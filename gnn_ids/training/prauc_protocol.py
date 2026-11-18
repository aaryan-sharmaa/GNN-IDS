"""CSE GNN: revised protocol with selection on validation PR-AUC (notebook sections 10.6-R, 10.7).

Stage 1  train, select the checkpoint and early-stop on validation PR-AUC.
Stage 2  restore the best checkpoint, select the F1-maximising threshold on VALIDATION only
         and freeze it into the checkpoint.
Stage 3  score the test split once at the frozen threshold (0.5 kept as a reference).
"""
import time

import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from ..config import (BATCH_SIZE, LEARNING_RATE, N_FEATURES, N_PARAMS_TOTAL, PATIENCE,
                      RANDOM_SEED, WEIGHT_DECAY, Locked, locked_assert)
from ..evaluation.metrics import evaluate_with_pr_auc
from ..evaluation.thresholds import select_threshold_full_curve
from ..models.attention import gather_flow_inputs


def make_loss_and_optimizer(modules, graph, device):
    """Positive-class weight from training labels only; Adam over all three components."""
    y_train_t = torch.from_numpy(graph.flow_label.numpy()).float()
    n_pos = int(y_train_t.sum().item())
    n_neg = int(y_train_t.numel() - n_pos)
    pos_weight_value = n_neg / n_pos

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device), reduction="sum")

    all_params = modules.parameters()
    optimizer = torch.optim.Adam(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    assert sum(p.numel() for p in all_params) == N_PARAMS_TOTAL
    assert len(all_params) == 12
    return criterion, optimizer, y_train_t, pos_weight_value


def train_prauc_stage1(modules, graph, criterion, optimizer, y_train_t, val, device,
                       checkpoint_path, num_epochs, patience=PATIENCE, batch_size=BATCH_SIZE,
                       pos_weight_value=None):
    """Stage 1. Returns (history, best_epoch, best_pr_auc)."""
    n_flows = graph.flow_src.shape[0]
    print("=" * 100)
    print("CSE TRAINING - selection on validation PR-AUC")
    print("=" * 100)
    print(f"  patience {patience} | pos_weight {pos_weight_value:.4f} | batch {batch_size:,} "
          f"| {num_epochs} epochs | seed {RANDOM_SEED}")
    print(f"  random PR baseline: {float(val.y.mean()):.6f} (validation positive rate)")
    print(f"  checkpoint        : {checkpoint_path}")
    print("-" * 100)

    best_pr, best_epoch, no_improve = -1.0, -1, 0
    history = []

    x_dev = graph.x.to(device)
    ei_dev = graph.edge_index.to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()
        modules.train()
        optimizer.zero_grad(set_to_none=True)

        emb = modules.gcn(x_dev, ei_dev)
        emb_det = emb.detach().requires_grad_(True)

        g = torch.Generator().manual_seed(RANDOM_SEED + epoch)
        perm = torch.randperm(n_flows, generator=g)

        train_loss = 0.0
        for s in range(0, n_flows, batch_size):
            idx = perm[s:s + batch_size]
            h_src, h_dst, xb = gather_flow_inputs(graph, emb_det, idx, device)
            yb = y_train_t[idx].to(device)
            z, _ = modules.attention(h_src, h_dst, xb)
            logits = modules.classifier(z)
            loss = criterion(logits, yb) / n_flows
            loss.backward()
            train_loss += loss.item()
            del h_src, h_dst, xb, yb, z, logits, loss

        assert emb_det.grad is not None, "No gradient reached the node embeddings"
        emb.backward(gradient=emb_det.grad)

        optimizer.step()
        del emb, emb_det

        with torch.no_grad():
            modules.gcn.eval()
            emb_eval = modules.gcn(x_dev, ei_dev)
        val_loss, m, _ = evaluate_with_pr_auc(modules, criterion, emb_eval, val.src, val.dst,
                                              val.X, val.y, batch_size, 0.5, device)
        del emb_eval

        improved = m["pr_auc"] > best_pr          # SELECTION ON PR-AUC
        if improved:
            best_pr, best_epoch, no_improve = m["pr_auc"], epoch, 0
            torch.save({
                "epoch": epoch,
                "val_pr_auc": best_pr,
                "val_roc_auc": m["roc_auc"],
                "val_f1_at_0.5": m["f1"],
                "gcn": modules.gcn.state_dict(),
                "attention": modules.attention.state_dict(),
                "classifier": modules.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "selection_metric": "validation PR-AUC",
                "threshold_selected": None,           # filled in Stage 2
                "threshold_reference": 0.5,
            }, checkpoint_path)
        else:
            no_improve += 1

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **m})
        peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f"Ep {epoch:02d}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| PR-AUC {m['pr_auc']:.6f} | ROC {m['roc_auc']:.4f} | "
              f"@0.5: P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} | "
              f"{time.time() - t_ep:5.1f}s | {peak:.0f}MB"
              f"{'  <- saved' if improved else f'  (no imp {no_improve}/{patience})'}")

        if no_improve >= patience:
            print(f"Early stopping: no validation PR-AUC improvement for {patience} epochs.")
            break

    print("-" * 100)
    print(f"Stage 1 complete | best epoch {best_epoch} | best val PR-AUC {best_pr:.6f} "
          f"| {len(history)} epochs in {(time.time() - t_start) / 60:.1f} min")
    del x_dev, ei_dev
    return history, best_epoch, best_pr


def freeze_threshold(modules, graph, val, device, checkpoint_path, batch_size=BATCH_SIZE,
                     criterion=None):
    """Stage 2: restore the best checkpoint, pick the threshold on validation, freeze it.

    Returns (frozen checkpoint dict, validation probabilities).
    """
    print("\n" + "=" * 100)
    print("STAGE 2 - THRESHOLD SELECTION (validation labels only; test untouched)")
    print("=" * 100)

    x_dev = graph.x.to(device)
    ei_dev = graph.edge_index.to(device)

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    modules.gcn.load_state_dict(ck["gcn"], strict=True)
    modules.attention.load_state_dict(ck["attention"], strict=True)
    modules.classifier.load_state_dict(ck["classifier"], strict=True)
    modules.eval()
    print(f"Restored epoch {ck['epoch']} (val PR-AUC {ck['val_pr_auc']:.6f})")

    with torch.no_grad():
        emb_best = modules.gcn(x_dev, ei_dev)
    _, m_val_05, val_probs = evaluate_with_pr_auc(
        modules, criterion, emb_best, val.src, val.dst, val.X, val.y, batch_size, 0.5, device)
    del emb_best

    yv = val.y.numpy()
    pv = val_probs.numpy()
    sel = select_threshold_full_curve(yv, pv)
    thr = sel["threshold"]

    print(f"Validation probability range : [{pv.min():.6f}, {pv.max():.6f}]")
    print(f"F1-maximizing threshold      : {thr:.6f}")
    print(f"  validation F1 there        : {sel['f1']:.6f} "
          f"(precision {sel['precision']:.4f}, recall {sel['recall']:.4f})")
    print(f"  validation F1 at 0.5       : {m_val_05['f1']:.6f} (secondary reference)")

    yhat_sel = (pv >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, yhat_sel).ravel()
    print(f"  accuracy {accuracy_score(yv, yhat_sel):.4f} | precision "
          f"{precision_score(yv, yhat_sel, zero_division=0):.4f} | recall "
          f"{recall_score(yv, yhat_sel, zero_division=0):.4f} | F1 "
          f"{f1_score(yv, yhat_sel, zero_division=0):.4f}")
    print(f"  confusion matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    # Freeze the threshold into the checkpoint
    ck["threshold_selected"] = thr
    ck["val_f1_at_selected"] = sel["f1"]
    torch.save(ck, checkpoint_path)
    print(f"FROZEN in {checkpoint_path}: epoch {ck['epoch']} | threshold_selected "
          f"{ck['threshold_selected']:.6f} | threshold_reference {ck['threshold_reference']}")

    del x_dev, ei_dev
    modules.gcn.zero_grad(set_to_none=True)
    modules.attention.zero_grad(set_to_none=True)
    modules.classifier.zero_grad(set_to_none=True)
    return ck, val_probs


def evaluate_frozen_on_test(modules, graph, test, device, checkpoint_path, batch_size=BATCH_SIZE):
    """Stage 3 (evaluation only): load the frozen checkpoint and score the test split once.

    Returns (ck_r, results dict keyed by threshold label, test probabilities).
    """
    assert checkpoint_path.exists(), f"Missing: {checkpoint_path}"
    ck_r = torch.load(checkpoint_path, map_location=device, weights_only=False)

    assert ck_r["threshold_selected"] is not None, "Threshold was never frozen in Stage 2"
    thr_primary = float(ck_r["threshold_selected"])   # from the file, not hardcoded
    thr_reference = float(ck_r["threshold_reference"])
    locked_assert(abs(thr_primary - Locked.CSE_THRESHOLD) < 1e-5,
                  f"Checkpoint threshold {thr_primary} != the frozen {Locked.CSE_THRESHOLD}")
    assert thr_reference == 0.5
    assert ck_r["selection_metric"] == "validation PR-AUC"

    modules.gcn.load_state_dict(ck_r["gcn"], strict=True)
    modules.attention.load_state_dict(ck_r["attention"], strict=True)
    modules.classifier.load_state_dict(ck_r["classifier"], strict=True)
    modules.eval()

    assert all(torch.isfinite(p).all() for p in modules.parameters())
    n_params = sum(p.numel() for p in modules.parameters())
    assert n_params == N_PARAMS_TOTAL, f"Parameter count changed: {n_params}"

    n_test = test.X.shape[0]
    locked_assert(n_test == Locked.CSE_TEST_FLOWS, f"Test flow count is {n_test}")
    assert test.X.shape[1] == N_FEATURES

    # Graph integrity: still the training graph, no test flows added
    assert graph.edge_index.shape[1] == 2 * graph.flow_src.shape[0], "edge_index changed"

    n_pos = int(test.y.sum())
    print("=" * 80)
    print("CSE GNN TEST EVALUATION (Stage 3, evaluation only)")
    print("=" * 80)
    print(f"Checkpoint {checkpoint_path} | frozen epoch {ck_r['epoch']} | "
          f"val PR-AUC {ck_r['val_pr_auc']:.6f} | val ROC-AUC {ck_r['val_roc_auc']:.6f}")
    print(f"Thresholds frozen before test access: primary {thr_primary:.6f} "
          f"(validation F1-max) | reference {thr_reference}")
    print(f"Test flows {n_test:,} | positives {n_pos:,} ({100 * n_pos / n_test:.3f}%)")

    # Node embeddings from the TRAINING graph only
    x_dev = graph.x.to(device)
    ei_dev = graph.edge_index.to(device)
    with torch.no_grad():
        emb_test = modules.gcn(x_dev, ei_dev)
    assert emb_test.shape == (graph.num_nodes, 64)
    assert torch.isfinite(emb_test).all()

    # Score the test flows once
    t0 = time.time()
    logits_t = torch.empty(n_test)
    with torch.no_grad():
        for s in range(0, n_test, batch_size):
            e = min(s + batch_size, n_test)
            h_src = emb_test[test.src[s:e].to(device)]
            h_dst = emb_test[test.dst[s:e].to(device)]
            xb = test.X[s:e].to(device)
            z, _ = modules.attention(h_src, h_dst, xb)
            logits_t[s:e] = modules.classifier(z).cpu()
            del h_src, h_dst, xb, z
    infer_time = time.time() - t0

    test_probs = torch.sigmoid(logits_t)
    pt = test_probs.numpy()
    yt = test.y.numpy()

    # Threshold-independent metrics (computed once)
    test_roc = roc_auc_score(yt, pt)
    test_pr = average_precision_score(yt, pt)
    print(f"ROC-AUC {test_roc:.6f} | PR-AUC {test_pr:.6f} | random PR baseline "
          f"{yt.mean():.6f} | lift {test_pr / yt.mean():.2f}x | inference {infer_time:.1f}s")

    # Metrics at both thresholds, from the same probability array
    results = {}
    for label, thr in [("PRIMARY (val-selected)", thr_primary),
                       ("REFERENCE (0.5)", thr_reference)]:
        yhat = (pt >= thr).astype(int)
        cm = confusion_matrix(yt, yhat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        res = {
            "threshold": thr,
            "accuracy": accuracy_score(yt, yhat),
            "precision": precision_score(yt, yhat, zero_division=0),
            "recall": recall_score(yt, yhat, zero_division=0),
            "f1": f1_score(yt, yhat, zero_division=0),
            "roc_auc": test_roc, "pr_auc": test_pr, "cm": cm,
        }
        results[label] = res
        print(f"\n{label} | threshold {thr:.6f}")
        print(f"  accuracy {res['accuracy']:.4f} | precision {res['precision']:.4f} | "
              f"recall {res['recall']:.4f} | F1 {res['f1']:.4f}")
        print(f"  TP={tp:,} TN={tn:,} FP={fp:,} FN={fn:,} | flagged {int(yhat.sum()):,} "
              f"({100 * yhat.mean():.3f}% of test flows)")

    del emb_test, x_dev, ei_dev, logits_t
    return ck_r, results, test_probs
