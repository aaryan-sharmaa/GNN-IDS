"""UNSW GNN training with the historical protocol: selection and early stopping on
validation F1 at the fixed 0.5 threshold (notebook sections 7.1-7.3).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..config import (BATCH_SIZE, DECISION_THRESHOLD, LEARNING_RATE, PATIENCE, RANDOM_SEED,
                      WEIGHT_DECAY)
from ..evaluation.metrics import evaluate
from ..models.attention import gather_flow_inputs


@dataclass
class F1TrainingResult:
    history: list = field(default_factory=list)
    best_epoch: int = -1
    best_f1: float = -1.0
    total_time_s: float = 0.0


def make_loss_and_optimizer(modules, graph, y_train_sub, device):
    """Imbalance-weighted loss (training labels only) and Adam over all three components."""
    y_train_tensor = graph.flow_label                      # aligned with flow order
    assert torch.equal(y_train_tensor, torch.from_numpy(y_train_sub.to_numpy(dtype=np.int64))), \
        "flow_label is not aligned with y_train"

    n_flows = graph.flow_src.shape[0]
    n_pos = int(y_train_tensor.sum())
    n_neg = int(n_flows - n_pos)
    pos_weight_value = n_neg / n_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")
    optimizer = torch.optim.Adam(modules.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    return criterion, optimizer, pos_weight_value


def train_f1_protocol(modules, graph, criterion, optimizer, val, device, checkpoint_path,
                      num_epochs, patience=PATIENCE, batch_size=BATCH_SIZE,
                      threshold=DECISION_THRESHOLD) -> F1TrainingResult:
    """Each epoch: one graph-wide GCN forward, mini-batches through gather -> attention ->
    classifier that backpropagate into a detached copy of the node embeddings, one backward
    through the GCN with the accumulated embedding gradient, one optimizer step, then
    validation. Selection on validation F1 at ``threshold``. No test data is touched.
    """
    n_flows = graph.flow_src.shape[0]
    y_train_t = graph.flow_label.float()

    best_f1, best_epoch, epochs_no_improve = -1.0, -1, 0
    history = []

    # Graph tensors to GPU once for the whole training run
    x_dev = graph.x.to(device)
    ei_dev = graph.edge_index.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    print(f"Training on {n_flows:,} flows | {graph.edge_index.shape[1]:,} MP edges "
          f"| {num_epochs} epochs max | patience {patience}")
    print("-" * 104)

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()

        # ---- Training ----
        modules.train()
        optimizer.zero_grad(set_to_none=True)

        emb = modules.gcn(x_dev, ei_dev)                # one graph-wide forward
        emb_det = emb.detach().requires_grad_(True)     # batches backprop into this leaf

        g = torch.Generator().manual_seed(RANDOM_SEED + epoch)
        perm = torch.randperm(n_flows, generator=g)

        train_loss = 0.0
        for start in range(0, n_flows, batch_size):
            idx = perm[start:start + batch_size]
            h_src, h_dst, xb = gather_flow_inputs(graph, emb_det, idx, device)
            yb = y_train_t[idx].to(device)

            z_flow, _ = modules.attention(h_src, h_dst, xb)
            logits = modules.classifier(z_flow)
            loss = criterion(logits, yb) / n_flows      # dataset-level mean
            loss.backward()
            train_loss += loss.item()

            del h_src, h_dst, xb, yb, z_flow, logits, loss

        assert emb_det.grad is not None, "No gradient reached the node embeddings"
        emb.backward(gradient=emb_det.grad)             # single backward through the GCN

        optimizer.step()
        del emb, emb_det

        # ---- Validation (training graph only, no grad) ----
        with torch.no_grad():
            modules.gcn.eval()
            emb_eval = modules.gcn(x_dev, ei_dev)
        val_loss, m, _ = evaluate(modules, criterion, emb_eval, val.src, val.dst, val.X, val.y,
                                  batch_size, threshold, device)
        del emb_eval

        elapsed = time.time() - t_epoch
        improved = m["f1"] > best_f1
        if improved:
            best_f1 = m["f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "val_f1": best_f1,
                "gcn": modules.gcn.state_dict(),
                "attention": modules.attention.state_dict(),
                "classifier": modules.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "threshold": threshold,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **m})

        print(f"Epoch {epoch:02d}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} "
              f"AUC {m['roc_auc']:.4f} | {elapsed:5.1f}s"
              f"{'  <- saved' if improved else f'  (no improve {epochs_no_improve}/{patience})'}")

        if epochs_no_improve >= patience:
            print(f"Early stopping: no validation F1 improvement for {patience} epochs.")
            break

    total_time = time.time() - t_start
    best = history[best_epoch - 1]
    tn, fp, fn, tp = best["cm"].ravel()

    print("-" * 104)
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation F1  : {best_f1:.4f}")
    print(f"  accuracy {best['accuracy']:.4f} | precision {best['precision']:.4f} "
          f"| recall {best['recall']:.4f} | ROC-AUC {best['roc_auc']:.4f}")
    print(f"  confusion matrix  : TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Total training time : {total_time / 60:.1f} min ({len(history)} epochs run)")
    return F1TrainingResult(history=history, best_epoch=best_epoch, best_f1=best_f1,
                            total_time_s=total_time)


def restore_checkpoint(modules, checkpoint_path, device):
    """Load the best-validation checkpoint (strict) into ``modules``; returns the checkpoint."""
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    modules.gcn.load_state_dict(ckpt["gcn"], strict=True)
    modules.attention.load_state_dict(ckpt["attention"], strict=True)
    modules.classifier.load_state_dict(ckpt["classifier"], strict=True)
    modules.eval()
    return ckpt
