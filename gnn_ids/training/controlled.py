"""Section 11: controlled GNN+Attention vs GCN-only ablation (UNSW, selection on PR-AUC).

One implementation serves both models; they differ only in ``head_fn``. All state that the
notebook kept in globals lives on ``ControlledExperiment``.

Protocol per model:
  Stage 1  train, select on validation PR-AUC.
  Stage 2  restore the best checkpoint, select the F1-max threshold on validation, freeze it.
  Stage 3  score the held-out split exactly once, at the frozen threshold.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from ..config import (ATT_DROPOUT, ATT_LATENT_DIM, ATT_SEED, ATT_USE_LAYERNORM, BATCH_SIZE,
                      CLS_DROPOUT, CLS_HIDDEN_DIM, GCN_HIDDEN_DIM, GCN_OUT_DIM, LEARNING_RATE,
                      N_FEATURES, PATIENCE, RANDOM_SEED, WEIGHT_DECAY, Locked, locked_assert)
from ..evaluation.metrics import ranking_metrics, threshold_metrics
from ..evaluation.thresholds import select_threshold_trimmed
from ..models.attention import PostGCNAttention
from ..models.classifier import FlowClassifier
from ..models.gcn import GCNEncoder
from ..utils import seed_torch

NAME_ATTN = "GNN+Attention (controlled, PR-AUC)"
NAME_GCN_ONLY = "GCN-only (mean of endpoints, PR-AUC)"


def head_attn(M, h_src, h_dst, x_edge):
    assert x_edge is not None
    z_flow, _ = M["attention"](h_src, h_dst, x_edge)
    return M["classifier"](z_flow)


def head_gcn_only(M, h_src, h_dst, x_edge):
    """z_flow = element-wise mean of the two endpoint embeddings. No learned
    combination layer and no flow features: x_edge must be None."""
    assert x_edge is None, "GCN-only head must never receive flow features"
    z_flow = 0.5 * (h_src + h_dst)
    return M["classifier"](z_flow)


def make_bundle(name, modules, head_fn, uses_edge_feats):
    """modules: ordered dict {'gcn': ..., [...], 'classifier': ...}."""
    params = [p for m in modules.values() for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return {"name": name, "modules": modules, "gcn": modules["gcn"],
            "head_fn": head_fn, "uses_edge_feats": uses_edge_feats,
            "params": params, "optimizer": optimizer}


def set_mode(bundle, train):
    for m in bundle["modules"].values():
        m.train(train)


class ControlledExperiment:
    """Holds the UNSW graph, validation/test tensors and the protocol helpers."""

    def __init__(self, graph, val, test, device, num_epochs, protected_paths,
                 pos_weight_value, num_nodes):
        self.graph = graph
        self.val = val
        self.test = test
        self.device = device
        self.num_epochs = num_epochs
        self.protected_paths = {str(p) for p in protected_paths}
        self.n_flows = graph.flow_src.shape[0]
        self.num_nodes = num_nodes
        self.heldout_accessed = set()

        # Loss (training labels only)
        self.y_train = graph.flow_label.float()
        n_pos = int(graph.flow_label.sum())
        n_neg = self.n_flows - n_pos
        locked_assert((n_pos, n_neg) == Locked.UNSW_POS_NEG)
        self.pos_weight = n_neg / n_pos
        assert abs(self.pos_weight - pos_weight_value) < 1e-12, "pos_weight differs from Section 7.1"

        # Same construction as Section 7.1: shape-[1] float32 tensor, reduction="sum"
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], dtype=torch.float32, device=device),
            reduction="sum")
        # CPU twin used only to log validation loss from CPU logits (same math, logging only)
        self.criterion_cpu = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], dtype=torch.float32), reduction="sum")

    # ----------------------------------------------------------------- builders
    def build_attn(self):
        """Fresh modules, seeded in the same order as Sections 5, 6 and 7.1."""
        seed_torch(RANDOM_SEED)
        gcn_m = GCNEncoder(in_channels=self.graph.x.shape[1], hidden_channels=GCN_HIDDEN_DIM,
                           out_channels=GCN_OUT_DIM).to(self.device)

        seed_torch(ATT_SEED)
        att_m = PostGCNAttention(node_dim=GCN_OUT_DIM, edge_in_dim=N_FEATURES,
                                 latent_dim=ATT_LATENT_DIM, use_layernorm=ATT_USE_LAYERNORM,
                                 dropout=ATT_DROPOUT).to(self.device)

        seed_torch(RANDOM_SEED)
        cls_m = FlowClassifier(in_dim=ATT_LATENT_DIM, hidden_dim=CLS_HIDDEN_DIM,
                               dropout=CLS_DROPOUT).to(self.device)

        return make_bundle(NAME_ATTN, {"gcn": gcn_m, "attention": att_m, "classifier": cls_m},
                           head_attn, uses_edge_feats=True)

    def build_gcn_only(self):
        """Fresh modules, seeded exactly as the GCN and classifier of ``build_attn``."""
        seed_torch(RANDOM_SEED)
        gcn_m = GCNEncoder(in_channels=self.graph.x.shape[1], hidden_channels=GCN_HIDDEN_DIM,
                           out_channels=GCN_OUT_DIM).to(self.device)

        seed_torch(RANDOM_SEED)
        cls_m = FlowClassifier(in_dim=GCN_OUT_DIM, hidden_dim=CLS_HIDDEN_DIM,
                               dropout=CLS_DROPOUT).to(self.device)

        return make_bundle(NAME_GCN_ONLY, {"gcn": gcn_m, "classifier": cls_m},
                           head_gcn_only, uses_edge_feats=False)

    # ------------------------------------------------------------------ scoring
    def gather(self, node_emb, idx, src_t, dst_t, X_t=None):
        """Endpoint embeddings for flows `idx`; flow features only if X_t is given."""
        src = src_t[idx].to(self.device, non_blocking=True)
        dst = dst_t[idx].to(self.device, non_blocking=True)
        x_edge = None if X_t is None else X_t[idx].to(self.device, non_blocking=True)
        return node_emb[src], node_emb[dst], x_edge

    @torch.no_grad()
    def _score(self, bundle, x_dev, ei_dev, src_t, dst_t, X_t):
        """Logits (CPU) for a fixed set of flows. Embeddings come from graph_train only."""
        set_mode(bundle, False)
        emb = bundle["gcn"](x_dev, ei_dev)
        M = bundle["modules"]
        n = src_t.shape[0]
        logits = torch.empty(n)
        for s in range(0, n, BATCH_SIZE):
            sl = slice(s, min(s + BATCH_SIZE, n))
            h_src, h_dst, xb = self.gather(emb, sl, src_t, dst_t, X_t)
            logits[sl] = bundle["head_fn"](M, h_src, h_dst, xb).cpu()
            del h_src, h_dst, xb
        del emb
        return logits

    def score_validation(self, bundle, x_dev, ei_dev):
        """Hard-wired to the validation split; takes no split argument."""
        X = self.val.X if bundle["uses_edge_feats"] else None
        return self._score(bundle, x_dev, ei_dev, self.val.src, self.val.dst, X)

    def _score_heldout(self, bundle, x_dev, ei_dev):
        """Held-out split scorer. Called ONLY from ``stage3``."""
        X = self.test.X if bundle["uses_edge_feats"] else None
        return self._score(bundle, x_dev, ei_dev, self.test.src, self.test.dst, X)

    # ------------------------------------------------------------------ Stage 1
    def train_stage1(self, bundle, ckpt_path):
        """Section 7.2.2 mechanics; selection on validation PR-AUC only."""
        name, M, gcn_m, opt = bundle["name"], bundle["modules"], bundle["gcn"], bundle["optimizer"]
        graph = self.graph
        assert str(ckpt_path) not in self.protected_paths
        assert not ckpt_path.exists(), f"Refusing to overwrite {ckpt_path}"
        X_flow = graph.flow_attr if bundle["uses_edge_feats"] else None
        yv = self.val.y.numpy().astype(int)

        x_dev = graph.x.to(self.device)
        ei_dev = graph.edge_index.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        best_pr, best_epoch, no_imp, hist = -1.0, -1, 0, []
        t_start = time.time()
        print(f"{name}: Stage 1 | selection = validation PR-AUC | patience {PATIENCE} "
              f"| cap {self.num_epochs} | edge features in head: {bundle['uses_edge_feats']}")
        print("-" * 118)

        for epoch in range(1, self.num_epochs + 1):
            t_ep = time.time()
            set_mode(bundle, True)
            opt.zero_grad(set_to_none=True)

            emb = gcn_m(x_dev, ei_dev)                        # one graph-wide forward
            emb_det = emb.detach().requires_grad_(True)       # batches backprop into this leaf

            g = torch.Generator().manual_seed(RANDOM_SEED + epoch)
            perm = torch.randperm(self.n_flows, generator=g)

            train_loss = 0.0
            for start in range(0, self.n_flows, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                h_src, h_dst, xb = self.gather(emb_det, idx, graph.flow_src,
                                               graph.flow_dst, X_flow)
                yb = self.y_train[idx].to(self.device)
                logits = bundle["head_fn"](M, h_src, h_dst, xb)
                loss = self.criterion(logits, yb) / self.n_flows    # dataset-level mean
                loss.backward()
                train_loss += loss.item()
                del h_src, h_dst, xb, yb, logits, loss

            assert emb_det.grad is not None, "No gradient reached the node embeddings"
            emb.backward(gradient=emb_det.grad)               # single backward through the GCN

            opt.step()
            del emb, emb_det

            # ---- validation (graph_train embeddings, no_grad) ----
            v_logits = self.score_validation(bundle, x_dev, ei_dev)
            val_loss = self.criterion_cpu(v_logits, self.val.y).item() / len(yv)
            pv = torch.sigmoid(v_logits).numpy()
            rk = ranking_metrics(yv, pv)
            m05 = threshold_metrics(yv, pv, 0.5)              # logged only

            improved = rk["pr_auc"] > best_pr                 # SELECTION: PR-AUC only
            if improved:
                best_pr, best_epoch, no_imp = rk["pr_auc"], epoch, 0
                ck = {"model_name": name, "epoch": epoch,
                      "val_pr_auc": rk["pr_auc"], "val_roc_auc": rk["roc_auc"],
                      "val_f1_at_0.5": m05["f1"],
                      "optimizer": opt.state_dict(),
                      "selection_metric": "validation PR-AUC",
                      "threshold_selected": None, "threshold_reference": 0.5,
                      "module_keys": list(M.keys()),
                      "uses_edge_feats": bundle["uses_edge_feats"],
                      "n_params": sum(p.numel() for p in bundle["params"])}
                ck.update({k: m.state_dict() for k, m in M.items()})
                torch.save(ck, ckpt_path)
            else:
                no_imp += 1

            dt = time.time() - t_ep
            peak = (torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available() else 0.0)
            hist.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         **rk, "precision_at_0.5": m05["precision"],
                         "recall_at_0.5": m05["recall"], "f1_at_0.5": m05["f1"],
                         "epoch_time_s": dt})
            print(f"Ep {epoch:02d}/{self.num_epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| PR-AUC {rk['pr_auc']:.6f} | ROC {rk['roc_auc']:.4f} | @0.5: "
                  f"P {m05['precision']:.4f} R {m05['recall']:.4f} F1 {m05['f1']:.4f} "
                  f"| {dt:5.1f}s | {peak:.0f}MB"
                  f"{'  <- saved' if improved else f'  (no imp {no_imp}/{PATIENCE})'}")

            if no_imp >= PATIENCE:
                print(f"Early stopping: no validation PR-AUC improvement for {PATIENCE} epochs.")
                break

        total = time.time() - t_start
        peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        del x_dev, ei_dev
        for m in M.values():
            m.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        assert graph.x.device.type == "cpu" and graph.flow_attr.device.type == "cpu"
        print("-" * 118)
        print(f"{name}: best epoch {best_epoch} | best val PR-AUC {best_pr:.6f} | "
              f"{len(hist)} epochs in {total / 60:.2f} min "
              f"({total / len(hist):.2f} s/epoch) | peak GPU {peak:.0f} MB")
        return {"history": hist, "best_epoch": best_epoch, "best_val_pr_auc": best_pr,
                "total_time_s": total, "peak_gpu_mb": peak}

    # ------------------------------------------------------------------ Stage 2
    def stage2(self, build_fn, ckpt_path):
        """Restore best checkpoint, score validation once, select and freeze the threshold."""
        assert ckpt_path.exists(), "Stage 1 checkpoint not found"
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        assert ck["threshold_selected"] is None, "Threshold already frozen; Stage 2 runs once"
        assert ck["selection_metric"] == "validation PR-AUC"

        bundle = build_fn()
        assert list(bundle["modules"].keys()) == ck["module_keys"]
        for k, m in bundle["modules"].items():
            m.load_state_dict(ck[k], strict=True)

        x_dev = self.graph.x.to(self.device)
        ei_dev = self.graph.edge_index.to(self.device)
        v_logits = self.score_validation(bundle, x_dev, ei_dev)
        del x_dev, ei_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        yv = self.val.y.numpy().astype(int)
        pv = torch.sigmoid(v_logits).numpy()
        rk = ranking_metrics(yv, pv)
        pr_diff = abs(rk["pr_auc"] - ck["val_pr_auc"])
        assert pr_diff < 1e-3, f"Restored model does not reproduce saved PR-AUC (diff {pr_diff:.2e})"

        thr = select_threshold_trimmed(yv, pv)                # validation labels only
        m_sel = threshold_metrics(yv, pv, thr)
        m_05 = threshold_metrics(yv, pv, 0.5)

        ck["threshold_selected"] = thr
        ck["val_metrics_at_selected"] = m_sel
        ck["val_metrics_at_0.5"] = m_05
        ck["val_prob_range"] = (float(pv.min()), float(pv.max()))
        torch.save(ck, ckpt_path)

        print(f"{ck['model_name']}: Stage 2 | restored epoch {ck['epoch']} "
              f"| val PR-AUC {rk['pr_auc']:.6f} (saved {ck['val_pr_auc']:.6f}, diff {pr_diff:.1e})")
        print(f"  val ROC-AUC {rk['roc_auc']:.6f} | val prob range "
              f"[{pv.min():.6f}, {pv.max():.6f}]")
        print(f"  FROZEN threshold (validation labels only): {thr:.6f}")
        for lbl, m in [("selected", m_sel), ("0.5", m_05)]:
            print(f"  val @ {lbl:<8}: acc {m['accuracy']:.4f} P {m['precision']:.4f} "
                  f"R {m['recall']:.4f} F1 {m['f1']:.4f} | TN={m['tn']:,} FP={m['fp']:,} "
                  f"FN={m['fn']:,} TP={m['tp']:,} | flagged {m['flagged_pct']:.3f}%")
        print("  Held-out split not accessed.")
        return {"ckpt_path": ckpt_path, "val_prob": pv, "threshold": thr,
                "val_ranking": rk, "val_at_selected": m_sel, "val_at_0.5": m_05}

    # ------------------------------------------------------------------ Stage 3
    def stage3(self, build_fn, ckpt_path):
        """First and only held-out access per model. Generates its own probabilities."""
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        name = ck["model_name"]
        thr = ck["threshold_selected"]
        assert isinstance(thr, float) and np.isfinite(thr), \
            f"{name}: threshold not frozen; Stage 2 must complete first"
        assert ck["threshold_reference"] == 0.5
        assert name not in self.heldout_accessed, f"{name}: held-out split already evaluated once"

        bundle = build_fn()
        assert list(bundle["modules"].keys()) == ck["module_keys"]
        for k, m in bundle["modules"].items():
            m.load_state_dict(ck[k], strict=True)

        x_dev = self.graph.x.to(self.device)
        ei_dev = self.graph.edge_index.to(self.device)
        print("=" * 84)
        print(f"TEST ACCESS - model and threshold frozen ({name}, epoch {ck['epoch']}, "
              f"threshold {thr:.6f})")
        print("=" * 84)
        self.heldout_accessed.add(name)
        t_logits = self._score_heldout(bundle, x_dev, ei_dev)
        del x_dev, ei_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        yt = self.test.y.numpy().astype(int)
        pt = torch.sigmoid(t_logits).numpy()
        rk = ranking_metrics(yt, pt)
        m_sel = threshold_metrics(yt, pt, thr)
        m_05 = threshold_metrics(yt, pt, 0.5)

        print(f"  ROC-AUC {rk['roc_auc']:.6f} | PR-AUC {rk['pr_auc']:.6f}")
        for lbl, m in [("PRIMARY (val-selected)", m_sel), ("REFERENCE (0.5)", m_05)]:
            print(f"  {lbl} @ {m['threshold']:.6f}: acc {m['accuracy']:.6f} "
                  f"P {m['precision']:.6f} R {m['recall']:.6f} F1 {m['f1']:.6f}")
            print(f"    TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,} "
                  f"| flagged {m['flagged_pct']:.6f}%")
        return {"name": name, "epoch": ck["epoch"], "threshold": thr, "prob": pt,
                "ranking": rk, "at_selected": m_sel, "at_0.5": m_05}
