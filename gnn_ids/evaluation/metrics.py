"""Metric helpers shared by the GNN and the classical baselines."""
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from ..config import DECISION_THRESHOLD


@torch.no_grad()
def evaluate(modules, criterion, node_emb, src_t, dst_t, X_t, y_t, batch_size, threshold, device):
    """Score flows using node embeddings from the training graph (UNSW / F1 protocol).

    No grad, no training. Returns (weighted loss per flow, metrics dict, probabilities).
    """
    modules.eval()
    n = X_t.shape[0]
    total_loss = 0.0
    probs = torch.empty(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        h_src = node_emb[src_t[start:end].to(device)]
        h_dst = node_emb[dst_t[start:end].to(device)]
        xb = X_t[start:end].to(device)
        yb = y_t[start:end].to(device)

        z_flow, _ = modules.attention(h_src, h_dst, xb)
        logits = modules.classifier(z_flow)
        total_loss += criterion(logits, yb).item()
        probs[start:end] = torch.sigmoid(logits).cpu()

    y_true = y_t.numpy()
    y_prob = probs.numpy()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "cm": confusion_matrix(y_true, y_pred),
    }
    return total_loss / n, metrics, probs


@torch.no_grad()
def evaluate_with_pr_auc(modules, criterion, node_emb, src_t, dst_t, X_t, y_t, batch_size,
                         threshold, device):
    """CSE / PR-AUC protocol: as ``evaluate`` plus PR-AUC. Returns (loss, metrics, probs)."""
    modules.eval()
    n = X_t.shape[0]
    total_loss = 0.0
    probs = torch.empty(n)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        h_src = node_emb[src_t[s:e].to(device)]
        h_dst = node_emb[dst_t[s:e].to(device)]
        xb = X_t[s:e].to(device)
        yb = y_t[s:e].to(device)
        z, _ = modules.attention(h_src, h_dst, xb)
        logits = modules.classifier(z)
        total_loss += criterion(logits, yb).item()
        probs[s:e] = torch.sigmoid(logits).cpu()
        del h_src, h_dst, xb, yb, z, logits
    yt, yp = y_t.numpy(), probs.numpy()
    yhat = (yp >= threshold).astype(int)
    return total_loss / n, {
        "pr_auc": average_precision_score(yt, yp),      # SELECTION METRIC
        "roc_auc": roc_auc_score(yt, yp),
        "accuracy": accuracy_score(yt, yhat),
        "precision": precision_score(yt, yhat, zero_division=0),
        "recall": recall_score(yt, yhat, zero_division=0),
        "f1": f1_score(yt, yhat, zero_division=0),
        "cm": confusion_matrix(yt, yhat),
    }, probs


def eval_baseline(name, y_true, y_prob, threshold=DECISION_THRESHOLD):
    """Same metrics and threshold as the GNN's ``evaluate`` (fixed 0.5 for the baselines)."""
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "cm": confusion_matrix(y_true, y_pred),
    }
    tn, fp, fn, tp = m["cm"].ravel()
    print(f"{name:<22} acc {m['accuracy']:.4f} | P {m['precision']:.4f} | "
          f"R {m['recall']:.4f} | F1 {m['f1']:.4f} | AUC {m['roc_auc']:.4f}")
    print(f"{'':<22} TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    return m


def metrics_at(y_true, y_prob, thr):
    """Threshold metrics used by the CSE baselines (also reports the flagged count)."""
    yhat = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y_true, yhat),
        "precision": precision_score(y_true, yhat, zero_division=0),
        "recall": recall_score(y_true, yhat, zero_division=0),
        "f1": f1_score(y_true, yhat, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "flagged": int(yhat.sum()), "flagged_pct": 100 * float(yhat.mean()),
    }


def ranking_metrics(y_true, y_prob):
    return {"roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob)}


def threshold_metrics(y_true, y_prob, thr):
    """Threshold metrics used by the controlled experiments (section 11)."""
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "flagged_pct": 100.0 * float(y_pred.mean())}
