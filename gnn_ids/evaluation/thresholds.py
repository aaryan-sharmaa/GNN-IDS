"""F1-maximising decision-threshold selection, always on VALIDATION data only.

Two rules exist in the original experiments and are intentionally kept separate because
they can pick different thresholds on tied probabilities:

* ``select_threshold_full_curve``  - CSE GNN and CSE baselines (division over the full
  precision/recall arrays, ``where=`` guard).
* ``select_threshold_trimmed``     - UNSW final table and section 11 (``[:-1]`` slice with
  a ``1e-12`` floor on the denominator).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def select_threshold_full_curve(y_true, y_prob):
    """CSE rule. Returns a dict with the threshold and the validation P/R/F1 at it."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1c = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                    where=(prec + rec) > 0)
    bi = int(np.argmax(f1c))
    return {"threshold": float(thr[min(bi, len(thr) - 1)]),
            "f1": float(f1c[bi]), "precision": float(prec[bi]), "recall": float(rec[bi])}


def select_threshold_trimmed(y_true, y_prob):
    """UNSW / section-11 rule."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precision[:-1] * recall[:-1]
                 / np.maximum(precision[:-1] + recall[:-1], 1e-12))
    return float(thresholds[int(np.argmax(f1_scores))])


def select_threshold_and_evaluate(name, val_prob, val_y, test_prob, test_y):
    """Stage 2: select the threshold on validation. Stage 3: score test at the frozen value."""
    val_prob = np.asarray(val_prob).reshape(-1)
    val_y = np.asarray(val_y).reshape(-1).astype(int)
    test_prob = np.asarray(test_prob).reshape(-1)
    test_y = np.asarray(test_y).reshape(-1).astype(int)

    # ---------------- Stage 2: validation threshold selection ----------------
    selected_threshold = select_threshold_trimmed(val_y, val_prob)
    val_pred = (val_prob >= selected_threshold).astype(int)

    val_metrics = {
        "val_threshold": selected_threshold,
        "val_accuracy": accuracy_score(val_y, val_pred),
        "val_precision": precision_score(val_y, val_pred, zero_division=0),
        "val_recall": recall_score(val_y, val_pred, zero_division=0),
        "val_f1": f1_score(val_y, val_pred, zero_division=0),
        "val_roc_auc": roc_auc_score(val_y, val_prob),
        "val_pr_auc": average_precision_score(val_y, val_prob),
    }

    # ---------------- Stage 3: frozen test evaluation ----------------
    test_pred = (test_prob >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred, labels=[0, 1]).ravel()

    test_metrics = {
        "test_accuracy": accuracy_score(test_y, test_pred),
        "test_precision": precision_score(test_y, test_pred, zero_division=0),
        "test_recall": recall_score(test_y, test_pred, zero_division=0),
        "test_f1": f1_score(test_y, test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(test_y, test_prob),
        "test_pr_auc": average_precision_score(test_y, test_prob),
        "test_tn": int(tn), "test_fp": int(fp), "test_fn": int(fn), "test_tp": int(tp),
        "test_flagged_pct": 100.0 * test_pred.mean(),
    }

    # ---------------- 0.5 reference ----------------
    ref_pred = (test_prob >= 0.5).astype(int)
    ref_metrics = {
        "test_accuracy_at_0.5": accuracy_score(test_y, ref_pred),
        "test_precision_at_0.5": precision_score(test_y, ref_pred, zero_division=0),
        "test_recall_at_0.5": recall_score(test_y, ref_pred, zero_division=0),
        "test_f1_at_0.5": f1_score(test_y, ref_pred, zero_division=0),
    }

    return {"model": name, **val_metrics, **test_metrics, **ref_metrics}
