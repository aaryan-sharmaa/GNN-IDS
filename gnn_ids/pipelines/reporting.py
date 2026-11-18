"""Final comparison tables (notebook sections 10.9 and 10.10)."""
import pandas as pd

from ..config import Locked, RunPaths, locked_assert
from ..evaluation.thresholds import select_threshold_and_evaluate


# ------------------------------------------------------------------ 10.9 CSE comparison
def build_cse_final_comparison(cse) -> pd.DataFrame:
    """Consolidate the frozen CSE test results for the GNN and the three baselines.

    Every figure is read from stored objects; nothing is retyped or recomputed.
    """
    ck_r = cse.ck_r
    gnn_sel = cse.test_results_r["PRIMARY (val-selected)"]
    gnn_05 = cse.test_results_r["REFERENCE (0.5)"]

    # Guard: the assembled GNN row must match the locked 10.7 figures
    locked_assert(abs(gnn_sel["f1"] - Locked.CSE_GNN_F1_PRIMARY) < 5e-4
                  and abs(gnn_05["f1"] - Locked.CSE_GNN_F1_REFERENCE) < 5e-4,
                  "GNN test metrics differ from the locked 10.7 values")
    locked_assert(abs(gnn_sel["pr_auc"] - Locked.CSE_GNN_PR_AUC) < 1e-5)

    g_tn, g_fp, g_fn, g_tp = gnn_sel["cm"].ravel()
    g5_tn, g5_fp, g5_fn, g5_tp = gnn_05["cm"].ravel()
    n_te_flows = int(g_tn + g_fp + g_fn + g_tp)

    gnn_row = {
        "model": "GNN + Post-GCN Attention",
        "threshold_selected": float(ck_r["threshold_selected"]),
        "val_roc_auc": float(ck_r["val_roc_auc"]), "val_pr_auc": float(ck_r["val_pr_auc"]),
        "test_roc_auc": gnn_sel["roc_auc"], "test_pr_auc": gnn_sel["pr_auc"],
        "test_accuracy": gnn_sel["accuracy"], "test_precision": gnn_sel["precision"],
        "test_recall": gnn_sel["recall"], "test_f1": gnn_sel["f1"],
        "test_tn": int(g_tn), "test_fp": int(g_fp),
        "test_fn": int(g_fn), "test_tp": int(g_tp),
        "test_flagged_pct": 100 * (g_tp + g_fp) / n_te_flows,
        "test_accuracy_at_0.5": gnn_05["accuracy"],
        "test_precision_at_0.5": gnn_05["precision"],
        "test_recall_at_0.5": gnn_05["recall"], "test_f1_at_0.5": gnn_05["f1"],
        "test_tn_at_0.5": int(g5_tn), "test_fp_at_0.5": int(g5_fp),
        "test_fn_at_0.5": int(g5_fn), "test_tp_at_0.5": int(g5_tp),
        "test_flagged_pct_at_0.5": 100 * (g5_tp + g5_fp) / n_te_flows,
        "uses_graph": True,
    }

    bl = cse.baseline_results.reset_index()
    bl["uses_graph"] = False
    comparison = pd.concat([pd.DataFrame([gnn_row]), bl], ignore_index=True).set_index("model")

    COLS_TI = ["test_roc_auc", "test_pr_auc"]
    COLS_TD = ["threshold_selected", "test_accuracy", "test_precision", "test_recall", "test_f1"]
    COLS_05 = ["test_accuracy_at_0.5", "test_precision_at_0.5", "test_recall_at_0.5",
               "test_f1_at_0.5"]
    COLS_CM = ["test_tn", "test_fp", "test_fn", "test_tp", "test_flagged_pct"]

    y_te = cse.y_test_np
    pos_rate = float(y_te.mean())

    print("=" * 96)
    print("CSE-CIC-IDS2018-v3 FINAL TEST COMPARISON")
    print(f"{n_te_flows:,} test flows | {int(y_te.sum()):,} attacks ({100 * pos_rate:.3f}%) "
          f"| {int((y_te == 0).sum()):,} benign")
    print("=" * 96)

    print("\n[A] THRESHOLD-INDEPENDENT METRICS (ranking quality; no decision boundary)")
    ti = comparison[COLS_TI].copy()
    ti["pr_auc_lift_vs_random"] = ti["test_pr_auc"] / pos_rate
    print(ti.round(6).to_string())
    print(f"    random PR-AUC baseline = test positive rate = {pos_rate:.6f}")

    print("\n[B] THRESHOLD-DEPENDENT METRICS at each model's validation-selected threshold")
    print(comparison[COLS_TD].round(6).to_string())

    print("\n[C] THRESHOLD-DEPENDENT METRICS at the fixed 0.5 reference")
    print(comparison[COLS_05].round(6).to_string())

    print("\n[D] CONFUSION MATRICES at the validation-selected threshold")
    print(comparison[COLS_CM].round(3).to_string())

    print("\n[E] EFFECT OF THRESHOLD CHOICE (selected minus 0.5, same frozen models)")
    delta = pd.DataFrame({
        "d_accuracy": comparison["test_accuracy"] - comparison["test_accuracy_at_0.5"],
        "d_precision": comparison["test_precision"] - comparison["test_precision_at_0.5"],
        "d_recall": comparison["test_recall"] - comparison["test_recall_at_0.5"],
        "d_f1": comparison["test_f1"] - comparison["test_f1_at_0.5"],
    })
    print(delta.round(4).to_string())

    print("\n[F] VALIDATION vs TEST (generalisation of the frozen selection)")
    gen = comparison[["val_pr_auc", "test_pr_auc", "val_roc_auc", "test_roc_auc"]].copy()
    gen["d_pr_auc"] = gen["test_pr_auc"] - gen["val_pr_auc"]
    gen["d_roc_auc"] = gen["test_roc_auc"] - gen["val_roc_auc"]
    print(gen.round(6).to_string())
    return comparison


# ---------------------------------------------------------- 10.10 UNSW threshold table
def build_unsw_threshold_table(unsw, paths: RunPaths) -> pd.DataFrame:
    """Re-select every UNSW model's threshold on validation, freeze it, score test."""
    d = unsw.inputs.data
    rows = [select_threshold_and_evaluate(
        "GNN + Post-GCN Attention", unsw.val_probs_best.cpu().numpy(), d.y_val,
        unsw.test_probs.cpu().numpy(), d.y_test)]
    for name, key in [("Logistic Regression", "Logistic Regression"),
                      ("Random Forest", "Random Forest"), ("XGBoost", "XGBoost")]:
        b = unsw.baseline_results[key]
        rows.append(select_threshold_and_evaluate(
            name, b["val_prob"], d.y_val, b["test_prob"], d.y_test))

    df = pd.DataFrame(rows)

    print("=" * 100)
    print("UNSW-NB15 FINAL THRESHOLD-SELECTED RESULTS")
    print("=" * 100)
    print(df[["model", "val_threshold", "test_accuracy", "test_precision", "test_recall",
              "test_f1", "test_roc_auc", "test_pr_auc"]].to_string(index=False))
    print("\nUNSW 0.5 REFERENCE")
    print(df[["model", "test_accuracy_at_0.5", "test_precision_at_0.5", "test_recall_at_0.5",
              "test_f1_at_0.5"]].to_string(index=False))
    print("\nUNSW CONFUSION MATRICES - VALIDATION-SELECTED THRESHOLD")
    print(df[["model", "test_tn", "test_fp", "test_fn", "test_tp",
              "test_flagged_pct"]].to_string(index=False))

    paths.make_dirs()
    df.to_csv(paths.backup_dir / "unsw_final_threshold_results.csv", index=False)
    return df


# ------------------------------------------------------------------ result sheets (CSV)
def format_metrics(df):
    out = df.copy()
    for col in ["Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
        if col in out.columns:
            out[col] = out[col].round(4)
    if "Flagged %" in out.columns:
        out["Flagged %"] = out["Flagged %"].round(3)
    return out


def build_result_tables(unsw_df: pd.DataFrame, cse_comparison: pd.DataFrame, paths: RunPaths):
    """Section 10.10: the six clean result tables, displayed and saved as CSV."""
    PRIMARY = ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    REFERENCE = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    CONFUSION = ["Model", "TN", "FP", "FN", "TP", "Flagged %"]

    unsw_primary = unsw_df[["model", "val_threshold", "test_accuracy", "test_precision",
                            "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]].copy()
    unsw_primary.columns = PRIMARY
    unsw_reference = unsw_df[["model", "test_accuracy_at_0.5", "test_precision_at_0.5",
                              "test_recall_at_0.5", "test_f1_at_0.5"]].copy()
    unsw_reference.columns = REFERENCE

    cse = cse_comparison.copy()
    cse["Model"] = cse.index.astype(str)
    cse_primary = cse[["Model", "threshold_selected", "test_accuracy", "test_precision",
                       "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]].copy()
    cse_primary.columns = PRIMARY
    cse_reference = cse[["Model", "test_accuracy_at_0.5", "test_precision_at_0.5",
                         "test_recall_at_0.5", "test_f1_at_0.5"]].copy()
    cse_reference.columns = REFERENCE

    unsw_confusion = unsw_df[["model", "test_tn", "test_fp", "test_fn", "test_tp",
                              "test_flagged_pct"]].copy()
    unsw_confusion.columns = CONFUSION
    cse_confusion = cse[["Model", "test_tn", "test_fp", "test_fn", "test_tp",
                         "test_flagged_pct"]].copy()
    cse_confusion.columns = CONFUSION

    tables = {
        "UNSW_primary_results": ("UNSW-NB15 - PRIMARY RESULTS", format_metrics(unsw_primary)),
        "UNSW_reference_0.5_results": ("UNSW-NB15 - 0.5 REFERENCE", format_metrics(unsw_reference)),
        "UNSW_confusion_matrix": ("UNSW-NB15 - CONFUSION MATRIX", format_metrics(unsw_confusion)),
        "CSE_primary_results": ("CSE-CIC-IDS2018-v3 - PRIMARY RESULTS", format_metrics(cse_primary)),
        "CSE_reference_0.5_results": ("CSE-CIC-IDS2018-v3 - 0.5 REFERENCE",
                                      format_metrics(cse_reference)),
        "CSE_confusion_matrix": ("CSE-CIC-IDS2018-v3 - CONFUSION MATRIX",
                                 format_metrics(cse_confusion)),
    }
    paths.make_dirs()
    for fname, (title, table) in tables.items():
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        print(table.to_string(index=False))
        table.to_csv(paths.backup_dir / f"{fname}.csv", index=False)

    print(f"\nSaved clean result tables to: {paths.backup_dir}")
    return {k: v[1] for k, v in tables.items()}
