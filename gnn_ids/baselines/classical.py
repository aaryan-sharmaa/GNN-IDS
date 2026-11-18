"""Classical flow-level baselines (Logistic Regression, Random Forest, XGBoost).

They use the same flows, the same split and the same 49 scaled features as the GNN, with no
graph information. UNSW baselines are scored at the fixed 0.5 threshold (notebook section 9);
CSE baselines follow the three-stage protocol with a validation-selected threshold (10.8).
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import BASELINE_SEED, N_FEATURES, Locked, locked_assert
from ..evaluation.metrics import eval_baseline, metrics_at
from ..evaluation.thresholds import select_threshold_full_curve


# ======================================================================== UNSW (section 9)
def run_unsw_baselines(data, graph):
    """Fit and score LR / RF / XGBoost on the UNSW splits.

    Returns ``baseline_results`` (name -> dict with val / test metrics, timings and the
    validation and test probability arrays).
    """
    # ---------------- Fairness: same training population as the GNN ----------------
    assert len(data.X_train) == graph.flow_src.shape[0]
    locked_assert(len(data.X_train) == Locked.UNSW_TRAIN_FLOWS,
                  "Baselines must use the GNN's 496,738 flows")
    assert data.X_train.index.equals(data.y_train.index), "Train feature/label index mismatch"
    locked_assert(len(data.X_val) == len(data.y_val) == Locked.UNSW_VAL_FLOWS)
    locked_assert(len(data.X_test) == len(data.y_test) == Locked.UNSW_TEST_FLOWS)
    assert list(data.X_train.columns) == list(data.X_val.columns) \
        == list(data.X_test.columns) == data.feature_cols
    assert len(data.feature_cols) == N_FEATURES

    FORBIDDEN = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS",
                 "FLOW_END_MILLISECONDS", "Label", "Attack", "orig_row_id",
                 "node", "emb", "alpha", "degree"]
    leaked = [c for c in data.feature_cols if any(f.lower() in c.lower() for f in FORBIDDEN)]
    assert not leaked, f"Graph/leakage columns present in features: {leaked}"

    # ---------------- Numpy views (float32 to limit memory) ----------------
    X_tr_np = data.X_train.to_numpy(dtype=np.float32)
    y_tr_np = data.y_train.to_numpy(dtype=np.int64)
    X_val_np = data.X_val.to_numpy(dtype=np.float32)
    y_val_np = data.y_val.to_numpy(dtype=np.int64)
    X_te_np = data.X_test.to_numpy(dtype=np.float32)
    y_te_np = data.y_test.to_numpy(dtype=np.int64)

    # Baseline labels must be the same targets the GNN trained on
    assert np.array_equal(y_tr_np, graph.flow_label.numpy()), \
        "Baseline training labels differ from the GNN's flow labels"

    # ---------------- Imbalance weight (training labels ONLY) ----------------
    bl_n_pos = int(y_tr_np.sum())
    bl_n_neg = int(len(y_tr_np) - bl_n_pos)
    scale_pos_weight = bl_n_neg / bl_n_pos

    results = {}

    def _score(label, model, train_time, extra=None):
        val_prob = model.predict_proba(X_val_np)[:, 1]
        val = eval_baseline(f"{label} (validation)", y_val_np, val_prob)
        t0 = time.time()
        test_prob = model.predict_proba(X_te_np)[:, 1]
        infer_time = time.time() - t0
        test = eval_baseline(f"{label} (test)", y_te_np, test_prob)
        print(f"Inference time: {infer_time:.1f}s for {len(y_te_np):,} flows")
        entry = {"val": val, "test": test, "train_time": train_time,
                 "val_prob": val_prob, "test_prob": test_prob}
        if extra:
            entry.update(extra)
        return entry

    # ---------------- 9.2 Logistic Regression ----------------
    lr_model = LogisticRegression(class_weight="balanced", random_state=BASELINE_SEED,
                                  solver="lbfgs", max_iter=1000, n_jobs=-1)
    t0 = time.time()
    lr_model.fit(X_tr_np, y_tr_np)
    lr_train_time = time.time() - t0
    print(f"LR training time {lr_train_time:.1f}s | iterations {int(lr_model.n_iter_[0])}/1000")
    results["Logistic Regression"] = _score("LR", lr_model, lr_train_time)

    # ---------------- 9.3 Random Forest ----------------
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", random_state=BASELINE_SEED, n_jobs=-1)
    t0 = time.time()
    rf_model.fit(X_tr_np, y_tr_np)
    rf_train_time = time.time() - t0
    print(f"RF training time {rf_train_time:.1f}s")
    results["Random Forest"] = _score("RF", rf_model, rf_train_time)

    # ---------------- 9.4 XGBoost (validation used for early stopping only) ----------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, objective="binary:logistic", eval_metric="aucpr",
        early_stopping_rounds=20, tree_method="hist", random_state=BASELINE_SEED, n_jobs=-1)
    t0 = time.time()
    xgb_model.fit(X_tr_np, y_tr_np, eval_set=[(X_val_np, y_val_np)], verbose=False)
    xgb_train_time = time.time() - t0
    print(f"XGB training time {xgb_train_time:.1f}s | best iteration "
          f"{getattr(xgb_model, 'best_iteration', None)} | best val aucpr "
          f"{getattr(xgb_model, 'best_score', None)}")
    results["XGBoost"] = _score("XGB", xgb_model, xgb_train_time,
                                {"best_iteration": int(xgb_model.best_iteration)})
    return results


def baseline_comparison_table(gnn_test_metrics, baseline_results, n_test, threshold):
    """Section 9.5: one test-set table for the GNN and the three baselines."""
    all_results = {"GNN + Post-GCN Attention": gnn_test_metrics}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        all_results[name] = baseline_results[name]["test"]

    rows = []
    for name, m in all_results.items():
        tn, fp, fn, tp = m["cm"].ravel()
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "F1": m["f1"], "ROC-AUC": m["roc_auc"],
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    comparison_df = pd.DataFrame(rows).set_index("Model")

    print("=" * 96)
    print(f"TEST-SET COMPARISON - NF-UNSW-NB15-v3 | {n_test:,} flows | threshold {threshold}")
    print("=" * 96)
    print(comparison_df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].round(4).to_string())
    print("\nConfusion matrices (TN / FP / FN / TP)")
    print(comparison_df[["TN", "FP", "FN", "TP"]].to_string())
    return comparison_df


# ================================================================== CSE (section 10.8)
class CseBaselineProtocol:
    """Three-stage evaluation shared by the three CSE baselines.

    Stage 1: fit on training data (validation only for XGBoost early stopping).
    Stage 2: select the F1-maximising threshold on validation and freeze it.
    Stage 3: evaluate the test split once at that threshold (0.5 as a reference).
    """

    def __init__(self, data):
        self.data = data
        self.y_tr = data.y_train.to_numpy(dtype=np.int64)
        self.y_va = data.y_val.to_numpy(dtype=np.int64)
        self.y_te = data.y_test.to_numpy(dtype=np.int64)
        n_tr, n_va, n_te = data.n_tr, data.n_va, data.n_te

        assert data.X_train.shape == (n_tr, N_FEATURES) and data.X_train.dtype == np.float32
        assert data.X_val.shape == (n_va, N_FEATURES) and data.X_test.shape == (n_te, N_FEATURES)
        assert len(self.y_tr) == n_tr and len(self.y_va) == n_va and len(self.y_te) == n_te
        assert data.scaler.n_samples_seen_ == n_tr, "Scaler was not fit on training rows only"
        assert np.isfinite(data.X_train).all() and np.isfinite(data.X_val).all() \
            and np.isfinite(data.X_test).all()

        self.seed = BASELINE_SEED
        n_pos = int(self.y_tr.sum())
        self.scale_pos_weight = (n_tr - n_pos) / n_pos
        self.rows = []

    # -------------------------------------------------------------- STAGE 2
    def stage2_select_threshold(self, name, model, cfg_lines):
        """Touches validation data only. Returns a dict carrying the frozen threshold."""
        d = self.data
        val_prob = model.predict_proba(d.X_val)[:, 1]

        thr_sel = select_threshold_full_curve(self.y_va, val_prob)["threshold"]

        val_roc = roc_auc_score(self.y_va, val_prob)
        val_pr = average_precision_score(self.y_va, val_prob)
        v_sel = metrics_at(self.y_va, val_prob, thr_sel)
        v_05 = metrics_at(self.y_va, val_prob, 0.5)

        print(f"\n{'-' * 74}\n{name} - STAGE 1 + 2 (training and validation only)\n{'-' * 74}")
        for line in cfg_lines:
            print(f"  {line}")
        print(f"  validation ROC-AUC {val_roc:.6f} | PR-AUC {val_pr:.6f} "
              f"(random baseline {self.y_va.mean():.6f})")
        print(f"  selected threshold : {thr_sel:.6f}  <- FROZEN, from validation labels only")
        print(f"  at selected : acc {v_sel['accuracy']:.4f} P {v_sel['precision']:.4f} "
              f"R {v_sel['recall']:.4f} F1 {v_sel['f1']:.4f}")
        print(f"  at 0.5      : acc {v_05['accuracy']:.4f} P {v_05['precision']:.4f} "
              f"R {v_05['recall']:.4f} F1 {v_05['f1']:.4f}  (reference)")

        return {"name": name, "model": model, "threshold_selected": thr_sel,
                "val_roc_auc": val_roc, "val_pr_auc": val_pr,
                "val_f1_selected": v_sel["f1"], "val_prob": val_prob}

    # -------------------------------------------------------------- STAGE 3
    def stage3_evaluate_test(self, frozen, extra=None):
        """FIRST test access for this model."""
        thr_sel = frozen["threshold_selected"]
        name, model = frozen["name"], frozen["model"]

        assert isinstance(thr_sel, float) and np.isfinite(thr_sel), \
            f"{name}: threshold not frozen; Stage 2 must complete before Stage 3"
        assert "val_prob" in frozen and frozen["val_prob"] is not None, \
            f"{name}: validation probabilities missing; Stage 2 did not run"

        print(f"\n{'=' * 74}")
        print(f"TEST ACCESS - model and threshold frozen ({name}, threshold {thr_sel:.6f})")
        print(f"{'=' * 74}")

        test_prob = model.predict_proba(self.data.X_test)[:, 1]   # first and only test access

        test_roc = roc_auc_score(self.y_te, test_prob)
        test_pr = average_precision_score(self.y_te, test_prob)
        t_sel = metrics_at(self.y_te, test_prob, thr_sel)
        t_05 = metrics_at(self.y_te, test_prob, 0.5)

        print(f"  ROC-AUC {test_roc:.6f} | PR-AUC {test_pr:.6f} "
              f"| lift {test_pr / self.y_te.mean():.2f}x")
        for lbl, m in [("PRIMARY (val-selected)", t_sel), ("REFERENCE (0.5)", t_05)]:
            print(f"  {lbl} @ {m['threshold']:.6f}: acc {m['accuracy']:.4f} | "
                  f"P {m['precision']:.4f} | R {m['recall']:.4f} | F1 {m['f1']:.4f} | "
                  f"TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,}")

        row = {
            "model": name, "threshold_selected": thr_sel,
            "val_roc_auc": frozen["val_roc_auc"], "val_pr_auc": frozen["val_pr_auc"],
            "val_f1_selected": frozen["val_f1_selected"],
            "test_roc_auc": test_roc, "test_pr_auc": test_pr,
            "test_accuracy": t_sel["accuracy"], "test_precision": t_sel["precision"],
            "test_recall": t_sel["recall"], "test_f1": t_sel["f1"],
            "test_tn": t_sel["tn"], "test_fp": t_sel["fp"],
            "test_fn": t_sel["fn"], "test_tp": t_sel["tp"],
            "test_flagged_pct": t_sel["flagged_pct"],
            "test_accuracy_at_0.5": t_05["accuracy"],
            "test_precision_at_0.5": t_05["precision"],
            "test_recall_at_0.5": t_05["recall"], "test_f1_at_0.5": t_05["f1"],
            "test_tn_at_0.5": t_05["tn"], "test_fp_at_0.5": t_05["fp"],
            "test_fn_at_0.5": t_05["fn"], "test_tp_at_0.5": t_05["tp"],
            "test_flagged_pct_at_0.5": t_05["flagged_pct"],
        }
        if extra:
            row.update(extra)
        self.rows.append(row)
        return row, test_prob


def run_cse_baselines(data, unsw_feature_cols):
    """Section 10.8: LR, RF and XGBoost on the CSE flows. Returns the results DataFrame."""
    assert data.feature_cols == unsw_feature_cols, "CSE features differ from UNSW"
    proto = CseBaselineProtocol(data)
    n_tr, n_va, n_te = data.n_tr, data.n_va, data.n_te
    seed = proto.seed
    sizes = f"train {n_tr:,} | val {n_va:,} | test {n_te:,} | features 49"

    # ---- Logistic Regression ----
    lr = LogisticRegression(class_weight="balanced", random_state=seed,
                            solver="lbfgs", max_iter=1000, n_jobs=-1)
    t0 = time.time()
    lr.fit(data.X_train, proto.y_tr)
    lr_time = time.time() - t0
    assert lr.n_features_in_ == N_FEATURES
    frozen = proto.stage2_select_threshold(
        "Logistic Regression", lr,
        [sizes,
         f"solver lbfgs | C 1.0 | class_weight balanced | seed {seed}",
         f"iterations {int(lr.n_iter_[0])}/1000 "
         f"({'converged' if int(lr.n_iter_[0]) < 1000 else 'HIT LIMIT'}) | "
         f"train time {lr_time:.1f}s"])
    proto.stage3_evaluate_test(frozen, extra={"train_time_s": lr_time})

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    t0 = time.time()
    rf.fit(data.X_train, proto.y_tr)
    rf_time = time.time() - t0
    assert rf.n_features_in_ == N_FEATURES
    frozen = proto.stage2_select_threshold(
        "Random Forest", rf,
        [sizes,
         "100 trees | max_depth None | min_samples_leaf 5 | max_features sqrt",
         f"class_weight balanced_subsample | seed {seed} | train time {rf_time / 60:.1f} min"])
    proto.stage3_evaluate_test(frozen, extra={"train_time_s": rf_time})

    # ---- XGBoost (validation used for early stopping only) ----
    xgb_m = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=proto.scale_pos_weight, objective="binary:logistic",
        eval_metric="aucpr", early_stopping_rounds=20, tree_method="hist",
        random_state=seed, n_jobs=-1)
    t0 = time.time()
    xgb_m.fit(data.X_train, proto.y_tr, eval_set=[(data.X_val, proto.y_va)], verbose=False)
    xgb_time = time.time() - t0
    assert xgb_m.n_features_in_ == N_FEATURES
    frozen = proto.stage2_select_threshold(
        "XGBoost", xgb_m,
        [sizes,
         "max_depth 6 | lr 0.1 | subsample 0.8 | colsample 0.8 | hist",
         f"scale_pos_weight {proto.scale_pos_weight:.4f} | eval_metric aucpr | seed {seed}",
         f"best iteration {xgb_m.best_iteration} of 500 cap | best val aucpr "
         f"{xgb_m.best_score:.6f}",
         "early stopping used VALIDATION only; test never seen during fit"])
    proto.stage3_evaluate_test(frozen, extra={"train_time_s": xgb_time,
                                              "best_iteration": int(xgb_m.best_iteration)})

    results = pd.DataFrame(proto.rows).set_index("model")
    print("=" * 100)
    print(f"CSE BASELINE TEST RESULTS | {n_te:,} flows | {int(proto.y_te.sum()):,} positives "
          f"({100 * proto.y_te.mean():.3f}%)")
    print("=" * 100)
    print(results[["threshold_selected", "test_accuracy", "test_precision", "test_recall",
                   "test_f1", "test_roc_auc", "test_pr_auc"]].round(6).to_string())
    return results, proto.y_te
