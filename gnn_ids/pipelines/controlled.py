"""Section 11 pipeline: controlled GNN+Attention vs GCN-only ablation on UNSW."""
import numpy as np
import pandas as pd
import torch

from ..config import Locked, RunConfig, RunPaths, locked_assert
from ..evaluation.diagnostics import (hostpair_diagnostic, hostpair_summary_frame,
                                      reproducibility_diagnostic)
from ..evaluation.thresholds import select_threshold_trimmed
from ..training.controlled import NAME_ATTN, NAME_GCN_ONLY, ControlledExperiment
from ..utils import get_device


def _guards(paths: RunPaths, cfg: RunConfig):
    """Section 11 never overwrites a checkpoint produced by an earlier section or by itself."""
    ckpts = (paths.s11_attn_checkpoint, paths.s11_gcn_only_checkpoint)
    if cfg.fresh:
        for p in ckpts:
            if p.exists():
                print(f"--fresh: deleting {p}")
                p.unlink()
    protected = {paths.unsw_checkpoint, paths.cse_checkpoint}
    for p in ckpts:
        assert p not in protected, f"{p} collides with a protected checkpoint"
        assert not p.exists(), (
            f"{p} already exists. Section 11 refuses to overwrite it. "
            f"Delete it deliberately (or pass --fresh) if a clean restart is intended.")
    assert paths.unsw_checkpoint.exists(), f"Historical checkpoint missing: {paths.unsw_checkpoint}"
    return protected


def run_controlled(paths: RunPaths, cfg: RunConfig, inputs, pos_weight_value, history,
                   val_probs_best, unsw_threshold_df, device=None):
    """Run notebook section 11 end to end and persist all CSVs to ``final_experiment_backup``."""
    device = device or get_device()
    paths.make_dirs()
    bk = paths.backup_dir
    protected = _guards(paths, cfg)

    graph, val, test, vocab = inputs.graph, inputs.val, inputs.test, inputs.vocab
    n_val, n_test = val.X.shape[0], test.X.shape[0]
    locked_assert((n_val, n_test) == (Locked.UNSW_VAL_FLOWS, Locked.UNSW_TEST_FLOWS))

    # Historical run log (used only for the reproducibility comparison below)
    locked_assert(len(history) == Locked.UNSW_HISTORY_LEN,
                  f"Expected the {Locked.UNSW_HISTORY_LEN}-epoch historical log, got {len(history)}")
    locked_assert(len(history) > 3 and abs(history[3]["f1"] - Locked.UNSW_EPOCH4_F1) < 5e-5,
                  "historical epoch-4 validation F1 differs from the locked value")

    exp = ControlledExperiment(graph, val, test, device, cfg.num_epochs, protected,
                               pos_weight_value, vocab.num_nodes)

    # ------------------------------------------------ 11.1 GNN + Attention (controlled)
    s11_attn = exp.build_attn()
    attn_stage1 = exp.train_stage1(s11_attn, paths.s11_attn_checkpoint)

    # Retrace check against the historical run (diagnostic only).
    TOL_AUC, TOL_THR = 1e-3, 5e-3
    new = attn_stage1["history"]
    n_cmp = min(len(new), len(history))
    print("\n" + "=" * 96)
    print(f"RETRACE CHECK vs historical Section 7.2.2 log (epochs 1-{n_cmp})")
    print("=" * 96)
    worst_auc, worst_thr = 0.0, 0.0
    for i in range(n_cmp):
        h, n = history[i], new[i]
        d_roc = n["roc_auc"] - h["roc_auc"]
        d_f1 = n["f1_at_0.5"] - h["f1"]
        d_rec = n["recall_at_0.5"] - h["recall"]
        worst_auc = max(worst_auc, abs(d_roc))
        worst_thr = max(worst_thr, abs(d_f1), abs(d_rec))
        print(f"{i + 1:>3} | ROC {h['roc_auc']:.4f} -> {n['roc_auc']:.4f} ({d_roc:+.2e}) | "
              f"F1@.5 {h['f1']:.4f} -> {n['f1_at_0.5']:.4f} ({d_f1:+.2e}) | R@.5 diff {d_rec:+.2e}")
    retrace_ok = worst_auc <= TOL_AUC and worst_thr <= TOL_THR
    print("RETRACE OK - only the selection rule differs from the historical run" if retrace_ok
          else "RETRACE DIVERGENCE - something other than the selection rule changed")
    pd.DataFrame(new).to_csv(bk / "s11_attn_prauc_stage1_history.csv", index=False)

    # Stage 2 - threshold selection
    # Threshold-rule pre-check: applying the rule to the historical epoch-4 validation
    # probabilities must reproduce the recorded UNSW threshold (validation data only).
    thr_hist = select_threshold_trimmed(val.y.numpy().astype(int), val_probs_best.numpy())
    thr_rec = float(unsw_threshold_df.loc[
        unsw_threshold_df["model"] == "GNN + Post-GCN Attention", "val_threshold"].iloc[0])
    assert abs(thr_hist - thr_rec) < 1e-12, f"Rule mismatch: {thr_hist} vs {thr_rec}"
    print(f"Threshold-rule pre-check: historical probs -> {thr_hist:.6f} "
          f"(recorded {thr_rec:.6f}) PASS\n")

    ck = torch.load(paths.s11_attn_checkpoint, map_location="cpu", weights_only=False)
    locked_assert(ck["epoch"] == attn_stage1["best_epoch"] == Locked.S11_ATTN_EPOCH)
    locked_assert(abs(ck["val_pr_auc"] - Locked.S11_ATTN_VAL_PR_AUC) < 5e-7,
                  f"Saved PR-AUC {ck['val_pr_auc']}")
    del ck

    attn_stage2 = exp.stage2(exp.build_attn, paths.s11_attn_checkpoint)
    pd.DataFrame([
        {"model": NAME_ATTN, "threshold": "selected", **attn_stage2["val_at_selected"]},
        {"model": NAME_ATTN, "threshold": "0.5", **attn_stage2["val_at_0.5"]},
    ]).assign(val_pr_auc=attn_stage2["val_ranking"]["pr_auc"],
              val_roc_auc=attn_stage2["val_ranking"]["roc_auc"],
              best_epoch=attn_stage1["best_epoch"]).to_csv(
        bk / "s11_attn_prauc_stage2_validation.csv", index=False)

    # ------------------------------------------------ 11.2 GCN-only ablation
    s11_gcn_only = exp.build_gcn_only()
    gcn_stage1 = exp.train_stage1(s11_gcn_only, paths.s11_gcn_only_checkpoint)

    a, g = attn_stage1["history"], gcn_stage1["history"]
    print("\n" + "=" * 92)
    print("VALIDATION LOG SIDE BY SIDE (Stage 1 only; not used for any decision)")
    print("=" * 92)
    for i in range(max(len(a), len(g))):
        fa = lambda k: f"{a[i][k]:.6f}" if i < len(a) else "-"
        fg = lambda k: f"{g[i][k]:.6f}" if i < len(g) else "-"
        print(f"{i + 1:>3} | PR-AUC attn {fa('pr_auc'):>9} gcn {fg('pr_auc'):>9} | "
              f"ROC attn {fa('roc_auc'):>9} gcn {fg('roc_auc'):>9}")
    print(f"best | Attn epoch {attn_stage1['best_epoch']} PR-AUC "
          f"{attn_stage1['best_val_pr_auc']:.6f} | GCN-only epoch "
          f"{gcn_stage1['best_epoch']} PR-AUC {gcn_stage1['best_val_pr_auc']:.6f}")
    pd.DataFrame(g).to_csv(bk / "s11_gcn_only_stage1_history.csv", index=False)

    ck = torch.load(paths.s11_gcn_only_checkpoint, map_location="cpu", weights_only=False)
    locked_assert(ck["epoch"] == gcn_stage1["best_epoch"] == Locked.S11_GCN_EPOCH,
                  f"Saved epoch {ck['epoch']}")
    del ck

    gcn_stage2 = exp.stage2(exp.build_gcn_only, paths.s11_gcn_only_checkpoint)

    ck = torch.load(paths.s11_gcn_only_checkpoint, map_location="cpu", weights_only=False)
    lo, hi = ck["val_prob_range"]
    del ck
    n_unique = int(np.unique(gcn_stage2["val_prob"]).size)
    pd.DataFrame([
        {"model": NAME_GCN_ONLY, "threshold": "selected", **gcn_stage2["val_at_selected"]},
        {"model": NAME_GCN_ONLY, "threshold": "0.5", **gcn_stage2["val_at_0.5"]},
    ]).assign(val_pr_auc=gcn_stage2["val_ranking"]["pr_auc"],
              val_roc_auc=gcn_stage2["val_ranking"]["roc_auc"],
              val_prob_min=lo, val_prob_max=hi, val_prob_unique=n_unique,
              best_epoch=gcn_stage1["best_epoch"]).to_csv(
        bk / "s11_gcn_only_stage2_validation.csv", index=False)

    # 11.2.3 diagnostic: read-only, validation only
    reproducibility_diagnostic(exp, vocab, gcn_stage2, attn_stage2,
                               paths.s11_gcn_only_checkpoint, paths.s11_attn_checkpoint)

    # ------------------------------------------------ 11.3 final evaluation & comparison
    assert len(exp.heldout_accessed) == 0, \
        f"Held-out split already accessed for {sorted(exp.heldout_accessed)}; 11.3 runs once"

    # Both thresholds are frozen BEFORE any test access; _EXPECTED is the stale-model guard.
    expected = {
        paths.s11_attn_checkpoint: (NAME_ATTN, Locked.S11_ATTN_EPOCH, Locked.S11_ATTN_THRESHOLD,
                                    ["gcn", "attention", "classifier"]),
        paths.s11_gcn_only_checkpoint: (NAME_GCN_ONLY, Locked.S11_GCN_EPOCH,
                                        Locked.S11_GCN_THRESHOLD, ["gcn", "classifier"]),
    }
    frozen = {}
    for path, (name, ep, thr_exp, keys) in expected.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        thr = ck["threshold_selected"]
        assert ck["model_name"] == name
        locked_assert(ck["epoch"] == ep)
        assert ck["module_keys"] == keys
        assert ck["selection_metric"] == "validation PR-AUC"
        assert isinstance(thr, float) and np.isfinite(thr), f"{name}: threshold not frozen"
        locked_assert(abs(thr - thr_exp) < 5e-7, f"{name}: threshold {thr} != frozen {thr_exp}")
        assert ck["threshold_reference"] == 0.5
        frozen[name] = thr
        del ck
    print("Both thresholds frozen before test access:")
    for n_, t_ in frozen.items():
        print(f"  {n_:<40} {t_:.8f}")

    # Stage 3: exactly one test evaluation per model
    res_attn = exp.stage3(exp.build_attn, paths.s11_attn_checkpoint)
    res_gcn = exp.stage3(exp.build_gcn_only, paths.s11_gcn_only_checkpoint)

    assert exp.heldout_accessed == {NAME_ATTN, NAME_GCN_ONLY}
    assert res_attn["threshold"] == frozen[NAME_ATTN]
    assert res_gcn["threshold"] == frozen[NAME_GCN_ONLY]
    for r in (res_attn, res_gcn):
        for m in (r["at_selected"], r["at_0.5"]):
            assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == n_test
            assert m["fn"] + m["tp"] == int(test.y.sum())

    def _row(res, stage2, stage1, which):
        m = res["at_selected"] if which == "selected" else res["at_0.5"]
        return {"model": res["name"], "threshold_type": which,
                "threshold": m["threshold"], "best_epoch": res["epoch"],
                "val_pr_auc": stage2["val_ranking"]["pr_auc"],
                "val_roc_auc": stage2["val_ranking"]["roc_auc"],
                "test_roc_auc": res["ranking"]["roc_auc"],
                "test_pr_auc": res["ranking"]["pr_auc"],
                "test_accuracy": m["accuracy"], "test_precision": m["precision"],
                "test_recall": m["recall"], "test_f1": m["f1"],
                "test_tn": m["tn"], "test_fp": m["fp"], "test_fn": m["fn"], "test_tp": m["tp"],
                "test_flagged_pct": m["flagged_pct"],
                "n_params": sum(p.numel() for p in (s11_attn if res["name"] == NAME_ATTN
                                                    else s11_gcn_only)["params"]),
                "stage1_epochs_run": len(stage1["history"]),
                "stage1_time_s": stage1["total_time_s"]}

    final_results = pd.DataFrame([
        _row(res_attn, attn_stage2, attn_stage1, "selected"),
        _row(res_attn, attn_stage2, attn_stage1, "0.5"),
        _row(res_gcn, gcn_stage2, gcn_stage1, "selected"),
        _row(res_gcn, gcn_stage2, gcn_stage1, "0.5"),
    ])

    MET = ["test_accuracy", "test_precision", "test_recall", "test_f1",
           "test_roc_auc", "test_pr_auc", "test_flagged_pct"]
    deltas = []
    for t in ["selected", "0.5"]:
        a_ = final_results.query("model == @NAME_ATTN and threshold_type == @t").iloc[0]
        g_ = final_results.query("model == @NAME_GCN_ONLY and threshold_type == @t").iloc[0]
        deltas.append({"threshold_type": t, **{f"delta_{k}": g_[k] - a_[k] for k in MET}})
    final_deltas = pd.DataFrame(deltas)

    # Context only: historical F1@0.5-selected GNN+Attention (different selection protocol)
    h = unsw_threshold_df.set_index("model").loc["GNN + Post-GCN Attention"]
    historical_context = pd.DataFrame([{
        "model": "GNN+Attention (historical, F1@0.5-selected, epoch 4)",
        "threshold": h["val_threshold"], "test_accuracy": h["test_accuracy"],
        "test_precision": h["test_precision"], "test_recall": h["test_recall"],
        "test_f1": h["test_f1"], "test_roc_auc": h["test_roc_auc"],
        "test_pr_auc": h["test_pr_auc"], "test_tn": h["test_tn"], "test_fp": h["test_fp"],
        "test_fn": h["test_fn"], "test_tp": h["test_tp"],
        "test_flagged_pct": h["test_flagged_pct"]}])

    pd.set_option("display.width", 200)
    print("\n" + "=" * 110)
    print(f"UNSW-NB15 CONTROLLED ABLATION - TEST RESULTS | {n_test:,} flows | "
          f"{int(test.y.sum()):,} attacks ({100 * float(test.y.mean()):.3f}%)")
    print("=" * 110)
    print("\n[A] THRESHOLD-INDEPENDENT (computed once per model)")
    print(final_results.query("threshold_type == 'selected'")
          .set_index("model")[["best_epoch", "val_pr_auc", "test_roc_auc", "test_pr_auc"]]
          .round(6).to_string())
    cols = ["threshold", "test_accuracy", "test_precision", "test_recall", "test_f1",
            "test_tn", "test_fp", "test_fn", "test_tp", "test_flagged_pct"]
    print("\n[B] PRIMARY - validation-selected threshold (frozen before test)")
    print(final_results.query("threshold_type == 'selected'").set_index("model")[cols]
          .round(6).to_string())
    print("\n[C] SECONDARY - reference threshold 0.5")
    print(final_results.query("threshold_type == '0.5'").set_index("model")[cols]
          .round(6).to_string())
    print("\n[D] DELTAS (GCN-only minus controlled GNN+Attention)")
    print(final_deltas.set_index("threshold_type").round(6).to_string())
    print("\n[E] CONTEXT ONLY - historical GNN+Attention (F1@0.5 checkpoint selection; "
          "NOT like-for-like)")
    print(historical_context.set_index("model").round(6).to_string())
    print("\n[F] CAVEATS")
    for line in CAVEATS:
        print("  " + line)

    final_results.to_csv(bk / "s11_final_test_results.csv", index=False)
    final_deltas.to_csv(bk / "s11_final_deltas_gcn_only_minus_attention.csv", index=False)
    historical_context.to_csv(bk / "s11_historical_attention_context.csv", index=False)
    np.save(bk / "s11_test_prob_attn_prauc.npy", res_attn["prob"])
    np.save(bk / "s11_test_prob_gcn_only.npy", res_gcn["prob"])

    # ------------------------------------------------ 11.4 host-pair diagnostic
    heldout_before = set(exp.heldout_accessed)
    assert heldout_before == {NAME_ATTN, NAME_GCN_ONLY}, "Unexpected held-out state"
    for p, t in [(paths.s11_attn_checkpoint, float(attn_stage2["threshold"])),
                 (paths.s11_gcn_only_checkpoint, float(gcn_stage2["threshold"]))]:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert abs(ck["threshold_selected"] - t) < 1e-8, f"Frozen threshold differs in {p}"
        del ck

    R = hostpair_diagnostic(exp, vocab, gcn_stage2, attn_stage2, n_val)
    hostpair_summary_frame(R).to_csv(bk / "s11_hostpair_diagnostic_summary.csv", index=False)
    R["per_pair"].reset_index(names="unordered_pair_key").to_csv(
        bk / "s11_hostpair_diagnostic_per_pair.csv", index=False)
    assert exp.heldout_accessed == heldout_before, "Held-out access state changed"

    return {"attn_stage1": attn_stage1, "attn_stage2": attn_stage2,
            "gcn_only_stage1": gcn_stage1, "gcn_only_stage2": gcn_stage2,
            "res_attn": res_attn, "res_gcn": res_gcn, "final_results": final_results,
            "final_deltas": final_deltas, "historical_context": historical_context,
            "hostpair": R}


CAVEATS = [
    "1. Not a pure attention-only ablation: removing attention also removes the only",
    "   pathway by which the 49 flow features reach the GNN classifier. The comparison is",
    "   'graph + attention + flow features' vs 'graph structure alone'.",
    "2. GCN-only combines endpoints with mean(h_src, h_dst): order-invariant, and with",
    "   all-ones node features every flow on the same host pair receives the same score.",
    "3. GCN-only's best validation PR-AUC came from epoch 1 (one optimizer step); later",
    "   epochs ranked worse.",
    "4. CUDA nondeterminism: PyG GCN aggregation uses non-deterministic GPU scatter-add.",
    "   For GCN-only, identical-weight GPU rescorings of validation varied PR-AUC by ~2e-4",
    "   (tied host-pair score blocks amplify ~1e-7 noise); CPU scoring is deterministic.",
    "   Test was scored once on GPU; results are reported as produced.",
    "5. Random stratified flow-level split (not temporal); absolute figures are optimistic",
    "   relative to deployment. One test flow maps to the placeholder node.",
]
