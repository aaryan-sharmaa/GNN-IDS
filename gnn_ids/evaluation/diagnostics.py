"""Read-only, validation-only diagnostics for section 11.

* ``reproducibility_diagnostic`` (11.7-D): is the GCN-only Stage-2 PR-AUC reproducible, and
  how fragile is its frozen threshold to GPU scatter-add noise?
* ``hostpair_diagnostic`` (11.4): oracle ceilings for any predictor that only sees the host pair.

Neither trains, changes a threshold, writes a checkpoint, or touches the held-out split.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import GCN_HIDDEN_DIM, GCN_OUT_DIM, CLS_HIDDEN_DIM
from ..models.classifier import FlowClassifier
from ..models.gcn import GCNEncoder
from .metrics import threshold_metrics


def hostpair_diagnostic(exp, vocab, gcn_stage2, attn_stage2, n_val_expected):
    """Validation-only host-pair analysis. Reads no held-out objects."""
    val = exp.val
    yv = val.y.numpy().astype(int)
    s, d = val.src.numpy(), val.dst.numpy()
    num_nodes = vocab.num_nodes
    p_gcn = np.asarray(gcn_stage2["val_prob"])      # frozen Stage-2, actual model
    p_att = np.asarray(attn_stage2["val_prob"])     # frozen Stage-2, contrast only
    assert len(yv) == len(s) == len(p_gcn) == len(p_att) == n_val_expected
    thr_g = float(gcn_stage2["threshold"])

    df = pd.DataFrame({"u": np.minimum(s, d) * num_nodes + np.maximum(s, d),
                       "dpair": s * num_nodes + d, "y": yv, "p_gcn": p_gcn, "p_att": p_att})
    df["score_block"] = pd.factorize(p_gcn)[0]                # model's own distinct-score blocks

    def ceiling(key):
        """ORACLE: score each flow by its group's validation attack rate (in-sample)."""
        g = df.groupby(key)["y"].agg(n="size", a="sum")
        g["rate"] = g["a"] / g["n"]
        oracle = df[key].map(g["rate"]).to_numpy()
        # Exact max F1 over all subsets of groups: prefixes in attack-rate order
        gs = g.sort_values("rate", ascending=False)
        tp, fp = gs["a"].cumsum().to_numpy(), (gs["n"] - gs["a"]).cumsum().to_numpy()
        P = int(yv.sum())
        f1 = 2 * tp / (2 * tp + fp + (P - tp))
        irreducible = int(np.minimum(g["a"], g["n"] - g["a"]).sum())
        return {"groups": len(g), "pr_auc": average_precision_score(yv, oracle),
                "roc_auc": roc_auc_score(yv, oracle), "max_f1": float(f1.max()),
                "max_accuracy": 1 - irreducible / len(yv), "irreducible_errors": irreducible,
                "mixed_groups": int(((g["rate"] > 0) & (g["rate"] < 1)).sum()),
                "flows_in_mixed": int(g.loc[(g["rate"] > 0) & (g["rate"] < 1), "n"].sum()),
                "table": g}

    c_dir, c_und, c_blk = ceiling("dpair"), ceiling("u"), ceiling("score_block")

    # Actual frozen GCN-only validation results (recomputed from frozen probs; must match)
    act = threshold_metrics(yv, p_gcn, thr_g)
    act_pr = average_precision_score(yv, p_gcn)
    act_roc = roc_auc_score(yv, p_gcn)
    assert abs(act["f1"] - gcn_stage2["val_at_selected"]["f1"]) < 1e-12
    assert abs(act_pr - gcn_stage2["val_ranking"]["pr_auc"]) < 1e-12

    # Within-pair prediction spread
    spread_u_g = df.groupby("u")["p_gcn"].agg(lambda x: x.max() - x.min())
    spread_d_g = df.groupby("dpair")["p_gcn"].agg(lambda x: x.max() - x.min())
    spread_u_a = df.groupby("u")["p_att"].agg(lambda x: x.max() - x.min())
    # Order invariance: pairs observed in both directions
    dmean = df.groupby("dpair")["p_gcn"].first()
    both_dir = [(k, (k % num_nodes) * num_nodes + k // num_nodes) for k in dmean.index
                if (k % num_nodes) * num_nodes + k // num_nodes in dmean.index
                and k // num_nodes < k % num_nodes]
    dir_diff = max((abs(dmean[a] - dmean[b]) for a, b in both_dir), default=0.0)

    per_pair = c_und["table"].copy()
    per_pair["node_a"], per_pair["node_b"] = per_pair.index // num_nodes, per_pair.index % num_nodes
    _ip = lambda n: "PLACEHOLDER" if n == vocab.placeholder_id else str(vocab.node_ips[n])
    per_pair["ip_a"] = per_pair["node_a"].map(_ip)
    per_pair["ip_b"] = per_pair["node_b"].map(_ip)
    per_pair["gcn_only_score"] = df.groupby("u")["p_gcn"].mean()
    per_pair["gcn_only_within_spread"] = spread_u_g
    per_pair["gcn_only_flagged_at_frozen_thr"] = per_pair["gcn_only_score"] >= thr_g
    per_pair["attn_within_spread"] = spread_u_a
    per_pair = per_pair.rename(columns={"n": "val_flows", "a": "val_attacks",
                                        "rate": "val_attack_rate"})

    R = dict(c_dir=c_dir, c_und=c_und, c_blk=c_blk, act=act, act_pr=act_pr,
             act_roc=act_roc, base=float(yv.mean()), n=len(yv),
             sizes=c_und["table"]["n"], spread_u_g=spread_u_g, spread_d_g=spread_d_g,
             spread_u_a=spread_u_a, n_both_dir=len(both_dir), dir_diff=dir_diff,
             per_pair=per_pair, n_distinct_scores=int(df["score_block"].nunique()))

    _print_hostpair_report(R, thr_g)
    return R


def _print_hostpair_report(R, thr_g):
    cu, cd, cb, act = R["c_und"], R["c_dir"], R["c_blk"], R["act"]
    print("=" * 96)
    print("11.4 HOST-PAIR DIAGNOSTIC - UNSW validation only (post-hoc; decides nothing)")
    print("=" * 96)
    print("[1] VALIDATION HOST-PAIR STRUCTURE")
    print(f"  validation flows              : {R['n']:,} | attack rate {R['base']:.6f}")
    print(f"  unique unordered / directed pairs: {cu['groups']:,} / {cd['groups']:,}")
    print(f"  flows per unordered pair      : median {int(R['sizes'].median()):,} | "
          f"max {int(R['sizes'].max()):,}")
    print(f"  mixed-label unordered pairs   : {cu['mixed_groups']:,} of {cu['groups']:,}")
    print(f"  flows in mixed-label pairs    : {cu['flows_in_mixed']:,} "
          f"({100 * cu['flows_in_mixed'] / R['n']:.3f}% of validation)")

    print("\n[2] ORACLE CEILINGS  <<< DIAGNOSTIC ONLY - not models, not results >>>")
    print(f"  {'grouping':<34}{'groups':>7}{'PR-AUC':>10}{'ROC-AUC':>10}{'max F1':>10}"
          f"{'max acc':>10}{'irreducible err':>17}")
    for lbl, c in [("directed pair (order-aware)", cd), ("unordered pair (GCN-only bound)", cu),
                   ("GCN-only's own score blocks", cb)]:
        print(f"  {lbl:<34}{c['groups']:>7,}{c['pr_auc']:>10.6f}{c['roc_auc']:>10.6f}"
              f"{c['max_f1']:>10.6f}{c['max_accuracy']:>10.6f}{c['irreducible_errors']:>17,}")

    print("\n[3] ACTUAL FROZEN GCN-only (validation, Stage-2 probabilities)")
    print(f"  threshold (frozen) {thr_g:.8f} | PR-AUC {R['act_pr']:.6f} | ROC-AUC "
          f"{R['act_roc']:.6f} | F1 {act['f1']:.6f} | acc {act['accuracy']:.6f}")
    print(f"  distinct validation scores {R['n_distinct_scores']:,} (vs {cu['groups']:,} "
          f"unordered pairs)")

    print("\n[4] ACTUAL AS A FRACTION OF THE UNORDERED-PAIR ORACLE CEILING")
    base = R["base"]
    print(f"  F1     : {act['f1']:.6f} / {cu['max_f1']:.6f} = {act['f1'] / cu['max_f1']:.4f}")
    print(f"  PR-AUC : {R['act_pr']:.6f} / {cu['pr_auc']:.6f} = {R['act_pr'] / cu['pr_auc']:.4f}")
    print(f"  PR-AUC above random (AP - base)/(ceiling - base): "
          f"{(R['act_pr'] - base) / (cu['pr_auc'] - base):.4f}  [base = {base:.6f}]")
    print(f"  cost of order-invariance (directed - unordered ceiling): "
          f"F1 {cd['max_f1'] - cu['max_f1']:+.6f} | PR-AUC {cd['pr_auc'] - cu['pr_auc']:+.6f}")

    sg = R["spread_u_g"]
    print("\n[5] WITHIN-PAIR PREDICTION SPREAD")
    print(f"  GCN-only, unordered pairs : non-constant {int((sg > 0).sum())} of {len(sg)} | "
          f"max spread {sg.max():.2e} | median {sg.median():.2e}")
    print(f"  GCN-only, directed pairs  : non-constant {int((R['spread_d_g'] > 0).sum())} of "
          f"{len(R['spread_d_g'])} | max spread {R['spread_d_g'].max():.2e}")
    print(f"  GCN-only, A->B vs B->A    : {R['n_both_dir']} pairs seen in both directions | "
          f"max score difference {R['dir_diff']:.2e}")
    print(f"  GNN+Attention (contrast)  : max within-pair spread {R['spread_u_a'].max():.4f} | "
          f"median {R['spread_u_a'].median():.4f}")
    print(f"  -> GCN-only predictions constant within pairs up to numerical noise: "
          f"{bool(sg.max() < 1e-5)}")

    print("\n[6] INTERPRETATION")
    print(f"  Any predictor whose score depends only on the unordered host pair (GCN-only, by "
          f"construction) cannot exceed validation F1 {cu['max_f1']:.4f} or PR-AUC "
          f"{cu['pr_auc']:.4f}, and must misclassify at least {cu['irreducible_errors']:,} "
          f"validation flows ({100 * cu['flows_in_mixed'] / R['n']:.2f}% of flows sit in pairs "
          f"with both benign and attack traffic). GCN-only reaches "
          f"{100 * act['f1'] / cu['max_f1']:.1f}% of that F1 ceiling.")


def hostpair_summary_frame(R):
    """Summary table written to ``s11_hostpair_diagnostic_summary.csv``."""
    cu, cd, cb, act = R["c_und"], R["c_dir"], R["c_blk"], R["act"]
    return pd.DataFrame([
        {"source": "ORACLE (diagnostic)", "grouping": g, "groups": c["groups"],
         "pr_auc": c["pr_auc"], "roc_auc": c["roc_auc"], "f1": c["max_f1"],
         "accuracy": c["max_accuracy"], "irreducible_errors": c["irreducible_errors"],
         "mixed_groups": c["mixed_groups"], "flows_in_mixed": c["flows_in_mixed"]}
        for g, c in [("directed_pair", cd), ("unordered_pair", cu), ("gcn_only_score_block", cb)]
    ] + [{"source": "ACTUAL frozen GCN-only (validation)", "grouping": "model",
          "groups": R["n_distinct_scores"], "pr_auc": R["act_pr"], "roc_auc": R["act_roc"],
          "f1": act["f1"], "accuracy": act["accuracy"], "irreducible_errors": np.nan,
          "mixed_groups": np.nan, "flows_in_mixed": np.nan}])


def reproducibility_diagnostic(exp, vocab, gcn_stage2, attn_stage2, ckpt_gcn_only, ckpt_attn):
    """11.7-D: GCN-only Stage-2 PR-AUC reproducibility (validation only, read-only)."""
    assert len(exp.heldout_accessed) == 0
    device = exp.device
    graph = exp.graph
    val = exp.val
    num_nodes = vocab.num_nodes

    yv = val.y.numpy().astype(int)
    ck_g = torch.load(ckpt_gcn_only, map_location="cpu", weights_only=False)
    ck_a = torch.load(ckpt_attn, map_location="cpu", weights_only=False)
    THR_G, THR_A = ck_g["threshold_selected"], ck_a["threshold_selected"]
    P2_G = gcn_stage2["val_prob"]          # Stage-2 scoring (produced the threshold)
    P2_A = attn_stage2["val_prob"]

    def _load(build_fn, ck):
        b = build_fn()
        for k, m in b["modules"].items():
            m.load_state_dict(ck[k], strict=True)
        return b

    def _gpu_probs(b):
        x_dev, ei_dev = graph.x.to(device), graph.edge_index.to(device)
        p = torch.sigmoid(exp.score_validation(b, x_dev, ei_dev)).numpy()
        del x_dev, ei_dev
        return p

    @torch.no_grad()
    def _cpu_probs_gcn_only(ck):
        g = GCNEncoder(1, GCN_HIDDEN_DIM, GCN_OUT_DIM); c = FlowClassifier(GCN_OUT_DIM, CLS_HIDDEN_DIM, 0.0)
        g.load_state_dict(ck["gcn"], strict=True); c.load_state_dict(ck["classifier"], strict=True)
        g.eval(); c.eval()
        emb = g(graph.x, graph.edge_index)
        return torch.sigmoid(c(0.5 * (emb[val.src] + emb[val.dst]))).numpy()

    print("=" * 90)
    print("11.7-D  GCN-only Stage-2 reproducibility diagnostic (validation only)")
    print("=" * 90)

    # (A) Checkpoint / restoration: are restored weights bit-identical?
    b_g = _load(exp.build_gcn_only, ck_g)
    exact = all(torch.equal(v.detach().cpu(), ck_g[k][n])
                for k, m in b_g["modules"].items() for n, v in m.state_dict().items())
    print(f"(A) restored weights bit-identical to checkpoint: {exact}")
    print(f"    saved (Stage 1) val PR-AUC : {ck_g['val_pr_auc']:.6f} | "
          f"Stage 2 val PR-AUC : {gcn_stage2['val_ranking']['pr_auc']:.6f}")

    # (B) Computation path: recompute AP from the stored Stage-2 probabilities
    ap_p2 = average_precision_score(yv, P2_G)
    print(f"(B) AP recomputed from stored Stage-2 probs: {ap_p2:.6f} "
          f"(equals Stage-2 value: {abs(ap_p2 - gcn_stage2['val_ranking']['pr_auc']) < 1e-12})")

    # (C) Repeated GPU scorings of the SAME restored weights
    R = 5
    runs = [_gpu_probs(b_g) for _ in range(R)]
    aps = [average_precision_score(yv, p) for p in runs]
    dmax = [float(np.abs(p - P2_G).max()) for p in runs]
    print(f"(C) {R} rescorings of identical weights:")
    for i, (ap, d) in enumerate(zip(aps, dmax)):
        print(f"    run {i}: AP {ap:.6f} | max |p - p_stage2| {d:.2e}")
    print(f"    AP spread {max(aps) - min(aps):.2e} | Stage1-vs-Stage2 gap "
          f"{abs(ck_g['val_pr_auc'] - ap_p2):.2e}")

    # (D) CPU scoring (deterministic scatter): repeated runs
    c1, c2 = _cpu_probs_gcn_only(ck_g), _cpu_probs_gcn_only(ck_g)
    print(f"(D) CPU rescoring x2 bit-identical: {np.array_equal(c1, c2)} | "
          f"AP {average_precision_score(yv, c1):.6f} | max |p_cpu - p_stage2| "
          f"{np.abs(c1 - P2_G).max():.2e}")

    # (E) Tie structure of GCN-only validation scores
    s, d = val.src.numpy(), val.dst.numpy()
    pair = np.minimum(s, d) * num_nodes + np.maximum(s, d)
    df = pd.DataFrame({"pair": pair, "p": P2_G})
    per_pair_spread = df.groupby("pair")["p"].agg(lambda x: x.max() - x.min())
    u, cnt = np.unique(P2_G, return_counts=True)
    gaps = np.diff(u)
    print(f"(E) unordered host pairs in validation : {df['pair'].nunique()}")
    print(f"    distinct probability values        : {u.size}")
    print(f"    pairs non-constant within one run  : {int((per_pair_spread > 0).sum())} "
          f"(max within-pair spread {per_pair_spread.max():.2e})")
    print(f"    tie-block size: median {int(np.median(cnt)):,} | max {cnt.max():,}")
    if gaps.size:
        print(f"    smallest gap between adjacent distinct scores: {gaps.min():.2e} "
              f"| gaps below GPU noise ({max(dmax):.1e}): {int((gaps < max(dmax)).sum())}")
    pair_p2 = df.groupby("pair")["p"].first()
    for i, p in enumerate(runs[:2]):
        pp = pd.Series(p).groupby(pair).first().reindex(pair_p2.index)
        a, b = pair_p2.to_numpy(), pp.to_numpy()
        flips = int((np.sign(a[:, None] - a[None, :]) != np.sign(b[:, None] - b[None, :])).sum() // 2)
        print(f"    run {i}: pair-order changes (incl. ties made/broken) vs Stage 2: {flips}")

    # (F) Frozen-threshold fragility (threshold unchanged; measurement only)
    blk = P2_G == THR_G
    print(f"(F) frozen threshold {THR_G:.8f} equals a Stage-2 score exactly: {bool(blk.any())}")
    print(f"    flows in that tie block: {int(blk.sum()):,} "
          f"({int(yv[blk].sum()):,} attacks) | pairs: {sorted(set(pair[blk].tolist()))}")
    below = u[u < THR_G]
    print(f"    gap to next lower distinct score: "
          f"{THR_G - below.max() if below.size else float('nan'):.2e}")
    for i, p in enumerate(runs):
        m = threshold_metrics(yv, p, THR_G)
        flip = int(((p >= THR_G) != (P2_G >= THR_G)).sum())
        print(f"    run {i}: block flagged {bool((p[blk] >= THR_G).all())} "
              f"| predictions changed vs Stage 2 {flip:,} | val F1 {m['f1']:.6f} "
              f"| flagged {m['flagged_pct']:.4f}%")

    # (G) Control: same diagnostic on the frozen attention model
    b_a = _load(exp.build_attn, ck_a)
    runs_a = [_gpu_probs(b_a) for _ in range(3)]
    aps_a = [average_precision_score(yv, p) for p in runs_a]
    print(f"(G) attention control: distinct val scores {np.unique(P2_A).size:,} of {len(P2_A):,} | "
          f"AP spread over 3 runs {max(aps_a) - min(aps_a):.2e} | "
          f"max predictions changed at its threshold "
          f"{max(int(((p >= THR_A) != (P2_A >= THR_A)).sum()) for p in runs_a)}")

    del b_g, b_a, runs, runs_a, c1, c2, ck_g, ck_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert len(exp.heldout_accessed) == 0
    print("Read-only diagnostic | held-out accessed: none | threshold unchanged")
