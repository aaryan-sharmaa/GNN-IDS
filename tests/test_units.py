"""Fast unit tests: architecture invariants, graph construction, threshold rules."""
import numpy as np
import pandas as pd
import torch

from gnn_ids.config import N_PARAMS_ATTENTION, N_PARAMS_TOTAL, Locked
from gnn_ids.evaluation.thresholds import (select_threshold_and_evaluate,
                                           select_threshold_full_curve,
                                           select_threshold_trimmed)
from gnn_ids.graph.builder import assemble_training_graph, build_unsw_vocab
from gnn_ids.models import (FlowClassifier, GCNEncoder, PostGCNAttention, build_gnn_modules)

Locked.mode = "off"   # synthetic data cannot match the published reference values


def test_parameter_counts():
    assert sum(p.numel() for p in GCNEncoder(1, 64, 64).parameters()) == 4_288
    assert sum(p.numel() for p in PostGCNAttention(64, 49, 64).parameters()) == N_PARAMS_ATTENTION
    assert sum(p.numel() for p in FlowClassifier(64, 32, 0.0).parameters()) == 2_113
    modules = build_gnn_modules(1, torch.device("cpu"))
    assert sum(p.numel() for p in modules.parameters()) == N_PARAMS_TOTAL == 10_180


def test_attention_weights_sum_to_one():
    att = PostGCNAttention(64, 49, 64)
    z, alpha = att(torch.randn(5, 64), torch.randn(5, 64), torch.rand(5, 49))
    assert z.shape == (5, 64)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_staged_seeding_is_deterministic():
    a = build_gnn_modules(1, torch.device("cpu"), staged_seeding=True)
    b = build_gnn_modules(1, torch.device("cpu"), staged_seeding=True)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def _tiny_graph():
    src = pd.Series(["a", "a", "b", "c", "c"])
    dst = pd.Series(["b", "c", "c", "a", "b"])
    vocab = build_unsw_vocab(src, dst)
    s, d = vocab.map_train(src), vocab.map_train(dst)
    X = np.random.default_rng(0).random((5, 49)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    g = assemble_training_graph(vocab, X, y, np.arange(5, dtype=np.int64), s, d)
    return vocab, g


def test_graph_has_forward_and_reverse_edges_and_placeholder():
    vocab, g = _tiny_graph()
    n = 5
    assert vocab.num_train_nodes == 3 and vocab.num_nodes == 4
    assert g.edge_index.shape == (2, 2 * n)
    assert torch.equal(g.edge_index[0, n:], g.edge_index[1, :n])      # reverse block mirrors
    assert g.flow_attr.shape == (n, 49) and g.flow_label.shape == (n,)
    mapped = vocab.map_eval(pd.Series(["a", "zzz"]))
    assert mapped.tolist() == [0, vocab.placeholder_id]               # unseen IP -> placeholder


def test_threshold_rules_pick_valid_thresholds():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(int)
    p = np.clip(0.5 * y + rng.random(500) * 0.6, 0, 1)
    full = select_threshold_full_curve(y, p)
    assert 0.0 <= full["threshold"] <= 1.0 and 0.0 < full["f1"] <= 1.0
    assert 0.0 <= select_threshold_trimmed(y, p) <= 1.0


def test_select_threshold_and_evaluate_freezes_validation_threshold():
    rng = np.random.default_rng(2)
    yv, yt = (rng.random(400) < 0.3).astype(int), (rng.random(400) < 0.3).astype(int)
    pv, pt = np.clip(0.4 * yv + rng.random(400) * 0.6, 0, 1), np.clip(0.4 * yt + rng.random(400) * 0.6, 0, 1)
    row = select_threshold_and_evaluate("m", pv, yv, pt, yt)
    assert row["val_threshold"] == select_threshold_trimmed(yv, pv)
    assert row["test_tn"] + row["test_fp"] + row["test_fn"] + row["test_tp"] == 400
    assert row["test_pr_auc"] > 0
