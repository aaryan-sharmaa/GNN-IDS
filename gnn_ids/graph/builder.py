"""Communication-graph construction (notebook sections 4 and 10.4).

Nodes are IP addresses from the TRAINING flows only, plus one shared placeholder node for
IPs never seen in training. Message passing uses the forward and reverse copies of every
training flow (2N edges, duplicates preserved, no weights); the N original flows are kept
as separate flow-level tensors (``flow_*``) used for classification.
"""
import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ..config import N_FEATURES, Locked, locked_assert


@dataclass
class GraphVocab:
    """Train-only node vocabulary. ``lookup`` is a dict (UNSW) or a pandas Index (CSE)."""
    node_ips: np.ndarray
    lookup: Any
    kind: str                    # "dict" | "index"

    @property
    def num_train_nodes(self) -> int:
        return len(self.node_ips)

    @property
    def placeholder_id(self) -> int:
        return self.num_train_nodes

    @property
    def num_nodes(self) -> int:
        return self.num_train_nodes + 1

    def map_train(self, ip_series: pd.Series) -> np.ndarray:
        """Strict mapping for training endpoints: every IP must resolve."""
        if self.kind == "dict":
            mapped = ip_series.map(self.lookup)
            assert not np.isnan(mapped.to_numpy(dtype=np.float64)).any(), \
                "Some training IP failed to map"
            arr = mapped.to_numpy(dtype=np.int64)
        else:
            arr = self.lookup.get_indexer(ip_series).astype(np.int64)
            assert arr.min() >= 0, "Unmapped training IP"
        assert arr.max() < self.num_train_nodes
        return arr

    def map_eval(self, ip_series: pd.Series) -> np.ndarray:
        """Validation/test endpoints: unseen IPs -> the single placeholder node."""
        if self.kind == "dict":
            return (ip_series.map(self.lookup)
                             .fillna(self.placeholder_id)
                             .astype(np.int64)
                             .to_numpy())
        idx = self.lookup.get_indexer(ip_series)
        return np.where(idx < 0, self.placeholder_id, idx).astype(np.int64)


def build_unsw_vocab(train_src_ip: pd.Series, train_dst_ip: pd.Series) -> GraphVocab:
    """Deterministic vocabulary: sorted unique IPs from the SAMPLED TRAINING endpoints."""
    train_ips = pd.concat([train_src_ip, train_dst_ip], ignore_index=True)
    node_ips = np.sort(train_ips.unique())
    ip_to_node = {ip: i for i, ip in enumerate(node_ips)}
    vocab = GraphVocab(node_ips=node_ips, lookup=ip_to_node, kind="dict")
    locked_assert(vocab.num_train_nodes == Locked.UNSW_TRAIN_IP_NODES,
                  f"Expected {Locked.UNSW_TRAIN_IP_NODES} training IP nodes, got {vocab.num_train_nodes}")
    print(f"Training IP nodes: {vocab.num_train_nodes} | placeholder id {vocab.placeholder_id} "
          f"| total nodes {vocab.num_nodes}")
    return vocab


def build_cse_vocab(train_src_ip: pd.Series, train_dst_ip: pd.Series) -> GraphVocab:
    """Same vocabulary rule; a pandas Index gives a vectorised lookup without a Python dict."""
    train_ips = pd.concat([train_src_ip, train_dst_ip], ignore_index=True)
    node_ips = np.sort(train_ips.unique())
    del train_ips
    gc.collect()
    vocab = GraphVocab(node_ips=node_ips, lookup=pd.Index(node_ips), kind="index")
    print(f"Training IP nodes: {vocab.num_train_nodes:,} | placeholder id {vocab.placeholder_id:,} "
          f"| total nodes {vocab.num_nodes:,}")
    return vocab


def graph_diagnostics(src_flow_np, dst_flow_np, vocab: GraphVocab, y: pd.Series) -> int:
    """Self-loops, host-pair multiplicity and degree ranges. Returns the self-loop count."""
    n_flows = len(src_flow_np)
    n_real = vocab.num_train_nodes

    self_loop_mask = src_flow_np == dst_flow_np
    n_self_loop = int(self_loop_mask.sum())
    print(f"Self-loop flows: {n_self_loop:,} ({100 * n_self_loop / n_flows:.4f}% of flows) "
          "- classified, but dropped by GCNConv's gcn_norm from message passing")

    key = src_flow_np * np.int64(vocab.num_nodes) + dst_flow_np
    _, pair_counts = np.unique(key, return_counts=True)
    print(f"Unique (src, dst) host pairs: {len(pair_counts):,} | flows per pair: "
          f"min {pair_counts.min():,} | median {int(np.median(pair_counts)):,} "
          f"| max {pair_counts.max():,}")

    out_deg = np.bincount(src_flow_np, minlength=vocab.num_nodes)
    in_deg = np.bincount(dst_flow_np, minlength=vocab.num_nodes)
    print(f"Out-degree: {out_deg[:n_real].min():,}..{out_deg[:n_real].max():,} | "
          f"in-degree: {in_deg[:n_real].min():,}..{in_deg[:n_real].max():,} | "
          f"placeholder degrees: out={out_deg[vocab.placeholder_id]}, in={in_deg[vocab.placeholder_id]}")
    assert out_deg[vocab.placeholder_id] == 0 and in_deg[vocab.placeholder_id] == 0
    return n_self_loop


def assemble_training_graph(vocab: GraphVocab, X_flow: np.ndarray, y_flow: np.ndarray,
                            row_id: np.ndarray, src_flow_np: np.ndarray,
                            dst_flow_np: np.ndarray) -> Data:
    """Build the PyG ``Data`` object for the training communication graph.

    ``edge_index`` holds the message-passing edges (forward then reverse block, 2N columns,
    unlabeled). ``flow_*`` hold the N original classification flows. The ``flow_`` prefix
    avoids PyG's convention that ``edge_*`` keys have one row per edge_index column.
    """
    n_flows = len(src_flow_np)
    num_nodes = vocab.num_nodes

    # Node features: all-ones, width 1 (locked)
    x_nodes = torch.ones((num_nodes, 1), dtype=torch.float32)

    # Message-passing edges: forward + reverse, duplicates preserved, no weights
    forward = np.stack([src_flow_np, dst_flow_np], axis=0)
    reverse = np.stack([dst_flow_np, src_flow_np], axis=0)
    edge_index_mp = torch.from_numpy(np.concatenate([forward, reverse], axis=1))
    del forward, reverse
    gc.collect()

    # Flow-level (classification) tensors: N rows each, original order preserved
    edge_attr_flow = torch.from_numpy(X_flow)
    edge_label_flow = torch.from_numpy(y_flow)
    src_flow = torch.from_numpy(src_flow_np)
    dst_flow = torch.from_numpy(dst_flow_np)
    row_id_flow = torch.from_numpy(row_id)

    assert edge_index_mp.shape == (2, 2 * n_flows), f"MP edges: {edge_index_mp.shape}"
    assert edge_index_mp.dtype == torch.long
    assert int(edge_index_mp.max()) < num_nodes and int(edge_index_mp.min()) >= 0
    assert edge_attr_flow.shape == (n_flows, N_FEATURES), f"Edge attr: {edge_attr_flow.shape}"
    assert edge_attr_flow.dtype == torch.float32
    assert edge_label_flow.shape[0] == n_flows
    assert src_flow.shape[0] == dst_flow.shape[0] == row_id_flow.shape[0] == n_flows
    assert torch.isfinite(edge_attr_flow).all(), "Non-finite value in edge features"
    assert set(edge_label_flow.unique().tolist()) <= {0, 1}
    # Reverse copies must be an exact mirror of the forward block
    assert torch.equal(edge_index_mp[0, n_flows:], edge_index_mp[1, :n_flows])
    assert torch.equal(edge_index_mp[1, n_flows:], edge_index_mp[0, :n_flows])

    graph = Data(
        x=x_nodes,
        edge_index=edge_index_mp,
        num_nodes=num_nodes,
        flow_src=src_flow,
        flow_dst=dst_flow,
        flow_attr=edge_attr_flow,
        flow_label=edge_label_flow,
        flow_row_id=row_id_flow,
    )
    graph.validate(raise_on_error=True)

    print(f"Training graph: {num_nodes:,} nodes ({vocab.num_train_nodes:,} IPs + 1 placeholder) | "
          f"{n_flows:,} flows | {2 * n_flows:,} message-passing edges | "
          f"positives {int(edge_label_flow.sum()):,}")
    return graph


def build_unsw_graph(data, vocab: GraphVocab):
    """UNSW training graph + validation/test endpoint node ids.

    Returns (graph_train, src_flow_np, dst_flow_np, mapped) where ``mapped`` holds the
    val/test source/destination node arrays.
    """
    # Row alignment check BEFORE mapping: every flow-level object must share one index.
    for name, obj in [("y_train", data.y_train), ("train_src_ip", data.train_src_ip),
                      ("train_dst_ip", data.train_dst_ip), ("train_row_id", data.train_row_id)]:
        assert data.X_train.index.equals(obj.index), f"Index misalignment: {name}"

    src_flow_np = vocab.map_train(data.train_src_ip)
    dst_flow_np = vocab.map_train(data.train_dst_ip)
    n_flows = len(data.X_train)
    assert len(src_flow_np) == len(dst_flow_np) == n_flows
    locked_assert(n_flows == Locked.UNSW_TRAIN_FLOWS, f"Expected {Locked.UNSW_TRAIN_FLOWS:,} training flows")

    graph_diagnostics(src_flow_np, dst_flow_np, vocab, data.y_train)

    graph = assemble_training_graph(
        vocab,
        X_flow=data.X_train.to_numpy(dtype=np.float32),
        y_flow=data.y_train.to_numpy(dtype=np.int64),
        row_id=data.train_row_id.to_numpy(dtype=np.int64),
        src_flow_np=src_flow_np, dst_flow_np=dst_flow_np,
    )
    mapped = map_eval_endpoints(vocab, data.val_src_ip, data.val_dst_ip,
                                data.test_src_ip, data.test_dst_ip)
    return graph, mapped


def build_cse_graph(data, vocab: GraphVocab):
    """CSE training graph + validation/test endpoint node ids (vectorised, memory-safe)."""
    src_flow_np = vocab.map_train(data.train_src_ip)
    dst_flow_np = vocab.map_train(data.train_dst_ip)
    assert len(src_flow_np) == len(dst_flow_np) == len(data.X_train)

    graph_diagnostics(src_flow_np, dst_flow_np, vocab, data.y_train)

    graph = assemble_training_graph(
        vocab,
        X_flow=data.X_train,                       # shares memory with the numpy array
        y_flow=data.y_train.to_numpy(dtype=np.int64),
        row_id=data.train_row_id.to_numpy(dtype=np.int64),
        src_flow_np=src_flow_np, dst_flow_np=dst_flow_np,
    )
    mapped = map_eval_endpoints(vocab, data.val_src_ip, data.val_dst_ip,
                                data.test_src_ip, data.test_dst_ip)
    return graph, mapped


def map_eval_endpoints(vocab: GraphVocab, val_src_ip, val_dst_ip, test_src_ip, test_dst_ip):
    """Map val/test endpoints onto the training vocabulary. Creates no edges."""
    out = {
        "val_src": vocab.map_eval(val_src_ip), "val_dst": vocab.map_eval(val_dst_ip),
        "test_src": vocab.map_eval(test_src_ip), "test_dst": vocab.map_eval(test_dst_ip),
    }
    for name in ("val", "test"):
        s, d = out[f"{name}_src"], out[f"{name}_dst"]
        unseen = (s == vocab.placeholder_id) | (d == vocab.placeholder_id)
        assert s.min() >= 0 and s.max() < vocab.num_nodes and d.max() < vocab.num_nodes
        print(f"{name}: {len(s):,} flows | flows using placeholder: "
              f"{int(unseen.sum()):,} ({100 * unseen.mean():.5f}%)")
    return out
