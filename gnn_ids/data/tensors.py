"""CPU tensors for a validation / test split (features stay on CPU, moved per mini-batch)."""
from dataclasses import dataclass

import numpy as np
import torch

from ..config import N_FEATURES


@dataclass
class SplitTensors:
    X: torch.Tensor      # [n, 49] float32
    y: torch.Tensor      # [n] float32
    src: torch.Tensor    # [n] int64 node ids
    dst: torch.Tensor    # [n] int64 node ids


def unsw_split_tensors(X_df, y_series, src_node, dst_node, num_nodes) -> SplitTensors:
    t = SplitTensors(
        X=torch.from_numpy(X_df.to_numpy(dtype=np.float32)),
        y=torch.from_numpy(y_series.to_numpy(dtype=np.float32)),
        src=torch.from_numpy(src_node.astype(np.int64)),
        dst=torch.from_numpy(dst_node.astype(np.int64)),
    )
    _check(t, num_nodes)
    return t


def cse_split_tensors(X_np, y_series, src_node, dst_node, num_nodes) -> SplitTensors:
    t = SplitTensors(
        X=torch.from_numpy(X_np).float(),
        y=torch.from_numpy(y_series.to_numpy(dtype=np.float32)),
        src=torch.as_tensor(src_node, dtype=torch.long),
        dst=torch.as_tensor(dst_node, dtype=torch.long),
    )
    _check(t, num_nodes)
    return t


def _check(t: SplitTensors, num_nodes: int):
    assert t.X.shape[1] == N_FEATURES
    assert t.src.shape[0] == t.dst.shape[0] == t.X.shape[0] == t.y.shape[0]
    assert int(t.src.max()) < num_nodes and int(t.dst.max()) < num_nodes
    assert torch.isfinite(t.X).all(), "Non-finite value in features"
