"""CSE-CIC-IDS2018 pipeline (notebook section 10): sample -> graph -> GNN (PR-AUC protocol)
-> frozen-threshold test evaluation -> classical baselines."""
from dataclasses import dataclass

import pandas as pd
import torch

from ..baselines.classical import run_cse_baselines
from ..config import RunConfig, RunPaths
from ..data.cse import CseData, prepare_cse
from ..data.tensors import SplitTensors, cse_split_tensors
from ..graph.builder import GraphVocab, build_cse_graph, build_cse_vocab
from ..models.classifier import build_gnn_modules
from ..training.prauc_protocol import (evaluate_frozen_on_test, freeze_threshold,
                                       make_loss_and_optimizer, train_prauc_stage1)
from ..utils import get_device


@dataclass
class CseResult:
    data: CseData
    vocab: GraphVocab
    graph: object
    val: SplitTensors
    test: SplitTensors
    history: list
    ck_r: dict
    test_results_r: dict
    val_probs_r: torch.Tensor
    test_probs_r: torch.Tensor
    baseline_results: pd.DataFrame
    y_test_np: object


def run_cse(paths: RunPaths, cfg: RunConfig, unsw_feature_cols, device=None) -> CseResult:
    """Run notebook section 10 end to end (10.1-10.8; the final comparison is in reporting)."""
    device = device or get_device()
    paths.make_dirs()

    # 10.1-10.3 sample, split, scale
    data = prepare_cse(paths.cse_csv, paths.cse_sample, unsw_feature_cols,
                       reuse_sample=cfg.reuse_cse_sample)

    # 10.4 communication graph
    vocab = build_cse_vocab(data.train_src_ip, data.train_dst_ip)
    graph, mapped = build_cse_graph(data, vocab)
    val = cse_split_tensors(data.X_val, data.y_val, mapped["val_src"], mapped["val_dst"],
                            vocab.num_nodes)
    test = cse_split_tensors(data.X_test, data.y_test, mapped["test_src"], mapped["test_dst"],
                             vocab.num_nodes)

    # 10.5-10.6-R fresh modules, train on validation PR-AUC
    modules = build_gnn_modules(graph.x.shape[1], device, staged_seeding=False)
    criterion, optimizer, y_train_t, pos_weight_value = make_loss_and_optimizer(
        modules, graph, device)
    history, best_epoch, best_pr = train_prauc_stage1(
        modules, graph, criterion, optimizer, y_train_t, val, device, paths.cse_checkpoint,
        num_epochs=cfg.cse_num_epochs, pos_weight_value=pos_weight_value)

    # Stage 2: validation-only threshold, frozen into the checkpoint
    _, val_probs_r = freeze_threshold(modules, graph, val, device, paths.cse_checkpoint,
                                      criterion=criterion)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 10.7 Stage 3: the single test evaluation at the frozen threshold
    ck_r, test_results_r, test_probs_r = evaluate_frozen_on_test(
        modules, graph, test, device, paths.cse_checkpoint)

    # 10.8 classical baselines (same flows, same split)
    baseline_results, y_test_np = run_cse_baselines(data, unsw_feature_cols)

    return CseResult(data=data, vocab=vocab, graph=graph, val=val, test=test, history=history,
                     ck_r=ck_r, test_results_r=test_results_r, val_probs_r=val_probs_r,
                     test_probs_r=test_probs_r, baseline_results=baseline_results,
                     y_test_np=y_test_np)
