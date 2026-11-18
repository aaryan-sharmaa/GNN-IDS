"""UNSW pipeline (notebook sections 2-9): data -> graph -> GNN -> test -> baselines."""
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Data

from ..baselines.classical import baseline_comparison_table, run_unsw_baselines
from ..config import (BATCH_SIZE, DECISION_THRESHOLD, RunConfig, RunPaths)
from ..data.tensors import SplitTensors, unsw_split_tensors
from ..data.unsw import UnswData, prepare_unsw
from ..evaluation.metrics import evaluate
from ..graph.builder import GraphVocab, build_unsw_graph, build_unsw_vocab
from ..models.classifier import GnnModules, build_gnn_modules
from ..training.f1_protocol import make_loss_and_optimizer, restore_checkpoint, train_f1_protocol
from ..utils import get_device


@dataclass
class UnswInputs:
    """Output of notebook sections 2-4: prepared data, training graph, val/test tensors."""
    data: UnswData
    vocab: GraphVocab
    graph: Data
    mapped: dict
    val: SplitTensors
    test: SplitTensors


@dataclass
class UnswResult:
    inputs: UnswInputs
    modules: GnnModules
    criterion: Any
    pos_weight_value: float
    history: list
    best_epoch: int
    checkpoint: dict
    val_probs_best: torch.Tensor
    test_probs: torch.Tensor
    m_test: dict
    baseline_results: dict
    comparison_df: pd.DataFrame


def prepare_unsw_inputs(paths: RunPaths) -> UnswInputs:
    """Sections 2-4: load/clean/split/scale/subsample, build the training graph."""
    data = prepare_unsw(paths.unsw_csv)
    vocab = build_unsw_vocab(data.train_src_ip, data.train_dst_ip)
    graph, mapped = build_unsw_graph(data, vocab)

    val = unsw_split_tensors(data.X_val, data.y_val, mapped["val_src"], mapped["val_dst"],
                             vocab.num_nodes)
    test = unsw_split_tensors(data.X_test, data.y_test, mapped["test_src"], mapped["test_dst"],
                              vocab.num_nodes)
    return UnswInputs(data=data, vocab=vocab, graph=graph, mapped=mapped, val=val, test=test)


def _test_diagnostics(inputs: UnswInputs, test_probs):
    """Section 8.4: unseen endpoints and metrics split by endpoint coverage."""
    test, vocab = inputs.test, inputs.vocab
    n_test = test.X.shape[0]
    unseen_mask = ((test.src == vocab.placeholder_id) | (test.dst == vocab.placeholder_id)).numpy()
    n_unseen = int(unseen_mask.sum())
    print(f"Endpoint diagnostics: flows with >=1 unseen endpoint {n_unseen:,} "
          f"({100 * n_unseen / n_test:.5f}%) | self-loop flows "
          f"{int((test.src == test.dst).sum()):,}")

    y_true = test.y.numpy()
    y_prob = test_probs.numpy()
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    for label, mask in [("both endpoints seen", ~unseen_mask), ("unseen endpoint", unseen_mask)]:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<22}: no flows")
            continue
        auc = (roc_auc_score(y_true[mask], y_prob[mask])
               if len(np.unique(y_true[mask])) > 1 else float("nan"))
        print(f"  {label:<22}: n={n:>9,} pos={int(y_true[mask].sum()):>7,} | acc "
              f"{accuracy_score(y_true[mask], y_pred[mask]):.4f} | F1 "
              f"{f1_score(y_true[mask], y_pred[mask], zero_division=0):.4f} | AUC {auc:.4f}")


def train_unsw_gnn(paths: RunPaths, cfg: RunConfig, inputs: UnswInputs, device):
    """Sections 5-8: build the model, train (F1 protocol), restore the best checkpoint and
    score validation and test once at the fixed 0.5 threshold."""
    graph, val, test = inputs.graph, inputs.val, inputs.test

    modules = build_gnn_modules(graph.x.shape[1], device, staged_seeding=True)
    criterion, optimizer, pos_weight_value = make_loss_and_optimizer(
        modules, graph, inputs.data.y_train, device)
    print(f"pos_weight = {pos_weight_value:.4f}")

    result = train_f1_protocol(modules, graph, criterion, optimizer, val, device,
                               paths.unsw_checkpoint, num_epochs=cfg.num_epochs)

    # 7.3: the live model holds the LAST epoch's weights; restore the best-validation ones
    ckpt = restore_checkpoint(modules, paths.unsw_checkpoint, device)
    x_dev = graph.x.to(device)
    ei_dev = graph.edge_index.to(device)
    with torch.no_grad():
        modules.gcn.eval()
        emb_best = modules.gcn(x_dev, ei_dev)
    val_loss_best, m_best, val_probs_best = evaluate(
        modules, criterion, emb_best, val.src, val.dst, val.X, val.y, BATCH_SIZE,
        DECISION_THRESHOLD, device)
    best = result.history[result.best_epoch - 1]
    print(f"Restored epoch {ckpt['epoch']} - validation F1 {m_best['f1']:.4f} "
          f"(training log: {best['f1']:.4f}, diff {abs(m_best['f1'] - best['f1']):.2e})")
    del emb_best

    # 8: single, final test evaluation
    with torch.no_grad():
        emb_test = modules.gcn(x_dev, ei_dev)
    assert emb_test.shape == (graph.num_nodes, 64)
    assert torch.isfinite(emb_test).all()
    test_loss, m_test, test_probs = evaluate(
        modules, criterion, emb_test, test.src, test.dst, test.X, test.y, BATCH_SIZE,
        DECISION_THRESHOLD, device)
    tn, fp, fn, tp = m_test["cm"].ravel()
    print("=" * 62)
    print(f"FINAL TEST RESULTS - NF-UNSW-NB15-v3 (threshold {DECISION_THRESHOLD})")
    print("=" * 62)
    print(f"Model      : GCN + post-GCN attention, epoch {ckpt['epoch']} checkpoint")
    print(f"  Loss {test_loss:.4f} | Accuracy {m_test['accuracy']:.4f} | Precision "
          f"{m_test['precision']:.4f} | Recall {m_test['recall']:.4f} | F1 {m_test['f1']:.4f} "
          f"| ROC-AUC {m_test['roc_auc']:.4f}")
    print(f"  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    print("  validation vs test (epoch {}):".format(ckpt["epoch"]))
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"    {k:<10} val {m_best[k]:.4f} | test {m_test[k]:.4f} | "
              f"diff {m_test[k] - m_best[k]:+.4f}")
    _test_diagnostics(inputs, test_probs)

    del emb_test, x_dev, ei_dev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return modules, criterion, pos_weight_value, result, ckpt, val_probs_best, test_probs, m_test


def run_unsw(paths: RunPaths, cfg: RunConfig, device=None) -> UnswResult:
    """Run notebook sections 2-9 end to end."""
    device = device or get_device()
    inputs = prepare_unsw_inputs(paths)
    (modules, criterion, pos_weight_value, result, ckpt,
     val_probs_best, test_probs, m_test) = train_unsw_gnn(paths, cfg, inputs, device)

    # 9: classical baselines on the same flows
    baseline_results = run_unsw_baselines(inputs.data, inputs.graph)
    comparison_df = baseline_comparison_table(m_test, baseline_results,
                                              len(inputs.data.y_test), DECISION_THRESHOLD)

    res = UnswResult(inputs=inputs, modules=modules, criterion=criterion,
                     pos_weight_value=pos_weight_value, history=result.history,
                     best_epoch=result.best_epoch, checkpoint=ckpt,
                     val_probs_best=val_probs_best, test_probs=test_probs, m_test=m_test,
                     baseline_results=baseline_results, comparison_df=comparison_df)
    save_unsw_artifacts(res, paths)
    return res


# ---------------------------------------------------------------- artifacts for --stage controlled
def save_unsw_artifacts(res: UnswResult, paths: RunPaths):
    """Persist what section 11 needs from the historical run (the notebook kept it in memory)."""
    paths.make_dirs()
    hist = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items()}
            for h in res.history]
    (paths.backup_dir / "unsw_history.json").write_text(json.dumps(hist), encoding="utf-8")
    np.save(paths.backup_dir / "unsw_val_probs_best.npy", res.val_probs_best.numpy())


def load_unsw_artifacts(paths: RunPaths):
    """Load the historical epoch log and best-epoch validation probabilities."""
    hist_path = paths.backup_dir / "unsw_history.json"
    probs_path = paths.backup_dir / "unsw_val_probs_best.npy"
    if not (hist_path.exists() and probs_path.exists()):
        raise FileNotFoundError(
            "Section 11 needs the historical UNSW run. Run `--stage unsw` first "
            f"(missing {hist_path.name} / {probs_path.name} in {paths.backup_dir}).")
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    for h in history:
        h["cm"] = np.asarray(h["cm"])
    return history, torch.from_numpy(np.load(probs_path))
