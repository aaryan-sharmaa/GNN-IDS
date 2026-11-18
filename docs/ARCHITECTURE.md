# Architecture

## Module map and dependency direction

```
main.py ──► pipelines ──► data ─────────┐
                │         graph ────────┤
                │         models ───────┤
                │         training ─────┼──► config, utils
                │         evaluation ───┤
                └───────► baselines ────┘
```

Lower layers never import from `pipelines`. `config.py` has no project imports.

| Package | Responsibility | Notebook sections |
|---|---|---|
| `data/unsw.py` | load, clean, stratified split, scale (train-only fit), 30% subsample | 2–3 |
| `data/cse.py` | chunked scan, 10% stratified sample, split, in-place scaling | 10.1–10.3 |
| `data/tensors.py` | CPU tensors for the validation/test splits | 7.2.1, 8.1, 10.6 |
| `graph/builder.py` | train-only IP vocabulary, flow→node mapping, PyG `Data` | 4, 10.4 |
| `models/` | `GCNEncoder`, `PostGCNAttention`, `FlowClassifier`, `GnnModules` | 5–7.1 |
| `training/f1_protocol.py` | UNSW loop, selection on validation F1 @ 0.5 | 7 |
| `training/prauc_protocol.py` | CSE loop, selection on PR-AUC, threshold freeze, frozen test eval | 10.6-R, 10.7 |
| `training/controlled.py` | §11 bundle helpers, Stage 1/2/3 | 11 |
| `evaluation/` | metrics, the two threshold rules, read-only diagnostics | 7–11 |
| `baselines/classical.py` | LR / RF / XGBoost for UNSW (fixed 0.5) and CSE (frozen threshold) | 9, 10.8 |
| `pipelines/` | one function per stage: `run_unsw`, `run_cse`, `run_controlled`, reporting | all |

## How state moves (replacing the notebook's globals)

* `run_unsw` returns an `UnswResult`: `inputs` (prepared data, vocabulary, graph, val/test
  tensors), trained modules, epoch `history`, `val_probs_best`, `test_probs`, baseline results.
* `run_cse` returns a `CseResult` (graph, frozen checkpoint `ck_r`, test results, baselines).
* `build_unsw_threshold_table` / `build_cse_final_comparison` / `build_result_tables` consume
  those results and write the CSVs.
* `run_controlled` receives the UNSW inputs, the historical `history` / `val_probs_best` and
  the UNSW threshold table. For `--stage controlled` on its own, the history and validation
  probabilities are reloaded from `outputs/final_experiment_backup/` (written by `--stage unsw`).

## Running the controlled stage independently (`--stage controlled`)

Section 11 reuses the historical UNSW run, so `--stage controlled` has prerequisites. `main.py`
checks them before any expensive work and exits with an error naming the missing file.

| Prerequisite | Produced by | Why section 11 needs it |
|---|---|---|
| `NF-UNSW-NB15-v3.csv` (path via `--unsw-csv`) | downloaded | the data and graph are rebuilt deterministically from the CSV |
| `outputs/gnn_ids_unsw_best.pt` | `--stage unsw` | historical checkpoint; its presence is asserted and it is a protected path |
| `outputs/final_experiment_backup/unsw_history.json` | `--stage unsw` | historical epoch log for the retrace check and the locked-value checks |
| `outputs/final_experiment_backup/unsw_val_probs_best.npy` | `--stage unsw` | epoch-4 validation probabilities for the threshold-rule pre-check |
| `outputs/final_experiment_backup/unsw_final_threshold_results.csv` | `--stage unsw` | historical GNN threshold and test metrics (pre-check and context table) |

Also required:

* The same `--output-dir` as the `--stage unsw` run.
* **No existing section-11 checkpoints** (`gnn_ids_unsw_attn_best_prauc.pt`,
  `gnn_ids_unsw_gcn_only_best.pt`). Section 11 refuses to overwrite them; pass `--fresh` to delete
  them deliberately.
* The CSE stage is **not** required, although in `--stage all` it runs first and the CSE
  checkpoint path is treated as protected.

`--stage all` needs none of this: it passes the in-memory results straight through.

## Deliberate duplications (kept from the notebook to protect numerical results)

* **Three training loops** (`f1_protocol`, `prauc_protocol`, `controlled`). They share
  mechanics but differ in selection rule, evaluation helper, checkpoint payload and logging.
* **Two threshold rules** (`thresholds.py`): the CSE rule divides over the full
  precision/recall arrays with a `where=` guard; the UNSW/§11 rule slices `[:-1]` and floors the
  denominator at `1e-12`. They can pick different thresholds on tied probabilities.
* **Two vocabulary lookups**: a Python dict for UNSW (43 hosts) and a pandas `Index` for CSE
  (57,811 hosts). The tensor assembly after the lookup is shared.
* **Two seeding schemes** in `build_gnn_modules`: UNSW resets the seed before each module
  (sections 5, 6, 7.1); CSE seeds once.

## Safety guards inherited from the notebook

* Test data is touched once per model, after the threshold is frozen. `ControlledExperiment`
  records every held-out access and asserts a model is never scored twice.
* §11 refuses to overwrite checkpoints (`--fresh` deletes them deliberately).
* `Locked` reference values (flow counts, thresholds, best epochs) are checked in `warn` mode by
  default; `--strict-locked` raises, `--no-locked-asserts` skips (subsets).
