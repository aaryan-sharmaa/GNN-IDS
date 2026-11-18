"""Single entry point for the GNN-IDS pipeline.

    python main.py                          # full pipeline: UNSW -> CSE -> tables -> section 11
    python main.py --stage unsw             # sections 2-9 + UNSW threshold table
    python main.py --stage cse              # section 10 (needs the UNSW CSV only for the feature list)
    python main.py --stage controlled       # section 11 (needs a finished `--stage unsw`)
    python main.py --stage report           # rebuild the six result CSVs from saved tables

See README.md for dataset download instructions.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from gnn_ids.config import Locked, RunConfig, RunPaths
from gnn_ids.data.unsw import unsw_feature_columns
from gnn_ids.pipelines.controlled import run_controlled
from gnn_ids.pipelines.cse import run_cse
from gnn_ids.pipelines.reporting import (build_cse_final_comparison, build_result_tables,
                                         build_unsw_threshold_table)
from gnn_ids.pipelines.unsw import (load_unsw_artifacts, prepare_unsw_inputs, run_unsw)
from gnn_ids.utils import get_device, print_environment

ROOT = Path(__file__).resolve().parent


class _Tee:
    """Mirror stdout to a log file."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GNN-based network intrusion detection pipeline")
    p.add_argument("--stage", choices=["all", "unsw", "cse", "controlled", "report"],
                   default="all")
    p.add_argument("--unsw-csv", default=str(ROOT / "data" / "NF-UNSW-NB15-v3.csv"))
    p.add_argument("--cse-csv", default=str(ROOT / "data" / "NF-CSE-CIC-IDS2018-v3.csv"))
    p.add_argument("--output-dir", default=str(ROOT / "outputs"))
    p.add_argument("--epochs", type=int, default=None,
                   help="override the 15-epoch cap (smoke tests only)")
    p.add_argument("--strict-locked", action="store_true",
                   help="abort if a published reference value differs (default: warn)")
    p.add_argument("--no-locked-asserts", action="store_true",
                   help="do not check the published reference values (data subsets)")
    p.add_argument("--smoke", action="store_true",
                   help="shortcut for --epochs 4 --no-locked-asserts")
    p.add_argument("--fresh", action="store_true",
                   help="delete existing section-11 checkpoints before running")
    p.add_argument("--no-reuse-cse-sample", action="store_true",
                   help="rebuild the 10%% CSE sample instead of reloading the parquet")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = RunConfig(fresh=args.fresh, reuse_cse_sample=not args.no_reuse_cse_sample)
    if args.smoke:
        args.epochs = args.epochs or 4
        args.no_locked_asserts = True
    if args.epochs is not None:
        cfg.num_epochs = cfg.cse_num_epochs = args.epochs
    if args.strict_locked:
        Locked.mode = "strict"
    if args.no_locked_asserts:
        Locked.mode = "off"

    paths = RunPaths(unsw_csv=args.unsw_csv, cse_csv=args.cse_csv, output_dir=args.output_dir)
    paths.make_dirs()
    sys.stdout = _Tee(paths.output_dir / "run.log")

    device = get_device()
    print_environment(device)
    print(f"Stage: {args.stage} | epochs {cfg.num_epochs}/{cfg.cse_num_epochs} | "
          f"locked checks {Locked.mode} | outputs {paths.output_dir}")

    stage = args.stage
    unsw = cse = unsw_df = cse_comparison = None

    # Fail fast, before any expensive work, if a stage's inputs are missing.
    need = {"all": ["unsw_csv", "cse_csv"], "unsw": ["unsw_csv"], "cse": ["unsw_csv"],
            "controlled": ["unsw_csv"], "report": []}[stage]
    if stage in ("all", "cse") and cfg.reuse_cse_sample and paths.cse_sample.exists():
        pass                                   # cached sample: the 4 GB CSV is not needed
    elif stage == "cse":
        need.append("cse_csv")
    for attr in need:
        if not getattr(paths, attr).exists():
            sys.exit(f"ERROR: {getattr(paths, attr)} not found. See data/README.md or pass "
                     f"--{attr.replace('_', '-')}.")
    required = {"controlled": ["unsw_history.json", "unsw_val_probs_best.npy",
                               "unsw_final_threshold_results.csv"],
                "report": ["unsw_final_threshold_results.csv", "cse_final_comparison.csv"]}
    for name in required.get(stage, []):
        if not (paths.backup_dir / name).exists():
            sys.exit(f"ERROR: {paths.backup_dir / name} not found. Run the earlier stage first "
                     f"(unsw -> cse -> controlled/report).")
    if stage == "controlled" and not paths.unsw_checkpoint.exists():
        sys.exit(f"ERROR: {paths.unsw_checkpoint} not found. Run `--stage unsw` first.")

    if stage in ("all", "unsw"):
        unsw = run_unsw(paths, cfg, device)                       # sections 2-9
        unsw_df = build_unsw_threshold_table(unsw, paths)         # 10.10 (UNSW half)

    if stage in ("all", "cse"):
        cse = run_cse(paths, cfg, unsw_feature_columns(paths.unsw_csv), device)   # section 10
        cse_comparison = build_cse_final_comparison(cse)          # 10.9
        cse_comparison.to_csv(paths.backup_dir / "cse_final_comparison.csv")

    if stage in ("all", "report", "controlled") and unsw_df is None:
        unsw_df = pd.read_csv(paths.backup_dir / "unsw_final_threshold_results.csv")

    if stage in ("all", "report"):
        if cse_comparison is None:
            cse_comparison = pd.read_csv(paths.backup_dir / "cse_final_comparison.csv",
                                         index_col="model")
        build_result_tables(unsw_df, cse_comparison, paths)       # 10.10 tables

    if stage in ("all", "controlled"):
        if unsw is not None:
            inputs, pos_weight = unsw.inputs, unsw.pos_weight_value
            history, val_probs_best = unsw.history, unsw.val_probs_best
        else:                                    # reuse the artifacts of an earlier run
            inputs = prepare_unsw_inputs(paths)
            y = inputs.graph.flow_label
            n_pos = int(y.sum())
            pos_weight = (len(y) - n_pos) / n_pos
            history, val_probs_best = load_unsw_artifacts(paths)
        run_controlled(paths, cfg, inputs, pos_weight, history, val_probs_best, unsw_df, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
