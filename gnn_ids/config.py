"""Central configuration: seeds, hyper-parameters, paths and locked reference values.

Every number below is copied from the notebook ``GNN-IDS-FINAL-SUBMIT-clean.ipynb``.
"""
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- seeds
RANDOM_SEED = 42       # split, shuffling, CSE sampling
SUBSAMPLE_SEED = 42    # UNSW training subsample
MODEL_SEED = 42        # GCN initialisation
ATT_SEED = 42          # attention initialisation
BASELINE_SEED = 42     # LR / RF / XGBoost

# ------------------------------------------------------------------- data / features
N_FEATURES = 49
IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
TIME_COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
DROP_FROM_X = IP_COLS + TIME_COLS + ["Label", "Attack", "orig_row_id"]
CSE_DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS",
            "FLOW_END_MILLISECONDS", "Label", "Attack"]
SCALE_CLIP = 10                # validation / test are clipped to [-10, 10]
TRAIN_FRACTION = 0.30          # UNSW training subsample

CSE_CHUNK = 1_000_000          # rows per chunk for the CSE scan / collection pass
CSE_SAMPLE_FRACTION = 0.10
CSE_SCALE_CHUNK = 200_000

# ---------------------------------------------------------------------------- model
GCN_HIDDEN_DIM = 64
GCN_OUT_DIM = 64
ATT_LATENT_DIM = GCN_OUT_DIM
ATT_USE_LAYERNORM = False
ATT_DROPOUT = 0.0
CLS_HIDDEN_DIM = 32
CLS_DROPOUT = 0.0
N_PARAMS_TOTAL = 10_180        # 4,288 (GCN) + 3,779 (attention) + 2,113 (classifier)
N_PARAMS_ATTENTION = 3_779

# -------------------------------------------------------------------------- training
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 15
PATIENCE = 5
DECISION_THRESHOLD = 0.5


@dataclass
class RunConfig:
    """Run-time switches (the defaults reproduce the published experiment)."""
    num_epochs: int = NUM_EPOCHS
    cse_num_epochs: int = NUM_EPOCHS
    fresh: bool = False           # delete existing Section-11 checkpoints before running
    reuse_cse_sample: bool = True  # reload the persisted 10% CSE sample if present


# ---------------------------------------------------------------- locked reference values
class Locked:
    """Published reference values (flow counts, thresholds, best epochs).

    ``mode`` controls what a mismatch does:
      "warn"   (default) print a WARNING and continue - a full-data re-run on other hardware
               is never aborted after hours of training because of a last-digit difference;
      "strict" raise AssertionError (use to certify a bit-for-bit reproduction);
      "off"    do not check (subsets, smoke tests).
    """
    mode = "warn"

    UNSW_ROWS = 2_365_424
    UNSW_COLS = 55
    UNSW_TRAIN_IP_NODES = 43
    UNSW_TRAIN_FLOWS = 496_738
    UNSW_VAL_FLOWS = 354_814
    UNSW_TEST_FLOWS = 354_814
    UNSW_POS_NEG = (26_815, 469_923)
    UNSW_HISTORY_LEN = 9
    UNSW_EPOCH4_F1 = 0.7225

    CSE_ROWS = 20_115_529
    CSE_CLASSES = 15
    CSE_THRESHOLD = 0.478357
    CSE_TEST_FLOWS = 301_733
    CSE_GNN_F1_PRIMARY = 0.6361
    CSE_GNN_F1_REFERENCE = 0.4550
    CSE_GNN_PR_AUC = 0.635817

    S11_ATTN_EPOCH = 11
    S11_ATTN_VAL_PR_AUC = 0.782290
    S11_ATTN_THRESHOLD = 0.510303
    S11_GCN_EPOCH = 1
    S11_GCN_THRESHOLD = 0.493669


def locked_assert(condition, message=""):
    """Check a published reference value according to ``Locked.mode``."""
    if Locked.mode == "off" or condition:
        return
    if not message:                      # name the call site when no message was given
        import inspect
        frame = inspect.stack()[1]
        message = f"{Path(frame.filename).name}:{frame.lineno}"
    if Locked.mode == "strict":
        raise AssertionError(message)
    print(f"WARNING (published reference value differs): {message}")


# ------------------------------------------------------------------------------- paths
@dataclass
class RunPaths:
    unsw_csv: Path
    cse_csv: Path
    output_dir: Path

    def __post_init__(self):
        self.unsw_csv = Path(self.unsw_csv)
        self.cse_csv = Path(self.cse_csv)
        self.output_dir = Path(self.output_dir)

    @property
    def backup_dir(self) -> Path:
        return self.output_dir / "final_experiment_backup"

    @property
    def unsw_checkpoint(self) -> Path:
        return self.output_dir / "gnn_ids_unsw_best.pt"

    @property
    def cse_sample(self) -> Path:
        return self.output_dir / "cse_sample_10pct.parquet"

    @property
    def cse_checkpoint(self) -> Path:
        return self.output_dir / "gnn_ids_cse_best_prauc.pt"

    @property
    def s11_attn_checkpoint(self) -> Path:
        return self.output_dir / "gnn_ids_unsw_attn_best_prauc.pt"

    @property
    def s11_gcn_only_checkpoint(self) -> Path:
        return self.output_dir / "gnn_ids_unsw_gcn_only_best.pt"

    def make_dirs(self):
        self.backup_dir.mkdir(parents=True, exist_ok=True)
