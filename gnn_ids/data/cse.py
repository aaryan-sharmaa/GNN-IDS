"""NF-CSE-CIC-IDS2018-v3: chunked scan, 10% stratified sample, split and scaling (section 10.1-10.3)."""
import gc
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..config import (CSE_CHUNK, CSE_DROP, CSE_SAMPLE_FRACTION, CSE_SCALE_CHUNK, N_FEATURES,
                      RANDOM_SEED, SCALE_CLIP, Locked, locked_assert)

_DTYPES = {"IPV4_SRC_ADDR": str, "IPV4_DST_ADDR": str, "Attack": str}


@dataclass
class CseData:
    feature_cols: list
    X_train: np.ndarray
    y_train: pd.Series
    train_src_ip: pd.Series
    train_dst_ip: pd.Series
    train_row_id: pd.Series
    X_val: np.ndarray
    y_val: pd.Series
    val_src_ip: pd.Series
    val_dst_ip: pd.Series
    X_test: np.ndarray
    y_test: pd.Series
    test_src_ip: pd.Series
    test_dst_ip: pd.Series
    scaler: MinMaxScaler
    n_tr: int
    n_va: int
    n_te: int
    strat_col: str


# ---------------------------------------------------------------- 10.1 chunked scan
def scan_cse(csv_path):
    """One chunked pass: row count, label counts and attack counts (+ NaN/inf check)."""
    cse_head = pd.read_csv(csv_path, nrows=1000)
    cse_columns = cse_head.columns.tolist()
    feature_cols = [c for c in cse_columns if c not in CSE_DROP]

    rows = 0
    label_counts, attack_counts = Counter(), Counter()
    n_nan = n_inf = 0

    t0 = time.time()
    reader = pd.read_csv(csv_path, chunksize=CSE_CHUNK, dtype=_DTYPES)
    for i, chunk in enumerate(reader):
        rows += len(chunk)
        label_counts.update(chunk["Label"].value_counts(dropna=False).to_dict())
        attack_counts.update(chunk["Attack"].value_counts(dropna=False).to_dict())
        feats = chunk[feature_cols]
        n_nan += int(feats.isna().sum().sum())
        n_inf += int(np.isinf(feats.to_numpy(dtype=np.float64)).sum())
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  chunk {i + 1:>2}: {rows:,} rows | {time.time() - t0:.0f}s")

    print(f"Scan complete: {rows:,} rows in {(time.time() - t0) / 60:.1f} min "
          f"| NaN {n_nan} | inf {n_inf}")
    locked_assert(rows == Locked.CSE_ROWS, f"CSE rows {rows:,} differ from the published {Locked.CSE_ROWS:,}")
    return {"rows": rows, "label_counts": label_counts, "attack_counts": attack_counts,
            "feature_cols": feature_cols}


# ------------------------------------------------------------ 10.2 stratified sample
def allocate_class_samples(attack_counts, total_rows, fraction=CSE_SAMPLE_FRACTION):
    """Per-class targets with largest-remainder rounding, so the total is exact."""
    target_rows = int(total_rows * fraction)

    full_counts = {str(k): int(v) for k, v in attack_counts.items()}
    assert sum(full_counts.values()) == total_rows
    locked_assert(len(full_counts) == Locked.CSE_CLASSES)

    classes = sorted(full_counts)          # sorted -> stable class ordering
    exact = {c: full_counts[c] * fraction for c in classes}
    alloc = {c: int(np.floor(exact[c])) for c in classes}
    shortfall = target_rows - sum(alloc.values())

    remainders = sorted(((exact[c] - alloc[c], c) for c in classes), reverse=True)
    for _, c in remainders[:shortfall]:
        alloc[c] += 1

    assert sum(alloc.values()) == target_rows, "Allocation does not sum to target"
    assert all(alloc[c] >= 1 for c in classes), "A class received zero samples"
    assert all(alloc[c] <= full_counts[c] for c in classes)
    print(f"Target sample: {target_rows:,} flows ({fraction:.0%}); "
          f"rounding shortfall {shortfall} row(s)")
    return classes, full_counts, alloc, target_rows


def build_selection_masks(classes, full_counts, alloc, seed=RANDOM_SEED):
    """Per-class RNG (seed + class index) -> boolean mask over position-within-class."""
    masks = {}
    for i, c in enumerate(classes):
        n_c = full_counts[c]
        rng = np.random.default_rng(seed + i)
        chosen = rng.choice(n_c, size=alloc[c], replace=False)
        mask = np.zeros(n_c, dtype=bool)
        mask[chosen] = True
        masks[c] = mask
    return masks


def collect_sample(csv_path, classes, masks, full_counts):
    """Single chunked pass that keeps only the selected rows."""
    offsets = {c: 0 for c in classes}
    parts = []

    t0 = time.time()
    reader = pd.read_csv(csv_path, chunksize=CSE_CHUNK, dtype=_DTYPES)
    for i, chunk in enumerate(reader):
        keep = np.zeros(len(chunk), dtype=bool)
        attack_vals = chunk["Attack"].to_numpy()

        for c in np.unique(attack_vals):
            pos = np.flatnonzero(attack_vals == c)
            ranks = offsets[c] + np.arange(len(pos))
            keep[pos] = masks[c][ranks]
            offsets[c] += len(pos)

        if keep.any():
            parts.append(chunk.loc[keep].copy())

        if (i + 1) % 5 == 0 or i == 0:
            kept = sum(len(p) for p in parts)
            print(f"  chunk {i + 1:>2}: kept {kept:,} so far | {time.time() - t0:.0f}s")

    sample = pd.concat(parts, ignore_index=True)
    del parts

    float_cols = sample.select_dtypes(include=["float64"]).columns
    sample[float_cols] = sample[float_cols].astype(np.float32)

    ok = all(offsets[c] == full_counts[c] for c in classes)
    print(f"Sampling complete in {(time.time() - t0) / 60:.1f} min | shape {sample.shape} "
          f"| offsets match full counts: {ok}")
    assert ok
    return sample


def load_or_build_sample(csv_path, sample_path, reuse=True):
    """Return the 10% stratified sample; reload the persisted parquet when available."""
    if reuse and sample_path.exists():
        print(f"Reloading persisted CSE sample: {sample_path}")
        return pd.read_parquet(sample_path)

    scan = scan_cse(csv_path)
    classes, full_counts, alloc, target_rows = allocate_class_samples(
        scan["attack_counts"], scan["rows"])
    masks = build_selection_masks(classes, full_counts, alloc)
    sample = collect_sample(csv_path, classes, masks, full_counts)
    assert len(sample) == target_rows, f"Sample size {len(sample):,} != target {target_rows:,}"
    sample.to_parquet(sample_path, index=False)
    print(f"Saved {sample_path} ({sample_path.stat().st_size / 1024**2:.1f} MB)")
    return sample


# -------------------------------------------------------- 10.3 split, clean, scale
def _extract(sample, feature_cols, idx):
    """Extract one split's objects from the sample."""
    X = sample[feature_cols].to_numpy(dtype=np.float32)[idx]
    return {
        "X": X,
        "y": sample["Label"].iloc[idx].reset_index(drop=True),
        "attack": sample["Attack"].iloc[idx].reset_index(drop=True),
        "src_ip": sample["IPV4_SRC_ADDR"].iloc[idx].reset_index(drop=True),
        "dst_ip": sample["IPV4_DST_ADDR"].iloc[idx].reset_index(drop=True),
        "row_id": sample["cse_orig_row_id"].iloc[idx].reset_index(drop=True),
    }


def scale_inplace(X, scaler, clip=False, chunk=CSE_SCALE_CHUNK):
    """Transform X in place, chunk by chunk, keeping it float32 throughout."""
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        block = scaler.transform(X[s:e])
        if clip:
            np.clip(block, -SCALE_CLIP, SCALE_CLIP, out=block)
        X[s:e] = block.astype(np.float32, copy=False)
        del block


def prepare_cse(csv_path, sample_path, unsw_feature_cols, reuse_sample=True) -> CseData:
    """Sections 10.1-10.3: sample, split 70/15/15 (stratified), fit scaler on train only."""
    sample = load_or_build_sample(csv_path, sample_path, reuse=reuse_sample)

    feature_cols = [c for c in sample.columns
                    if c not in CSE_DROP and c != "cse_orig_row_id"]
    assert feature_cols == unsw_feature_cols, "CSE features differ from UNSW"
    assert len(feature_cols) == N_FEATURES

    # 10.3.1 row ids and a column-by-column inf/NaN check (no full-width float64 copy)
    sample["cse_orig_row_id"] = np.arange(len(sample), dtype=np.int64)
    n_inf_found = n_nan_found = 0
    for col in feature_cols:
        vals = sample[col].to_numpy()
        n_i = int(np.isinf(vals).sum())
        n_n = int(np.isnan(vals).sum())
        if n_i or n_n:
            n_inf_found += n_i
            n_nan_found += n_n
            sample[col] = sample[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        del vals
    print(f"Rows {len(sample):,} | inf fixed {n_inf_found} | NaN fixed {n_nan_found}")

    # 10.3.2 stratified 70/15/15 split on index arrays
    attack_all = sample["Attack"].to_numpy()
    positions = np.arange(len(sample))
    try:
        idx_train, idx_temp = train_test_split(
            positions, test_size=0.30, random_state=RANDOM_SEED, stratify=attack_all)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=attack_all[idx_temp])
        strat_col = "Attack"
    except ValueError as e:
        print(f"Attack-stratified split failed ({e}); falling back to Label.")
        label_all = sample["Label"].to_numpy()
        idx_train, idx_temp = train_test_split(
            positions, test_size=0.30, random_state=RANDOM_SEED, stratify=label_all)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=label_all[idx_temp])
        strat_col = "Label"

    # Sorted so extraction is sequential and preserves sample order
    idx_train.sort(); idx_val.sort(); idx_test.sort()
    del idx_temp, positions
    gc.collect()

    n_tr, n_va, n_te = len(idx_train), len(idx_val), len(idx_test)
    assert n_tr + n_va + n_te == len(sample)
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0
    print(f"Stratified by {strat_col}: train {n_tr:,} | val {n_va:,} | test {n_te:,}")

    # 10.3.3 extract one split at a time, then release the sample
    _tr = _extract(sample, feature_cols, idx_train); gc.collect()
    _va = _extract(sample, feature_cols, idx_val); gc.collect()
    _te = _extract(sample, feature_cols, idx_test); gc.collect()
    del sample, attack_all
    gc.collect()

    assert _tr["X"].shape == (n_tr, N_FEATURES) and _tr["X"].dtype == np.float32
    assert _va["X"].shape == (n_va, N_FEATURES) and _te["X"].shape == (n_te, N_FEATURES)

    # 10.3.4 fit the scaler on training rows only, transform in place
    scaler = MinMaxScaler()
    scaler.fit(_tr["X"])
    scale_inplace(_tr["X"], scaler, clip=False)
    gc.collect()
    scale_inplace(_va["X"], scaler, clip=True)
    gc.collect()
    scale_inplace(_te["X"], scaler, clip=True)
    gc.collect()
    assert scaler.n_samples_seen_ == n_tr, "Scaler was not fit on training rows only"
    print(f"Scaler fitted on {scaler.n_samples_seen_:,} training rows; val/test clipped to "
          f"[-{SCALE_CLIP}, {SCALE_CLIP}]")

    for name, X in [("train", _tr["X"]), ("val", _va["X"]), ("test", _te["X"])]:
        assert np.isfinite(X).all(), f"non-finite values in scaled {name} features"

    return CseData(
        feature_cols=feature_cols,
        X_train=_tr["X"], y_train=_tr["y"], train_src_ip=_tr["src_ip"],
        train_dst_ip=_tr["dst_ip"], train_row_id=_tr["row_id"],
        X_val=_va["X"], y_val=_va["y"], val_src_ip=_va["src_ip"], val_dst_ip=_va["dst_ip"],
        X_test=_te["X"], y_test=_te["y"], test_src_ip=_te["src_ip"], test_dst_ip=_te["dst_ip"],
        scaler=scaler, n_tr=n_tr, n_va=n_va, n_te=n_te, strat_col=strat_col,
    )
