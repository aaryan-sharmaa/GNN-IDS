"""NF-UNSW-NB15-v3: load, clean, split, scale and subsample (notebook sections 2-3)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..config import (DROP_FROM_X, N_FEATURES, RANDOM_SEED, SCALE_CLIP, SUBSAMPLE_SEED,
                      TRAIN_FRACTION, Locked, locked_assert)


@dataclass
class UnswData:
    """Everything the later stages need. Train arrays are the 30% subsample."""
    feature_cols: list
    X_train: pd.DataFrame
    y_train: pd.Series
    train_src_ip: pd.Series
    train_dst_ip: pd.Series
    train_row_id: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    val_src_ip: pd.Series
    val_dst_ip: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_src_ip: pd.Series
    test_dst_ip: pd.Series
    scaler: MinMaxScaler
    n_train_full: int
    strat_col: str


def unsw_feature_columns(csv_path) -> list:
    """The 49 feature columns, read from the CSV header only."""
    cols = pd.read_csv(csv_path, nrows=0).columns
    return [c for c in cols if c not in DROP_FROM_X]


def load_and_clean(csv_path) -> pd.DataFrame:
    """Section 2 + 3.1: read the CSV, add a row id, replace inf/NaN by 0 in numeric columns."""
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")

    locked_assert(df.shape[0] == Locked.UNSW_ROWS, f"Unexpected row count: {df.shape[0]}")
    locked_assert(df.shape[1] == Locked.UNSW_COLS, f"Unexpected column count: {df.shape[1]}")

    # Preserve original row identity before any reindexing/filtering
    df["orig_row_id"] = np.arange(len(df))

    # Clean: inf -> NaN -> 0, numeric columns only (IPs and Attack are untouched)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop("orig_row_id")  # never touch the row id

    n_inf_before = np.isinf(df[numeric_cols].to_numpy()).sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    n_nan_after_inf = df[numeric_cols].isna().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print(f"Infinite values replaced: {n_inf_before}")
    print(f"NaN values (including former infinities) filled with 0: {n_nan_after_inf}")
    return df


def stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Two-stage stratified split: 70% train, then 15/15 from the remaining 30%.

    Stratify by Attack (finer than Label) so rare classes appear in every split;
    fall back to Label only if Attack stratification fails.
    """
    try:
        df_train, df_temp = train_test_split(
            df, test_size=0.30, random_state=seed, stratify=df["Attack"])
        df_val, df_test = train_test_split(
            df_temp, test_size=0.50, random_state=seed, stratify=df_temp["Attack"])
        strat_col_used = "Attack"
    except ValueError as e:
        print(f"Attack-stratified split failed ({e}); falling back to Label.")
        df_train, df_temp = train_test_split(
            df, test_size=0.30, random_state=seed, stratify=df["Label"])
        df_val, df_test = train_test_split(
            df_temp, test_size=0.50, random_state=seed, stratify=df_temp["Label"])
        strat_col_used = "Label"

    del df_temp
    print(f"Stratification column used: {strat_col_used}")
    print(f"train: {len(df_train):,} | val: {len(df_val):,} | test: {len(df_test):,}")
    return df_train, df_val, df_test, strat_col_used


def _split_frame(d, feature_cols):
    return {
        "X": d[feature_cols].copy(),
        "y": d["Label"].copy(),
        "attack": d["Attack"].copy(),
        "src_ip": d["IPV4_SRC_ADDR"].copy(),
        "dst_ip": d["IPV4_DST_ADDR"].copy(),
        "row_id": d["orig_row_id"].copy(),
    }


def prepare_unsw(csv_path, train_fraction: float = TRAIN_FRACTION) -> UnswData:
    """Run notebook sections 2-3 and return the prepared splits."""
    df = load_and_clean(csv_path)
    df_train, df_val, df_test, strat_col_used = stratified_split(df)

    # --- 3.3 separate features, IPs, target and attack labels
    feature_cols = [c for c in df.columns if c not in DROP_FROM_X]
    assert len(feature_cols) == N_FEATURES, f"Expected {N_FEATURES} feature columns, got {len(feature_cols)}"

    train_parts = _split_frame(df_train, feature_cols)
    val_parts = _split_frame(df_val, feature_cols)
    test_parts = _split_frame(df_test, feature_cols)
    del df, df_train, df_val, df_test

    X_train, y_train, attack_train = train_parts["X"], train_parts["y"], train_parts["attack"]
    X_val, y_val = val_parts["X"], val_parts["y"]
    X_test, y_test = test_parts["X"], test_parts["y"]
    train_src_ip, train_dst_ip = train_parts["src_ip"], train_parts["dst_ip"]
    train_row_id = train_parts["row_id"]

    # --- 3.4 fit the scaler on the full training split ONLY
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), columns=feature_cols, index=X_train.index)
    X_val = pd.DataFrame(np.clip(scaler.transform(X_val), -SCALE_CLIP, SCALE_CLIP),
                         columns=feature_cols, index=X_val.index)
    X_test = pd.DataFrame(np.clip(scaler.transform(X_test), -SCALE_CLIP, SCALE_CLIP),
                          columns=feature_cols, index=X_test.index)

    # --- 3.6 stratified training subsample (validation and test stay full size)
    n_train_full = len(X_train)
    if train_fraction >= 1.0:
        sub_idx = X_train.index
        print("TRAIN_FRACTION >= 1.0 -> using the full training split")
    else:
        # Sample within each Attack class so the binary ratio and every rare class survive.
        sub_idx = (
            attack_train.groupby(attack_train, group_keys=False)
            .sample(frac=train_fraction, random_state=SUBSAMPLE_SEED)
            .index.sort_values()          # keep original row order for determinism
        )

    X_train_sub = X_train.loc[sub_idx]
    y_train_sub = y_train.loc[sub_idx]
    train_src_ip_sub = train_src_ip.loc[sub_idx]
    train_dst_ip_sub = train_dst_ip.loc[sub_idx]
    train_row_id_sub = train_row_id.loc[sub_idx]

    assert len(X_train_sub) == len(y_train_sub) == len(train_src_ip_sub) == len(train_row_id_sub)
    assert X_train_sub.index.equals(train_src_ip_sub.index), "Index alignment broken"

    print(f"TRAIN_FRACTION = {train_fraction} (seed {SUBSAMPLE_SEED})")
    print(f"Training flows: {n_train_full:,} -> {len(X_train_sub):,}")
    print(f"Validation / test unchanged: {len(X_val):,} / {len(X_test):,}")
    print(f"Scaler fitted on {scaler.n_samples_seen_:,} rows (full training split)")

    return UnswData(
        feature_cols=feature_cols,
        X_train=X_train_sub, y_train=y_train_sub,
        train_src_ip=train_src_ip_sub, train_dst_ip=train_dst_ip_sub,
        train_row_id=train_row_id_sub,
        X_val=X_val, y_val=y_val,
        val_src_ip=val_parts["src_ip"], val_dst_ip=val_parts["dst_ip"],
        X_test=X_test, y_test=y_test,
        test_src_ip=test_parts["src_ip"], test_dst_ip=test_parts["dst_ip"],
        scaler=scaler, n_train_full=n_train_full, strat_col=strat_col_used,
    )
