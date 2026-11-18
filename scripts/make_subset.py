"""Build small subsets of the two datasets for quick smoke runs on a laptop.

    python scripts/make_subset.py --unsw-csv data/NF-UNSW-NB15-v3.csv \
        --cse-csv data/NF-CSE-CIC-IDS2018-v3.csv --out-dir data/subset
    python main.py --smoke --unsw-csv data/subset/unsw_subset.csv \
        --cse-csv data/subset/cse_subset.csv --output-dir outputs/subset

UNSW subset: every attack flow plus the first 60,000 benign flows.
CSE subset : at most 12,000 flows per attack class (file order).
The subsets are NOT the experiment; results from them are not comparable to the published ones.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 200_000


def make_unsw_subset(src, dst, n_benign=60_000):
    parts, taken = [], 0
    for ch in pd.read_csv(src, chunksize=CHUNK):
        att, ben = ch[ch["Label"] == 1], ch[ch["Label"] == 0]
        b = ben.iloc[:max(0, n_benign - taken)]
        taken += len(b)
        parts.append(pd.concat([att, b]).sort_index())
    df = pd.concat(parts)
    df.to_csv(dst, index=False)
    print(f"UNSW subset {df.shape} -> {dst}")


def make_cse_subset(src, dst, per_class=12_000):
    counts, parts = {}, []
    dtypes = {"IPV4_SRC_ADDR": str, "IPV4_DST_ADDR": str, "Attack": str}
    for ch in pd.read_csv(src, chunksize=CHUNK, dtype=dtypes):
        keep = np.zeros(len(ch), bool)
        attack = ch["Attack"].to_numpy()
        for c in np.unique(attack):
            pos = np.flatnonzero(attack == c)
            take = max(0, per_class - counts.get(c, 0))
            keep[pos[:take]] = True
            counts[c] = counts.get(c, 0) + min(take, len(pos))
        if keep.any():
            parts.append(ch[keep])
    df = pd.concat(parts)
    df.to_csv(dst, index=False)
    print(f"CSE subset {df.shape} -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsw-csv", required=True)
    ap.add_argument("--cse-csv", required=True)
    ap.add_argument("--out-dir", default="data/subset")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    make_unsw_subset(a.unsw_csv, out / "unsw_subset.csv")
    make_cse_subset(a.cse_csv, out / "cse_subset.csv")
