"""Shared fixtures: a tiny synthetic NetFlow-like dataset with the real 55-column schema."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn_ids.config import DROP_FROM_X  # noqa: E402

FEATURES = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES",
    "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE",
    "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE", "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]
assert len(FEATURES) == 49


def make_netflow_csv(path, n_rows=6000, n_hosts=14, classes=("Benign", "Exploits", "Fuzzers"),
                     seed=0):
    """Write a synthetic CSV with the 55 NF-v3 columns. Attack flows are linearly separable
    from benign ones by a few features, so the tiny models have something to learn."""
    rng = np.random.default_rng(seed)
    attack = rng.choice(classes, size=n_rows, p=[0.7, 0.2, 0.1])
    hosts = np.array([f"10.0.0.{i}" for i in range(n_hosts)])
    X = rng.integers(0, 1000, size=(n_rows, len(FEATURES))).astype(np.int64)
    X[attack != "Benign", 4] += 800            # IN_BYTES
    X[attack == "Fuzzers", 14] += 500          # MIN_TTL
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "FLOW_START_MILLISECONDS", np.arange(n_rows) * 10)
    df.insert(1, "FLOW_END_MILLISECONDS", np.arange(n_rows) * 10 + 5)
    df.insert(2, "IPV4_SRC_ADDR", rng.choice(hosts, n_rows))
    df.insert(4, "IPV4_DST_ADDR", rng.choice(hosts, n_rows))
    df["Label"] = (attack != "Benign").astype(int)
    df["Attack"] = attack
    assert df.shape[1] == 55
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def feature_cols():
    return [c for c in FEATURES]


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    return make_netflow_csv(tmp_path_factory.mktemp("data") / "synthetic.csv")
