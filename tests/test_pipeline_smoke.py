"""End-to-end smoke test of the single entry point on a tiny synthetic dataset (CPU, ~1 min).

It checks that every stage runs and writes its artifacts. It does NOT check the published
numbers - those need the real datasets (see README).
"""
import pytest

from main import main
from conftest import make_netflow_csv


@pytest.mark.slow
def test_full_pipeline_on_synthetic_data(tmp_path, synthetic_csv):
    cse_csv = make_netflow_csv(tmp_path / "cse.csv", n_rows=8000, seed=1)
    out = tmp_path / "out"
    main(["--stage", "all", "--smoke", "--epochs", "3",
          "--unsw-csv", str(synthetic_csv), "--cse-csv", str(cse_csv),
          "--output-dir", str(out)])

    bk = out / "final_experiment_backup"
    for name in ["unsw_final_threshold_results.csv", "cse_final_comparison.csv",
                 "UNSW_primary_results.csv", "CSE_primary_results.csv",
                 "s11_final_test_results.csv", "s11_hostpair_diagnostic_summary.csv"]:
        assert (bk / name).exists(), name
    for name in ["gnn_ids_unsw_best.pt", "gnn_ids_cse_best_prauc.pt",
                 "gnn_ids_unsw_attn_best_prauc.pt", "gnn_ids_unsw_gcn_only_best.pt"]:
        assert (out / name).exists(), name
