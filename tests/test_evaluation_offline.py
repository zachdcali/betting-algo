import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pandas as pd
from evaluation import offline


def _write_summary(d: Path, payload: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(payload))


def test_discovers_summary_json_and_maps_keys(tmp_path):
    exp = tmp_path / "2026-04-25" / "xgboost" / "performance_v1__xgb_depth5_recency_hl_12y"
    _write_summary(exp, {
        "family": "xgboost", "feature_set": "performance_v1", "feature_mode": "one_hot",
        "n_features": 198, "split_label": "fixed_2022_val_2023plus_test",
        "config": {"slug": "xgb_depth5_recency_hl_12y"},
        "test_accuracy": 0.6712, "test_auc": 0.7374,
        "test_brier": 0.2073, "test_log_loss": 0.6008,
        # test_ece intentionally omitted -> should be NaN, not dropped
    })
    df = offline.discover_experiment_metrics(str(tmp_path))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["source"] == "offline"
    assert row["family"] == "xgboost"
    assert row["experiment"] == "xgb_depth5_recency_hl_12y"
    assert abs(row["accuracy"] - 0.6712) < 1e-9
    assert abs(row["log_loss"] - 0.6008) < 1e-9
    assert pd.isna(row["ece"])
    assert row["run_date"] == "2026-04-25"


def test_handles_missing_dir(tmp_path):
    df = offline.discover_experiment_metrics(str(tmp_path / "does_not_exist"))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_multiple_experiments_and_families(tmp_path):
    _write_summary(tmp_path / "2026-04-24" / "nn" / "nn_logits_128_64_robust", {
        "family": "nn", "split_label": "fixed", "config": {"slug": "nn_logits_128_64_robust"},
        "test_accuracy": 0.6627, "test_auc": 0.7268, "test_brier": 0.2109,
        "test_log_loss": 0.609, "test_ece": 0.012,
    })
    _write_summary(tmp_path / "2026-04-25" / "lightgbm" / "lgbm_native", {
        "family": "lightgbm", "split_label": "fixed", "config": {"slug": "lgbm_native"},
        "test_accuracy": 0.6708, "test_auc": 0.7301, "test_brier": 0.2081,
        "test_log_loss": 0.6020, "test_ece": 0.009,
    })
    df = offline.discover_experiment_metrics(str(tmp_path))
    assert set(df["family"]) == {"nn", "lightgbm"}
    assert len(df) == 2
