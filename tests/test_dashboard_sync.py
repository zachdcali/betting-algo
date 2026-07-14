import json
from pathlib import Path
import sys

import pandas as pd

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

import dashboard_sync  # noqa: E402
from evaluation import cohorts  # noqa: E402
from models.inference import EXACT_141_FEATURES  # noqa: E402


def _aggregate_feature(snapshot_id: str) -> dict:
    payload = {name: 0.0 for name in EXACT_141_FEATURES}
    payload.update({
        "Surface_Hard": 1.0, "Level_A": 1.0, "Round_R32": 1.0,
        "P1_Hand_U": 1.0, "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0, "P2_Country_Other": 1.0,
    })
    return {
        "feature_snapshot_id": snapshot_id,
        "build_status": "ok",
        "features_complete": True,
        "features_json": json.dumps(payload),
    }


def test_materialized_metrics_use_remote_plus_local_merged_cohort(monkeypatch):
    base = {
        "features_complete": True,
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "model_p1_prob": 0.60,
        "p1_odds_decimal": 1.8,
        "p2_odds_decimal": 2.1,
    }
    local = pd.DataFrame([{
        **base, "match_uid": "local", "actual_winner": 1,
        "feature_snapshot_id": "feat_local", "feature_snapshot_verified": True,
    }])
    merged = pd.DataFrame([
        local.iloc[0].to_dict(),
        {**base, "match_uid": "remote", "actual_winner": 2,
         "feature_snapshot_id": "feat_remote"},
    ])
    features = pd.DataFrame([
        _aggregate_feature("feat_local"),
        _aggregate_feature("feat_remote"),
    ])
    monkeypatch.setattr(cohorts, "load_prediction_log", lambda _prod: local)

    result = dashboard_sync._build_model_metrics(
        "sync_test", pred_log=merged, shadow_log=pd.DataFrame(), feature_log=features,
    )

    nn_gold = result[(result["model"] == "nn") & (result["tier"] == "gold")]
    assert int(nn_gold.iloc[0]["n"]) == 2
    assert nn_gold.iloc[0]["sync_id"] == "sync_test"
    assert nn_gold.iloc[0]["dashboard_row_key"] == "gold:nn"
    assert result["dashboard_row_key"].is_unique


def test_feature_projection_recovers_exact_rows_from_immutable_run_files(tmp_path):
    aggregate = tmp_path / "feature_vectors.csv"
    pd.DataFrame([{
        "p1": "Legacy", "p2": "Row", "match_date": "2026-07-12",
        "logged_at": "2026-07-12T10:00:00Z",
    }]).to_csv(aggregate, index=False)
    run_feature = {name: 0.0 for name in EXACT_141_FEATURES}
    run_feature.update({
        "Surface_Hard": 1.0, "Level_A": 1.0, "Round_R32": 1.0,
        "P1_Hand_U": 1.0, "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0, "P2_Country_Other": 1.0,
        "player1_raw": "Player One", "player2_raw": "Player Two",
        "meta_match_date": "2026-07-13", "timestamp": "2026-07-13T10:00:00Z",
        "run_id": "run_1", "match_uid": "match_1",
        "feature_snapshot_id": "feature_1", "status": "ok",
        "meta_defaulted_features": "", "Player1_Rank": 12,
        "P1_Hand_R": 1, "P1_Hand_L": 0, "P1_Hand_U": 0,
        "P2_Hand_R": 0, "P2_Hand_L": 1, "P2_Hand_U": 0,
    })
    pd.DataFrame([run_feature]).to_csv(tmp_path / "features_20260713_100000.csv", index=False)

    projected = dashboard_sync._load_feature_state(aggregate)
    recovered = projected[projected["feature_snapshot_id"] == "feature_1"].iloc[0]

    assert recovered["match_uid"] == "match_1"
    assert recovered["run_id"] == "run_1"
    assert recovered["features_complete"]
    assert len(recovered["feature_vector_sha256"]) == 64
    assert recovered["p1_hand"] == "R"
    assert recovered["p2_hand"] == "L"
    assert json.loads(recovered["features_json"])["Player1_Rank"] == 12.0


def test_manifest_separates_latest_attempt_from_prediction_slate():
    runs = pd.DataFrame([
        {"run_id": "run_good", "run_kind": "prediction_pipeline",
         "status": "success", "started_at": "2026-07-13T10:00:00Z"},
        {"run_id": "run_running", "run_kind": "prediction_pipeline",
         "status": "running", "started_at": "2026-07-13T11:00:00Z"},
    ])
    snapshots = pd.DataFrame([{
        "run_id": "run_good", "logged_at": "2026-07-13T10:30:00Z",
    }])

    assert dashboard_sync._latest_run_id(runs) == "run_running"
    assert dashboard_sync._accepted_prediction_run_id(runs, snapshots) == "run_good"


def test_manifest_never_accepts_snapshot_from_explicit_failed_run():
    runs = pd.DataFrame([{
        "run_id": "run_failed", "run_kind": "prediction_pipeline",
        "status": "failed", "started_at": "2026-07-13T12:00:00Z",
    }])
    snapshots = pd.DataFrame([{
        "run_id": "run_failed", "logged_at": "2026-07-13T12:01:00Z",
    }])

    assert dashboard_sync._accepted_prediction_run_id(runs, snapshots) == ""


def test_latest_run_prefers_valid_started_at_over_missing_timestamp():
    runs = pd.DataFrame([
        {"run_id": "run_valid", "run_kind": "prediction_pipeline",
         "status": "success", "started_at": "2026-07-13T12:00:00Z"},
        {"run_id": "run_missing_time", "run_kind": "prediction_pipeline",
         "status": "running", "started_at": ""},
    ])

    assert dashboard_sync._latest_run_id(runs) == "run_valid"


def test_feature_projection_normalizes_nonfinite_json_values():
    assert dashboard_sync._json_scalar(float("nan")) is None
    assert dashboard_sync._json_scalar(float("inf")) is None
    assert dashboard_sync._json_scalar("-Infinity") is None
