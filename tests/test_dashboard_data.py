from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data import (  # noqa: E402
    build_apples_to_apples_rows,
    build_derived_run_history,
    build_live_latest_snapshots,
    latest_pipeline_run_id,
    normalize_prediction_log,
)


def _row(**overrides):
    row = {
        "match_uid": "m1", "p1": "A", "p2": "B",
        "logging_quality": "snapshot_v2", "rescore_quality": "exact_feature_snapshot",
        "features_complete": "True", "feature_snapshot_verified": "True", "actual_winner": 1,
        "model_p1_prob": 0.6, "xgb_p1_prob": 0.6, "rf_p1_prob": 0.6,
        "market_p1_prob": 0.5,
    }
    row.update(overrides)
    return row


def test_gold_and_settlement_flags_parse_explicitly():
    normalized = normalize_prediction_log(pd.DataFrame([
        _row(match_uid="good"),
        _row(match_uid="false_string", features_complete="False"),
        _row(match_uid="unverified", feature_snapshot_verified="False"),
        _row(match_uid="missing_verification", feature_snapshot_verified=None),
        _row(match_uid="void", actual_winner=-1),
    ]))
    by_id = normalized.set_index("match_uid")
    assert bool(by_id.loc["good", "decision_grade"])
    assert not bool(by_id.loc["false_string", "decision_grade"])
    assert not bool(by_id.loc["unverified", "decision_grade"])
    assert not bool(by_id.loc["missing_verification", "decision_grade"])
    assert not bool(by_id.loc["void", "is_settled"])
    assert bool(by_id.loc["good", "market_has_pick"])  # even-money is valid


def test_common_cohort_requires_gold_and_derives_truth_from_winner():
    normalized = normalize_prediction_log(pd.DataFrame([
        _row(match_uid="gold", model_correct=0, market_correct=0),
        _row(match_uid="incomplete", features_complete="False"),
        _row(match_uid="unverified", feature_snapshot_verified=False),
    ]))
    apples = build_apples_to_apples_rows(normalized)
    assert list(apples["match_uid"]) == ["gold"]
    assert apples.iloc[0]["p1_won"] == 1


def test_live_slate_uses_only_latest_prediction_run():
    snapshots = pd.DataFrame([
        {"run_id": "run_old", "match_uid": "old", "prediction_uid": "p1",
         "logged_at": pd.Timestamp("2026-07-13T10:00:00Z"), "p1": "A", "p2": "B"},
        {"run_id": "run_new", "match_uid": "new", "prediction_uid": "p2",
         "logged_at": pd.Timestamp("2026-07-13T11:00:00Z"), "p1": "C", "p2": "D"},
    ])
    latest = build_live_latest_snapshots(snapshots)
    assert list(latest["match_uid"]) == ["new"]


def test_live_slate_never_falls_back_after_filtering_latest_run():
    snapshots = pd.DataFrame([
        {"run_id": "run_old", "match_uid": "old", "prediction_uid": "p1",
         "logged_at": pd.Timestamp("2026-07-13T10:00:00"), "surface": "Clay"},
        {"run_id": "run_new", "match_uid": "new", "prediction_uid": "p2",
         "logged_at": pd.Timestamp("2026-07-13T11:00:00"), "surface": "Hard"},
    ])
    runs = pd.DataFrame([
        {"run_id": "run_old", "run_kind": "prediction_pipeline",
         "started_at": pd.Timestamp("2026-07-13T10:00:00")},
        {"run_id": "run_new", "run_kind": "prediction_pipeline",
         "started_at": pd.Timestamp("2026-07-13T11:00:00")},
    ])
    run_id = latest_pipeline_run_id(runs, snapshots)
    clay_only = snapshots[(snapshots["run_id"] == run_id) & (snapshots["surface"] == "Clay")]
    assert run_id == "run_new"
    assert build_live_latest_snapshots(clay_only, run_id=run_id).empty


def test_live_slate_uses_latest_accepted_prediction_run_not_latest_attempt():
    snapshots = pd.DataFrame([
        {"run_id": "run_good", "match_uid": "good", "prediction_uid": "p1",
         "logged_at": pd.Timestamp("2026-07-13T10:00:00Z")},
        {"run_id": "run_failed", "match_uid": "failed", "prediction_uid": "p2",
         "logged_at": pd.Timestamp("2026-07-13T11:00:00Z")},
    ])
    runs = pd.DataFrame([
        {"run_id": "run_good", "run_kind": "prediction_pipeline", "status": "success",
         "started_at": "2026-07-13T10:00:00Z"},
        {"run_id": "run_failed", "run_kind": "prediction_pipeline", "status": "failed",
         "started_at": "2026-07-13T11:00:00Z"},
        {"run_id": "run_running", "run_kind": "prediction_pipeline", "status": "running",
         "started_at": "2026-07-13T12:00:00Z"},
    ])

    assert latest_pipeline_run_id(runs, snapshots) == "run_good"


def test_odds_only_derived_run_history_does_not_crash():
    odds = pd.DataFrame([{
        "run_id": "run_1", "logged_at": pd.Timestamp("2026-07-13T11:00:00Z"),
        "odds_snapshot_uid": "odds_1",
    }])
    result = build_derived_run_history(pd.DataFrame(), odds)
    assert result.iloc[0]["run_id"] == "run_1"
    assert result.iloc[0]["odds_rows_fetched"] == 1


def test_prediction_timestamps_accept_mixed_iso_and_space_formats():
    normalized = normalize_prediction_log(pd.DataFrame([
        _row(match_uid="iso", logged_at="2026-07-13T10:00:00+00:00"),
        _row(match_uid="space", logged_at="2026-07-13 11:00:00"),
    ]))
    assert normalized["effective_logged_at"].notna().all()
