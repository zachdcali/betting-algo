from pathlib import Path
import sys

import pandas as pd

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

from operational_state import STATE_SPECS, merge_state_frames  # noqa: E402


SPECS = {spec.table: spec for spec in STATE_SPECS}


def test_prediction_merge_preserves_remote_only_settlement():
    existing = pd.DataFrame([
        {"match_uid": "m1", "actual_winner": "1", "features_complete": "True",
         "model_p1_prob": "0.60", "settled_at": "2026-07-10T00:00:00Z"},
    ])
    incoming = pd.DataFrame([
        {"match_uid": "m2", "actual_winner": "", "features_complete": "True",
         "model_p1_prob": "0.55", "logged_at": "2026-07-11T00:00:00Z"},
    ])
    merged = merge_state_frames(existing, incoming, SPECS["dash_predictions"])
    assert set(merged["match_uid"]) == {"m1", "m2"}
    assert merged.loc[merged["match_uid"] == "m1", "actual_winner"].iloc[0] == "1"


def test_prediction_merge_cannot_unsettle_or_downgrade_a_row():
    existing = pd.DataFrame([
        {"match_uid": "m1", "actual_winner": "2.0", "features_complete": "True",
         "model_p1_prob": "0.62", "settled_at": "2026-07-10T00:00:00Z"},
    ])
    incoming = pd.DataFrame([
        {"match_uid": "m1", "actual_winner": "", "features_complete": "False",
         "model_p1_prob": "", "logged_at": "2026-07-12T00:00:00Z"},
    ])
    merged = merge_state_frames(existing, incoming, SPECS["dash_predictions"])
    assert len(merged) == 1
    assert float(merged.iloc[0]["actual_winner"]) == 2.0
    assert merged.iloc[0]["model_p1_prob"] == "0.62"


def test_prediction_merge_joins_settlement_to_best_complete_inference():
    settled_incomplete = pd.DataFrame([{
        "match_uid": "m1", "actual_winner": "2", "score": "4-6 6-7",
        "record_status": "settled", "settled_at": "2026-07-12T12:00:00Z",
        "features_complete": "False", "model_p1_prob": "0.52",
        "feature_snapshot_id": "feat_incomplete",
    }])
    pending_complete = pd.DataFrame([{
        "match_uid": "m1", "actual_winner": "", "record_status": "pending",
        "features_complete": "True", "model_p1_prob": "0.61",
        "feature_snapshot_id": "feat_complete", "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
    }])
    merged = merge_state_frames(
        settled_incomplete, pending_complete, SPECS["dash_predictions"]
    )
    row = merged.iloc[0]
    assert len(merged) == 1
    assert row["actual_winner"] == "2"
    assert row["record_status"] == "settled"
    assert row["feature_snapshot_id"] == "feat_complete"
    assert row["model_p1_prob"] == "0.61"
    assert row["features_complete"] == "True"


def test_prediction_merge_cannot_resurrect_identity_conflict():
    durable_pending = pd.DataFrame([{
        "match_uid": "m1", "actual_winner": "", "record_status": "pending",
        "features_complete": "True", "model_p1_prob": "0.61",
        "identity_status": "canonical", "feature_snapshot_id": "feat_open",
        "latest_logged_at": "2026-07-14T09:00:00Z",
    }])
    local_conflict = pd.DataFrame([{
        "match_uid": "m1", "actual_winner": "",
        "record_status": "identity_conflict", "features_complete": "False",
        "model_p1_prob": "0.61", "identity_status": "conflict",
        "identity_related_match_uid": "m0",
        "identity_conflict_fields": "round",
        "defaulted_features": "match_identity_conflict",
        "latest_logged_at": "2026-07-14T10:00:00Z",
    }])

    merged = merge_state_frames(
        durable_pending, local_conflict, SPECS["dash_predictions"]
    )
    row = merged.iloc[0]

    assert row["record_status"] == "identity_conflict"
    assert row["features_complete"] == "False"
    assert row["identity_status"] == "conflict"
    assert row["identity_conflict_fields"] == "round"


def test_newer_settlement_cannot_override_older_identity_conflict():
    identity_conflict = pd.DataFrame([{
        "match_uid": "m1", "actual_winner": "",
        "record_status": "identity_conflict", "features_complete": "False",
        "model_p1_prob": "0.61", "identity_status": "conflict",
        "identity_related_match_uid": "m0",
        "identity_conflict_fields": "match_date",
        "defaulted_features": "match_identity_conflict",
        "latest_logged_at": "2026-07-14T09:00:00Z",
    }])
    newer_settlement = pd.DataFrame([{
        "match_uid": "m1", "actual_winner": "1",
        "record_status": "settled", "features_complete": "True",
        "model_p1_prob": "0.61", "identity_status": "canonical",
        "settled_at": "2026-07-14T12:00:00Z",
    }])

    merged = merge_state_frames(
        identity_conflict, newer_settlement, SPECS["dash_predictions"]
    )
    row = merged.iloc[0]

    assert row["record_status"] == "identity_conflict"
    assert row["features_complete"] == "False"
    assert row["identity_status"] == "conflict"
    assert row["identity_conflict_fields"] == "match_date"


def test_terminal_run_and_bet_outrank_newer_running_or_pending_copy():
    run_existing = pd.DataFrame([{"run_id": "r1", "status": "success",
                                  "completed_at": "2026-07-10T00:00:00Z"}])
    run_incoming = pd.DataFrame([{"run_id": "r1", "status": "running",
                                  "started_at": "2026-07-11T00:00:00Z"}])
    bet_existing = pd.DataFrame([{"bet_id": "b1", "status": "settled", "outcome": "win"}])
    bet_incoming = pd.DataFrame([{"bet_id": "b1", "status": "pending", "outcome": ""}])
    assert merge_state_frames(run_existing, run_incoming, SPECS["dash_runs"]).iloc[0]["status"] == "success"
    assert merge_state_frames(bet_existing, bet_incoming, SPECS["dash_bets"]).iloc[0]["outcome"] == "win"


def test_same_quality_prefers_fresher_copy_not_incoming_order():
    existing = pd.DataFrame([{"match_uid": "m1", "features_complete": "True",
                              "model_p1_prob": "0.61", "logged_at": "2026-07-12T00:00:00Z"}])
    stale_incoming = pd.DataFrame([{"match_uid": "m1", "features_complete": "True",
                                    "model_p1_prob": "0.52", "logged_at": "2026-07-10T00:00:00Z"}])
    merged = merge_state_frames(existing, stale_incoming, SPECS["dash_predictions"])
    assert merged.iloc[0]["model_p1_prob"] == "0.61"


def test_same_quality_freshness_parses_mixed_timestamp_formats():
    existing = pd.DataFrame([{
        "match_uid": "m1", "features_complete": "True",
        "model_p1_prob": "0.61", "logged_at": "2026-07-12T12:00:00+00:00",
    }])
    stale_incoming = pd.DataFrame([{
        "match_uid": "m1", "features_complete": "True",
        "model_p1_prob": "0.52", "logged_at": "2026-07-11 12:00:00",
    }])
    merged = merge_state_frames(existing, stale_incoming, SPECS["dash_predictions"])
    assert merged.iloc[0]["model_p1_prob"] == "0.61"


def test_empty_prediction_generation_preserves_schema():
    empty = pd.DataFrame(columns=["match_uid", "model_p1_prob"])
    merged = merge_state_frames(empty, empty, SPECS["dash_predictions"])
    assert merged.empty
    assert list(merged.columns) == ["match_uid", "model_p1_prob"]
