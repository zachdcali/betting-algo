from pathlib import Path
import sys

import pandas as pd
import pytest

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


def test_bet_attribution_enrichment_outranks_newer_blank_settled_copy():
    durable_exact = pd.DataFrame([{
        "bet_id": "b1", "status": "settled", "outcome": "win",
        "settled_timestamp": "2026-07-14T10:00:00Z",
        "settlement_quality": "authoritative_result_exact_match_uid",
        "attribution_quality": "exact_match_uid", "metric_eligible": "true",
        "result_evidence_kind": "auto_settle_atp_results",
        "result_evidence_sha256": "a" * 64,
    }])
    stale_local_blank = pd.DataFrame([{
        "bet_id": "b1", "status": "settled", "outcome": "win",
        "settled_timestamp": "2026-07-14T10:00:00Z",
        "attribution_quality": "", "metric_eligible": "",
        "result_evidence_kind": "", "result_evidence_sha256": "",
    }])

    merged = merge_state_frames(
        durable_exact, stale_local_blank, SPECS["dash_bets"]
    ).iloc[0]

    assert merged["metric_eligible"] == "true"
    assert merged["attribution_quality"] == "exact_match_uid"
    assert merged["result_evidence_sha256"] == "a" * 64


def test_bet_repairable_unknown_can_upgrade_to_exact():
    common = {
        "bet_id": "b1",
        "status": "settled",
        "outcome": "win",
        "stake": "10",
        "odds_decimal": "2",
        "actual_profit": "10",
        "bankroll_after": "1010",
        "settled_timestamp": "2026-07-14T10:00:00Z",
        "settlement_quality": "authoritative_result_exact_match_uid",
        "result_evidence_kind": "auto_settle_atp_results",
        "result_evidence_sha256": "a" * 64,
    }
    unknown = pd.DataFrame([{
        **common,
        "attribution_quality": "exact_match_uid_unverified_feature_snapshot",
        "metric_eligible": "",
    }])
    exact = pd.DataFrame([{
        **common,
        "attribution_quality": "exact_match_uid",
        "metric_eligible": "true",
    }])

    merged = merge_state_frames(unknown, exact, SPECS["dash_bets"]).iloc[0]
    assert merged["metric_eligible"] == "true"
    assert merged["attribution_quality"] == "exact_match_uid"
    assert merged["result_evidence_sha256"] == "a" * 64


def test_bet_merge_rejects_conflicting_explicit_metric_eligibility():
    common = {
        "bet_id": "b1", "status": "settled", "outcome": "win",
        "settlement_quality": "authoritative_result_exact_match_uid",
        "attribution_quality": "exact_match_uid",
        "result_evidence_kind": "auto_settle_atp_results",
        "result_evidence_sha256": "a" * 64,
    }
    exact = pd.DataFrame([{
        **common, "metric_eligible": "true",
    }])
    accounting_only = pd.DataFrame([{
        **common, "metric_eligible": "false",
    }])

    with pytest.raises(
        RuntimeError, match="conflicting immutable bet metric_eligible"
    ):
        merge_state_frames(exact, accounting_only, SPECS["dash_bets"])


def test_bet_merge_accepts_csv_postgres_numeric_round_trip_noise():
    remote = pd.DataFrame([{
        "bet_id": "bet_20260715_153101_2",
        "status": "pending",
        "outcome": "",
        "match": "Tyler Zink vs Alexis Galarneau",
        "match_uid": "match_d69ad9449e3a285a201d",
        "feature_snapshot_id": "feat_01676af0ba615e2ade36",
        "run_id": "run_20260715T150502Z",
        "bet_on": "Tyler Zink",
        "bet_on_player1": "true",
        "stake": "18.836159128126734",
        "odds_decimal": "3.25",
    }])
    local = remote.copy()
    local.loc[0, "stake"] = "18.83615912812673"
    local.loc[0, "odds_decimal"] = "3.2500000000000004"

    merged = merge_state_frames(remote, local, SPECS["dash_bets"])

    assert len(merged) == 1
    assert merged.iloc[0]["bet_id"] == "bet_20260715_153101_2"


def test_bet_merge_rejects_numeric_change_above_transport_tolerance():
    existing = pd.DataFrame([{
        "bet_id": "b1", "status": "pending", "outcome": "",
        "stake": "18.83615912812673", "odds_decimal": "3.25",
    }])
    changed = existing.copy()
    changed.loc[0, "stake"] = "18.83615912822673"

    with pytest.raises(RuntimeError, match="conflicting immutable bet stake"):
        merge_state_frames(existing, changed, SPECS["dash_bets"])


def test_bet_merge_caps_tolerance_for_extreme_numeric_values():
    existing = pd.DataFrame([{
        "bet_id": "b1", "status": "pending", "outcome": "",
        "stake": "1000000000000000", "odds_decimal": "3.25",
    }])
    changed = existing.copy()
    changed.loc[0, "stake"] = "1000000000000001"

    with pytest.raises(RuntimeError, match="conflicting immutable bet stake"):
        merge_state_frames(existing, changed, SPECS["dash_bets"])


def test_terminal_bet_merge_accepts_numeric_round_trip_noise():
    common = {
        "bet_id": "b1",
        "status": "settled",
        "outcome": "loss",
        "stake": "4.975670098212187",
        "odds_decimal": "2.1",
        "settled_timestamp": "2026-07-15T16:51:11Z",
        "settlement_quality": "authoritative_result_exact_match_uid",
        "attribution_quality": "exact_match_uid",
        "metric_eligible": "true",
        "result_evidence_kind": "auto_settle_atp_results",
        "result_evidence_sha256": "a" * 64,
    }
    remote = pd.DataFrame([{
        **common,
        "actual_profit": "-4.975670098212187",
        "bankroll_after": "565.4660846227465",
    }])
    local = pd.DataFrame([{
        **common,
        "actual_profit": "-4.9756700982121875",
        "bankroll_after": "565.4660846227466",
    }])

    merged = merge_state_frames(remote, local, SPECS["dash_bets"])

    assert len(merged) == 1
    assert merged.iloc[0]["outcome"] == "loss"


def test_terminal_bet_merge_still_rejects_material_bankroll_conflict():
    common = {
        "bet_id": "b1",
        "status": "settled",
        "outcome": "loss",
        "stake": "4.975670098212187",
        "odds_decimal": "2.1",
        "actual_profit": "-4.975670098212187",
        "settled_timestamp": "2026-07-15T16:51:11Z",
        "settlement_quality": "authoritative_result_exact_match_uid",
        "attribution_quality": "exact_match_uid",
        "metric_eligible": "true",
        "result_evidence_kind": "auto_settle_atp_results",
        "result_evidence_sha256": "a" * 64,
    }
    remote = pd.DataFrame([{
        **common, "bankroll_after": "565.4660846227465",
    }])
    changed = pd.DataFrame([{
        **common, "bankroll_after": "565.4760846227465",
    }])

    with pytest.raises(
        RuntimeError, match="conflicting terminal bet bankroll_after"
    ):
        merge_state_frames(remote, changed, SPECS["dash_bets"])


def test_bet_merge_rejects_conflicting_result_evidence_and_terminal_pnl():
    common = {
        "bet_id": "b1",
        "status": "settled",
        "outcome": "win",
        "stake": "10",
        "odds_decimal": "2",
        "actual_profit": "10",
        "settlement_quality": "authoritative_result_exact_match_uid",
        "attribution_quality": "exact_match_uid",
        "metric_eligible": "true",
        "result_evidence_kind": "auto_settle_atp_results",
    }
    durable = pd.DataFrame([{
        **common,
        "settled_timestamp": "2026-07-14T10:00:00Z",
        "result_evidence_sha256": "a" * 64,
    }])
    changed_hash = pd.DataFrame([{
        **common,
        "settled_timestamp": "2026-07-15T10:00:00Z",
        "result_evidence_sha256": "b" * 64,
    }])
    with pytest.raises(
        RuntimeError, match="conflicting immutable bet result_evidence_sha256"
    ):
        merge_state_frames(durable, changed_hash, SPECS["dash_bets"])

    changed_pnl = pd.DataFrame([{
        **common,
        "outcome": "loss",
        "actual_profit": "-10",
        "settled_timestamp": "2026-07-15T10:00:00Z",
        "result_evidence_sha256": "a" * 64,
    }])
    with pytest.raises(RuntimeError, match="conflicting terminal bet outcome"):
        merge_state_frames(durable, changed_pnl, SPECS["dash_bets"])

    pending_claim = pd.DataFrame([{
        **common,
        "status": "pending",
        "outcome": "",
        "actual_profit": "",
        "result_evidence_sha256": "a" * 64,
    }])
    with pytest.raises(
        RuntimeError, match="nonterminal bet carries attribution bundle"
    ):
        merge_state_frames(pd.DataFrame(), pending_claim, SPECS["dash_bets"])


def test_bet_merge_rejects_internally_invalid_terminal_accounting():
    common = {
        "bet_id": "b1",
        "match": "Player One vs Player Two",
        "match_uid": "m1",
        "feature_snapshot_id": "f1",
        "run_id": "r1",
        "bet_on": "Player One",
        "bet_on_player1": "true",
        "stake": "10",
        "odds_decimal": "2",
        "bankroll_after": "990",
        "settled_timestamp": "2026-07-14T10:00:00Z",
        "settlement_quality": "authoritative_result_exact_match_uid",
        "attribution_quality": "exact_match_uid",
        "metric_eligible": "true",
        "result_evidence_kind": "auto_settle_atp_results",
        "result_evidence_sha256": "a" * 64,
    }
    wrong_pnl = pd.DataFrame([{
        **common,
        "status": "settled",
        "outcome": "win",
        "actual_profit": "-10",
    }])
    with pytest.raises(RuntimeError, match="invalid P&L arithmetic"):
        merge_state_frames(pd.DataFrame(), wrong_pnl, SPECS["dash_bets"])

    pending_result = pd.DataFrame([{
        **common,
        "status": "pending",
        "outcome": "win",
        "actual_profit": "10",
    }])
    with pytest.raises(RuntimeError, match="nonterminal bet carries attribution"):
        merge_state_frames(
            pd.DataFrame(), pending_result, SPECS["dash_bets"]
        )

    for status, outcome in (("void", "void"), ("cancel", "canceled")):
        invalid_refund = pd.DataFrame([{
            **common,
            "status": status,
            "outcome": outcome,
            "actual_profit": "10",
            "settlement_quality": "result_recorded_without_attribution_proof",
            "attribution_quality": "unverified",
            "metric_eligible": "false",
            "result_evidence_kind": "",
            "result_evidence_sha256": "",
        }])
        with pytest.raises(RuntimeError, match="is not refunded"):
            merge_state_frames(
                pd.DataFrame(), invalid_refund, SPECS["dash_bets"]
            )

    invalid_void_metric = pd.DataFrame([{
        **common,
        "status": "void",
        "outcome": "void",
        "actual_profit": "0",
    }])
    with pytest.raises(RuntimeError, match="cannot be metric eligible"):
        merge_state_frames(
            pd.DataFrame(), invalid_void_metric, SPECS["dash_bets"]
        )


def test_bet_merge_requires_canonical_status_and_clean_nonterminal_state():
    for status in ("", "pendng", "open"):
        invalid_status = pd.DataFrame([{
            "bet_id": f"bad-{status or 'blank'}",
            "status": status,
            "outcome": "",
        }])
        with pytest.raises(RuntimeError, match="invalid canonical bet status"):
            merge_state_frames(
                pd.DataFrame(), invalid_status, SPECS["dash_bets"]
            )

    pending_with_terminal = pd.DataFrame([{
        "bet_id": "pending-terminal",
        "status": "pending",
        "outcome": "win",
        "actual_profit": "10",
    }])
    with pytest.raises(RuntimeError, match="pending bet carries terminal state"):
        merge_state_frames(
            pd.DataFrame(), pending_with_terminal, SPECS["dash_bets"]
        )


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
