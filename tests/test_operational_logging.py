from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))


def test_orchestrator_logs_incomplete_feature_predictions(monkeypatch):
    import main

    calls = []

    def fake_log_prediction(**kwargs):
        calls.append(kwargs)
        return "created"

    def fail_skipped_log(**_kwargs):
        raise AssertionError("incomplete feature predictions should be logged, not skipped")

    monkeypatch.setattr(main, "log_prediction", fake_log_prediction)
    monkeypatch.setattr(main, "log_skipped_live_match", fail_skipped_log)

    orchestrator = main.LiveBettingOrchestrator.__new__(main.LiveBettingOrchestrator)
    orchestrator.run_id = "run_test"
    orchestrator.run_started_at = "2026-05-13T04:05:26+00:00"

    predictions_df = pd.DataFrame(
        [
            {
                "player1_normalized": "Player One",
                "player2_normalized": "Player Two",
                "player1_raw": "Player One",
                "player2_raw": "Player Two",
                "player1_win_prob": 0.63,
                "prediction_status": "success",
                "_has_defaulted_features": True,
                "meta_defaulted_features": "round_code=None",
                "meta_match_date": "2026-05-13",
                "meta_surface_input": "Clay",
                "meta_level_input": "A",
                "meta_round_input": "",
                "meta_resolver_source": "level_hint",
                "run_id": "run_test",
                "match_uid": "match_test",
                "feature_snapshot_id": "feature_test",
                "match_time": "5/13/26 6:00 AM",
                "timestamp": "2026-05-13T03:00:00+00:00",
                "Player1_Rank": 7,
                "Player2_Rank": 18,
                "xgb_p1_prob": 0.58,
                "xgb_p2_prob": 0.42,
                "rf_p1_prob": 0.55,
                "rf_p2_prob": 0.45,
            }
        ]
    )
    odds_df = pd.DataFrame(
        [
            {
                "player1_normalized": "Player One",
                "player2_normalized": "Player Two",
                "player1_implied_prob": 0.60,
                "player2_implied_prob": 0.50,
                "player1_odds_american": -150,
                "player2_odds_american": 120,
                "player1_odds_decimal": 1.67,
                "player2_odds_decimal": 2.20,
                "spread_handicap": None,
                "spread_odds_p1": None,
                "spread_odds_p2": None,
                "total_games": None,
                "total_odds_over": None,
                "total_odds_under": None,
                "tourney_name": "Rome",
                "match_time": "5/13/26 6:00 AM",
                "scrape_time_utc": "2026-05-13T03:00:00+00:00",
            }
        ]
    )

    stats = orchestrator._log_all_predictions(predictions_df, odds_df, pd.DataFrame())

    assert stats == {
        "attempts": 1,
        "created": 1,
        "updated": 0,
        "skipped_incomplete": 0,
        "identity_aliases": 0,
        "identity_conflicts": 0,
        "identity_conflict_match_uids": [],
    }
    assert len(calls) == 1
    assert calls[0]["features_complete"] is False
    assert calls[0]["defaulted_features"] == "round_code=None"
    assert calls[0]["round_code"] is None
    assert calls[0]["match_start_at_utc"] == "2026-05-13T10:00:00+00:00"


def test_missing_market_join_is_logged_as_missing_not_even_money(monkeypatch):
    import main

    calls = []
    monkeypatch.setattr(main, "log_prediction", lambda **kwargs: calls.append(kwargs) or "created")
    orchestrator = main.LiveBettingOrchestrator.__new__(main.LiveBettingOrchestrator)
    orchestrator.run_id = "run_missing_market"
    orchestrator.run_started_at = "2026-05-13T04:05:26+00:00"
    predictions = pd.DataFrame([{
        "player1_normalized": "Player One", "player2_normalized": "Player Two",
        "player1_raw": "Player One", "player2_raw": "Player Two",
        "player1_win_prob": 0.63, "prediction_status": "success",
        "_has_defaulted_features": False, "meta_match_date": "2026-05-13",
        "meta_surface_input": "Clay", "meta_level_input": "A",
        "meta_round_input": "R32", "run_id": "run_missing_market",
        "match_uid": "match_missing_market", "feature_snapshot_id": "feat_missing_market",
    }])
    unmatched_odds = pd.DataFrame([{
        "player1_normalized": "Someone Else", "player2_normalized": "Another Player",
    }])

    orchestrator._log_all_predictions(predictions, unmatched_odds, pd.DataFrame())

    assert calls[0]["market_p1_prob"] is None
    assert calls[0]["market_p2_prob"] is None


def test_prediction_identity_contract_is_fatal_even_without_durable_mode(monkeypatch):
    import main

    monkeypatch.delenv("REQUIRE_DURABLE_STATE", raising=False)

    def reject_identity(**_kwargs):
        raise main.LiveMatchIdentityError("snapshot belongs to another match")

    monkeypatch.setattr(main, "log_prediction", reject_identity)
    orchestrator = main.LiveBettingOrchestrator.__new__(main.LiveBettingOrchestrator)
    orchestrator.run_id = "run_identity_failure"
    orchestrator.run_started_at = "2026-07-14T18:00:00+00:00"
    predictions = pd.DataFrame([{
        "player1_normalized": "Player One",
        "player2_normalized": "Player Two",
        "player1_raw": "Player One",
        "player2_raw": "Player Two",
        "player1_win_prob": 0.63,
        "prediction_status": "success",
        "_has_defaulted_features": False,
        "meta_match_date": "2026-07-14",
        "meta_surface_input": "Hard",
        "meta_level_input": "C",
        "meta_round_input": "R32",
        "run_id": "run_identity_failure",
        "match_uid": "match_wrong",
        "feature_snapshot_id": "feat_wrong",
    }])
    unmatched_odds = pd.DataFrame([{
        "player1_normalized": "Someone Else",
        "player2_normalized": "Another Player",
    }])

    with pytest.raises(RuntimeError, match="prediction identity contract failed"):
        orchestrator._log_all_predictions(
            predictions, unmatched_odds, pd.DataFrame(),
        )


def test_unexpected_prediction_logging_error_is_fatal_before_decisions(monkeypatch):
    import main

    monkeypatch.delenv("REQUIRE_DURABLE_STATE", raising=False)

    def unavailable_log(**_kwargs):
        raise OSError("prediction log unavailable")

    monkeypatch.setattr(main, "log_prediction", unavailable_log)
    orchestrator = main.LiveBettingOrchestrator.__new__(main.LiveBettingOrchestrator)
    orchestrator.run_id = "run_logging_failure"
    orchestrator.run_started_at = "2026-07-14T18:00:00+00:00"
    predictions = pd.DataFrame([{
        "player1_normalized": "Player One",
        "player2_normalized": "Player Two",
        "player1_raw": "Player One",
        "player2_raw": "Player Two",
        "player1_win_prob": 0.63,
        "prediction_status": "success",
        "_has_defaulted_features": False,
        "meta_match_date": "2026-07-14",
        "meta_surface_input": "Hard",
        "meta_level_input": "C",
        "meta_round_input": "R32",
        "run_id": "run_logging_failure",
        "match_uid": "match_unavailable",
        "feature_snapshot_id": "feat_unavailable",
    }])
    unmatched_odds = pd.DataFrame([{
        "player1_normalized": "Someone Else",
        "player2_normalized": "Another Player",
    }])

    with pytest.raises(
        RuntimeError, match="prediction identity preflight/logging failed"
    ):
        orchestrator._log_all_predictions(
            predictions, unmatched_odds, pd.DataFrame(),
        )


def test_logging_gate_returns_both_sides_of_same_run_identity_conflict(monkeypatch):
    import main

    calls = 0

    def classify_prediction(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "created"
        kwargs["identity_conflict_uids_out"].update(
            {"match_opening", "match_shifted"}
        )
        return "identity_conflict"

    monkeypatch.setattr(main, "log_prediction", classify_prediction)
    monkeypatch.setattr(main, "log_skipped_live_match", lambda **_kwargs: None)
    orchestrator = main.LiveBettingOrchestrator.__new__(main.LiveBettingOrchestrator)
    orchestrator.run_id = "run_same_slate"
    orchestrator.run_started_at = "2026-07-14T18:00:00+00:00"
    base = {
        "player1_normalized": "Player One",
        "player2_normalized": "Player Two",
        "player1_raw": "Player One",
        "player2_raw": "Player Two",
        "player1_win_prob": 0.63,
        "prediction_status": "success",
        "_has_defaulted_features": False,
        "meta_match_date": "2026-07-14",
        "meta_surface_input": "Hard",
        "meta_level_input": "C",
        "meta_round_input": "R32",
        "run_id": "run_same_slate",
    }
    predictions = pd.DataFrame([
        {
            **base,
            "match_uid": "match_opening",
            "feature_snapshot_id": "feat_opening",
        },
        {
            **base,
            "meta_round_input": "R16",
            "match_uid": "match_shifted",
            "feature_snapshot_id": "feat_shifted",
        },
    ])
    unmatched_odds = pd.DataFrame([{
        "player1_normalized": "Someone Else",
        "player2_normalized": "Another Player",
    }])

    stats = orchestrator._log_all_predictions(
        predictions, unmatched_odds, pd.DataFrame()
    )

    assert stats["identity_conflicts"] == 1
    assert set(stats["identity_conflict_match_uids"]) == {
        "match_opening", "match_shifted",
    }


def test_prediction_terminal_status_rejects_all_failed_rows():
    import main

    status, successes, errors = main.prediction_terminal_status(pd.DataFrame([
        {"prediction_status": "error"},
        {"prediction_status": "skipped_missing_data"},
    ]))
    assert (status, successes, errors) == ("no_predictions", 0, 2)

    status, successes, errors = main.prediction_terminal_status(pd.DataFrame([
        {"prediction_status": "success", "player1_win_prob": 0.61},
        {"prediction_status": "error", "player1_win_prob": None},
    ]))
    assert (status, successes, errors) == ("partial", 1, 1)

    status, successes, errors = main.prediction_terminal_status(pd.DataFrame([
        {"prediction_status": "success", "player1_win_prob": float("nan")},
        {"prediction_status": "success", "player1_win_prob": 1.2},
    ]))
    assert (status, successes, errors) == ("no_predictions", 0, 2)


def test_failed_or_nan_predictions_never_reach_snapshot_logger(monkeypatch):
    import main

    calls = []
    monkeypatch.setattr(
        main, "log_prediction", lambda **kwargs: calls.append(kwargs) or "created",
    )
    orchestrator = main.LiveBettingOrchestrator.__new__(
        main.LiveBettingOrchestrator,
    )
    orchestrator.run_id = "run_failed_output"
    orchestrator.run_started_at = "2026-07-14T18:00:00+00:00"
    rows = pd.DataFrame([
        {
            "prediction_status": "skipped_missing_data",
            "player1_win_prob": float("nan"),
            "player1_raw": "Skipped One",
            "player2_raw": "Skipped Two",
        },
        {
            "prediction_status": "success",
            "player1_win_prob": float("nan"),
            "player1_raw": "Invalid One",
            "player2_raw": "Invalid Two",
        },
    ])

    stats = orchestrator._log_all_predictions(rows, pd.DataFrame(), pd.DataFrame())

    assert calls == []
    assert stats["attempts"] == 0
    assert stats["skipped_incomplete"] == 1


def test_generate_predictions_reclassifies_nan_success_and_audits_it(monkeypatch):
    import main

    audits = []
    monkeypatch.setattr(
        main,
        "log_skipped_live_match",
        lambda **kwargs: audits.append(kwargs),
    )
    orchestrator = main.LiveBettingOrchestrator.__new__(
        main.LiveBettingOrchestrator,
    )
    orchestrator.run_id = "run_invalid_prediction"
    orchestrator.run_started_at = "2026-07-14T18:00:00+00:00"
    orchestrator.predictor = SimpleNamespace(
        predict_slate=lambda _features: pd.DataFrame([{
            "prediction_status": "success",
            "player1_win_prob": float("nan"),
            "player2_win_prob": float("nan"),
            "player1_raw": "Player One",
            "player2_raw": "Player Two",
            "run_id": "run_invalid_prediction",
            "match_uid": "match_invalid_prediction",
            "feature_snapshot_id": "feat_invalid_prediction",
            "meta_match_date": "2026-07-14",
            "meta_surface_input": "Hard",
            "meta_level_input": "C",
            "meta_round_input": "R32",
        }]),
    )
    orchestrator.xgb_predictor = SimpleNamespace(
        is_loaded=False, load_model=lambda: False,
    )
    orchestrator.rf_predictor = SimpleNamespace(
        is_loaded=False, load_model=lambda: False,
    )

    result = orchestrator.generate_predictions(pd.DataFrame([{"feature": 1.0}]))

    assert result.loc[0, "prediction_status"] == "failed"
    assert result.loc[0, "error"] == "invalid_primary_probability"
    assert len(audits) == 1
    assert audits[0]["stage"] == "prediction_generation"
    assert audits[0]["skip_reason_code"] == "prediction_output_invalid"


def test_only_ok_feature_rows_receive_exact_snapshot_ids():
    import main

    common = {
        "match_uid": "match_exact",
        "run_id": "run_exact",
        "p1": "Player One",
        "p2": "Player Two",
    }

    assert main.exact_feature_snapshot_id(status="skip", **common) == ""
    assert main.exact_feature_snapshot_id(status="error", **common) == ""
    assert main.exact_feature_snapshot_id(status="ok", **common).startswith(
        "feat_"
    )


def test_missing_match_start_is_an_inference_guard():
    import main

    orchestrator = main.LiveBettingOrchestrator.__new__(main.LiveBettingOrchestrator)
    assert orchestrator.get_inference_guard_reason("") == (
        None, "match_start_time_missing",
    )
    assert orchestrator.get_inference_guard_reason("Unknown") == (
        None, "match_start_time_missing",
    )


def test_auto_settle_reports_ta_unfinished_before_opponent_not_found(monkeypatch):
    import auto_settle

    def fake_get_player_matches(*_args, **_kwargs):
        return pd.DataFrame()

    def fake_get_upcoming_match(*_args, **_kwargs):
        return {
            "round": "R32",
            "surface": "Clay",
            "event": "Rome",
            "date": "20260513",
        }

    monkeypatch.setattr(auto_settle.SCRAPER, "get_player_matches", fake_get_player_matches)
    monkeypatch.setattr(auto_settle.SCRAPER, "get_upcoming_match", fake_get_upcoming_match)

    result = auto_settle.try_settle_from_ta(
        "Player One",
        "Player Two",
        "2026-05-13",
        SimpleNamespace(player_slug_map={}),
        session_cache={},
        dry_run=True,
    )

    assert result["status"] == "ta_match_unfinished"
    assert result["ta_round_found"] == "R32"
    assert result["ta_event_found"] == "Rome"
    assert result["ta_match_date_found"] == "2026-05-13"


def test_auto_settle_prefers_context_match_over_closer_wrong_event():
    import auto_settle

    candidates = pd.DataFrame(
        [
            {
                "date": "2026-05-13",
                "event": "Lyon",
                "surface": "Clay",
                "round": "R32",
                "opp_name": "Player Two",
                "result": "W",
            },
            {
                "date": "2026-05-10",
                "event": "Rome",
                "surface": "Clay",
                "round": "R32",
                "opp_name": "Player Two",
                "result": "L",
            },
        ]
    )
    context = auto_settle._build_settlement_context(
        tournament="ATP - Rome",
        round_code="R32",
        surface="Clay",
    )

    row, status, evidence = auto_settle._select_best_settlement_candidate(
        candidates,
        pd.Timestamp("2026-05-13"),
        context,
    )

    assert status == "matched"
    assert row["event"] == "Rome"
    assert evidence["score"] >= auto_settle.MIN_SETTLEMENT_SCORE


def test_auto_settle_leaves_repeated_same_opponent_ambiguous():
    import auto_settle

    candidates = pd.DataFrame(
        [
            {
                "date": "2026-05-13",
                "event": "Rome",
                "surface": "Clay",
                "round": "R32",
                "opp_name": "Player Two",
                "result": "W",
            },
            {
                "date": "2026-05-13",
                "event": "Rome",
                "surface": "Clay",
                "round": "R32",
                "opp_name": "Player Two",
                "result": "L",
            },
        ]
    )
    context = auto_settle._build_settlement_context(
        tournament="Rome",
        round_code="R32",
        surface="Clay",
    )

    row, status, evidence = auto_settle._select_best_settlement_candidate(
        candidates,
        pd.Timestamp("2026-05-13"),
        context,
    )

    assert row is None
    assert status == "ambiguous_match"
    assert evidence["candidates"] == 2


def test_auto_settle_skips_rows_before_settlement_grace_period():
    import auto_settle

    assert auto_settle.DEFAULT_MIN_SETTLEMENT_AGE_HOURS == 18.0
    assert auto_settle.DEFAULT_MAX_CANDIDATES == 75
    assert auto_settle.DEFAULT_RETRY_BACKOFF_HOURS == 18.0

    row = pd.Series(
        {
            "match_start_time": "5/13/26 6:00 AM",
            "match_date": "2026-05-13",
        }
    )

    eligible, reason = auto_settle._is_old_enough_to_settle(
        row,
        min_age_hours=18,
        now=pd.Timestamp("2026-05-13 12:00:00").to_pydatetime(),
    )

    assert eligible is False
    assert "min_age_hours=18.0" in reason


def test_auto_settle_prefers_latest_exact_utc_over_stale_display_time():
    import auto_settle

    row = pd.Series(
        {
            "match_start_time": "7/14/26 8:00 AM",
            "match_start_at_utc": "2026-07-14T12:00:00Z",
            "latest_match_start_at_utc": "2026-07-14T08:00:00Z",
            "match_date": "2026-07-14",
        }
    )

    eligible, reason = auto_settle._is_old_enough_to_settle(
        row,
        min_age_hours=18,
        now=pd.Timestamp("2026-07-15T02:30:00Z").to_pydatetime(),
    )

    assert eligible is True
    assert reason == ""


def test_auto_settle_legacy_clock_accepts_an_aware_utc_reference():
    import auto_settle

    row = pd.Series(
        {
            "match_start_time": "5/13/26 6:00 AM",
            "match_date": "2026-05-13",
        }
    )

    eligible, reason = auto_settle._is_old_enough_to_settle(
        row,
        min_age_hours=18,
        now=pd.Timestamp("2026-05-13T16:00:00Z").to_pydatetime(),
    )

    assert eligible is False
    assert "match_start_age_hours=6.0" in reason


def test_auto_settle_prioritizes_only_direct_or_canonical_alias_pending_bets():
    import auto_settle

    pending = pd.DataFrame(
        [
            {"match_uid": "prediction-only", "identity_status": "canonical"},
            {"match_uid": "direct-bet", "identity_status": "canonical"},
            {
                "match_uid": "safe-new-uid",
                "identity_status": "canonical_alias",
                "identity_related_match_uid": "old-bet-uid",
            },
            {
                "match_uid": "unsafe-new-uid",
                "identity_status": "conflict",
                "identity_related_match_uid": "old-bet-uid",
            },
        ]
    )
    pending_bets = pd.DataFrame(
        {"match_uid": ["direct-bet", "old-bet-uid"]}
    )

    prioritized = auto_settle._prioritize_tracked_pending_matches(
        pending,
        pending_bets,
    )

    assert prioritized["_tracked_bet_priority"].tolist() == [False, True, True, False]


def test_auto_settle_tracked_bet_priority_is_applied_before_candidate_cap():
    import auto_settle

    pending = pd.DataFrame(
        [
            {
                "match_uid": "newer-prediction-only",
                "match_date": "2026-07-15",
                "match_start_at_utc": "2026-07-15T08:00:00Z",
                "p1": "New A",
                "p2": "New B",
            },
            {
                "match_uid": "tracked-bet",
                "match_date": "2026-07-14",
                "match_start_at_utc": "2026-07-14T08:00:00Z",
                "p1": "Tracked A",
                "p2": "Tracked B",
            },
        ]
    )
    pending_bets = pd.DataFrame({"match_uid": ["tracked-bet"]})

    ordered, tracked_count = auto_settle._order_settlement_candidates(
        pending,
        pending_bets,
        max_candidates=1,
    )

    assert ordered["match_uid"].tolist() == ["tracked-bet"]
    assert tracked_count == 1


def test_auto_settle_orders_oldest_tracked_exposure_before_newer_tracked_rows():
    import auto_settle

    pending = pd.DataFrame(
        [
            {
                "match_uid": "tracked-new",
                "match_date": "2026-07-15",
                "match_start_at_utc": "2026-07-15T08:00:00Z",
            },
            {
                "match_uid": "tracked-old",
                "match_date": "2026-07-13",
                "latest_match_date": "2026-07-14",
                "match_start_at_utc": "2026-07-14T08:00:00Z",
            },
            {
                "match_uid": "prediction-only",
                "match_date": "2026-07-16",
                "match_start_at_utc": "2026-07-16T08:00:00Z",
            },
        ]
    )
    pending_bets = pd.DataFrame(
        {"match_uid": ["tracked-new", "tracked-old"]}
    )

    ordered, tracked_count = auto_settle._order_settlement_candidates(
        pending,
        pending_bets,
        max_candidates=2,
    )

    assert ordered["match_uid"].tolist() == ["tracked-old", "tracked-new"]
    assert tracked_count == 2


def test_auto_settle_recent_attempt_backoff_ignores_dry_runs(tmp_path):
    import auto_settle

    audit_path = tmp_path / "settlement_audit.csv"
    pd.DataFrame(
        [
            {
                "logged_at": "2026-05-13T12:00:00+00:00",
                "dry_run": False,
                "row_index": 12,
            },
            {
                "logged_at": "2026-05-13T12:00:00+00:00",
                "dry_run": True,
                "row_index": 13,
            },
            {
                "logged_at": "2026-05-12T12:00:00+00:00",
                "dry_run": False,
                "row_index": 14,
            },
        ]
    ).to_csv(audit_path, index=False)

    recent = auto_settle._recently_attempted_row_indexes(
        audit_path=audit_path,
        backoff_hours=18,
        now=pd.Timestamp("2026-05-13T18:00:00+00:00"),
    )

    assert recent == {12}


def test_auto_settle_recent_attempt_backoff_uses_stable_identity_keys(tmp_path):
    import auto_settle

    audit_path = tmp_path / "settlement_audit.csv"
    pd.DataFrame(
        [
            {
                "logged_at": "2026-05-13T12:00:00+00:00",
                "dry_run": False,
                "row_index": 12,
                "match_uid": "match_shared",
                "prediction_uid": "prediction_shared",
            },
            {
                "logged_at": "2026-05-13T12:00:00+00:00",
                "dry_run": True,
                "row_index": 13,
                "match_uid": "match_dry_run",
                "prediction_uid": "prediction_dry_run",
            },
        ]
    ).to_csv(audit_path, index=False)

    indexes, match_uids, prediction_uids = (
        auto_settle._recently_attempted_identity_keys(
            audit_path=audit_path,
            backoff_hours=18,
            now=pd.Timestamp("2026-05-13T18:00:00+00:00"),
        )
    )

    assert indexes == {12}
    assert match_uids == {"match_shared"}
    assert prediction_uids == {"prediction_shared"}

    pending = pd.DataFrame(
        [
            {
                "match_uid": "match_shared",
                "prediction_uid": "prediction_shared",
            },
            {
                "match_uid": "match_shared",
                "prediction_uid": "prediction_duplicate_row",
            },
            {
                "match_uid": "match_fresh",
                "prediction_uid": "prediction_fresh",
            },
        ],
        index=[12, 99, 100],
    )
    mask = auto_settle._recently_attempted_mask(
        pending,
        row_indexes=indexes,
        match_uids=match_uids,
        prediction_uids=prediction_uids,
    )

    assert mask.to_dict() == {12: True, 99: True, 100: False}


def test_bet_tracker_skips_duplicate_pending_bets():
    from utils.bet_tracker import BetTracker

    with tempfile.TemporaryDirectory() as tmp:
        tracker = BetTracker(tmp)
        session_id = tracker.start_session(1000.0, 0.18, "duplicate test")
        bet_slips = pd.DataFrame(
            [
                {
                    "event": "Rome",
                    "match": "Player One vs Player Two",
                    "match_uid": "match_test",
                    "feature_snapshot_id": "feature_test",
                    "run_id": "run_test",
                    "bet_on": "Player One",
                    "bet_on_player1": True,
                    "odds_decimal": 1.90,
                    "stake": 10.0,
                    "stake_fraction": 0.01,
                    "model_prob": 0.62,
                    "market_prob": 0.53,
                    "edge": 0.09,
                    "kelly_fraction": 0.03,
                    "potential_profit": 9.0,
                    "potential_loss": 10.0,
                    "bankroll": 1000.0,
                    "model_version": "v-test",
                    "match_date": "2026-05-13",
                    "match_start_time": "5/13/26 6:00 AM",
                }
            ]
        )

        first_count = tracker.log_bets(bet_slips, session_id, 1000.0)
        second_count = tracker.log_bets(bet_slips, session_id, 1000.0)
        all_bets = pd.read_csv(tracker.bets_file)

        assert first_count == 1
        assert second_count == 0
        assert len(all_bets) == 1


def test_bet_tracker_discards_zero_exposure_session(tmp_path):
    from utils.bet_tracker import BetTracker

    tracker = BetTracker(str(tmp_path))
    session_id = tracker.start_session(1000.0, 0.18, "empty test")
    assert tracker.discard_empty_session(session_id)
    assert pd.read_csv(tracker.session_file).empty
    assert pd.read_csv(tracker.bankroll_file).empty


def test_bet_tracker_settles_string_false_as_player_two(tmp_path):
    """CSV round-tripping may make ``False`` a string; it must not become
    truthy and reverse the winner when a P2 bet settles."""
    from utils.bet_tracker import BetTracker

    tracker = BetTracker(str(tmp_path))
    session_id = tracker.start_session(1000.0, 0.18, "settlement bool test")
    slip = pd.DataFrame([{
        "event": "Rome", "match": "Player One vs Player Two",
        "match_uid": "match_bool", "feature_snapshot_id": "feature_bool",
        "run_id": "run_bool", "bet_on": "Player Two",
        "bet_on_player1": False, "odds_decimal": 2.0, "stake": 10.0,
        "stake_fraction": 0.01, "model_prob": 0.60, "market_prob": 0.50,
        "edge": 0.10, "kelly_fraction": 0.03, "potential_profit": 10.0,
        "potential_loss": 10.0, "bankroll": 1000.0,
        "model_version": "v-test", "match_date": "2026-05-13",
        "match_start_time": "5/13/26 6:00 AM",
    }])
    assert tracker.log_bets(slip, session_id, 1000.0) == 1

    # Force the exact object/string representation that triggered the bug.
    bets = pd.read_csv(tracker.bets_file, dtype={"bet_on_player1": object})
    bets.loc[0, "bet_on_player1"] = "False"
    bets.to_csv(tracker.bets_file, index=False)

    assert tracker.settle_pending_bets_for_match(
        match_uid="match_bool", p1="Player One", p2="Player Two",
        actual_winner=2,
    ) == 1
    settled = pd.read_csv(tracker.bets_file).iloc[0]
    assert settled["status"] == "settled"
    assert settled["outcome"] == "win"
    assert settled["actual_profit"] == 10.0


def test_bet_settlement_never_fuzzy_matches_a_different_nonblank_uid(tmp_path):
    from utils.bet_tracker import BETS_COLUMNS, BetTracker
    from settlement_attribution import (
        FeatureAttributionEvidence,
        build_auto_result_evidence,
    )

    tracker = BetTracker(str(tmp_path))
    session_id = tracker.start_session(1000.0, 0.18, "identity settlement")
    rows = []
    for bet_id, match_uid in (
        ("exact", "match_canonical"),
        ("exact_wrong_run", "match_canonical"),
        ("explicit_alias", "match_old_alias"),
        ("different_uid", "match_other_round"),
        ("legacy_blank", ""),
    ):
        row = {column: "" for column in BETS_COLUMNS}
        row.update({
            "bet_id": bet_id,
            "session_id": session_id,
            "match": "Player One vs Player Two",
            "match_uid": match_uid,
            "bet_on": "Player One",
            "bet_on_player1": True,
            "odds_decimal": 2.0,
            "stake": 10.0,
            "status": "pending",
        })
        if bet_id in {"exact", "exact_wrong_run"}:
            row["feature_snapshot_id"] = "feat_exact"
            row["run_id"] = (
                "run_exact" if bet_id == "exact" else "run_other"
            )
        rows.append(row)
    pd.DataFrame(rows, columns=BETS_COLUMNS).to_csv(tracker.bets_file, index=False)

    feature_evidence = {
        "feat_exact": FeatureAttributionEvidence(
            feature_snapshot_id="feat_exact",
            run_id="run_exact",
            match_uid="match_canonical",
            p1="Player One",
            p2="Player Two",
            feature_schema_sha256="a" * 64,
            feature_vector_sha256="b" * 64,
        )
    }
    bound_result_evidence = build_auto_result_evidence(
        source_evidence={"source": "tennis_abstract", "candidate": "exact"},
        match_uid="match_canonical",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
        score="6-4 6-4",
    )
    result_evidence_kind, result_evidence_sha256 = bound_result_evidence

    assert tracker.settle_pending_bets_for_match(
        match_uid="match_canonical",
        alias_match_uids=["match_old_alias"],
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
        exact_feature_evidence=feature_evidence,
        result_evidence_kind=result_evidence_kind,
        result_evidence_sha256=result_evidence_sha256,
        bound_result_evidence=bound_result_evidence,
    ) == 3
    first_pass = pd.read_csv(tracker.bets_file).set_index("bet_id")
    assert first_pass.loc["exact", "status"] == "settled"
    assert first_pass.loc["explicit_alias", "status"] == "settled"
    assert first_pass.loc["different_uid", "status"] == "pending"
    assert first_pass.loc["legacy_blank", "status"] == "pending"
    assert first_pass.loc["exact", "settlement_quality"] == (
        "authoritative_result_exact_match_uid"
    )
    assert first_pass.loc["exact", "attribution_quality"] == "exact_match_uid"
    assert str(first_pass.loc["exact", "metric_eligible"]).lower() == "true"
    assert first_pass.loc["exact", "result_evidence_kind"] == (
        "auto_settle_tennis_abstract"
    )
    assert first_pass.loc["exact", "result_evidence_sha256"] == (
        result_evidence_sha256
    )
    assert first_pass.loc["explicit_alias", "attribution_quality"] == (
        "unattributed_rotated_match_uid"
    )
    assert str(
        first_pass.loc["explicit_alias", "metric_eligible"]
    ).lower() == "false"
    assert first_pass.loc["exact_wrong_run", "attribution_quality"] == (
        "exact_match_uid_unverified_feature_snapshot"
    )
    assert pd.isna(first_pass.loc["exact_wrong_run", "metric_eligible"])

    # A result carrying a UID can never fall through to pair-only matching,
    # even for a legacy blank-UID bet.
    assert tracker.settle_pending_bets_for_match(
        match_uid="match_canonical",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    ) == 0
    second_pass = pd.read_csv(tracker.bets_file).set_index("bet_id")
    assert second_pass.loc["legacy_blank", "status"] == "pending"
    assert second_pass.loc["different_uid", "status"] == "pending"

    # Pair-only compatibility is blank-result-UID to blank-bet-UID only. The
    # modern nonblank bet remains pending despite the identical player pair.
    assert tracker.settle_pending_bets_for_match(
        match_uid="",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    ) == 1
    second_pass = pd.read_csv(tracker.bets_file).set_index("bet_id")
    assert second_pass.loc["legacy_blank", "status"] == "settled"
    assert second_pass.loc["legacy_blank", "attribution_quality"] == "uid_unlinked"
    assert str(
        second_pass.loc["legacy_blank", "metric_eligible"]
    ).lower() == "false"
    assert second_pass.loc["different_uid", "status"] == "pending"


def test_auto_settlement_uses_winner_identity_across_orientation_changes(tmp_path):
    from utils.bet_tracker import BETS_COLUMNS, BetTracker

    tracker = BetTracker(str(tmp_path))
    session_id = tracker.start_session(1000.0, 0.18, "orientation settlement")
    row = {column: "" for column in BETS_COLUMNS}
    row.update({
        "bet_id": "flipped",
        "session_id": session_id,
        "match": "Player Two vs Player One",
        "match_uid": "match_orientation",
        "bet_on": "Player Two",
        "bet_on_player1": True,
        "odds_decimal": 2.0,
        "stake": 10.0,
        "status": "pending",
    })
    wrong_pair = dict(row)
    wrong_pair.update({
        "bet_id": "one_player_overlap",
        "match": "Player One vs Different Opponent",
        "bet_on": "Player One",
    })
    pd.DataFrame([row, wrong_pair], columns=BETS_COLUMNS).to_csv(
        tracker.bets_file, index=False
    )

    # The settlement result is oriented Player One vs Player Two. Numeric P1
    # would incorrectly mark this flipped bet as a win; winner identity says loss.
    assert tracker.settle_pending_bets_for_match(
        match_uid="match_orientation",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    ) == 1
    rows = pd.read_csv(tracker.bets_file).set_index("bet_id")
    settled = rows.loc["flipped"]
    assert settled["outcome"] == "loss"
    assert settled["actual_profit"] == -10.0
    assert settled["attribution_quality"] == (
        "exact_match_uid_unverified_feature_snapshot"
    )
    assert pd.isna(settled["metric_eligible"])
    assert rows.loc["one_player_overlap", "status"] == "pending"


def test_auto_settlement_without_result_participants_stays_pending(tmp_path):
    from utils.bet_tracker import BETS_COLUMNS, BetTracker

    tracker = BetTracker(str(tmp_path))
    row = {column: "" for column in BETS_COLUMNS}
    row.update({
        "bet_id": "missing_identity",
        "match": "Player Two vs Player One",
        "match_uid": "match_missing_identity",
        "bet_on": "Player Two",
        "bet_on_player1": True,
        "odds_decimal": 2.0,
        "stake": 10.0,
        "status": "pending",
    })
    pd.DataFrame([row], columns=BETS_COLUMNS).to_csv(
        tracker.bets_file, index=False
    )

    assert tracker.settle_pending_bets_for_match(
        match_uid="match_missing_identity",
        p1=None,
        p2=None,
        actual_winner=1,
    ) == 0
    pending = pd.read_csv(tracker.bets_file, keep_default_na=False).iloc[0]
    assert pending["status"] == "pending"
    assert pending["outcome"] == ""
    assert tracker.get_current_bankroll() == 1000.0


def test_metric_true_fails_closed_without_bound_result_payload(tmp_path):
    from settlement_attribution import (
        ATTRIBUTION_QUALITY_EXACT_UID,
        SETTLEMENT_QUALITY_EXACT_UID,
        FeatureAttributionEvidence,
        build_auto_result_evidence,
    )
    from utils.bet_tracker import BETS_COLUMNS, BetTracker

    tracker = BetTracker(str(tmp_path))
    rows = []
    for bet_id in ("forward", "direct"):
        row = {column: "" for column in BETS_COLUMNS}
        row.update({
            "bet_id": bet_id,
            "match": "Player One vs Player Two",
            "match_uid": "match_exact",
            "feature_snapshot_id": "feat_exact",
            "run_id": "run_exact",
            "bet_on": "Player One",
            "bet_on_player1": True,
            "odds_decimal": 2.0,
            "stake": 10.0,
            "status": "pending",
        })
        rows.append(row)
    pd.DataFrame(rows, columns=BETS_COLUMNS).to_csv(tracker.bets_file, index=False)
    feature_evidence = {
        "feat_exact": FeatureAttributionEvidence(
            feature_snapshot_id="feat_exact",
            run_id="run_exact",
            match_uid="match_exact",
            p1="Player One",
            p2="Player Two",
            feature_schema_sha256="a" * 64,
            feature_vector_sha256="b" * 64,
        )
    }
    bound = build_auto_result_evidence(
        source_evidence={"source": "atp_results", "card": "exact"},
        match_uid="match_exact",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
        score="6-4 6-4",
    )
    kind, digest = bound

    # A kind/hash pair without its canonical payload cannot authorize metrics.
    assert tracker.settle_pending_bets_for_match(
        match_uid="match_exact",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
        exact_feature_evidence=feature_evidence,
        result_evidence_kind=kind,
        result_evidence_sha256=digest,
    ) == 2
    settled = pd.read_csv(
        tracker.bets_file, keep_default_na=False, dtype=str
    ).set_index("bet_id")
    assert set(settled["metric_eligible"]) == {""}
    assert set(settled["result_evidence_kind"]) == {""}

    # The lower-level writer independently rejects a caller-asserted true claim.
    settled.loc["direct", [
        "status", "outcome", "actual_profit", "bankroll_after",
        "settled_timestamp", "settlement_quality", "attribution_quality",
        "metric_eligible", "result_evidence_kind", "result_evidence_sha256",
    ]] = ["pending", "", "", "", "", "", "", "", "", ""]
    settled.reset_index().reindex(columns=BETS_COLUMNS).to_csv(
        tracker.bets_file, index=False
    )
    tracker.settle_bet(
        "direct",
        won=True,
        settlement_quality=SETTLEMENT_QUALITY_EXACT_UID,
        attribution_quality=ATTRIBUTION_QUALITY_EXACT_UID,
        metric_eligible=True,
        result_evidence_kind=kind,
        result_evidence_sha256=digest,
        exact_feature_evidence=feature_evidence,
    )
    direct = pd.read_csv(
        tracker.bets_file, keep_default_na=False
    ).set_index("bet_id").loc["direct"]
    assert direct["metric_eligible"] == ""
    assert direct["settlement_quality"] == (
        "authoritative_result_exact_match_uid"
    )
    assert direct["attribution_quality"] == (
        "exact_match_uid_unverified_feature_snapshot"
    )
    assert direct["result_evidence_kind"] == ""


def test_direct_uid_unknown_upgrades_when_exact_evidence_recovers(tmp_path):
    from settlement_attribution import (
        FeatureAttributionEvidence,
        build_auto_result_evidence,
    )
    from utils.bet_tracker import BETS_COLUMNS, BetTracker

    tracker = BetTracker(str(tmp_path))
    row = {column: "" for column in BETS_COLUMNS}
    row.update({
        "bet_id": "repairable",
        "session_id": "session_repairable",
        "match": "Player One vs Player Two",
        "match_uid": "match_repairable",
        "feature_snapshot_id": "feat_repairable",
        "run_id": "run_repairable",
        "bet_on": "Player One",
        "bet_on_player1": True,
        "odds_decimal": 2.0,
        "stake": 10.0,
        "status": "pending",
    })
    pd.DataFrame([row], columns=BETS_COLUMNS).to_csv(
        tracker.bets_file, index=False
    )
    result = build_auto_result_evidence(
        source_evidence={"source": "atp_results", "event": "Rome"},
        match_uid="match_repairable",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
        score="6-4 6-4",
    )
    assert tracker.settle_pending_bets_for_match(
        match_uid="match_repairable",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
        exact_feature_evidence={},
        result_evidence_kind=result.kind,
        result_evidence_sha256=result.sha256,
        bound_result_evidence=result,
    ) == 1
    unknown = pd.read_csv(
        tracker.bets_file, dtype=str, keep_default_na=False
    ).iloc[0]
    assert unknown["metric_eligible"] == ""
    assert unknown["attribution_quality"] == (
        "exact_match_uid_unverified_feature_snapshot"
    )

    predictions = pd.DataFrame([{
        "match_uid": "match_repairable",
        "prediction_uid": "pred_repairable",
        "p1": "Player One",
        "p2": "Player Two",
        "actual_winner": 1,
        "score": "6-4 6-4",
        "settled_at": "2026-07-15T00:00:00Z",
        "record_status": "settled",
        "identity_status": "canonical",
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "features_complete": True,
    }])
    feature_evidence = {
        "feat_repairable": FeatureAttributionEvidence(
            "feat_repairable", "run_repairable", "match_repairable",
            "Player One", "Player Two", "a" * 64, "b" * 64,
        )
    }
    assert tracker.repair_settled_bet_attribution(
        predictions, feature_evidence
    ) == 1
    repaired = pd.read_csv(
        tracker.bets_file, dtype=str, keep_default_na=False
    ).iloc[0]
    assert repaired["metric_eligible"] == "true"
    assert repaired["attribution_quality"] == "exact_match_uid"
    assert repaired["result_evidence_kind"] == result.kind
    assert repaired["result_evidence_sha256"] == result.sha256


def test_legacy_attribution_repair_upgrades_only_fully_proven_blank_row():
    from settlement_attribution import (
        FeatureAttributionEvidence,
        repair_settled_bet_attribution_frame,
    )
    from utils.bet_tracker import BETS_COLUMNS

    def bet_row(bet_id, uid, outcome, metric="", actual_profit=None):
        row = {column: "" for column in BETS_COLUMNS}
        row.update({
            "bet_id": bet_id,
            "status": "settled",
            "match": "Player One vs Player Two",
            "match_uid": uid,
            "feature_snapshot_id": "feat_exact",
            "run_id": "run_exact",
            "bet_on": "Player One",
            "bet_on_player1": True,
            "outcome": outcome,
            "stake": 10.0,
            "odds_decimal": 2.0,
            "actual_profit": (
                actual_profit
                if actual_profit is not None
                else 10.0 if outcome == "win" else -10.0
            ),
            "metric_eligible": metric,
        })
        if metric:
            row["settlement_quality"] = "existing"
            row["attribution_quality"] = "existing"
        return row

    bets = pd.DataFrame([
        bet_row("proven", "match_exact", "win"),
        bet_row("explicit_false", "match_exact", "win", "false"),
        bet_row("wrong_outcome", "match_exact", "loss"),
        bet_row("wrong_pnl", "match_exact", "win", actual_profit=-10.0),
        bet_row("uid_unjoined", "match_missing", "win"),
    ], columns=BETS_COLUMNS)
    predictions = pd.DataFrame([{
        "match_uid": "match_exact",
        "prediction_uid": "pred_exact",
        "p1": "Player One",
        "p2": "Player Two",
        "actual_winner": 1,
        "score": "6-4 6-4",
        "settled_at": "2026-07-14T12:00:00+00:00",
        "record_status": "settled",
        "identity_status": "canonical",
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "features_complete": True,
        # The settled prediction row is result identity evidence. Its later
        # enriched observation need not repeat the bet-time feature hashes;
        # the exact bet snapshot authority below owns that contract.
        "feature_snapshot_verified": False,
        "feature_schema_sha256": "",
        "feature_vector_sha256": "",
    }])
    evidence = {
        "feat_exact": FeatureAttributionEvidence(
            feature_snapshot_id="feat_exact",
            run_id="run_exact",
            match_uid="match_exact",
            p1="Player One",
            p2="Player Two",
            feature_schema_sha256="a" * 64,
            feature_vector_sha256="b" * 64,
        )
    }

    repaired, count = repair_settled_bet_attribution_frame(
        bets, predictions, evidence
    )
    repaired = repaired.set_index("bet_id")

    assert count == 1
    assert repaired.loc["proven", "metric_eligible"] == "true"
    assert repaired.loc["proven", "attribution_quality"] == "exact_match_uid"
    assert repaired.loc["proven", "result_evidence_kind"] == (
        "prediction_log_exact_match_uid_feature_snapshot_bound"
    )
    assert len(repaired.loc["proven", "result_evidence_sha256"]) == 64
    assert repaired.loc["explicit_false", "metric_eligible"] == "false"
    assert repaired.loc["explicit_false", "settlement_quality"] == "existing"
    assert repaired.loc["wrong_outcome", "metric_eligible"] == ""
    assert repaired.loc["wrong_pnl", "metric_eligible"] == ""
    assert repaired.loc["uid_unjoined", "metric_eligible"] == ""

    tombstoned = predictions.copy()
    tombstoned.loc[0, "record_status"] = "identity_conflict"
    _, tombstone_count = repair_settled_bet_attribution_frame(
        bets.iloc[[0]].copy(), tombstoned, evidence
    )
    assert tombstone_count == 0


def test_legacy_repair_accepts_compatible_duplicate_uid_consensus():
    from settlement_attribution import (
        FeatureAttributionEvidence,
        prediction_match_supports_exact_attribution,
        repair_settled_bet_attribution_frame,
    )
    from utils.bet_tracker import BETS_COLUMNS

    bet = {column: "" for column in BETS_COLUMNS}
    bet.update({
        "bet_id": "duplicate_consensus",
        "status": "settled",
        "match": "Player One vs Player Two",
        "match_uid": "match_duplicate",
        "feature_snapshot_id": "feat_duplicate",
        "run_id": "run_duplicate",
        "bet_on": "Player One",
        "bet_on_player1": True,
        "outcome": "win",
        "stake": 10.0,
        "odds_decimal": 2.0,
        "actual_profit": 10.0,
    })
    common = {
        "match_uid": "match_duplicate",
        "score": "6-4 6-4",
        "settled_at": "2026-07-15T00:00:00Z",
        "record_status": "settled",
        "identity_status": "canonical",
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "features_complete": True,
    }
    predictions = pd.DataFrame([
        {
            **common, "prediction_uid": "pred_a", "p1": "Player One",
            "p2": "Player Two", "actual_winner": 1,
        },
        {
            **common, "prediction_uid": "pred_b", "p1": "Player Two",
            "p2": "Player One", "actual_winner": 2,
        },
    ])
    evidence = {
        "feat_duplicate": FeatureAttributionEvidence(
            "feat_duplicate", "run_duplicate", "match_duplicate",
            "Player One", "Player Two", "a" * 64, "b" * 64,
        )
    }

    assert prediction_match_supports_exact_attribution(
        predictions,
        "match_duplicate",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    )
    repaired, count = repair_settled_bet_attribution_frame(
        pd.DataFrame([bet], columns=BETS_COLUMNS), predictions, evidence
    )
    assert count == 1
    assert repaired.loc[0, "metric_eligible"] == "true"

    predictions.loc[1, "actual_winner"] = 1
    assert not prediction_match_supports_exact_attribution(
        predictions,
        "match_duplicate",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    )
    _, conflicting_count = repair_settled_bet_attribution_frame(
        pd.DataFrame([bet], columns=BETS_COLUMNS), predictions, evidence
    )
    assert conflicting_count == 0

    predictions.loc[1, "actual_winner"] = 2
    predictions.loc[1, "p2"] = "Different Opponent"
    assert not prediction_match_supports_exact_attribution(
        predictions,
        "match_duplicate",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    )
    _, conflicting_pair_count = repair_settled_bet_attribution_frame(
        pd.DataFrame([bet], columns=BETS_COLUMNS), predictions, evidence
    )
    assert conflicting_pair_count == 0


def test_auto_result_evidence_hash_is_source_bound_and_deterministic():
    from dataclasses import replace

    from settlement_attribution import (
        build_auto_result_evidence,
        build_prediction_result_evidence,
        result_evidence_matches_settlement,
    )

    kwargs = {
        "source_evidence": {
            "source": "atp_results",
            "event": "Rome",
            "card_round": "QF",
        },
        "match_uid": "match_exact",
        "p1": "Player One",
        "p2": "Player Two",
        "actual_winner": 1,
        "score": "6-4 6-4",
    }
    first = build_auto_result_evidence(**kwargs)
    second = build_auto_result_evidence(**kwargs)
    changed = build_auto_result_evidence(**{**kwargs, "actual_winner": 2})

    assert first == second
    assert first[0] == "auto_settle_atp_results"
    assert len(first[1]) == 64
    assert changed[1] != first[1]
    assert result_evidence_matches_settlement(
        first,
        match_uid="match_exact",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    )
    assert not result_evidence_matches_settlement(
        replace(first, canonical_payload_json=first.canonical_payload_json + " "),
        match_uid="match_exact",
        p1="Player One",
        p2="Player Two",
        actual_winner=1,
    )
    assert not result_evidence_matches_settlement(
        replace(first, actual_winner=2),
        match_uid="match_exact",
        p1="Player One",
        p2="Player Two",
        actual_winner=2,
    )

    prediction_kind, prediction_hash = build_prediction_result_evidence({
        "match_uid": "match_exact",
        "prediction_uid": "pred_exact",
        "p1": "Player One",
        "p2": "Player Two",
        "actual_winner": 1,
        "score": "6-4 6-4",
        "settled_at": "2026-07-14T12:00:00+00:00",
    })
    assert prediction_kind == "prediction_log_exact_match_uid"
    assert len(prediction_hash) == 64


def test_fractional_actual_winner_never_becomes_ground_truth(tmp_path):
    from settlement_attribution import (
        build_prediction_result_evidence,
        parse_actual_winner,
    )
    from utils.bet_tracker import BETS_COLUMNS, BetTracker

    assert parse_actual_winner(1) == 1
    assert parse_actual_winner("2.0") == 2
    assert parse_actual_winner(1.9) is None
    assert parse_actual_winner(True) is None
    assert parse_actual_winner(float("nan")) is None
    evidence = build_prediction_result_evidence({
        "match_uid": "match_fractional",
        "prediction_uid": "pred_fractional",
        "p1": "Player One",
        "p2": "Player Two",
        "actual_winner": 1.9,
        "settled_at": "2026-07-15T00:00:00Z",
    })
    assert evidence.kind == ""
    assert evidence.sha256 == ""

    tracker = BetTracker(str(tmp_path))
    pending = {column: "" for column in BETS_COLUMNS}
    pending.update({
        "bet_id": "fractional_winner",
        "match": "Player One vs Player Two",
        "match_uid": "match_fractional",
        "bet_on": "Player One",
        "bet_on_player1": True,
        "odds_decimal": 2.0,
        "stake": 10.0,
        "status": "pending",
    })
    pd.DataFrame([pending], columns=BETS_COLUMNS).to_csv(
        tracker.bets_file, index=False
    )
    assert tracker.settle_pending_bets_for_match(
        match_uid="match_fractional",
        p1="Player One",
        p2="Player Two",
        actual_winner=1.9,
    ) == 0
    row = pd.read_csv(
        tracker.bets_file, dtype=str, keep_default_na=False
    ).iloc[0]
    assert row["status"] == "pending"
    assert row["outcome"] == ""


def test_generic_bovada_tournament_titles_resolve_to_useful_metadata():
    from tournaments.resolve_tournament import TournamentResolver, level_hint_from_title

    resolver = TournamentResolver(str(REPO_ROOT / "data" / "tournaments_map.csv"))

    french_open, _ = resolver.resolve_soft("French Open - French Open Men's Singles (41)")
    rome, _ = resolver.resolve_soft("ATP - Rome (2)")
    bengaluru, _ = resolver.resolve_soft("Challenger - Bengaluru (4)")
    oeiras, _ = resolver.resolve_soft("Challenger - Oeiras (4)")
    reggio, _ = resolver.resolve_soft("ITF Men's - ITF Men Reggio Emilia (4)")

    assert french_open is not None
    assert french_open.level == "G"
    assert french_open.surface == "Clay"
    assert french_open.draw_size == 128
    assert rome is not None
    assert rome.level == "M"
    assert rome.surface == "Clay"
    assert rome.draw_size >= 96
    assert bengaluru is not None
    assert bengaluru.level == "C"
    assert bengaluru.draw_size == 32
    assert oeiras is not None
    assert oeiras.surface == "Clay"
    assert oeiras.level == "C"
    assert reggio is not None
    assert reggio.level == "15"
    assert level_hint_from_title("ITF Men's - ITF Men Gaborone (8)") == "15"


def test_fallback_heuristics_do_not_default_known_events_to_atp():
    from tournaments.fallback_heuristics import get_fallback_tournament_meta

    french_open = get_fallback_tournament_meta("French Open - French Open Men's Singles (41)")
    generic_itf = get_fallback_tournament_meta("ITF Men's - ITF Men Gaborone (8)")

    assert french_open.level == "G"
    assert french_open.surface == "Clay"
    assert french_open.draw_size == 128
    assert generic_itf.level == "15"
    assert generic_itf.draw_size == 32


def test_ta_upcoming_surface_overrides_fallback_metadata_only():
    from features.ta_feature_calculator import reconcile_upcoming_surface

    assert reconcile_upcoming_surface("Hard", "Clay", "fallback_heuristic") == ("Clay", True)
    assert reconcile_upcoming_surface("Hard", "Clay", "default") == ("Clay", True)
    assert reconcile_upcoming_surface("Hard", "Clay", "resolved") == ("Hard", False)


def test_betting_edges_preserve_event_for_bet_slips():
    from models.inference import calculate_betting_edges
    from utils.stake_calculator import KellyStakeCalculator

    predictions = pd.DataFrame(
        [
            {
                "player1_raw": "Player One",
                "player2_raw": "Player Two",
                "player1_win_prob": 0.65,
                "player2_win_prob": 0.35,
                "event": "ATP - Rome (2)",
                "match_time": "5/13/26 6:00 AM",
                "meta_match_date": "2026-05-13",
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "player1_raw": "Player One",
                "player2_raw": "Player Two",
                "player1_odds_decimal": 2.10,
                "player2_odds_decimal": 1.80,
                "player1_implied_prob": 0.48,
                "player2_implied_prob": 0.52,
                "event": "ATP - Rome (2)",
                "match_time": "5/13/26 6:00 AM",
            }
        ]
    )

    edges = calculate_betting_edges(predictions, odds)
    calculator = KellyStakeCalculator(edge_threshold=0.01, min_stake_dollars=1.0)
    slips = calculator.generate_bet_slips(calculator.allocate_block_stakes(edges, bankroll=1000.0))

    assert edges.loc[0, "event"] == "ATP - Rome (2)"
    assert edges.loc[0, "match_time"] == "5/13/26 6:00 AM"
    assert slips.loc[0, "event"] == "ATP - Rome (2)"
    assert slips.loc[0, "match_start_time"] == "5/13/26 6:00 AM"
    assert slips.loc[0, "match_date"] == "2026-05-13"


def test_staking_enforces_per_bet_run_and_pending_exposure_caps():
    from utils.stake_calculator import KellyStakeCalculator

    calculator = KellyStakeCalculator(
        kelly_multiplier=0.18, max_stake_fraction=0.05,
        edge_threshold=0.02, min_stake_dollars=1.0,
    )
    matches = pd.DataFrame([
        {"bet_prob": 0.90, "bet_odds": 2.0, "match_time": "7/14/26 1:00 PM"},
        {"bet_prob": 0.90, "bet_odds": 2.0, "match_time": "7/15/26 1:00 PM"},
        {"bet_prob": 0.90, "bet_odds": 2.0, "match_time": "7/16/26 1:00 PM"},
    ])
    stakes = calculator.allocate_block_stakes(
        matches, bankroll=1000.0, available_bankroll=80.0,
    )

    assert stakes["final_stake"].max() <= 50.0
    assert stakes["final_stake"].sum() <= 80.0
    assert stakes["final_stake"].sum() <= 180.0


def test_staking_fails_closed_on_invalid_probability_or_odds():
    from utils.stake_calculator import KellyStakeCalculator

    calculator = KellyStakeCalculator()
    assert calculator.kelly_fraction(float("nan"), 2.0) == 0.0
    assert calculator.kelly_fraction(1.01, 2.0) == 0.0
    assert calculator.kelly_fraction(0.6, float("inf")) == 0.0
    assert calculator.kelly_fraction("not-a-number", 2.0) == 0.0


def test_account_equity_and_available_capital_do_not_reset_per_session(tmp_path):
    from utils.bet_tracker import BetTracker

    tracker = BetTracker(str(tmp_path), initial_bankroll=1000.0)
    session_id = tracker.start_session(1000.0, 0.18, "account test")
    slip = pd.DataFrame([{
        "event": "Rome", "match": "A vs B", "match_uid": "m_account",
        "feature_snapshot_id": "f_account", "run_id": "r_account",
        "bet_on": "A", "bet_on_player1": True, "odds_decimal": 2.0,
        "stake": 50.0, "stake_fraction": 0.05, "model_prob": 0.7,
        "market_prob": 0.5, "edge": 0.2, "kelly_fraction": 0.4,
        "potential_profit": 50.0, "potential_loss": 50.0,
        "bankroll": 1000.0, "model_version": "v-test",
        "match_date": "2026-07-14", "match_start_time": "7/14/26 1:00 PM",
    }])
    assert tracker.log_bets(slip, session_id, 1000.0) == 1
    assert tracker.get_current_bankroll() == 1000.0
    assert tracker.get_pending_exposure() == 50.0
    assert tracker.get_available_bankroll() == 950.0
    tracker.settle_bet(pd.read_csv(tracker.bets_file).iloc[0]["bet_id"], won=False)
    assert tracker.get_current_bankroll() == 950.0
    second_session = tracker.start_session(950.0, 0.18, "next account slice")
    assert second_session != session_id
    assert tracker.get_current_bankroll() == 950.0


def test_bet_tracker_rejects_batch_that_exceeds_unreserved_capital(tmp_path):
    from utils.bet_tracker import BetTracker

    tracker = BetTracker(str(tmp_path), initial_bankroll=100.0)
    session_id = tracker.start_session(100.0, 0.18, "writer gate")
    slip = pd.DataFrame([{
        "event": "Rome", "match": "A vs B", "match_uid": "m_writer_gate",
        "feature_snapshot_id": "f_writer_gate", "run_id": "r_writer_gate",
        "bet_on": "A", "bet_on_player1": True, "odds_decimal": 2.0,
        "stake": 101.0, "stake_fraction": 1.01, "model_prob": 0.7,
        "market_prob": 0.5, "edge": 0.2, "kelly_fraction": 0.4,
        "potential_profit": 101.0, "potential_loss": 101.0,
        "bankroll": 100.0, "model_version": "v-test",
        "match_date": "2026-07-14", "match_start_time": "7/14/26 1:00 PM",
    }])

    assert tracker.log_bets(slip, session_id, 100.0) == 0
    assert tracker.get_pending_bets().empty
    assert tracker.get_available_bankroll() == 100.0


def test_pending_exposure_is_case_insensitive(tmp_path):
    from utils.bet_tracker import BETS_COLUMNS, BetTracker

    tracker = BetTracker(str(tmp_path), initial_bankroll=1000.0)
    row = {column: "" for column in BETS_COLUMNS}
    row.update({"bet_id": "case_pending", "status": "PENDING", "stake": 25.0})
    pd.DataFrame([row], columns=BETS_COLUMNS).to_csv(tracker.bets_file, index=False)

    assert tracker.get_pending_exposure() == 25.0
    assert tracker.get_available_bankroll() == 975.0


def test_incomplete_prediction_upgrades_to_first_complete(tmp_path, monkeypatch):
    """An incomplete-featured row must be REPLACED by the first complete
    prediction (probs+odds together); complete rows stay frozen (opening line)."""
    import prediction_logger as PL
    monkeypatch.setattr(PL, "LOG_PATH", tmp_path / "prediction_log.csv")
    monkeypatch.setattr(PL, "SNAPSHOT_LOG_PATH", tmp_path / "prediction_snapshots.csv")
    monkeypatch.setattr(PL, "ODDS_HISTORY_LOG_PATH", tmp_path / "odds_history.csv")
    common = dict(p1="Player A", p2="Player B", match_date="2026-07-08",
                  tournament="Testville", surface="Hard", level="C", round_code="R32",
                  market_p1_prob=0.5, market_p2_prob=0.5,
                  p1_odds_american="-110", p2_odds_american="-110",
                  p1_odds_decimal=1.91, p2_odds_decimal=1.91)
    # first log: incomplete features
    PL.log_prediction(model_p1_prob=0.40, model_p2_prob=0.60, model_version="v1",
                      features_complete=False, defaulted_features="round_code=None", **common)
    # second log: complete features, new probs -> must overwrite
    r = PL.log_prediction(model_p1_prob=0.55, model_p2_prob=0.45, model_version="v1",
                          features_complete=True, defaulted_features="", **common)
    assert r == "updated"
    import pandas as pd
    df = pd.read_csv(tmp_path / "prediction_log.csv")
    assert len(df) == 1
    assert abs(df.iloc[0]["model_p1_prob"] - 0.55) < 1e-9
    assert str(df.iloc[0]["features_complete"]) in ("True", "1", "1.0")
    # third log: still complete, different probs -> must PRESERVE (opening line)
    PL.log_prediction(model_p1_prob=0.70, model_p2_prob=0.30, model_version="v1",
                      features_complete=True, defaulted_features="", **common)
    df = pd.read_csv(tmp_path / "prediction_log.csv")
    assert abs(df.iloc[0]["model_p1_prob"] - 0.55) < 1e-9  # frozen at first COMPLETE


def test_regime_bump_unfreezes_complete_rows(tmp_path, monkeypatch):
    """A model-regime version bump must re-price a frozen complete row (the
    fix that regime marks is exactly why the stored probabilities are stale);
    same-version re-logs stay frozen."""
    import pandas as pd
    import prediction_logger as PL
    monkeypatch.setattr(PL, "LOG_PATH", tmp_path / "prediction_log.csv")
    monkeypatch.setattr(PL, "SNAPSHOT_LOG_PATH", tmp_path / "prediction_snapshots.csv")
    monkeypatch.setattr(PL, "ODDS_HISTORY_LOG_PATH", tmp_path / "odds_history.csv")
    common = dict(p1="Player A", p2="Player B", match_date="2026-07-08",
                  tournament="Testville", surface="Hard", level="C", round_code="R32",
                  market_p1_prob=0.5, market_p2_prob=0.5,
                  p1_odds_american="-110", p2_odds_american="-110",
                  p1_odds_decimal=1.91, p2_odds_decimal=1.91)
    PL.log_prediction(model_p1_prob=0.55, model_p2_prob=0.45, model_version="v1.2.2",
                      nn_model_version="v1.2.2", features_complete=True,
                      defaulted_features="", **common)
    # same regime -> frozen at first complete
    PL.log_prediction(model_p1_prob=0.70, model_p2_prob=0.30, model_version="v1.2.2",
                      nn_model_version="v1.2.2", features_complete=True,
                      defaulted_features="", **common)
    df = pd.read_csv(tmp_path / "prediction_log.csv")
    assert abs(df.iloc[0]["model_p1_prob"] - 0.55) < 1e-9
    # regime bump -> refresh probs
    r = PL.log_prediction(model_p1_prob=0.62, model_p2_prob=0.38, model_version="v1.2.3",
                          nn_model_version="v1.2.3", features_complete=True,
                          defaulted_features="", **common)
    assert r == "updated"
    df = pd.read_csv(tmp_path / "prediction_log.csv")
    assert len(df) == 1
    assert abs(df.iloc[0]["model_p1_prob"] - 0.62) < 1e-9
    assert df.iloc[0]["nn_model_version"] == "v1.2.3"
    # an INCOMPLETE build under an even newer regime must NOT downgrade the row
    PL.log_prediction(model_p1_prob=0.50, model_p2_prob=0.50, model_version="v1.2.4",
                      nn_model_version="v1.2.4", features_complete=False,
                      defaulted_features="height=missing", **common)
    df = pd.read_csv(tmp_path / "prediction_log.csv")
    assert abs(df.iloc[0]["model_p1_prob"] - 0.62) < 1e-9
    assert str(df.iloc[0]["features_complete"]) in ("True", "1", "1.0")
