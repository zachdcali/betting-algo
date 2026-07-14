from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import pandas as pd


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

    assert stats == {"attempts": 1, "created": 1, "updated": 0, "skipped_incomplete": 0}
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


def test_prediction_terminal_status_rejects_all_failed_rows():
    import main

    status, successes, errors = main.prediction_terminal_status(pd.DataFrame([
        {"prediction_status": "error"},
        {"prediction_status": "skipped_missing_data"},
    ]))
    assert (status, successes, errors) == ("no_predictions", 0, 2)

    status, successes, errors = main.prediction_terminal_status(pd.DataFrame([
        {"prediction_status": "success"}, {"prediction_status": "error"},
    ]))
    assert (status, successes, errors) == ("partial", 1, 1)


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
        match_uid="match_bool", actual_winner=2,
    ) == 1
    settled = pd.read_csv(tracker.bets_file).iloc[0]
    assert settled["status"] == "settled"
    assert settled["outcome"] == "win"
    assert settled["actual_profit"] == 10.0


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
