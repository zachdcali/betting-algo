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
    assert slips.loc[0, "event"] == "ATP - Rome (2)"
