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
