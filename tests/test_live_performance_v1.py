import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from features.performance_v1 import build_match_performance_features, parse_score_summary  # noqa: E402


def test_live_parse_score_summary_handles_deciding_set():
    summary = parse_score_summary("6-4 3-6 7-6(5)", best_of=3)

    assert summary["valid"] is True
    assert summary["winner_sets"] == 2
    assert summary["loser_sets"] == 1
    assert summary["winner_tiebreaks"] == 1
    assert summary["deciding_set"] is True


def test_live_performance_features_are_prematch_and_player_oriented():
    p1_matches = pd.DataFrame(
        [
            _ta_match("2024-01-18", "W", "6-1 6-1", serve_points=44, opp_serve_points=42),
            _ta_match("2024-01-08", "L", "6-4 6-4", serve_points=60, opp_serve_points=58),
            _ta_match("2024-01-01", "W", "6-4 6-4", serve_points=55, opp_serve_points=57),
        ]
    )
    p2_matches = pd.DataFrame(
        [
            _ta_match("2024-01-07", "W", "7-6(4) 6-7(5) 6-3", serve_points=70, opp_serve_points=68),
        ]
    )

    features = build_match_performance_features(
        p1_matches,
        p2_matches,
        match_date="2024-01-15",
    )

    assert features["P1_Score_Matches_Last10"] == 2
    assert features["P2_Score_Matches_Last10"] == 1
    assert features["Score_Matches_Last10_Diff"] == 1
    assert features["P1_Stat_Matches_Last10"] == 2
    assert pd.notna(features["Service_Points_Won_Last10_Diff"])
    assert features["P1_Game_WinRate_Last10"] < 1.0


def _ta_match(date, result, score, serve_points, opp_serve_points):
    first_in = int(serve_points * 0.62)
    first_won = int(first_in * 0.72)
    second_won = int((serve_points - first_in) * 0.50)
    opp_first_in = int(opp_serve_points * 0.60)
    opp_first_won = int(opp_first_in * 0.68)
    opp_second_won = int((opp_serve_points - opp_first_in) * 0.46)
    return {
        "date": date,
        "event": "Example Open",
        "level": "A",
        "round": "R32",
        "result": result,
        "score": score,
        "max_sets": 3,
        "serve_points": serve_points,
        "first_serves_in": first_in,
        "first_serve_won": first_won,
        "second_serve_won": second_won,
        "aces": 5,
        "double_faults": 2,
        "bp_saved": 3,
        "bp_faced": 5,
        "opp_serve_points": opp_serve_points,
        "opp_first_serves_in": opp_first_in,
        "opp_first_serve_won": opp_first_won,
        "opp_second_serve_won": opp_second_won,
        "opp_bp_saved": 2,
        "opp_bp_faced": 6,
        "minutes": 92,
    }
