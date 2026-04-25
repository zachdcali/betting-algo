import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "src" / "models" / "professional_tennis"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from performance_features import add_performance_features, parse_score_summary  # noqa: E402


def test_parse_score_summary_deciding_tiebreak():
    summary = parse_score_summary("6-4 3-6 7-6(5)", best_of=3)

    assert summary["valid"] is True
    assert summary["winner_sets"] == 2
    assert summary["loser_sets"] == 1
    assert summary["winner_games"] == 16
    assert summary["loser_games"] == 16
    assert summary["winner_tiebreaks"] == 1
    assert summary["deciding_set"] is True


def test_parse_score_summary_retirement_invalid_for_form():
    summary = parse_score_summary("6-4 2-1 RET", best_of=3)

    assert summary["retired"] is True
    assert summary["valid"] is False


def test_add_performance_features_are_prematch_and_oriented():
    rows = [
        _row(
            date="20240101",
            match_num=1,
            p1="A",
            p2="B",
            p1_wins=1,
            score="6-4 6-4",
            w_svpt=60,
            l_svpt=58,
        ),
        _row(
            date="20240108",
            match_num=1,
            p1="A",
            p2="C",
            p1_wins=0,
            score="7-5 6-2",
            w_svpt=62,
            l_svpt=55,
        ),
        _row(
            date="20240115",
            match_num=1,
            p1="A",
            p2="B",
            p1_wins=1,
            score="6-3 6-3",
            w_svpt=50,
            l_svpt=49,
        ),
    ]
    df = pd.DataFrame(rows)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df["inferred_match_date"] = df["tourney_date"]

    out = add_performance_features(df)
    third = out.iloc[2]

    assert third["P1_Score_Matches_Last10"] == 2
    assert third["P2_Score_Matches_Last10"] == 1
    assert third["Score_Matches_Last10_Diff"] == 1
    assert pd.notna(third["P1_Service_Points_Won_Last10"])
    assert pd.notna(third["P2_Return_Points_Won_Last10"])


def _row(date, match_num, p1, p2, p1_wins, score, w_svpt, l_svpt):
    return {
        "tourney_date": date,
        "match_num": match_num,
        "surface": "Hard",
        "best_of": 3,
        "Player1_ID": p1,
        "Player1_Name": p1,
        "Player2_ID": p2,
        "Player2_Name": p2,
        "Player1_Wins": p1_wins,
        "score": score,
        "minutes": 90,
        "w_ace": 5,
        "w_df": 2,
        "w_svpt": w_svpt,
        "w_1stIn": 38,
        "w_1stWon": 28,
        "w_2ndWon": 12,
        "w_bpSaved": 3,
        "w_bpFaced": 4,
        "l_ace": 3,
        "l_df": 4,
        "l_svpt": l_svpt,
        "l_1stIn": 35,
        "l_1stWon": 22,
        "l_2ndWon": 10,
        "l_bpSaved": 2,
        "l_bpFaced": 6,
    }
