from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PERFORMANCE_FEATURE_SET = "performance_v1"


PLAYER_PERFORMANCE_METRICS = [
    "Score_Matches_Last10",
    "Stat_Matches_Last10",
    "Set_WinRate_Last10",
    "Game_WinRate_Last10",
    "Game_WinRate_90d",
    "Straight_Set_WinRate_Last10",
    "Deciding_Set_WinRate_Career",
    "Tiebreak_WinRate_Career",
    "Service_Points_Won_Last10",
    "Return_Points_Won_Last10",
    "First_Serve_In_Last10",
    "First_Serve_Won_Last10",
    "Second_Serve_Won_Last10",
    "Ace_Rate_Last10",
    "Double_Fault_Rate_Last10",
    "BP_Save_Rate_Last10",
    "BP_Convert_Rate_Last10",
    "Avg_Minutes_Last10",
    "Avg_Service_Points_Last10",
]

DIFF_METRICS = [
    "Score_Matches_Last10",
    "Stat_Matches_Last10",
    "Set_WinRate_Last10",
    "Game_WinRate_Last10",
    "Game_WinRate_90d",
    "Straight_Set_WinRate_Last10",
    "Deciding_Set_WinRate_Career",
    "Tiebreak_WinRate_Career",
    "Service_Points_Won_Last10",
    "Return_Points_Won_Last10",
    "First_Serve_In_Last10",
    "First_Serve_Won_Last10",
    "Second_Serve_Won_Last10",
    "Ace_Rate_Last10",
    "Double_Fault_Rate_Last10",
    "BP_Save_Rate_Last10",
    "BP_Convert_Rate_Last10",
    "Avg_Minutes_Last10",
    "Avg_Service_Points_Last10",
]

PERFORMANCE_FEATURES: List[str] = (
    [f"P1_{metric}" for metric in PLAYER_PERFORMANCE_METRICS]
    + [f"P2_{metric}" for metric in PLAYER_PERFORMANCE_METRICS]
    + [f"{metric}_Diff" for metric in DIFF_METRICS]
)


SET_SCORE_RE = re.compile(r"(?P<winner_games>\d{1,2})-(?P<loser_games>\d{1,2})(?:\([^)]*\))?")
RETIREMENT_MARKERS = ("RET", "W/O", "WO", "DEF", "ABD", "ABN", "DEFAULT", "WALKOVER")


def laplace_rate(successes: float, total: float, alpha: float = 3.0, prior: float = 0.5) -> float:
    if total <= 0:
        return np.nan
    return float((successes + prior * alpha) / (total + alpha))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator and denominator > 0:
        return float(numerator / denominator)
    return np.nan


def parse_score_summary(score: object, best_of: object = None) -> Dict[str, object]:
    """
    Parse a Sackmann/TA score string from the recorded winner's perspective.

    Retirements/defaults are marked invalid for score-form features because the
    partial score can exaggerate or understate true match quality.
    """
    text = "" if score is None or pd.isna(score) else str(score).strip()
    upper = text.upper()
    retired = any(marker in upper for marker in RETIREMENT_MARKERS)
    set_scores = []

    for match in SET_SCORE_RE.finditer(upper):
        winner_games = int(match.group("winner_games"))
        loser_games = int(match.group("loser_games"))
        if winner_games == loser_games:
            continue
        winner_set_won = winner_games > loser_games
        set_scores.append(
            {
                "winner_games": winner_games,
                "loser_games": loser_games,
                "winner_set_won": winner_set_won,
                "has_tiebreak": "(" in match.group(0),
            }
        )

    winner_sets = sum(1 for item in set_scores if item["winner_set_won"])
    loser_sets = sum(1 for item in set_scores if not item["winner_set_won"])
    valid = bool(set_scores) and not retired and winner_sets > loser_sets

    try:
        best_of_int = int(best_of)
    except (TypeError, ValueError):
        best_of_int = 5 if winner_sets >= 3 else 3
    sets_to_win = best_of_int // 2 + 1

    return {
        "valid": valid,
        "retired": retired,
        "winner_sets": winner_sets,
        "loser_sets": loser_sets,
        "winner_games": sum(item["winner_games"] for item in set_scores),
        "loser_games": sum(item["loser_games"] for item in set_scores),
        "winner_tiebreaks": sum(
            1 for item in set_scores if item["has_tiebreak"] and item["winner_set_won"]
        ),
        "loser_tiebreaks": sum(
            1 for item in set_scores if item["has_tiebreak"] and not item["winner_set_won"]
        ),
        "straight_sets": valid and loser_sets == 0,
        "deciding_set": valid and winner_sets == sets_to_win and loser_sets == sets_to_win - 1,
    }


def _to_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        if value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _oriented_value(row: pd.Series, p1_wins: bool, winner_col: str, loser_col: str, player_num: int) -> float:
    if player_num == 1:
        source_col = winner_col if p1_wins else loser_col
    else:
        source_col = loser_col if p1_wins else winner_col
    return _to_float(row.get(source_col))


def _player_match_performance(row: pd.Series, player_num: int, score_summary: Dict[str, object]) -> Dict[str, object]:
    p1_wins = bool(row.get("Player1_Wins") == 1)
    player_won = p1_wins if player_num == 1 else not p1_wins

    if player_won:
        sets_won = score_summary["winner_sets"]
        sets_lost = score_summary["loser_sets"]
        games_won = score_summary["winner_games"]
        games_lost = score_summary["loser_games"]
        tb_won = score_summary["winner_tiebreaks"]
        tb_lost = score_summary["loser_tiebreaks"]
    else:
        sets_won = score_summary["loser_sets"]
        sets_lost = score_summary["winner_sets"]
        games_won = score_summary["loser_games"]
        games_lost = score_summary["winner_games"]
        tb_won = score_summary["loser_tiebreaks"]
        tb_lost = score_summary["winner_tiebreaks"]

    serve_points = _oriented_value(row, p1_wins, "w_svpt", "l_svpt", player_num)
    first_serves_in = _oriented_value(row, p1_wins, "w_1stIn", "l_1stIn", player_num)
    first_won = _oriented_value(row, p1_wins, "w_1stWon", "l_1stWon", player_num)
    second_won = _oriented_value(row, p1_wins, "w_2ndWon", "l_2ndWon", player_num)
    aces = _oriented_value(row, p1_wins, "w_ace", "l_ace", player_num)
    double_faults = _oriented_value(row, p1_wins, "w_df", "l_df", player_num)
    bp_saved = _oriented_value(row, p1_wins, "w_bpSaved", "l_bpSaved", player_num)
    bp_faced = _oriented_value(row, p1_wins, "w_bpFaced", "l_bpFaced", player_num)

    opponent_num = 2 if player_num == 1 else 1
    opp_serve_points = _oriented_value(row, p1_wins, "w_svpt", "l_svpt", opponent_num)
    opp_first_won = _oriented_value(row, p1_wins, "w_1stWon", "l_1stWon", opponent_num)
    opp_second_won = _oriented_value(row, p1_wins, "w_2ndWon", "l_2ndWon", opponent_num)
    opp_bp_saved = _oriented_value(row, p1_wins, "w_bpSaved", "l_bpSaved", opponent_num)
    opp_bp_faced = _oriented_value(row, p1_wins, "w_bpFaced", "l_bpFaced", opponent_num)

    service_points_won = first_won + second_won
    return_points_won = opp_serve_points - (opp_first_won + opp_second_won)
    bp_converted = opp_bp_faced - opp_bp_saved
    minutes = _to_float(row.get("minutes"))

    stats_valid = bool(pd.notna(serve_points) and serve_points > 0 and pd.notna(opp_serve_points) and opp_serve_points > 0)

    return {
        "date": row.get("inferred_match_date", row.get("tourney_date")),
        "surface": row.get("surface"),
        "won": player_won,
        "score_valid": bool(score_summary["valid"]),
        "sets_won": float(sets_won),
        "sets_lost": float(sets_lost),
        "games_won": float(games_won),
        "games_lost": float(games_lost),
        "straight_set_win": bool(player_won and score_summary["straight_sets"]),
        "deciding_set": bool(score_summary["deciding_set"]),
        "deciding_set_win": bool(player_won and score_summary["deciding_set"]),
        "tb_won": float(tb_won),
        "tb_lost": float(tb_lost),
        "stats_valid": stats_valid,
        "serve_points": serve_points,
        "service_points_won": service_points_won,
        "return_points": opp_serve_points,
        "return_points_won": return_points_won,
        "first_serves_in": first_serves_in,
        "first_won": first_won,
        "second_won": second_won,
        "aces": aces,
        "double_faults": double_faults,
        "bp_saved": bp_saved,
        "bp_faced": bp_faced,
        "bp_converted": bp_converted,
        "bp_chances": opp_bp_faced,
        "minutes": minutes,
    }


def _take_recent(history: Iterable[Dict[str, object]], max_matches: int, target_date: Optional[pd.Timestamp] = None, days: Optional[int] = None) -> List[Dict[str, object]]:
    items = list(history)
    if target_date is not None and days is not None:
        cutoff = target_date - pd.Timedelta(days=days)
        items = [item for item in items if cutoff <= item["date"] < target_date]
    return list(reversed(items))[:max_matches]


def _sum_valid(items: Iterable[Dict[str, object]], numerator: str, denominator: str) -> float:
    num = 0.0
    den = 0.0
    for item in items:
        item_num = _to_float(item.get(numerator))
        item_den = _to_float(item.get(denominator))
        if pd.notna(item_num) and pd.notna(item_den) and item_den > 0:
            num += item_num
            den += item_den
    return safe_ratio(num, den)


def _mean_valid(items: Iterable[Dict[str, object]], key: str) -> float:
    values = [_to_float(item.get(key)) for item in items]
    values = [value for value in values if pd.notna(value)]
    if not values:
        return np.nan
    return float(np.mean(values))


def _count_valid(items: Iterable[Dict[str, object]], key: str) -> int:
    return sum(1 for item in items if item.get(key))


def _history_features(prefix: str, history: deque, career: Dict[str, float], target_date: pd.Timestamp) -> Dict[str, float]:
    last10 = _take_recent(history, 10)
    score_last10 = [item for item in last10 if item.get("score_valid")]
    stats_last10 = [item for item in last10 if item.get("stats_valid")]
    score_90d = [
        item
        for item in _take_recent(history, max_matches=80, target_date=target_date, days=90)
        if item.get("score_valid")
    ]

    sets_won = sum(item["sets_won"] for item in score_last10)
    sets_total = sets_won + sum(item["sets_lost"] for item in score_last10)
    games_won = sum(item["games_won"] for item in score_last10)
    games_total = games_won + sum(item["games_lost"] for item in score_last10)
    games_won_90d = sum(item["games_won"] for item in score_90d)
    games_total_90d = games_won_90d + sum(item["games_lost"] for item in score_90d)
    straight_wins = sum(1 for item in score_last10 if item.get("straight_set_win"))

    deciding_rate = laplace_rate(career["deciding_wins"], career["deciding_total"])
    tb_total = career["tb_wins"] + career["tb_losses"]
    tiebreak_rate = laplace_rate(career["tb_wins"], tb_total)

    return {
        f"{prefix}_Score_Matches_Last10": float(len(score_last10)),
        f"{prefix}_Stat_Matches_Last10": float(len(stats_last10)),
        f"{prefix}_Set_WinRate_Last10": laplace_rate(sets_won, sets_total),
        f"{prefix}_Game_WinRate_Last10": laplace_rate(games_won, games_total),
        f"{prefix}_Game_WinRate_90d": laplace_rate(games_won_90d, games_total_90d),
        f"{prefix}_Straight_Set_WinRate_Last10": laplace_rate(straight_wins, len(score_last10)),
        f"{prefix}_Deciding_Set_WinRate_Career": deciding_rate,
        f"{prefix}_Tiebreak_WinRate_Career": tiebreak_rate,
        f"{prefix}_Service_Points_Won_Last10": _sum_valid(stats_last10, "service_points_won", "serve_points"),
        f"{prefix}_Return_Points_Won_Last10": _sum_valid(stats_last10, "return_points_won", "return_points"),
        f"{prefix}_First_Serve_In_Last10": _sum_valid(stats_last10, "first_serves_in", "serve_points"),
        f"{prefix}_First_Serve_Won_Last10": _sum_valid(stats_last10, "first_won", "first_serves_in"),
        f"{prefix}_Second_Serve_Won_Last10": _sum_valid(
            stats_last10,
            "second_won",
            "__second_serve_attempts",
        ),
        f"{prefix}_Ace_Rate_Last10": _sum_valid(stats_last10, "aces", "serve_points"),
        f"{prefix}_Double_Fault_Rate_Last10": _sum_valid(stats_last10, "double_faults", "serve_points"),
        f"{prefix}_BP_Save_Rate_Last10": _sum_valid(stats_last10, "bp_saved", "bp_faced"),
        f"{prefix}_BP_Convert_Rate_Last10": _sum_valid(stats_last10, "bp_converted", "bp_chances"),
        f"{prefix}_Avg_Minutes_Last10": _mean_valid(last10, "minutes"),
        f"{prefix}_Avg_Service_Points_Last10": _mean_valid(stats_last10, "serve_points"),
    }


def _add_second_serve_attempts(item: Dict[str, object]) -> None:
    serve_points = _to_float(item.get("serve_points"))
    first_serves_in = _to_float(item.get("first_serves_in"))
    if pd.notna(serve_points) and pd.notna(first_serves_in):
        item["__second_serve_attempts"] = max(serve_points - first_serves_in, 0.0)
    else:
        item["__second_serve_attempts"] = np.nan


def _update_career(career: Dict[str, float], match_perf: Dict[str, object]) -> None:
    if not match_perf.get("score_valid"):
        return
    if match_perf.get("deciding_set"):
        career["deciding_total"] += 1
        if match_perf.get("deciding_set_win"):
            career["deciding_wins"] += 1
    career["tb_wins"] += float(match_perf.get("tb_won", 0.0))
    career["tb_losses"] += float(match_perf.get("tb_lost", 0.0))


def add_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-match score and serve/return form features.

    The calculation is chronological: each row receives features from player
    histories before the current match is added to those histories.
    """
    if "inferred_match_date" in df.columns:
        sorted_df = df.sort_values(["inferred_match_date", "match_num"]).reset_index(drop=True).copy()
    else:
        sorted_df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True).copy()

    sorted_df["tourney_date"] = pd.to_datetime(sorted_df["tourney_date"])
    if "inferred_match_date" in sorted_df.columns:
        sorted_df["inferred_match_date"] = pd.to_datetime(sorted_df["inferred_match_date"])

    histories = defaultdict(lambda: deque(maxlen=80))
    careers = defaultdict(lambda: defaultdict(float))
    feature_rows: List[Dict[str, float]] = []

    for idx, row in sorted_df.iterrows():
        if idx % 50000 == 0:
            print(f"  Performance features: processed {idx:,}/{len(sorted_df):,} matches")

        match_date = row.get("inferred_match_date", row.get("tourney_date"))
        p1_id = row["Player1_ID"] if pd.notna(row.get("Player1_ID")) else row.get("Player1_Name")
        p2_id = row["Player2_ID"] if pd.notna(row.get("Player2_ID")) else row.get("Player2_Name")

        features: Dict[str, float] = {}
        features.update(_history_features("P1", histories[p1_id], careers[p1_id], match_date))
        features.update(_history_features("P2", histories[p2_id], careers[p2_id], match_date))
        for metric in DIFF_METRICS:
            p1_value = features.get(f"P1_{metric}")
            p2_value = features.get(f"P2_{metric}")
            features[f"{metric}_Diff"] = (
                float(p1_value - p2_value)
                if pd.notna(p1_value) and pd.notna(p2_value)
                else np.nan
            )
        feature_rows.append(features)

        score_summary = parse_score_summary(row.get("score"), row.get("best_of"))
        p1_perf = _player_match_performance(row, 1, score_summary)
        p2_perf = _player_match_performance(row, 2, score_summary)
        _add_second_serve_attempts(p1_perf)
        _add_second_serve_attempts(p2_perf)
        histories[p1_id].append(p1_perf)
        histories[p2_id].append(p2_perf)
        _update_career(careers[p1_id], p1_perf)
        _update_career(careers[p2_id], p2_perf)

    feature_df = pd.DataFrame(feature_rows)
    for col in PERFORMANCE_FEATURES:
        sorted_df[col] = feature_df[col].values if col in feature_df.columns else np.nan

    return sorted_df
