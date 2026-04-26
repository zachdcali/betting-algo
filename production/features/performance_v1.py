from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from round_offsets import get_round_day_offset, infer_draw_size
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .round_offsets import get_round_day_offset, infer_draw_size


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

DIFF_METRICS = list(PLAYER_PERFORMANCE_METRICS)

PERFORMANCE_FEATURES: List[str] = (
    [f"P1_{metric}" for metric in PLAYER_PERFORMANCE_METRICS]
    + [f"P2_{metric}" for metric in PLAYER_PERFORMANCE_METRICS]
    + [f"{metric}_Diff" for metric in DIFF_METRICS]
)

SET_SCORE_RE = re.compile(r"(?P<winner_games>\d{1,2})-(?P<loser_games>\d{1,2})(?:\([^)]*\))?")
RETIREMENT_MARKERS = ("RET", "W/O", "WO", "DEF", "ABD", "ABN", "DEFAULT", "WALKOVER")


def _to_float(value) -> float:
    try:
        if value is None or pd.isna(value) or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.notna(numerator) and pd.notna(denominator) and denominator > 0:
        return float(numerator / denominator)
    return np.nan


def _laplace(successes: float, total: float, alpha: float = 3.0, prior: float = 0.5) -> float:
    if total <= 0:
        return np.nan
    return float((successes + prior * alpha) / (total + alpha))


def parse_score_summary(score, best_of=None) -> Dict[str, object]:
    text = "" if score is None or pd.isna(score) else str(score).strip()
    upper = text.upper()
    retired = any(marker in upper for marker in RETIREMENT_MARKERS)
    set_scores = []
    for match in SET_SCORE_RE.finditer(upper):
        winner_games = int(match.group("winner_games"))
        loser_games = int(match.group("loser_games"))
        if winner_games == loser_games:
            continue
        set_scores.append(
            {
                "winner_games": winner_games,
                "loser_games": loser_games,
                "winner_set_won": winner_games > loser_games,
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
        "winner_tiebreaks": sum(1 for item in set_scores if item["has_tiebreak"] and item["winner_set_won"]),
        "loser_tiebreaks": sum(1 for item in set_scores if item["has_tiebreak"] and not item["winner_set_won"]),
        "straight_sets": valid and loser_sets == 0,
        "deciding_set": valid and winner_sets == sets_to_win and loser_sets == sets_to_win - 1,
    }


def apply_round_offsets(matches: pd.DataFrame, ref) -> pd.DataFrame:
    if matches.empty or "date" not in matches.columns:
        return matches.copy()
    out = matches.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    ref_ts = pd.Timestamp(ref)

    def _infer(row):
        if pd.isna(row.get("date")):
            return row.get("date")
        level = str(row.get("level", "")) if pd.notna(row.get("level")) else ""
        event = str(row.get("event", "")) if pd.notna(row.get("event")) else ""
        draw = infer_draw_size(event, level)
        round_code = str(row.get("round", ""))
        return row["date"] + pd.Timedelta(
            days=get_round_day_offset(level, draw, round_code, tourney_date=row["date"])
        )

    out["date"] = out.apply(_infer, axis=1)
    return out[out["date"] < ref_ts].copy()


def _row_to_match(row: pd.Series) -> Dict[str, object]:
    player_won = str(row.get("result", "")).upper() == "W"
    score = parse_score_summary(row.get("score"), row.get("max_sets"))
    if player_won:
        sets_won = score["winner_sets"]
        sets_lost = score["loser_sets"]
        games_won = score["winner_games"]
        games_lost = score["loser_games"]
        tb_won = score["winner_tiebreaks"]
        tb_lost = score["loser_tiebreaks"]
    else:
        sets_won = score["loser_sets"]
        sets_lost = score["winner_sets"]
        games_won = score["loser_games"]
        games_lost = score["winner_games"]
        tb_won = score["loser_tiebreaks"]
        tb_lost = score["winner_tiebreaks"]

    serve_points = _to_float(row.get("serve_points"))
    first_serves_in = _to_float(row.get("first_serves_in"))
    first_won = _to_float(row.get("first_serve_won"))
    second_won = _to_float(row.get("second_serve_won"))
    opp_serve_points = _to_float(row.get("opp_serve_points"))
    opp_first_won = _to_float(row.get("opp_first_serve_won"))
    opp_second_won = _to_float(row.get("opp_second_serve_won"))
    opp_bp_faced = _to_float(row.get("opp_bp_faced"))
    opp_bp_saved = _to_float(row.get("opp_bp_saved"))

    return_points_won = opp_serve_points - (opp_first_won + opp_second_won)
    bp_converted = opp_bp_faced - opp_bp_saved
    stats_valid = pd.notna(serve_points) and serve_points > 0 and pd.notna(opp_serve_points) and opp_serve_points > 0
    second_serve_attempts = serve_points - first_serves_in if pd.notna(serve_points) and pd.notna(first_serves_in) else np.nan

    return {
        "date": row.get("date"),
        "score_valid": bool(score["valid"]),
        "stats_valid": bool(stats_valid),
        "sets_won": float(sets_won),
        "sets_lost": float(sets_lost),
        "games_won": float(games_won),
        "games_lost": float(games_lost),
        "straight_set_win": bool(player_won and score["straight_sets"]),
        "deciding_set": bool(score["deciding_set"]),
        "deciding_set_win": bool(player_won and score["deciding_set"]),
        "tb_won": float(tb_won),
        "tb_lost": float(tb_lost),
        "serve_points": serve_points,
        "service_points_won": first_won + second_won,
        "return_points": opp_serve_points,
        "return_points_won": return_points_won,
        "first_serves_in": first_serves_in,
        "first_won": first_won,
        "second_won": second_won,
        "__second_serve_attempts": max(second_serve_attempts, 0.0) if pd.notna(second_serve_attempts) else np.nan,
        "aces": _to_float(row.get("aces")),
        "double_faults": _to_float(row.get("double_faults")),
        "bp_saved": _to_float(row.get("bp_saved")),
        "bp_faced": _to_float(row.get("bp_faced")),
        "bp_converted": bp_converted,
        "bp_chances": opp_bp_faced,
        "minutes": _to_float(row.get("minutes")),
    }


def _recent(items: List[Dict[str, object]], max_matches: int, ref=None, days: int | None = None) -> List[Dict[str, object]]:
    candidates = items
    if ref is not None and days is not None:
        cutoff = pd.Timestamp(ref) - pd.Timedelta(days=days)
        candidates = [item for item in candidates if cutoff <= item["date"] < pd.Timestamp(ref)]
    return sorted(candidates, key=lambda item: item["date"], reverse=True)[:max_matches]


def _sum_rate(items: Iterable[Dict[str, object]], numerator: str, denominator: str) -> float:
    num = 0.0
    den = 0.0
    for item in items:
        item_num = _to_float(item.get(numerator))
        item_den = _to_float(item.get(denominator))
        if pd.notna(item_num) and pd.notna(item_den) and item_den > 0:
            num += item_num
            den += item_den
    return _safe_ratio(num, den)


def _mean(items: Iterable[Dict[str, object]], key: str) -> float:
    vals = [_to_float(item.get(key)) for item in items]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def player_performance_features(matches: pd.DataFrame, ref, prefix: str) -> Dict[str, float]:
    if matches.empty:
        history: List[Dict[str, object]] = []
    else:
        prepared = matches.dropna(subset=["date"]).copy()
        history = [_row_to_match(row) for _, row in prepared.iterrows()]

    last10 = _recent(history, 10)
    score_last10 = [item for item in last10 if item.get("score_valid")]
    stats_last10 = [item for item in last10 if item.get("stats_valid")]
    score_90d = [item for item in _recent(history, 80, ref=ref, days=90) if item.get("score_valid")]

    sets_won = sum(item["sets_won"] for item in score_last10)
    sets_total = sets_won + sum(item["sets_lost"] for item in score_last10)
    games_won = sum(item["games_won"] for item in score_last10)
    games_total = games_won + sum(item["games_lost"] for item in score_last10)
    games_won_90d = sum(item["games_won"] for item in score_90d)
    games_total_90d = games_won_90d + sum(item["games_lost"] for item in score_90d)
    deciding_total = sum(1 for item in history if item.get("deciding_set") and item.get("score_valid"))
    deciding_wins = sum(1 for item in history if item.get("deciding_set_win") and item.get("score_valid"))
    tb_wins = sum(item.get("tb_won", 0.0) for item in history if item.get("score_valid"))
    tb_losses = sum(item.get("tb_lost", 0.0) for item in history if item.get("score_valid"))

    return {
        f"{prefix}_Score_Matches_Last10": float(len(score_last10)),
        f"{prefix}_Stat_Matches_Last10": float(len(stats_last10)),
        f"{prefix}_Set_WinRate_Last10": _laplace(sets_won, sets_total),
        f"{prefix}_Game_WinRate_Last10": _laplace(games_won, games_total),
        f"{prefix}_Game_WinRate_90d": _laplace(games_won_90d, games_total_90d),
        f"{prefix}_Straight_Set_WinRate_Last10": _laplace(
            sum(1 for item in score_last10 if item.get("straight_set_win")),
            len(score_last10),
        ),
        f"{prefix}_Deciding_Set_WinRate_Career": _laplace(deciding_wins, deciding_total),
        f"{prefix}_Tiebreak_WinRate_Career": _laplace(tb_wins, tb_wins + tb_losses),
        f"{prefix}_Service_Points_Won_Last10": _sum_rate(stats_last10, "service_points_won", "serve_points"),
        f"{prefix}_Return_Points_Won_Last10": _sum_rate(stats_last10, "return_points_won", "return_points"),
        f"{prefix}_First_Serve_In_Last10": _sum_rate(stats_last10, "first_serves_in", "serve_points"),
        f"{prefix}_First_Serve_Won_Last10": _sum_rate(stats_last10, "first_won", "first_serves_in"),
        f"{prefix}_Second_Serve_Won_Last10": _sum_rate(stats_last10, "second_won", "__second_serve_attempts"),
        f"{prefix}_Ace_Rate_Last10": _sum_rate(stats_last10, "aces", "serve_points"),
        f"{prefix}_Double_Fault_Rate_Last10": _sum_rate(stats_last10, "double_faults", "serve_points"),
        f"{prefix}_BP_Save_Rate_Last10": _sum_rate(stats_last10, "bp_saved", "bp_faced"),
        f"{prefix}_BP_Convert_Rate_Last10": _sum_rate(stats_last10, "bp_converted", "bp_chances"),
        f"{prefix}_Avg_Minutes_Last10": _mean(last10, "minutes"),
        f"{prefix}_Avg_Service_Points_Last10": _mean(stats_last10, "serve_points"),
    }


def build_match_performance_features(matches1: pd.DataFrame, matches2: pd.DataFrame, match_date) -> Dict[str, float]:
    ref = pd.Timestamp(match_date)
    p1_matches = apply_round_offsets(matches1, ref)
    p2_matches = apply_round_offsets(matches2, ref)
    features: Dict[str, float] = {}
    features.update(player_performance_features(p1_matches, ref, "P1"))
    features.update(player_performance_features(p2_matches, ref, "P2"))
    for metric in DIFF_METRICS:
        p1_value = features.get(f"P1_{metric}")
        p2_value = features.get(f"P2_{metric}")
        features[f"{metric}_Diff"] = (
            float(p1_value - p2_value)
            if pd.notna(p1_value) and pd.notna(p2_value)
            else np.nan
        )
    return features
