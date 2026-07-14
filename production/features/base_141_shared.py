"""Pure candidate semantics for the ordered ``base_141`` representation.

This module is the implementation candidate for
``base_141_shared@1.0.0``.  It deliberately performs no I/O and mutates no
caller-owned objects.  Historical preprocessing and live serving may opt in
through their adapters, but the candidate is *not* the active semantics for
any promoted artifact.

The kernel accepts player-perspective match observations and owns the disputed
as-of formulas: activity and set windows, form, ranking movement/volatility,
surface transition behavior, and P1-oriented head-to-head summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
import math
import re
from statistics import pstdev
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd


SEMANTICS_ID = "base_141_shared@1.0.0"
TIME_GRANULARITY = "calendar_day"
LAPLACE_ALPHA = 3.0
FORM_DECAY_DAYS = 15.0
UNSCORED_MATCH_SET_ESTIMATE = 2.5
FIRST_HISTORY_DAYS_SINCE_DEFAULT = 60


class CanonicalDateError(ValueError):
    """Raised when a clock-bearing timestamp is used as an event-local date."""


@dataclass(frozen=True)
class MatchObservation:
    """A source-neutral, player-perspective historical match."""

    date: pd.Timestamp
    won: Optional[bool] = None
    surface: Optional[str] = None
    score: Optional[str] = None
    rank: Optional[float] = None
    opponent: Any = None
    # ``order`` is populated only from explicit chronological evidence.  A
    # DataFrame row position or a source-local ``match_num`` is not proof.
    order: Optional[float] = None
    order_scope: str = ""
    source_key: str = ""


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _timestamp(value: Any) -> Optional[pd.Timestamp]:
    if not _present(value):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        raise CanonicalDateError(
            "base_141_shared requires a timezone-naive canonical event-local "
            "date; keep timezone-aware kickoff values in match_start_at_utc "
            "lineage and pass canonical_match_date separately"
        )
    # Sackmann history is date-granular.  The candidate therefore compares
    # both adapters at calendar-day granularity and never lets a same-day row
    # become evidence merely because the live kickoff carries a clock time.
    return stamp.normalize()


def as_of_day(value: Any) -> pd.Timestamp:
    """Return the shared candidate's normalized calendar-day timestamp."""

    stamp = _timestamp(value)
    if stamp is None:
        raise ValueError(f"invalid candidate timestamp: {value!r}")
    return stamp


def _result(value: Any) -> Optional[bool]:
    if not _present(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().upper()
    if normalized in {"W", "WIN", "TRUE", "1"}:
        return True
    if normalized in {"L", "LOSS", "FALSE", "0"}:
        return False
    return None


def _rank(value: Any) -> Optional[float]:
    if not _present(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _finite(value: Any, default: float, *, minimum: Optional[float] = None) -> float:
    if _present(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            numeric = float("nan")
        if math.isfinite(numeric) and (minimum is None or numeric >= minimum):
            return numeric
    return float(default)


def normalize_player_snapshot(
    *,
    height: Any,
    age: Any,
    rank: Any,
    rank_points: Any,
    hand: Any,
    country: Any,
) -> dict[str, Any]:
    """Shared finite defaults for profile fields used by derived features."""

    normalized_rank = _rank(rank)
    rank_was_missing = normalized_rank is None
    if rank_was_missing:
        normalized_rank = 999.0
    points_default = 0.0 if rank_was_missing or normalized_rank >= 999 else 500.0
    normalized_hand = str(hand).strip().upper() if _present(hand) else "U"
    if normalized_hand not in {"R", "L", "U", "A"}:
        normalized_hand = "U"
    normalized_country = str(country).strip().upper() if _present(country) else "OTHER"
    if not normalized_country:
        normalized_country = "OTHER"
    return {
        "height": _finite(height, 180.0, minimum=1.0),
        "age": _finite(age, 25.0, minimum=0.0),
        "rank": float(normalized_rank),
        "rank_points": _finite(rank_points, points_default, minimum=0.0),
        "hand": normalized_hand,
        "country": normalized_country,
    }


def feature_default(feature_name: str) -> float:
    """Finite last-resort default shared by both candidate adapters."""

    if "Rank" in feature_name and "Points" not in feature_name:
        return 100.0
    if "Points" in feature_name:
        return 500.0
    if "Age" in feature_name:
        return 25.0
    if "Height" in feature_name:
        return 180.0
    if "WinRate" in feature_name:
        return 0.5
    return 0.0


def _order(value: Any) -> Optional[float]:
    if not _present(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _explicit_order(row: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """Return source chronology plus the scope where values are comparable."""

    order = _order(row.get("chronological_order"))
    if order is not None:
        scope = row.get("chronological_order_scope", "__global__")
        return order, str(scope or "__global__")
    round_order = _order(row.get("round_ord"))
    event_scope = row.get("order_scope", row.get("tourney_id", row.get("event")))
    if round_order is not None and _present(event_scope):
        return round_order, f"round:{str(event_scope).strip().casefold()}"
    chronology_flag = row.get("order_is_chronological", False)
    if isinstance(chronology_flag, str):
        chronology_is_proven = chronology_flag.strip().casefold() in {
            "1", "true", "yes", "y",
        }
    else:
        chronology_is_proven = _present(chronology_flag) and bool(chronology_flag)
    if chronology_is_proven:
        for name in ("order", "match_order", "match_num"):
            order = _order(row.get(name))
            if order is not None:
                scope = row.get("chronological_order_scope", "__global__")
                return order, str(scope or "__global__")
    return None, ""


def _source_key(row: Mapping[str, Any]) -> str:
    """Stable identity for deterministic storage, never chronological proof."""

    for name in ("source_key", "match_id"):
        value = row.get(name)
        if _present(value):
            return str(value).strip()
    fields = (
        row.get("event"),
        row.get("opponent", row.get("opp_id", row.get("opp_name"))),
        row.get("result", row.get("won")),
        row.get("surface"),
        row.get("round"),
        row.get("score"),
        row.get("rank"),
    )
    return "\x1f".join("" if not _present(value) else str(value) for value in fields)


def observations_from_records(
    records: Iterable[Mapping[str, Any]] | pd.DataFrame,
) -> tuple[MatchObservation, ...]:
    """Normalize TA-frame or historical-dict rows without applying an as-of.

    Supported aliases intentionally match the two existing adapters:
    ``result``/``won`` and ``opp_id``/``opponent``/``opp_name``.
    Invalid dates remain absent from the returned history rather than becoming
    maximally recent evidence. Timezone-aware dates fail closed because a
    kickoff instant cannot establish the canonical event-local calendar day.
    """

    if records is None:
        return ()
    if isinstance(records, pd.DataFrame):
        raw_records = records.to_dict("records")
    else:
        raw_records = list(records)

    normalized: list[MatchObservation] = []
    for row in raw_records:
        date = _timestamp(row.get("date"))
        if date is None:
            continue
        result_value = row.get("won") if "won" in row else row.get("result")
        opponent = row.get("opp_id")
        if not _present(opponent):
            opponent = row.get("opponent")
        if not _present(opponent):
            opponent = row.get("opp_name")
        surface = row.get("surface")
        score = row.get("score")
        order, order_scope = _explicit_order(row)
        normalized.append(
            MatchObservation(
                date=date,
                won=_result(result_value),
                surface=str(surface).strip().title() if _present(surface) else None,
                score=str(score).strip() if _present(score) else None,
                rank=_rank(row.get("rank")),
                opponent=opponent if _present(opponent) else None,
                order=order,
                order_scope=order_scope,
                source_key=_source_key(row),
            )
        )
    return tuple(normalized)


def _observation_sort_key(row: MatchObservation) -> tuple[Any, ...]:
    """Stable storage order; it does not invent chronology for tied dates."""

    return (
        row.date,
        row.order is not None,
        row.order if row.order is not None else 0.0,
        row.order_scope,
        row.source_key,
        "" if row.opponent is None else str(row.opponent),
        "" if row.surface is None else row.surface,
        "" if row.score is None else row.score,
        "" if row.won is None else str(int(row.won)),
        float("-inf") if row.rank is None else row.rank,
    )


def _day_groups(
    history: Sequence[MatchObservation],
) -> tuple[tuple[MatchObservation, ...], ...]:
    """Newest-first calendar-day groups in deterministic storage order."""

    ordered = sorted(history, key=_observation_sort_key, reverse=True)
    return tuple(
        tuple(rows)
        for _, rows in groupby(ordered, key=lambda row: row.date)
    )


def _proven_day_order(
    rows: Sequence[MatchObservation],
) -> Optional[tuple[MatchObservation, ...]]:
    """Newest-first rows only when every tie has unique order evidence."""

    if len(rows) <= 1:
        return tuple(rows)
    orders = [row.order for row in rows]
    scopes = {row.order_scope for row in rows}
    if (
        any(order is None for order in orders)
        or len(set(orders)) != len(orders)
        or len(scopes) != 1
    ):
        return None
    return tuple(sorted(rows, key=lambda row: float(row.order), reverse=True))


def _recent_complete_day_rows(
    history: Sequence[MatchObservation],
    limit: int,
) -> tuple[MatchObservation, ...]:
    """Take recent evidence without selecting an arbitrary part of a tied day.

    An explicitly ordered day can be truncated at ``limit``.  An ambiguous
    date-only group is included only when the whole group fits; otherwise the
    capped sample stops before that day.
    """

    selected: list[MatchObservation] = []
    for group in _day_groups(history):
        remaining = limit - len(selected)
        if remaining <= 0:
            break
        proven = _proven_day_order(group)
        if proven is not None:
            selected.extend(proven[:remaining])
            continue
        if len(group) > remaining:
            break
        selected.extend(group)
    return tuple(selected)


def history_before(
    history: Sequence[MatchObservation], as_of: datetime | pd.Timestamp
) -> tuple[MatchObservation, ...]:
    """Return valid observations strictly before ``as_of``, newest first."""

    ref = _timestamp(as_of)
    if ref is None:
        raise ValueError(f"invalid as_of timestamp: {as_of!r}")
    return tuple(
        sorted(
            (row for row in history if row.date < ref),
            key=_observation_sort_key,
            reverse=True,
        )
    )


def laplace(wins: int, total: int, alpha: float = LAPLACE_ALPHA) -> float:
    """Neutral-prior Laplace estimate: ``(wins + alpha/2)/(total+alpha)``."""

    if total < 0 or wins < 0 or wins > total:
        raise ValueError(f"invalid win count: wins={wins}, total={total}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return (float(wins) + alpha / 2.0) / (float(total) + alpha)


def _window(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    days: int,
) -> tuple[MatchObservation, ...]:
    """Canonical half-open lookback window ``[as_of-days, as_of)``."""

    if days <= 0:
        raise ValueError("days must be positive")
    ref = _timestamp(as_of)
    if ref is None:
        raise ValueError(f"invalid as_of timestamp: {as_of!r}")
    cutoff = ref - pd.Timedelta(days=days)
    return tuple(row for row in history_before(history, ref) if row.date >= cutoff)


def count_matches(
    history: Sequence[MatchObservation], as_of: datetime | pd.Timestamp, days: int
) -> int:
    return len(_window(history, as_of, days))


def _same_surface(left: Optional[str], right: Optional[str]) -> bool:
    return bool(left and right) and left.casefold() == right.casefold()


def count_surface_matches(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    surface: str,
    days: Optional[int] = None,
) -> int:
    rows = history_before(history, as_of) if days is None else _window(history, as_of, days)
    return sum(1 for row in rows if _same_surface(row.surface, surface))


def surface_win_rate(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    surface: str,
    days: int = 90,
) -> float:
    rows = [
        row for row in _window(history, as_of, days)
        if _same_surface(row.surface, surface) and row.won is not None
    ]
    return laplace(sum(row.won is True for row in rows), len(rows))


def recent_win_rate(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    max_matches: int = 10,
    max_days: int = 120,
) -> float:
    rows = [row for row in _window(history, as_of, max_days) if row.won is not None]
    rows = list(_recent_complete_day_rows(rows, max_matches))
    return laplace(sum(row.won is True for row in rows), len(rows))


def current_streak(history: Sequence[MatchObservation]) -> int:
    rows = [row for row in history if row.won is not None]
    first: Optional[bool] = None
    length = 0
    for group in _day_groups(rows):
        proven = _proven_day_order(group)
        if proven is None:
            outcomes = {row.won for row in group}
            # A mixed date-only day has no knowable final result.  It cannot
            # start or extend a streak without inventing a clock.
            if len(outcomes) != 1:
                break
            sequence = group
        else:
            sequence = proven
        for row in sequence:
            if first is None:
                first = row.won
            if row.won is not first:
                return length if first else -length
            length += 1
    if first is None:
        return 0
    return length if first else -length


def form_trend(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    days: int = 30,
) -> float:
    """Exponentially weighted win share using the legacy 15-day time constant.

    Every valid in-window result contributes.  In particular, one or two
    observations are evidence; the live legacy minimum-of-three reset is not
    part of the shared contract.
    """

    ref = _timestamp(as_of)
    if ref is None:
        raise ValueError(f"invalid as_of timestamp: {as_of!r}")
    rows = [row for row in _window(history, ref, days) if row.won is not None]
    if not rows:
        return 0.5
    weighted_wins = 0.0
    total_weight = 0.0
    for row in rows:
        age_days = (ref - row.date).total_seconds() / 86_400.0
        weight = math.exp(-age_days / FORM_DECAY_DAYS)
        total_weight += weight
        if row.won:
            weighted_wins += weight
    return weighted_wins / total_weight if total_weight else 0.5


_SET_TOKEN = re.compile(r"^\d{1,2}-\d{1,2}(?:\(\d+\))?$")
_NO_PLAY_SCORE = re.compile(r"^(?:W/?O|WALKOVER|DEF(?:AULT)?)$", re.IGNORECASE)


def set_count_from_score(score: Optional[str]) -> Optional[int]:
    """Return completed/started set tokens, or ``None`` when score is absent.

    Walkovers/defaults contribute zero sets.  A non-empty score with no
    parseable set token is treated as unscored so the caller can apply the
    explicit estimate rather than silently manufacturing zero workload.
    """

    if not score:
        return None
    value = str(score).strip()
    if _NO_PLAY_SCORE.fullmatch(value):
        return 0
    tokens = [token.strip(";,.") for token in value.split()]
    parsed = sum(bool(_SET_TOKEN.fullmatch(token)) for token in tokens)
    return parsed if parsed else None


def sets_played(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    days: int = 14,
    unscored_estimate: float = UNSCORED_MATCH_SET_ESTIMATE,
) -> float:
    total = 0.0
    for row in _window(history, as_of, days):
        parsed = set_count_from_score(row.score)
        total += float(unscored_estimate if parsed is None else parsed)
    return total


def _monday(value: pd.Timestamp) -> pd.Timestamp:
    return (value - pd.Timedelta(days=int(value.weekday()))).normalize()


def days_since_last_tournament(
    history: Sequence[MatchObservation], as_of: datetime | pd.Timestamp
) -> Optional[int]:
    """Days from ``as_of`` to the latest prior tournament-week Monday."""

    ref = _timestamp(as_of)
    if ref is None:
        raise ValueError(f"invalid as_of timestamp: {as_of!r}")
    current_week = _monday(ref)
    previous_weeks = {
        _monday(row.date)
        for row in history_before(history, ref)
        if _monday(row.date) < current_week
    }
    if not previous_weeks:
        return None
    return int((ref - max(previous_weeks)).days)


def _unambiguous_latest_rank(
    rows: Sequence[MatchObservation],
) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    ranked = [row for row in rows if row.rank is not None]
    groups = _day_groups(ranked)
    if not groups:
        return None, None
    latest = groups[0]
    proven = _proven_day_order(latest)
    if proven is not None:
        return proven[0].rank, proven[0].date
    ranks = {float(row.rank) for row in latest}
    if len(ranks) == 1:
        return ranks.pop(), latest[0].date
    # Conflicting ranks on a date-only tie have no source-neutral anchor.
    return None, latest[0].date


def rank_change(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    days: int,
    rank_as_of: Optional[float],
) -> float:
    """``rank_at_or_before_cutoff - rank_as_of_ref``.

    Positive values therefore mean the player's ranking improved.  The anchor
    must be no more than half a lookback window older than the cutoff, matching
    the historical lineage's explicit staleness bound.
    """

    ref = _timestamp(as_of)
    if ref is None:
        raise ValueError(f"invalid as_of timestamp: {as_of!r}")
    current = _rank(rank_as_of)
    ranked = [row for row in history_before(history, ref) if row.rank is not None]
    if current is None:
        current, _ = _unambiguous_latest_rank(ranked)
    if current is None:
        return 0.0
    cutoff = ref - pd.Timedelta(days=days)
    anchors = [row for row in ranked if row.date <= cutoff]
    if not anchors:
        return 0.0
    anchor_rank, anchor_date = _unambiguous_latest_rank(anchors)
    if anchor_rank is None or anchor_date is None:
        return 0.0
    if (cutoff - anchor_date).total_seconds() > (days / 2.0) * 86_400.0:
        return 0.0
    return float(anchor_rank) - float(current)


def rank_volatility(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    days: int = 90,
) -> float:
    """Population standard deviation of at least three in-window ranks."""

    ranks = [row.rank for row in _window(history, as_of, days) if row.rank is not None]
    return float(pstdev(ranks)) if len(ranks) >= 3 else 0.0


def last_surface(
    history: Sequence[MatchObservation], as_of: datetime | pd.Timestamp
) -> Optional[str]:
    for group in _day_groups(history_before(history, as_of)):
        with_surface = tuple(row for row in group if row.surface)
        if not with_surface:
            continue
        proven = _proven_day_order(group)
        if proven is not None:
            for row in proven:
                if row.surface:
                    return row.surface
        normalized = {str(row.surface).casefold(): row.surface for row in with_surface}
        if len(normalized) == 1:
            return next(iter(normalized.values()))
        # Multiple surfaces on an unordered day have no knowable "last" one.
        return None
    return None


def surface_transition_flag(
    p1_last_surface: Optional[str],
    p2_last_surface: Optional[str],
    current_surface: str,
) -> int:
    """Flag a known transition; no history is neutral rather than ``Hard``."""

    known = (p1_last_surface, p2_last_surface)
    return int(any(value and not _same_surface(value, current_surface) for value in known))


def player_temporal_features(
    history: Sequence[MatchObservation],
    as_of: datetime | pd.Timestamp,
    surface: str,
    *,
    rank_as_of: Optional[float],
    surface_experience: Optional[int] = None,
) -> dict[str, Any]:
    """Calculate the candidate's disputed player-temporal fields."""

    prior = history_before(history, as_of)
    observed_days_since = days_since_last_tournament(prior, as_of)
    days_since = (
        FIRST_HISTORY_DAYS_SINCE_DEFAULT
        if observed_days_since is None else observed_days_since
    )
    return {
        "matches_14d": count_matches(prior, as_of, 14),
        "matches_30d": count_matches(prior, as_of, 30),
        "matches_90d": count_matches(prior, as_of, 90),
        "surface_matches_30d": count_surface_matches(prior, as_of, surface, 30),
        "surface_matches_90d": count_surface_matches(prior, as_of, surface, 90),
        "surface_experience": (
            int(surface_experience)
            if surface_experience is not None
            else count_surface_matches(prior, as_of, surface, None)
        ),
        "surface_winrate_90d": surface_win_rate(prior, as_of, surface, 90),
        "winrate_last10_120d": recent_win_rate(prior, as_of, 10, 120),
        "streak": current_streak(prior),
        "form_trend_30d": form_trend(prior, as_of, 30),
        "days_since_last": days_since,
        # Unknown history is neutral, not evidence that the player is rusty.
        "rust_flag": int(observed_days_since is not None and days_since > 21),
        "sets_14d": sets_played(prior, as_of, 14),
        "last_surface": last_surface(prior, as_of),
        "rank_change_30d": rank_change(prior, as_of, 30, rank_as_of),
        "rank_change_90d": rank_change(prior, as_of, 90, rank_as_of),
        "rank_volatility_90d": rank_volatility(prior, as_of, 90),
    }


def h2h_features_from_counts(
    *,
    total: int,
    p1_wins: int,
    recent_p1_results: Sequence[bool],
) -> dict[str, float | int]:
    """Build P1-oriented H2H fields from career counts and newest-first results."""

    if total < 0 or p1_wins < 0 or p1_wins > total:
        raise ValueError(f"invalid H2H counts: p1_wins={p1_wins}, total={total}")
    recent = tuple(bool(value) for value in recent_p1_results[:3])
    recent_advantage = laplace(sum(recent), len(recent)) - 0.5
    return {
        "H2H_Total_Matches": int(total),
        "H2H_P1_Wins": int(p1_wins),
        "H2H_P2_Wins": int(total - p1_wins),
        "H2H_P1_WinRate": float(laplace(p1_wins, total)),
        "H2H_Recent_P1_Advantage": float(recent_advantage),
    }


def h2h_features_from_history(
    history: Sequence[MatchObservation],
    opponent: Any,
    as_of: datetime | pd.Timestamp,
) -> dict[str, float | int]:
    rows = [
        row for row in history_before(history, as_of)
        if row.opponent == opponent and row.won is not None
    ]
    recent = _recent_complete_day_rows(rows, 3)
    return h2h_features_from_counts(
        total=len(rows),
        p1_wins=sum(row.won is True for row in rows),
        recent_p1_results=[bool(row.won) for row in recent],
    )


def h2h_features_from_frame(
    frame: pd.DataFrame,
    as_of: datetime | pd.Timestamp,
) -> dict[str, float | int]:
    """H2H summary for a frame already filtered to P1-perspective meetings."""

    rows = [row for row in history_before(observations_from_records(frame), as_of) if row.won is not None]
    recent = _recent_complete_day_rows(rows, 3)
    return h2h_features_from_counts(
        total=len(rows),
        p1_wins=sum(row.won is True for row in rows),
        recent_p1_results=[bool(row.won) for row in recent],
    )
