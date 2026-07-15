#!/usr/bin/env python3
"""
Auto-settle pending predictions by checking Tennis Abstract match history.

For each unsettled row in prediction_log.csv:
  1. Fetch p1's recent matches from TA
  2. Look for a completed match against p2 on or after the logged match_date
  3. If found, record the winner, score, and compute model_correct / market_correct

Usage:
    python auto_settle.py           # check and settle all pending
    python auto_settle.py --dry-run # show what would be settled without writing
    python auto_settle.py --stats   # show accuracy stats only (no settling)
"""

import argparse
import json
import math
import sys
import os
import re
import time
from collections import Counter
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scraping"))

from ta_scraper import TennisAbstractScraper
from features.ta_feature_calculator import TAFeatureCalculator
from utils.bet_tracker import BetTracker
from audit_logger import SETTLEMENT_AUDIT_LOG_PATH, log_settlement_event, upsert_run_history
from logging_utils import (
    canonicalize_live_event_key,
    make_run_id,
    normalize_name,
    normalize_text,
    utc_now,
)
from prediction_logger import upgrade_prediction_log
from settlement_attribution import (
    build_auto_result_evidence,
    load_feature_attribution_evidence,
    load_verified_prediction_log,
    prediction_match_supports_exact_attribution,
)

LOG_PATH = Path(__file__).parent / "prediction_log.csv"
DEFAULT_RATE_LIMIT_DELAY = 8.0
# Conservative production defaults. Hourly prediction capture and result
# settlement are separate concerns; the wider grace period avoids treating a
# delayed/unfinished listing as final and the cap keeps Tennis Abstract pacing
# predictable. Deeper backlog passes require deliberate CLI overrides.
DEFAULT_MIN_SETTLEMENT_AGE_HOURS = 18.0
DEFAULT_MAX_CANDIDATES = 75
DEFAULT_MAX_RATE_LIMITS = 5
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 120.0
DEFAULT_RETRY_BACKOFF_HOURS = 18.0
MIN_SETTLEMENT_SCORE = 65
AMBIGUITY_MARGIN = 6

SCRAPER = TennisAbstractScraper(rate_limit_delay=DEFAULT_RATE_LIMIT_DELAY)
IDENTITY_TERMINAL_STATUSES = {'identity_conflict', 'superseded_identity'}


def _identity_signature(row: pd.Series) -> tuple[str, ...]:
    """Comparable oriented metadata for duplicate operational match UIDs."""
    event_key = normalize_text(row.get('identity_event_key'))
    if not event_key:
        event_key = canonicalize_live_event_key(row.get('tournament'))
    return (
        normalize_name(row.get('p1')),
        normalize_name(row.get('p2')),
        normalize_text(row.get('match_date')),
        event_key,
        normalize_text(row.get('round')),
        normalize_text(row.get('surface')),
    )


def _settlement_uid_gate(df: pd.DataFrame, pending: pd.DataFrame) -> tuple[
    pd.DataFrame, dict[str, int]
]:
    """Block unsafe UID groups and attempt compatible duplicates once.

    Durable hydration can temporarily expose more than one operational row for
    a match UID.  A tombstone or existing terminal result on any member blocks
    the whole UID.  Compatible pending duplicates consume one TA candidate;
    inconsistent metadata is marked as an identity conflict and fails closed.
    """
    counts = {
        'identity_terminal_uid': 0,
        'already_settled_uid': 0,
        'duplicate_identity_conflict': 0,
        'compatible_duplicate_uid': 0,
    }
    if pending.empty or 'match_uid' not in df.columns:
        return pending, counts

    uid = df['match_uid'].fillna('').astype(str).str.strip()
    pending_uid = pending.get(
        'match_uid', pd.Series('', index=pending.index)
    ).fillna('').astype(str).str.strip()
    status = df.get(
        'record_status', pd.Series('', index=df.index)
    ).fillna('').astype(str).str.strip().str.lower()
    winner = pd.to_numeric(
        df.get('actual_winner', pd.Series('', index=df.index)),
        errors='coerce',
    )
    pending_uid_values = {
        match_uid for match_uid in pending_uid if match_uid
    }

    identity_blocked = {
        match_uid for match_uid in uid[status.isin(IDENTITY_TERMINAL_STATUSES)]
        if match_uid and match_uid in pending_uid_values
    }
    settled_blocked = {
        match_uid for match_uid in uid[winner.isin([1, 2])]
        if match_uid and match_uid in pending_uid_values
    }
    counts['identity_terminal_uid'] = len(identity_blocked)
    counts['already_settled_uid'] = len(settled_blocked - identity_blocked)
    blocked = identity_blocked | settled_blocked
    if blocked:
        pending = pending.loc[~pending_uid.isin(blocked)].copy()

    pending_uid = pending.get(
        'match_uid', pd.Series('', index=pending.index)
    ).fillna('').astype(str).str.strip()
    duplicate_uids = pending_uid[pending_uid.ne('')].value_counts()
    duplicate_uids = set(duplicate_uids[duplicate_uids.gt(1)].index)
    incompatible_uids: set[str] = set()
    for match_uid in duplicate_uids:
        group_indices = pending.index[pending_uid.eq(match_uid)]
        signatures = {
            _identity_signature(pending.loc[index]) for index in group_indices
        }
        if len(signatures) == 1:
            counts['compatible_duplicate_uid'] += 1
            continue
        incompatible_uids.add(match_uid)
        counts['duplicate_identity_conflict'] += 1
        all_indices = df.index[uid.eq(match_uid)]
        df.loc[all_indices, 'record_status'] = 'identity_conflict'
        df.loc[all_indices, 'identity_status'] = 'conflict'
        df.loc[all_indices, 'identity_conflict_fields'] = (
            'duplicate_match_uid_metadata'
        )
        df.loc[all_indices, 'features_complete'] = False

    if incompatible_uids:
        pending = pending.loc[~pending_uid.isin(incompatible_uids)].copy()
    return pending, counts


def _dedupe_pending_match_uids(pending: pd.DataFrame) -> pd.DataFrame:
    """Keep one settlement candidate per nonblank compatible match UID."""
    if pending.empty or 'match_uid' not in pending.columns:
        return pending
    pending_uid = pending['match_uid'].fillna('').astype(str).str.strip()
    keep = pending_uid.eq('') | ~pending_uid.duplicated(keep='first')
    return pending.loc[keep].copy()


def _compatible_pending_group_indices(
    df: pd.DataFrame, selected_index: int
) -> list[int]:
    """Return every compatible pending row represented by one candidate."""
    selected = df.loc[selected_index]
    raw_match_uid = selected.get('match_uid')
    match_uid = (
        '' if raw_match_uid is None or pd.isna(raw_match_uid)
        else str(raw_match_uid).strip()
    )
    if not match_uid or 'match_uid' not in df.columns:
        return [selected_index]
    uid = df['match_uid'].fillna('').astype(str).str.strip()
    winner = pd.to_numeric(df['actual_winner'], errors='coerce')
    status = df.get(
        'record_status', pd.Series('', index=df.index)
    ).fillna('').astype(str).str.strip().str.lower()
    indices = list(df.index[
        uid.eq(match_uid)
        & winner.isna()
        & ~status.isin(IDENTITY_TERMINAL_STATUSES)
    ])
    if not indices:
        return [selected_index]
    signatures = {_identity_signature(df.loc[index]) for index in indices}
    return indices if len(signatures) == 1 else [selected_index]


def _prediction_correct(probability, actual_winner: int):
    """Return 0/1 correctness for a finite probability, otherwise no score."""
    try:
        value = float(probability)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None
    return int(
        (actual_winner == 1 and value > 0.5)
        or (actual_winner == 2 and value < 0.5)
    )


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

def _last_name(name: str) -> str:
    """Extract last name, lowercased."""
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def _names_match(opp_name: str, candidate: str) -> bool:
    """
    Check if TA opponent name matches a logged player name.
    TA opp_name is often 'FirstLast' or 'F. Last' or just 'Last'.
    Candidate is the full name from Bovada e.g. 'Luca Van Assche'.
    """
    opp = opp_name.lower().strip()
    cand = candidate.lower().strip()

    # Exact
    if opp == cand:
        return True

    # Last name match (both)
    if _last_name(opp) == _last_name(cand) and _last_name(cand):
        return True

    # TA sometimes stores 'Last' only — if opp is a single token, check last name
    if " " not in opp and opp == _last_name(cand):
        return True

    # TA 'F. Last' format
    m = re.match(r"^([a-z])\.?\s+(.+)$", opp)
    if m:
        first_initial = m.group(1)
        last = m.group(2).strip()
        cand_parts = cand.split()
        if cand_parts and cand_parts[0][0] == first_initial and last == _last_name(cand):
            return True

    return False


_NAME_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv'}

def _strip_suffixes(name: str) -> str:
    """Strip name suffixes (Jr, Sr, II, etc.) before slug derivation."""
    parts = name.strip().split()
    clean = [p for p in parts if p.lower().rstrip('.') not in _NAME_SUFFIXES]
    return ' '.join(clean) if len(clean) >= 2 else name


def _resolve_slug(player_name: str, calc: TAFeatureCalculator) -> str:
    """Get TA slug for a player name via mapping or derivation."""
    name_lower = player_name.lower().strip()
    # Check player mapping
    if name_lower in calc.player_slug_map:
        return calc.player_slug_map[name_lower]
    # Strip Jr/Sr/II/III suffixes and try again
    clean = _strip_suffixes(player_name)
    if clean != player_name:
        clean_lower = clean.lower().strip()
        if clean_lower in calc.player_slug_map:
            return calc.player_slug_map[clean_lower]
        return TennisAbstractScraper.name_to_slug(clean)
    return TennisAbstractScraper.name_to_slug(player_name)


# ---------------------------------------------------------------------------
# Match identity scoring
# ---------------------------------------------------------------------------

def _clean_optional(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_text(value) -> str:
    text = _clean_optional(value).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_set(value) -> set[str]:
    stop = {
        "atp",
        "wta",
        "itf",
        "men",
        "mens",
        "women",
        "womens",
        "singles",
        "open",
        "cup",
        "challenger",
        "french",
    }
    return {tok for tok in _normalize_text(value).split() if len(tok) > 2 and tok not in stop}


def _event_similarity(expected: str, found: str) -> float | None:
    expected_clean = _normalize_text(expected)
    found_clean = _normalize_text(found)
    if not expected_clean or not found_clean:
        return None
    expected_tokens = _token_set(expected_clean)
    found_tokens = _token_set(found_clean)
    token_score = 0.0
    if expected_tokens and found_tokens:
        token_score = len(expected_tokens & found_tokens) / len(expected_tokens | found_tokens)
    text_score = SequenceMatcher(None, expected_clean, found_clean).ratio()
    return max(token_score, text_score)


def _normalize_round(value: str) -> str:
    text = _clean_optional(value).upper().replace(" ", "")
    aliases = {
        "ROUND128": "R128",
        "ROUND64": "R64",
        "ROUND32": "R32",
        "ROUND16": "R16",
        "1R": "R128",
        "2R": "R64",
        "3R": "R32",
        "4R": "R16",
        "QUARTERFINAL": "QF",
        "QUARTERFINALS": "QF",
        "SEMIFINAL": "SF",
        "SEMIFINALS": "SF",
        "FINAL": "F",
    }
    return aliases.get(text, text)


def _normalize_surface(value: str) -> str:
    text = _clean_optional(value).strip().title()
    aliases = {"Indoor Hard": "Hard"}
    return aliases.get(text, text)


def _score_date_diff(diff_days: float | None) -> int:
    if diff_days is None:
        return 0
    if diff_days <= 1:
        return 25
    if diff_days <= 7:
        return 20
    if diff_days <= 14:
        return 12
    if diff_days <= 21:
        return 4
    return -18


def _build_settlement_context(
    *,
    tournament: str = "",
    round_code: str = "",
    surface: str = "",
) -> dict:
    return {
        "tournament": _clean_optional(tournament),
        "round": _normalize_round(round_code),
        "surface": _normalize_surface(surface),
    }


def _score_settlement_candidate(candidate: pd.Series, match_date, context: dict) -> tuple[int, dict]:
    """
    Score a same-opponent TA result against the logged prediction metadata.

    TA dates are tournament start dates in this table, not exact match dates, so
    the score intentionally combines a wide date signal with tournament,
    surface, and round evidence instead of requiring exact date equality.
    """
    score = 50
    evidence: dict[str, object] = {}

    found_date = pd.to_datetime(candidate.get("date"), errors="coerce")
    diff_days = None
    if pd.notna(found_date) and pd.notna(match_date):
        diff_days = abs((found_date.normalize() - match_date.normalize()).days)
    evidence["date_diff_days"] = diff_days
    score += _score_date_diff(diff_days)

    expected_event = context.get("tournament", "")
    found_event = _clean_optional(candidate.get("event", ""))
    event_similarity = _event_similarity(expected_event, found_event)
    evidence["event_similarity"] = round(event_similarity, 3) if event_similarity is not None else None
    if event_similarity is not None:
        if event_similarity >= 0.72:
            score += 22
        elif event_similarity >= 0.45:
            score += 10
        elif expected_event and found_event:
            score -= 14

    expected_surface = context.get("surface", "")
    found_surface = _normalize_surface(candidate.get("surface", ""))
    evidence["surface_match"] = None
    if expected_surface:
        evidence["surface_match"] = found_surface == expected_surface
        score += 10 if found_surface == expected_surface else -12

    expected_round = context.get("round", "")
    found_round = _normalize_round(candidate.get("round", ""))
    evidence["round_match"] = None
    if expected_round:
        evidence["round_match"] = found_round == expected_round
        score += 12 if found_round == expected_round else -6

    evidence["ta_event"] = found_event
    evidence["ta_surface"] = found_surface
    evidence["ta_round"] = found_round
    evidence["score"] = score
    return score, evidence


def _select_best_settlement_candidate(
    found: pd.DataFrame,
    match_date,
    context: dict,
) -> tuple[pd.Series | None, str, dict]:
    scored = []
    for _, candidate in found.iterrows():
        score, evidence = _score_settlement_candidate(candidate, match_date, context)
        scored.append((score, evidence, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None, "opponent_not_found", {}

    top_score, top_evidence, top_row = scored[0]
    if top_score < MIN_SETTLEMENT_SCORE:
        return None, "low_confidence_match", {
            "best_score": top_score,
            "best_evidence": top_evidence,
            "candidates": len(scored),
        }

    if len(scored) > 1:
        second_score, second_evidence, _ = scored[1]
        if top_score - second_score <= AMBIGUITY_MARGIN:
            return None, "ambiguous_match", {
                "best_score": top_score,
                "second_score": second_score,
                "best_evidence": top_evidence,
                "second_evidence": second_evidence,
                "candidates": len(scored),
            }

    top_evidence["candidates"] = len(scored)
    return top_row, "matched", top_evidence


def _parse_match_start_time(match_start_time: str, match_date: str = "", now: datetime | None = None):
    now = now or datetime.now()
    text = _clean_optional(match_start_time)
    for fmt in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p", "%m/%d/%y", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            if "%I:%M %p" not in fmt:
                parsed = parsed.replace(hour=12, minute=0)
            return parsed
        except ValueError:
            pass

    time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", text, re.I)
    parsed_time = None
    if time_match:
        try:
            parsed_time = datetime.strptime(time_match.group(1).upper(), "%I:%M %p")
        except ValueError:
            parsed_time = None

    lower = text.lower()
    if lower.startswith("today") and parsed_time:
        return now.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0, microsecond=0)
    if lower.startswith("tomorrow") and parsed_time:
        base = now + timedelta(days=1)
        return base.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0, microsecond=0)

    parsed_date = pd.to_datetime(match_date, errors="coerce")
    if pd.notna(parsed_date):
        fallback = parsed_date.to_pydatetime()
        if parsed_time:
            return fallback.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0, microsecond=0)
        return fallback.replace(hour=12, minute=0, second=0, microsecond=0)
    return None


def _parse_exact_match_start_utc(row: pd.Series):
    """Return the best immutable match start as an aware UTC timestamp.

    ``latest_match_start_at_utc`` is the dashboard's corrected start clock.
    Older rows may only have the original exact UTC field, while legacy rows
    fall back to the Eastern display-string parser below.
    """
    for column in ("latest_match_start_at_utc", "match_start_at_utc"):
        value = _clean_optional(row.get(column, ""))
        if not value:
            continue
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed
    return None


def _effective_settlement_match_date(row: pd.Series) -> str:
    """Best corrected calendar date for every settlement decision.

    ``match_date`` is immutable opening lineage. Odds refreshes can correct a
    stale Bovada date into ``latest_match_date``; ordering already honored that
    correction, so result-source selection and auditing must use it too.
    """
    return (
        _clean_optional(row.get("latest_match_date"))
        or _clean_optional(row.get("match_date"))
    )


def _is_old_enough_to_settle(row: pd.Series, min_age_hours: float, now: datetime | None = None) -> tuple[bool, str]:
    exact_start = _parse_exact_match_start_utc(row)
    if exact_start is not None:
        reference = pd.Timestamp(now if now is not None else utc_now())
        if reference.tzinfo is None:
            reference = reference.tz_localize("UTC")
        else:
            reference = reference.tz_convert("UTC")
        age_hours = (reference - exact_start).total_seconds() / 3600
        if age_hours < min_age_hours:
            return False, (
                f"match_start_age_hours={age_hours:.1f} < "
                f"min_age_hours={min_age_hours:.1f}"
            )
        return True, ""

    # Bovada start times are US/Eastern display strings; comparing them naive
    # against the runner's local clock made eligibility depend on which
    # machine ran settlement (UTC cloud vs local laptop disagreed).
    from zoneinfo import ZoneInfo
    if now is None:
        now = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    else:
        reference = pd.Timestamp(now)
        if reference.tzinfo is not None:
            reference = reference.tz_convert("America/New_York").tz_localize(None)
        now = reference.to_pydatetime()
    start_dt = _parse_match_start_time(
        str(row.get("match_start_time", "")),
        _effective_settlement_match_date(row),
        now=now,
    )
    if start_dt is None:
        return True, ""
    age_hours = (now - start_dt).total_seconds() / 3600
    if age_hours < min_age_hours:
        return False, f"match_start_age_hours={age_hours:.1f} < min_age_hours={min_age_hours:.1f}"
    return True, ""


def _prioritize_tracked_pending_matches(
    pending: pd.DataFrame,
    pending_bets: pd.DataFrame,
) -> pd.DataFrame:
    """Mark prediction rows that can settle currently reserved paper capital.

    Direct match UIDs are preferred. A related UID counts only when the
    prediction logger already classified the relationship as a canonical
    alias; conflict or merely-related metadata must never broaden settlement.
    """
    prioritized = pending.copy()
    prioritized["_tracked_bet_priority"] = False
    if prioritized.empty or pending_bets.empty:
        return prioritized

    bet_uids = {
        str(value).strip()
        for value in pending_bets.get(
            "match_uid", pd.Series(dtype=object)
        ).dropna()
        if str(value).strip()
    }
    if not bet_uids:
        return prioritized

    direct = prioritized.get(
        "match_uid", pd.Series("", index=prioritized.index)
    ).fillna("").astype(str).str.strip().isin(bet_uids)

    def _has_safe_alias(row: pd.Series) -> bool:
        if _clean_optional(row.get("identity_status")).lower() != "canonical_alias":
            return False
        aliases = {
            value.strip()
            for value in _clean_optional(
                row.get("identity_related_match_uid")
            ).split("|")
            if value.strip()
        }
        return bool(aliases & bet_uids)

    alias = prioritized.apply(_has_safe_alias, axis=1)
    prioritized["_tracked_bet_priority"] = direct | alias
    return prioritized


def _order_settlement_candidates(
    pending: pd.DataFrame,
    pending_bets: pd.DataFrame,
    max_candidates: int | None,
) -> tuple[pd.DataFrame, int]:
    """Order the bounded lookup queue, reserving capacity for tracked bets."""
    if pending.empty:
        return pending.copy(), 0

    ordered = _prioritize_tracked_pending_matches(pending, pending_bets)
    ordered["_start_sort"] = ordered.apply(
        lambda row: _parse_exact_match_start_utc(row), axis=1
    )
    legacy_start = pd.to_datetime(
        ordered.get("match_start_time", pd.Series("", index=ordered.index)),
        errors="coerce",
        utc=True,
    )
    ordered["_start_sort"] = ordered["_start_sort"].where(
        ordered["_start_sort"].notna(), legacy_start
    )
    effective_date = ordered.apply(_effective_settlement_match_date, axis=1)
    date_rank = pd.to_datetime(
        effective_date,
        errors="coerce",
        utc=True,
    ).map(lambda value: value.timestamp() if pd.notna(value) else float("inf"))
    # Oldest tracked exposure first. Prediction-only backlog keeps the prior
    # newest-day-first policy so ancient zombies do not crowd out likely fresh
    # official results after all reserved-capital candidates are queued.
    ordered["_queue_date_sort"] = [
        rank if tracked else (-rank if math.isfinite(rank) else rank)
        for rank, tracked in zip(
            date_rank,
            ordered["_tracked_bet_priority"].fillna(False).astype(bool),
        )
    ]
    sort_cols = [
        column for column in
        ["_tracked_bet_priority", "_queue_date_sort", "_start_sort", "p1", "p2"]
        if column in ordered.columns
    ]
    if sort_cols:
        ascending = [
            column != "_tracked_bet_priority"
            for column in sort_cols
        ]
        ordered = ordered.sort_values(sort_cols, ascending=ascending)

    # One compatible operational duplicate may survive durable hydration. It
    # should consume one external lookup, not one per CSV row.
    ordered = _dedupe_pending_match_uids(ordered)
    if max_candidates and max_candidates > 0:
        ordered = ordered.head(max_candidates)
    tracked_count = int(
        ordered["_tracked_bet_priority"].fillna(False).astype(bool).sum()
    )
    return (
        ordered.drop(
            columns=["_start_sort", "_queue_date_sort", "_tracked_bet_priority"],
            errors="ignore",
        ),
        tracked_count,
    )


def _recently_attempted_identity_keys(
    *,
    audit_path: Path = SETTLEMENT_AUDIT_LOG_PATH,
    backoff_hours: float = DEFAULT_RETRY_BACKOFF_HOURS,
    now=None,
) -> tuple[set[int], set[str], set[str]]:
    """
    Return stable identity keys attempted recently by real settlement runs.

    This keeps catch-up runs moving forward: an unresolved old row should not
    block every immediate rerun while TA has no matching result. Row indexes
    remain a legacy fallback, but match/prediction UIDs prevent a compatible
    duplicate row (or a shifted CSV index) from bypassing the cooldown.
    """
    if backoff_hours <= 0 or not Path(audit_path).exists():
        return set(), set(), set()
    try:
        audit = pd.read_csv(audit_path)
    except Exception:
        return set(), set(), set()
    if audit.empty or not {"logged_at", "dry_run", "row_index"}.issubset(audit.columns):
        return set(), set(), set()

    logged_at = pd.to_datetime(audit["logged_at"], errors="coerce", utc=True)
    reference = pd.Timestamp(now if now is not None else utc_now())
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    cutoff = reference.tz_convert("UTC") - pd.Timedelta(hours=backoff_hours)

    dry_run = audit.get("dry_run", pd.Series(False, index=audit.index)).fillna(False).astype(str).str.lower()
    is_real_attempt = ~dry_run.isin({"true", "1", "yes"})
    recent = audit[is_real_attempt & logged_at.ge(cutoff)].copy()
    indexes = pd.to_numeric(recent["row_index"], errors="coerce").dropna().astype(int)

    def _stable_values(column: str) -> set[str]:
        if column not in recent.columns:
            return set()
        values = recent[column].fillna("").astype(str).str.strip()
        return {
            value for value in values
            if value and value.casefold() not in {"nan", "none", "null"}
        }

    return set(indexes), _stable_values("match_uid"), _stable_values("prediction_uid")


def _recently_attempted_row_indexes(
    *,
    audit_path: Path = SETTLEMENT_AUDIT_LOG_PATH,
    backoff_hours: float = DEFAULT_RETRY_BACKOFF_HOURS,
    now=None,
) -> set[int]:
    """Backward-compatible row-index view of the stable retry identity set."""
    indexes, _, _ = _recently_attempted_identity_keys(
        audit_path=audit_path,
        backoff_hours=backoff_hours,
        now=now,
    )
    return indexes


def _recently_attempted_mask(
    pending: pd.DataFrame,
    *,
    row_indexes: set[int],
    match_uids: set[str],
    prediction_uids: set[str],
) -> pd.Series:
    """Mark every pending row covered by a recent stable settlement identity."""
    mask = pd.Series(pending.index.isin(row_indexes), index=pending.index, dtype=bool)
    if match_uids and "match_uid" in pending.columns:
        values = pending["match_uid"].fillna("").astype(str).str.strip()
        mask |= values.isin(match_uids)
    if prediction_uids and "prediction_uid" in pending.columns:
        values = pending["prediction_uid"].fillna("").astype(str).str.strip()
        mask |= values.isin(prediction_uids)
    return mask


# ---------------------------------------------------------------------------
# Core settle logic
# ---------------------------------------------------------------------------

def try_settle_from_ta(p1: str, p2: str, match_date_str: str,
                        calc: TAFeatureCalculator,
                        tournament: str = "",
                        round_code: str = "",
                        surface: str = "",
                        session_cache: dict | None = None,
                        dry_run: bool = False) -> dict:
    """
    Fetch p1's recent matches and look for a result vs p2 on/after match_date.
    Returns a dict with a status code plus settlement metadata when available.
    """
    match_date = pd.to_datetime(match_date_str, errors='coerce')
    if pd.isna(match_date):
        print(f"  ⚠️  Could not parse match_date '{match_date_str}' — skipping")
        return {
            'status': 'parse_error',
            'outcome_detail': f"Could not parse match_date '{match_date_str}'",
            'ta_player_slug': '',
        }

    slug1 = _resolve_slug(p1, calc)
    current_year = datetime.now().year
    start_year = min(int(match_date.year), current_year)
    end_year = max(int(match_date.year), current_year)
    years = list(range(start_year, end_year + 1))

    context = _build_settlement_context(
        tournament=tournament,
        round_code=round_code,
        surface=surface,
    )

    print(f"  Checking TA for {p1} ({slug1}) vs {p2}...")
    matches = SCRAPER.get_player_matches(
        slug1,
        years=years,
        force_refresh=True,
        session_cache=session_cache,
    )
    upcoming = SCRAPER.get_upcoming_match(slug1, p2, session_cache=session_cache)

    def _unfinished_response() -> dict:
        raw_date = str(upcoming.get('date', '')) if upcoming else ''
        ta_date = raw_date
        if len(raw_date) == 8 and raw_date.isdigit():
            try:
                ta_date = datetime.strptime(raw_date, '%Y%m%d').strftime('%Y-%m-%d')
            except Exception:
                ta_date = raw_date
        return {
            'status': 'ta_match_unfinished',
            'outcome_detail': f"TA lists an upcoming/unfinished match vs '{p2}', but no result is posted yet",
            'ta_player_slug': slug1,
            'ta_match_date_found': ta_date,
            'ta_event_found': str(upcoming.get('event', '')) if upcoming else '',
            'ta_round_found': str(upcoming.get('round', '')) if upcoming else '',
        }

    if matches.empty:
        if upcoming:
            print(f"    TA lists upcoming/unfinished match vs '{p2}', no result posted yet")
            return _unfinished_response()
        print(f"    No match data found for {slug1}")
        return {
            'status': 'ta_empty',
            'outcome_detail': f"No match data found for {slug1}",
            'ta_player_slug': slug1,
        }

    # TA stores the tournament START DATE for all rounds (same as Sackmann CSV),
    # not the actual match date. So we can't use a tight date filter.
    # Use a wide window: 21 days before (covers tournament start offset)
    # and 14 days after (covers delayed runs, rain delays, rescheduled matches).
    # Rely on opponent name matching to identify the specific match.
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    window_start = match_date - timedelta(days=21)
    window_end   = match_date + timedelta(days=14)
    recent = matches[
        (matches['date'] >= window_start) &
        (matches['date'] <= window_end)
    ].copy()

    if recent.empty:
        if upcoming:
            print(f"    TA lists upcoming/unfinished match vs '{p2}', no result posted yet")
            return _unfinished_response()
        print(f"    No matches found within window of {match_date_str}")
        return {
            'status': 'outside_window',
            'outcome_detail': f"No matches found within window of {match_date_str}",
            'ta_player_slug': slug1,
        }

    # Look for a match vs p2
    found = recent[recent['opp_name'].apply(lambda n: _names_match(str(n), p2))]

    if found.empty:
        if upcoming:
            print(f"    TA lists upcoming/unfinished match vs '{p2}', no result posted yet")
            return _unfinished_response()
        print(f"    No result found yet vs '{p2}'")
        return {
            'status': 'opponent_not_found',
            'outcome_detail': f"No result found yet vs '{p2}'",
            'ta_player_slug': slug1,
        }

    row, selection_status, selection_evidence = _select_best_settlement_candidate(
        found,
        match_date,
        context,
    )
    if row is None:
        detail = json.dumps(selection_evidence, sort_keys=True)
        if selection_status == "ambiguous_match":
            print(f"    Ambiguous TA matches vs '{p2}' — leaving unsettled")
            return {
                'status': 'ambiguous_match',
                'outcome_detail': detail,
                'ta_player_slug': slug1,
            }
        print(f"    Low-confidence TA match vs '{p2}' — leaving unsettled")
        return {
            'status': selection_status,
            'outcome_detail': detail,
            'ta_player_slug': slug1,
        }

    result = str(row.get('result', '')).upper()
    score = str(row.get('score', ''))
    match_date_found = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '?'

    if result == 'W':
        actual_winner = 1
        winner_name = p1
    elif result == 'L':
        actual_winner = 2
        winner_name = p2
    else:
        print(f"    Unexpected result value '{result}' — skipping")
        return {
            'status': 'unexpected_result',
            'outcome_detail': f"Unexpected result value '{result}'",
            'ta_player_slug': slug1,
            'ta_match_date_found': match_date_found,
            'ta_event_found': str(row.get('event', '')),
            'ta_round_found': str(row.get('round', '')),
            'ta_surface_found': str(row.get('surface', '')),
            'settlement_score': selection_evidence.get('score'),
            'settlement_evidence': selection_evidence,
        }

    print(f"    Found result ({match_date_found}): {winner_name} won  |  score: {score}")
    return {
        'status': 'matched_and_settled',
        'actual_winner': actual_winner,
        'score': score,
        'settled_at': datetime.now().isoformat(),
        'ta_player_slug': slug1,
        'ta_match_date_found': match_date_found,
        'ta_event_found': str(row.get('event', '')),
        'ta_round_found': str(row.get('round', '')),
        'ta_surface_found': str(row.get('surface', '')),
        'settlement_score': selection_evidence.get('score'),
        'settlement_evidence': selection_evidence,
        'outcome_detail': (
            f"{winner_name} won | settlement_score={selection_evidence.get('score')} | "
            f"evidence={json.dumps(selection_evidence, sort_keys=True)}"
        ),
    }


# ---------------------------------------------------------------------------
# ATP results fallback (secondary source when TA has not posted results yet)
# ---------------------------------------------------------------------------

# Tournament -> immutable official results instance for labels that cannot be
# resolved through ATP event discovery. TA can lag days-to-weeks behind an
# event (e.g. it posted nothing between Halle and mid-Wimbledon 2026). A static
# binding must never use ATP's mutable ``/current/`` route: that route can go
# empty after completion or roll to a later edition while the configured ID,
# start date, and evidence still claim the old tournament. Explicit event
# labels may be contained in a richer Bovada title; generic bucket labels must
# match exactly so a title such as "French Open Men's Singles" can never be
# rebound to Wimbledon merely because its surface/date fields are also bad.
_ATP_FALLBACK_EVENTS = [
    {
        "explicit_labels": ("wimbledon",),
        "generic_labels": ("men s singles",),  # normalized: apostrophe strips to a space
        "surface": "grass",
        "event_name": "Wimbledon",
        "id": "540",
        "start_date": "2026-06-29",
        # The official results page includes qualifying matches from the week
        # before the main draw.  Keep that explicit rather than weakening the
        # general event/date contract for every other tournament.
        "match_window_start": "2026-06-21",
        "match_window_end": "2026-07-13",
        "date_verified": True,
        "date_source": "static_registry",
        "url": "https://www.atptour.com/en/scores/archive/wimbledon/540/2026/results",
    },
]

# Try the official ATP/ITF fallback sources for ANY status TA didn't
# confidently settle — NOT just a hand-picked few. The fallback sources carry
# their own conservative both-name identity + round corroboration, so an
# authoritative official-page result should never be withheld because TA was
# ambiguous/low-confidence/errored (the exact cases where you MOST want a
# second source). Only a clean TA settlement ('matched_and_settled') skips them.
# (Bug: 'low_confidence_match'/'ambiguous_match'/'parse_error' were silently
# non-eligible, stranding matches whose result sat ready on the ITF page.)
_TA_SETTLED_STATUS = "matched_and_settled"
_ATP_SETTLEMENT_CONFIDENCE = 90  # both full names matched on the official ATP results page


def _atp_results_source_for(tournament: str, surface: str, match_date_str: str) -> dict | None:
    label = _normalize_text(tournament)
    surf = _normalize_text(surface)
    match_date = pd.to_datetime(match_date_str, errors="coerce")
    # Static URLs deliberately bind generic labels to one event. They are safe
    # only with explicit compatible surface and date evidence; missing evidence
    # must not silently bypass those guards.
    if not label or not surf or pd.isna(match_date):
        return None
    match_date = pd.Timestamp(match_date).normalize()
    for ev in _ATP_FALLBACK_EVENTS:
        explicit_match = any(l in label for l in ev.get("explicit_labels", ()))
        generic_match = label in set(ev.get("generic_labels", ()))
        if not (explicit_match or generic_match):
            continue
        if ev["surface"] and ev["surface"] not in surf:
            continue
        window_start = pd.to_datetime(ev.get("match_window_start"), errors="coerce")
        window_end = pd.to_datetime(ev.get("match_window_end"), errors="coerce")
        if pd.isna(window_start) or pd.isna(window_end):
            continue
        if not (pd.Timestamp(window_start).normalize()
                <= match_date
                <= pd.Timestamp(window_end).normalize()):
            continue
        return ev
    return None


def _candidate_match_date(value) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_convert(None).normalize()


def _archive_atp_events(match_date: pd.Timestamp, cache: dict) -> list[dict]:
    """Official results-archive events for a candidate's calendar year.

    Old pending rows must not be compared with this week's similarly named
    event.  The archive URL carries the event ID, year, and official start
    date, and is fetched at most once per year in a settlement run.
    """
    archive_cache = cache.setdefault("_results_archive_by_year", {})
    events: list[dict] = []
    # A late-December event can finish in January. Load every year touched by
    # the conservative event/qualifying lookback, then dedupe by official
    # event instance instead of silently ignoring the prior-year archive.
    years = {
        int(match_date.year),
        int((match_date - pd.Timedelta(days=16)).year),
    }
    for year in sorted(years):
        if year not in archive_cache:
            year_events: list[dict] = []
            try:
                from scraping.atp_results_scraper import _fetch_rendered, parse_results_archive

                url = f"https://www.atptour.com/en/scores/results-archive?year={year}"
                html = _fetch_rendered(url, "a[href*='/en/scores/archive/']")
                frame = parse_results_archive(html, year)
                for _, row in frame.iterrows():
                    event_url = _clean_optional(row.get("url"))
                    event_id = _clean_optional(row.get("id"))
                    start_date = _clean_optional(row.get("start_date"))
                    if not event_url or not event_id or not start_date:
                        continue
                    year_events.append({
                        "event": _clean_optional(row.get("event")),
                        "slug": _clean_optional(row.get("slug")),
                        "id": event_id,
                        "url": event_url,
                        "start_date": start_date,
                        "date_verified": True,
                        "date_source": "results_archive",
                        "surface": _clean_optional(row.get("surface")),
                        "static_binding": False,
                    })
            except Exception as exc:
                print(f"  ⚠️ ATP results-archive discovery unavailable for {year}: {exc}")
            archive_cache[year] = year_events
        events.extend(archive_cache[year])
    return events


def _event_identity(event: dict) -> tuple[str, ...]:
    event_id = str(event.get("id") or "").strip()
    start_date = str(event.get("start_date") or "").strip()
    if event_id and start_date:
        return ("official", event_id, start_date)
    return ("url", str(event.get("url") or "").strip())


def _active_atp_events(cache: dict, *, match_date: str = "") -> list[dict]:
    """Active events for the settlement fallback.

    Use the same hub + official calendar discovery as live feature building.
    The ATP live-scores hubs occasionally render without event links even while
    the tournament results pages are healthy; relying on the hub alone leaves
    already-finished matches permanently pending.  Calendar discovery is the
    durable fallback because it carries stable event IDs and results URLs.
    """
    candidate_date = _candidate_match_date(match_date)
    if candidate_date is None:
        return []
    date_key = candidate_date.date().isoformat()
    events_by_date = cache.setdefault("_events_by_match_date", {})
    if date_key in events_by_date:
        return events_by_date[date_key]

    static_events = [
        {
            "event": ev["event_name"],
            "slug": normalize_text(ev["event_name"]).replace(" ", "-"),
            "id": str(ev.get("id") or ""),
            "url": ev["url"],
            "start_date": str(ev.get("start_date") or ""),
            "date_verified": ev.get("date_verified") is True,
            "date_source": str(ev.get("date_source") or "static_registry"),
            "surface": str(ev.get("surface") or ""),
            "static_binding": True,
        }
        for ev in _ATP_FALLBACK_EVENTS
    ]
    dynamic: dict[tuple[str, ...], dict] = {}
    try:
        from features.history_stitch import get_active_events

        discovered = get_active_events(
            candidate_date,
            cache.setdefault("_event_discovery", {}),
        )
        for ev in discovered:
            url = str(ev.get("url") or "").strip()
            event_id = str(ev.get("id") or "").strip()
            start_date = str(ev.get("start_date") or "").strip()
            if not url or not event_id or not start_date:
                continue
            event = {
                "event": str(ev.get("event") or ""),
                "slug": str(ev.get("slug") or ""),
                "id": event_id,
                "url": url,
                "start_date": start_date,
                "date_verified": ev.get("date_verified") is True,
                "date_source": str(ev.get("date_source") or ""),
                "surface": str(ev.get("surface") or ""),
                "static_binding": False,
            }
            dynamic[_event_identity(event)] = event
    except Exception as exc:
        print(f"  ⚠️ event discovery unavailable for settlement fallback: {exc}")

    # Archive rows are year-pinned and therefore outrank a current/calendar URL
    # for the same official event identity when replaying an old backlog row.
    for event in _archive_atp_events(candidate_date, cache):
        identity = _event_identity(event)
        previous = dynamic.get(identity, {})
        merged = {**previous, **event}
        if not _clean_optional(event.get("surface")):
            merged["surface"] = _clean_optional(previous.get("surface"))
        dynamic[identity] = merged

    events = [*static_events, *dynamic.values()]
    events_by_date[date_key] = events
    return events


def _date_compatible_events(
    events: list[dict],
    match_date: str,
    *,
    round_code: str = "",
) -> list[dict]:
    """Return one non-overlapping official event window for ``match_date``.

    ATP pages expose tournament start dates rather than per-card dates.  Use a
    conservative fourteen-day maximum, truncated at the day before the next
    same-label candidate starts. This makes consecutive-week rematches bind to
    one event instead of page iteration order. Unverified dates never qualify.
    """
    candidate_date = _candidate_match_date(match_date)
    if candidate_date is None:
        return []

    dated: list[tuple[pd.Timestamp, dict]] = []
    for event in events:
        if event.get("date_verified") is not True:
            continue
        start = _candidate_match_date(event.get("start_date"))
        if start is None:
            continue
        dated.append((start, event))
    dated.sort(key=lambda item: (item[0], _event_identity(item[1])))

    qualifying = _normalize_round(round_code).startswith("Q")
    compatible: list[dict] = []
    for index, (start, event) in enumerate(dated):
        later_starts = [other for other, _ in dated[index + 1:] if other > start]
        window_start = start - pd.Timedelta(days=3) if qualifying else start
        end = start + pd.Timedelta(days=13)
        if later_starts:
            end = min(end, later_starts[0] - pd.Timedelta(days=1))
        if window_start <= candidate_date <= end:
            bound = dict(event)
            bound.update({
                "match_date_bound": candidate_date.date().isoformat(),
                "selection_window_start": window_start.date().isoformat(),
                "selection_window_end": end.date().isoformat(),
            })
            compatible.append(bound)
    return compatible


def _candidate_atp_events(
    tournament: str,
    cache: dict,
    *,
    surface: str = "",
    match_date: str = "",
    round_code: str = "",
) -> list[dict]:
    """Prefer the official event matching this operational tournament label.

    Fetching every active ATP/Challenger page for each candidate is both slow
    and unnecessary.  Exact/contained event or slug labels select the narrow
    source set. A generic label may use the explicit, surface/date-bounded static
    registry. Every other unmatched label fails closed: searching every calendar
    event and accepting the first same-pair/same-round card can choose the wrong
    rematch when the players met more than once. ITF has its own official API path
    and should never fan out through ATP pages first.
    """
    tournament_key = _normalize_text(tournament)
    candidate_date = _candidate_match_date(match_date)
    if "itf" in tournament_key or candidate_date is None:
        return []

    if not tournament_key:
        return []
    events = _active_atp_events(cache, match_date=match_date)
    surface_key = _normalize_text(surface)

    matched: list[dict] = []
    for event in events:
        # Static generic-label mappings have stronger surface/date contracts
        # below and cannot enter through ordinary label containment.
        if bool(event.get("static_binding")):
            continue
        labels = {
            _normalize_text(event.get("event", "")),
            _normalize_text(event.get("slug", "")),
        } - {""}
        event_surface = _normalize_text(event.get("surface", ""))
        if (
            surface_key
            and event_surface
            and surface_key not in event_surface
            and event_surface not in surface_key
        ):
            continue
        if any(
            tournament_key in label or label in tournament_key
            for label in labels
        ):
            matched.append(event)
    matched = _date_compatible_events(
        matched,
        match_date,
        round_code=round_code,
    )
    if len(matched) == 1:
        return matched
    if matched:
        return []

    static_source = _atp_results_source_for(tournament, surface, match_date)
    if static_source:
        static_url = str(static_source.get("url") or "")
        static_matches = [
            event for event in events
            if event.get("url") == static_url
            and event.get("static_binding") is True
            and event.get("date_verified") is True
            and str(event.get("id") or "").strip()
            and str(event.get("start_date") or "").strip()
        ]
        if len(static_matches) != 1:
            return []
        bound = dict(static_matches[0])
        bound.update({
            "match_date_bound": candidate_date.date().isoformat(),
            "selection_window_start": str(static_source["match_window_start"]),
            "selection_window_end": str(static_source["match_window_end"]),
        })
        return [bound]
    return []


def _fetch_atp_results_cached(url: str, cache: dict) -> pd.DataFrame:
    if url not in cache:
        from scraping.atp_results_scraper import fetch_tournament_results
        print(f"  Fetching ATP results page (fallback source): {url}")
        try:
            cache[url] = fetch_tournament_results(url)
            print(f"    ATP results: {len(cache[url])} completed matches parsed")
        except Exception as exc:  # network/parse failure must not break the TA path
            print(f"    ⚠️ ATP results fetch failed (non-fatal): {exc}")
            cache[url] = pd.DataFrame()
    return cache[url]


def _combined_score(p1_sets: str, p2_sets: str) -> str:
    a, b = str(p1_sets).split(), str(p2_sets).split()
    return " ".join(f"{x}-{y}" for x, y in zip(a, b))


def _strict_full_name_key(value) -> str:
    """Normalized full name for official-result orientation.

    ATP cards publish full player names, so a one-token surname or a fuzzy
    last-name fallback is neither needed nor safe. Requiring two normalized
    tokens prevents Francisco/Juan Manuel Cerundolo from both matching the
    same card side while still normalizing accents and hyphenation.
    """
    normalized = normalize_name(_clean_optional(value))
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if len(normalized.split()) >= 2 else ""


def _strict_full_names_match(left, right) -> bool:
    left_key = _strict_full_name_key(left)
    right_key = _strict_full_name_key(right)
    return bool(left_key and right_key and left_key == right_key)


def try_settle_from_atp(p1: str, p2: str, round_code: str,
                        results: pd.DataFrame, event: dict) -> dict | None:
    """Settle from an ATP tournament-results page. Same result shape as
    try_settle_from_ta so the existing write path applies unchanged.

    Conservative identity rule: BOTH normalized full names must match a single
    results card in exactly one orientation; a known round must corroborate.
    The caller must also supply the verified official event instance that was
    date-bound during source selection. Ambiguity or incomplete provenance
    fails closed.
    """
    if results is None or results.empty:
        return None
    if not isinstance(event, dict):
        return None

    event_name = _clean_optional(event.get("event"))
    event_id = _clean_optional(event.get("id"))
    event_start_date = _clean_optional(event.get("start_date"))
    event_date_source = _clean_optional(event.get("date_source"))
    source_url = _clean_optional(event.get("url"))
    match_date_bound = _clean_optional(event.get("match_date_bound"))
    selection_window_start = _clean_optional(event.get("selection_window_start"))
    selection_window_end = _clean_optional(event.get("selection_window_end"))
    bound_date = _candidate_match_date(match_date_bound)
    bound_start = _candidate_match_date(selection_window_start)
    bound_end = _candidate_match_date(selection_window_end)
    if (
        not event_name
        or not event_id
        or _candidate_match_date(event_start_date) is None
        or event.get("date_verified") is not True
        or not event_date_source
        or not source_url
        or bound_date is None
        or bound_start is None
        or bound_end is None
        or not (bound_start <= bound_date <= bound_end)
    ):
        return None

    candidates = []
    for _, card in results.iterrows():
        c1, c2 = str(card.get("p1", "")), str(card.get("p2", ""))
        if _strict_full_names_match(c1, p1) and _strict_full_names_match(c2, p2):
            candidates.append((card, False))  # our p1 == card p1
        if _strict_full_names_match(c1, p2) and _strict_full_names_match(c2, p1):
            candidates.append((card, True))   # orientations flipped

    rc = _normalize_round(round_code)
    if rc and len(candidates) > 1:
        candidates = [(c, f) for c, f in candidates if _normalize_round(str(c["round"])) == rc]
    if len(candidates) != 1:
        return None  # not found, or ambiguous — leave pending
    card, flipped = candidates[0]
    if rc and _normalize_round(str(card["round"])) not in ("", rc):
        return None  # round contradicts our row — do not settle

    card_winner = card.get("winner")
    if card_winner not in (1, 2):
        return None
    actual_winner = int(card_winner) if not flipped else (2 if card_winner == 1 else 1)
    winner_name = p1 if actual_winner == 1 else p2

    if flipped:
        score = _combined_score(card.get("p2_sets", ""), card.get("p1_sets", ""))
    else:
        score = _combined_score(card.get("p1_sets", ""), card.get("p2_sets", ""))

    evidence = {
        "source": "atp_results",
        "event": event_name,
        "event_id": event_id,
        "event_start_date": event_start_date,
        "start_date": event_start_date,
        "event_instance": f"{event_id}:{event_start_date}",
        "event_date_verified": True,
        "date_verified": True,
        "event_date_source": event_date_source,
        "date_source": event_date_source,
        "source_url": source_url,
        "url": source_url,
        "match_date_bound": match_date_bound,
        "selection_window_start": selection_window_start,
        "selection_window_end": selection_window_end,
        "identity_binding": "strict_normalized_full_name",
        "card_p1": str(card["p1"]),
        "card_p2": str(card["p2"]),
        "card_round": str(card["round"]),
        "orientation_flipped": flipped,
        "score": _ATP_SETTLEMENT_CONFIDENCE,
    }
    print(f"    Found on ATP results ({event_name}): {winner_name} won  |  score: {score}")
    return {
        "status": "matched_and_settled",
        "actual_winner": actual_winner,
        "score": score,
        "settled_at": datetime.now().isoformat(),
        "ta_player_slug": "",
        "ta_match_date_found": event_start_date,
        "ta_event_found": event_name,
        "ta_round_found": str(card["round"]),
        "ta_surface_found": _clean_optional(event.get("surface")),
        "settlement_score": _ATP_SETTLEMENT_CONFIDENCE,
        "settlement_evidence": evidence,
        "outcome_detail": (
            f"{winner_name} won | source=atp_results | evidence={json.dumps(evidence, sort_keys=True)}"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show_stats(df: pd.DataFrame):
    settled = df[df['actual_winner'].notna()].copy()
    if settled.empty:
        print("No settled predictions yet.")
        return

    n_all = len(settled)
    print(f"\n{'='*70}")
    print(f"  LIVE ACCURACY REPORT")
    print(f"{'='*70}")
    print(f"  Total settled: {n_all}  (includes ITF/no-model entries)")

    # Clean stats: only rows with complete features and a real model prediction
    clean = settled[
        settled['model_correct'].notna() &
        settled['features_complete'].fillna(True).astype(bool)
    ].copy()

    if clean.empty:
        print("  No complete-feature predictions settled yet.")
        return

    # Exclude 50/50 market (market not making a pick) for fair comparison
    has_market_pick = clean['market_p1_prob'] != 0.50
    clean_no5050 = clean[has_market_pick]
    n_5050 = len(clean) - len(clean_no5050)

    n = len(clean_no5050)
    model_acc  = clean_no5050['model_correct'].mean()
    market_acc = clean_no5050['market_correct'].mean()
    print(f"\n  [Complete features, market has pick — {n} matches]")
    print(f"  (Excluded {n_5050} matches where market was 50/50)")
    print(f"  Model:  {model_acc:.1%}  ({int(clean_no5050['model_correct'].sum())}/{n})")
    print(f"  Market: {market_acc:.1%}  ({int(clean_no5050['market_correct'].sum())}/{n})")
    print(f"  Edge:   {model_acc - market_acc:+.1%}")

    # By model version
    if 'model_version' in clean_no5050.columns:
        print(f"\n  {'─'*66}")
        print(f"  By model version:")
        for ver, grp in clean_no5050.groupby('model_version'):
            if grp.empty:
                continue
            ma = grp['model_correct'].mean()
            mk = grp['market_correct'].mean()
            edge = ma - mk
            print(f"    {ver}: model {ma:.1%}  market {mk:.1%}  edge {edge:+.1%}  ({len(grp)} matches)")

    # By surface
    if 'surface' in clean_no5050.columns and clean_no5050['surface'].notna().any():
        print(f"\n  {'─'*66}")
        print(f"  By surface:")
        for surf, grp in clean_no5050.groupby('surface'):
            if grp.empty:
                continue
            ma = grp['model_correct'].mean()
            mk = grp['market_correct'].mean()
            print(f"    {surf}: model {ma:.1%}  market {mk:.1%}  edge {ma-mk:+.1%}  ({len(grp)} matches)")

    # Feature completeness summary
    all_settled = df[df['actual_winner'].notna()]
    incomplete = all_settled[all_settled['features_complete'].fillna(True).astype(bool) == False]
    if len(incomplete) > 0:
        print(f"\n  {'─'*66}")
        print(f"  Feature completeness:")
        print(f"    Complete: {len(all_settled) - len(incomplete)}  |  Incomplete (excluded): {len(incomplete)}")
        # Show which features defaulted most
        all_defaults = []
        for _, r in incomplete.iterrows():
            if pd.notna(r.get('defaulted_features', '')) and str(r.get('defaulted_features', '')).strip():
                all_defaults.extend(str(r['defaulted_features']).split(','))
        if all_defaults:
            from collections import Counter
            top = Counter(all_defaults).most_common(5)
            print(f"    Top defaulted features:")
            for feat, cnt in top:
                print(f"      {feat.strip()}: {cnt}x")

    print(f"{'='*70}")


def run(
    dry_run: bool = False,
    stats_only: bool = False,
    stale_days: int = 7,
    include_market_only: bool = False,
    run_id: str | None = None,
    record_run_history: bool | None = None,
    min_age_hours: float = DEFAULT_MIN_SETTLEMENT_AGE_HOURS,
    max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
    rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    max_rate_limits: int = DEFAULT_MAX_RATE_LIMITS,
    rate_limit_cooldown_seconds: float = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    retry_backoff_hours: float = DEFAULT_RETRY_BACKOFF_HOURS,
):
    started = utc_now()
    owns_run_id = run_id is None
    if run_id is None:
        run_id = make_run_id(started, prefix='settle')
    if record_run_history is None:
        record_run_history = owns_run_id

    summary = {
        'run_id': run_id,
        'run_kind': 'auto_settle',
        'started_at': started.replace(microsecond=0).isoformat(),
        'completed_at': '',
        'status': 'running',
        'auto_settle_enabled': True,
        'rankings_refresh_enabled': '',
        'odds_rows_fetched': 0,
        'odds_rows_candidate': 0,
        'feature_rows_total': 0,
        'feature_rows_ok': 0,
        'feature_rows_skipped': 0,
        'feature_skip_reason_summary': {},
        'prediction_rows_total': 0,
        'prediction_rows_success': 0,
        'prediction_rows_error': 0,
        'prediction_log_attempts': 0,
        'prediction_log_created': 0,
        'prediction_log_updated': 0,
        'prediction_log_skipped_incomplete': 0,
        'prediction_identity_aliases': 0,
        'prediction_identity_conflicts': 0,
        'bet_opportunities': 0,
        'bets_logged': 0,
        'settlement_candidates': 0,
        'settlement_newly_settled': 0,
        'settlement_auto_settled_bets': 0,
        'settlement_reason_summary': {},
        'error_message': '',
    }
    SCRAPER.rate_limit_delay = rate_limit_delay
    if hasattr(SCRAPER, 'rate_limit_hits'):
        SCRAPER.rate_limit_hits = 0
    if record_run_history:
        upsert_run_history(summary)

    if not LOG_PATH.exists():
        print("No prediction_log.csv found.")
        summary['status'] = 'missing_prediction_log'
        summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
        if record_run_history:
            upsert_run_history(summary)
        return summary

    df = upgrade_prediction_log(LOG_PATH, stale_days=stale_days, write=not dry_run)

    if stats_only:
        show_stats(df)
        summary['status'] = 'stats_only'
        summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
        if record_run_history:
            upsert_run_history(summary)
        return summary

    tracker = BetTracker(str(Path(__file__).parent / "logs"))
    verified_predictions = pd.DataFrame()
    exact_feature_evidence = {}
    attribution_repaired = 0
    attribution_evidence_error = ""
    try:
        # This uses the same feature-lineage verifier as the evaluation ledger.
        # A corrupt or contradictory lineage source disables model attribution
        # but must not prevent a known result from closing paper exposure.
        verified_predictions = load_verified_prediction_log(
            Path(__file__).parent
        )
        exact_feature_evidence = load_feature_attribution_evidence(
            Path(__file__).parent
        )
        if not dry_run:
            attribution_repaired = tracker.repair_settled_bet_attribution(
                verified_predictions, exact_feature_evidence
            )
            if attribution_repaired:
                print(
                    "Repaired exact attribution on "
                    f"{attribution_repaired} previously unclassified settled bet(s)"
                )
    except Exception as exc:
        attribution_evidence_error = str(exc)
        verified_predictions = pd.DataFrame()
        exact_feature_evidence = {}
        print(
            "  ⚠️ Exact bet attribution disabled for this run; accounting "
            f"settlement remains available: {exc}"
        )
    tracked_pending_bets = tracker.get_pending_bets()

    pending = df[df['actual_winner'].isna()].copy()
    pending, identity_gate_counts = _settlement_uid_gate(df, pending)
    if 'record_status' in pending.columns:
        pending = pending[~pending['record_status'].isin([
            'stale_no_model',
            'expired_unsettled',
            'identity_conflict',
            'superseded_identity',
        ])]
    if not include_market_only:
        pending = pending[pending['model_p1_prob'].notna()]
    age_skip_reasons = []
    if min_age_hours > 0 and not pending.empty:
        eligible_mask = []
        for _, candidate in pending.iterrows():
            eligible, reason = _is_old_enough_to_settle(candidate, min_age_hours=min_age_hours)
            eligible_mask.append(eligible)
            if not eligible:
                age_skip_reasons.append(reason)
        pending = pending[pd.Series(eligible_mask, index=pending.index)].copy()

    (
        recent_attempt_indexes,
        recent_attempt_match_uids,
        recent_attempt_prediction_uids,
    ) = _recently_attempted_identity_keys(backoff_hours=retry_backoff_hours)
    recent_attempt_count = 0
    if (
        recent_attempt_indexes
        or recent_attempt_match_uids
        or recent_attempt_prediction_uids
    ) and not pending.empty:
        before_recent_filter = len(pending)
        recently_attempted = _recently_attempted_mask(
            pending,
            row_indexes=recent_attempt_indexes,
            match_uids=recent_attempt_match_uids,
            prediction_uids=recent_attempt_prediction_uids,
        )
        pending = pending[~recently_attempted].copy()
        recent_attempt_count = before_recent_filter - len(pending)

    tracked_candidate_count = 0
    if not pending.empty:
        pending, tracked_candidate_count = _order_settlement_candidates(
            pending,
            tracked_pending_bets,
            max_candidates,
        )

    summary['settlement_candidates'] = len(pending)
    if pending.empty:
        gate_reasons = {
            reason: count for reason, count in identity_gate_counts.items()
            if count
        }
        if attribution_repaired:
            gate_reasons['bet_attribution_repaired'] = attribution_repaired
        if attribution_evidence_error:
            gate_reasons['bet_attribution_evidence_error'] = 1
        if identity_gate_counts['duplicate_identity_conflict'] and not dry_run:
            df.to_csv(LOG_PATH, index=False)
        if gate_reasons:
            print("No identity-safe unsettled predictions.")
            summary['settlement_reason_summary'] = gate_reasons
        elif age_skip_reasons:
            print(f"No old-enough unsettled predictions. Skipped {len(age_skip_reasons)} too-recent row(s).")
            summary['settlement_reason_summary'] = {'too_recent_to_settle': len(age_skip_reasons)}
        elif recent_attempt_count:
            print(f"No unsettled predictions outside retry backoff. Skipped {recent_attempt_count} recently-attempted row(s).")
            summary['settlement_reason_summary'] = {'recently_attempted': recent_attempt_count}
        else:
            print("No unsettled predictions.")
        show_stats(df)
        summary['status'] = 'success'
        summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
        if record_run_history:
            upsert_run_history(summary)
        return summary

    print(f"Found {len(pending)} unsettled prediction(s) to check")
    if tracked_candidate_count:
        print(
            f"Prioritized {tracked_candidate_count} candidate(s) linked to "
            "pending paper exposure"
        )
    if age_skip_reasons:
        print(f"Skipped {len(age_skip_reasons)} too-recent prediction(s) before hitting TA")
    if recent_attempt_count:
        print(f"Skipped {recent_attempt_count} recently-attempted prediction(s) inside {retry_backoff_hours:.1f}h retry backoff")
    print(f"TA pacing: {rate_limit_delay:.1f}s minimum between requests, stop after {max_rate_limits} rate-limit hit(s)\n")

    # Load player mapping once
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.player_slug_map = TAFeatureCalculator._load_player_mapping(calc)
    session_cache = {}
    atp_results_cache = {}

    newly_settled = 0
    newly_settled_bets = 0
    reason_counts = Counter()
    reason_counts.update({
        reason: count for reason, count in identity_gate_counts.items()
        if count
    })
    if age_skip_reasons:
        reason_counts['too_recent_to_settle'] = len(age_skip_reasons)
    if recent_attempt_count:
        reason_counts['recently_attempted'] = recent_attempt_count
    if attribution_repaired:
        reason_counts['bet_attribution_repaired'] = attribution_repaired
    if attribution_evidence_error:
        reason_counts['bet_attribution_evidence_error'] = 1
    stopped_for_rate_limit = False

    for idx, row in pending.iterrows():
        p1 = str(row['p1'])
        p2 = str(row['p2'])
        match_date = _effective_settlement_match_date(row)
        print(f"\n[{idx}] {p1} vs {p2}  ({row.get('tournament', '')}  {match_date})")
        print(f"     Model: {float(row['model_p1_prob']):.0%} P1 | Market: {float(row['market_p1_prob']):.0%} P1")

        rate_hits_before = getattr(SCRAPER, 'rate_limit_hits', 0)
        result = try_settle_from_ta(
            p1,
            p2,
            match_date,
            calc,
            tournament=str(row.get('tournament', '')),
            round_code=str(row.get('round', '')),
            surface=str(row.get('surface', '')),
            session_cache=session_cache,
            dry_run=dry_run,
        )
        # Secondary source: when TA has no result yet, try the official ATP
        # results pages of every active event (source=atp_results). Both-name
        # identity + round corroboration keep this conservative.
        if result.get('status') != _TA_SETTLED_STATUS:
            for ev in _candidate_atp_events(
                str(row.get('tournament', '')),
                atp_results_cache,
                surface=str(row.get('surface', '')),
                match_date=match_date,
                round_code=str(row.get('round', '')),
            ):
                atp_df = _fetch_atp_results_cached(ev["url"], atp_results_cache)
                if atp_df is None or atp_df.empty:
                    continue
                atp_result = try_settle_from_atp(
                    p1, p2, str(row.get('round', '')), atp_df, ev,
                )
                if atp_result:
                    result = atp_result
                    break

        # Tertiary source: ITF order-of-play (itftennis.com API) for ITF-tier
        # matches — the ATP pages never carry these. Same conservative matching.
        if result.get('status') != _TA_SETTLED_STATUS and "itf" in str(row.get('tournament', '')).lower():
            try:
                from features.history_stitch import _itf_event_for, _itf_event_matches, _names_loosely_match
                ev = _itf_event_for(str(row.get('tournament', '')), match_date, atp_results_cache)
                if ev:
                    em = _itf_event_matches(ev["key"], atp_results_cache)
                    if em is not None and not em.empty:
                        done = em[em["completed"] & em["winner"].notna()]
                        for _, card in done.iterrows():
                            fwd = _names_loosely_match(card["p1"], p1) and _names_loosely_match(card["p2"], p2)
                            rev = _names_loosely_match(card["p1"], p2) and _names_loosely_match(card["p2"], p1)
                            if not (fwd or rev):
                                continue
                            won_p1 = (int(card["winner"]) == 1) if fwd else (int(card["winner"]) == 2)
                            # ITF cards store scores WINNER-first; our log stores P1-first.
                            # Winner-first == our-P1-first exactly when OUR P1 won.
                            score = str(card["score"])
                            if not won_p1:
                                score = " ".join("-".join(reversed(seg.split("-"))) for seg in score.split())
                            from datetime import datetime as _dt
                            result = {
                                'status': 'matched_and_settled',
                                'actual_winner': 1 if won_p1 else 2,
                                'score': score,
                                'settled_at': _dt.now().isoformat(),
                                'ta_player_slug': '', 'ta_match_date_found': str(card.get("date", "")),
                                'ta_event_found': ev["event"], 'ta_round_found': str(card.get("round", "")),
                                'ta_surface_found': ev.get("surface", ""),
                                'settlement_score': 85,
                                'settlement_evidence': {'source': 'itf_oop', 'event': ev["event"]},
                                'outcome_detail': f"{p1 if won_p1 else p2} won | source=itf_oop | event={ev['event']}",
                            }
                            print(f"    Found on ITF ({ev['event']}): {p1 if won_p1 else p2} won  |  score: {score}")
                            break
            except Exception as _itf_exc:
                print(f"    ⚠️ ITF settle fallback failed (non-fatal): {_itf_exc}")

        outcome_code = result.get('status', 'unknown')
        reason_counts[outcome_code] += 1
        rate_hits_after = getattr(SCRAPER, 'rate_limit_hits', 0)
        log_settlement_event(
            run_id=run_id,
            dry_run=dry_run,
            row_index=idx,
            record_status_before=str(row.get('record_status', '')),
            match_uid=row.get('match_uid', ''),
            prediction_uid=row.get('prediction_uid', ''),
            match_date=match_date,
            match_start_time=str(row.get('match_start_time', '')),
            tournament=str(row.get('tournament', '')),
            round_code=str(row.get('round', '')),
            surface=str(row.get('surface', '')),
            p1=p1,
            p2=p2,
            model_version=str(row.get('model_version', '')),
            ta_player_slug=result.get('ta_player_slug', ''),
            outcome_code=outcome_code,
            outcome_detail=result.get('outcome_detail', ''),
            ta_match_date_found=result.get('ta_match_date_found', ''),
            ta_event_found=result.get('ta_event_found', ''),
            ta_round_found=result.get('ta_round_found', ''),
            ta_surface_found=result.get('ta_surface_found', ''),
            settlement_score=result.get('settlement_score', ''),
            settlement_evidence=result.get('settlement_evidence', {}),
            actual_winner=result.get('actual_winner'),
            score=result.get('score', ''),
        )

        if rate_hits_after > rate_hits_before:
            hit_delta = rate_hits_after - rate_hits_before
            reason_counts['ta_rate_limited'] += hit_delta
            print(f"  ⚠️ TA rate limit observed ({rate_hits_after} total hit(s))")
            if rate_hits_after >= max_rate_limits:
                print("  Stopping settlement early to protect TA access; rerun later to resume.")
                stopped_for_rate_limit = True
                break
            if rate_limit_cooldown_seconds > 0:
                print(f"  Cooling down {rate_limit_cooldown_seconds:.0f}s before the next candidate...")
                time.sleep(rate_limit_cooldown_seconds)

        if outcome_code != 'matched_and_settled':
            print(f"  → Not settled yet ({outcome_code})")
            continue

        if dry_run:
            winner_label = p1 if result['actual_winner'] == 1 else p2
            print(f"  [DRY RUN] Would settle: winner={winner_label}, score={result['score']}")
            continue

        # Write one authoritative result onto every compatible pending copy of
        # this exact UID. This clears hydrated duplicate backlog without a
        # second external lookup while retaining every immutable observation.
        settlement_indices = _compatible_pending_group_indices(df, idx)
        w = result['actual_winner']
        if 'xgb_correct' not in df.columns:
            df['xgb_correct'] = None
        if 'rf_correct' not in df.columns:
            df['rf_correct'] = None
        model_correct = None
        market_correct = None
        for settlement_index in settlement_indices:
            settlement_row = df.loc[settlement_index]
            df.at[settlement_index, 'actual_winner'] = w
            df.at[settlement_index, 'score'] = result['score']
            df.at[settlement_index, 'settled_at'] = result['settled_at']
            df.at[settlement_index, 'record_status'] = 'settled'
            row_model_correct = _prediction_correct(
                settlement_row.get('model_p1_prob'), w
            )
            row_market_correct = _prediction_correct(
                settlement_row.get('market_p1_prob'), w
            )
            row_xgb_correct = _prediction_correct(
                settlement_row.get('xgb_p1_prob'), w
            )
            row_rf_correct = _prediction_correct(
                settlement_row.get('rf_p1_prob'), w
            )
            if row_model_correct is not None:
                df.at[settlement_index, 'model_correct'] = row_model_correct
            if row_market_correct is not None:
                df.at[settlement_index, 'market_correct'] = row_market_correct
            if row_xgb_correct is not None:
                df.at[settlement_index, 'xgb_correct'] = row_xgb_correct
            if row_rf_correct is not None:
                df.at[settlement_index, 'rf_correct'] = row_rf_correct
            if settlement_index == idx:
                model_correct = row_model_correct
                market_correct = row_market_correct

        winner_name = p1 if w == 1 else p2
        model_str = 'N/A' if model_correct is None else ('✓' if model_correct else '✗')
        market_str = 'N/A' if market_correct is None else ('✓' if market_correct else '✗')
        print(
            f"  ✓ Settled: {winner_name} won | Model {model_str} | "
            f"Market {market_str} | rows={len(settlement_indices)}"
        )
        newly_settled += 1

        bound_result_evidence = build_auto_result_evidence(
            source_evidence=result.get('settlement_evidence', {}),
            match_uid=row.get('match_uid'),
            p1=p1,
            p2=p2,
            actual_winner=w,
            score=result.get('score', ''),
        )
        result_evidence_kind, result_evidence_sha256 = bound_result_evidence
        direct_attribution_supported = (
            prediction_match_supports_exact_attribution(
                verified_predictions,
                row.get('match_uid'),
                p1=p1,
                p2=p2,
                actual_winner=w,
            )
        )

        settled_bets = tracker.settle_pending_bets_for_match(
            match_uid=row.get('match_uid'),
            alias_match_uids=(
                [
                    value.strip()
                    for value in str(
                        row.get('identity_related_match_uid') or ''
                    ).split('|')
                    if value.strip()
                ]
                if str(row.get('identity_status') or '').strip().lower()
                == 'canonical_alias'
                else []
            ),
            p1=p1,
            p2=p2,
            actual_winner=w,
            notes=(
                "Auto-settled | "
                f"evidence={result_evidence_kind or 'unavailable'} | "
                f"score={result['score']}"
            ),
            exact_feature_evidence=(
                exact_feature_evidence
                if direct_attribution_supported
                else {}
            ),
            result_evidence_kind=result_evidence_kind,
            result_evidence_sha256=result_evidence_sha256,
            bound_result_evidence=bound_result_evidence,
        )
        if settled_bets:
            print(f"  💰 Auto-settled {settled_bets} pending bet(s)")
            newly_settled_bets += settled_bets

    if not dry_run:
        df.to_csv(LOG_PATH, index=False)
        df = upgrade_prediction_log(LOG_PATH, stale_days=stale_days, write=True)
        try:
            from shadow.performance_v1_shadow import sync_shadow_settlements

            shadow_synced = sync_shadow_settlements(
                Path(__file__).parent / "logs" / "performance_v1_shadow_predictions.csv",
                LOG_PATH,
            )
            if shadow_synced:
                print(f"Synced settlement outcomes onto {shadow_synced} shadow prediction row(s)")
        except Exception as exc:
            print(f"  ⚠️ Shadow settlement sync failed (non-fatal): {exc}")
        print(f"\nSaved {newly_settled} newly settled prediction(s) to prediction_log.csv")
        if newly_settled_bets:
            print(f"Auto-settled {newly_settled_bets} pending tracked bet(s)")

    summary['settlement_newly_settled'] = newly_settled
    summary['settlement_auto_settled_bets'] = newly_settled_bets
    summary['settlement_reason_summary'] = dict(reason_counts)
    summary['status'] = 'stopped_rate_limited' if stopped_for_rate_limit else 'success'
    summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
    # terminal state for the permanently unsettleable: no source produced a
    # result 21+ days after the match. They stop consuming candidate slots but
    # remain in the log (honest record); reversible by clearing record_status.
    try:
        if not dry_run:
            df2 = pd.read_csv(LOG_PATH, low_memory=False)
            md = pd.to_datetime(df2['match_date'], errors='coerce')
            # rows with unparseable dates must not dodge expiry forever —
            # fall back to the freshest odds/log timestamp as their age
            fallback = pd.to_datetime(
                df2.get('latest_odds_scraped_at', pd.Series('', index=df2.index))
                   .fillna(df2.get('odds_scraped_at', ''))
                   .fillna(df2.get('latest_logged_at', ''))
                   .fillna(df2.get('logged_at', '')),
                errors='coerce', utc=True).dt.tz_localize(None)
            age_basis = md.fillna(fallback)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=21)
            cond = (df2['actual_winner'].isna()
                    & (age_basis.isna() | (age_basis < cutoff))
                    & (~df2.get(
                        'record_status', pd.Series('', index=df2.index)
                    ).astype(str).isin([
                        'expired_unsettled',
                        'settled',
                        'identity_conflict',
                        'superseded_identity',
                    ])))
            n_exp = int(cond.sum())
            if n_exp:
                df2.loc[cond, 'record_status'] = 'expired_unsettled'
                df2.to_csv(LOG_PATH, index=False)
                print(f"  ⏳ expired {n_exp} unsettleable row(s) (>21d, no source result)")
                summary['rows_expired'] = n_exp
    except Exception as _exp_exc:
        print(f"  ⚠️ expiry pass failed (non-fatal): {_exp_exc}")

    if record_run_history:
        upsert_run_history(summary)

    show_stats(df)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-settle predictions from Tennis Abstract results')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be settled without writing')
    parser.add_argument('--stats', action='store_true', help='Show accuracy stats only')
    parser.add_argument('--stale-days', type=int, default=7, help='Mark legacy no-model rows older than this as stale and skip them')
    parser.add_argument('--include-market-only', action='store_true', help='Also check old market-only rows with no model prediction')
    parser.add_argument('--min-age-hours', type=float, default=DEFAULT_MIN_SETTLEMENT_AGE_HOURS, help='Only check matches at least this many hours after scheduled start/date fallback')
    parser.add_argument('--max-candidates', type=int, default=DEFAULT_MAX_CANDIDATES, help='Maximum eligible unsettled rows to check in this run; use 0 for no cap')
    parser.add_argument('--rate-limit-delay', type=float, default=DEFAULT_RATE_LIMIT_DELAY, help='Minimum seconds between Tennis Abstract HTTP requests')
    parser.add_argument('--max-rate-limits', type=int, default=DEFAULT_MAX_RATE_LIMITS, help='Stop the run after this many observed TA 429 responses')
    parser.add_argument('--rate-limit-cooldown-seconds', type=float, default=DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS, help='Cooldown after a TA 429 response before continuing')
    parser.add_argument('--retry-backoff-hours', type=float, default=DEFAULT_RETRY_BACKOFF_HOURS, help='Skip rows attempted by real settlement runs within this many hours; use 0 to disable')
    args = parser.parse_args()

    run(
        dry_run=args.dry_run,
        stats_only=args.stats,
        stale_days=args.stale_days,
        include_market_only=args.include_market_only,
        min_age_hours=args.min_age_hours,
        max_candidates=None if args.max_candidates == 0 else args.max_candidates,
        rate_limit_delay=args.rate_limit_delay,
        max_rate_limits=args.max_rate_limits,
        rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
        retry_backoff_hours=args.retry_backoff_hours,
    )
