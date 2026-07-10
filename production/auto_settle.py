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
from logging_utils import make_run_id, utc_now
from prediction_logger import upgrade_prediction_log

LOG_PATH = Path(__file__).parent / "prediction_log.csv"
DEFAULT_RATE_LIMIT_DELAY = 8.0
# Tuned for the hourly cloud cadence: settlement sources are one cached fetch
# per EVENT per run (ATP results / ITF order-of-play), so re-checking a
# candidate is nearly free. The old 18h values date from daily manual runs
# where every candidate cost a Tennis Abstract fetch — they made same-day
# results wait until the next morning.
DEFAULT_MIN_SETTLEMENT_AGE_HOURS = 6.0
DEFAULT_MAX_CANDIDATES = 150
DEFAULT_MAX_RATE_LIMITS = 5
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 120.0
DEFAULT_RETRY_BACKOFF_HOURS = 2.0
MIN_SETTLEMENT_SCORE = 65
AMBIGUITY_MARGIN = 6

SCRAPER = TennisAbstractScraper(rate_limit_delay=DEFAULT_RATE_LIMIT_DELAY)


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


def _is_old_enough_to_settle(row: pd.Series, min_age_hours: float, now: datetime | None = None) -> tuple[bool, str]:
    # Bovada start times are US/Eastern display strings; comparing them naive
    # against the runner's local clock made eligibility depend on which
    # machine ran settlement (UTC cloud vs local laptop disagreed).
    from zoneinfo import ZoneInfo
    now = now or datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    start_dt = _parse_match_start_time(
        str(row.get("match_start_time", "")),
        str(row.get("match_date", "")),
        now=now,
    )
    if start_dt is None:
        return True, ""
    age_hours = (now - start_dt).total_seconds() / 3600
    if age_hours < min_age_hours:
        return False, f"match_start_age_hours={age_hours:.1f} < min_age_hours={min_age_hours:.1f}"
    return True, ""


def _recently_attempted_row_indexes(
    *,
    audit_path: Path = SETTLEMENT_AUDIT_LOG_PATH,
    backoff_hours: float = DEFAULT_RETRY_BACKOFF_HOURS,
    now=None,
) -> set[int]:
    """
    Return prediction-log row indexes attempted recently by real settlement runs.

    This keeps catch-up runs moving forward: an unresolved old row should not
    block every immediate rerun while TA has no matching result.
    """
    if backoff_hours <= 0 or not Path(audit_path).exists():
        return set()
    try:
        audit = pd.read_csv(audit_path, usecols=["logged_at", "dry_run", "row_index"])
    except Exception:
        return set()
    if audit.empty:
        return set()

    logged_at = pd.to_datetime(audit["logged_at"], errors="coerce", utc=True)
    reference = pd.Timestamp(now if now is not None else utc_now())
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    cutoff = reference.tz_convert("UTC") - pd.Timedelta(hours=backoff_hours)

    dry_run = audit.get("dry_run", pd.Series(False, index=audit.index)).fillna(False).astype(str).str.lower()
    is_real_attempt = ~dry_run.isin({"true", "1", "yes"})
    recent = audit[is_real_attempt & logged_at.ge(cutoff)].copy()
    indexes = pd.to_numeric(recent["row_index"], errors="coerce").dropna().astype(int)
    return set(indexes)


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

# Tournament -> live results URL. TA can lag days-to-weeks behind an in-progress
# event (e.g. it posted nothing between Halle and mid-Wimbledon 2026); the ATP
# scores page is current same-day. Keyed by (label substring, surface, month
# window) so Bovada's generic "Men's Singles" label maps to the right Slam.
_ATP_FALLBACK_EVENTS = [
    {
        "labels": ("wimbledon", "men s singles"),  # normalized: apostrophes strip to spaces
        "surface": "grass",
        "months": (6, 7),
        "event_name": "Wimbledon",
        "url": "https://www.atptour.com/en/scores/current/wimbledon/540/results",
    },
]

_ATP_FALLBACK_ELIGIBLE = {"opponent_not_found", "ta_match_unfinished", "ta_empty", "outside_window"}
_ATP_SETTLEMENT_CONFIDENCE = 90  # both full names matched on the official ATP results page


def _atp_results_source_for(tournament: str, surface: str, match_date_str: str) -> dict | None:
    label = _normalize_text(tournament)
    surf = _normalize_text(surface)
    month = pd.to_datetime(match_date_str, errors="coerce")
    month = int(month.month) if pd.notna(month) else None
    for ev in _ATP_FALLBACK_EVENTS:
        if not any(l in label for l in ev["labels"]):
            continue
        if ev["surface"] and surf and ev["surface"] not in surf:
            continue
        if month is not None and month not in ev["months"]:
            continue
        return ev
    return None


def _active_atp_events(cache: dict) -> list[dict]:
    """Active events for the settlement fallback: auto-discovered from the ATP
    live-scores hubs + the static registry. Cached per settlement run."""
    if "_events" in cache:
        return cache["_events"]
    events = [{"event": ev["event_name"], "url": ev["url"]} for ev in _ATP_FALLBACK_EVENTS]
    try:
        from scraping.atp_results_scraper import discover_active_events
        known = {e["url"] for e in events}
        for ev in discover_active_events():
            if ev["url"] not in known:
                events.append({"event": ev["event"], "url": ev["url"]})
    except Exception as exc:
        print(f"  ⚠️ event discovery unavailable for settlement fallback: {exc}")
    cache["_events"] = events
    return events


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


def try_settle_from_atp(p1: str, p2: str, round_code: str,
                        results: pd.DataFrame, event_name: str) -> dict | None:
    """Settle from an ATP tournament-results page. Same result shape as
    try_settle_from_ta so the existing write path applies unchanged.

    Conservative identity rule: BOTH names must match a single results card
    (either orientation); a known round must corroborate. Ambiguity -> None.
    """
    if results is None or results.empty:
        return None

    candidates = []
    for _, card in results.iterrows():
        c1, c2 = str(card["p1"]), str(card["p2"])
        if _names_match(c1, p1) and _names_match(c2, p2):
            candidates.append((card, False))  # our p1 == card p1
        elif _names_match(c1, p2) and _names_match(c2, p1):
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
        "ta_match_date_found": "",
        "ta_event_found": event_name,
        "ta_round_found": str(card["round"]),
        "ta_surface_found": "",
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

    pending = df[df['actual_winner'].isna()].copy()
    if 'record_status' in pending.columns:
        pending = pending[~pending['record_status'].isin(['stale_no_model'])]
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

    recent_attempt_indexes = _recently_attempted_row_indexes(backoff_hours=retry_backoff_hours)
    recent_attempt_count = 0
    if recent_attempt_indexes and not pending.empty:
        before_recent_filter = len(pending)
        pending = pending[~pending.index.isin(recent_attempt_indexes)].copy()
        recent_attempt_count = before_recent_filter - len(pending)

    if not pending.empty:
        sort_cols = [col for col in ["match_date", "match_start_time", "p1", "p2"] if col in pending.columns]
        # newest match DAYS first (zombies to the back), earliest STARTS first
        # within a day (most likely finished). Start times must be parsed —
        # string order puts "11:00 AM" before "2:00 PM" and let same-day ITF
        # floods crowd out finished tour matches.
        if "match_start_time" in pending.columns:
            pending["_start_sort"] = pd.to_datetime(
                pending["match_start_time"], errors="coerce")
        sort_cols = [c for c in ["match_date", "_start_sort", "p1", "p2"] if c in pending.columns]
        if sort_cols:
            ascending = [c != "match_date" for c in sort_cols]
            pending = pending.sort_values(sort_cols, ascending=ascending)
            pending = pending.drop(columns=["_start_sort"], errors="ignore")
        if max_candidates and max_candidates > 0:
            pending = pending.head(max_candidates)

    summary['settlement_candidates'] = len(pending)
    if pending.empty:
        if age_skip_reasons:
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
    if age_skip_reasons:
        print(f"Skipped {len(age_skip_reasons)} too-recent prediction(s) before hitting TA")
    if recent_attempt_count:
        print(f"Skipped {recent_attempt_count} recently-attempted prediction(s) inside {retry_backoff_hours:.1f}h retry backoff")
    print(f"TA pacing: {rate_limit_delay:.1f}s minimum between requests, stop after {max_rate_limits} rate-limit hit(s)\n")

    # Load player mapping once
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.player_slug_map = TAFeatureCalculator._load_player_mapping(calc)
    tracker = BetTracker(str(Path(__file__).parent / "logs"))
    session_cache = {}
    atp_results_cache = {}

    newly_settled = 0
    newly_settled_bets = 0
    reason_counts = Counter()
    if age_skip_reasons:
        reason_counts['too_recent_to_settle'] = len(age_skip_reasons)
    if recent_attempt_count:
        reason_counts['recently_attempted'] = recent_attempt_count
    stopped_for_rate_limit = False

    for idx, row in pending.iterrows():
        p1 = str(row['p1'])
        p2 = str(row['p2'])
        match_date = str(row.get('match_date', ''))
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
        if result.get('status') in _ATP_FALLBACK_ELIGIBLE:
            for ev in _active_atp_events(atp_results_cache):
                atp_df = _fetch_atp_results_cached(ev["url"], atp_results_cache)
                if atp_df is None or atp_df.empty:
                    continue
                atp_result = try_settle_from_atp(
                    p1, p2, str(row.get('round', '')), atp_df, ev["event"],
                )
                if atp_result:
                    result = atp_result
                    break

        # Tertiary source: ITF order-of-play (itftennis.com API) for ITF-tier
        # matches — the ATP pages never carry these. Same conservative matching.
        if result.get('status') in _ATP_FALLBACK_ELIGIBLE and "itf" in str(row.get('tournament', '')).lower():
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

        # Write result back
        df.at[idx, 'actual_winner'] = result['actual_winner']
        df.at[idx, 'score'] = result['score']
        df.at[idx, 'settled_at'] = result['settled_at']

        import math
        model_p1_raw = row['model_p1_prob']
        market_p1 = float(row['market_p1_prob'])
        w = result['actual_winner']
        # Only score model_correct if there was actually a model prediction
        if pd.isna(model_p1_raw) or (isinstance(model_p1_raw, float) and math.isnan(model_p1_raw)):
            model_correct = float('nan')
        else:
            model_p1 = float(model_p1_raw)
            model_correct = int((w == 1 and model_p1 > 0.5) or (w == 2 and model_p1 < 0.5))
        market_correct = int((w == 1 and market_p1 > 0.5) or (w == 2 and market_p1 < 0.5))
        if not pd.isna(model_correct):
            df.at[idx, 'model_correct'] = model_correct
        df.at[idx, 'market_correct'] = market_correct

        # Score XGBoost if prediction exists
        xgb_p1_raw = row.get('xgb_p1_prob')
        if 'xgb_correct' not in df.columns:
            df['xgb_correct'] = None
        if pd.notna(xgb_p1_raw) and not (isinstance(xgb_p1_raw, float) and math.isnan(xgb_p1_raw)):
            xgb_p1 = float(xgb_p1_raw)
            xgb_correct = int((w == 1 and xgb_p1 > 0.5) or (w == 2 and xgb_p1 < 0.5))
            df.at[idx, 'xgb_correct'] = xgb_correct

        # Score Random Forest if prediction exists
        rf_p1_raw = row.get('rf_p1_prob')
        if 'rf_correct' not in df.columns:
            df['rf_correct'] = None
        if pd.notna(rf_p1_raw) and not (isinstance(rf_p1_raw, float) and math.isnan(rf_p1_raw)):
            rf_p1 = float(rf_p1_raw)
            rf_correct = int((w == 1 and rf_p1 > 0.5) or (w == 2 and rf_p1 < 0.5))
            df.at[idx, 'rf_correct'] = rf_correct

        winner_name = p1 if w == 1 else p2
        model_str = ('✓' if model_correct else '✗') if not (isinstance(model_correct, float) and math.isnan(model_correct)) else 'N/A'
        print(f"  ✓ Settled: {winner_name} won | Model {model_str} | Market {'✓' if market_correct else '✗'}")
        newly_settled += 1

        settled_bets = tracker.settle_pending_bets_for_match(
            match_uid=row.get('match_uid'),
            p1=p1,
            p2=p2,
            actual_winner=w,
            notes=f"Auto-settled from Tennis Abstract | score={result['score']}",
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
