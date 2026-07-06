"""Stitch ATP-sourced match rows onto a Tennis Abstract match history.

TA can lag days-to-weeks behind (it posted nothing between Halle and
mid-Wimbledon 2026), which corrupts the temporal features (days-since-last,
streaks, form windows) for any player whose latest matches are missing. This
module appends the missing matches from atptour.com — real results with honest
provenance (``source`` column), never silent defaults.

Two ATP sources (see scraping/atp_results_scraper.py):
- player-activity pages → completed tournaments (rows carry tournament start
  date + round, same date convention TA uses)
- an in-progress event's results page → current-tournament rows (assigned the
  event's start date, matching TA's tournament-start convention, then the
  calculator's ``_apply_round_offsets`` treats them identically to TA rows)

Dedupe rule: an ATP row is appended only if TA has no row within ±3 days with
the same opponent (last-name match). On overlap TA wins, so once TA catches up
its rows take precedence automatically and reconciliation can compare sources.

Columns that only TA provides (own rank, opponent hand/rank, per-match stats)
are left as NaN on ATP rows — rank/hand-based features must keep computing on
TA rows only, never silently on NaN (see AGENTS.md no-silent-fallbacks).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import pandas as pd

# In-progress events whose results pages we can stitch from. start_date uses
# TA's convention (tournament start for every round). Extend as events come up;
# a future build derives this from ATP's results archive automatically.
CURRENT_EVENT_REGISTRY = [
    {
        "event": "Wimbledon",
        "url": "https://www.atptour.com/en/scores/current/wimbledon/540/results",
        "start_date": "2026-06-29",
        "surface": "Grass",
        "level": "G",
        "window": ("2026-06-29", "2026-07-13"),
    },
]

# Stitch only when TA's newest row is at least this many days older than the
# reference date — below this we cannot distinguish "TA lags" from "player rested".
STALENESS_THRESHOLD_DAYS = 12

# Chronological order of rounds within one tournament (later round = higher).
_ROUND_ORDER = {
    "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4,
    "R128": 5, "R64": 6, "R32": 7, "R16": 8,
    "RR": 8, "QF": 9, "SF": 10, "BR": 11, "F": 12,
}


def _last_name(name: str) -> str:
    parts = str(name).strip().split()
    return parts[-1].lower() if parts else ""


def _names_loosely_match(a: str, b: str) -> bool:
    """Match 'J. Cerundolo' ~ 'Juan Manuel Cerundolo', 'C. Tabur' ~ 'Clement Tabur'."""
    la, lb = _last_name(a), _last_name(b)
    if not la or not lb:
        return False
    if la == lb:
        ia = str(a).strip()[0].lower()
        ib = str(b).strip()[0].lower()
        return ia == ib or "." in a or "." in b  # initials tolerated
    return False


def _flip_score_if_loss(score: str, result: str) -> str:
    """TA stores scores winner-first; activity-page rows are own-first.

    For losses, flip each 'a-b' set so the stitched row matches TA convention.
    """
    if result != "L" or not score:
        return score
    flipped = []
    for s in str(score).split():
        m = re.match(r"^(\d+(?:\(\d+\))?)-(\d+(?:\(\d+\))?)$", s)
        flipped.append(f"{m.group(2)}-{m.group(1)}" if m else s)
    return " ".join(flipped)


def _parse_activity_date(raw: str) -> Optional[pd.Timestamp]:
    """'25 May, 26' -> 2026-05-25."""
    try:
        return pd.to_datetime(datetime.strptime(str(raw).strip(), "%d %B, %y"))
    except Exception:
        try:
            return pd.to_datetime(datetime.strptime(str(raw).strip(), "%d %b, %y"))
        except Exception:
            return None


_SLAM_LEVELS = {"wimbledon": "G", "roland garros": "G", "australian open": "G", "us open": "G"}


def activity_to_ta_schema(activity_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize player-activity rows to the TA match-history schema."""
    if activity_df is None or activity_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in activity_df.iterrows():
        result = r.get("result")
        if result not in ("W", "L"):
            continue  # unresolvable rows (odd tiebreak streams) stay out — no guesses
        date = _parse_activity_date(r.get("start_date", ""))
        if date is None:
            continue
        event = str(r.get("event", ""))
        rows.append({
            "date": date,
            "event": event,
            "surface": str(r.get("surface", "")).title(),
            "round": str(r.get("round", "")),
            "level": _SLAM_LEVELS.get(event.lower(), ""),
            "rank": float("nan"),
            "opp_name": str(r.get("opponent", "")),
            "opp_rank": float("nan"),
            "opp_hand": "",
            "opp_country": "",
            "score": _flip_score_if_loss(str(r.get("score", "")), result),
            "result": result,
            "source": "atp_activity",
        })
    return pd.DataFrame(rows)


def event_results_to_ta_schema(results_df: pd.DataFrame, player_name: str, event: dict) -> pd.DataFrame:
    """Rows for one player from an in-progress event's results page."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    rows = []
    for _, card in results_df.iterrows():
        winner = card.get("winner")
        if winner not in (1, 2):
            continue
        p1, p2 = str(card["p1"]), str(card["p2"])
        if _names_loosely_match(p1, player_name):
            own_side = 1
        elif _names_loosely_match(p2, player_name):
            own_side = 2
        else:
            continue
        opp = p2 if own_side == 1 else p1
        result = "W" if winner == own_side else "L"
        w_sets = card["p1_sets"] if winner == 1 else card["p2_sets"]
        l_sets = card["p2_sets"] if winner == 1 else card["p1_sets"]
        score = " ".join(
            f"{a}-{b}" for a, b in zip(str(w_sets).split(), str(l_sets).split())
        )  # winner-first, TA convention
        rows.append({
            "date": pd.to_datetime(event["start_date"]),
            "event": event["event"],
            "surface": event["surface"],
            "round": str(card.get("round", "")),
            "level": event.get("level", ""),
            "rank": float("nan"),
            "opp_name": opp,
            "opp_rank": float("nan"),
            "opp_hand": "",
            "opp_country": "",
            "score": score,
            "result": result,
            "source": "atp_event_results",
        })
    return pd.DataFrame(rows)


def stitch_history(ta_df: pd.DataFrame, atp_df: pd.DataFrame) -> pd.DataFrame:
    """Append ATP rows TA doesn't have. TA rows always win on overlap.

    Overlap = TA row within ±3 days with the same opponent (loose name match).
    Returns a frame with a ``source`` column ('ta' for original rows).
    """
    if ta_df is None:
        ta_df = pd.DataFrame()
    out = ta_df.copy()
    if "source" not in out.columns:
        out["source"] = "ta"
    if atp_df is None or atp_df.empty:
        return out

    ta_dates = pd.to_datetime(ta_df["date"], errors="coerce") if not ta_df.empty and "date" in ta_df.columns else pd.Series(dtype="datetime64[ns]")
    keep = []
    for _, row in atp_df.iterrows():
        dup = False
        if not ta_df.empty:
            window = (ta_dates - row["date"]).abs() <= pd.Timedelta(days=3)
            for _, ta_row in ta_df[window.fillna(False)].iterrows():
                if _names_loosely_match(str(ta_row.get("opp_name", "")), str(row["opp_name"])):
                    dup = True
                    break
        if not dup:
            keep.append(row)
    if not keep:
        return out
    stitched = pd.concat([out, pd.DataFrame(keep)], ignore_index=True)
    # TA emits string dates, ATP rows are Timestamps — normalize before sorting
    # (downstream consumers all coerce with pd.to_datetime themselves)
    stitched["date"] = pd.to_datetime(stitched["date"], errors="coerce")
    # TA's contract is MOST-RECENT-FIRST and rows within a tournament share the
    # tournament-start date, so order ties by round (final first). Without this,
    # streak/last_surface would read an arbitrary within-event row as "latest".
    stitched["_round_order"] = stitched["round"].map(_ROUND_ORDER).fillna(0)
    stitched = stitched.sort_values(
        ["date", "_round_order"], ascending=[False, False], kind="stable",
    ).drop(columns=["_round_order"])
    return stitched.reset_index(drop=True)


def needs_stitching(ta_df: pd.DataFrame, ref_date) -> bool:
    """True when TA's newest row is stale relative to the prediction date."""
    if ta_df is None or ta_df.empty or "date" not in ta_df.columns:
        return True
    newest = pd.to_datetime(ta_df["date"], errors="coerce").max()
    if pd.isna(newest):
        return True
    return (pd.Timestamp(ref_date) - newest).days > STALENESS_THRESHOLD_DAYS


def active_event_for(ref_date) -> Optional[dict]:
    """Registry entry whose window covers ref_date, else None."""
    ts = pd.Timestamp(ref_date)
    for ev in CURRENT_EVENT_REGISTRY:
        lo, hi = pd.Timestamp(ev["window"][0]), pd.Timestamp(ev["window"][1])
        if lo <= ts <= hi:
            return ev
    return None


def lookup_player_url(name: str, rankings_df: Optional[pd.DataFrame]) -> Optional[str]:
    """Find a player's atptour.com profile URL in the rankings CSV.

    Rankings names are abbreviated ('J. Sinner'); match on last name, then
    disambiguate by first initial. None when not uniquely resolvable.
    """
    if rankings_df is None or rankings_df.empty or "player_url" not in rankings_df.columns:
        return None
    last = _last_name(name)
    if not last:
        return None
    cand = rankings_df[rankings_df["player_name"].astype(str).str.lower().str.endswith(" " + last)]
    if len(cand) > 1:
        initial = str(name).strip()[0].lower()
        cand = cand[cand["player_name"].astype(str).str.lower().str.startswith(initial)]
    if len(cand) != 1:
        return None
    url = str(cand.iloc[0]["player_url"])
    return url or None


def gather_atp_rows(display_name: str, ref_date, rankings_df: Optional[pd.DataFrame],
                    session_cache: Optional[dict] = None) -> pd.DataFrame:
    """Fetch + normalize all available ATP rows for a player (cached per run).

    Sources: the active in-progress event's results page (one fetch shared by
    every player in the run) and the player's activity page (one fetch per
    player). Failures are non-fatal — we return what we could get.
    """
    cache = session_cache if session_cache is not None else {}
    frames = []

    ev = active_event_for(ref_date)
    if ev is not None:
        results_cache = cache.setdefault("atp_event_results", {})
        if ev["url"] not in results_cache:
            try:
                from atp_results_scraper import fetch_tournament_results
            except ImportError:
                from scraping.atp_results_scraper import fetch_tournament_results
            try:
                print(f"      🧵 Fetching ATP results page for {ev['event']} (shared, cached)")
                results_cache[ev["url"]] = fetch_tournament_results(ev["url"])
            except Exception as exc:
                print(f"      ⚠️ ATP event results fetch failed (non-fatal): {exc}")
                results_cache[ev["url"]] = pd.DataFrame()
        frames.append(event_results_to_ta_schema(results_cache[ev["url"]], display_name, ev))

    url = lookup_player_url(display_name, rankings_df)
    if url:
        act_cache = cache.setdefault("atp_activity", {})
        if url not in act_cache:
            try:
                from atp_results_scraper import fetch_player_activity
            except ImportError:
                from scraping.atp_results_scraper import fetch_player_activity
            try:
                print(f"      🧵 Fetching ATP activity page for {display_name}")
                act_cache[url] = fetch_player_activity(url)
            except Exception as exc:
                print(f"      ⚠️ ATP activity fetch failed (non-fatal): {exc}")
                act_cache[url] = pd.DataFrame()
        frames.append(activity_to_ta_schema(act_cache[url]))

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # the event-results page and the activity page can both carry the same match
    # (once the event completes); keep the first occurrence (event-results rows
    # come first and carry the registry's surface/level)
    out["_opp_last"] = out["opp_name"].map(_last_name)
    out = out.drop_duplicates(subset=["date", "round", "_opp_last"], keep="first").drop(columns=["_opp_last"])
    return out.reset_index(drop=True)
