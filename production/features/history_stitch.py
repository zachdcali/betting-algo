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
    # hyphens split to spaces so "Auger-Aliassime" (ATP) matches
    # "Auger Aliassime" (TA display form); apostrophes/dots stripped
    cleaned = str(name).replace("-", " ").replace("'", "").replace(".", ". ").strip()
    parts = cleaned.split()
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
            "_stats_url": card.get("stats_url"),
        })
    return pd.DataFrame(rows)


# Bound per-run stats-page fetches so a large slate can't stall the pipeline;
# most-recent rows enrich first (they drive the Last10 stat windows).
MAX_STATS_FETCHES_PER_PLAYER = 6

_STAT_COLS = (
    "aces", "double_faults", "serve_points", "first_serves_in",
    "first_serve_won", "second_serve_won", "bp_saved", "bp_faced",
    "opp_aces", "opp_double_faults", "opp_serve_points", "opp_first_serves_in",
    "opp_first_serve_won", "opp_second_serve_won", "opp_bp_saved", "opp_bp_faced",
)


def enrich_event_rows_with_stats(df: pd.DataFrame, player_name: str,
                                 session_cache: Optional[dict] = None,
                                 _fetch=None) -> pd.DataFrame:
    """Attach real serve/BP stats to stitched event rows via their stats pages.

    Fills the exact TA stat columns so the granular (performance_v1) features
    compute on stitched rows instead of leaning on older stat-bearing matches.
    Non-fatal per row; rows without stats stay score-only (counted honestly by
    Stat_Matches_Last10).
    """
    if df is None or df.empty or "_stats_url" not in df.columns:
        return df
    cache = (session_cache if session_cache is not None else {}).setdefault("atp_match_stats", {})
    if _fetch is None:
        try:
            from atp_results_scraper import fetch_match_stats as _fetch
        except ImportError:
            from scraping.atp_results_scraper import fetch_match_stats as _fetch

    out = df.copy()
    for col in _STAT_COLS:
        if col not in out.columns:
            out[col] = float("nan")
    fetched = 0
    for idx in out.sort_values("date", ascending=False).index:
        url = out.at[idx, "_stats_url"]
        if not isinstance(url, str) or not url:
            continue
        if url not in cache:
            if fetched >= MAX_STATS_FETCHES_PER_PLAYER:
                continue
            fetched += 1
            try:
                cache[url] = _fetch(url)
            except Exception as exc:
                print(f"      ⚠️ match-stats fetch failed (non-fatal): {exc}")
                cache[url] = None
        stats = cache[url]
        if not stats:
            continue
        if _names_loosely_match(stats["p1_name"], player_name):
            side = stats["p1"]
        elif _names_loosely_match(stats["p2_name"], player_name):
            side = stats["p2"]
        else:
            continue
        for col in _STAT_COLS:
            val = side.get(col)
            if val is not None:
                out.at[idx, col] = float(val)
    return out


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
    """Static-registry entry whose window covers ref_date, else None."""
    ts = pd.Timestamp(ref_date)
    for ev in CURRENT_EVENT_REGISTRY:
        lo, hi = pd.Timestamp(ev["window"][0]), pd.Timestamp(ev["window"][1])
        if lo <= ts <= hi:
            return ev
    return None


def _monday_of(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=int(ts.weekday()))).normalize()


def get_active_events(ref_date, session_cache: Optional[dict] = None) -> list[dict]:
    """All active tournaments: auto-discovered from the ATP live-scores hubs,
    with static-registry entries overriding by slug (they carry exact
    surface/level/start dates). Discovered events get a Monday-snapped start
    date (tour events run Mon-Sun; Slams' true start comes from the registry).
    Cached per run."""
    cache = session_cache if session_cache is not None else {}
    if "atp_active_events" in cache:
        return cache["atp_active_events"]
    try:
        from atp_results_scraper import discover_active_events
    except ImportError:
        from scraping.atp_results_scraper import discover_active_events
    monday = _monday_of(pd.Timestamp(ref_date))
    events: list[dict] = []
    try:
        for ev in discover_active_events():
            ev = dict(ev)
            ev["start_date"] = str(monday.date())
            events.append(ev)
    except Exception as exc:
        print(f"      ⚠️ event discovery unavailable ({exc}); using static registry only")
    # calendar: catches events Bovada prices before the live hub lists them
    try:
        try:
            from atp_results_scraper import parse_challenger_calendar, _fetch_rendered
        except ImportError:
            from scraping.atp_results_scraper import parse_challenger_calendar, _fetch_rendered
        cal_cache = cache.setdefault("atp_calendar", {})
        if "df" not in cal_cache:
            html = _fetch_rendered("https://www.atptour.com/en/atp-challenger-tour/calendar",
                                   "a[href*='/en/scores/']")
            cal_cache["df"] = parse_challenger_calendar(html)
        cal = cal_cache["df"]
        if cal.empty:
            raise RuntimeError("challenger calendar parse returned no events")
        ref = pd.Timestamp(ref_date)
        seen_ids = {e.get("id") for e in events}
        window = cal[(pd.to_datetime(cal["start_date"]) <= ref)
                     & (pd.to_datetime(cal["start_date"]) >= ref - pd.Timedelta(days=8))]
        for _, r in window.iterrows():
            if r["id"] in seen_ids:
                continue
            events.append({"event": r["event"], "slug": r["slug"], "id": r["id"],
                           "url": r["url"], "level": "C", "surface": "",
                           "start_date": r["start_date"]})
    except Exception as exc:
        print(f"      ⚠️ calendar discovery unavailable ({exc})")
    static_slugs = set()
    for sev in CURRENT_EVENT_REGISTRY:
        lo, hi = pd.Timestamp(sev["window"][0]), pd.Timestamp(sev["window"][1])
        if lo <= pd.Timestamp(ref_date) <= hi:
            static_slugs.add(sev["event"].lower())
            events = [e for e in events if sev["event"].lower() not in e["event"].lower()
                      and e.get("slug", "") != sev["event"].lower()]
            events.append(sev)
    cache["atp_active_events"] = events
    return events


def round_from_draws(p1: str, p2: str, ref_date, session_cache: Optional[dict] = None) -> Optional[str]:
    """Resolve an upcoming match's round from active events' DRAW pages.

    Covers what result-based inference can't: first-round matches at a
    just-started event. Both names must match one bracket pairing (either
    orientation); draws are fetched lazily and cached per run.
    """
    cache = session_cache if session_cache is not None else {}
    draws_cache = cache.setdefault("atp_event_draws", {})
    try:
        from atp_results_scraper import fetch_event_draw
    except ImportError:
        from scraping.atp_results_scraper import fetch_event_draw
    for ev in get_active_events(ref_date, cache):
        url = ev["url"]
        if url not in draws_cache:
            try:
                print(f"      🧵 Fetching draw for {ev['event']} (shared, cached)")
                draws_cache[url] = fetch_event_draw(url)
            except Exception as exc:
                print(f"      ⚠️ draw fetch failed (non-fatal): {exc}")
                draws_cache[url] = pd.DataFrame()
        draw = draws_cache[url]
        if draw is None or draw.empty:
            continue
        for _, row in draw.iterrows():
            a, b = str(row["p1"]), str(row["p2"])
            if (_names_loosely_match(a, p1) and _names_loosely_match(b, p2)) or \
               (_names_loosely_match(a, p2) and _names_loosely_match(b, p1)):
                return str(row["round"])
    return None


_NEXT_ROUND_ANY_WINDOW_DAYS = 16


def infer_next_round_any(matches1: pd.DataFrame, matches2: pd.DataFrame, ref_date) -> Optional[str]:
    """Registry-free round inference: if BOTH players' newest rows are the same
    event's same round, played within the last ~two weeks of ref_date, the
    upcoming match is the next round. Any mismatch -> None (stays unresolved)."""
    tops = []
    for m in (matches1, matches2):
        if m is None or m.empty:
            return None
        top = m.iloc[0]
        d = pd.to_datetime(top.get("date"), errors="coerce")
        if pd.isna(d) or (pd.Timestamp(ref_date) - d).days > _NEXT_ROUND_ANY_WINDOW_DAYS:
            return None
        tops.append((str(top.get("event", "")).strip().lower(),
                     str(top.get("round", "")).strip().upper()))
    if tops[0][0] != tops[1][0] or not tops[0][0]:
        return None
    if tops[0][1] != tops[1][1]:
        return None
    return _NEXT_ROUND.get(tops[0][1])


_NEXT_ROUND = {"R128": "R64", "R64": "R32", "R32": "R16", "R16": "QF", "QF": "SF", "SF": "F"}


def infer_next_round(matches1: pd.DataFrame, matches2: pd.DataFrame, event_name: str) -> Optional[str]:
    """Infer the upcoming round from both players' completed rows at this event.

    Real data, not a generic fallback: if BOTH players' newest match is this
    event's round R, the match being predicted is the next round. Any mismatch
    (different rounds, different events, empty history) returns None so the row
    stays honestly unresolved.
    """
    rounds = []
    for m in (matches1, matches2):
        if m is None or m.empty:
            return None
        top = m.iloc[0]
        if str(top.get("event", "")).strip().lower() != event_name.strip().lower():
            return None
        rounds.append(str(top.get("round", "")).strip().upper())
    if rounds[0] != rounds[1]:
        return None
    return _NEXT_ROUND.get(rounds[0])


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


ITF_LEVELS = {"15", "25", "S", "F"}


def _with_itf_client(fn):
    """Run fn(client) in a short-lived ItfClient. A PERSISTENT client can't
    coexist with the pipeline's other sync_playwright users in one thread
    ('Sync API inside the asyncio loop') — so open, fetch, close. Results are
    cached per run, so this costs a handful of opens per slate."""
    try:
        from itf_results_scraper import ItfClient
    except ImportError:
        from scraping.itf_results_scraper import ItfClient
    client = ItfClient()
    try:
        return fn(client)
    finally:
        client.close()


def _itf_event_for(tournament_label: str, ref_date, cache: dict) -> Optional[dict]:
    """Map a Bovada 'ITF Men <City>' label to an ITF calendar event by location."""
    try:
        from itf_results_scraper import get_calendar
    except ImportError:
        from scraping.itf_results_scraper import get_calendar
    if "itf_calendar" not in cache:
        ref = pd.Timestamp(ref_date)
        try:
            cache["itf_calendar"] = _with_itf_client(lambda c: get_calendar(
                c,
                str((ref - pd.Timedelta(days=10)).date()),
                str((ref + pd.Timedelta(days=3)).date()),
            ))
        except Exception as exc:
            print(f"      ⚠️ ITF calendar unavailable ({exc})")
            cache["itf_calendar"] = pd.DataFrame()
    cal = cache["itf_calendar"]
    if cal.empty:
        return None
    # Bovada labels look like "ITF Men's - ITF Men Wuning (7)" — take the text
    # after the LAST 'ITF Men/Women' marker, then strip counters/punctuation
    raw = str(tournament_label)
    m = list(re.finditer(r"itf\s+(?:men|women)(?:'s)?", raw, re.I))
    city = raw[m[-1].end():] if m else raw
    city = re.sub(r"\(.*?\)", " ", city)
    city = re.sub(r"[^a-zA-Z\s]", " ", city).strip().lower()
    city = re.sub(r"\s+", " ", city)
    if not city:
        return None
    hit = cal[cal["location"].astype(str).str.lower().str.contains(city, regex=False)
              | cal["event"].astype(str).str.lower().str.contains(city, regex=False)]
    return hit.iloc[0].to_dict() if len(hit) >= 1 else None


def _itf_event_matches(key: str, cache: dict) -> pd.DataFrame:
    matches_cache = cache.setdefault("itf_event_matches", {})
    if key not in matches_cache:
        try:
            from itf_results_scraper import get_event_matches
        except ImportError:
            from scraping.itf_results_scraper import get_event_matches
        try:
            print(f"      🧵 Fetching ITF order-of-play for {key} (shared, cached)")
            matches_cache[key] = _with_itf_client(lambda c: get_event_matches(c, key))
        except Exception as exc:
            print(f"      ⚠️ ITF event fetch failed (non-fatal): {exc}")
            matches_cache[key] = pd.DataFrame()
    return matches_cache[key]


def gather_itf_rows(display_name: str, tournament_label: str, ref_date,
                    session_cache: Optional[dict] = None) -> pd.DataFrame:
    """Completed ITF rows for a player at the current event, TA schema.

    Dates use the event's start date (tournament-start convention = training
    parity); rows carry source='itf_results'. Score-only (no serve stats —
    ITF's API publishes scores; coverage features count this honestly).
    """
    cache = session_cache if session_cache is not None else {}
    ev = _itf_event_for(tournament_label, ref_date, cache)
    if not ev:
        return pd.DataFrame()
    em = _itf_event_matches(ev["key"], cache)
    if em.empty:
        return pd.DataFrame()
    done = em[em["completed"] & em["winner"].notna()]
    rows = []
    for _, m in done.iterrows():
        if _names_loosely_match(m["p1"], display_name):
            side = 1
        elif _names_loosely_match(m["p2"], display_name):
            side = 2
        else:
            continue
        won = int(m["winner"]) == side
        opp = m["p2"] if side == 1 else m["p1"]
        rows.append({
            "date": pd.Timestamp(ev["start_date"]),
            "event": ev["event"],
            "surface": ev.get("surface", ""),
            "round": m["round"] or "",
            "level": "25" if "25" in str(ev.get("category", "")) else "15",
            "rank": float("nan"),
            "opp_name": opp,
            "opp_rank": float("nan"),
            "opp_hand": "",
            "opp_country": "",
            "score": m["score"],  # winner-first already
            "result": "W" if won else "L",
            "source": "itf_results",
        })
    return pd.DataFrame(rows)


def itf_round_for(p1_name: str, p2_name: str, tournament_label: str, ref_date,
                  session_cache: Optional[dict] = None) -> Optional[str]:
    """Round of an upcoming ITF match from the event's order of play."""
    cache = session_cache if session_cache is not None else {}
    ev = _itf_event_for(tournament_label, ref_date, cache)
    if not ev:
        return None
    em = _itf_event_matches(ev["key"], cache)
    if em.empty:
        return None
    for _, m in em.iterrows():
        pair = ((_names_loosely_match(m["p1"], p1_name) and _names_loosely_match(m["p2"], p2_name))
                or (_names_loosely_match(m["p1"], p2_name) and _names_loosely_match(m["p2"], p1_name)))
        if pair and not m["completed"]:
            return m["round"]
    # completed same-pair match also pins the round (late odds on a finished match)
    for _, m in em.iterrows():
        pair = ((_names_loosely_match(m["p1"], p1_name) and _names_loosely_match(m["p2"], p2_name))
                or (_names_loosely_match(m["p1"], p2_name) and _names_loosely_match(m["p2"], p1_name)))
        if pair:
            return m["round"]
    return None


def gather_atp_rows(display_name: str, ref_date, rankings_df: Optional[pd.DataFrame],
                    session_cache: Optional[dict] = None,
                    level_hint: str = "") -> pd.DataFrame:
    """Fetch + normalize all available ATP rows for a player (cached per run).

    Sources: the active in-progress event's results page (one fetch shared by
    every player in the run) and the player's activity page (one fetch per
    player). Failures are non-fatal — we return what we could get.
    """
    cache = session_cache if session_cache is not None else {}
    frames = []

    try:
        from atp_results_scraper import fetch_tournament_results
    except ImportError:
        from scraping.atp_results_scraper import fetch_tournament_results
    results_cache = cache.setdefault("atp_event_results", {})
    meta_cache = cache.setdefault("atp_event_meta", {})
    for ev in get_active_events(ref_date, cache):
        if ev["url"] not in results_cache:
            try:
                print(f"      🧵 Fetching ATP results page for {ev['event']} (shared, cached)")
                results_cache[ev["url"]] = fetch_tournament_results(ev["url"])
                meta_cache[ev["url"]] = ev
            except Exception as exc:
                print(f"      ⚠️ ATP event results fetch failed (non-fatal): {exc}")
                results_cache[ev["url"]] = pd.DataFrame()
        ev_rows = event_results_to_ta_schema(results_cache[ev["url"]], display_name, ev)
        if ev_rows.empty:
            continue
        ev_rows = enrich_event_rows_with_stats(ev_rows, display_name, cache)
        if "_stats_url" in ev_rows.columns:
            ev_rows = ev_rows.drop(columns=["_stats_url"])
        frames.append(ev_rows)

    # atptour activity pages only list tour/Challenger events — for ITF-level
    # matches the fetch cannot return gap rows, so skip the ~25s browser trip.
    url = None if str(level_hint) in ITF_LEVELS else lookup_player_url(display_name, rankings_df)
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
