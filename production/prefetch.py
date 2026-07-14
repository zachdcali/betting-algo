"""Parallel prefetch of event pages into the session cache.

The per-match pipeline resolves rounds/surfaces/history by lazily fetching
event pages (results, draws, daily schedule, ITF order-of-play) the first time
each is needed — correct, but sequential. On week-boundary days a run
discovers ~15 fresh tournaments and those first-touch fetches dominated wall
clock (40-minute Sunday runs).

This module warms the same caches up front with a small thread pool. Browsers
are thread-local (scraping/browser_session.py), so each worker owns its own
Chromium; results are merged into the session cache on the caller's thread —
downstream code finds warm keys and never refetches. Every failure is loud and
non-fatal: a missed prefetch just means that page falls back to the old lazy
path.

ITF is fetched with fewer workers and a stagger — itftennis.com sits behind
Incapsula and hammering it from one IP invites session churn.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scraping"))
sys.path.insert(0, str(Path(__file__).parent / "features"))

import pandas as pd

ATP_WORKERS = 3
ITF_WORKERS = 2
ITF_STAGGER_S = 1.0


def _fetch_atp_bundle(ev: dict) -> tuple[str, dict]:
    from atp_results_scraper import (fetch_tournament_results, fetch_event_draw,
                                     fetch_daily_schedule)
    url = ev["url"]
    out: dict = {}
    errors: list[str] = []
    for key, fn in (("results", fetch_tournament_results),
                    ("draw", fetch_event_draw),
                    ("schedule", fetch_daily_schedule)):
        try:
            out[key] = fn(url)
        except Exception as exc:
            print(f"      ⚠️ prefetch {key} failed for {ev.get('event','?')} (non-fatal): {exc}")
            errors.append(key)
    out["_errors"] = errors
    return url, out


def _fetch_itf_event(key: str) -> tuple[str, pd.DataFrame]:
    """Fresh client per event, one retry on a second fresh client — the same
    rotation pattern that fixed the backfill's Incapsula churn."""
    from itf_results_scraper import ItfClient, get_event_matches
    time.sleep(ITF_STAGGER_S)
    last_exc = None
    for attempt in (1, 2):
        client = ItfClient()
        try:
            return key, get_event_matches(client, key)
        except Exception as exc:
            last_exc = exc
            time.sleep(3)
        finally:
            try:
                client.close()
            except Exception:
                pass
    raise last_exc


def prefetch_event_pages(session_cache: dict, ref_date=None) -> dict:
    """Discover active events, then warm results/draw/schedule/OOP caches in
    parallel. Returns a small stats dict for the run log."""
    from history_stitch import cache_event_ingest_metadata, get_active_events

    ref = pd.Timestamp(ref_date) if ref_date is not None else pd.Timestamp.today()
    t0 = time.time()
    events = get_active_events(ref, session_cache)  # sequential: 3 calendar pages

    results_cache = session_cache.setdefault("atp_event_results", {})
    draws_cache = session_cache.setdefault("atp_event_draws", {})
    sched_cache = session_cache.setdefault("atp_event_schedules", {})

    todo = [ev for ev in events if ev["url"] not in results_cache
            or ev["url"] not in draws_cache or ev["url"] not in sched_cache]
    stats = {"atp_events": len(todo), "itf_events": 0, "errors": 0}
    if todo:
        with ThreadPoolExecutor(max_workers=ATP_WORKERS, thread_name_prefix="atp-prefetch") as pool:
            futures = {pool.submit(_fetch_atp_bundle, ev): ev for ev in todo}
            for fut in as_completed(futures):
                ev = futures[fut]
                try:
                    url, bundle = fut.result()
                except Exception as exc:
                    print(f"      ⚠️ prefetch bundle failed for {ev.get('event','?')}: {exc}")
                    stats["errors"] += 1
                    continue
                # Only warm validated, non-empty responses. An exception or
                # parse-empty result must leave the key absent so the existing
                # lazy path gets one real fallback attempt.
                for key, cache in (
                    ("results", results_cache),
                    ("draw", draws_cache),
                    ("schedule", sched_cache),
                ):
                    value = bundle.get(key)
                    if isinstance(value, pd.DataFrame) and not value.empty:
                        cache.setdefault(url, value)
                    else:
                        stats["errors"] += 1
                if url in results_cache:
                    cache_event_ingest_metadata(session_cache, ev)

    # ITF: the calendar is already in cache via get_active_events'
    # history-stitch helpers only when an ITF label was seen; warm it here
    # explicitly so event fan-out has keys to work with
    try:
        from history_stitch import _itf_event_for  # populates itf_calendar
        _itf_event_for("ITF Men __warm__", ref, session_cache)
    except Exception:
        pass
    cal = session_cache.get("itf_calendar")
    em_cache = session_cache.setdefault("itf_event_matches", {})
    if cal is not None and not getattr(cal, "empty", True):
        live = cal[(pd.to_datetime(cal["start_date"]) <= ref + pd.Timedelta(days=1))
                   & (pd.to_datetime(cal["end_date"]) >= ref - pd.Timedelta(days=1))]
        keys = [k for k in live["key"].astype(str) if k not in em_cache]
        stats["itf_events"] = len(keys)
        if keys:
            with ThreadPoolExecutor(max_workers=ITF_WORKERS, thread_name_prefix="itf-prefetch") as pool:
                futures = {pool.submit(_fetch_itf_event, k): k for k in keys}
                for fut in as_completed(futures):
                    k = futures[fut]
                    try:
                        key, em = fut.result()
                        em_cache.setdefault(key, em)
                    except Exception as exc:
                        print(f"      ⚠️ ITF prefetch failed for {k} (non-fatal): {str(exc)[:60]}")
                        stats["errors"] += 1
    stats["seconds"] = round(time.time() - t0, 1)
    print(f"  ⚡ prefetch: {stats['atp_events']} ATP + {stats['itf_events']} ITF events "
          f"warmed in {stats['seconds']}s ({stats['errors']} errors)")
    return stats
