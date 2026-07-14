#!/usr/bin/env python3
"""
ATP Height Scraper

Fallback height lookup from atptour.com for players where Tennis Abstract
returns height_cm=None.

Strategy:
  1. Load player profile URLs from the ATP rankings CSV (populated by atp_rankings_scraper.py).
  2. Fuzzy-match the player name to find their URL.
  3. Navigate to their /bio page with Playwright (JS-rendered).
  4. Extract height in cm via regex.
  5. In today's default legacy mode, retain the existing JSON cache behavior.
     After explicit eligibility cutover (ELIGIBILITY_PROVENANCE_MODE=required),
     read only a schema/generation-pinned ops export and never write locally.

Usage (standalone test):
    python -m production.scraping.atp_height_scraper "Jacob Fearnley"
"""

import json
import os
import re
import time
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from storage.eligibility import (
        ELIGIBILITY_GENERATION_ENV, ELIGIBILITY_PROJECTION_SEAL_ENV,
        EligibilityContractError, EligibilityMode, eligibility_mode,
    )
    from eligibility_cache import (
        VerifiedEligibilityBundle, load_verified_profile_bundle,
    )
except ImportError:  # pragma: no cover - package-style execution
    from production.storage.eligibility import (  # type: ignore
        ELIGIBILITY_GENERATION_ENV, ELIGIBILITY_PROJECTION_SEAL_ENV,
        EligibilityContractError, EligibilityMode, eligibility_mode,
    )
    from production.eligibility_cache import (  # type: ignore
        VerifiedEligibilityBundle, load_verified_profile_bundle,
    )

CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "atp_heights.json"
HANDS_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "atp_hands.json"
CACHE_MANIFEST_PATH = (
    Path(__file__).parent.parent.parent / "data" / "eligibility_cache_manifest.json"
)
RANKINGS_PATH = Path(__file__).parent.parent.parent / "data" / "atp_rankings.csv"
ATP_BASE = "https://www.atptour.com"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_required_bundle() -> Optional[VerifiedEligibilityBundle]:
    """Load the all-or-nothing accepted bundle configured for this process."""
    generation = os.environ.get(ELIGIBILITY_GENERATION_ENV, "").strip().lower()
    seal = os.environ.get(ELIGIBILITY_PROJECTION_SEAL_ENV, "").strip().lower()
    if not generation or not seal:
        return None
    return load_verified_profile_bundle(
        output_dir=CACHE_MANIFEST_PATH.parent,
        expected_generation_sha256=generation,
        expected_projection_seal_sha256=seal,
    )


def _load_cache() -> dict:
    if _provenance_required():
        # Required mode must never expose a plain cache dictionary as accepted
        # evidence. Public lookups consume the verified ID-bearing bundle.
        return {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    if _provenance_required():
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(name: str) -> str:
    return name.strip().lower()


def _load_hands_cache() -> dict:
    if _provenance_required():
        return {}
    if HANDS_CACHE_PATH.exists():
        with open(HANDS_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_hands_cache(cache: dict):
    if _provenance_required():
        return
    HANDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HANDS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _provenance_required() -> bool:
    return eligibility_mode() is EligibilityMode.REQUIRED


def _new_browser_page():
    """Resolve BrowserSession in both package and legacy script execution."""
    try:
        from .browser_session import new_page
    except ImportError:  # pragma: no cover - legacy production/ on sys.path
        from browser_session import new_page
    return new_page()


# ---------------------------------------------------------------------------
# URL lookup from rankings CSV
# ---------------------------------------------------------------------------

def _load_url_map() -> dict:
    """
    Build {normalized_name: bio_url} from atp_rankings.csv.
    Requires the CSV to have a player_url column (populated when rankings
    scraper is re-run after the atp_rankings_scraper.py update).
    """
    if not RANKINGS_PATH.exists():
        return {}
    df = pd.read_csv(RANKINGS_PATH)
    if "player_url" not in df.columns:
        return {}

    url_map = {}
    for _, row in df.iterrows():
        url = row.get("player_url")
        name = str(row.get("player_name", "")).strip()
        if pd.isna(url) or not url or not name:
            continue
        url_map[name.lower()] = str(url)
    return url_map


def _find_profile_url(player_name: str, url_map: dict) -> Optional[str]:
    """
    Find ATP bio URL for player_name. Tries:
      1. Exact name match (case-insensitive)
      2. Last-name match when unique
      3. Last-name + first-initial match
    """
    name_lower = player_name.strip().lower()

    if name_lower in url_map:
        return url_map[name_lower]

    parts = name_lower.split()
    if not parts:
        return None

    last = parts[-1]
    first_initial = parts[0][0] if parts else ""

    # Last-name candidates
    candidates = {k: v for k, v in url_map.items() if k.split()[-1] == last}
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if len(candidates) > 1 and first_initial:
        narrowed = {k: v for k, v in candidates.items() if k[0] == first_initial}
        if len(narrowed) == 1:
            return next(iter(narrowed.values()))

    return None


# ---------------------------------------------------------------------------
# Height extraction
# ---------------------------------------------------------------------------

def _extract_height_cm(text: str) -> Optional[int]:
    """Extract height in cm from rendered bio page text."""
    # ATP shows: 6'0" (183cm) or 183 cm or 183cm
    m = re.search(r'(\d{2,3})\s*cm', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 150 <= val <= 230:
            return val
    return None


def _extract_hand(text: str) -> Optional[str]:
    """'Plays: Right-Handed' / 'Left-Handed' on the bio page -> 'R'/'L'."""
    m = re.search(r"(right|left)\s*-?\s*handed", text, re.IGNORECASE)
    if m:
        return "R" if m.group(1).lower() == "right" else "L"
    return None


def _fetch_profile_text(page, profile_url: str) -> str:
    full_url = ATP_BASE + profile_url if profile_url.startswith("/") else profile_url
    try:
        page.goto(full_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(5)
        return page.inner_text("body")
    except Exception as e:
        print(f"  ATP profile page error ({full_url}): {e}")
        return ""


def _scrape_profile(page, profile_url: str) -> Optional[int]:
    """Navigate to an ATP overview page and extract height."""
    return _extract_height_cm(_fetch_profile_text(page, profile_url))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_height_cm(
    player_name: str,
    cache: Optional[dict | VerifiedEligibilityBundle] = None,
) -> Optional[int]:
    """
    Return height in cm for player_name from ATP website.
    Checks cache first; only launches Playwright when needed.
    """
    if _provenance_required():
        if cache is not None:
            raise EligibilityContractError(
                "required eligibility mode rejects caller-supplied cache objects"
            )
        bundle = _load_required_bundle()
        profile = None if bundle is None else bundle.profile_for(player_name)
        value = None if profile is None else profile.get("height_cm")
        return None if value is None else int(float(value))

    own_cache = cache is None
    if cache is None:
        cache = _load_cache()
    assert isinstance(cache, dict)

    key = _cache_key(player_name)
    if key in cache:
        return cache[key]
    url_map = _load_url_map()
    bio_url = _find_profile_url(player_name, url_map)
    if not bio_url:
        print(f"  ATP: no URL found for '{player_name}' (re-run atp_rankings_scraper.py to refresh)")
        cache[key] = None
        if own_cache:
            _save_cache(cache)
        return None

    print(f"  ATP height lookup: {player_name}")
    pg = _new_browser_page()
    try:
        height = _scrape_profile(pg, bio_url)
    finally:
        pg.close()

    cache[key] = height
    if own_cache:
        _save_cache(cache)

    if height is not None:
        print(f"  ATP height found: {player_name} → {height}cm")
    else:
        print(f"  ATP height not found: {player_name}")
    return height


def batch_get_profiles(player_names: list, verbose: bool = True) -> dict:
    """
    Fetch height AND handedness for multiple players, one page fetch each,
    shared browser. Returns {name: {"height_cm": int|None, "hand": 'R'/'L'/None}}.
    Default legacy mode retains today's persistent caches. Required cutover
    mode reads only a generation-pinned export and returns the accepted
    canonical player ID with each profile so callers can reject identity
    mismatches. Fresh values remain in-memory pending normalized
    ingestion/review.
    """
    if _provenance_required():
        bundle = _load_required_bundle()
        results: dict = {}
        for name in player_names:
            profile = None if bundle is None else bundle.profile_for(name)
            results[name] = {
                "canonical_player_id": (
                    None if profile is None else profile.get("canonical_player_id")
                ),
                "height_cm": None if profile is None else profile.get("height_cm"),
                "hand": None if profile is None else profile.get("hand"),
            }
        return results

    h_cache = _load_cache()
    hd_cache = _load_hands_cache()
    results: dict = {}
    to_scrape = []

    for name in player_names:
        key = _cache_key(name)
        if key in h_cache and key in hd_cache:
            results[name] = {"height_cm": h_cache[key], "hand": hd_cache[key]}
        else:
            to_scrape.append(name)

    if not to_scrape:
        return results
    url_map = _load_url_map()
    with_url = [(n, _find_profile_url(n, url_map)) for n in to_scrape]
    needs_scraping = [(n, u) for n, u in with_url if u]
    no_url = [n for n, u in with_url if not u]

    for name in no_url:
        results[name] = {"height_cm": h_cache.get(_cache_key(name)),
                         "hand": hd_cache.get(_cache_key(name))}
        h_cache.setdefault(_cache_key(name), None)
        hd_cache.setdefault(_cache_key(name), None)
        if verbose:
            print(f"  ATP: no URL for '{name}' — skipping profile lookup")

    if needs_scraping:
        if verbose:
            print(f"  ATP profile scraper: fetching {len(needs_scraping)} players...")
        pg = _new_browser_page()
        try:
            for name, bio_url in needs_scraping:
                text = _fetch_profile_text(pg, bio_url)
                h = _extract_height_cm(text)
                hd = _extract_hand(text)
                results[name] = {"height_cm": h, "hand": hd}
                h_cache[_cache_key(name)] = h
                hd_cache[_cache_key(name)] = hd
                if verbose:
                    print(f"    {name}: {h or '?'}cm hand={hd or '?'}")
        finally:
            pg.close()

    _save_cache(h_cache)
    _save_hands_cache(hd_cache)
    return results


def batch_get_heights(player_names: list, verbose: bool = True) -> dict:
    """Back-compat wrapper: heights only."""
    profs = batch_get_profiles(player_names, verbose=verbose)
    return {n: (v or {}).get("height_cm") for n, v in profs.items()}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Resolve one ATP player height")
    parser.add_argument("player_name", nargs="+", help="ATP player name")
    args = parser.parse_args(argv)
    name = " ".join(args.player_name)
    h = get_height_cm(name)
    print(f"\nResult: {name} → {h}cm" if h else f"\nResult: {name} → not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
