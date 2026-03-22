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
  5. Cache all results to data/atp_heights.json.

Usage (standalone test):
    python scraping/atp_height_scraper.py "Jacob Fearnley"
"""

from playwright.sync_api import sync_playwright
import json
import re
import time
import pandas as pd
from pathlib import Path
from typing import Optional

CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "atp_heights.json"
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

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(name: str) -> str:
    return name.strip().lower()


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


def _scrape_profile(page, profile_url: str) -> Optional[int]:
    """Navigate to an ATP overview page and extract height."""
    full_url = ATP_BASE + profile_url if profile_url.startswith("/") else profile_url
    try:
        page.goto(full_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(5)
        return _extract_height_cm(page.inner_text("body"))
    except Exception as e:
        print(f"  ATP profile page error ({full_url}): {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_height_cm(player_name: str, cache: Optional[dict] = None) -> Optional[int]:
    """
    Return height in cm for player_name from ATP website.
    Checks cache first; only launches Playwright when needed.
    """
    own_cache = cache is None
    if cache is None:
        cache = _load_cache()

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
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.set_extra_http_headers({"User-Agent": _UA})
        height = _scrape_profile(pg, bio_url)
        browser.close()

    cache[key] = height
    if own_cache:
        _save_cache(cache)

    if height is not None:
        print(f"  ATP height found: {player_name} → {height}cm")
    else:
        print(f"  ATP height not found: {player_name}")
    return height


def batch_get_heights(player_names: list, verbose: bool = True) -> dict:
    """
    Fetch heights for multiple players in a single Playwright browser session.
    Returns {name: height_cm_or_None}.
    New results are merged into the persistent cache.
    """
    cache = _load_cache()
    results = {}
    to_scrape = []

    for name in player_names:
        key = _cache_key(name)
        if key in cache:
            results[name] = cache[key]
        else:
            to_scrape.append(name)

    if not to_scrape:
        return results

    url_map = _load_url_map()

    # Split into those with known URLs vs those without
    with_url = [(n, _find_profile_url(n, url_map)) for n in to_scrape]
    needs_scraping = [(n, u) for n, u in with_url if u]
    no_url = [n for n, u in with_url if not u]

    for name in no_url:
        results[name] = None
        cache[_cache_key(name)] = None
        if verbose:
            print(f"  ATP: no URL for '{name}' — skipping height lookup")

    if not needs_scraping:
        _save_cache(cache)
        return results

    if verbose:
        print(f"  ATP height scraper: fetching {len(needs_scraping)} players...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.set_extra_http_headers({"User-Agent": _UA})

        for name, bio_url in needs_scraping:
            h = _scrape_profile(pg, bio_url)
            results[name] = h
            cache[_cache_key(name)] = h
            if verbose:
                status = f"{h}cm" if h else "not found"
                print(f"    {name}: {status}")

        browser.close()

    _save_cache(cache)
    return results


if __name__ == "__main__":
    import sys
    name = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Jacob Fearnley"
    h = get_height_cm(name)
    print(f"\nResult: {name} → {h}cm" if h else f"\nResult: {name} → not found")
