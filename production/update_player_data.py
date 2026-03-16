#!/usr/bin/env python3
# update_player_data.py
#
# Men-only, upcoming-only Bovada slate → update/append each player's
# UTR profile, rating history, and match history (year-by-year).
#
# Usage example (8 workers, dynamic years + backfill 3 seasons, long timeouts):
#   export UTR_EMAIL="your_email"
#   export UTR_PASSWORD="your_password"
#   python update_player_data.py \
#       --from-bovada \
#       --years 2025 2024 \
#       --dynamic-years \
#       --backfill-years 3 \
#       --max-workers 8 \
#       --force-ratings \
#       --headless \
#       --nav-timeout-ms 180000 \
#       --wait-timeout-ms 90000
#
# Notes:
# - Requires your utr_scraper_cloud. We call ONLY its actual methods:
#     start_browser, close_browser, login, search_player, get_player_profile,
#     get_player_rating_history, get_player_match_history,
#     select_year_from_dropdown, select_singles_from_dropdown,
#     select_verified_singles, update_player_mapping, update_tournament_mapping
# - Appends to CSVs (creates them if missing) and de-dupes by `match_id`.

import os
import re
import sys
import csv
import time
import math
import json
import argparse
import asyncio
import random as rnd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from multiprocessing import Pool, get_context

import pandas as pd

# Add path to utr_scraper_cloud
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scraping"))

from utr_scraper_cloud import UTRScraper, logger
from playwright.async_api import async_playwright

# ---------------------------
# Self-healing player mapping helpers
# ---------------------------

def append_player_mapping_row(data_dir: Path, utr_id: str, primary_name: str, bovada_name: str = ""):
    """
    Append a single row to player_mapping.csv (creates file if missing).
    Thread-safe for parallel workers via file locking would be ideal, but for now we assume
    main.py calls update_slate_players sequentially.
    """
    mapping_file = data_dir / "player_mapping.csv"
    file_exists = mapping_file.exists()

    # Read existing to check for duplicates
    existing_ids = set()
    if file_exists:
        try:
            with open(mapping_file, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    existing_ids.add(row.get("utr_id", "").strip())
        except Exception:
            pass

    if utr_id in existing_ids:
        return  # Already present

    # Append new row
    with open(mapping_file, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "utr_id", "primary_name", "name_variants", "bovada_name",
            "current_utr", "country", "last_updated"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "utr_id": utr_id,
            "primary_name": primary_name,
            "name_variants": "",  # Will be enriched by scraper later
            "bovada_name": bovada_name or primary_name,
            "current_utr": "",
            "country": "",
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    print(f"   📝 Appended {primary_name} (id={utr_id}) to player_mapping.csv")


# ---------------------------
# Adapter class for main.py integration
# ---------------------------

class PlayerDataUpdater:
    """
    Adapter class to allow main.py to call the async worker infrastructure.
    Keeps the CLI intact while providing a clean programmatic interface.
    """
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)

    async def _resolve_or_create_mapping(
        self,
        scraper: UTRScraper,
        player_name: str,
        cached_map: Dict[str, str]
    ) -> Optional[str]:
        """
        Resolve player name to UTR ID. If not in mapping, search UTR and append to mapping.csv.

        Returns:
            UTR ID if found/resolved, None if search failed
        """
        # Try cache first
        utr_id = await resolve_utr_id(scraper, player_name, cached_map)

        if utr_id:
            # Check if this is a new resolution (not in mapping file yet)
            norm = normalize_name(player_name)
            if norm not in cached_map:
                # New player discovered - append to mapping
                try:
                    profile = await scraper.get_player_profile(utr_id)
                    if profile:
                        primary_name = profile.get('name', player_name)
                        append_player_mapping_row(
                            self.data_dir,
                            utr_id,
                            primary_name,
                            bovada_name=player_name
                        )
                        # Update cache for subsequent lookups
                        cached_map[norm] = utr_id
                        cached_map[normalize_name(primary_name)] = utr_id
                except Exception as e:
                    print(f"   ⚠️  Could not append mapping for {player_name}: {e}")

        return utr_id

    def update_slate_players(self, odds_df: pd.DataFrame, max_age_days: int = 7,
                           num_workers: int = 8, dynamic_years: bool = True,
                           backfill_years: int = 2, headless: bool = True) -> bool:
        """
        Update UTR data for all unique players in the given odds DataFrame.

        Args:
            odds_df: DataFrame with player columns (player1_normalized, player2_normalized, etc.)
            max_age_days: Not currently used, reserved for future freshness checks
            num_workers: Number of parallel worker processes (default: 8, set to 1 for single-worker)
            dynamic_years: Discover available years per player via UTR UI dropdown
            backfill_years: Also include last N seasons from the current year
            headless: Run browsers headless

        Returns:
            True if update succeeded, False if UTR credentials missing or error
        """
        # Collect unique player names from all player columns
        names = set()
        for col in ('player1_normalized', 'player2_normalized', 'player1_raw', 'player2_raw'):
            if col in odds_df.columns:
                for val in odds_df[col].dropna():
                    if isinstance(val, str) and val.strip():
                        names.add(val.strip())

        if not names:
            print("⚠️  No player names found in odds DataFrame")
            return False

        name_list = sorted(names)
        print(f"🔄 Updating UTR data for {len(name_list)} players using {num_workers} process(es)")

        # Get credentials
        email = os.environ.get("UTR_EMAIL", "").strip()
        password = os.environ.get("UTR_PASSWORD", "").strip()
        if not email or not password:
            print("⚠️  Missing UTR_EMAIL or UTR_PASSWORD; skipping updates")
            return False

        # Build task config
        cfg = TaskConfig(
            data_dir=self.data_dir,
            email=email,
            password=password,
            years=[],  # Will use dynamic + backfill
            headless=headless,
            force_ratings=False,  # Use fast incremental path (only fetch new ratings)
            nav_timeout_ms=90000,
            wait_timeout_ms=45000,
            dynamic_years=dynamic_years,
            backfill_years=backfill_years,
            stagger_logins=True,
            require_fast_path=False  # main.py always allows fallback
        )

        # SINGLE-WORKER PATH (fallback for simplicity)
        if num_workers <= 1:
            try:
                summary = asyncio.run(worker_async(0, name_list, cfg))

                print(f"✅ Update complete (single worker):")
                print(f"   Resolved: {summary.get('resolved', 0)}")
                print(f"   Profiles: {summary.get('profiles', 0)}")
                print(f"   Ratings:  {summary.get('ratings', 0)}")
                print(f"   Matches:  {summary.get('matches', 0)}")
                if summary.get('errors', 0) > 0:
                    print(f"   ⚠️  Errors: {summary['errors']}")

                return summary.get('resolved', 0) > 0

            except Exception as e:
                print(f"❌ Update failed (single worker): {e}")
                return False

        # PARALLEL PATH
        chunks = split_into_chunks(name_list, num_workers)
        jobs = [(i, chunk, cfg) for i, chunk in enumerate(chunks)]

        totals = {
            "resolved": 0, "profiles": 0, "ratings": 0, "matches": 0, "errors": 0,
            "ratings_fast": 0, "ratings_full": 0, "ratings_fallback": 0,
            "matches_fast": 0, "matches_full": 0, "matches_fallback": 0,
        }

        try:
            # Use spawn context for cross-platform compatibility
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                for res in pool.imap_unordered(worker_entry, jobs):
                    for k, v in res.items():
                        totals[k] = totals.get(k, 0) + int(v or 0)
                    print(f"   ➕ Worker completed: resolved={res.get('resolved', 0)}, "
                          f"profiles={res.get('profiles', 0)}, "
                          f"matches={res.get('matches', 0)} (🚀{res.get('matches_fast', 0)} 📥{res.get('matches_full', 0)}), "
                          f"errors={res.get('errors', 0)}")

            print(f"✅ Parallel update complete ({num_workers} workers):")
            for k, v in totals.items():
                print(f"   {k}: {v}")

            return totals["resolved"] > 0

        except Exception as e:
            print(f"❌ Parallel update failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# ---------------------------
# CLI & config
# ---------------------------

@dataclass
class TaskConfig:
    data_dir: Path
    email: str
    password: str
    years: List[str]
    headless: bool
    force_ratings: bool
    nav_timeout_ms: int
    wait_timeout_ms: int
    dynamic_years: bool
    backfill_years: int
    stagger_logins: bool
    require_fast_path: bool  # If True, abort instead of falling back to full scrape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update men-only UTR data for upcoming matches (Bovada).")
    p.add_argument("--from-bovada", action="store_true",
                   help="Scrape Bovada tennis page for upcoming men-only matches (ATP/Challenger/ITF Men).")
    p.add_argument("--players-csv", type=str, default="",
                   help="Optional CSV with a 'player' column of names to process (used if not --from-bovada).")
    p.add_argument("--players", nargs="*", default=[],
                   help="Optional list of player names on the CLI (used if not --from-bovada).")
    p.add_argument("--headless", action="store_true", help="Run browsers headless.")
    p.add_argument("--max-workers", type=int, default=4, help="Parallel worker processes (default: 4).")
    p.add_argument("--years", nargs="*", default=[],
                   help="Explicit years to include (e.g. 2025 2024). Still de-duped with dynamic/backfill.")
    p.add_argument("--dynamic-years", action="store_true",
                   help="Discover available years per player via UTR UI dropdown.")
    p.add_argument("--backfill-years", type=int, default=0,
                   help="Also include last N seasons from the current year (safe with de-dupe).")
    p.add_argument("--force-ratings", action="store_true",
                   help="Fetch rating history even if a ratings CSV already exists.")
    p.add_argument("--nav-timeout-ms", type=int, default=90000,
                   help="Navigation timeout ms (0 = no timeout).")
    p.add_argument("--wait-timeout-ms", type=int, default=45000,
                   help="Wait-for-selector timeout ms (0 = no timeout).")
    p.add_argument("--no-stagger", action="store_true",
                   help="Disable staggered logins (not recommended).")
    p.add_argument("--require-fast-path", action="store_true",
                   help="Abort on fast path failure instead of falling back (testing mode).")
    return p.parse_args()


# ---------------------------
# Bovada scraping (upcoming-only, men-only)
# ---------------------------

MEN_TOUR_LABELS = (
    "ATP", "Challenger", "ITF Men", "ITF Men's", "ITF Men -", "US Open - Men's",
)

def _looks_live(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return ("live" in t) or ("in-play" in t) or ("in play" in t) or ("inplay" in t)


async def scrape_bovada_upcoming_men_names(headless: bool = True) -> Tuple[Set[str], int, int]:
    """
    Returns (unique_player_names, men_event_count, match_count).
    - Only upcoming (non-live)
    - Only men tours: ATP / Challenger / ITF Men (and e.g. "US Open - Men's Singles").
    """
    url = "https://www.bovada.lv/sports/tennis"
    names: Set[str] = set()
    men_events = 0
    total_matches = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context()
        page = await ctx.new_page()

        # Load
        await page.goto(url, timeout=120_000)
        await asyncio.sleep(2.5)

        # Click "Show More" repeatedly (common on Bovada)
        print("🔍 Looking for 'Show More' buttons...")
        for attempt in range(1, 10):
            buttons = await page.locator("button:has-text('Show More')").all()
            if not buttons:
                print(f"   No more Show More buttons found (attempt {attempt})")
                break
            print(f"   Found {len(buttons)} Show More button(s) on attempt {attempt}")
            for i, b in enumerate(buttons, 1):
                try:
                    await b.click(timeout=3000)
                    print(f"      Clicked button {i}")
                    await asyncio.sleep(1.0)
                except Exception:
                    pass

        # Event groups
        # Bovada DOMs shift, so we keep this broad: each "group" usually has a header and a list of games.
        groups = await page.locator("section, div, article").filter(
            has=page.locator("h4, h3, header, .sport-name, .section-header")
        ).all()

        print(f"📊 Found {len(groups)} event groups")

        for g in groups:
            try:
                header_text = (await g.inner_text()).strip()
            except Exception:
                continue
            header_text = re.sub(r"\s+", " ", header_text)
            is_men = any(lbl in header_text for lbl in MEN_TOUR_LABELS)
            is_women = ("Women" in header_text) or ("WTA" in header_text) or ("ITF Women" in header_text)

            if not is_men or is_women:
                # Skip non-men tours
                if is_women:
                    print(f"   🚫 Skipping: {header_text[:80]}")
                continue

            # skip live groups quickly
            if _looks_live(header_text):
                continue

            # collect matches within this group (competitor-name spans usually)
            try:
                match_nodes = await g.locator("span.competitor-name").all()
                # fallback to generic spans with 'vs' in nearby text
                if not match_nodes:
                    match_nodes = await g.locator("span, div").all()
            except Exception:
                match_nodes = []

            # Count games by pairing names 2-by-2
            local_names: List[str] = []
            for node in match_nodes:
                try:
                    t = (await node.inner_text()).strip()
                    if not t or len(t) < 2:
                        continue
                    if _looks_live(t):
                        continue
                    # Avoid obvious junk like "More Bets"
                    if re.search(r"more bets|bet now|lines", t, re.I):
                        continue
                    # Player-ish: contains alphabet letters, not all-caps code, etc.
                    if re.search(r"[A-Za-z]", t) and len(t) <= 40:
                        local_names.append(t)
                except Exception:
                    continue

            # Attempt to infer the number of games
            # Usually names are [P1, P2, P3, P4, ...] => games ≈ len//2
            games = len(local_names) // 2
            if games > 0:
                print(f"   ✅ Processing ATP event: {header_text.splitlines()[0][:70]}")
                print(f"      Found {games} games")
                men_events += 1
                total_matches += games
                # Add to set
                for nm in local_names:
                    names.add(nm)

        await ctx.close()
        await browser.close()

    # Filter upcoming vs live at the end as well (defensive)
    names = {n for n in names if not _looks_live(n)}
    print("\n📈 Summary:")
    print(f"   ATP/Challenger events found: {men_events}")
    print(f"   Total matches found: {total_matches}")
    print(f"   Upcoming matches (non-live): {total_matches}")
    return names, men_events, total_matches


# ---------------------------
# Player mapping helpers
# ---------------------------

def load_player_mapping(data_dir: Path) -> Dict[str, str]:
    """
    Returns normalized_name -> utr_id mapping
    (includes primary_name and name_variants).
    """
    mapping: Dict[str, str] = {}
    f = data_dir / "player_mapping.csv"
    if not f.exists():
        return mapping

    try:
        with open(f, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pid = row.get("utr_id", "").strip()
                primary = row.get("primary_name", "").strip()
                variants = (row.get("name_variants", "") or "").split("|")
                cands = [primary] + variants
                for nm in cands:
                    nm_clean = normalize_name(nm)
                    if nm_clean and pid:
                        mapping[nm_clean] = pid
    except Exception as e:
        logger.warning(f"Could not read player_mapping.csv: {e}")
    return mapping


def normalize_name(n: str) -> str:
    return re.sub(r"\s+", " ", (n or "").strip().lower())


async def resolve_utr_id(scraper: UTRScraper, raw_name: str, cached_map: Dict[str, str]) -> Optional[str]:
    """
    Resolve a raw Bovada name to a UTR ID.
    1) Exact mapping hit (primary or variants) -> id
    2) scraper.search_player(name) -> best match
    """
    q = normalize_name(raw_name)
    if q in cached_map:
        return cached_map[q]

    # fall back to search
    try:
        results = await scraper.search_player(raw_name)
    except Exception as e:
        logger.warning(f"Search failed for '{raw_name}': {e}")
        results = []

    if not results:
        return None

    # Pick best: exact case-insensitive name hit first, else highest UTR
    exact = [r for r in results if normalize_name(r.get("name", "")) == q]
    chosen = exact[0] if exact else max(results, key=lambda r: (r.get("utr") or 0.0))
    pid = chosen.get("id")
    # After we fetch the profile, update_player_mapping will enrich mapping.csv
    return pid


# ---------------------------
# Worker
# ---------------------------

def split_into_chunks(items: List[str], n_chunks: int) -> List[List[str]]:
    n = max(1, n_chunks)
    k = math.ceil(len(items) / n)
    return [items[i:i + k] for i in range(0, len(items), k)]


def worker_entry(args):
    """
    Spawned process worker: async runner with one Playwright/UTR session.
    """
    return asyncio.run(worker_async(*args))


async def worker_async(worker_idx: int, names: List[str], cfg: TaskConfig) -> Dict[str, int]:
    """
    Each worker logs in (staggered), resolves UTR IDs, then updates ratings & matches.
    Returns counters summary.
    """
    # Stagger logins to avoid hitting UTR at once
    if cfg.stagger_logins:
        base = 5.0  # seconds
        delay = base * worker_idx + rnd.uniform(0.0, 3.0)
        await asyncio.sleep(delay)

    scraper = UTRScraper(email=cfg.email, password=cfg.password, headless=cfg.headless, data_dir=cfg.data_dir)

    summary = {
        "resolved": 0,
        "profiles": 0,
        "ratings": 0,
        "matches": 0,
        "errors": 0,
        "ratings_fast": 0,
        "ratings_full": 0,
        "ratings_fallback": 0,
        "matches_fast": 0,
        "matches_full": 0,
        "matches_fallback": 0,
    }

    try:
        await scraper.start_browser()

        # Optional: override timeouts after start (page.set_default_* in your class is sync)
        try:
            if cfg.nav_timeout_ms >= 0:
                scraper.page.set_default_navigation_timeout(cfg.nav_timeout_ms)
                scraper.context.set_default_navigation_timeout(cfg.nav_timeout_ms)
            if cfg.wait_timeout_ms >= 0:
                scraper.page.set_default_timeout(cfg.wait_timeout_ms)
        except Exception as e:
            logger.warning(f"[W{worker_idx}] Could not set custom timeouts: {e}")

        logged_in = await scraper.login()
        if not logged_in:
            logger.error(f"[W{worker_idx}] Login failed")
            return summary

        mapping_cache = load_player_mapping(cfg.data_dir)

        for raw_name in names:
            try:
                nm_clean = raw_name.strip()
                if not nm_clean:
                    continue

                utr_id = await resolve_utr_id(scraper, nm_clean, mapping_cache)
                if not utr_id:
                    logger.warning(f"[W{worker_idx}] No UTR id for '{nm_clean}'")
                    continue
                summary["resolved"] += 1

                # Profile (also refreshes mapping.csv via update_player_mapping inside scraper)
                prof = await scraper.get_player_profile(utr_id)
                if prof:
                    summary["profiles"] += 1

                # Ratings (incremental update with fast path)
                ratings_path = cfg.data_dir / "ratings" / f"player_{utr_id}_ratings.csv"

                if ratings_path.exists() and ratings_path.stat().st_size > 0:
                    # FAST PATH: Player has existing ratings, only fetch new ones
                    logger.info(f"[W{worker_idx}] 🚀 FAST RATINGS PATH for {utr_id}")
                    try:
                        # Import relative to this file's directory
                        import sys
                        from pathlib import Path
                        prod_dir = Path(__file__).parent
                        if str(prod_dir) not in sys.path:
                            sys.path.insert(0, str(prod_dir))
                        from ratings_fast import fetch_ratings_incremental
                        incr = await fetch_ratings_incremental(scraper, utr_id, str(ratings_path))
                        if not incr.empty:
                            # Merge with existing
                            base = pd.read_csv(ratings_path)
                            combo = pd.concat([base, incr], ignore_index=True)
                            if 'date' in combo.columns:
                                combo['date'] = pd.to_datetime(combo['date'], errors='coerce')
                                combo = combo.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                                logger.info(f"[W{worker_idx}] ✅ Added {len(incr)} new ratings for {utr_id}")
                            combo.to_csv(ratings_path, index=False)
                            summary["ratings"] += 1
                            summary["ratings_fast"] += 1
                        else:
                            logger.info(f"[W{worker_idx}] ✅ No new ratings for {utr_id}")
                            summary["ratings_fast"] += 1
                    except Exception as e:
                        if cfg.require_fast_path:
                            logger.error(f"[W{worker_idx}] ❌ FAST PATH REQUIRED but failed for {utr_id}: {e}")
                            raise RuntimeError(f"Fast ratings path failed (--require-fast-path): {e}")
                        logger.warning(f"[W{worker_idx}] ⚠️  Fast ratings failed for {utr_id}, FALLING BACK: {e}")
                        summary["ratings_fallback"] += 1
                        # Fallback to full ratings scrape
                        try:
                            rdf = await scraper.get_player_rating_history(utr_id)
                            if isinstance(rdf, pd.DataFrame):
                                summary["ratings"] += 1
                        except Exception as e2:
                            logger.warning(f"[W{worker_idx}] Full ratings fetch also failed for {utr_id}: {e2}")
                else:
                    # NEW PLAYER: Full ratings history
                    logger.info(f"[W{worker_idx}] 📥 FULL RATINGS FETCH for {utr_id} (new player)")
                    try:
                        rdf = await scraper.get_player_rating_history(utr_id)
                        if isinstance(rdf, pd.DataFrame):
                            summary["ratings"] += 1
                            summary["ratings_full"] += 1
                    except Exception as e:
                        logger.warning(f"[W{worker_idx}] ratings error id={utr_id}: {e}")

                # Years to pull
                player_years = list(cfg.years)

                # Optional backfill from current year
                if cfg.backfill_years and cfg.backfill_years > 0:
                    from datetime import datetime
                    now_y = datetime.utcnow().year
                    extra = [str(y) for y in range(now_y, now_y - cfg.backfill_years, -1)]
                    player_years = list({*player_years, *extra})

                # Discover years from UTR dropdown (scraper clicks the year menu)
                if cfg.dynamic_years:
                    try:
                        # We rely on the same UI the scraper uses in get_player_match_history(year=...)
                        # This method name exists in your class:
                        #   select_year_from_dropdown(year) is used internally,
                        #   so here we probe availability by trying years, or fetch from helper if you added one.
                        # If you have scraper.get_available_years, use it; otherwise derive by probing.
                        avail = await _discover_years_for_player(scraper, utr_id)
                        if avail:
                            player_years = list({*player_years, *[str(y) for y in avail]})
                    except Exception as e:
                        logger.warning(f"[W{worker_idx}] year discovery failed id={utr_id}: {e}")

                # Normalize + sort desc so newest first
                player_years = sorted({str(y) for y in player_years if str(y).isdigit()}, reverse=True)

                # Check if player already has match history (FAST PATH optimization)
                main_file = cfg.data_dir / "matches" / f"player_{utr_id}_matches.csv"

                if main_file.exists() and main_file.stat().st_size > 100:
                    # FAST PATH: Player exists, only update current year
                    logger.info(f"[W{worker_idx}] 🚀 FAST MATCH PATH for {utr_id}")
                    try:
                        # Import relative to this file's directory
                        import sys
                        from pathlib import Path
                        prod_dir = Path(__file__).parent
                        if str(prod_dir) not in sys.path:
                            sys.path.insert(0, str(prod_dir))
                        from scraper_fast import fetch_current_year_incremental_full
                        new_df = await fetch_current_year_incremental_full(scraper, utr_id, str(main_file))
                        if not new_df.empty:
                            # Merge with existing
                            old_df = pd.read_csv(main_file)
                            combo = pd.concat([old_df, new_df], ignore_index=True)
                            if 'match_id' in combo.columns:
                                before_dedup = len(combo)
                                combo = combo.drop_duplicates(subset=['match_id'], keep='first')
                                logger.info(f"[W{worker_idx}] ✅ Added {len(new_df)} new matches for {utr_id}, removed {before_dedup - len(combo)} duplicates")
                            combo.to_csv(main_file, index=False)
                            summary["matches"] += 1
                            summary["matches_fast"] += 1
                        else:
                            logger.info(f"[W{worker_idx}] ✅ No new matches for {utr_id}")
                            summary["matches_fast"] += 1
                    except Exception as e:
                        if cfg.require_fast_path:
                            logger.error(f"[W{worker_idx}] ❌ FAST PATH REQUIRED but failed for {utr_id}: {e}")
                            raise RuntimeError(f"Fast match path failed (--require-fast-path): {e}")
                        logger.warning(f"[W{worker_idx}] ⚠️  Fast match update failed for {utr_id}, FALLING BACK: {e}")
                        summary["matches_fallback"] += 1
                        # Fall back to full backfill on error
                        pulled_any = False
                        for y in player_years:
                            try:
                                mdf = await scraper.get_player_match_history(utr_id, year=str(y), limit=200)
                                if isinstance(mdf, pd.DataFrame) and not mdf.empty:
                                    pulled_any = True
                            except Exception as e2:
                                logger.warning(f"[W{worker_idx}] matches error id={utr_id} year={y}: {e2}")
                            await asyncio.sleep(rnd.uniform(0.8, 1.8))
                        if pulled_any:
                            summary["matches"] += 1
                else:
                    # NEW PLAYER: Full backfill (scrape all years)
                    logger.info(f"[W{worker_idx}] 📥 FULL MATCH BACKFILL for {utr_id} (new player)")
                    summary["matches_full"] += 1
                    pulled_any = False
                    for y in player_years:
                        try:
                            mdf = await scraper.get_player_match_history(utr_id, year=str(y), limit=200)
                            if isinstance(mdf, pd.DataFrame) and not mdf.empty:
                                pulled_any = True
                        except Exception as e:
                            logger.warning(f"[W{worker_idx}] matches error id={utr_id} year={y}: {e}")
                        await asyncio.sleep(rnd.uniform(0.8, 1.8))

                    if pulled_any:
                        summary["matches"] += 1

                # small jitter per player
                await asyncio.sleep(rnd.uniform(0.5, 1.2))

            except Exception as e:
                summary["errors"] += 1
                logger.error(f"[W{worker_idx}] error processing '{raw_name}': {e}")

    finally:
        try:
            await scraper.close_browser()
        except Exception:
            pass

    return summary


async def _discover_years_for_player(scraper: UTRScraper, utr_id: str) -> List[int]:
    """
    Conservative year discovery:
    - First try the most common recent seasons (current back to 2018).
    - Use scraper.select_year_from_dropdown(year) to check availability quickly.
    """
    from datetime import datetime
    now_y = datetime.utcnow().year
    candidates = list(range(now_y, 2017, -1))  # now..2018

    # Navigate to results tab once; subsequent year clicks are cheap
    try:
        _ = await scraper.get_player_match_history(utr_id, year=None, limit=5)  # opens the page & Singles view
    except Exception:
        pass

    found: List[int] = []
    for y in candidates:
        try:
            sel = await scraper.select_year_from_dropdown(str(y))
            if sel is True:
                found.append(y)
            elif sel == "year_not_found":
                # stop only if we have found newer years already and hit a gap of multiple years
                # but to be safe, just continue
                pass
        except Exception:
            pass
    return found


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent  # /production
    data_dir = (base_dir.parent / "data")
    for d in [data_dir, data_dir / "players", data_dir / "matches", data_dir / "ratings"]:
        d.mkdir(parents=True, exist_ok=True)

    email = os.environ.get("UTR_EMAIL", "").strip()
    password = os.environ.get("UTR_PASSWORD", "").strip()
    if not email or not password:
        print("Error: UTR_EMAIL and UTR_PASSWORD environment variables must be set.")
        print("  export UTR_EMAIL=your_email@example.com")
        print("  export UTR_PASSWORD=your_password")
        sys.exit(1)

    # Collect target player list
    men_names: Set[str] = set()

    if args.from_bovada:
        names, men_events, match_count = asyncio.run(scrape_bovada_upcoming_men_names(headless=args.headless))
        # We only keep player-like strings (bovada sometimes repeats headers)
        names = {n for n in names if re.search(r"[A-Za-z]", n)}
        # Bovada produces both players; we just want the set of unique names
        # Also filter out obvious noise like "To Win Match", etc.
        bad = re.compile(r"(to win|set betting|total games|handicap|correct score)", re.I)
        names = {n for n in names if not bad.search(n)}
        # sanity: names not containing "Women's" (defensive)
        names = {n for n in names if "women" not in n.lower()}
        men_names.update(names)

        print(f"🎾 Men-only slate: {len(men_names)} unique players")
    else:
        if args.players_csv:
            try:
                df = pd.read_csv(args.players_csv)
                if "player" in df.columns:
                    men_names.update([str(x) for x in df["player"].dropna().tolist()])
            except Exception as e:
                print(f"Failed to read --players-csv: {e}")
        if args.players:
            men_names.update(args.players)

    if not men_names:
        print("No players to process. Provide --from-bovada, --players-csv, or --players.")
        sys.exit(0)

    years = [str(y) for y in args.years]
    cfg = TaskConfig(
        data_dir=data_dir,
        email=email,
        password=password,
        years=years,
        headless=bool(args.headless),
        force_ratings=bool(args.force_ratings),
        nav_timeout_ms=int(args.nav_timeout_ms),
        wait_timeout_ms=int(args.wait_timeout_ms),
        dynamic_years=bool(args.dynamic_years),
        backfill_years=int(args.backfill_years),
        stagger_logins=not bool(args.no_stagger),
        require_fast_path=bool(args.require_fast_path),
    )

    max_workers = max(1, int(args.max_workers))
    names_list = sorted(men_names)
    chunks = split_into_chunks(names_list, max_workers)

    print(f"🚀 Updating {len(names_list)} players across {max_workers} processes")

    # Use spawn (macOS default) and explicit context to be safe on Py3.13
    ctx = get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        jobs = [
            (i, chunk, cfg)
            for i, chunk in enumerate(chunks)
        ]
        totals = {
            "resolved": 0, "profiles": 0, "ratings": 0, "matches": 0, "errors": 0,
            "ratings_fast": 0, "ratings_full": 0, "ratings_fallback": 0,
            "matches_fast": 0, "matches_full": 0, "matches_fallback": 0,
        }
        try:
            for res in pool.imap_unordered(worker_entry, jobs):
                for k, v in res.items():
                    totals[k] = totals.get(k, 0) + int(v or 0)
        except KeyboardInterrupt:
            print("Interrupted by user. Attempting to close workers cleanly...")
            pool.terminate()
            pool.join()
            raise
        except Exception as e:
            print(f"Worker error: {e}")
            pool.terminate()
            pool.join()
            raise

    print("\n✅ Done.")
    print(f"Resolved IDs:  {totals['resolved']}")
    print(f"Profiles:      {totals['profiles']}")
    print(f"\nRatings files: {totals['ratings']} (created/updated)")
    print(f"  🚀 Fast path:   {totals.get('ratings_fast', 0)}")
    print(f"  📥 Full fetch:  {totals.get('ratings_full', 0)}")
    print(f"  ⚠️  Fallback:    {totals.get('ratings_fallback', 0)}")
    print(f"\nMatch files:   {totals['matches']} (created/updated)")
    print(f"  🚀 Fast path:   {totals.get('matches_fast', 0)}")
    print(f"  📥 Full fetch:  {totals.get('matches_full', 0)}")
    print(f"  ⚠️  Fallback:    {totals.get('matches_fallback', 0)}")
    print(f"\nErrors:        {totals['errors']}")


if __name__ == "__main__":
    # Make Ctrl+C behave on macOS/Windows
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
