#!/usr/bin/env python3
"""
ATP Rankings Scraper
Fetches current rank + points for all ranked ATP players from atptour.com.
Saves to data/atp_rankings.csv for use by ta_feature_calculator.py.

Usage:
    python scraping/atp_rankings_scraper.py
    # -> writes data/atp_rankings.csv with columns: rank, player_name, points
"""

from playwright.sync_api import sync_playwright
import pandas as pd
import re
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

ATP_RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=0-5000"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "atp_rankings.csv"


def _normalize_name(name: str) -> str:
    """Normalize player name: strip extra whitespace, title-case."""
    return " ".join(name.strip().split())


def fetch_atp_rankings(headless: bool = True, timeout_ms: int = 60000) -> pd.DataFrame:
    """
    Scrape the ATP singles rankings page and return a DataFrame with:
        rank (int), player_name (str), points (int)
    """
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

        print(f"Loading ATP rankings page...")
        page.goto(ATP_RANKINGS_URL, wait_until="domcontentloaded", timeout=timeout_ms)
        time.sleep(8)  # ATP page is JS-rendered — wait for React to populate the table

        # ATP rankings use class "lower-row" for each player row
        table_rows = page.query_selector_all("tr.lower-row")
        if not table_rows:
            # Fallback: any tbody tr
            table_rows = page.query_selector_all("table tbody tr")
        print(f"Found {len(table_rows)} ranking rows")
        print(f"Found {len(table_rows)} table rows")

        for tr in table_rows:
            cells = tr.query_selector_all("td")
            if len(cells) < 3:
                continue

            # Cell layout from atptour.com lower-row: rank | player | points | (other)
            try:
                rank_text = cells[0].inner_text().strip()
                rank = int(re.sub(r"[^\d]", "", rank_text))
            except (ValueError, IndexError):
                continue

            try:
                player_cell = cells[1]
                player_link = player_cell.query_selector("a")
                player_name = player_link.inner_text().strip() if player_link else player_cell.inner_text().strip()
                player_name = _normalize_name(player_name)
                player_url = player_link.get_attribute("href") if player_link else None
            except Exception:
                continue

            try:
                points_text = cells[2].inner_text().strip()
                points = int(re.sub(r"[^\d]", "", points_text))
            except (ValueError, IndexError):
                points = 0

            if player_name:
                rows.append({"rank": rank, "player_name": player_name, "points": points, "player_url": player_url})

        browser.close()

    df = pd.DataFrame(rows)

    # The ATP page renders two row sets: abbreviated names with real points, and
    # full names with "tournaments played" (~20-40). Deduplicate by rank, keeping
    # the row with the highest points (which is always the real ranking points).
    if not df.empty:
        df = df.sort_values("points", ascending=False).drop_duplicates(subset="rank", keep="first").sort_values("rank").reset_index(drop=True)

    print(f"Scraped {len(df)} ranked players")
    return df


def save_rankings(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df["scraped_at"] = datetime.now().isoformat()
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    return path


def load_rankings(path: Path = OUTPUT_PATH) -> Optional[pd.DataFrame]:
    """Load cached rankings CSV. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def _lookup(player_name: str, col: str, df: pd.DataFrame) -> Optional[int]:
    """
    Look up rank or points for a player. Strategy:
    1. Exact full-name match (e.g. "Matteo Berrettini" == "Matteo Berrettini")
    2. Abbreviated-name match (e.g. "M. Berrettini" from "Matteo Berrettini")
    3. Last-name + first-initial disambiguation (handles siblings like Berrettinis)
    4. Last-name only if unique
    """
    name = player_name.strip()
    name_lower = name.lower()
    df_lower = df["player_name"].str.lower().str.strip()

    # 1. Exact match
    match = df[df_lower == name_lower]
    if not match.empty:
        return int(match.iloc[0][col])

    parts = name.split()
    last_name = parts[-1].lower() if parts else ""
    first_initial = parts[0][0].lower() if parts else ""

    # 2. Abbreviated name: "F. Last" format
    abbrev = f"{first_initial}. {last_name}"
    match = df[df_lower == abbrev]
    if not match.empty:
        return int(match.iloc[0][col])

    # 3. Last-name candidates, then disambiguate by first initial
    if last_name:
        candidates = df[df_lower.str.contains(rf"\b{re.escape(last_name)}\b", na=False)]
        if len(candidates) == 1:
            return int(candidates.iloc[0][col])
        if len(candidates) > 1 and first_initial:
            # Filter by first initial
            narrowed = candidates[candidates["player_name"].str[0].str.lower() == first_initial]
            if len(narrowed) == 1:
                return int(narrowed.iloc[0][col])

    # 4. Reversed-name fallback for Asian/non-Western names stored as "Initial. FamilyName"
    # e.g. "Bu Yunchaokete" on TA → ATP stores as "Y. Bu" (family=Bu, given=Yunchaokete)
    if len(parts) >= 2:
        # Treat first word as family name, build "given_initial. family" abbrev
        family = parts[0].lower()
        given_initial = parts[1][0].lower()
        reversed_abbrev = f"{given_initial}. {family}"
        match = df[df_lower == reversed_abbrev]
        if not match.empty:
            return int(match.iloc[0][col])
        # Also try family-name-only search
        candidates = df[df_lower.str.contains(rf"\b{re.escape(family)}\b", na=False)]
        if len(candidates) == 1:
            return int(candidates.iloc[0][col])
        if len(candidates) > 1 and given_initial:
            narrowed = candidates[candidates["player_name"].str.split(r"\.\s*").str[-1].str.lower().str.strip() == family]
            if len(narrowed) == 1:
                return int(narrowed.iloc[0][col])

    return None


def get_player_points(player_name: str, df: Optional[pd.DataFrame] = None) -> Optional[int]:
    """Look up a player's ATP points by name. Returns None if not found."""
    if df is None:
        df = load_rankings()
    if df is None or df.empty:
        return None
    return _lookup(player_name, "points", df)


def get_player_rank(player_name: str, df: Optional[pd.DataFrame] = None) -> Optional[int]:
    """Look up a player's ATP rank by name. Used to cross-validate against TA rank. Returns None if not found."""
    if df is None:
        df = load_rankings()
    if df is None or df.empty:
        return None
    return _lookup(player_name, "rank", df)


if __name__ == "__main__":
    df = fetch_atp_rankings()
    if df.empty:
        print("ERROR: No data scraped — check selector or page structure")
    else:
        print(df.head(20).to_string(index=False))
        save_rankings(df)
