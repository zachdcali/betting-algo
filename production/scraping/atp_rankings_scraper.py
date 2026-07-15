#!/usr/bin/env python3
"""
ATP Rankings Scraper
Fetches current rank + points for all ranked ATP players from atptour.com.
Saves to data/atp_rankings.csv for use by ta_feature_calculator.py.

Usage:
    python scraping/atp_rankings_scraper.py
    # -> writes data/atp_rankings.csv with columns: rank, player_name, points
"""

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

    from browser_session import new_page

    page = new_page()
    try:

        print("Loading ATP rankings page...")
        # The page is JS-rendered. A blind sleep intermittently parsed an empty
        # table (the bug that silently aged out rankings). Instead, wait until the
        # rows actually populate, and retry the load once if they don't.
        table_rows = []
        for attempt in range(2):
            if attempt == 0:
                page.goto(ATP_RANKINGS_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            else:
                print("  ATP table came back empty — retrying load once...")
                page.reload(wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_function(
                    "document.querySelectorAll('tr.lower-row, table tbody tr').length > 50",
                    timeout=timeout_ms,
                )
            except Exception:
                time.sleep(8)  # fall back to the old fixed wait if the predicate times out
            table_rows = page.query_selector_all("tr.lower-row") or page.query_selector_all("table tbody tr")
            if len(table_rows) > 50:
                break
        print(f"Found {len(table_rows)} ranking rows")

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

    finally:
        page.close()

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


def _letters(token: str) -> str:
    """Return the case-folded alphabetic identity carried by one name token."""
    return "".join(char for char in str(token).casefold() if char.isalpha())


def _given_identity_matches(query_token: str, candidate_name: str) -> bool:
    """Conservatively bind a query given name to a ranking-row given name.

    ATP ranking rows are commonly abbreviated (``M. Berrettini``), so an
    initial may bind to a full given name.  When both sides expose a full given
    name, however, they must agree exactly.  A surname-only match may never
    override a different initial or full given name.
    """
    candidate_parts = str(candidate_name).strip().split()
    query_given = _letters(query_token)
    candidate_given = _letters(candidate_parts[0]) if candidate_parts else ""
    if not query_given or not candidate_given:
        return False
    if query_given[0] != candidate_given[0]:
        return False
    if len(query_given) > 1 and len(candidate_given) > 1:
        return query_given == candidate_given
    return True


def _given_compatible_candidates(
    candidates: pd.DataFrame,
    query_token: str,
) -> pd.DataFrame:
    """Keep only surname candidates with compatible given-name evidence."""
    if candidates.empty or not query_token:
        return candidates.iloc[0:0]
    compatible = candidates["player_name"].map(
        lambda value: _given_identity_matches(query_token, value)
    )
    return candidates[compatible]


def _lookup(player_name: str, col: str, df: pd.DataFrame) -> Optional[int]:
    """
    Look up rank or points for a player. Strategy:
    1. Exact full-name match (e.g. "Matteo Berrettini" == "Matteo Berrettini")
    2. Abbreviated-name match (e.g. "M. Berrettini" from "Matteo Berrettini")
    3. Last-name candidates with compatible given-name evidence
    4. Reversed-name abbreviation with the same identity requirement

    A unique surname is not identity evidence by itself.  This deliberately
    fails closed when, for example, ``Vito Antonio Darderi`` is queried against
    the sole ranking row ``L. Darderi``.
    """
    name = player_name.strip()
    name_lower = name.lower()
    df_lower = df["player_name"].str.lower().str.strip()

    # 1. Exact match
    match = df[df_lower == name_lower]
    if not match.empty:
        return int(match.iloc[0][col])

    parts = name.replace("-", " ").split()
    last_name = parts[-1].lower() if parts else ""
    first_initial = parts[0][0].lower() if parts else ""

    # 2b. Multi-surname names: ATP's display may abbreviate to ANY surname
    # token ("Diego Dedura Palomero" appears as "D. Dedura", rank feed form)
    for tok in parts[1:-1]:
        t = tok.lower()
        if len(t) >= 4:
            m = df[df_lower == f"{first_initial}. {t}"]
            if not m.empty:
                return int(m.iloc[0][col])
    # 2. Abbreviated name: "F. Last" format
    abbrev = f"{first_initial}. {last_name}"
    match = df[df_lower == abbrev]
    if not match.empty:
        return int(match.iloc[0][col])

    # 3. Last-name candidates.  A unique surname is insufficient: require the
    # query and candidate given-name evidence to agree as a full name or an
    # initial.  This prevents one family member inheriting another's rank.
    if last_name and len(parts) >= 2:
        candidates = df[df_lower.str.contains(rf"\b{re.escape(last_name)}\b", na=False)]
        narrowed = _given_compatible_candidates(candidates, parts[0])
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
        # Also search the family name, but retain the same given-name binding
        # contract.  A unique family name may not override a different initial.
        candidates = df[df_lower.str.contains(rf"\b{re.escape(family)}\b", na=False)]
        narrowed = _given_compatible_candidates(candidates, parts[1])
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


def resolve_rankings(headless: bool = True, _fetch=None, _load=None):
    """Get current rankings with a robust fallback chain.

    Tries a live scrape first; if that returns nothing or errors, falls back to
    the cached ``data/atp_rankings.csv``. Returns ``(df, source)`` where source is:
      - ``"fresh"``        live scrape succeeded
      - ``"cached@<ts>"``  live failed/empty; using cached CSV (ts = its scraped_at)
      - ``"none"``         no live data and no usable cache (callers default to 500)

    ``_fetch``/``_load`` are injectable for testing.
    """
    fetch = _fetch or fetch_atp_rankings
    load = _load or load_rankings
    try:
        df = fetch(headless=headless)
        if df is not None and not df.empty:
            return df, "fresh"
        print("  ⚠️  ATP rankings live scrape returned no rows")
    except Exception as e:  # network/playwright errors must not crash the pipeline
        print(f"  ⚠️  ATP rankings live scrape error (non-fatal): {e}")

    cached = load()
    if cached is not None and not cached.empty:
        asof = "unknown date"
        if "scraped_at" in cached.columns and len(cached):
            asof = str(cached["scraped_at"].iloc[0])
        return cached, f"cached@{asof}"
    return pd.DataFrame(), "none"


if __name__ == "__main__":
    df = fetch_atp_rankings()
    if df.empty:
        print("ERROR: No data scraped — check selector or page structure")
    else:
        print(df.head(20).to_string(index=False))
        save_rankings(df)
