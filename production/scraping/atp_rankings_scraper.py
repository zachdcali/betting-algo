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
import unicodedata
from pathlib import Path
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse

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
    normalized = unicodedata.normalize("NFKD", str(token).casefold())
    return "".join(
        char for char in normalized
        if char.isalpha() and not unicodedata.combining(char)
    )


def _normalized_name_tokens(value: str) -> tuple[str, ...]:
    """Return comparable alphabetic tokens for a display name or URL slug."""
    text = re.sub(r"\s*\([a-z]{2,5}\)\s*$", "", str(value), flags=re.I)
    tokens = tuple(
        token for token in (_letters(part) for part in re.split(r"[\s-]+", text))
        if token
    )
    if tokens and tokens[-1] in {"jr", "junior"}:
        return tokens[:-1]
    return tokens


def _profile_name_tokens(player_url: str) -> tuple[str, ...]:
    """Extract the full player-name slug from an official ATP profile URL."""
    try:
        parts = [part for part in urlparse(str(player_url)).path.split("/") if part]
        player_index = next(
            index for index, part in enumerate(parts)
            if part.casefold() == "players"
        )
        return _normalized_name_tokens(parts[player_index + 1])
    except (StopIteration, IndexError, ValueError):
        return ()


def _normalized_profile_url(player_url: str) -> str:
    """Normalize an ATP profile URL to its host-independent path identity."""
    try:
        path = urlparse(str(player_url or "").strip()).path
    except ValueError:
        return ""
    return path.rstrip("/").casefold()


def _profile_identity_matches(
    query_name: str,
    player_url: str,
    candidate_name: str = "",
) -> bool:
    """Require a full query to agree with the official profile-name slug."""
    query = _normalized_name_tokens(query_name)
    profile = _profile_name_tokens(player_url)
    if not query or not profile:
        return False
    if query == profile:
        return True
    candidate_tokens = _normalized_name_tokens(candidate_name)
    candidate_surname = candidate_tokens[-1] if candidate_tokens else ""
    # ATP can omit a second family name that another feed includes (Diego
    # Dedura vs Diego Dedura Palomero).  A strict prefix is accepted only when
    # the ranking display's surname anchors the shared prefix boundary.
    shorter, longer = (
        (query, profile) if len(query) < len(profile) else (profile, query)
    )
    if (
        len(shorter) >= 2
        and longer[:len(shorter)] == shorter
        and candidate_surname == shorter[-1]
    ):
        return True
    # Feeds can add or omit middle names and second family names. Permit only
    # an ordered subsequence with the same full first and last identity; never
    # an unordered token set (which would erase Juan Manuel vs Juan Martin).
    ordered_subsequence = iter(longer)
    if (
        len(shorter) >= 2
        and shorter[0] == longer[0]
        and shorter[-1] == longer[-1]
        and all(token in ordered_subsequence for token in shorter)
    ):
        return True
    # Some feeds split a compound given name that ATP concatenates (Soon Woo
    # Kwon vs Soonwoo Kwon; Ye Cong Mo vs Yecong Mo).
    if (
        len(query) >= 2
        and len(profile) >= 2
        and query[-1] == profile[-1]
        and "".join(query[:-1]) == "".join(profile[:-1])
    ):
        return True
    # ATP and upstream feeds occasionally use family-name-first display order
    # for two-token East Asian names.  Do not permit a general token-set match:
    # that would erase middle-name identity such as Juan Manuel vs Juan Martin.
    return len(query) == 2 and query == tuple(reversed(profile))


def _given_identity_matches(
    query_name: str,
    candidate_name: str,
    candidate_url: str = "",
) -> bool:
    """Conservatively bind a query identity to a ranking-row identity.

    ATP ranking rows are commonly abbreviated (``M. Berrettini``), so an
    initial can identify a candidate only when the row's official profile URL
    supplies the agreeing full name.  When both displays expose full names,
    they must agree exactly.  A surname or same initial is never sufficient by
    itself; otherwise two same-initial relatives remain indistinguishable.
    """
    query_parts = str(query_name).strip().split()
    candidate_parts = str(candidate_name).strip().split()
    query_given = _letters(query_parts[0]) if query_parts else ""
    candidate_given = _letters(candidate_parts[0]) if candidate_parts else ""
    if not query_given or not candidate_given:
        return False
    if query_given[0] != candidate_given[0]:
        return False
    if len(query_given) > 1 and len(candidate_given) > 1:
        return query_given == candidate_given
    if len(query_given) > 1 and len(candidate_given) == 1:
        return _profile_identity_matches(
            query_name,
            candidate_url,
            candidate_name,
        )
    # An abbreviated upstream query carries no more identity than the row.
    # Preserve that legacy lookup path, but callers with a canonical ATP URL
    # should pass it to ``get_player_rank``/``get_player_points`` below.
    return len(query_given) == 1


def _given_compatible_candidates(
    candidates: pd.DataFrame,
    query_name: str,
) -> pd.DataFrame:
    """Keep only surname candidates with compatible given-name evidence."""
    if candidates.empty or not query_name:
        return candidates.iloc[0:0]
    compatible = candidates.apply(
        lambda row: _given_identity_matches(
            query_name,
            row.get("player_name", ""),
            row.get("player_url", ""),
        ),
        axis=1,
    )
    return candidates[compatible]


def _matching_row(
    player_name: str,
    df: pd.DataFrame,
    *,
    player_url: str = "",
) -> Optional[pd.Series]:
    """
    Resolve exactly one ranking row for a player. Strategy:
    0. Exact canonical ATP profile URL, when the caller has one
    1. Exact full-name match (e.g. "Matteo Berrettini" == "Matteo Berrettini")
    2. Abbreviated-name match bound to the full official profile URL slug
    3. Last-name candidates with compatible given-name evidence
    4. Reversed-name abbreviation with the same identity requirement

    A unique surname is not identity evidence by itself.  This deliberately
    fails closed when, for example, ``Vito Antonio Darderi`` is queried against
    the sole ranking row ``L. Darderi``.  A full-name query also fails closed
    against an initial-only row whose ``player_url`` is missing: same-initial
    people cannot be distinguished from display text alone.
    """
    name = player_name.strip()
    name_lower = name.lower()
    df_lower = df["player_name"].str.lower().str.strip()

    # 0. A canonical URL is stronger than display-name heuristics.  If the
    # caller supplies one, absence or duplication is a hard miss; never fall
    # back to a possibly different same-name person.
    canonical_url = _normalized_profile_url(player_url)
    if canonical_url:
        if "player_url" not in df.columns:
            return None
        url_match = df[
            df["player_url"].map(_normalized_profile_url).eq(canonical_url)
        ]
        return url_match.iloc[0] if len(url_match) == 1 else None

    # 1. Exact match
    match = df[df_lower == name_lower]
    if len(match) == 1:
        return match.iloc[0]
    if len(match) > 1:
        return None

    parts = name.replace("-", " ").split()
    last_name = parts[-1].lower() if parts else ""
    first_initial = parts[0][0].lower() if parts else ""

    # 2b. Multi-surname names: ATP's display may abbreviate to ANY surname
    # token ("Diego Dedura Palomero" appears as "D. Dedura", rank feed form)
    for tok in parts[1:-1]:
        t = tok.lower()
        if len(t) >= 4:
            m = df[df_lower == f"{first_initial}. {t}"]
            m = _given_compatible_candidates(m, name)
            if len(m) == 1:
                return m.iloc[0]
    # 2. Abbreviated name: "F. Last" format
    abbrev = f"{first_initial}. {last_name}"
    match = df[df_lower == abbrev]
    match = _given_compatible_candidates(match, name)
    if len(match) == 1:
        return match.iloc[0]

    # 3. Last-name candidates.  A unique surname is insufficient: require the
    # query and candidate given-name evidence to agree as a full name or an
    # initial.  This prevents one family member inheriting another's rank.
    if last_name and len(parts) >= 2:
        candidates = df[df_lower.str.contains(rf"\b{re.escape(last_name)}\b", na=False)]
        narrowed = _given_compatible_candidates(candidates, name)
        if len(narrowed) == 1:
            return narrowed.iloc[0]

    # 4. Reversed-name fallback for Asian/non-Western names stored as "Initial. FamilyName"
    # e.g. "Bu Yunchaokete" on TA → ATP stores as "Y. Bu" (family=Bu, given=Yunchaokete)
    if len(parts) >= 2:
        # Treat first word as family name, build "given_initial. family" abbrev
        family = parts[0].lower()
        given_initial = parts[1][0].lower()
        reversed_abbrev = f"{given_initial}. {family}"
        match = df[df_lower == reversed_abbrev]
        match = _given_compatible_candidates(
            match,
            " ".join([*parts[1:], parts[0]]),
        )
        if len(match) == 1:
            return match.iloc[0]
        # Also search the family name, but retain the same given-name binding
        # contract.  A unique family name may not override a different initial.
        candidates = df[df_lower.str.contains(rf"\b{re.escape(family)}\b", na=False)]
        narrowed = _given_compatible_candidates(
            candidates,
            " ".join([*parts[1:], parts[0]]),
        )
        if len(narrowed) == 1:
            return narrowed.iloc[0]

    return None


def _lookup(
    player_name: str,
    col: str,
    df: pd.DataFrame,
    *,
    player_url: str = "",
) -> Optional[int]:
    row = _matching_row(player_name, df, player_url=player_url)
    if row is None:
        return None
    try:
        return int(row[col])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def get_player_points(
    player_name: str,
    df: Optional[pd.DataFrame] = None,
    *,
    player_url: str = "",
) -> Optional[int]:
    """Look up a player's ATP points by name. Returns None if not found."""
    if df is None:
        df = load_rankings()
    if df is None or df.empty:
        return None
    return _lookup(player_name, "points", df, player_url=player_url)


def get_player_rank(
    player_name: str,
    df: Optional[pd.DataFrame] = None,
    *,
    player_url: str = "",
) -> Optional[int]:
    """Look up a player's ATP rank by name. Used to cross-validate against TA rank. Returns None if not found."""
    if df is None:
        df = load_rankings()
    if df is None or df.empty:
        return None
    return _lookup(player_name, "rank", df, player_url=player_url)


def get_player_url(
    player_name: str,
    df: Optional[pd.DataFrame] = None,
    *,
    player_url: str = "",
) -> Optional[str]:
    """Return the exact resolved ATP profile URL, or ``None`` if ambiguous."""
    if df is None:
        df = load_rankings()
    if df is None or df.empty or "player_url" not in df.columns:
        return None
    row = _matching_row(player_name, df, player_url=player_url)
    if row is None:
        return None
    value = str(row.get("player_url", "")).strip()
    return value or None


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
