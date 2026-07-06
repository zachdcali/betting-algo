"""ATP results scraper — secondary live source for match results.

Covers the Tennis Abstract update gap (TA can lag days-to-weeks behind, e.g. it
posted nothing between Halle and mid-Wimbledon 2026). Two entry points:

- ``fetch_tournament_results(url)``  → all completed matches of an *in-progress*
  event from an atptour.com scores page, e.g.
  ``https://www.atptour.com/en/scores/current/wimbledon/540/results``
  (round, players, winner, score). This is what TA lacks during a tournament.
- ``fetch_player_activity(player_url)`` → per-player completed tournaments from
  ``/en/players/<slug>/<id>/player-activity?matchType=Singles``
  (event, date, surface, round, opponent, W/L, score).

Both pages are JS-rendered → Playwright, same approach as atp_rankings_scraper.
Parsers are separated from fetchers so they can be tested on saved HTML fixtures.
Rows produced here must be labeled with ``source='atp_results'`` by callers —
real values with honest provenance, never silent blending (see AGENTS.md).
"""
from __future__ import annotations

import re
import time
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

ROUND_CODES = {"R128", "R64", "R32", "R16", "QF", "SF", "F", "Q1", "Q2", "Q3", "Q4", "RR", "ER", "BR"}

ROUND_NAME_TO_CODE = {
    "round of 128": "R128",
    "round of 64": "R64",
    "round of 32": "R32",
    "round of 16": "R16",
    "quarterfinals": "QF",
    "quarter-finals": "QF",
    "quarterfinal": "QF",
    "semifinals": "SF",
    "semi-finals": "SF",
    "semifinal": "SF",
    "final": "F",
    "finals": "F",
}


def _round_code_from_header(header_text: str) -> Optional[str]:
    """'Round of 64 - Court 14' -> 'R64'; 'Final - Centre Court' -> 'F'."""
    head = header_text.split("-")[0].strip().lower()
    return ROUND_NAME_TO_CODE.get(head)


def _player_set_scores(stats_item) -> list[str]:
    """Set scores for one player from a results-page stats-item.

    Each set renders as a games digit plus an optional tiebreak digit in a
    sub-span. We read the direct text of each score span defensively and emit
    strings like '6' or '6(8)'.
    """
    scores_div = stats_item.select_one("div.scores")
    if scores_div is None:
        return []
    sets: list[str] = []
    for item in scores_div.find_all(["div", "span"], recursive=False):
        txt = item.get_text(" ", strip=True)
        nums = re.findall(r"\d+", txt)
        if not nums:
            continue
        if len(nums) == 1:
            sets.append(nums[0])
        else:  # games + tiebreak points
            sets.append(f"{nums[0]}({nums[1]})")
    if sets:
        return sets
    # fallback: flat digit stream (games first, tiebreaks can't be attributed)
    return re.findall(r"\d+", scores_div.get_text(" ", strip=True))


def parse_tournament_results(html: str) -> pd.DataFrame:
    """Parse an atptour.com scores/results page into one row per completed match.

    Columns: round (code or raw header), p1, p2, winner (1|2|None),
    p1_sets, p2_sets (space-joined set strings), stats_url (may be None).
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for card in soup.select("div.match"):
        header = card.select_one("div.match-header")
        header_text = header.get_text(" ", strip=True) if header else ""
        rnd = _round_code_from_header(header_text)

        items = card.select("div.match-stats div.stats-item")
        if len(items) != 2:
            continue

        names, sets, winner_flags = [], [], []
        for it in items:
            name_el = it.select_one("div.player-info div.name a") or it.select_one("div.player-info div.name")
            name = name_el.get_text(" ", strip=True) if name_el else ""
            name = re.sub(r"\s*\(.*?\)\s*$", "", name)  # strip seed/(Q) suffix
            names.append(name)
            sets.append(_player_set_scores(it))
            winner_flags.append(bool(it.select_one("[class*=winner]")) or "winner" in (it.get("class") or []))

        if not names[0] or not names[1]:
            continue
        winner = 1 if winner_flags[0] and not winner_flags[1] else 2 if winner_flags[1] and not winner_flags[0] else None

        stats_a = None
        for a in card.select("div.match-cta a"):
            if "stats" in a.get_text(strip=True).lower():
                stats_a = a.get("href")
                break

        rows.append({
            "round": rnd or header_text,
            "p1": names[0],
            "p2": names[1],
            "winner": winner,
            "p1_sets": " ".join(sets[0]),
            "p2_sets": " ".join(sets[1]),
            "stats_url": stats_a,
        })
    return pd.DataFrame(rows)


def _parse_activity_rows(tokens: list[str]) -> list[dict]:
    """Split a tournament block's token stream into match rows anchored on round codes."""
    rows = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ROUND_CODES:
            # opponent = next non-numeric token that isn't a rank '(56)' or 'Bye'
            j = i + 1
            opp = None
            while j < len(tokens) and tokens[j] not in ROUND_CODES:
                t = tokens[j]
                if t == "Bye":
                    break
                if opp is None and not re.fullmatch(r"\(?\d+\)?", t):
                    opp = t
                j += 1
            digits = [t for t in tokens[i + 1:j] if re.fullmatch(r"\d+", t)]
            if opp:
                rows.append({"round": tok, "opponent": opp, "digits": digits})
            i = j
        else:
            i += 1
    return rows


def _result_from_digits(digits: list[str]) -> tuple[Optional[str], str]:
    """Infer W/L + score string from the player-first digit stream.

    Digits alternate own-games/opp-games per set; a tiebreak adds one extra
    digit which we cannot always attribute, so we pair greedily and mark the
    result only when the set pairing is unambiguous (even count).
    """
    if not digits or len(digits) % 2 == 1:
        return None, " ".join(digits)
    own = digits[0::2]
    opp = digits[1::2]
    won = sum(1 for a, b in zip(own, opp) if int(a) > int(b))
    lost = sum(1 for a, b in zip(own, opp) if int(a) < int(b))
    score = " ".join(f"{a}-{b}" for a, b in zip(own, opp))
    if won == lost:
        return None, score
    return ("W" if won > lost else "L"), score


def parse_player_activity(html: str) -> pd.DataFrame:
    """Parse a player-activity page into one row per completed match.

    Columns: event, location, start_date (raw '25 May, 26'), surface,
    round, opponent, result ('W'/'L'/None), score.
    """
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one("div.atp_player-activity") or soup
    rows = []
    for blk in container.select("div.tournament"):
        name_el = blk.select_one("h3")
        event = name_el.get_text(" ", strip=True) if name_el else ""
        loc_el = blk.select_one("span")
        location = loc_el.get_text(" ", strip=True) if loc_el else ""
        blk_text = blk.get_text(" | ", strip=True)
        m_date = re.search(r"(\d{1,2} \w+, \d{2})", blk_text)
        start_date = m_date.group(1) if m_date else ""
        m_surface = re.search(r"\b(Hard|Clay|Grass|Carpet)\b", blk_text)
        surface = m_surface.group(1) if m_surface else ""

        tokens = [t.strip() for t in blk_text.split("|") if t.strip()]
        for r in _parse_activity_rows(tokens):
            result, score = _result_from_digits(r["digits"])
            rows.append({
                "event": event,
                "location": location,
                "start_date": start_date,
                "surface": surface,
                "round": r["round"],
                "opponent": r["opponent"],
                "result": result,
                "score": score,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fetchers (Playwright; same wait-for-render + retry pattern as rankings scraper)
# ---------------------------------------------------------------------------

def _fetch_rendered(url: str, ready_selector: str, timeout_ms: int = 60000) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({"User-Agent": USER_AGENT})
        html = ""
        for attempt in range(2):
            if attempt == 0:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            else:
                page.reload(wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector(ready_selector, timeout=20000)
            except Exception:
                time.sleep(6)
            html = page.content()
            if len(html) > 50_000:
                break
        browser.close()
    return html


def fetch_tournament_results(url: str) -> pd.DataFrame:
    """All completed matches from an in-progress event's results page."""
    html = _fetch_rendered(url, "div.match")
    return parse_tournament_results(html)


def parse_match_stats(html: str) -> Optional[dict]:
    """Parse an atptour.com match-stats page into TA stat-column names.

    Returns {'p1_name', 'p2_name', 'p1': {...}, 'p2': {...}} where each side's
    dict uses the exact columns ta_scraper emits (aces, double_faults,
    serve_points, first_serves_in, first_serve_won, second_serve_won, bp_saved,
    bp_faced) plus the opp_* mirrors filled from the other side. None when the
    page has no stats table.
    """
    soup = BeautifulSoup(html, "lxml")
    names = [el.get_text(" ", strip=True) for el in soup.select("div.player-info")][:2]
    if len(names) < 2:
        return None

    def _vals(li):
        own = li.select_one("div.player-stats-item div.value")
        opp = li.select_one("div.opponent-stats-item div.value")
        return (own.get_text(" ", strip=True) if own else "", opp.get_text(" ", strip=True) if opp else "")

    def _fraction(text):
        m = re.search(r"\((\d+)/(\d+)\)", text)
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

    def _int(text):
        m = re.search(r"\d+", text)
        return int(m.group(0)) if m else None

    p1: dict = {}
    p2: dict = {}
    for li in soup.select("div.stats-group-items li"):
        legend = li.select_one("div.stats-item-legend")
        if legend is None:
            continue
        label = legend.get_text(" ", strip=True).lower()
        v1, v2 = _vals(li)
        if label == "aces":
            p1["aces"], p2["aces"] = _int(v1), _int(v2)
        elif label == "double faults":
            p1["double_faults"], p2["double_faults"] = _int(v1), _int(v2)
        elif label == "1st serve":
            n1, d1 = _fraction(v1)
            n2, d2 = _fraction(v2)
            p1["first_serves_in"], p1["serve_points"] = n1, d1
            p2["first_serves_in"], p2["serve_points"] = n2, d2
        elif label == "1st serve points won":
            p1["first_serve_won"], _ = _fraction(v1)
            p2["first_serve_won"], _ = _fraction(v2)
        elif label == "2nd serve points won":
            p1["second_serve_won"], _ = _fraction(v1)
            p2["second_serve_won"], _ = _fraction(v2)
        elif label == "break points saved":
            s1, f1 = _fraction(v1)
            s2, f2 = _fraction(v2)
            p1["bp_saved"], p1["bp_faced"] = s1, f1
            p2["bp_saved"], p2["bp_faced"] = s2, f2

    if not p1 or not p2:
        return None
    # opponent mirrors: each side's opp_* = the other side's own serve stats
    for own, other in ((p1, p2), (p2, p1)):
        for col in ("aces", "double_faults", "serve_points", "first_serves_in",
                    "first_serve_won", "second_serve_won", "bp_saved", "bp_faced"):
            own[f"opp_{col}"] = other.get(col)
    return {"p1_name": names[0], "p2_name": names[1], "p1": p1, "p2": p2}


def fetch_match_stats(url: str) -> Optional[dict]:
    """Fetch + parse one match-stats page (accepts relative ATP paths)."""
    if url.startswith("/"):
        url = "https://www.atptour.com" + url
    html = _fetch_rendered(url, "div.stats-group-items li")
    return parse_match_stats(html)


def fetch_player_activity(player_url: str) -> pd.DataFrame:
    """Completed tournaments for one player.

    ``player_url`` may be the overview path stored in data/atp_rankings.csv
    (e.g. '/en/players/jannik-sinner/s0ag/overview'); it is rewritten to the
    player-activity page.
    """
    path = player_url
    if path.endswith("/overview"):
        path = path[: -len("/overview")] + "/player-activity?matchType=Singles"
    if path.startswith("/"):
        path = "https://www.atptour.com" + path
    html = _fetch_rendered(path, "div.tournament")
    return parse_player_activity(html)
