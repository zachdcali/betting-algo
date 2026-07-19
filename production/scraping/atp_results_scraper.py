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

import io
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

try:
    from tennis_enums import ACTIVE_ROUND_CODES
except ImportError:  # pragma: no cover - package-style execution
    from production.tennis_enums import ACTIVE_ROUND_CODES  # type: ignore

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

ROUND_CODES = ACTIVE_ROUND_CODES

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
    # qualifying rounds appear on results-page match headers, not draws pages
    "1st round qualifying": "Q1",
    "first round qualifying": "Q1",
    "2nd round qualifying": "Q2",
    "second round qualifying": "Q2",
    "3rd round qualifying": "Q3",
    "final round qualifying": "Q2",  # challengers: 2-round quali; header 'Final Round' = last
    "qualifying round": "Q1",
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

def _fetch_rendered(
    url: str,
    ready_selector: str,
    timeout_ms: int = 60000,
    require_selector: bool = False,
) -> str:
    from browser_session import new_page

    page = new_page()
    try:
        html = ""
        for attempt in range(2):
            if attempt == 0:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            else:
                page.reload(wait_until="domcontentloaded", timeout=timeout_ms)
            selector_ready = False
            try:
                page.wait_for_selector(ready_selector, timeout=20000)
                selector_ready = True
            except Exception:
                time.sleep(6)
            html = page.content()
            # ATP's navigation shell is already >50 kB and contains broad
            # score/tournament links before the calendar or score cards are
            # injected. Returning merely because the shell is large caused
            # cloud runs to parse zero events while the official page became
            # ready a few seconds later. Discovery callers opt into strict
            # readiness; ordinary result pages retain the bounded large-body
            # fallback because a legitimate no-match page has no match card.
            if selector_ready or (not require_selector and len(html) > 50_000):
                break
    finally:
        page.close()
    return html


def fetch_tournament_results(url: str) -> pd.DataFrame:
    """All completed matches from an in-progress event's results page."""
    html = _fetch_rendered(url, "div.match")
    return parse_tournament_results(html)


def parse_active_events(html: str, level: str = "") -> list[dict]:
    """Active tournaments from an atptour live-scores page (tour or challenger).

    Each event card links to /en/scores/current[-challenger]/{slug}/{id}; the
    card header carries the display name + location. Surface/dates are not on
    these pages — callers assign a Monday-snapped start date and leave surface
    empty (honest) unless a registry override provides it.
    """
    soup = BeautifulSoup(html, "lxml")
    seen: dict[str, dict] = {}
    for a in soup.select("a[href*='/en/scores/current']"):
        href = a.get("href", "")
        m = re.match(r"^(/en/scores/current(?:-challenger)?/([a-z0-9-]+)/(\d+))", href)
        if not m:
            continue
        base, slug, eid = m.group(1), m.group(2), m.group(3)
        if eid in seen:
            continue
        name = ""
        node = a
        for _ in range(6):
            node = node.parent
            if node is None:
                break
            h = node.find(["h2", "h3"])
            if h and h.get_text(strip=True):
                cand = h.get_text(" ", strip=True)
                # accessibility headings render as literal "Header 2" — not a name
                if not re.fullmatch(r"header\s*\d*", cand, re.I):
                    name = cand
                    break
        seen[eid] = {
            "event": name or slug.replace("-", " ").title(),
            "slug": slug,
            "id": eid,
            "url": f"https://www.atptour.com{base}/results",
            "level": level,
            "surface": "",
        }
    return list(seen.values())


def discover_active_events() -> list[dict]:
    """Fetch active tour + challenger events from the live-scores hubs."""
    events: list[dict] = []
    # tour hub defaults to 'A' (Sackmann tour-level code); slams are corrected
    # by the registry override downstream, Davis Cup by name here
    for url, level, ready_selector in [
        (
            "https://www.atptour.com/en/scores/current",
            "A",
            "a[href^='/en/scores/current/'][href*='/draws']",
        ),
        (
            "https://www.atptour.com/en/scores/current-challenger",
            "C",
            "a[href^='/en/scores/current-challenger/'][href*='/draws']",
        ),
    ]:
        try:
            html = _fetch_rendered(url, ready_selector, require_selector=True)
            found = parse_active_events(html, level)
            if not found:
                raise RuntimeError("rendered page contained no event cards")
            for ev in found:
                if "davis cup" in str(ev.get("event", "")).lower():
                    ev["level"] = "D"
            events.extend(found)
        except Exception as exc:  # discovery is best-effort; stitching falls back to registry
            print(f"      ⚠️ active-event discovery failed for {url}: {exc}")
    # dedupe by id, tour page wins on collisions
    out: dict[str, dict] = {}
    for ev in events:
        out.setdefault(ev["id"], ev)
    return list(out.values())


_DRAW_NONNAMES = re.compile(r"^(\(\d+\)|\(WC\)|\(Q\)|\(LL\)|\(PR\)|\(SE\)|\(ALT\)|H2H|BYE|-)$", re.I)


def parse_event_draw(html: str) -> pd.DataFrame:
    """Full bracket from an atptour draws page: one row per pairing.

    Columns: round (code), p1, p2. Placeholder slots ('Qualifier / Lucky
    Loser', BYE) yield no usable names and are skipped — a first-round pairing
    resolves only once both names are real.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for header in soup.select("[class*=draw-header]"):
        rnd = _round_code_from_header(header.get_text(" ", strip=True))
        if rnd is None:
            continue
        content = header.find_next_sibling(class_=re.compile("draw-content"))
        if content is None:
            continue
        for item in content.select("[class*=draw-item]"):
            toks = [t.strip() for t in item.get_text("|", strip=True).split("|") if t.strip()]
            names = [t for t in toks
                     if not _DRAW_NONNAMES.match(t)
                     and "qualifier" not in t.lower() and "lucky loser" not in t.lower()
                     and len(t) > 2]
            if len(names) >= 2:
                rows.append({"round": rnd, "p1": names[0], "p2": names[1]})
    return pd.DataFrame(rows)


def fetch_event_draw(url: str) -> pd.DataFrame:
    """Fetch + parse a tournament draw from ATP's official static PDF first.

    ``protennislive.com`` is ATP's own publication host for immutable draw and
    order-of-play PDFs.  Those files are substantially more reliable from a
    cloud runner than the JS site (which can return only its navigation shell
    to data-centre IPs).  The rendered page remains a compatibility fallback.
    """
    pdf = fetch_official_draw_pdf(url)
    if pdf is not None and not pdf.empty:
        return pdf
    if url.endswith("/results"):
        url = url[: -len("/results")] + "/draws"
    html = _fetch_rendered(url, "[class*=draw-item]", require_selector=True)
    return parse_event_draw(html)


_SCHED_ROUND = re.compile(
    r"\b(Q\d|R128|R64|R32|R16|QF|SF|F)\b(?:\s+\1)?\s+(.*?)\s+[Vv]s\.?\s+(.*?)(?=\s*(?:–|—|H2H|$))")
_PAREN = re.compile(r"\([^)]*\)")


def parse_daily_schedule(html: str) -> pd.DataFrame:
    """(round, p1, p2) for every singles match on an event's daily-schedule page.

    The page labels each match with its exact round code (Q2, R32, ...) — the
    only official source that covers QUALIFYING matches, which draw sheets
    omit until the main draw. Never infer quali rounds from the weekday: a
    rain-shifted Q2 on Sunday would silently mislabel every feature.
    """
    from bs4 import BeautifulSoup as _BS
    text = _BS(html, "lxml").get_text(" ", strip=True)
    rows = []
    for m in _SCHED_ROUND.finditer(text):
        rnd, a, b = m.group(1), m.group(2), m.group(3)
        a = " ".join(_PAREN.sub(" ", a).split())
        b = " ".join(_PAREN.sub(" ", b).split())
        if not a or not b or "/" in a or "/" in b:   # doubles pairs
            continue
        # tour pages render doubles without a slash ("Q. Halys P. Herbert") —
        # two initialed tokens on one side = a team, never a singles player
        if len(re.findall(r"\b[A-Z]\.", a)) >= 2 or len(re.findall(r"\b[A-Z]\.", b)) >= 2:
            continue
        # a trailing scheduling phrase can prefix the name capture; keep the
        # tail tokens closest to 'Vs' (names are short: initial + surname(s))
        a = " ".join(a.split()[-4:])
        b = " ".join(b.split()[:4])
        rows.append({"round": rnd, "p1": a, "p2": b})
    return pd.DataFrame(rows)


def fetch_daily_schedule(url: str) -> pd.DataFrame:
    """Daily schedule for an event (accepts the event's results URL).

    Prefer ATP's official order-of-play PDF, then require the data-bearing
    schedule container on the JS page.  Waiting for merely ``body`` returned
    a valid-looking 90 kB navigation shell before any match cards rendered.
    """
    pdf = fetch_official_order_of_play_pdf(url)
    if pdf is not None and not pdf.empty:
        return pdf
    if url.endswith("/results"):
        url = url[: -len("/results")] + "/daily-schedule"
    html = _fetch_rendered(url, ".schedule-content", require_selector=True)
    return parse_daily_schedule(html)


# ---------------------------------------------------------------------------
# Official ATP PDF fallbacks (static, no browser / cloud-IP rendering issue)
# ---------------------------------------------------------------------------

_PDF_SLOT = re.compile(r"^\s*(\d{1,3})\s+(.*?)\s*$")
_PDF_PLAYER = re.compile(
    r"([A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ .\-'’…]+),\s*"
    r"([A-Za-zÀ-ÖØ-öø-ÿ .\-'’…]+?)\s+([A-Z]{3})(?:\s|$)"
)
_PDF_NON_PLAYER_PREFIX = re.compile(r"^(?:(?:\d+|WC|Q|LL|ALT|PR|SE|JR)\s+)+", re.I)
_PDF_NATION = re.compile(r"^\([A-Z]{3}\)$")
_PDF_SEED = re.compile(
    r"^(?:\[(?:\d+|WC|Q|LL|ALT|PR|SE|JR)\]|\(\d+\)|WC|Q|LL|ALT|PR|SE|JR)$",
    re.I,
)


def _official_pdf_base(event_url: str) -> Optional[str]:
    """Return ATP's official posting base for a current/archive event URL."""
    value = str(event_url or "")
    archive = re.search(r"/scores/archive/[^/]+/(\d+)/(\d{4})/", value)
    if archive:
        event_id, year = archive.group(1), archive.group(2)
    else:
        current = re.search(r"/scores/current(?:-challenger)?/[^/]+/(\d+)(?:/|$)", value)
        if not current:
            return None
        event_id = current.group(1)
        year = str(datetime.now(timezone.utc).year)
    return f"https://www.protennislive.com/posting/{year}/{event_id}"


def _fetch_official_pdf(event_url: str, filename: str):
    """Return ``(text, words)`` from an ATP posting PDF, or ``None``."""
    base = _official_pdf_base(event_url)
    if not base:
        return None
    import pdfplumber
    import requests

    pdf_url = f"{base}/{filename}"
    try:
        response = requests.get(
            pdf_url,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            return None
        text_parts: list[str] = []
        words: list[dict] = []
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text_parts.append(page.extract_text() or "")
                for word in page.extract_words():
                    item = dict(word)
                    item["page"] = page_number
                    words.append(item)
        return {
            "url": pdf_url,
            "text": "\n".join(text_parts),
            "words": words,
        }
    except Exception as exc:
        print(f"      ⚠️ official ATP PDF unavailable ({pdf_url}): {exc}")
        return None


def _pdf_slot_player(value: str) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower().startswith(("bye", "qualifier", "lucky loser", "tba")):
        return None
    match = _PDF_PLAYER.search(raw)
    if not match:
        # A very long name can lose its given name and comma in ATP's text
        # layer (for example ``REJCHTMAN VINCIGUERR… SWE``). Preserve the
        # substantive surname tokens so the order-of-play name can still bind
        # to this exact draw slot.
        fallback = re.match(r"(.+?)\s+[A-Z]{3}(?:\s|$)", raw)
        if not fallback:
            return None
        surname_only = _PDF_NON_PLAYER_PREFIX.sub("", fallback.group(1)).strip()
        return surname_only if surname_only else None
    surname = _PDF_NON_PLAYER_PREFIX.sub("", match.group(1)).strip()
    given = match.group(2).strip()
    if not surname or not given:
        return None
    return " ".join(f"{given} {surname}".split())


def parse_official_draw_pdf_text(text: str) -> pd.DataFrame:
    """Parse first-round pairings from an ATP main-draw PDF text layer."""
    slots: dict[int, Optional[str]] = {}
    in_draw = False
    for line in str(text or "").splitlines():
        if "Main Draw Singles" in line:
            in_draw = True
            continue
        if in_draw and re.search(r"\bRound of (?:128|64|32|16)\b", line, re.I):
            break
        if not in_draw:
            continue
        match = _PDF_SLOT.match(line)
        if not match:
            continue
        slot = int(match.group(1))
        slots[slot] = _pdf_slot_player(match.group(2))
    if not slots:
        return pd.DataFrame()
    max_slot = max(slots)
    draw_size = next((size for size in (128, 64, 32, 16, 8) if max_slot >= size), None)
    if draw_size is None:
        return pd.DataFrame()
    round_code = f"R{draw_size}" if draw_size > 8 else "QF"
    rows = []
    for slot in range(1, draw_size + 1, 2):
        p1, p2 = slots.get(slot), slots.get(slot + 1)
        if p1 and p2:
            rows.append({"round": round_code, "p1": p1, "p2": p2})
    return pd.DataFrame(rows)


def _pdf_round_from_label(label: str, qualifying_final_code: str) -> Optional[str]:
    value = " ".join(str(label or "").upper().split())
    if "QUALIFYING FINAL" in value or "FINAL QUALIFYING" in value:
        return qualifying_final_code
    for pattern, code in (
        (r"\bQ3\b", "Q3"), (r"\bQ2\b", "Q2"), (r"\bQ1\b", "Q1"),
        (r"QUALIFYING (?:ROUND )?3", "Q3"),
        (r"QUALIFYING (?:ROUND )?2", "Q2"),
        (r"QUALIFYING (?:ROUND )?1", "Q1"),
        (r"ROUND OF 128", "R128"), (r"ROUND OF 64", "R64"),
        (r"ROUND OF 32", "R32"), (r"ROUND OF 16", "R16"),
        (r"QUARTER ?FINALS?", "QF"), (r"SEMI ?FINALS?", "SF"),
    ):
        if re.search(pattern, value):
            return code
    if re.fullmatch(r"(?:SINGLES )?FINAL", value):
        return "F"
    return None


def _words_on_line(words: list[dict], top: float, cx: float, radius: float = 105.0) -> list[dict]:
    return sorted(
        [word for word in words
         if abs(float(word.get("top", 0.0)) - top) <= 1.0
         and cx - radius <= (float(word.get("x0", 0.0)) + float(word.get("x1", 0.0))) / 2 <= cx + radius],
        key=lambda word: float(word.get("x0", 0.0)),
    )


def _player_from_pdf_line(words: list[dict], top: float, cx: float) -> Optional[str]:
    tokens = []
    for word in _words_on_line(words, top, cx):
        token = str(word.get("text", "")).strip()
        if not token or _PDF_NATION.match(token) or _PDF_SEED.match(token):
            continue
        tokens.append(token)
    value = " ".join(tokens).strip()
    return value or None


def _qualifying_slots(text: str) -> dict[int, Optional[str]]:
    slots: dict[int, Optional[str]] = {}
    in_draw = False
    for line in str(text or "").splitlines():
        if "Qualifying Singles" in line:
            in_draw = True
            continue
        if in_draw and "Qualifying Round" in line:
            break
        if not in_draw:
            continue
        match = _PDF_SLOT.match(line)
        if match:
            slots[int(match.group(1))] = _pdf_slot_player(match.group(2))
    return slots


def _pdf_name_tokens(name: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(name or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("…", " ")
    return re.findall(r"[a-z]+", normalized.casefold())


def _pdf_names_match(left: str, right: str) -> bool:
    a, b = _pdf_name_tokens(left), _pdf_name_tokens(right)
    if not a or not b:
        return False
    if a[-1] == b[-1] and a[0][0] == b[0][0]:
        return True
    # ATP truncates long given/compound names with an ellipsis in the PDF text
    # layer. Require a common substantive token and matching first initial.
    shared = {token for token in set(a) & set(b) if len(token) >= 4}
    # Binding remains fail-closed at the caller: each side must resolve to one
    # and only one official slot before a qualifying round is accepted.
    return bool(shared)


def _qualifying_round_for_pair(p1: str, p2: str, qualifying_text: str) -> Optional[str]:
    slots = _qualifying_slots(qualifying_text)
    if not slots:
        return None
    p1_slots = [slot for slot, name in slots.items() if name and _pdf_names_match(p1, name)]
    p2_slots = [slot for slot, name in slots.items() if name and _pdf_names_match(p2, name)]
    if len(p1_slots) != 1 or len(p2_slots) != 1 or p1_slots[0] == p2_slots[0]:
        return None
    max_round = int(_qualifying_final_code(qualifying_text)[1:])
    a, b = p1_slots[0] - 1, p2_slots[0] - 1
    for round_number in range(1, max_round + 1):
        group_size = 2 ** round_number
        if a // group_size == b // group_size:
            return f"Q{round_number}"
    return None


def parse_official_order_of_play_words(
    words: list[dict],
    *,
    qualifying_final_code: str = "Q2",
    qualifying_text: str = "",
) -> pd.DataFrame:
    """Parse singles pairings from positioned ATP order-of-play PDF words."""
    rows = []
    pages = sorted({int(word.get("page", 0)) for word in words})
    for page_number in pages:
        page_words = [word for word in words if int(word.get("page", 0)) == page_number]
        nations = [word for word in page_words if _PDF_NATION.match(str(word.get("text", "")))]
        for versus in [word for word in page_words if str(word.get("text", "")).lower() == "vs"]:
            cx = (float(versus["x0"]) + float(versus["x1"])) / 2
            vy = float(versus["top"])
            above = [word for word in nations
                     if 0 < vy - float(word["top"]) <= 60
                     and abs(((float(word["x0"]) + float(word["x1"])) / 2) - cx) <= 115]
            below = [word for word in nations
                     if 0 < float(word["top"]) - vy <= 60
                     and abs(((float(word["x0"]) + float(word["x1"])) / 2) - cx) <= 115]
            if not above or not below:
                continue
            horizontal_distance = lambda word: abs(
                ((float(word["x0"]) + float(word["x1"])) / 2) - cx
            )
            upper_nation = min(above, key=horizontal_distance)
            lower_nation = min(below, key=horizontal_distance)
            upper_top, lower_top = float(upper_nation["top"]), float(lower_nation["top"])
            if vy - upper_top > 60 or lower_top - vy > 60:
                continue
            p1 = _player_from_pdf_line(page_words, upper_top, cx)
            p2 = _player_from_pdf_line(page_words, lower_top, cx)
            if not p1 or not p2 or "/" in p1 or "/" in p2:
                continue
            label_candidates: list[tuple[float, str]] = []
            line_tops = sorted({round(float(word.get("top", 0.0)), 1) for word in page_words})
            for top in line_tops:
                if not (upper_top - 90 <= top < upper_top):
                    continue
                label = " ".join(
                    str(word.get("text", ""))
                    for word in _words_on_line(page_words, top, cx, radius=140.0)
                )
                if _pdf_round_from_label(label, qualifying_final_code):
                    label_candidates.append((top, label))
            if label_candidates:
                _, label = max(label_candidates, key=lambda item: item[0])
                round_code = _pdf_round_from_label(label, qualifying_final_code)
            else:
                # A few single-court PDFs position the lone round heading well
                # outside the player column. Accept only a page-wide line that
                # is itself one unambiguous round label; never choose among
                # mixed round headings from a multi-court schedule.
                global_round = None
                for top in line_tops:
                    if not (upper_top - 90 <= top < upper_top):
                        continue
                    whole_line = " ".join(
                        str(word.get("text", ""))
                        for word in sorted(
                            [word for word in page_words
                             if abs(float(word.get("top", 0.0)) - top) <= 1.0],
                            key=lambda word: float(word.get("x0", 0.0)),
                        )
                    )
                    candidate = _pdf_round_from_label(whole_line, qualifying_final_code)
                    if candidate and re.fullmatch(
                        r"(?:SINGLES )?FINAL|QUALIFYING FINAL|"
                        r"Q[1-4]|R(?:128|64|32|16)|QF|SF",
                        " ".join(whole_line.upper().split()),
                    ):
                        global_round = candidate
                round_code = global_round or _qualifying_round_for_pair(
                    p1, p2, qualifying_text,
                )
            if not round_code:
                continue
            rows.append({"round": round_code, "p1": p1, "p2": p2})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def _qualifying_final_code(text: str) -> str:
    rounds = [int(value) for value in re.findall(r"Qualifying Round\s+(\d+)", str(text or ""), re.I)]
    return f"Q{max(rounds)}" if rounds else "Q2"


def fetch_official_draw_pdf(event_url: str) -> pd.DataFrame:
    payload = _fetch_official_pdf(event_url, "mds.pdf")
    if not payload:
        return pd.DataFrame()
    draw = parse_official_draw_pdf_text(payload["text"])
    if not draw.empty:
        print(f"      📄 Official ATP main-draw PDF: {len(draw)} pairings ({payload['url']})")
    return draw


def fetch_official_order_of_play_pdf(event_url: str) -> pd.DataFrame:
    order = _fetch_official_pdf(event_url, "op.pdf")
    if not order:
        return pd.DataFrame()
    qualifying = _fetch_official_pdf(event_url, "qs.pdf")
    final_code = _qualifying_final_code(qualifying["text"] if qualifying else "")
    schedule = parse_official_order_of_play_words(
        order["words"],
        qualifying_final_code=final_code,
        qualifying_text=qualifying["text"] if qualifying else "",
    )
    if not schedule.empty:
        print(f"      📄 Official ATP order-of-play PDF: {len(schedule)} pairings ({order['url']})")
    return schedule


def parse_tour_calendar(html: str) -> pd.DataFrame:
    """Season calendar from atptour.com/en/tournaments: event, slug, id, url,
    start_date. Tour-level twin of parse_challenger_calendar — the live hub
    only lists in-progress events, so Sunday's next-week tour matches (which
    Bovada already prices) need the calendar to be discoverable."""
    from bs4 import BeautifulSoup as _BS
    soup = _BS(html, "lxml")
    out, seen = [], set()
    for a in soup.select("a[href*='/en/tournaments/']"):
        m = re.match(r"^/en/tournaments/([a-z0-9-]+)/(\d+)/", a.get("href", ""))
        if not m or m.group(2) in seen:
            continue
        card = a
        for _ in range(6):
            if card.parent is None:
                break
            card = card.parent
            txt = card.get_text(" ", strip=True)
            start = _parse_start_date(txt)
            if start:
                break
        else:
            continue
        if not start:
            continue
        seen.add(m.group(2))
        name = m.group(1).replace("-", " ").title()
        surf_m = re.search(r"\b(Clay|Grass|Hard|Carpet)\b", txt, re.I)
        out.append({"event": name, "slug": m.group(1), "id": m.group(2),
                    "url": f"https://www.atptour.com/en/scores/current/{m.group(1)}/{m.group(2)}/results",
                    "start_date": start,
                    "surface": (surf_m.group(1).title() if surf_m else "")})
    return pd.DataFrame(out)


_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

# "15 - 21 June, 2026" or "28 December, 2025 - 3 January, 2026"
_DATE_RANGE = re.compile(
    r"(\d{1,2})\s*(?:(\w+),?\s*(\d{4})?)?\s*[-–]\s*\d{1,2}\s+(\w+),?\s*(\d{4})")


def _parse_start_date(text: str) -> Optional[str]:
    m = _DATE_RANGE.search(text)
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2) or m.group(4)   # "15 - 21 June, 2026" -> start month = end month
    year = m.group(3) or m.group(5)
    month = _MONTHS.get(str(month_name).capitalize())
    if not month or not year:
        return None
    return f"{int(year):04d}-{month:02d}-{day:02d}"


def parse_results_archive(html: str, year: int) -> pd.DataFrame:
    """Completed tournaments from an atptour results-archive listing.

    One row per event card (``ul.events > li`` — card-scoped, never page-level
    containers): event, slug, id, url (archive results URL), location,
    start_date (ISO). Cards without a parseable name+date are skipped.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for li in soup.select("ul.events > li"):
        a = li.find("a", href=re.compile(rf"/en/scores/archive/([a-z0-9-]+)/(\d+)/{year}/results"))
        if a is None:
            continue
        m = re.search(rf"/en/scores/archive/([a-z0-9-]+)/(\d+)/{year}/results", a.get("href", ""))
        text = re.sub(r"\s+", " ", li.get_text(" | ", strip=True))
        name = text.split(" | ")[0].strip()
        start = _parse_start_date(text)
        if not name or not start or name.lower() in ("results", "facebook"):
            continue
        loc = ""
        parts = [p.strip() for p in text.split(" | ") if p.strip()]
        if len(parts) > 1 and "," in parts[1]:
            loc = parts[1]
        rows.append({
            "event": name, "slug": m.group(1), "id": m.group(2),
            "url": f"https://www.atptour.com/en/scores/archive/{m.group(1)}/{m.group(2)}/{year}/results",
            "location": loc, "start_date": start,
        })
    return pd.DataFrame(rows)


def parse_challenger_calendar(html: str) -> pd.DataFrame:
    """Events from the atptour challenger calendar: event, slug, id, url,
    start_date. Card-scoped: each scores-link's nearest ancestor that contains
    exactly one date-range (never page-level containers — see the archive
    parser's 'Facebook' contamination lesson)."""
    soup = BeautifulSoup(html, "lxml")
    rows, seen = [], set()
    for a in soup.select("a[href*='/en/scores/']"):
        href = str(a.get("href", "")).strip()
        m = re.search(r"/en/scores/(?:current(?:-challenger)?|archive)/([a-z0-9-]+)/(\d+)", href)
        if not m or m.group(2) in seen:
            continue
        node, start = a, None
        for _ in range(5):
            node = node.parent
            if node is None:
                break
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
            found = _parse_start_date(text)
            if found:
                start, card_text = found, text
                break
        if not start:
            continue
        seen.add(m.group(2))
        name = card_text.split("|")[0].strip() if "|" in card_text else card_text[:60].strip()
        surf = re.search(r"\b(Clay|Grass|Hard|Carpet)\b", card_text, re.I)
        # Historical calendar cards already carry an immutable
        # ``/archive/<slug>/<id>/<year>/results`` URL. Rewriting every card to
        # ``current-challenger`` makes an old backlog lookup silently fetch the
        # present edition of a recurring event instead of the dated instance.
        # Preserve the official event-instance path while selecting its results
        # view (calendar buttons sometimes link to ``/draws`` instead).
        results_href = re.sub(
            r"/(?:draws|daily-schedule)(?=([?#]|$))",
            "/results",
            href,
        )
        url = (
            results_href
            if results_href.startswith("http")
            else f"https://www.atptour.com{results_href}"
        )
        rows.append({
            "event": name, "slug": m.group(1), "id": m.group(2),
            "url": url,
            "start_date": start,
            "surface": surf.group(1).title() if surf else None,
        })
    return pd.DataFrame(rows)


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
