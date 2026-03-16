# production/scraper_fast.py
import re
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List

# Reuse your real scraper + helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scraping"))
from utr_scraper_cloud import UTRScraper

EARLY_KNOWN_STREAK = 12      # stop after N consecutive known IDs
GRACE_DAYS = 10              # date cushion for late-posted results
MAX_TO_SCAN = 400            # hard cap on scanned cards

async def _get_event_headers(scraper: UTRScraper) -> List[dict]:
    """
    Mirror your 'event headers' discovery so we can assign tournament/draw
    for each card by DOM position. Returns sorted headers [{name, draw, position, index}, ...]
    """
    event_headers = []
    try:
        event_header_selector = 'div.eventItem__eventHeaderContainer__3pg9m'
        headers = await scraper.page.query_selector_all(event_header_selector)

        if headers:
            for idx, header in enumerate(headers):
                # main event name
                event_name = ""
                try:
                    elem = await header.query_selector('div.eventItem__eventName__6hntZ > span')
                    if elem:
                        event_name = (await elem.inner_text()).strip()
                except Exception:
                    pass

                # draw (fallback)
                draw_name = ""
                try:
                    elem = await header.query_selector('div.eventItem__drawName__29qpC')
                    if elem:
                        draw_name = (await elem.inner_text()).strip()
                        draw_name = re.sub(r'^\s*•\s*', '', draw_name).strip()
                except Exception:
                    pass

                if not event_name and draw_name:
                    event_name = draw_name

                if not event_name or event_name.isspace():
                    event_name = "Unknown Tournament"
                if not draw_name or draw_name.isspace():
                    draw_name = "Unknown Draw"

                # strip explicit times/days
                event_name = re.sub(r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)', '', event_name)
                event_name = re.sub(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\w+\s+\d{1,2}', '', event_name)
                event_name = event_name.strip()

                # DOM Y position
                y_position = await scraper.page.evaluate(
                    "(el) => el.getBoundingClientRect().top", header
                )

                event_headers.append({
                    "name": event_name,
                    "draw": draw_name or "Unknown Draw",
                    "position": y_position,
                    "index": idx
                })
        else:
            event_headers = []
    except Exception:
        event_headers = []

    if not event_headers:
        return [{"name": "Unknown Tournament", "draw": "Unknown Draw", "position": float("-inf"), "index": 0}]

    event_headers.sort(key=lambda x: x["position"])
    return event_headers

def _event_for_position(event_headers: List[dict], y_pos: float) -> tuple:
    """Choose last header above the card's Y position."""
    if not event_headers:
        return "Unknown Tournament", "Unknown Draw"
    best = None
    for h in event_headers:
        if h["position"] <= y_pos:
            if best is None or h["position"] > best["position"]:
                best = h
    if best:
        return best["name"], best["draw"]
    return "Unknown Tournament", "Unknown Draw"

def _compute_match_id(match_date: Optional[str], player_id: str, opponent_id: Optional[str], idx: int) -> str:
    if match_date and opponent_id:
        return f"{match_date}_{player_id}_{opponent_id}"
    return f"{player_id}_{opponent_id or 'unknown'}_{idx}"

async def fetch_current_year_incremental_full(
    scraper: UTRScraper,
    player_id: str,
    existing_csv_path: str,              # path to player_{id}_matches.csv
    year: Optional[int] = None,          # default: current year
) -> pd.DataFrame:
    """
    Incremental, full-fidelity fetch for *current year* only.
    - Navigates once
    - Captures ALL fields required by match_info
    - Early-stops after a run of known matches and/or crossing date floor
    - Returns ONLY new rows (empty DataFrame if nothing new)
    Falls back to full scraper if anything critical fails.
    """
    now_y = year or datetime.utcnow().year

    # Load existing index + date floor
    known_ids = set()
    latest_dt = None
    try:
        existing_df = pd.read_csv(existing_csv_path)
        if not existing_df.empty:
            if 'match_id' in existing_df.columns:
                # only current-year match_ids for early-known streak
                if 'date' in existing_df.columns:
                    existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
                    in_year = existing_df['date'].dt.year == now_y
                    known_ids = set(existing_df.loc[in_year, 'match_id'].dropna().astype(str))
                else:
                    known_ids = set(existing_df['match_id'].dropna().astype(str))
            # overall latest date for date floor
            if 'date' in existing_df.columns:
                existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
                if existing_df['date'].notna().any():
                    latest_dt = existing_df['date'].max()
    except Exception:
        pass

    # Navigate to profile's results tab (Singles, current year)
    try:
        profile_url = f"https://app.utrsports.net/profiles/{player_id}?t=2"
        async def _go():
            await scraper.page.goto(profile_url, wait_until="load", timeout=60000)
        await scraper.retry_with_backoff(_go)
        await asyncio.sleep(1.2)
        await scraper.select_singles_from_dropdown()
        sel = await scraper.select_year_from_dropdown(str(now_y))
        if sel == "year_not_found":
            # nothing to do—no results for current year
            return pd.DataFrame()
    except Exception:
        # fallback to full scrape for safety
        try:
            return await scraper.get_player_match_history(player_id, year=str(now_y))
        except Exception:
            return pd.DataFrame()

    # Event headers (for tournament/draw mapping)
    event_headers = await _get_event_headers(scraper)

    # Locate match cards (reusing your selectors)
    sels = [
        '.utr-card.score-card',
        '.scorecard__scorecard__3oNJK',
        '.eventItem__eventItem__2xpsd',
        'div[class*="eventItem"]',
        'div[class*="match-card"]',
        'div[class*="scorecard"]',
        'div[class*="event-card"]'
    ]
    match_cards = []
    for s in sels:
        cards = await scraper.page.query_selector_all(s)
        if cards:
            match_cards = cards
            break
    if not match_cards:
        return pd.DataFrame()

    # Prepare early-stop thresholds
    date_floor = (latest_dt - timedelta(days=GRACE_DAYS)).date() if latest_dt is not None else None
    known_streak = 0
    new_rows = []

    # Iterate newest→oldest
    for idx, card in enumerate(match_cards[:MAX_TO_SCAN]):
        # --- retired? (don't skip; just mark) ---
        retired = False
        try:
            t = (await card.inner_text()) or ""
            if "retired" in t.lower():
                retired = True
        except Exception:
            pass

        # --- header: time | date | round ---
        header_text = None
        for hsel in (
            'div[class*="scorecard_header_2iDdF"]',
            '[class*="header_2iDdF"]',
            'div[class*="header"] > div',
            '.scorecard__header__2iDdF',
            '[class*="header"]',
            'div.date',
        ):
            elem = await card.query_selector(hsel)
            if elem:
                header_text = (await elem.inner_text()).strip()
                # Prefer the one with pipes
                if '|' in header_text:
                    break

        match_time = None
        match_date = None
        match_round = None

        if header_text and '|' in header_text:
            parts = [p.strip() for p in header_text.split('|')]
            if len(parts) >= 1:
                match_time = parts[0]
            if len(parts) >= 2:
                date_part = parts[1]
                try:
                    match_date = datetime.strptime(f"{date_part} {now_y}", "%b %d %Y").strftime("%Y-%m-%d")
                except Exception:
                    pass
            if len(parts) >= 3:
                match_round = re.sub(r'\s*•\s*Singles', '', parts[2], flags=re.IGNORECASE)
                if match_round.lower() == "singles":
                    match_round = "Unknown Round"
        else:
            # loose regex
            if header_text:
                m = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})', header_text)
                if m:
                    try:
                        match_date = datetime.strptime(f"{m.group(1)} {m.group(2)} {now_y}", "%b %d %Y").strftime("%Y-%m-%d")
                    except Exception:
                        pass
                r = re.search(r'(Round of \d+|Final|Semi-?final|Quarter-?final|Qualifier)', header_text, re.IGNORECASE)
                if r:
                    match_round = r.group(1)

        if not match_round:
            # element-based round fallback
            elem = await card.query_selector('.round, [class*="round"], [class*="phase"]')
            if elem:
                rt = (await elem.inner_text()).strip()
                rt = re.sub(r'\s*•\s*Singles', '', rt, flags=re.IGNORECASE)
                match_round = "Unknown Round" if rt.lower() == "singles" else rt
            if not match_round:
                match_round = "Unknown Round"

        # --- opponent info & UTRs (player/opponent names and displayed UTRs) ---
        player_name = None
        opponent_name = None
        player_utr = None
        opponent_utr = None
        opponent_id = None

        # try to find name/utr elements
        try:
            # names
            name_elems = await card.query_selector_all('[class*="player-name"], .player-name, div.name')
            if name_elems and len(name_elems) >= 2:
                player_name = (await name_elems[0].inner_text()).strip()
                opponent_name = (await name_elems[1].inner_text()).strip()
        except Exception:
            pass

        # opponent id
        try:
            opp_link = await card.query_selector('a[href*="profiles"]')
            if opp_link:
                href = await opp_link.get_attribute('href')
                mo = re.search(r'/profiles/(\d+)', href or '')
                if mo:
                    opponent_id = mo.group(1)
        except Exception:
            pass

        # displayed UTRs
        try:
            # try to collect two UTR values; fallback is fine if not present
            player_utr_elem = await card.query_selector('.team .utr, .team [class*="utr"], .team [class*="rating"]')
            opponent_utr_elem = await card.query_selector('.team:nth-of-type(2) .utr, .team:nth-of-type(2) [class*="utr"], .team:nth-of-type(2) [class*="rating"]')
            if player_utr_elem:
                txt = (await player_utr_elem.inner_text()).strip()
                if txt == "UR":
                    player_utr = "UR"
                else:
                    m = re.search(r'([\d.]+)', txt)
                    if m:
                        player_utr = float(m.group(1))
            if opponent_utr_elem:
                txt = (await opponent_utr_elem.inner_text()).strip()
                if txt == "UR":
                    opponent_utr = "UR"
                else:
                    m = re.search(r'([\d.]+)', txt)
                    if m:
                        opponent_utr = float(m.group(1))
        except Exception:
            pass

        # --- score (use your cleaner) ---
        player_scores, opponent_scores = [], []
        cleaned_score, raw_score = "", ""
        try:
            p_elems = await card.query_selector_all('.score-item, [class*="score-item"]')
            o_elems = await card.query_selector_all('.score-item, [class*="score-item"]')
            for i in range(min(len(p_elems), len(o_elems))):
                p_raw = ((await p_elems[i].inner_text()) or "").strip().replace('\n', '')
                o_raw = ((await o_elems[i].inner_text()) or "").strip().replace('\n', '')
                # scoreboard cells are often one side per column; we still join into "p-o"
                player_scores.append(p_raw)
                opponent_scores.append(o_raw)
            if player_scores and opponent_scores:
                raw_score = " ".join([f"{p}-{o}" for p, o in zip(player_scores, opponent_scores)])
                cleaned_score = scraper.clean_tennis_score(raw_score)  # <-- reuse your method
        except Exception:
            pass

        # --- result (winner) ---
        is_winner = None
        try:
            # keep it simple but robust
            win_sel = '[class*="winner-display-container"] [class*="winning-scores"], .winning-scores, div[class*="winner-display"] div[class*="winning"]'
            player_w = await card.query_selector(win_sel)  # if this finds inside player's side first, it's W
            # If ambiguous, fallback to set counting
            if player_w:
                is_winner = True
            else:
                # fallback via set wins
                pw, ow = 0, 0
                for ps, os in zip(player_scores, opponent_scores):
                    mp = re.search(r'(\d+)', ps or "")
                    mo = re.search(r'(\d+)', os or "")
                    if mp and mo:
                        p, o = int(mp.group(1)), int(mo.group(1))
                        if p > o: pw += 1
                        elif o > p: ow += 1
                if pw or ow:
                    is_winner = pw > ow
        except Exception:
            is_winner = None

        result = "W" if is_winner else ("L" if is_winner is not None else None)

        # --- tournament/draw via event header position ---
        try:
            y_pos = await scraper.page.evaluate("(el)=>el.getBoundingClientRect().top", card)
        except Exception:
            y_pos = float('inf')
        tournament, draw = _event_for_position(event_headers, y_pos)

        # --- tournament type / exhibition flag ---
        tournament_type = "Regular"
        is_exhibition = False
        for term in ('exhibition', 'laver cup', 'show match', 'charity'):
            if term in (tournament or "").lower():
                tournament_type = "Exhibition"
                is_exhibition = True
                break

        # --- compute match_id + early-stop checks ---
        m_id = _compute_match_id(match_date, player_id, opponent_id, idx)
        if m_id in known_ids:
            known_streak += 1
        else:
            known_streak = 0
        if known_streak >= EARLY_KNOWN_STREAK:
            break

        if match_date and date_floor:
            try:
                dt = datetime.strptime(match_date, "%Y-%m-%d").date()
                if dt < date_floor and known_streak >= 3:
                    break
            except Exception:
                pass

        # --- append only if NEW ---
        if m_id not in known_ids:
            # Excel date coercion guard (same as your pipeline)
            if cleaned_score:
                # add a leading apostrophe only if contains month abbrev; otherwise keep as cleaner output
                mo = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', cleaned_score)
                if mo:
                    cleaned_score = re.sub(r'(?=.)', "'", cleaned_score, count=1)

            new_rows.append({
                "match_id": m_id,
                "player_id": player_id,
                "player_name": player_name,
                "player_utr_displayed": player_utr,
                "date": match_date,
                "time": match_time,
                "tournament": tournament,
                "draw": draw,
                "tournament_type": tournament_type,
                "round": match_round,
                "opponent_name": opponent_name,
                "opponent_id": opponent_id,
                "opponent_utr_displayed": opponent_utr,
                "score": cleaned_score,
                "raw_score": raw_score or cleaned_score,
                "result": result,
                "retired": retired,
                "is_exhibition": is_exhibition
            })

    return pd.DataFrame(new_rows)
