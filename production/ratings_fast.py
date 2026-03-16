# production/ratings_fast.py
import re
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scraping"))
from utr_scraper_cloud import UTRScraper

EARLY_OLD_STREAK = 8       # stop after N consecutive old/known entries
GRACE_DAYS = 3             # small cushion for late-posted entries
MAX_ITEMS = 500            # hard cap for safety

async def fetch_ratings_incremental(
    scraper: UTRScraper,
    player_id: str,
    rating_csv_path: str,   # path to data/ratings/player_{id}_ratings.csv
) -> pd.DataFrame:
    """
    Incremental ratings fetch:
    - Reads existing ratings; finds latest date
    - Navigates to stats (t=6), selects Verified Singles, clicks 'Show all' if present
    - Scans newest->oldest history items
    - Returns ONLY rows with date > latest_date_in_file (0+ rows)
    """
    # 1) Load existing
    latest_dt = None
    existing = None
    try:
        existing = pd.read_csv(rating_csv_path)
        if not existing.empty and {'date', 'utr'}.issubset(existing.columns):
            existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
            if existing['date'].notna().any():
                latest_dt = existing['date'].max()
    except Exception:
        existing = None

    # Grace floor (keep a small window in case a late entry appears)
    date_floor = (latest_dt - timedelta(days=GRACE_DAYS)).date() if latest_dt is not None else None

    # 2) Navigate to stats page (Verified Singles)
    stats_url = f"https://app.utrsports.net/profiles/{player_id}?t=6"
    try:
        async def _go():
            await scraper.page.goto(stats_url, wait_until="load", timeout=60000)
        await scraper.retry_with_backoff(_go)
    except Exception:
        # Fallback to the full method for safety
        try:
            return await scraper.get_player_rating_history(player_id)
        except Exception:
            return pd.DataFrame(columns=['date','utr'])

    # Allow load, then ensure verified singles
    await asyncio.sleep(1.5)
    try:
        await scraper.select_verified_singles()
    except Exception:
        pass
    await asyncio.sleep(1.0)

    # 3) Click "Show all" if present
    try:
        for sel in ('a:has-text("Show all")',
                    'a.underline:has-text("Show all")',
                    'button:has-text("Show all")',
                    'div:has-text("Show all")'):
            btn = await scraper.page.query_selector(sel)
            if btn:
                await btn.click()
                await asyncio.sleep(2.5)
                break
    except Exception:
        pass

    # 4) Collect history items (reuse robust selector list from your scraper)
    history_selectors = [
        '.newStatsTabContent__historyItem__INNPC',
        'div[class*="historyItem"]',
        'div.history-item',
        '.rating-history-item',
        '[class*="rating-history"] div',
        'div[class*="history"]'
    ]
    items = []
    for sel in history_selectors:
        found = await scraper.page.query_selector_all(sel)
        if found:
            items = found
            break
    if not items:
        # fallback: HTML scrape (your function also tries this, but we only need delta)
        html = await scraper.page.content()
        pairs = re.findall(r'"date":"(\d{4}-\d{2}-\d{2})","rating":([\d.]+)', html)
        rows = []
        for d, r in pairs[:MAX_ITEMS]:
            try:
                d_dt = datetime.strptime(d, "%Y-%m-%d").date()
                if date_floor is None or d_dt > date_floor:
                    rows.append({"date": d, "utr": float(r)})
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if not df.empty and latest_dt is not None:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[df['date'] > latest_dt].copy()
        return df.sort_values('date') if not df.empty else pd.DataFrame(columns=['date','utr'])

    # 5) Parse newest->oldest; early-stop when we've hit only old/known
    new_rows = []
    old_streak = 0

    for idx, it in enumerate(items[:MAX_ITEMS]):
        # Extract date
        date_str = None
        for dsel in ('.newStatsTabContent__historyItemDate__JFjy',
                     'div[class*="Date"]', 'div[class*="date"]', 'span[class*="date"]'):
            el = await it.query_selector(dsel)
            if el:
                date_str = (await el.inner_text()).strip()
                break
        if not date_str:
            txt = await it.inner_text()
            m = re.search(r'(\d{4}-\d{2}-\d{2})', txt or '')
            if m:
                date_str = m.group(1)

        # Extract rating
        rating_str = None
        for rsel in ('.newStatsTabContent__historyItemRating__GQUXX',
                     'div[class*="Rating"]', 'div[class*="rating"]', 'span[class*="rating"]'):
            el = await it.query_selector(rsel)
            if el:
                rating_str = (await el.inner_text()).strip()
                break
        if not rating_str and date_str:
            txt = await it.inner_text()
            # remove the date once to search the remainder
            txt2 = (txt or '').replace(date_str, '', 1)
            m = re.search(r'([\d.]{2,5})', txt2)
            if m:
                rating_str = m.group(1)

        if not date_str or not rating_str:
            continue

        # Parse + compare to floor/latest
        try:
            d_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            m = re.search(r'([\d.]+)', rating_str)
            if not m:
                continue
            utr_val = float(m.group(1))
        except Exception:
            continue

        is_old = False
        if date_floor is not None and d_dt <= date_floor:
            is_old = True
        if latest_dt is not None and d_dt <= latest_dt.date():
            is_old = True

        if is_old:
            old_streak += 1
            if old_streak >= EARLY_OLD_STREAK:
                break
            continue

        old_streak = 0
        new_rows.append({"date": d_dt.strftime("%Y-%m-%d"), "utr": utr_val})

    df = pd.DataFrame(new_rows)
    return df.sort_values('date') if not df.empty else pd.DataFrame(columns=['date','utr'])
