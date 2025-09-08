#!/usr/bin/env python3
"""
Bovada Tennis Odds Fetcher (Hybrid row discovery, doubles-aware expected counts)

Key points:
- HYBRID row discovery:
  • From each coupon, collect sp-score-coupon rows; climb to nearest section.coupon-content if present
  • Also collect section.coupon-content that directly contains competitor names
  • Union & de-dup
- Singles-only extraction; doubles detected by slashes/&/commas or 4+ tokens per side
- Page-level + in-coupon "Show more" draining (bounded)
- Uses parentheses "(N)" in headers as expected counts, but now computes an
  Adjusted Expected (singles) = header (N) - doubles_detected_in_bucket
- Prints both raw and adjusted expected totals and compares found vs adjusted
"""

from playwright.sync_api import sync_playwright
import pandas as pd
from datetime import datetime, timezone
import re
import time
import unicodedata
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# =========================
# Configuration / Filters
# =========================

ALLOWED_EVENTS = ("ATP", "Challenger", "ITF", "Masters", "Grand Slam", "Davis Cup", "Men's")
BLOCKED_EVENTS = ("Women", "WTA", "Exhibition", "UTR", "Doubles")

# Logging
VERBOSE = False  # set True for extra debug prints

# Row acceptance policy
REQUIRE_WIN = False         # require moneyline present to keep a row
ALLOW_PARTIAL_ROWS = True   # if REQUIRE_WIN == False, keep rows without ML (names + time only)

# Scroll / timing caps
SCROLL_PASSES_INITIAL = 5
SCROLL_PASSES_SETTLE = 2
SCROLL_DY = 2200
SCROLL_PAUSE_MS = 200
SHOW_MORE_PASSES = 8
SHOW_MORE_PAUSE_MS = 220

# In-coupon "Show more" caps
COUPON_SHOW_MORE_MAX_PASSES = 5
COUPON_SHOW_MORE_CLICK_CAP = 10
COUPON_SHOW_MORE_PAUSE_MS = 200

# Global waits
DEFAULT_TIMEOUT_MS = 16000
NAV_TIMEOUT_MS = 35000

# =========================
# Utilities
# =========================

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# Doubles only if names contain a slash (e.g., "Hsu/Sakellaridis")
DBL_TEAM_PAT = re.compile(r"/")
LEVEL_PAT = re.compile(r"(ATP|Challenger|ITF(?:\s+M\d+)?|Masters|Grand Slam|Davis Cup)", re.I)

def parse_level_and_name(event_title: str) -> tuple[str, str]:
    # event_title like "Challenger - Rennes (14)" or "ITF Men's - ITF M25 Plaisir (12)"
    t = event_title or ""
    # Drop the trailing "(N)"
    core = re.sub(r"\s*\(\d+\)\s*$", "", t).strip()
    # Split on first " - " to separate circuit from tourney name when present
    if " - " in core:
        left, right = core.split(" - ", 1)
        level_match = LEVEL_PAT.search(left) or LEVEL_PAT.search(right)
        level = (level_match.group(0) if level_match else left).strip()
        tourney = right.strip()
    else:
        level_match = LEVEL_PAT.search(core)
        level = (level_match.group(0) if level_match else core).strip()
        tourney = core.strip()
    return (level, tourney)

def looks_doubles(name: str) -> bool:
    if not name:
        return False
    return bool(DBL_TEAM_PAT.search(name))

def names_look_doubles(p1: str, p2: str) -> bool:
    # treat as doubles if either side has a slash
    return looks_doubles(p1) or looks_doubles(p2)

def is_blocked_bucket(title: str) -> bool:
    t = (title or "").lower()
    if "doubles" in t:
        return True
    return any(b.lower() in t for b in BLOCKED_EVENTS)

def is_allowed_bucket(title: str) -> bool:
    t = (title or "").lower()
    return any(a.lower() in t for a in ALLOWED_EVENTS)

def expected_count_from_title(title: str) -> int:
    m = re.search(r"\((\d+)\)", title or "")
    return int(m.group(1)) if m else -1

def normalize_name(name_str: str) -> str:
    if not name_str or pd.isna(name_str):
        return ""
    name = unicodedata.normalize("NFKD", str(name_str))
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"[^a-zA-Z\s\-]", "", name).lower().strip()
    name = re.sub(r"\s+", " ", name)
    return name

def american_to_decimal(odds_str: Optional[str]) -> float:
    if not odds_str:
        return 2.0
    s = str(odds_str).strip().upper()
    if s in {"EVEN", "PK", "PICK", "PICK'EM"}:
        return 2.0
    try:
        if s.startswith("+") or s.startswith("-"):
            s = s.replace("+", "")
        american = int(float(s))
        return 1 + (american / 100.0) if american > 0 else 1 + (100.0 / abs(american))
    except Exception:
        return 2.0

def parse_match_time(text: str) -> str:
    if not text:
        return "Unknown"
    m_time = re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", text, re.I)
    m_date = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
    if m_date and m_time:
        return f"{m_date.group(0)} {m_time.group(0)}"
    if m_date:
        return m_date.group(0)
    if m_time:
        m_rel = re.search(r"\b(Today|Tomorrow)\b", text, re.I)
        return f"{m_rel.group(0)} {m_time.group(0)}" if m_rel else m_time.group(0)
    return "Unknown"

def dedupe_key(bucket_title: str, p1: str, p2: str, match_time: str,
               win_tuple: Optional[Tuple[Optional[str], Optional[str]]]) -> str:
    a, b = sorted([normalize_name(p1), normalize_name(p2)])
    w1, w2 = "", ""
    if win_tuple:
        w1 = win_tuple[0] or ""
        w2 = win_tuple[1] or ""
    return f"{bucket_title}|{a}|{b}|{match_time}|{w1}|{w2}"

# =========================
# Playwright helpers
# =========================

def dismiss_banners(page) -> None:
    try:
        sel = ("button:has-text('Accept All'), button:has-text('I Agree'), "
               "button:has-text('Got it'), button:has-text('OK')")
        btns = page.locator(sel)
        if btns.count():
            btns.first.click(timeout=1000)
            page.wait_for_timeout(150)
    except Exception:
        pass

def lazy_scroll(page, passes=SCROLL_PASSES_INITIAL, dy=SCROLL_DY, pause_ms=SCROLL_PAUSE_MS) -> None:
    for _ in range(passes):
        j = random.randint(-200, 200)
        page.mouse.wheel(0, dy + j)
        page.wait_for_timeout(pause_ms + random.randint(0, 80))

def drain_bottom_show_more(page, max_passes=SHOW_MORE_PASSES, pause_ms=SHOW_MORE_PAUSE_MS) -> None:
    for _ in range(max_passes):
        btns = page.locator("button:has-text('Show More'), button:has-text('Show more')")
        if not btns.count():
            return
        page.mouse.wheel(0, 4000)
        b = btns.last
        if not b.is_visible():
            return
        try:
            b.click(timeout=1500)
            page.wait_for_timeout(pause_ms)
        except Exception:
            return

def get_next_events_root(page):
    """
    Return the container for TENNIS - NEXT EVENTS.
    Avoids scraping the LIVE BETTING ODDS bucket.
    """
    for sel in ("sp-next-events-bucket", "sp-next-events", "div.next-events__bucket"):
        loc = page.locator(sel)
        if loc.count():
            return loc.first
    # Fallback: if we can't find the container, use page (old behavior)
    return page

def stabilize_headers(page, min_headers=10, max_wait_ms=7000, poll_ms=250) -> int:
    from datetime import datetime as _dt
    root = get_next_events_root(page)
    start = _dt.now()
    prev = -1
    while True:
        headers = root.locator("h4.header-collapsible.league-header-collapsible")
        cur = headers.count()
        if cur >= min_headers and cur == prev:
            return cur
        if (_dt.now() - start).total_seconds() * 1000 > max_wait_ms:
            return cur
        prev = cur
        page.mouse.wheel(0, 1600)
        drain_bottom_show_more(page, max_passes=1, pause_ms=120)
        page.wait_for_timeout(poll_ms)

def find_tournament_buckets(page) -> List[Any]:
    """
    Only buckets under TENNIS - NEXT EVENTS.
    Explicitly avoid LIVE containers (sp-happening-now-bucket).
    """
    root = get_next_events_root(page)
    # safety: exclude anything whose ancestor is happening-now
    blocks = root.locator("div.grouped-events").filter(
        has_not=page.locator("xpath=ancestor::sp-happening-now-bucket")
    )
    out = []
    for i in range(blocks.count()):
        blk = blocks.nth(i)
        if blk.locator("h4.header-collapsible").count():
            out.append(blk)
    return out

def get_bucket_title(bucket) -> str:
    try:
        a = bucket.locator("h4.header-collapsible a.league-header-collapsible__description")
        if a.count():
            return a.first.inner_text().strip()
        return bucket.locator("h4.header-collapsible").first.inner_text().strip()
    except Exception:
        return ""

def expand_bucket_if_needed(bucket) -> None:
    try:
        h = bucket.locator("h4.header-collapsible").first
        cls = h.get_attribute("class") or ""
        if "league-header-collapsible--collapsed" in cls:
            h.click(timeout=2000)
            h.page.wait_for_timeout(180)
        bucket.evaluate("el => el.scrollIntoView({block:'center'})")
        h.page.wait_for_timeout(100)
    except Exception:
        pass

def coupons_in_bucket(bucket) -> List[Any]:
    cps = bucket.locator("sp-coupon")
    return [cps.nth(i) for i in range(cps.count())]

def drain_coupon_until_stable(page, coupon,
                              max_passes=COUPON_SHOW_MORE_MAX_PASSES,
                              pause_ms=COUPON_SHOW_MORE_PAUSE_MS,
                              click_cap=COUPON_SHOW_MORE_CLICK_CAP) -> None:
    btn_sel = ("button:has-text('Show More'), button:has-text('Show more'), "
               "button:has-text('Show more bets'), button:has-text('More Bets')")
    last_count = -1
    total_clicked = 0
    for _ in range(max_passes):
        rows = coupon.locator("sp-score-coupon")
        cur = rows.count()
        if cur <= last_count:
            break
        last_count = cur
        btns = coupon.locator(btn_sel)
        n = min(btns.count(), max(0, click_cap - total_clicked))
        clicked = 0
        for i in range(n):
            try:
                b = btns.nth(i)
                if b.is_visible():
                    b.click(timeout=1200)
                    total_clicked += 1
                    clicked += 1
                    if total_clicked >= click_cap:
                        break
                    page.wait_for_timeout(pause_ms)
            except Exception:
                continue
        page.mouse.wheel(0, 1400)
        page.wait_for_timeout(pause_ms)
        if clicked == 0:
            break

# =========================
# Market parsing helpers
# =========================

def parse_prices(node) -> List[str]:
    ps = node.locator("span.bet-price")
    return [ps.nth(i).inner_text().strip() for i in range(ps.count())]

def classify_market_group(group_node) -> str:
    # Only treat as TOTAL/SPREAD if the DOM actually exposes those structures.
    html = group_node.inner_html().lower()
    txt = group_node.inner_text().lower()

    # TOTAL must have explicit total affordances
    if ("both-handicaps" in html) or \
       group_node.locator("span.market-line.bet-handicap.both-handicaps").count() > 0 or \
       re.search(r"\bover\b|\bunder\b|\b o\b|\b u\b", txt):
        return "TOTAL"

    # SPREAD must have a handicap element
    if ("sp-spread-outcome" in html) or \
       group_node.locator("span.market-line.bet-handicap:not(.both-handicaps)").count() > 0:
        return "SPREAD"

    # Otherwise it's WIN (moneyline)
    return "WIN"

def extract_totals_value(group_node) -> Optional[str]:
    try:
        val = group_node.locator("span.market-line.bet-handicap.both-handicaps")
        if val.count():
            m = re.search(r"\d+(?:\.\d+)?", val.first.inner_text())
            if m:
                return m.group(0)
        m = re.search(r"\d+(?:\.\d+)?", group_node.inner_text())
        return m.group(0) if m else None
    except Exception:
        return None

def is_group_suspended(group_node) -> bool:
    """
    Consider the group suspended if:
      - It or any ancestor/descendant has class containing 'suspend'
      - There is <ul class="suspended"> or <li class="suspended|disabled">
      - aria-disabled="true" exists on self/ancestor
      - prices are struck out via <s> or <del> inside bet-price
    """
    # Fast path: explicit UL/LI flags
    if group_node.locator("ul.suspended, li.suspended, li.disabled").count() > 0:
        return True

    # Any 'suspend' class on self/ancestor
    if group_node.locator("xpath=ancestor-or-self::*[contains(translate(@class,'SUSPEND','suspend'),'suspend')]").count() > 0:
        return True

    # aria-disabled flags
    if group_node.locator("xpath=ancestor-or-self::*[@aria-disabled='true']").count() > 0:
        return True

    # strike-through price content
    if group_node.locator("span.bet-price s, span.bet-price del").count() > 0:
        return True

    # Text safety (rare)
    txt = (group_node.inner_text() or "").lower()
    if "suspended" in txt:
        return True

    return False

def is_empty_market(group_node) -> bool:
    # Bovada renders empty cells as <li class="empty-bet">
    if group_node.locator("li.empty-bet").count() >= 1:
        # Also ensure there are no prices in this group
        if group_node.locator("span.bet-price").count() == 0:
            return True
    return False

def extract_spread_value(group_node) -> Optional[str]:
    try:
        val = group_node.locator("span.market-line.bet-handicap:not(.both-handicaps)")
        if val.count():
            m = re.search(r"[-+]\d+(?:\.\d+)?", val.first.inner_text())
            if m:
                return m.group(0)
        m = re.search(r"[-+]\d+(?:\.\d+)?", group_node.inner_text())
        return m.group(0) if m else None
    except Exception:
        return None

# =========================
# Row discovery (HYBRID)
# =========================

def nearest_section_host(node):
    """Return nearest section.coupon-content ancestor if any, else the node itself."""
    sec = node.locator("xpath=ancestor::section[contains(@class,'coupon-content')][1]")
    return sec if sec.count() else node

def discover_match_hosts(bucket) -> List[Any]:
    """
    Prefer DOM order: take section.coupon-content blocks first (as they appear on the page),
    then add any extra hosts we only reach via sp-score-coupon mapping.
    """
    hosts = []
    seen_handles = set()

    def add_host(h):
        try:
            oid = h.evaluate("el => el ? (el.__unique || (el.__unique = Math.random().toString(36).slice(2))) : ''")
        except Exception:
            oid = None
        if oid and oid in seen_handles:
            return
        if oid:
            seen_handles.add(oid)
        hosts.append(h)

    # B) Sections in DOM order (primary source of truth for order)
    sections = bucket.locator("section.coupon-content:has(h4.competitor-name span.name)")
    for i in range(sections.count()):
        add_host(sections.nth(i))

    # A) Any rows reachable via sp-score-coupon → map to nearest section host (fills gaps)
    cps = coupons_in_bucket(bucket)
    for cp in cps:
        try:
            drain_coupon_until_stable(cp.page, cp)
        except Exception:
            pass
        rows = cp.locator("sp-score-coupon")
        for i in range(rows.count()):
            host = nearest_section_host(rows.nth(i))
            add_host(host)

    return hosts

# =========================
# Scraping routine
# =========================

def fetch_bovada_tennis_odds(headless: bool = True, max_retries: int = 3) -> pd.DataFrame:
    url = "https://www.bovada.lv/sports/tennis"

    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(
                    viewport={"width": 1360, "height": 900},
                    user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/118.0.0.0 Safari/537.36"),
                    java_script_enabled=True,
                    locale="en-US",
                )
                page = context.new_page()
                page.set_default_timeout(DEFAULT_TIMEOUT_MS)

                # Block heavy assets
                page.route("**/*", lambda route: route.abort()
                           if route.request.resource_type in {"image", "media", "font"}
                           else route.continue_())

                # Navigate
                page.goto(url, timeout=NAV_TIMEOUT_MS)
                page.wait_for_load_state("domcontentloaded")
                try:
                    page.wait_for_selector("sp-coupon", timeout=15000)
                except Exception:
                    vprint("⚠️  'sp-coupon' not seen; waiting for '.grouped-events' instead…")
                    page.wait_for_selector(".grouped-events", timeout=10000)

                # STEP 1: page-wide load + bottom drain
                print("🔄 Initial lazy scroll and Show More drain...")
                lazy_scroll(page, passes=SCROLL_PASSES_INITIAL, dy=SCROLL_DY, pause_ms=SCROLL_PAUSE_MS)
                dismiss_banners(page)
                drain_bottom_show_more(page, max_passes=SHOW_MORE_PASSES, pause_ms=SHOW_MORE_PAUSE_MS)

                # Stabilize headers
                print("🧭 Stabilizing header list…")
                hdr_count = stabilize_headers(page, min_headers=10)
                print(f"   Stabilized at ~{hdr_count} tournament headers")

                if hdr_count < 10:
                    print("♻️  Low header count; soft reload & re-drain…")
                    page.reload(wait_until="domcontentloaded")
                    dismiss_banners(page)
                    lazy_scroll(page, passes=SCROLL_PASSES_INITIAL, dy=SCROLL_DY, pause_ms=SCROLL_PAUSE_MS)
                    drain_bottom_show_more(page, max_passes=SHOW_MORE_PASSES, pause_ms=SHOW_MORE_PAUSE_MS)
                    hdr_count = stabilize_headers(page, min_headers=10)
                    print(f"   After reload: {hdr_count} tournament headers")

                # STEP 2: Expand allowed headers
                print("🎾 Expanding all tournament headers...")
                root = get_next_events_root(page)
                headers = root.locator("h4.header-collapsible.league-header-collapsible")
                header_total = headers.count()
                print(f"   Found {header_total} tournament headers")

                expanded = 0
                for i in range(header_total):
                    h = headers.nth(i)
                    try:
                        title = h.inner_text().strip()
                    except Exception:
                        title = ""
                    
                    cls = h.get_attribute("class") or ""
                    already_open = "league-header-collapsible--collapsed" not in cls
                    
                    if is_blocked_bucket(title) or not is_allowed_bucket(title):
                        print(f"   {i+1:2d}. 🚫 Skip bucket: {title}")
                        continue
                        
                    if already_open:
                        print(f"   {i+1:2d}. ⚪ Already open: {title}")
                    else:
                        h.click(timeout=2000)
                        page.wait_for_timeout(180)
                        expanded += 1
                        print(f"   {i+1:2d}. ✅ Expanded: {title}")
                    
                    h.evaluate("el => el.scrollIntoView({block:'center'})")
                    page.wait_for_timeout(90)

                print(f"   Successfully expanded {expanded} tournaments (of {header_total} total headers)")

                # STEP 3: settle + drain again
                print("🧩 Settling coupons…")
                lazy_scroll(page, passes=SCROLL_PASSES_SETTLE, dy=SCROLL_DY, pause_ms=SCROLL_PAUSE_MS)
                drain_bottom_show_more(page, max_passes=SHOW_MORE_PASSES, pause_ms=SHOW_MORE_PAUSE_MS)
                page.wait_for_load_state("networkidle")

                # STEP 4: Process buckets (HYBRID)
                print("🔍 Processing individual tournament buckets...")
                seen: set = set()
                out_rows: List[Dict[str, Any]] = []
                discrepancies: Dict[str, Dict[str, int]] = {}

                # For adjusted expected accounting:
                # bucket_stats[title] = {"expected_raw": N, "doubles_detected": D, "found_singles": K}
                bucket_stats: Dict[str, Dict[str, int]] = {}

                buckets = find_tournament_buckets(page)
                allowed = [b for b in buckets if is_allowed_bucket(get_bucket_title(b)) and not is_blocked_bucket(get_bucket_title(b))]
                print(f"   Buckets on page (allowed/total): {len(allowed)}/{len(buckets)}")

                # Sum expected counts (raw; will also compute adjusted later)
                sum_expected_raw = 0
                for b in buckets:
                    t = get_bucket_title(b)
                    if is_blocked_bucket(t) or not is_allowed_bucket(t):
                        continue
                    exp = expected_count_from_title(t)
                    if exp > 0:
                        sum_expected_raw += exp
                # adjusted sum will be printed after processing when doubles are known
                print(f"🧮 Expected-by-titles (men's buckets) RAW: ~{sum_expected_raw}")

                def process_hosts(bucket, title: str, doubles_keys_seen: set, level: str, tourney_name: str, bucket_index: int) -> Tuple[int, int]:
                    """
                    Returns: (found_added, doubles_seen_here)
                    """
                    added_here = 0
                    doubles_here = 0
                    host_index = 0

                    hosts = discover_match_hosts(bucket)
                    print(f"      • rows/hosts discovered: {len(hosts)}")

                    for host in hosts:
                        # two names?
                        names = host.locator("h4.competitor-name span.name")
                        if names.count() < 2:
                            names = host.locator(".competitor-name .name, span.name")
                            if names.count() < 2:
                                continue

                        p1 = names.nth(0).inner_text().strip()
                        p2 = names.nth(1).inner_text().strip()

                        # ----- doubles guard: skip saving, but count once for expected adjustment -----
                        if names_look_doubles(p1, p2):
                            # de-dup doubles seen so we don't subtract twice if the row is discovered via two paths
                            when_el = host.locator("time.clock")
                            match_time = when_el.first.inner_text().strip() if when_el.count() else parse_match_time(host.inner_text())
                            dbl_key = dedupe_key(title, p1, p2, match_time, None)
                            if dbl_key not in doubles_keys_seen:
                                doubles_keys_seen.add(dbl_key)
                                doubles_here += 1
                            continue
                        # ---------------------------------------------------------------------------

                        # (optional) live guard – skip pure live rows with no concrete time
                        try:
                            ttxt = host.inner_text().lower()
                            if (" live " in f" {ttxt} ") and not re.search(r"\b(today|tomorrow|\d{1,2}:\d{2}\s*(am|pm))\b", ttxt):
                                continue
                        except Exception:
                            pass

                        groups = host.locator("sp-two-way-vertical")
                        markets = {"WIN": None, "SPREAD": None, "TOTAL": None}

                        for gi in range(groups.count()):
                            g = groups.nth(gi)

                            # NEW: ignore suspended or empty market blocks
                            if is_group_suspended(g) or is_empty_market(g):
                                continue

                            kind = classify_market_group(g)
                            prices = parse_prices(g)

                            if kind == "WIN" and markets["WIN"] is None and len(prices) >= 2:
                                markets["WIN"] = (prices[0], prices[1])

                            elif kind == "SPREAD" and markets["SPREAD"] is None:
                                hcap = extract_spread_value(g)
                                if hcap and len(prices) >= 2:
                                    markets["SPREAD"] = (hcap, prices[0], prices[1])

                            elif kind == "TOTAL" and markets["TOTAL"] is None:
                                tot = extract_totals_value(g)
                                if tot and len(prices) >= 2:
                                    markets["TOTAL"] = (tot, prices[0], prices[1])


                        # If nothing is bettable (all markets missing/suspended), treat this row as suspended
                        row_bettable = any(m is not None for m in markets.values())
                        if not row_bettable:
                            # count once toward the "subtract from expected" bucket accounting, like doubles
                            when_el = host.locator("time.clock")
                            match_time = when_el.first.inner_text().strip() if when_el.count() else parse_match_time(host.inner_text())
                            sus_key = dedupe_key(title, p1, p2, match_time, None)
                            if sus_key not in doubles_keys_seen:  # reuse the "seen once" set so we don't subtract twice
                                doubles_keys_seen.add(sus_key)
                                doubles_here += 1  # reuse same counter so adjusted expected = raw - doubles/suspended
                            continue

                        when_el = host.locator("time.clock")
                        match_time = when_el.first.inner_text().strip() if when_el.count() else parse_match_time(host.inner_text())

                        # row acceptance policy
                        if REQUIRE_WIN and markets["WIN"] is None:
                            continue
                        if (not REQUIRE_WIN) and (not ALLOW_PARTIAL_ROWS) and markets["WIN"] is None:
                            continue

                        key = dedupe_key(title, p1, p2, match_time, markets["WIN"])
                        if key in seen:
                            continue
                        seen.add(key)

                        # Market flags and metadata
                        has_win = markets["WIN"] is not None
                        has_spread = markets["SPREAD"] is not None
                        has_total = markets["TOTAL"] is not None
                        row_bettable = any((has_win, has_spread, has_total))
                        is_live = bool(re.search(r"\blive\b", host.inner_text().lower()))
                        host_index += 1

                        o1 = o2 = d1 = d2 = None
                        if markets["WIN"] is not None:
                            o1, o2 = markets["WIN"]
                            d1, d2 = american_to_decimal(o1), american_to_decimal(o2)

                        spread_val = spread_o1 = spread_o2 = None
                        total_val = total_o1 = total_o2 = None
                        if markets["SPREAD"]:
                            spread_val, spread_o1, spread_o2 = markets["SPREAD"]
                        if markets["TOTAL"]:
                            total_val, total_o1, total_o2 = markets["TOTAL"]

                        out_rows.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "scrape_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "source": "bovada",
                            "event": title,
                            "tourney_level": level,
                            "tourney_name": tourney_name,
                            "bucket_index": bucket_index,
                            "match_index": host_index,
                            "player1_raw": p1,
                            "player2_raw": p2,
                            "player1_normalized": normalize_name(p1),
                            "player2_normalized": normalize_name(p2),
                            "match_time": match_time,
                            "match_time_raw": match_time,
                            "is_live": is_live,
                            "player1_odds_american": o1,
                            "player2_odds_american": o2,
                            "player1_odds_decimal": d1,
                            "player2_odds_decimal": d2,
                            "player1_implied_prob": (1.0 / d1) if d1 and d1 > 0 else None,
                            "player2_implied_prob": (1.0 / d2) if d2 and d2 > 0 else None,
                            "spread_handicap": spread_val,
                            "spread_odds_p1": spread_o1,
                            "spread_odds_p2": spread_o2,
                            "total_games": total_val,
                            "total_odds_over": total_o1,
                            "total_odds_under": total_o2,
                            "has_win": has_win,
                            "has_spread": has_spread,
                            "has_total": has_total,
                            "is_bettable_row": row_bettable,
                            "row_key": key,
                        })
                        added_here += 1

                        print(f"         ✅ {p1} vs {p2} | "
                              f"Win {o1 if o1 else '-'} / {o2 if o2 else '-'} | "
                              f"Spread {spread_val or '-'} ({spread_o1}/{spread_o2}) | "
                              f"Total {total_val or '-'} ({total_o1}/{total_o2})")

                    return added_here, doubles_here

                for bi, bucket in enumerate(buckets, start=1):
                    title = get_bucket_title(bucket)
                    if is_blocked_bucket(title) or not is_allowed_bucket(title):
                        continue

                    print(f"   📦 Bucket {bi}/{len(buckets)}: {title}")
                    expand_bucket_if_needed(bucket)
                    
                    level, tourney_name = parse_level_and_name(title)
                    bucket_index = bi

                    expected_raw = expected_count_from_title(title)
                    found = 0
                    doubles_seen = 0
                    doubles_keys_seen = set()

                    added, dbls = process_hosts(bucket, title, doubles_keys_seen, level, tourney_name, bucket_index)
                    found += added
                    doubles_seen += dbls

                    # Retry once if we think there should be more singles
                    if expected_raw > 0:
                        expected_adjusted = max(expected_raw - min(doubles_seen, expected_raw), 0)
                    else:
                        expected_adjusted = -1
                    if expected_raw > 0 and found < expected_adjusted:
                        print(f"   ↩️  Only {found}/{expected_adjusted} (adj from {expected_raw}) for '{title}'. "
                              f"Retrying after bottom & in-coupon drains…")
                        drain_bottom_show_more(page, max_passes=2, pause_ms=160)
                        for cp in coupons_in_bucket(bucket):
                            drain_coupon_until_stable(page, cp)
                        expand_bucket_if_needed(bucket)
                        added, dbls = process_hosts(bucket, title, doubles_keys_seen, level, tourney_name, bucket_index)
                        found += added
                        doubles_seen += dbls
                        if expected_raw > 0:
                            expected_adjusted = max(expected_raw - min(doubles_seen, expected_raw), 0)
                        else:
                            expected_adjusted = -1

                    bucket_stats[title] = {
                        "expected_raw": max(expected_raw, 0),
                        "doubles_detected": doubles_seen,
                        "expected_adjusted": max(expected_adjusted, 0) if expected_adjusted >= 0 else 0,
                        "found_singles": found,
                    }

                    if expected_raw > 0 and found != expected_adjusted:
                        discrepancies[title] = {"expected": expected_adjusted, "found": found}

                # Snapshot
                try:
                    page.screenshot(path="bovada_tennis_debug.png", full_page=True)
                except Exception:
                    pass

                browser.close()

                # Summary with adjusted totals
                sum_expected_adjusted = sum(s["expected_adjusted"] for s in bucket_stats.values())
                print("\n📈 Summary:")
                print(f"   Buckets scanned: {header_total}")
                print(f"   Matches kept: {len(out_rows)}")
                print(f"   Expected RAW total: ~{sum_expected_raw}")
                print(f"   Adjusted Expected (singles): ~{sum_expected_adjusted}")

                if discrepancies:
                    print("\n⚠️ Buckets with mismatched counts (found vs adjusted expected):")
                    for t, d in discrepancies.items():
                        raw = bucket_stats[t]["expected_raw"]
                        dbl = bucket_stats[t]["doubles_detected"]
                        adj = bucket_stats[t]["expected_adjusted"]
                        print(f"   • {t}: found {d['found']}/{adj} (raw {raw}, minus doubles {dbl})")

                return pd.DataFrame(out_rows)

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(3)

    return pd.DataFrame()

# =========================
# Saving / CLI
# =========================

def save_odds_data(df: pd.DataFrame, output_dir: Optional[str] = None) -> str:
    if output_dir is None:
        output_dir = Path(__file__).parent / "data"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bovada_tennis_{timestamp}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    print(f"💾 Odds data saved to: {filepath}")

    latest_path = output_dir / "bovada_tennis_latest.csv"
    df.to_csv(latest_path, index=False)
    print(f"💾 Latest data saved to: {latest_path}")

    return str(filepath)

def main():
    print("🎾 Fetching ATP/Challenger tennis odds from Bovada...")
    try:
        df = fetch_bovada_tennis_odds(headless=True)
        if not df.empty:
            save_odds_data(df)
            print(f"\n✅ Success! {len(df)} matches saved")
            print("\n📋 Sample matches:")
            print(df[['event', 'player1_raw', 'player2_raw',
                      'player1_odds_decimal', 'player2_odds_decimal']].head())
        else:
            print("\n❌ No ATP/Challenger matches found")
    except Exception as e:
        print(f"\n💥 Error fetching odds: {e}")
        raise

if __name__ == "__main__":
    main()
