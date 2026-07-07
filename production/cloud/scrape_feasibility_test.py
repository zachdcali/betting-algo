"""Cloud scrape feasibility probe.

The single make-or-break question for cloud: will Bovada (Cloudflare) and Tennis
Abstract serve a *datacenter* IP, or block it the way anti-bot systems often block
cloud ranges? This runs ONLY the scrapers (no models, no torch, no DB) so it's a
fast, cheap, isolated test — meant to run on a cloud runner (e.g. GitHub Actions)
before investing in the full cloud build.

Exit code 0 = both sources served us; 1 = at least one blocked (would need a proxy).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # put production/ on path


def test_tennis_abstract() -> bool:
    from scraping.ta_scraper import TennisAbstractScraper
    s = TennisAbstractScraper()
    try:
        df = s.get_player_matches(s.name_to_slug("Novak Djokovic"), years=[2024])
        rows = 0 if df is None else len(df)
        status = getattr(s, "last_http_status", None)
        ok = rows > 0
        print(f"[Tennis Abstract] http={status} rows={rows} -> {'OK' if ok else 'BLOCKED/EMPTY'}")
        return ok
    except Exception as e:  # noqa: BLE001 - we want any failure to read as blocked
        print(f"[Tennis Abstract] EXCEPTION {type(e).__name__}: {e} -> BLOCKED")
        return False


def test_bovada() -> bool:
    from odds.fetch_bovada import fetch_bovada_tennis_odds
    try:
        df = fetch_bovada_tennis_odds(headless=True)
        rows = 0 if df is None else len(df)
        ok = rows > 0
        print(f"[Bovada] rows={rows} -> {'OK' if ok else 'BLOCKED/EMPTY'}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[Bovada] EXCEPTION {type(e).__name__}: {e} -> BLOCKED")
        return False


def test_atptour() -> bool:
    """atptour.com via headless Chromium (rankings page = heaviest dependency).

    Since features-from-store (2026-07-08) the live pipeline needs atptour +
    Bovada + Supabase at predict time — TA is optional. This probe decides cloud
    viability."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            pg = b.new_page()
            pg.set_extra_http_headers({"User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")})
            pg.goto("https://www.atptour.com/en/rankings/singles?rankRange=0-5000",
                    wait_until="domcontentloaded", timeout=60000)
            try:
                pg.wait_for_function(
                    "document.querySelectorAll('tr.lower-row, table tbody tr').length > 50",
                    timeout=30000)
            except Exception:
                pass
            rows = pg.locator("tr.lower-row, table tbody tr").count()
            # second dependency shape: live-scores hub (event discovery)
            pg.goto("https://www.atptour.com/en/scores/current", wait_until="domcontentloaded", timeout=60000)
            html = pg.content()
            b.close()
        hub_ok = "/en/scores/current" in html and len(html) > 30000
        ok = rows > 50 and hub_ok
        print(f"[atptour.com] ranking rows={rows} scores-hub={'OK' if hub_ok else 'EMPTY/BLOCKED'} -> {'OK' if ok else 'BLOCKED'}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[atptour.com] EXCEPTION {type(e).__name__}: {e} -> BLOCKED")
        return False


def main() -> int:
    print("=== Cloud scrape feasibility (datacenter IP) ===")
    atp_ok = test_atptour()
    bovada_ok = test_bovada()
    ta_ok = test_tennis_abstract()
    print()
    print(f"VERDICT: atptour = {'OK' if atp_ok else 'BLOCKED'} | Bovada = {'OK' if bovada_ok else 'BLOCKED'} | "
          f"Tennis Abstract (optional since features-from-store) = {'OK' if ta_ok else 'BLOCKED'}")
    if atp_ok and bovada_ok:
        print("Required sources (atptour + Bovada) serve this cloud IP -> hourly cloud runs are viable "
              "(store mode needs no TA at predict time).")
        return 0
    print("A REQUIRED source blocked the cloud IP -> home box or proxy needed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
