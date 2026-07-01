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


def main() -> int:
    print("=== Cloud scrape feasibility (datacenter IP) ===")
    ta_ok = test_tennis_abstract()
    bovada_ok = test_bovada()
    print()
    print(f"VERDICT: Tennis Abstract = {'OK' if ta_ok else 'BLOCKED'} | Bovada = {'OK' if bovada_ok else 'BLOCKED'}")
    if ta_ok and bovada_ok:
        print("Both sources served the cloud IP -> cloud scraping is viable (~free on Actions + Supabase).")
        return 0
    print("At least one source blocked the cloud IP -> would need a residential proxy or an alternative.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
