"""One-shot: backfill ITF M15/M25 results June 1 -> today into the store.

Closes the gap between Sackmann's last futures drop (2026-06-01) and the live
hourly ingest that starts today: without these weeks, ITF players' recent
histories are invisible (inflated Days_Since_Last, thin form windows, the
retro-settlement sweep has nothing to match against).
"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scraping"); sys.path.insert(0, "features")
from datetime import date, timedelta
from scraping.itf_results_scraper import ItfClient, get_calendar, get_event_matches
from canonical_store import connect, ingest_itf_results

start = date(2026, 6, 1)
end = date.today()
client = ItfClient()
total = {"inserted": 0, "skipped": 0, "created": 0}
try:
    cal = get_calendar(client, str(start), str(end), take=400)
    cal = cal[cal["category"].isin(["M15", "M25"])]
    print(f"events to backfill: {len(cal)}")
    with connect() as conn:
        for i, ev in cal.iterrows():
            try:
                em = get_event_matches(client, ev["key"])
                if em is None or em.empty:
                    continue
                r = ingest_itf_results(conn, em, event=str(ev["event"]),
                                       start_date=str(ev["start_date"]),
                                       surface=(str(ev.get("surface")) or None),
                                       level="25" if "25" in str(ev.get("category","")) else "15")
                for k in total: total[k] += r.get(k, 0)
            except Exception as e:
                print(f"  ⚠️ {ev['event']} failed: {e}")
                time.sleep(5)
finally:
    client.close()
print(f"BACKFILL DONE: {total}")
