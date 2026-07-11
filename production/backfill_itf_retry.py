"""Targeted retry for events the main backfill lost to session churn.

Same ingest path (label-blind guard + per-event transactions = idempotent);
fresh client per event, longer waits — these events failed twice already.
"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scraping")
sys.path.insert(0, "features")
from datetime import date

from canonical_store import connect, ingest_itf_results
from scraping.itf_results_scraper import ItfClient, get_calendar, get_event_matches

RETRY = {"M15 Bergamo", "M15 Getxo", "M15 Kayseri", "M15 Kursumlijska Banja",
         "M15 Rosbach", "M25 Bakio", "M25 Klosters", "M25 Nivelles", "M25 Skopje"}

client = ItfClient()
cal = get_calendar(client, str(date(2026, 6, 1)), str(date.today()), take=400)
client.close()
cal = cal[cal["event"].isin(RETRY)].reset_index(drop=True)
print(f"retrying {len(cal)} events", flush=True)

total = {"inserted": 0, "skipped": 0, "created": 0}
with connect() as conn:
    for _, ev in cal.iterrows():
        em = None
        for attempt in (1, 2, 3):
            client = ItfClient()
            try:
                em = get_event_matches(client, ev["key"])
                break
            except Exception as e:
                print(f"  ⚠️ {ev['event']} attempt {attempt}: {str(e)[:80]}", flush=True)
                time.sleep(15)
            finally:
                client.close()
        if em is None or em.empty:
            print(f"  ❌ gave up: {ev['event']}", flush=True)
            continue
        with conn.transaction():
            r = ingest_itf_results(conn, em, event=str(ev["event"]),
                                   start_date=str(ev["start_date"]),
                                   surface=(str(ev.get("surface")) or None),
                                   level="25" if "25" in str(ev.get("category", "")) else "15")
        for k in total:
            total[k] += r.get(k, 0)
        print(f"  ✅ {ev['event']}: +{r['inserted']}", flush=True)
        time.sleep(3)
print(f"RETRY DONE: {total}", flush=True)
