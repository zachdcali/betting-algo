"""Backfill ITF M15/M25 results June 1 -> today into the store.

Incapsula kills long-lived sessions: rotate the browser client every few
events, back off between fetches, retry each event once on a fresh client.
"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scraping"); sys.path.insert(0, "features")
from datetime import date
from scraping.itf_results_scraper import ItfClient, get_calendar, get_event_matches
from canonical_store import connect, ingest_itf_results

ROTATE_EVERY = 8
start, end = date(2026, 6, 1), date.today()

client = ItfClient()
cal = get_calendar(client, str(start), str(end), take=400)
client.close()
cal = cal[cal["category"].isin(["M15", "M25"])].reset_index(drop=True)
print(f"events to backfill: {len(cal)}", flush=True)

total = {"inserted": 0, "skipped": 0, "created": 0}
client = None
since_rotate = 0
with connect() as conn:
    for i, ev in cal.iterrows():
        if client is None or since_rotate >= ROTATE_EVERY:
            if client: client.close()
            client = ItfClient(); since_rotate = 0
            time.sleep(2)
        ok = False
        for attempt in (1, 2):
            try:
                em = get_event_matches(client, ev["key"])
                ok = True
                break
            except Exception as e:
                print(f"  ⚠️ {ev['event']} attempt {attempt}: {str(e)[:60]}", flush=True)
                client.close(); client = ItfClient(); since_rotate = 0
                time.sleep(8)
        since_rotate += 1
        if not ok or em is None or em.empty:
            continue
        try:
            r = ingest_itf_results(conn, em, event=str(ev["event"]),
                                   start_date=str(ev["start_date"]),
                                   surface=(str(ev.get("surface")) or None),
                                   level="25" if "25" in str(ev.get("category", "")) else "15")
            for k in total: total[k] += r.get(k, 0)
        except Exception as e:
            print(f"  ⚠️ ingest {ev['event']}: {str(e)[:60]}", flush=True)
        time.sleep(1.5)
        if i % 20 == 0:
            print(f"  progress {i}/{len(cal)} — {total}", flush=True)
if client: client.close()
print(f"BACKFILL DONE: {total}", flush=True)
