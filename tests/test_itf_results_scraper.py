import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from scraping.itf_results_scraper import parse_calendar, parse_oop_matches

FIX = REPO_ROOT / "tests" / "fixtures" / "itf"


def _load(name):
    with gzip.open(FIX / name, "rt") as fh:
        return json.load(fh)


def test_parse_calendar():
    df = parse_calendar(_load("itf_calendar.json.gz"))
    assert len(df) >= 20
    assert (df["key"].str.match(r"^[mw]-itf-[a-z]{3}-\d{4}-\d+$")).all()
    assert df["surface"].str.len().gt(0).any()
    assert (df["category"].isin(["M15", "M25"])).any()


def test_parse_oop_completed_matches():
    df = parse_oop_matches(_load("itf_oop_day.json.gz"), "2026-07-05")
    assert len(df) >= 4
    done = df[df.completed]
    assert len(done) >= 1
    r = done.iloc[0]
    assert r["winner"] in (1, 2)
    assert "-" in r["score"]
    assert r["round"] in {"Q1", "Q2", "Q3", "R128", "R64", "R32", "R16", "QF", "SF", "F"}
    assert r["p1_id"] and r["p2_id"]


def test_round_mapping_quali_vs_main():
    from scraping.itf_results_scraper import _round_code
    assert _round_code({"eventClassificationDesc": "Qualifying Draw", "roundGroupDesc": "1st Round", "roundCode": "32"}) == "Q1"
    assert _round_code({"eventClassificationDesc": "Main Draw", "roundGroupDesc": "1st Round", "roundCode": "32"}) == "R32"
    assert _round_code({"eventClassificationDesc": "Main Draw", "roundGroupDesc": "Semifinals", "roundCode": ""}) == "SF"


def test_gather_itf_rows_and_round_with_fixtures(monkeypatch):
    """Integration shape: Bovada label -> event -> player rows + upcoming round."""
    import pandas as pd
    from features import history_stitch as hs
    cal = parse_calendar(_load("itf_calendar.json.gz"))
    ev = cal.iloc[0].to_dict()
    em = __import__("scraping.itf_results_scraper", fromlist=["parse_oop_matches"]).parse_oop_matches(
        _load("itf_oop_day.json.gz"), "2026-07-05")
    cache = {"itf_calendar": cal, "itf_event_matches": {ev["key"]: em}}
    monkeypatch.setattr(hs, "_itf_event_for", lambda label, ref, c: ev)
    done = em[em.completed]
    player = done.iloc[0]["p1"] if done.iloc[0]["winner"] == 1 else done.iloc[0]["p2"]
    rows = hs.gather_itf_rows(player, "ITF Men Test", "2026-07-08", cache)
    assert len(rows) >= 1
    r = rows.iloc[0]
    assert r["source"] == "itf_results" and r["result"] in ("W", "L") and r["level"] in ("15", "25")
    assert r["round"].startswith(("Q", "R", "S", "F"))
    # upcoming round resolution for the same completed pair pins the round
    rc = hs.itf_round_for(done.iloc[0]["p1"], done.iloc[0]["p2"], "ITF Men Test", "2026-07-08", cache)
    assert rc == done.iloc[0]["round"]
