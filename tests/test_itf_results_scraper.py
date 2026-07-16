import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from scraping.itf_results_scraper import (
    get_player_profiles_resilient,
    parse_calendar,
    parse_oop_matches,
    parse_player_profile,
    profile_refs_for_names,
)

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
    assert r["p1_profile_url"].startswith("/en/players/")
    assert r["p2_profile_url"].startswith("/en/players/")
    assert len(r["p1_nationality"]) == 3


def test_round_mapping_quali_vs_main():
    from scraping.itf_results_scraper import _round_code
    assert _round_code({"eventClassificationDesc": "Qualifying Draw", "roundGroupDesc": "1st Round", "roundCode": "32"}) == "Q1"
    assert _round_code({"eventClassificationDesc": "Main Draw", "roundGroupDesc": "1st Round", "roundCode": "32"}) == "R32"
    assert _round_code({"eventClassificationDesc": "Main Draw", "roundGroupDesc": "Semifinals", "roundCode": ""}) == "SF"


def test_profile_ref_and_profile_body_require_same_name_and_itf_id():
    matches = parse_oop_matches(_load("itf_oop_day.json.gz"), "2026-07-05")
    first = matches.iloc[0]
    refs = profile_refs_for_names(
        {"event": matches},
        [first["p1"], "Not On This Order Of Play"],
    )
    ref = refs[first["p1"]]
    assert ref["itf_player_id"] == str(first["p1_id"])
    assert ref["profile_url"] == first["p1_profile_url"]
    assert "Not On This Order Of Play" not in refs

    html = f"""
      <html><head>
        <meta name="keywords" content="{first['p1']}" />
        <link rel="canonical"
          href="https://www.itftennis.com{first['p1_profile_url']}overview/" />
      </head><body>
        <span id="ga__player-plays-hand" class="player-hero__value">Right Handed</span>
      </body></html>
    """
    parsed = parse_player_profile(
        html,
        expected_name=first["p1"],
        expected_player_id=first["p1_id"],
    )
    assert parsed["status"] == "resolved"
    assert parsed["hand"] == "R"
    assert parsed["height_cm"] is None

    wrong = parse_player_profile(
        html,
        expected_name=first["p1"],
        expected_player_id="800000000",
    )
    assert wrong["status"] == "identity_mismatch"
    assert wrong["hand"] is None

    blocked = parse_player_profile(
        "<html><head><title>Request unsuccessful</title></head><body>"
        "Incapsula incident ID</body></html>",
        expected_name=first["p1"],
        expected_player_id=first["p1_id"],
    )
    assert blocked["status"] == "fetch_error"
    assert blocked["hand"] is None


def test_profile_batch_rotates_sessions_and_retries_only_transient_blocks():
    refs = {
        "Alpha Player": {
            "itf_player_id": "800000001",
            "profile_url": "/en/players/alpha-player/800000001/usa/mt/s/",
        },
        "Beta Player": {
            "itf_player_id": "800000002",
            "profile_url": "/en/players/beta-player/800000002/gbr/mt/s/",
        },
        "Conflict Player": {
            "itf_player_id": "800000003",
            "profile_url": "/en/players/conflict-player/800000003/fra/mt/s/",
        },
    }
    clients = []

    def _profile_html(name, player_id, hand="Right"):
        slug = name.casefold().replace(" ", "-")
        return f"""
          <meta name="keywords" content="{name}" />
          <link rel="canonical" href="https://www.itftennis.com/en/players/{slug}/{player_id}/usa/mt/s/overview/" />
          <span id="ga__player-plays-hand">{hand} Handed</span>
        """

    class _Client:
        def __init__(self):
            self.number = len(clients) + 1
            self.closed = False
            clients.append(self)

        def fetch_text(self, url):
            if "beta-player" in url and self.number == 1:
                return "<html><body>Incapsula incident ID</body></html>"
            if "conflict-player" in url:
                return _profile_html("Different Player", "800009999")
            if "alpha-player" in url:
                return _profile_html("Alpha Player", "800000001")
            return _profile_html("Beta Player", "800000002", hand="Left")

        def close(self):
            self.closed = True

    results = get_player_profiles_resilient(
        refs,
        batch_size=2,
        max_attempts=2,
        client_factory=_Client,
    )

    assert len(clients) == 3
    assert all(client.closed for client in clients)
    assert results["Alpha Player"]["status"] == "resolved"
    assert results["Alpha Player"]["attempt_count"] == 1
    assert results["Beta Player"]["status"] == "resolved"
    assert results["Beta Player"]["hand"] == "L"
    assert results["Beta Player"]["attempt_count"] == 2
    assert results["Conflict Player"]["status"] == "identity_mismatch"
    assert results["Conflict Player"]["attempt_count"] == 1


def test_gather_itf_rows_and_round_with_fixtures(monkeypatch):
    """Integration shape: Bovada label -> event -> player rows + upcoming round."""
    import pandas as pd
    from features import history_stitch as hs
    cal = parse_calendar(_load("itf_calendar.json.gz"))
    ev = cal.iloc[0].to_dict()
    em = __import__("scraping.itf_results_scraper", fromlist=["parse_oop_matches"]).parse_oop_matches(
        _load("itf_oop_day.json.gz"), "2026-07-05")
    cache = {"itf_calendar": cal, "itf_event_matches": {ev["key"]: em}}
    monkeypatch.setattr(hs, "_itf_event_for", lambda label, ref, c, players=None: ev)
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
