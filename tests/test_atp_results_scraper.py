import gzip
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pytest
from scraping.atp_results_scraper import parse_player_activity, parse_tournament_results

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "atp"


def _load(name: str) -> str:
    with gzip.open(FIXTURES / name, "rt") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def results_df():
    return parse_tournament_results(_load("atp_wimbledon_results.html.gz"))


@pytest.fixture(scope="module")
def activity_df():
    return parse_player_activity(_load("atp_activity_sinner.html.gz"))


def test_results_page_yields_many_matches(results_df):
    # Wimbledon fixture captured mid-event: 200+ completed matches
    assert len(results_df) >= 150
    assert set(["round", "p1", "p2", "winner", "p1_sets", "p2_sets"]) <= set(results_df.columns)


def test_results_hurkacz_ofner_r64(results_df):
    m = results_df[(results_df.p1.str.contains("Hurkacz")) & (results_df.p2.str.contains("Ofner"))]
    assert len(m) == 1, "expected exactly one Hurkacz-Ofner card"
    row = m.iloc[0]
    assert row["round"] == "R64"
    assert row["winner"] == 1  # Hurkacz won 7-6(8) 6-4 6-4


def test_results_winner_marker_coverage(results_df):
    # the winner marker should resolve for the overwhelming majority of matches
    resolved = results_df["winner"].notna().mean()
    assert resolved > 0.9, f"winner resolved for only {resolved:.0%} of matches"


def test_activity_tournaments_parsed(activity_df):
    assert activity_df["event"].nunique() >= 5
    rg = activity_df[activity_df.event.str.contains("Roland Garros")]
    assert not rg.empty
    assert (rg["surface"] == "Clay").all()
    assert (rg["start_date"] == "25 May, 26").all()


def test_activity_sinner_rg_rows(activity_df):
    rg = activity_df[activity_df.event.str.contains("Roland Garros")]
    r64 = rg[rg["round"] == "R64"]
    assert len(r64) == 1
    assert "Cerundolo" in r64.iloc[0]["opponent"]
    assert r64.iloc[0]["result"] == "L"  # Sinner lost 6-3 6-2 5-7 1-6 1-6
    r128 = rg[rg["round"] == "R128"]
    assert len(r128) == 1
    assert "Tabur" in r128.iloc[0]["opponent"]
    assert r128.iloc[0]["result"] == "W"


def test_parse_match_stats_sinner_fixture():
    from scraping.atp_results_scraper import parse_match_stats
    stats = parse_match_stats(_load("atp_match_stats_sinner.html.gz"))
    assert stats is not None
    assert "Sinner" in stats["p1_name"] and "Mochizuki" in stats["p2_name"]
    p1, p2 = stats["p1"], stats["p2"]
    assert p1["aces"] == 15 and p1["double_faults"] == 3
    assert p1["serve_points"] == 85 and p1["first_serves_in"] == 63
    assert p1["first_serve_won"] == 52 and p1["second_serve_won"] == 12
    assert p1["bp_saved"] == 5 and p1["bp_faced"] == 5
    # opponent mirrors: Sinner's opp_* = Mochizuki's own serve stats
    assert p1["opp_serve_points"] == 116 and p1["opp_first_serves_in"] == 70
    assert p2["opp_serve_points"] == 85
