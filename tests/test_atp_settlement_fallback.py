import gzip
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pytest
from auto_settle import _atp_results_source_for, try_settle_from_atp
from scraping.atp_results_scraper import parse_tournament_results

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "atp"


@pytest.fixture(scope="module")
def wimbledon_results():
    with gzip.open(FIXTURES / "atp_wimbledon_results.html.gz", "rt") as fh:
        return parse_tournament_results(fh.read())


def test_url_mapping_matches_bovada_label():
    ev = _atp_results_source_for("Men's Singles", "Grass", "2026-07-01")
    assert ev and ev["event_name"] == "Wimbledon"
    # wrong surface or month must not map
    assert _atp_results_source_for("Men's Singles", "Clay", "2026-07-01") is None
    assert _atp_results_source_for("Men's Singles", "Grass", "2026-01-15") is None
    assert _atp_results_source_for("ITF Men Monastir", "Hard", "2026-07-01") is None


def test_settles_hurkacz_ofner_from_fixture(wimbledon_results):
    r = try_settle_from_atp("Hubert Hurkacz", "Sebastian Ofner", "", wimbledon_results, "Wimbledon")
    assert r is not None and r["status"] == "matched_and_settled"
    assert r["actual_winner"] == 1  # Hurkacz won 7-6(8) 6-4 6-4
    assert r["settlement_evidence"]["source"] == "atp_results"
    assert "6-4" in r["score"]


def test_flipped_orientation(wimbledon_results):
    r = try_settle_from_atp("Sebastian Ofner", "Hubert Hurkacz", "", wimbledon_results, "Wimbledon")
    assert r is not None and r["actual_winner"] == 2  # our p1=Ofner lost


def test_round_contradiction_blocks_settlement(wimbledon_results):
    r = try_settle_from_atp("Hubert Hurkacz", "Sebastian Ofner", "F", wimbledon_results, "Wimbledon")
    assert r is None  # card is R64; a claimed Final must not settle


def test_unknown_matchup_stays_pending(wimbledon_results):
    r = try_settle_from_atp("Hubert Hurkacz", "Casper Ruud", "R64", wimbledon_results, "Wimbledon")
    # Hurkacz played Ruud in R128, not R64 — with the wrong round this must not settle
    r2 = try_settle_from_atp("Nobody Realman", "Also Fake", "", wimbledon_results, "Wimbledon")
    assert r2 is None
