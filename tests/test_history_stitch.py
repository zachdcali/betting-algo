import gzip
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pandas as pd
import pytest
from features.history_stitch import (
    activity_to_ta_schema,
    active_event_for,
    event_results_to_ta_schema,
    needs_stitching,
    stitch_history,
)
from scraping.atp_results_scraper import parse_player_activity, parse_tournament_results

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "atp"


def _load(name):
    with gzip.open(FIXTURES / name, "rt") as fh:
        return fh.read()


def _ta_history():
    return pd.DataFrame([
        {"date": pd.Timestamp("2026-06-15"), "event": "Halle", "surface": "Grass",
         "round": "F", "level": "A", "rank": 4.0, "opp_name": "Taylor Fritz",
         "opp_rank": 5.0, "opp_hand": "R", "opp_country": "USA",
         "score": "6-4 6-4", "result": "W"},
        {"date": pd.Timestamp("2026-05-25"), "event": "Roland Garros", "surface": "Clay",
         "round": "R64", "level": "G", "rank": 4.0, "opp_name": "Juan Manuel Cerundolo",
         "opp_rank": 56.0, "opp_hand": "R", "opp_country": "ARG",
         "score": "7-5 6-1 6-1", "result": "L"},
    ])


# --- parity: stitching must be a no-op when there is nothing to stitch ---

def test_parity_no_atp_rows_is_identity():
    ta = _ta_history()
    out = stitch_history(ta, pd.DataFrame())
    assert len(out) == len(ta)
    # identity path must not reorder or alter the TA frame in any way
    pd.testing.assert_frame_equal(out.drop(columns=["source"]), ta)
    assert (out["source"] == "ta").all()


def test_parity_overlapping_rows_prefer_ta():
    ta = _ta_history()
    # ATP offers the same Halle final (abbreviated name, same date window) -> must NOT duplicate
    atp = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-15"), "event": "ATP Halle", "surface": "Grass",
        "round": "F", "level": "", "rank": float("nan"), "opp_name": "T. Fritz",
        "opp_rank": float("nan"), "opp_hand": "", "opp_country": "",
        "score": "6-4 6-4", "result": "W", "source": "atp_activity",
    }])
    out = stitch_history(ta, atp)
    assert len(out) == len(ta)
    assert (out["source"] == "ta").all()


def test_new_rows_are_appended_with_provenance():
    ta = _ta_history()
    atp = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-29"), "event": "Wimbledon", "surface": "Grass",
        "round": "R128", "level": "G", "rank": float("nan"), "opp_name": "Luca Nardi",
        "opp_rank": float("nan"), "opp_hand": "", "opp_country": "",
        "score": "6-3 6-3 6-3", "result": "W", "source": "atp_event_results",
    }])
    out = stitch_history(ta, atp)
    assert len(out) == 3
    newest = out.sort_values("date").iloc[-1]
    assert newest["source"] == "atp_event_results"
    assert newest["opp_name"] == "Luca Nardi"


# --- schema conversion from real fixtures ---

def test_activity_fixture_converts_with_ta_conventions():
    act = parse_player_activity(_load("atp_activity_sinner.html.gz"))
    conv = activity_to_ta_schema(act)
    assert not conv.empty
    rg = conv[conv["event"].str.contains("Roland Garros")]
    r64 = rg[rg["round"] == "R64"].iloc[0]
    assert r64["result"] == "L"
    # score flipped to winner-perspective (Sinner's own view '6-3 6-2 5-7 1-6 1-6'
    # becomes Cerundolo-first '3-6 2-6 7-5 6-1 6-1', TA convention)
    assert r64["score"] == "3-6 2-6 7-5 6-1 6-1"
    assert r64["level"] == "G"
    assert r64["date"] == pd.Timestamp("2026-05-25")
    assert (conv["source"] == "atp_activity").all()


def test_event_results_fixture_extracts_player_rows():
    res = parse_tournament_results(_load("atp_wimbledon_results.html.gz"))
    ev = {"event": "Wimbledon", "start_date": "2026-06-29", "surface": "Grass", "level": "G"}
    hurk = event_results_to_ta_schema(res, "Hubert Hurkacz", ev)
    assert len(hurk) >= 2  # R128 + R64 completed in fixture
    r64 = hurk[hurk["round"] == "R64"].iloc[0]
    assert r64["result"] == "W" and "Ofner" in r64["opp_name"]
    assert r64["score"] == "7-6(8) 6-4 6-4"
    assert r64["date"] == pd.Timestamp("2026-06-29")


# --- gating helpers ---

def test_needs_stitching_thresholds():
    ta = _ta_history()  # newest row 2026-06-15
    assert needs_stitching(ta, "2026-07-05") is True    # 20 days stale
    assert needs_stitching(ta, "2026-06-20") is False   # 5 days: could be rest
    assert needs_stitching(pd.DataFrame(), "2026-07-05") is True


def test_active_event_window():
    assert active_event_for("2026-07-05")["event"] == "Wimbledon"
    assert active_event_for("2026-08-20") is None


def test_infer_next_round():
    from features.history_stitch import infer_next_round
    m1 = pd.DataFrame([{"event": "Wimbledon", "round": "R32", "date": pd.Timestamp("2026-06-29")}])
    m2 = pd.DataFrame([{"event": "Wimbledon", "round": "R32", "date": pd.Timestamp("2026-06-29")}])
    assert infer_next_round(m1, m2, "Wimbledon") == "R16"
    # mismatched rounds -> None (conservative)
    m3 = pd.DataFrame([{"event": "Wimbledon", "round": "R64", "date": pd.Timestamp("2026-06-29")}])
    assert infer_next_round(m1, m3, "Wimbledon") is None
    # different event on top -> None
    m4 = pd.DataFrame([{"event": "Halle", "round": "F", "date": pd.Timestamp("2026-06-15")}])
    assert infer_next_round(m1, m4, "Wimbledon") is None
    # final has no next round
    mf = pd.DataFrame([{"event": "Wimbledon", "round": "F", "date": pd.Timestamp("2026-06-29")}])
    assert infer_next_round(mf, mf, "Wimbledon") is None


def test_enrich_event_rows_with_stats_fake_fetch():
    from features.history_stitch import enrich_event_rows_with_stats
    rows = pd.DataFrame([{
        "date": pd.Timestamp("2026-06-29"), "event": "Wimbledon", "round": "R128",
        "opp_name": "Shintaro Mochizuki", "result": "W", "score": "6-2 6-2 6-2",
        "source": "atp_event_results", "_stats_url": "/en/scores/match-stats/x",
    }])
    fake = lambda url: {"p1_name": "J. Sinner", "p2_name": "S. Mochizuki",
                        "p1": {"aces": 15, "serve_points": 85, "opp_serve_points": 116},
                        "p2": {"aces": 4, "serve_points": 116, "opp_serve_points": 85}}
    out = enrich_event_rows_with_stats(rows, "Jannik Sinner", {}, _fetch=fake)
    assert out.iloc[0]["aces"] == 15.0
    assert out.iloc[0]["serve_points"] == 85.0
    assert out.iloc[0]["opp_serve_points"] == 116.0
    # unmatched player -> untouched
    out2 = enrich_event_rows_with_stats(rows, "Someone Else", {}, _fetch=fake)
    assert pd.isna(out2.iloc[0]["aces"])


def test_infer_next_round_any_registry_free():
    from features.history_stitch import infer_next_round_any
    ref = pd.Timestamp("2026-07-08")
    m1 = pd.DataFrame([{"event": "Concord Iasi Open", "round": "R32", "date": pd.Timestamp("2026-07-06")}])
    m2 = pd.DataFrame([{"event": "Concord Iasi Open", "round": "R32", "date": pd.Timestamp("2026-07-06")}])
    assert infer_next_round_any(m1, m2, ref) == "R16"
    # different events on top -> None
    m3 = pd.DataFrame([{"event": "Trieste", "round": "R32", "date": pd.Timestamp("2026-07-06")}])
    assert infer_next_round_any(m1, m3, ref) is None
    # stale top rows (>16d old) -> None
    old = pd.DataFrame([{"event": "Concord Iasi Open", "round": "R32", "date": pd.Timestamp("2026-06-01")}])
    assert infer_next_round_any(old, old, ref) is None


def test_names_match_hyphen_variants():
    from features.history_stitch import _names_loosely_match
    # TA strips hyphens from display names; ATP keeps them
    assert _names_loosely_match("F. Auger-Aliassime", "Felix Auger Aliassime")
    assert _names_loosely_match("Felix Auger Aliassime", "F. Auger-Aliassime")
    assert _names_loosely_match("J. Struff", "Jan Lennard Struff")
    assert _names_loosely_match("Christopher O'Connell", "Christopher Oconnell")
    assert not _names_loosely_match("Novak Djokovic", "Felix Auger-Aliassime")


def test_names_match_truncated_double_surname():
    from features.history_stitch import _names_loosely_match
    assert _names_loosely_match("Diego Dedura", "Diego Dedura-Palomero")
    assert _names_loosely_match("Diego Dedura-Palomero", "Diego Dedura")
    assert not _names_loosely_match("Alex de Minaur", "Alex Michelsen")
    assert not _names_loosely_match("Jan Choinski", "Jan-Lennard Struff")
