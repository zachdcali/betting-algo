import gzip
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import auto_settle
from auto_settle import (
    _active_atp_events,
    _archive_atp_events,
    _atp_results_source_for,
    _candidate_atp_events,
    try_settle_from_atp,
)
from scraping.atp_results_scraper import (
    parse_challenger_calendar,
    parse_tournament_results,
)

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "atp"


def _event(
    name="Wimbledon",
    slug="wimbledon",
    event_id="540",
    start_date="2026-06-29",
    match_date="2026-07-01",
    url="https://www.atptour.com/en/scores/archive/wimbledon/540/2026/results",
    **overrides,
):
    event = {
        "event": name,
        "slug": slug,
        "id": event_id,
        "url": url,
        "start_date": start_date,
        "date_verified": True,
        "date_source": "test_official_calendar",
        "surface": "Grass",
        "static_binding": False,
        "match_date_bound": match_date,
        "selection_window_start": start_date,
        "selection_window_end": "2026-07-12",
    }
    event.update(overrides)
    return event


@pytest.fixture(scope="module")
def wimbledon_results():
    with gzip.open(FIXTURES / "atp_wimbledon_results.html.gz", "rt") as fh:
        return parse_tournament_results(fh.read())


def test_url_mapping_matches_only_explicit_or_exact_generic_label():
    ev = _atp_results_source_for("Men's Singles", "Grass", "2026-07-01")
    assert ev and ev["event_name"] == "Wimbledon"
    assert ev["url"] == (
        "https://www.atptour.com/en/scores/archive/wimbledon/540/2026/results"
    )
    assert "/current/" not in ev["url"]
    assert _atp_results_source_for("Wimbledon Men's Singles", "Grass", "2026-07-01")
    # Contradictory explicit text must not be swallowed by the generic alias.
    assert _atp_results_source_for(
        "French Open Men's Singles", "Grass", "2026-07-01"
    ) is None
    assert _atp_results_source_for("Men's Singles", "Clay", "2026-07-01") is None
    assert _atp_results_source_for("Men's Singles", "Grass", "2025-07-01") is None
    assert _atp_results_source_for("Men's Singles", "Grass", "2026-01-15") is None
    assert _atp_results_source_for("ITF Men Monastir", "Hard", "2026-07-01") is None
    assert _atp_results_source_for("Men's Singles", "", "2026-07-01") is None
    assert _atp_results_source_for("Men's Singles", "Grass", "") is None
    assert _atp_results_source_for("Men's Singles", "Grass", "not-a-date") is None


def test_settles_hurkacz_ofner_from_fixture_with_event_evidence(wimbledon_results):
    r = try_settle_from_atp(
        "Hubert Hurkacz", "Sebastian Ofner", "", wimbledon_results, _event()
    )
    assert r is not None and r["status"] == "matched_and_settled"
    assert r["actual_winner"] == 1
    assert "6-4" in r["score"]
    evidence = r["settlement_evidence"]
    assert evidence["source"] == "atp_results"
    assert evidence["event_instance"] == "540:2026-06-29"
    assert evidence["event_date_verified"] is True
    assert evidence["event_date_source"] == "test_official_calendar"
    assert evidence["start_date"] == "2026-06-29"
    assert evidence["date_verified"] is True
    assert evidence["date_source"] == "test_official_calendar"
    assert evidence["source_url"].endswith("/archive/wimbledon/540/2026/results")
    assert "/current/" not in evidence["source_url"]
    assert evidence["match_date_bound"] == "2026-07-01"
    assert evidence["identity_binding"] == "strict_normalized_full_name"
    assert r["ta_match_date_found"] == "2026-06-29"


def test_flipped_orientation(wimbledon_results):
    r = try_settle_from_atp(
        "Sebastian Ofner", "Hubert Hurkacz", "", wimbledon_results, _event()
    )
    assert r is not None and r["actual_winner"] == 2


def test_same_surname_players_use_strict_full_name_orientation():
    cards = pd.DataFrame([{
        "round": "R32",
        "p1": "Francisco Cerundolo",
        "p2": "Juan Manuel Cerundolo",
        "winner": 1,
        "p1_sets": "6 6",
        "p2_sets": "4 3",
    }])
    result = try_settle_from_atp(
        "Juan Manuel Cerundolo",
        "Francisco Cerundolo",
        "R32",
        cards,
        _event(name="Argentina Open", slug="buenos-aires", event_id="506"),
    )
    assert result is not None
    assert result["actual_winner"] == 2
    assert result["settlement_evidence"]["orientation_flipped"] is True
    assert try_settle_from_atp(
        "J. Cerundolo",
        "Francisco Cerundolo",
        "R32",
        cards,
        _event(name="Argentina Open", slug="buenos-aires", event_id="506"),
    ) is None


def test_unverified_or_incomplete_event_provenance_fails_closed(wimbledon_results):
    for broken in (
        _event(date_verified=False),
        _event(event_id=""),
        _event(start_date=""),
        _event(url=""),
        _event(match_date_bound=""),
    ):
        assert try_settle_from_atp(
            "Hubert Hurkacz", "Sebastian Ofner", "R64", wimbledon_results, broken
        ) is None
    assert try_settle_from_atp(
        "Hubert Hurkacz", "Sebastian Ofner", "R64", wimbledon_results, "Wimbledon"
    ) is None


def test_round_contradiction_blocks_settlement(wimbledon_results):
    assert try_settle_from_atp(
        "Hubert Hurkacz", "Sebastian Ofner", "F", wimbledon_results, _event()
    ) is None


def test_unknown_matchup_stays_pending(wimbledon_results):
    assert try_settle_from_atp(
        "Hubert Hurkacz", "Casper Ruud", "R64", wimbledon_results, _event()
    ) is None
    assert try_settle_from_atp(
        "Nobody Realman", "Also Fake", "", wimbledon_results, _event()
    ) is None


def test_active_events_preserve_official_calendar_metadata(monkeypatch):
    discovered = [_event(
        name="Bastad",
        slug="bastad",
        event_id="316",
        start_date="2026-07-13",
        match_date="2026-07-15",
        url="https://www.atptour.com/en/scores/current/bastad/316/results",
        date_source="tour_calendar",
        surface="Clay",
    )]
    monkeypatch.setattr(
        "features.history_stitch.get_active_events",
        lambda ref_date, cache: discovered,
    )
    monkeypatch.setattr(auto_settle, "_archive_atp_events", lambda date, cache: [])

    events = _active_atp_events({}, match_date="2026-07-15")
    bastad = next(event for event in events if event["event"] == "Bastad")
    assert bastad["id"] == "316"
    assert bastad["start_date"] == "2026-07-13"
    assert bastad["date_verified"] is True
    assert bastad["date_source"] == "tour_calendar"
    assert any(event["event"] == "Wimbledon" for event in events)


def test_candidate_event_selection_is_narrow_and_itf_skips_atp(monkeypatch):
    events = [
        _event(
            name="Bastad", slug="bastad", event_id="316",
            start_date="2026-07-13", match_date="2026-07-15",
            url="bastad-results", surface="Clay",
        ),
        _event(
            name="Gstaad", slug="gstaad", event_id="314",
            start_date="2026-07-13", match_date="2026-07-15",
            url="gstaad-results", surface="Clay",
        ),
    ]
    monkeypatch.setattr(
        auto_settle, "_active_atp_events", lambda cache, match_date="": events
    )

    selected = _candidate_atp_events(
        "Bastad", {}, surface="Clay", match_date="2026-07-15"
    )
    assert len(selected) == 1 and selected[0]["id"] == "316"
    assert _candidate_atp_events(
        "Bastad", {}, surface="Grass", match_date="2026-07-15"
    ) == []
    assert _candidate_atp_events(
        "Nordea Open", {}, surface="Clay", match_date="2026-07-15"
    ) == []
    assert _candidate_atp_events("", {}, match_date="2026-07-15") == []
    assert _candidate_atp_events(
        "ITF Men Villa Constitucion", {}, match_date="2026-07-15"
    ) == []


def test_generic_label_requires_explicit_surface_and_date_registry(monkeypatch):
    static_source = _atp_results_source_for(
        "Men's Singles", "Grass", "2026-07-01"
    )
    assert static_source is not None
    wimbledon = _event(
        url=static_source["url"],
        static_binding=True,
        date_source="static_registry",
    )
    mutable_current = _event(
        url="https://www.atptour.com/en/scores/current/wimbledon/540/results",
        static_binding=False,
        date_source="tour_calendar",
    )
    bastad = _event(
        name="Bastad", slug="bastad", event_id="316",
        start_date="2026-07-13", match_date="2026-07-15",
        url="bastad-results", surface="Clay",
    )
    monkeypatch.setattr(
        auto_settle,
        "_active_atp_events",
        lambda cache, match_date="": [wimbledon, mutable_current, bastad],
    )

    selected = _candidate_atp_events(
        "Men's Singles", {}, surface="Grass", match_date="2026-07-01"
    )
    assert len(selected) == 1 and selected[0]["id"] == "540"
    assert selected[0]["url"] == static_source["url"]
    assert "/archive/wimbledon/540/2026/results" in selected[0]["url"]
    assert "/current/" not in selected[0]["url"]
    assert selected[0]["match_date_bound"] == "2026-07-01"
    assert _candidate_atp_events(
        "Men's Singles", {}, surface="Clay", match_date="2026-07-01"
    ) == []
    assert _candidate_atp_events(
        "Wimbledon", {}, surface="Clay", match_date="2026-07-01"
    ) == []
    assert _candidate_atp_events(
        "Wimbledon", {}, surface="Grass", match_date="2026-01-01"
    ) == []
    assert _candidate_atp_events("Wimbledon", {}) == []


def test_consecutive_week_event_instances_are_date_bound(monkeypatch):
    first = _event(
        name="Oeiras Open", slug="oeiras", event_id="2831",
        start_date="2026-01-19", match_date="2026-01-24",
        url="https://www.atptour.com/en/scores/archive/oeiras/2831/2026/results",
        surface="Hard", selection_window_end="2026-01-25",
    )
    second = _event(
        name="Oeiras Open", slug="oeiras", event_id="2833",
        start_date="2026-01-26", match_date="2026-01-27",
        url="https://www.atptour.com/en/scores/archive/oeiras/2833/2026/results",
        surface="Hard", selection_window_end="2026-02-08",
    )
    monkeypatch.setattr(
        auto_settle,
        "_active_atp_events",
        lambda cache, match_date="": [first, second],
    )

    week_one = _candidate_atp_events(
        "Oeiras Open", {}, surface="Hard", match_date="2026-01-24"
    )
    week_two = _candidate_atp_events(
        "Oeiras Open", {}, surface="Hard", match_date="2026-01-27"
    )
    assert [event["id"] for event in week_one] == ["2831"]
    assert [event["id"] for event in week_two] == ["2833"]


def test_old_challenger_backlog_keeps_fixture_archive_instance(monkeypatch):
    with gzip.open(FIXTURES / "atp_chall_calendar.html.gz", "rt") as fh:
        calendar = parse_challenger_calendar(fh.read())
    rows = calendar[calendar["id"].astype(str).isin({"2831", "2833"})]
    events = []
    for _, row in rows.iterrows():
        events.append({
            **row.to_dict(),
            "date_verified": True,
            "date_source": "challenger_calendar",
            "static_binding": False,
        })
    monkeypatch.setattr(
        auto_settle, "_active_atp_events", lambda cache, match_date="": events
    )

    selected = _candidate_atp_events(
        "Oeiras Open", {}, surface="Hard", match_date="2026-01-24"
    )
    assert len(selected) == 1
    assert selected[0]["id"] == "2831"
    assert selected[0]["url"].endswith("/archive/oeiras/2831/2026/results")


def test_two_date_compatible_event_instances_fail_closed(monkeypatch):
    events = [
        _event(
            name="Bastad", slug="bastad", event_id="316",
            start_date="2026-07-13", match_date="2026-07-15",
            url="tour-results", surface="Clay",
        ),
        _event(
            name="Bastad Challenger", slug="bastad-challenger", event_id="9316",
            start_date="2026-07-13", match_date="2026-07-15",
            url="challenger-results", surface="Clay",
        ),
    ]
    monkeypatch.setattr(
        auto_settle, "_active_atp_events", lambda cache, match_date="": events
    )
    assert _candidate_atp_events(
        "Bastad", {}, surface="Clay", match_date="2026-07-15"
    ) == []


def test_qualifying_only_gets_bounded_prestart_window(monkeypatch):
    miami = _event(
        name="Miami Open", slug="miami", event_id="403",
        start_date="2026-03-18", match_date="2026-03-18",
        url="miami-results", surface="Hard",
    )
    monkeypatch.setattr(
        auto_settle, "_active_atp_events", lambda cache, match_date="": [miami]
    )

    assert _candidate_atp_events(
        "Miami Open", {}, surface="Hard", match_date="2026-03-15", round_code="Q1"
    )
    assert _candidate_atp_events(
        "Miami Open", {}, surface="Hard", match_date="2026-03-15", round_code="R128"
    ) == []
    assert _candidate_atp_events(
        "Miami Open", {}, surface="Hard", match_date="2026-03-14", round_code="Q1"
    ) == []


def test_old_tour_backlog_selects_year_pinned_archive(monkeypatch):
    fixture_html = gzip.open(
        FIXTURES / "atp_archive_tour_2026.html.gz", "rt"
    ).read()
    monkeypatch.setattr(
        "scraping.atp_results_scraper._fetch_rendered",
        lambda url, selector: fixture_html,
    )
    monkeypatch.setattr(
        "features.history_stitch.get_active_events", lambda ref_date, cache: []
    )

    selected = _candidate_atp_events(
        "Terra Wortmann Open", {}, surface="Grass", match_date="2026-06-18"
    )
    assert len(selected) == 1
    assert selected[0]["id"] == "500"
    assert selected[0]["start_date"] == "2026-06-15"
    assert selected[0]["url"].endswith("/archive/halle/500/2026/results")


def test_archive_discovery_covers_prior_year_at_january_boundary(monkeypatch):
    requested = []
    monkeypatch.setattr(
        "scraping.atp_results_scraper._fetch_rendered",
        lambda url, selector: requested.append(url) or "<html></html>",
    )
    _archive_atp_events(pd.Timestamp("2026-01-05"), {})
    assert any("year=2025" in url for url in requested)
    assert any("year=2026" in url for url in requested)
