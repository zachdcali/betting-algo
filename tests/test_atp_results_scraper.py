import gzip
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_parse_active_events_challenger_hub():
    from scraping.atp_results_scraper import parse_active_events
    events = parse_active_events(_load("atp_scores_challenger.html.gz"), level="C")
    assert len(events) >= 10  # 12 active challengers in the fixture week
    slugs = {e["slug"] for e in events}
    assert {"iasi", "bogota", "braunschweig", "trieste"} <= slugs
    ev = next(e for e in events if e["slug"] == "iasi")
    assert ev["url"].startswith("https://www.atptour.com/en/scores/current") and ev["url"].endswith("/results")
    assert ev["level"] == "C"
    assert ev["event"]  # display name extracted or slug-titled


def test_parse_event_draw_iasi():
    from scraping.atp_results_scraper import parse_event_draw
    draw = parse_event_draw(_load("atp_draw_iasi.html.gz"))
    assert len(draw) >= 20  # 31 slots minus qualifier/bye placeholders
    assert set(draw["round"]) <= {"R32", "R16", "QF", "SF", "F"}
    r32 = draw[draw["round"] == "R32"]
    blob = " ".join(r32["p1"]) + " " + " ".join(r32["p2"])
    assert "Royer" in blob and "Jianu" in blob  # top seed pairing present


def test_parse_official_main_draw_pdf_ignores_date_number_and_keeps_slot_20():
    from scraping.atp_results_scraper import parse_official_draw_pdf_text

    text = """
    Example Open
    20 July — 26 July 2026 | Clay
    Main Draw Singles
    1 1 PLAYER, Alpha USA
    2 Bye
    19 WAWRINKA, Stan SUI
    20 BURRUCHAGA, Roman … ARG
    21 MERIDA, Daniel ESP
    22 Qualifier
    31 Bye
    32 2 PLAYER, Omega GBR
    Round of 32 Round of 16 Quarterfinals Semifinals Final Winner
    """

    draw = parse_official_draw_pdf_text(text)

    assert len(draw) == 1
    assert draw.iloc[0].to_dict() == {
        "round": "R32",
        "p1": "Stan WAWRINKA",
        "p2": "Roman … BURRUCHAGA",
    }


def test_parse_official_order_of_play_infers_q1_from_qualifying_slots():
    from scraping.atp_results_scraper import parse_official_order_of_play_words

    words = [
        {"text": "William", "x0": 100, "x1": 135, "top": 100, "page": 0},
        {"text": "REJCHTMAN", "x0": 137, "x1": 195, "top": 100, "page": 0},
        {"text": "VINCIGUERRA", "x0": 197, "x1": 260, "top": 100, "page": 0},
        {"text": "(SWE)", "x0": 262, "x1": 290, "top": 100, "page": 0},
        {"text": "vs", "x0": 190, "x1": 202, "top": 115, "page": 0},
        {"text": "Louis", "x0": 130, "x1": 155, "top": 130, "page": 0},
        {"text": "WESSELS", "x0": 157, "x1": 210, "top": 130, "page": 0},
        {"text": "(GER)", "x0": 212, "x1": 240, "top": 130, "page": 0},
    ]
    qualifying_text = """
    Qualifying Singles
    23 REJCHTMAN VINCIGUERR… SWE
    24 10 WESSELS, Louis GER
    Qualifying Round 1 Qualifying Round 2 Qualifier
    """

    schedule = parse_official_order_of_play_words(
        words,
        qualifying_final_code="Q2",
        qualifying_text=qualifying_text,
    )

    assert schedule.to_dict("records") == [{
        "round": "Q1",
        "p1": "William REJCHTMAN VINCIGUERRA",
        "p2": "Louis WESSELS",
    }]


def test_parse_results_archive_card_scoped():
    from scraping.atp_results_scraper import parse_results_archive
    df = parse_results_archive(_load("atp_archive_tour_2026.html.gz"), 2026)
    assert len(df) >= 30  # 34 completed tour events in fixture
    assert not (df["event"].str.lower() == "facebook").any()  # the depth-6 bug
    halle = df[df["slug"] == "halle"].iloc[0]
    assert halle["event"] == "Terra Wortmann Open"
    assert halle["start_date"] == "2026-06-15"
    ao = df[df["slug"] == "australian-open"].iloc[0]
    assert ao["start_date"].startswith("2026-01")  # January, not June!
    assert df["start_date"].nunique() > 5  # dates vary per card, not one shared date


def test_quali_round_headers_map():
    from scraping.atp_results_scraper import _round_code_from_header
    assert _round_code_from_header("1st Round Qualifying - Court 2") == "Q1"
    assert _round_code_from_header("2nd Round Qualifying") == "Q2"
    assert _round_code_from_header("Final Round Qualifying - Clamex Court") == "Q2"
    assert _round_code_from_header("Round of 32 - Center") == "R32"


def test_parse_challenger_calendar_fixture():
    from scraping.atp_results_scraper import parse_challenger_calendar
    df = parse_challenger_calendar(_load("atp_chall_calendar.html.gz"))
    assert len(df) >= 100
    assert df["start_date"].nunique() > 10  # per-card dates, not one shared (Facebook-bug guard)
    # Historical cards must keep their immutable year-pinned result URL. A
    # current-challenger rewrite can point an old backlog lookup at a later
    # edition of the same recurring event.
    first = df.iloc[0]
    assert first["url"].endswith(
        f"/en/scores/archive/{first['slug']}/{first['id']}/2026/results"
    )
    assert df["url"].str.endswith("/results").all()
    week = df[(df.start_date >= "2026-07-06") & (df.start_date <= "2026-07-13")]
    assert {"iasi", "bogota", "braunschweig"} <= set(week["slug"])


def test_parse_tour_calendar_modern_event_card_keeps_surface_and_date():
    from scraping.atp_results_scraper import parse_tour_calendar

    html = """
    <ul class="events show">
      <li>
        <div>
          <a href="/en/tournaments/estoril/7290/overview">
            <div class="list-wrapper">
              Estoril, Portugal Millennium Estoril Open | 20 - 26 July, 2026
            </div>
          </a>
          <span>sgl 28 dbl 16 Clay Outdoor</span>
          <a href="/en/scores/current/estoril/7290/draws">Draws</a>
        </div>
      </li>
    </ul>
    """

    df = parse_tour_calendar(html)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["event"] == "Estoril"
    assert row["start_date"] == "2026-07-20"
    assert row["surface"] == "Clay"
    assert row["url"].endswith("/en/scores/current/estoril/7290/results")


def test_rendered_fetch_does_not_accept_large_navigation_shell(monkeypatch):
    from scraping import atp_results_scraper

    class FakePage:
        def __init__(self):
            self.attempt = 0
            self.closed = False

        def goto(self, *_args, **_kwargs):
            self.attempt = 1

        def reload(self, *_args, **_kwargs):
            self.attempt = 2

        def wait_for_selector(self, *_args, **_kwargs):
            if self.attempt == 1:
                raise RuntimeError("event cards not injected yet")

        def content(self):
            if self.attempt == 1:
                return "navigation" + ("x" * 60_000)
            return "<a href='/en/tournaments/estoril/7290/overview'>Estoril</a>"

        def close(self):
            self.closed = True

    page = FakePage()
    monkeypatch.setitem(
        sys.modules,
        "browser_session",
        SimpleNamespace(new_page=lambda: page),
    )
    monkeypatch.setattr(atp_results_scraper.time, "sleep", lambda _seconds: None)

    html = atp_results_scraper._fetch_rendered(
        "https://www.atptour.com/en/tournaments",
        "a[href$='/overview']",
        require_selector=True,
    )

    assert "Estoril" in html
    assert page.attempt == 2
    assert page.closed is True
