import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from scraping.atp_rankings_scraper import (
    get_player_lookup_status,
    get_player_points,
    get_player_rank,
    get_player_url,
)


def _abbreviated_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 18, "player_name": "L. Darderi", "points": 2210,
                "player_url": "/en/players/luciano-darderi/d0fj/overview",
            },
            {
                "rank": 43, "player_name": "M. Berrettini", "points": 1075,
                "player_url": "/en/players/matteo-berrettini/bk40/overview",
            },
            {
                "rank": 109, "player_name": "Y. Bu", "points": 604,
                "player_url": "/en/players/yunchaokete-bu/y09v/overview",
            },
            {
                "rank": 257, "player_name": "D. Dedura", "points": 219,
                "player_url": "/en/players/diego-dedura/d0lj/overview",
            },
            {
                "rank": 434, "player_name": "A. Boitan", "points": 108,
                "player_url": "/en/players/adrian-boitan/b0c0/overview",
            },
            {
                "rank": 546, "player_name": "J. Berrettini", "points": 74,
                "player_url": "/en/players/jacopo-berrettini/b0g9/overview",
            },
            {
                "rank": 969, "player_name": "Micah Braswell", "points": 24,
                "player_url": "/en/players/micah-braswell/b0j8/overview",
            },
            {
                "rank": 1125, "player_name": "Seongwoo Cho", "points": 28,
                "player_url": "/en/players/seongwoo-cho/c0nx/overview",
            },
        ]
    )


def test_unique_surname_cannot_override_mismatched_initial_darderi_regression():
    rankings = _abbreviated_rankings()

    assert get_player_rank("Vito Antonio Darderi", rankings) is None
    assert get_player_points("Vito Antonio Darderi", rankings) is None
    assert get_player_rank("Luciano Darderi", rankings) == 18
    assert get_player_points("Luciano Darderi", rankings) == 2210


def test_all_observed_live_given_name_collisions_fail_closed():
    rankings = _abbreviated_rankings()

    for mismatched_name in (
        "Gabi Adrian Boitan",
        "Jonah Braswell",
        "Min Hyuk Cho",
        "Vito Antonio Darderi",
    ):
        assert get_player_rank(mismatched_name, rankings) is None
        assert get_player_points(mismatched_name, rankings) is None

    # The actual ranked identities remain available, including ATP's mix of
    # abbreviated and full ranking-row names.
    assert get_player_rank("Adrian Boitan", rankings) == 434
    assert get_player_rank("Micah Braswell", rankings) == 969
    assert get_player_rank("Seongwoo Cho", rankings) == 1125
    assert get_player_rank("Luciano Darderi", rankings) == 18


def test_unique_surname_cannot_override_mismatched_full_given_name():
    rankings = pd.DataFrame(
        [{"rank": 18, "player_name": "Luciano Darderi", "points": 2210}]
    )

    assert get_player_rank("Vito Darderi", rankings) is None
    assert get_player_rank("Luca Darderi", rankings) is None
    assert get_player_rank("Luciano Darderi", rankings) == 18


def test_legitimate_abbreviated_siblings_still_resolve_by_initial():
    rankings = _abbreviated_rankings()

    assert get_player_rank("Matteo Berrettini", rankings) == 43
    assert get_player_points("Matteo Berrettini", rankings) == 1075
    assert get_player_rank("Jacopo Berrettini", rankings) == 546
    assert get_player_points("Jacopo Berrettini", rankings) == 74


def test_abbreviated_query_can_bind_to_full_candidate_with_matching_initial():
    rankings = pd.DataFrame(
        [
            {"rank": 43, "player_name": "Matteo Berrettini", "points": 1075},
            {"rank": 546, "player_name": "Jacopo Berrettini", "points": 74},
        ]
    )

    assert get_player_rank("M. Berrettini", rankings) == 43
    assert get_player_rank("J. Berrettini", rankings) == 546


def test_supported_multi_surname_and_reversed_names_remain_resolvable():
    rankings = _abbreviated_rankings()

    assert get_player_rank("Diego Dedura Palomero", rankings) == 257
    assert get_player_rank("Bu Yunchaokete", rankings) == 109


def test_reversed_and_surname_only_fallbacks_fail_closed_without_given_evidence():
    rankings = _abbreviated_rankings()

    assert get_player_rank("Bu Zhao", rankings) is None
    assert get_player_rank("Darderi", rankings) is None


def test_same_initial_abbreviation_requires_full_profile_url_identity():
    rankings = pd.DataFrame([{
        "rank": 18,
        "player_name": "V. Example",
        "points": 2210,
        "player_url": "/en/players/victor-example/e001/overview",
    }])

    # An initial-only display cannot silently bind a different same-initial
    # full name. The official URL slug supplies the disambiguating identity.
    assert get_player_rank("Vito Example", rankings) is None
    assert get_player_rank("Victor Example", rankings) == 18
    assert get_player_url("Victor Example", rankings) == (
        "/en/players/victor-example/e001/overview"
    )

    no_url = rankings.drop(columns=["player_url"])
    assert get_player_rank("Victor Example", no_url) is None


def test_canonical_player_url_is_an_exact_fail_closed_lookup_key():
    rankings = pd.DataFrame([
        {
            "rank": 18,
            "player_name": "J. Cerundolo",
            "points": 2210,
            "player_url": "/en/players/juan-manuel-cerundolo/c0c1/overview",
        },
        {
            "rank": 22,
            "player_name": "J. Cerundolo",
            "points": 1880,
            "player_url": "/en/players/juan-martin-cerundolo/c0c2/overview",
        },
    ])

    assert get_player_rank("Juan Manuel Cerundolo", rankings) == 18
    assert get_player_rank("Juan Martin Cerundolo", rankings) == 22
    assert get_player_rank("J. Cerundolo", rankings) is None
    assert get_player_rank(
        "J. Cerundolo",
        rankings,
        player_url="https://www.atptour.com/en/players/juan-martin-cerundolo/c0c2/bio/",
    ) == 22
    assert get_player_rank(
        "J. Cerundolo",
        rankings,
        player_url="/en/players/not-present/z999/overview",
    ) is None


def test_profile_identity_allows_ordered_middle_name_and_compound_spacing():
    rankings = pd.DataFrame([
        {
            "rank": 360,
            "player_name": "D. Stricker",
            "points": 140,
            "player_url": "/en/players/dominic-stricker/s0la/overview",
        },
        {
            "rank": 155,
            "player_name": "S. Kwon",
            "points": 410,
            "player_url": "/en/players/soonwoo-kwon/kf17/overview",
        },
    ])

    assert get_player_rank("Dominic Stephan Stricker", rankings) == 360
    assert get_player_rank("Soon Woo Kwon", rankings) == 155


def test_real_alias_misses_are_identity_unresolved_not_unranked():
    rankings = pd.DataFrame([
        {
            "rank": 111,
            "player_name": "C. Wong",
            "points": 520,
            "player_url": "/en/players/coleman-wong/w0bh/overview",
        },
        {
            "rank": 100,
            "player_name": "A. Shevchenko",
            "points": 615,
            "player_url": "/en/players/aleksandr-shevchenko/s0h2/overview",
        },
        {
            "rank": 595,
            "player_name": "S. Popovic",
            "points": 61,
            "player_url": "/en/players/stefan-popovic/p0g5/overview",
        },
    ])

    for unresolved in (
        "Chak Lam Coleman Wong",
        "Alexander Shevchenko",
        "Stevan Popovic",
    ):
        assert get_player_rank(unresolved, rankings) is None
        assert get_player_lookup_status(unresolved, rankings) == (
            "identity_unresolved"
        )

    assert get_player_lookup_status("Unknown Futures Player", rankings) == (
        "not_ranked"
    )
    assert get_player_lookup_status("Coleman Wong", rankings) == "resolved"
