import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from scraping.atp_rankings_scraper import get_player_points, get_player_rank


def _abbreviated_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rank": 18, "player_name": "L. Darderi", "points": 2210},
            {"rank": 43, "player_name": "M. Berrettini", "points": 1075},
            {"rank": 109, "player_name": "Y. Bu", "points": 604},
            {"rank": 257, "player_name": "D. Dedura", "points": 219},
            {"rank": 434, "player_name": "A. Boitan", "points": 108},
            {"rank": 546, "player_name": "J. Berrettini", "points": 74},
            {"rank": 969, "player_name": "Micah Braswell", "points": 24},
            {"rank": 1125, "player_name": "Seongwoo Cho", "points": 28},
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
