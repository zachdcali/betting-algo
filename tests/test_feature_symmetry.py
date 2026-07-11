"""P1/P2 symmetry: build_141_features_from_slugs(A, B) must mirror (B, A).

Orientation bugs (P1 stats landing in P2 slots, one-sided diff signs) kept
surfacing in hand audits of live matches. This pins the invariant:

- P1_/P2_-prefixed features swap exactly under player swap
- *_Diff features flip sign (P1 minus P2 convention)
- H2H_P1_Wins/H2H_P2_Wins swap
- context features (Surface_/Round_/Level_/Draw/Best_of) are identical

Everything is stubbed — no network, no store, no TA. Two synthetic players
with distinct profiles/histories and two shared H2H meetings recorded from
both perspectives.
"""
import os
import sys
from datetime import datetime

import pandas as pd
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..", "production")
for p in (BASE, os.path.join(BASE, "features"), os.path.join(BASE, "scraping")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["USE_STORE_HISTORY"] = "0"

from features.ta_feature_calculator import TAFeatureCalculator  # noqa: E402

REF = datetime(2026, 7, 9)

PROFILES = {
    "alpha-one": {
        "name": "Alpha One", "hand": "R", "height_cm": 198.0, "country": "USA",
        "birthdate": "1999-05-01", "age": 27.2, "current_rank": 45,
        "rank_points": 1100, "player_id": None,
    },
    "beta-two": {
        "name": "Beta Two", "hand": "L", "height_cm": 178.0, "country": "ESP",
        "birthdate": "2004-02-10", "age": 22.4, "current_rank": 130,
        "rank_points": 480, "player_id": None,
    },
}


def _history(own: str, opp: str, own_rank: int, opp_rank: int,
             results: str, opp_hand: str) -> pd.DataFrame:
    """Weekly matches walking back from REF-3d; `results` like 'WWLW...'."""
    rows = []
    surfaces = ["Hard", "Clay", "Grass"]
    for i, res in enumerate(results):
        date = pd.Timestamp(REF) - pd.Timedelta(days=3 + 7 * i)
        rows.append({
            "date": date, "event": f"Event {i}", "surface": surfaces[i % 3],
            "round": "QF", "level": "C", "rank": float(own_rank + i),
            "opp_name": f"{opp} Filler{i}", "opp_rank": float(opp_rank + 2 * i),
            "opp_hand": opp_hand, "opp_country": "FRA",
            "score": "6-4 6-4" if res == "W" else "4-6 4-6",
            "result": res, "source": "ta",
        })
    return pd.DataFrame(rows)


def _with_h2h(df: pd.DataFrame, opp_display: str, own_results: str) -> pd.DataFrame:
    """Two direct meetings, appended with fixed dates so both players carry
    the same matches from opposite perspectives."""
    meetings = []
    for j, res in enumerate(own_results):
        meetings.append({
            "date": pd.Timestamp(REF) - pd.Timedelta(days=40 + 30 * j),
            "event": f"H2H Event {j}", "surface": "Clay", "round": "SF",
            "level": "C", "rank": float("nan"),
            "opp_name": opp_display, "opp_rank": float("nan"),
            "opp_hand": PROFILES["beta-two" if opp_display == "Beta Two" else "alpha-one"]["hand"],
            "opp_country": "", "score": "7-5 6-3" if res == "W" else "5-7 3-6",
            "result": res, "source": "ta",
        })
    out = pd.concat([df, pd.DataFrame(meetings)], ignore_index=True)
    return out.sort_values("date", ascending=False).reset_index(drop=True)


HISTS = {
    "alpha-one": _with_h2h(
        _history("Alpha One", "Opp", 45, 90, "WWWLWWLWWWWLWWWWLWWW", "R"),
        "Beta Two", "WLW"),
    "beta-two": _with_h2h(
        _history("Beta Two", "Opp", 130, 150, "WLLWLLWWLLWLLWWLLWLL", "R"),
        "Alpha One", "LWL"),
}


class StubScraper:
    def get_player_profile(self, slug, **kw):
        return dict(PROFILES[slug])

    def get_player_matches(self, slug, years=None, **kw):
        return HISTS[slug].copy()

    def get_upcoming_match(self, *a, **kw):
        return None


def _build(slug1: str, slug2: str) -> dict:
    calc = TAFeatureCalculator(StubScraper())
    calc.use_store = False
    calc._atp_rankings = None  # symmetric default path, no CSV dependence
    return calc.build_141_features_from_slugs(
        slug1=slug1, slug2=slug2, match_date=REF, surface="Clay",
        tournament_level="C", draw_size=32, round_code="QF",
        force_refresh=False, persist=False, session_cache={},
        match_date_is_explicit=True,
    )


def _counterpart(name: str) -> str:
    n = name.replace("Player1", "\x00").replace("Player2", "Player1").replace("\x00", "Player2")
    n = n.replace("P1", "\x00").replace("P2", "P1").replace("\x00", "P2")
    if n.startswith("Handedness_Matchup_"):
        n = {"Handedness_Matchup_RL": "Handedness_Matchup_LR",
             "Handedness_Matchup_LR": "Handedness_Matchup_RL"}.get(n, n)
    return n


# One-sided training features with non-trivial swap semantics
# (see _h2h_from_matches: Laplace-smoothed, P1 perspective only)
MIRROR_ONE_MINUS = {"H2H_P1_WinRate"}       # swap -> 1 - x
MIRROR_NEGATE = {"H2H_Recent_P1_Advantage"}  # centered at 0, swap -> -x


def test_swap_symmetry():
    f12 = _build("alpha-one", "beta-two")
    f21 = _build("beta-two", "alpha-one")

    problems = []
    for name, v12 in f12.items():
        if name.startswith("_"):
            continue  # debug stamps
        if name in MIRROR_ONE_MINUS or name in MIRROR_NEGATE:
            a, b = float(v12), float(f21[name])
            want = 1.0 - b if name in MIRROR_ONE_MINUS else -b
            if a != pytest.approx(want, rel=1e-9, abs=1e-9):
                problems.append(f"{name}: {a} should mirror as {want} (got pair {a}, {b})")
            continue
        other = _counterpart(name)
        if other not in f21:
            problems.append(f"{name}: counterpart {other} missing")
            continue
        v21 = f21[other]
        try:
            a, b = float(v12), float(v21)
        except (TypeError, ValueError):
            if str(v12) != str(v21):
                problems.append(f"{name}: {v12!r} vs {other} {v21!r}")
            continue
        if other == name and "Diff" in name:
            # P1-minus-P2 convention: flips sign under swap
            if a != pytest.approx(-b, rel=1e-9, abs=1e-9):
                problems.append(f"{name}: expected sign flip, {a} vs {b}")
        elif a != pytest.approx(b, rel=1e-9, abs=1e-9):
            problems.append(f"{name} -> {other}: {a} vs {b}")

    assert not problems, "\n".join(problems[:40]) + (
        f"\n... +{len(problems) - 40} more" if len(problems) > 40 else "")


def test_h2h_orientation():
    f12 = _build("alpha-one", "beta-two")
    # Alpha leads the ledger 2-1: counts must be oriented, not just mirrored
    assert float(f12["H2H_Total_Matches"]) == 3.0
    assert float(f12["H2H_P1_Wins"]) == 2.0
    assert float(f12["H2H_P2_Wins"]) == 1.0
    assert float(f12["H2H_P1_WinRate"]) > 0.5
    f21 = _build("beta-two", "alpha-one")
    assert float(f21["H2H_P1_Wins"]) == 1.0
    assert float(f21["H2H_P2_Wins"]) == 2.0
    assert float(f21["H2H_P1_WinRate"]) < 0.5


def test_days_since_last_week_boundary():
    """A ref with a clock time must not turn the current week's rows into a
    'previous tournament' (the Days_Since_Last=5 bug, task #38): with rows in
    the current week and one previous event, days-since counts from the
    PREVIOUS event's Monday."""
    calc = TAFeatureCalculator(StubScraper())
    df = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-08"), "result": "W"},   # current week (Mon Jul 6)
        {"date": pd.Timestamp("2026-06-29"), "result": "W"},   # previous event Monday
    ])
    ref = datetime(2026, 7, 11, 12, 20)  # Saturday afternoon — carries a time
    days = calc._days_since_last_tournament(df, ref)
    assert days == (pd.Timestamp(ref) - pd.Timestamp("2026-06-29")).days == 12
