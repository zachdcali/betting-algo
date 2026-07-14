"""Chronological golden contract for ``base_141_shared@1.0.0``.

The same source ledger is adapted through the historical preprocessing state
machine and the TA live builder.  The target match is always evaluated as of a
hard cutoff; a deliberately impossible future result is then appended to both
sources to prove it cannot alter any of the 141 ordered fields.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
for path in (
    PRODUCTION,
    PRODUCTION / "features",
    PRODUCTION / "scraping",
    ROOT / "src" / "models" / "professional_tennis",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feature_contract import normalize_feature_vector, vector_sha256  # noqa: E402
from features.base_141_shared import (  # noqa: E402
    CanonicalDateError,
    SEMANTICS_ID as SHARED_SEMANTICS_ID,
    count_matches,
    current_streak,
    form_trend,
    h2h_features_from_counts,
    h2h_features_from_frame,
    last_surface,
    observations_from_records,
    player_temporal_features,
    rank_change,
    rank_volatility,
    sets_played,
    surface_transition_flag,
)
from features.ta_feature_calculator import (  # noqa: E402
    EXACT_141_FEATURES,
    TAFeatureCalculator,
    UnsafeToInferError,
)
import features.ta_feature_calculator as ta_feature_module  # noqa: E402
import preprocess as preprocess_module  # noqa: E402
from preprocess import (  # noqa: E402
    calculate_temporal_features,
    preprocess_jeffsackmann_data_for_ml,
)
from versioning import (  # noqa: E402
    HISTORICAL_SEMANTICS_ID,
    LIVE_SEMANTICS_ID,
    SHARED_SEMANTICS_CANDIDATE_ID,
)


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "base_141_shared_v1.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())
AS_OF = datetime.fromisoformat(FIXTURE["as_of"])

SURFACE_KEYS = ("Hard", "Clay", "Grass", "Carpet")
LEVEL_KEYS = ("G", "M", "A", "C", "S", "F", "25", "15", "O", "D")
ROUND_KEYS = ("R128", "R64", "R32", "R16", "Q1", "Q2", "Q3", "Q4", "QF", "SF", "F", "RR", "ER", "BR")
HAND_KEYS = ("R", "L", "U", "A")
COUNTRY_KEYS = ("USA", "GBR", "FRA", "ITA", "AUS", "SRB", "CZE", "ESP", "SUI", "GER", "ARG", "RUS")


def _age(player: dict, at: datetime | pd.Timestamp) -> float:
    return (pd.Timestamp(at).to_pydatetime() - datetime.strptime(player["birthdate"], "%Y-%m-%d")).days / 365.25


def _structural_columns(row: dict) -> dict:
    p1_rank = float(row["Player1_Rank"])
    p2_rank = float(row["Player2_Rank"])
    p1_points = float(row["Player1_Rank_Points"])
    p2_points = float(row["Player2_Rank_Points"])
    p1_height = float(row["Player1_Height"])
    p2_height = float(row["Player2_Height"])
    p1_age = float(row["Player1_Age"])
    p2_age = float(row["Player2_Age"])
    row.update({
        "Rank_Diff": p1_rank - p2_rank,
        "Rank_Points_Diff": p1_points - p2_points,
        "Avg_Rank": (p1_rank + p2_rank) / 2.0,
        "Avg_Rank_Points": (p1_points + p2_points) / 2.0,
        "Rank_Ratio": max(p1_rank, p2_rank) / min(p1_rank, p2_rank),
        "Height_Diff": p1_height - p2_height,
        "Avg_Height": (p1_height + p2_height) / 2.0,
        "Age_Diff": p1_age - p2_age,
        "Avg_Age": (p1_age + p2_age) / 2.0,
    })
    surface = str(row["surface"]).title()
    for key in SURFACE_KEYS:
        row[f"Surface_{key}"] = int(surface == key)
    level = str(row["tourney_level"]).upper()
    for key in LEVEL_KEYS:
        row[f"Level_{key}"] = int(level == key)
    round_code = str(row["round"]).upper()
    for key in ROUND_KEYS:
        row[f"Round_{key}"] = int(round_code == key)
    for prefix, hand in (("P1", row["Player1_Hand"]), ("P2", row["Player2_Hand"])):
        for key in HAND_KEYS:
            row[f"{prefix}_Hand_{key}"] = int(str(hand).upper() == key)
    for prefix, country in (("P1", row["Player1_IOC"]), ("P2", row["Player2_IOC"])):
        normalized = str(country).upper()
        for key in COUNTRY_KEYS:
            row[f"{prefix}_Country_{key}"] = int(normalized == key)
        row[f"{prefix}_Country_Other"] = int(normalized not in COUNTRY_KEYS)
    return row


def _historical_row(match: dict, ordinal: int) -> dict:
    players = FIXTURE["players"]
    p1 = players[match["player1"]]
    p2 = players[match["player2"]]
    at = pd.Timestamp(match["date"])
    context = FIXTURE["context"]
    return _structural_columns({
        "tourney_id": match["id"],
        "tourney_name": f"Fixture {match['id']}",
        "tourney_date": at,
        "inferred_match_date": at,
        "match_num": ordinal,
        "surface": match.get("surface", context["surface"]),
        "tourney_level": context["tournament_level"],
        "round": context["round_code"],
        "draw_size": context["draw_size"],
        "best_of": 3,
        "score": match.get("score", ""),
        "Player1_ID": p1["id"],
        "Player1_Name": p1["name"],
        "Player1_Hand": p1["hand"],
        "Player1_Height": p1["height_cm"],
        "Player1_IOC": p1["country"],
        "Player1_Age": _age(p1, at),
        "Player1_Rank": match["p1_rank"],
        "Player1_Rank_Points": p1["rank_points"],
        "Player2_ID": p2["id"],
        "Player2_Name": p2["name"],
        "Player2_Hand": p2["hand"],
        "Player2_Height": p2["height_cm"],
        "Player2_IOC": p2["country"],
        "Player2_Age": _age(p2, at),
        "Player2_Rank": match["p2_rank"],
        "Player2_Rank_Points": p2["rank_points"],
        "Player1_Wins": int(match["player1_wins"]),
    })


def _target_row(ordinal: int) -> dict:
    target = FIXTURE["target"]
    target_match = {
        **target,
        # Sackmann carries a date-only midnight while live serving carries an
        # actual kickoff clock.  The shared contract must make those equal.
        "date": pd.Timestamp(FIXTURE["as_of"]).normalize().isoformat(),
        "surface": FIXTURE["context"]["surface"],
        "score": "",
        # The target is never applied to its own features; a deterministic
        # placeholder result merely satisfies the historical row shape.
        "player1_wins": 0,
    }
    return _historical_row(target_match, ordinal)


def _historical_vector(*, include_future: bool = False) -> dict[str, float]:
    matches = list(FIXTURE["matches"])
    rows = [_historical_row(match, idx) for idx, match in enumerate(matches)]
    rows.append(_target_row(len(rows)))
    if include_future:
        rows.extend(
            _historical_row(match, len(rows) + idx)
            for idx, match in enumerate(FIXTURE["future_matches"])
        )
    built = calculate_temporal_features(
        pd.DataFrame(rows),
        feature_semantics_id=SHARED_SEMANTICS_ID,
    )
    target = built.loc[built["tourney_id"] == FIXTURE["target"]["id"]].iloc[0]
    missing = [name for name in EXACT_141_FEATURES if name not in target.index]
    assert not missing, f"historical candidate missing schema fields: {missing}"
    return {name: float(target[name]) for name in EXACT_141_FEATURES}


def _perspective_history(player_key: str, *, include_future: bool) -> pd.DataFrame:
    players = FIXTURE["players"]
    rows = []
    ledger = list(FIXTURE["matches"])
    if include_future:
        ledger += FIXTURE["future_matches"]
    for match in ledger:
        if player_key not in (match["player1"], match["player2"]):
            continue
        own_is_p1 = match["player1"] == player_key
        opponent_key = match["player2"] if own_is_p1 else match["player1"]
        opponent = players[opponent_key]
        won = bool(match["player1_wins"]) if own_is_p1 else not bool(match["player1_wins"])
        rows.append({
            "date": pd.Timestamp(match["date"]),
            "event": f"Fixture {match['id']}",
            "surface": match["surface"],
            "round": FIXTURE["context"]["round_code"],
            "level": FIXTURE["context"]["tournament_level"],
            "rank": float(match["p1_rank"] if own_is_p1 else match["p2_rank"]),
            "opp_name": opponent["name"],
            "opp_rank": float(match["p2_rank"] if own_is_p1 else match["p1_rank"]),
            "opp_hand": opponent["hand"],
            "opp_country": opponent["country"],
            "score": match["score"],
            "result": "W" if won else "L",
            "source": "golden_fixture",
        })
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


class FixtureScraper:
    def __init__(self, *, include_future: bool = False):
        self.include_future = include_future
        self.by_slug = {player["slug"]: key for key, player in FIXTURE["players"].items()}

    def get_player_profile(self, slug, **_kwargs):
        player = dict(FIXTURE["players"][self.by_slug[slug]])
        return {
            "name": player["name"],
            "slug": player["slug"],
            "hand": player["hand"],
            "height_cm": player["height_cm"],
            "country": player["country"],
            "birthdate": player["birthdate"],
            "current_rank": player["current_rank"],
            "player_id": None,
        }

    def get_player_matches(self, slug, years=None, **_kwargs):
        del years
        return _perspective_history(self.by_slug[slug], include_future=self.include_future)

    def get_upcoming_match(self, *_args, **_kwargs):
        return None


def _live_vector(
    *,
    include_future: bool = False,
    match_date=AS_OF,
    canonical_match_date=None,
) -> dict[str, float]:
    players = FIXTURE["players"]
    calc = TAFeatureCalculator(
        FixtureScraper(include_future=include_future),
        feature_semantics_id=SHARED_SEMANTICS_ID,
    )
    calc.use_store = False
    calc._atp_rankings = pd.DataFrame([
        {
            "player_name": player["name"],
            "rank": player["current_rank"],
            "points": player["rank_points"],
        }
        for player in players.values()
    ])
    features = calc.build_141_features_from_slugs(
        slug1=players["alpha"]["slug"],
        slug2=players["beta"]["slug"],
        match_date=match_date,
        canonical_match_date=canonical_match_date,
        surface=FIXTURE["context"]["surface"],
        tournament_level=FIXTURE["context"]["tournament_level"],
        draw_size=FIXTURE["context"]["draw_size"],
        round_code=FIXTURE["context"]["round_code"],
        expected_event_title=FIXTURE["context"]["expected_event_title"],
        force_refresh=False,
        persist=False,
        session_cache={},
        match_date_is_explicit=True,
    )
    assert features["_feature_semantics_id"] == SHARED_SEMANTICS_ID
    return {name: float(features[name]) for name in EXACT_141_FEATURES}


def _assert_structurally_valid(vector: dict[str, float]) -> None:
    assert list(vector) == EXACT_141_FEATURES
    normalized, issues = normalize_feature_vector(vector, EXACT_141_FEATURES)
    assert not issues
    assert len(normalized) == 141
    assert all(math.isfinite(value) for value in normalized.values())


def test_candidate_is_reserved_and_both_legacy_defaults_are_unchanged():
    assert SHARED_SEMANTICS_ID == SHARED_SEMANTICS_CANDIDATE_ID
    default_live = TAFeatureCalculator(FixtureScraper())
    assert default_live.feature_semantics_id == LIVE_SEMANTICS_ID
    assert HISTORICAL_SEMANTICS_ID != SHARED_SEMANTICS_ID

    registry = json.loads((PRODUCTION / "models" / "model_registry.json").read_text())
    contract = registry["feature_contracts"]["semantics"][SHARED_SEMANTICS_ID]
    assert contract == {"status": "reserved_candidate", "parity_verified": False}
    current_entries = (
        registry["models"][registry["current_version"]],
        registry["xgboost"]["models"][registry["xgboost"]["current_version"]],
        registry["random_forest"]["models"][registry["random_forest"]["current_version"]],
    )
    for current in current_entries:
        assert current["training_feature_semantics_id"] == HISTORICAL_SEMANTICS_ID
        assert current["live_feature_semantics_id"] == LIVE_SEMANTICS_ID


def test_historical_candidate_cannot_overwrite_active_legacy_dataset():
    active = ROOT / "data" / "JeffSackmann" / "jeffsackmann_ml_ready_SURFACE_FIX.csv"
    with pytest.raises(ValueError, match="cannot overwrite the active legacy dataset"):
        preprocess_jeffsackmann_data_for_ml(
            output_path=active,
            feature_semantics_id=SHARED_SEMANTICS_ID,
        )


def test_historical_candidate_rejects_hardlink_and_casefolded_active_aliases(
    tmp_path, monkeypatch
):
    active = tmp_path / "active.csv"
    hardlink = tmp_path / "candidate.csv"
    active.write_text("legacy\n", encoding="utf-8")
    os.link(active, hardlink)
    monkeypatch.setattr(preprocess_module, "_ACTIVE_LEGACY_DATASET", active)

    with pytest.raises(ValueError, match="cannot overwrite the active legacy dataset"):
        preprocess_jeffsackmann_data_for_ml(
            output_path=hardlink,
            feature_semantics_id=SHARED_SEMANTICS_ID,
        )

    assert preprocess_module._paths_alias(
        tmp_path / "ACTIVE.CSV", active
    )


def test_full_missing_profile_preprocessing_is_finite_and_live_equivalent(
    tmp_path, monkeypatch
):
    raw = pd.DataFrame([{
        "data_source": "missingness_fixture",
        "tourney_id": "missing-1",
        "tourney_name": "Missingness Cup",
        "tourney_date": 20260720,
        "surface": "Hard",
        "draw_size": 32,
        "tourney_level": "A",
        "match_num": 1,
        "round": "R32",
        "score": "",
        "best_of": 3,
        "winner_id": 1,
        "winner_name": "Missing Alpha",
        "winner_hand": np.nan,
        "winner_ht": np.nan,
        "winner_ioc": np.nan,
        "winner_age": np.nan,
        "winner_seed": np.nan,
        "winner_entry": np.nan,
        "winner_rank": np.nan,
        "winner_rank_points": np.nan,
        "loser_id": 2,
        "loser_name": "Missing Beta",
        "loser_hand": np.nan,
        "loser_ht": np.nan,
        "loser_ioc": np.nan,
        "loser_age": np.nan,
        "loser_seed": np.nan,
        "loser_entry": np.nan,
        "loser_rank": np.nan,
        "loser_rank_points": np.nan,
    }])
    candidate_path = tmp_path / "base_141_shared_missingness.csv"
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            preprocess_module.pd,
            "read_csv",
            lambda *_args, **_kwargs: raw.copy(),
        )
        historical_frame, feature_names = preprocess_jeffsackmann_data_for_ml(
            output_path=candidate_path,
            feature_semantics_id=SHARED_SEMANTICS_ID,
        )

    assert feature_names == EXACT_141_FEATURES
    historical = {
        name: float(historical_frame.iloc[0][name])
        for name in EXACT_141_FEATURES
    }
    _assert_structurally_valid(historical)
    assert historical["Player1_Rank"] == 999.0
    assert historical["Player2_Rank"] == 999.0
    assert historical["Player1_Rank_Points"] == 0.0
    assert historical["Player2_Rank_Points"] == 0.0
    assert historical["Player1_Height"] == 180.0
    assert historical["Player2_Height"] == 180.0
    assert historical["Player1_Age"] == 25.0
    assert historical["Player2_Age"] == 25.0
    assert historical["Rank_Diff"] == 0.0
    assert historical["Avg_Rank"] == 999.0
    assert historical["Rank_Ratio"] == 1.0
    assert historical["Rank_Points_Diff"] == 0.0
    assert historical["Avg_Rank_Points"] == 0.0
    assert historical["Height_Diff"] == 0.0
    assert historical["Avg_Height"] == 180.0
    assert historical["Age_Diff"] == 0.0
    assert historical["Avg_Age"] == 25.0

    class MissingProfileScraper:
        @staticmethod
        def get_player_profile(slug, **_kwargs):
            return {
                "name": "Missing Alpha" if slug == "missing-alpha" else "Missing Beta",
                "slug": slug,
                "hand": None,
                "height_cm": None,
                "country": None,
                "birthdate": None,
                "age": None,
                "current_rank": None,
                "player_id": None,
            }

        @staticmethod
        def get_player_matches(*_args, **_kwargs):
            return pd.DataFrame()

        @staticmethod
        def get_upcoming_match(*_args, **_kwargs):
            return None

    monkeypatch.setattr(ta_feature_module, "needs_stitching", lambda *_args: False)
    monkeypatch.setattr(
        ta_feature_module, "batch_get_profiles", lambda *_args, **_kwargs: {}
    )
    calc = TAFeatureCalculator(
        MissingProfileScraper(), feature_semantics_id=SHARED_SEMANTICS_ID
    )
    calc.use_store = False
    calc._atp_rankings = None
    live_features = calc.build_141_features_from_slugs(
        slug1="missing-alpha",
        slug2="missing-beta",
        match_date=datetime(2026, 7, 20, 12),
        surface="Hard",
        tournament_level="A",
        draw_size=32,
        round_code="R32",
        force_refresh=False,
        persist=False,
        session_cache={},
        match_date_is_explicit=True,
    )
    live = {name: float(live_features[name]) for name in EXACT_141_FEATURES}
    _assert_structurally_valid(live)
    assert historical == live


def test_ranked_missing_points_and_duplicate_name_use_shared_500_not_legacy_curve(
    tmp_path, monkeypatch
):
    raw = pd.DataFrame([{
        "data_source": "missing_points_fixture",
        "tourney_id": "missing-points-1",
        "tourney_name": "Missing Points Cup",
        "tourney_date": 20260720,
        "surface": "Hard",
        "draw_size": 32,
        "tourney_level": "A",
        "match_num": 1,
        "round": "R32",
        "score": "",
        "best_of": 3,
        "winner_id": 11,
        "winner_name": "Known Alpha",
        "winner_hand": "U",
        "winner_ht": 180,
        "winner_ioc": np.nan,
        "winner_age": 25,
        "winner_seed": np.nan,
        "winner_entry": np.nan,
        "winner_rank": 250,
        "winner_rank_points": np.nan,
        "loser_id": 22,
        "loser_name": "Known Beta",
        "loser_hand": "U",
        "loser_ht": 180,
        "loser_ioc": np.nan,
        "loser_age": 25,
        "loser_seed": np.nan,
        "loser_entry": np.nan,
        "loser_rank": 250,
        "loser_rank_points": np.nan,
    }])
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            preprocess_module.pd,
            "read_csv",
            lambda *_args, **_kwargs: raw.copy(),
        )
        historical_frame, feature_names = preprocess_jeffsackmann_data_for_ml(
            output_path=tmp_path / "base_141_shared_missing_points.csv",
            feature_semantics_id=SHARED_SEMANTICS_ID,
        )
    assert feature_names == EXACT_141_FEATURES
    historical = {
        name: float(historical_frame.iloc[0][name])
        for name in EXACT_141_FEATURES
    }
    assert historical["Player1_Rank"] == 250.0
    assert historical["Player2_Rank"] == 250.0
    assert historical["Player1_Rank_Points"] == 500.0
    assert historical["Player2_Rank_Points"] == 500.0

    class KnownRankScraper:
        @staticmethod
        def get_player_profile(slug, **_kwargs):
            return {
                "name": "Known Alpha" if slug == "known-alpha" else "Known Beta",
                "slug": slug,
                "hand": "U",
                "height_cm": 180,
                "country": None,
                "birthdate": None,
                "age": 25,
                "current_rank": 250,
                "player_id": 11 if slug == "known-alpha" else 22,
            }

        @staticmethod
        def get_player_matches(*_args, **_kwargs):
            return pd.DataFrame()

        @staticmethod
        def get_upcoming_match(*_args, **_kwargs):
            return None

    curve = pd.DataFrame({
        "player_name": [f"Unrelated Player {idx}" for idx in range(1, 61)],
        "rank": list(range(1, 61)),
        "points": [2000 - idx * 10 for idx in range(1, 61)],
    })
    ambiguous_curve = pd.concat([
        curve,
        pd.DataFrame([
            {"player_name": "Known Alpha", "rank": 250, "points": 777},
            {"player_name": "Known Alpha", "rank": 250, "points": 777},
        ]),
    ], ignore_index=True)
    monkeypatch.setattr(ta_feature_module, "needs_stitching", lambda *_args: False)
    monkeypatch.setattr(
        ta_feature_module, "batch_get_profiles", lambda *_args, **_kwargs: {}
    )

    def _build(semantics_id, rankings=curve):
        calc = TAFeatureCalculator(
            KnownRankScraper(), feature_semantics_id=semantics_id
        )
        calc.use_store = False
        calc._atp_rankings = rankings.copy()
        return calc.build_141_features_from_slugs(
            slug1="known-alpha",
            slug2="known-beta",
            match_date=datetime(2026, 7, 20, 12),
            surface="Hard",
            tournament_level="A",
            draw_size=32,
            round_code="R32",
            force_refresh=False,
            persist=False,
            session_cache={},
            match_date_is_explicit=True,
        )

    live_candidate_features = _build(
        SHARED_SEMANTICS_ID,
        rankings=ambiguous_curve,
    )
    live_candidate = {
        name: float(live_candidate_features[name]) for name in EXACT_141_FEATURES
    }
    _assert_structurally_valid(historical)
    _assert_structurally_valid(live_candidate)
    assert live_candidate == historical

    duplicate_one_null = pd.concat([
        curve,
        pd.DataFrame([
            {"player_name": "Known Alpha", "rank": 250, "points": 777},
            {"player_name": "Known Alpha", "rank": 250, "points": np.nan},
        ]),
    ], ignore_index=True)
    live_duplicate_one_null = _build(
        SHARED_SEMANTICS_ID,
        rankings=duplicate_one_null,
    )
    assert float(live_duplicate_one_null["Player1_Rank_Points"]) == 500.0

    stable_id_rankings = pd.DataFrame([
        {"player_id": 11, "player_name": "Known Alpha", "rank": 250, "points": 777},
        {"player_id": 99, "player_name": "Known Alpha", "rank": 250, "points": 888},
        {"player_id": 22, "player_name": "Known Beta", "rank": 250, "points": 555},
    ])
    live_stable_ids = _build(
        SHARED_SEMANTICS_ID,
        rankings=stable_id_rankings,
    )
    assert float(live_stable_ids["Player1_Rank_Points"]) == 777.0
    assert float(live_stable_ids["Player2_Rank_Points"]) == 555.0

    legacy = _build(LIVE_SEMANTICS_ID)
    expected_curve_edge = float(curve.sort_values("rank").iloc[-1]["points"])
    assert expected_curve_edge != 500.0
    assert float(legacy["Player1_Rank_Points"]) == expected_curve_edge
    assert float(legacy["Player2_Rank_Points"]) == expected_curve_edge

    legacy_duplicate_name = _build(
        LIVE_SEMANTICS_ID,
        rankings=ambiguous_curve,
    )
    assert float(legacy_duplicate_name["Player1_Rank_Points"]) == 777.0


def test_chronological_golden_is_field_exact_across_all_141_features():
    historical = _historical_vector()
    live = _live_vector()
    _assert_structurally_valid(historical)
    _assert_structurally_valid(live)

    problems = {
        name: (historical[name], live[name])
        for name in EXACT_141_FEATURES
        if historical[name] != pytest.approx(live[name], rel=0.0, abs=1e-12)
    }
    assert not problems, problems

    for name, expected in FIXTURE["expected_disputed"].items():
        assert historical[name] == pytest.approx(float(expected), rel=0.0, abs=1e-12)
        assert live[name] == pytest.approx(float(expected), rel=0.0, abs=1e-12)
    assert vector_sha256(historical, EXACT_141_FEATURES) == FIXTURE["expected_vector_sha256"]
    assert vector_sha256(live, EXACT_141_FEATURES) == FIXTURE["expected_vector_sha256"]


def test_future_rows_cannot_change_either_candidate_vector():
    historical = _historical_vector()
    live = _live_vector()
    assert _historical_vector(include_future=True) == historical
    assert _live_vector(include_future=True) == live


def test_first_surface_is_neutral_without_prior_history(monkeypatch):
    assert surface_transition_flag(None, None, "Clay") == 0
    player = FIXTURE["players"]["alpha"]
    temporal = player_temporal_features(
        (), AS_OF, "Clay", rank_as_of=player["current_rank"]
    )
    assert temporal["last_surface"] is None
    assert temporal["days_since_last"] == 60
    assert temporal["rust_flag"] == 0

    historical = calculate_temporal_features(
        pd.DataFrame([_target_row(0)]),
        feature_semantics_id=SHARED_SEMANTICS_ID,
    )
    assert float(historical.iloc[0]["Surface_Transition_Flag"]) == 0.0
    assert float(historical.iloc[0]["P1_Days_Since_Last"]) == 60.0
    assert float(historical.iloc[0]["P2_Days_Since_Last"]) == 60.0
    assert float(historical.iloc[0]["P1_Rust_Flag"]) == 0.0
    assert float(historical.iloc[0]["P2_Rust_Flag"]) == 0.0

    class EmptyHistoryScraper(FixtureScraper):
        def get_player_matches(self, slug, years=None, **_kwargs):
            del slug, years
            return pd.DataFrame()

    # Empty history normally triggers the upstream stitcher. This test owns
    # only feature semantics, so hold source discovery at an empty result.
    monkeypatch.setattr(ta_feature_module, "needs_stitching", lambda *_args: False)
    calc = TAFeatureCalculator(
        EmptyHistoryScraper(), feature_semantics_id=SHARED_SEMANTICS_ID
    )
    calc.use_store = False
    calc._atp_rankings = pd.DataFrame([
        {
            "player_name": value["name"],
            "rank": value["current_rank"],
            "points": value["rank_points"],
        }
        for value in FIXTURE["players"].values()
    ])
    live = calc.build_141_features_from_slugs(
        slug1=FIXTURE["players"]["alpha"]["slug"],
        slug2=FIXTURE["players"]["beta"]["slug"],
        match_date=AS_OF,
        surface="Clay",
        tournament_level="A",
        draw_size=32,
        round_code="R32",
        force_refresh=False,
        persist=False,
        session_cache={},
        match_date_is_explicit=True,
    )
    assert float(live["Surface_Transition_Flag"]) == 0.0
    assert float(live["P1_Days_Since_Last"]) == 60.0
    assert float(live["P2_Days_Since_Last"]) == 60.0
    assert float(live["P1_Rust_Flag"]) == 0.0
    assert float(live["P2_Rust_Flag"]) == 0.0


def test_shared_h2h_uses_stable_id_or_exact_full_name_and_fails_on_ambiguity():
    calc = TAFeatureCalculator(
        FixtureScraper(), feature_semantics_id=SHARED_SEMANTICS_ID
    )
    surname_collision = pd.DataFrame([
        {
            "date": "2026-06-01",
            "opp_name": "Francisco Cerundolo",
            "result": "W",
        },
        {
            "date": "2026-06-08",
            "opp_name": "Juan Manuel Cerundolo",
            "result": "L",
        },
    ])
    exact = calc._shared_h2h_stats(
        surname_collision, "Francisco Cerundolo", AS_OF
    )
    assert exact["H2H_Total_Matches"] == 1
    assert exact["H2H_P1_Wins"] == 1

    id_evidence = pd.DataFrame([
        {
            "date": "2026-06-01",
            "opp_name": "F. Cerundolo",
            "opp_id": 707.0,
            "result": "W",
        },
        {
            "date": "2026-06-08",
            "opp_name": "Francisco Cerundolo",
            "opp_id": 808.0,
            "result": "L",
        },
    ])
    stable = calc._shared_h2h_stats(
        id_evidence,
        "any display alias",
        AS_OF,
        p2_player_id=707,
    )
    assert stable["H2H_Total_Matches"] == 1
    assert stable["H2H_P1_Wins"] == 1

    ambiguous = pd.DataFrame([
        {
            "date": "2026-06-01",
            "opp_name": "Francisco Cerundolo",
            "opp_id": 707,
            "result": "W",
        },
        {
            "date": "2026-06-08",
            "opp_name": "Francisco Cerundolo",
            "opp_id": 808,
            "result": "L",
        },
    ])
    with pytest.raises(
        UnsafeToInferError,
        match="shared_h2h_ambiguous_opponent_identity",
    ):
        calc._shared_h2h_stats(ambiguous, "Francisco Cerundolo", AS_OF)


def test_rank_change_sign_and_recent_h2h_smoothing_are_pinned():
    alpha = _perspective_history("alpha", include_future=True)
    temporal = player_temporal_features(
        observations_from_records(alpha),
        AS_OF,
        "Clay",
        rank_as_of=FIXTURE["players"]["alpha"]["current_rank"],
    )
    # Positive means improved: rank at/before cutoff minus rank as of ref.
    assert temporal["rank_change_30d"] == 10.0
    assert temporal["rank_change_90d"] == 24.0

    one_win = h2h_features_from_counts(total=1, p1_wins=1, recent_p1_results=[True])
    one_loss = h2h_features_from_counts(total=1, p1_wins=0, recent_p1_results=[False])
    assert one_win["H2H_Recent_P1_Advantage"] == pytest.approx(0.125)
    assert one_loss["H2H_Recent_P1_Advantage"] == pytest.approx(-0.125)


def test_candidate_windows_are_calendar_day_granular_across_adapter_clocks():
    day = pd.Timestamp("2026-07-20")
    frame = pd.DataFrame([
        {
            "date": day - pd.Timedelta(days=14) + pd.Timedelta(hours=23),
            "result": "W",
            "surface": "Hard",
            "score": "6-4 6-4",
            "rank": 20,
        },
        {
            "date": day - pd.Timedelta(days=1) + pd.Timedelta(hours=23),
            "result": "L",
            "surface": "Clay",
            "score": "4-6 4-6",
            "rank": 21,
        },
        {
            # Same calendar day is never prior evidence, even though midnight
            # is earlier than a live noon kickoff.
            "date": day + pd.Timedelta(minutes=1),
            "result": "W",
            "surface": "Clay",
            "score": "6-0 6-0",
            "rank": 1,
        },
    ])
    history = observations_from_records(frame)
    historical_midnight = player_temporal_features(
        history, day, "Clay", rank_as_of=19
    )
    live_kickoff = player_temporal_features(
        history, day + pd.Timedelta(hours=19), "Clay", rank_as_of=19
    )

    assert historical_midnight == live_kickoff
    assert live_kickoff["matches_14d"] == 2
    assert live_kickoff["sets_14d"] == 4.0


def test_aware_kickoff_cannot_infer_event_date_without_explicit_canonical_date():
    naive_eastern_date = datetime(2026, 7, 20, 23, 30)
    aware_eastern = naive_eastern_date.replace(
        tzinfo=ZoneInfo("America/New_York")
    )
    aware_utc = aware_eastern.astimezone(ZoneInfo("UTC"))
    assert aware_utc.date() == date(2026, 7, 21)

    # The kernel accepts the canonical event-local day but rejects both clock
    # representations; converting either clock would silently choose a date.
    player_temporal_features((), naive_eastern_date, "Clay", rank_as_of=40)
    for clock in (aware_eastern, aware_utc):
        with pytest.raises(CanonicalDateError, match="canonical event-local date"):
            player_temporal_features((), clock, "Clay", rank_as_of=40)
        aware_historical = _target_row(0)
        aware_historical["inferred_match_date"] = clock
        with pytest.raises(ValueError, match="match_start_at_utc"):
            calculate_temporal_features(
                pd.DataFrame([aware_historical]),
                feature_semantics_id=SHARED_SEMANTICS_ID,
            )
        with pytest.raises(ValueError, match="match_start_at_utc"):
            _live_vector(match_date=clock)

    canonical = date(2026, 7, 20)
    expected = _live_vector(match_date=naive_eastern_date)
    assert _live_vector(
        match_date=aware_eastern,
        canonical_match_date=canonical,
    ) == expected
    assert _live_vector(
        match_date=aware_utc,
        canonical_match_date=canonical,
    ) == expected


def test_pure_kernel_tied_day_policy_is_neutral_unless_order_is_proven():
    tied = pd.DataFrame([
        {
            "date": "2026-07-10",
            "event": f"Tie {idx}",
            "opp_name": "Beta Two",
            "result": result,
            "surface": surface,
            "rank": rank,
            "score": score,
        }
        for idx, (result, surface, rank, score) in enumerate((
            ("W", "Clay", 10, "6-1 6-1"),
            ("W", "Hard", 20, "6-2 6-2"),
            ("W", "Clay", 10, "6-3 6-3"),
            ("L", "Hard", 20, "3-6 3-6"),
        ))
    ])
    forward = observations_from_records(tied)
    reverse = observations_from_records(tied.iloc[::-1].reset_index(drop=True))
    for history in (forward, reverse):
        assert current_streak(history) == 0
        assert last_surface(history, datetime(2026, 7, 11)) is None
    assert h2h_features_from_frame(tied, datetime(2026, 7, 11)) == (
        h2h_features_from_frame(tied.iloc[::-1], datetime(2026, 7, 11))
    )
    assert h2h_features_from_frame(
        tied, datetime(2026, 7, 11)
    )["H2H_Recent_P1_Advantage"] == 0.0
    assert rank_change(forward, datetime(2026, 8, 9), 30, 5) == 0.0
    assert rank_change(reverse, datetime(2026, 8, 9), 30, 5) == 0.0

    explicitly_ordered = tied.iloc[:2].copy()
    explicitly_ordered.loc[explicitly_ordered.index[0], "result"] = "L"
    explicitly_ordered["event"] = "One Ordered Event"
    explicitly_ordered["round_ord"] = [1, 2]
    ordered_forward = observations_from_records(explicitly_ordered)
    ordered_reverse = observations_from_records(explicitly_ordered.iloc[::-1])
    for history in (ordered_forward, ordered_reverse):
        assert current_streak(history) == 1
        assert last_surface(history, datetime(2026, 7, 11)) == "Hard"
        assert rank_change(history, datetime(2026, 8, 9), 30, 5) == 15.0

    different_events = explicitly_ordered.copy()
    different_events["event"] = ["Event A", "Event B"]
    different_scope_history = observations_from_records(different_events)
    assert current_streak(different_scope_history) == 0
    assert last_surface(different_scope_history, datetime(2026, 7, 11)) is None
    assert rank_change(
        different_scope_history, datetime(2026, 8, 9), 30, 5
    ) == 0.0

    misleading_order = explicitly_ordered.drop(columns=["round_ord"]).copy()
    misleading_order["order"] = [1, 2]
    misleading_forward = observations_from_records(misleading_order)
    misleading_reverse = observations_from_records(misleading_order.iloc[::-1])
    for history in (misleading_forward, misleading_reverse):
        assert current_streak(history) == 0
        assert last_surface(history, datetime(2026, 7, 11)) is None
        assert rank_change(history, datetime(2026, 8, 9), 30, 5) == 0.0


def test_equal_day_historical_rows_share_one_snapshot_and_are_permutation_invariant():
    prior = {
        "id": "prior",
        "date": "2026-07-01",
        "player1": "alpha",
        "player2": "gamma",
        "player1_wins": 1,
        "p1_rank": 45,
        "p2_rank": 190,
        "surface": "Clay",
        "score": "6-4 6-4",
    }
    same_a = {
        "id": "same-a",
        "date": "2026-07-10T08:00:00",
        "player1": "alpha",
        "player2": "beta",
        "player1_wins": 1,
        "p1_rank": 42,
        "p2_rank": 115,
        "surface": "Clay",
        "score": "6-3 6-3",
    }
    same_b = {
        **same_a,
        "id": "same-b",
        "date": "2026-07-10T20:00:00",
        "player1_wins": 0,
        "score": "3-6 3-6",
    }
    next_day = {
        **same_a,
        "id": "next-day",
        "date": "2026-07-11T12:00:00",
        "p1_rank": 41,
        "p2_rank": 114,
    }

    def _build(day_rows):
        rows = [_historical_row(prior, 0)]
        for match in day_rows:
            row = _historical_row(match, 1)
            # Deliberately tie the source order.  The batch apply order must
            # still be deterministic and must not affect same-day features.
            row["match_num"] = 1
            rows.append(row)
        rows.append(_historical_row(next_day, 2))
        built = calculate_temporal_features(
            pd.DataFrame(rows),
            feature_semantics_id=SHARED_SEMANTICS_ID,
        )
        return built.set_index("tourney_id")

    forward = _build([same_a, same_b])
    reverse = _build([same_b, same_a])
    for match_id in ("same-a", "same-b", "next-day"):
        assert {
            name: float(forward.loc[match_id, name])
            for name in EXACT_141_FEATURES
        } == {
            name: float(reverse.loc[match_id, name])
            for name in EXACT_141_FEATURES
        }

    # Both same-day matches read only the prior-day state.  In particular the
    # second row cannot see the first through career, H2H, lefty, or form state.
    same_a_vector = {
        name: float(forward.loc["same-a", name])
        for name in EXACT_141_FEATURES
    }
    same_b_vector = {
        name: float(forward.loc["same-b", name])
        for name in EXACT_141_FEATURES
    }
    assert same_a_vector == same_b_vector
    assert forward.loc["same-a", "P1_Level_Matches_Career"] == 1
    assert forward.loc["same-b", "P1_Level_Matches_Career"] == 1
    assert forward.loc["same-a", "P1_Round_WinRate_Career"] == forward.loc[
        "same-b", "P1_Round_WinRate_Career"
    ]
    assert forward.loc["same-a", "P1_Surface_Experience"] == 1
    assert forward.loc["same-b", "P1_Surface_Experience"] == 1
    assert forward.loc["same-a", "P1_Matches_14d"] == 1
    assert forward.loc["same-b", "P1_Matches_14d"] == 1
    assert forward.loc["same-a", "H2H_Total_Matches"] == 0
    assert forward.loc["same-b", "H2H_Total_Matches"] == 0
    assert forward.loc["same-a", "P1_vs_Lefty_WinRate"] == forward.loc[
        "same-b", "P1_vs_Lefty_WinRate"
    ]

    # The next day sees both results exactly once, after the atomic day flush.
    assert forward.loc["next-day", "P1_Level_Matches_Career"] == 3
    assert forward.loc["next-day", "H2H_Total_Matches"] == 2


def test_live_tied_day_forward_reverse_equals_historical_next_day(monkeypatch):
    prior = {
        "id": "prior-live",
        "date": "2026-06-29",
        "player1": "alpha",
        "player2": "gamma",
        "player1_wins": 1,
        "p1_rank": 45,
        "p2_rank": 190,
        "surface": "Clay",
        "score": "6-4 6-4",
    }
    tied = [
        {
            "id": f"live-tie-{idx}",
            "date": "2026-07-06",
            "player1": "alpha",
            "player2": "beta",
            "player1_wins": result,
            "p1_rank": 42,
            "p2_rank": 115,
            "surface": surface,
            "score": score,
        }
        for idx, (result, surface, score) in enumerate((
            (1, "Clay", "6-1 6-1"),
            (1, "Hard", "6-2 6-2"),
            (1, "Clay", "6-3 6-3"),
            (0, "Hard", "3-6 3-6"),
        ))
    ]
    target = {
        "id": "live-next-day",
        "date": "2026-07-07",
        "player1": "alpha",
        "player2": "beta",
        "player1_wins": 0,
        "p1_rank": 40,
        "p2_rank": 110,
        "surface": "Clay",
        "score": "",
    }
    historical_rows = [_historical_row(prior, 0)]
    historical_rows.extend(_historical_row(match, 1) for match in tied)
    historical_rows.append(_historical_row(target, 2))
    historical_frame = calculate_temporal_features(
        pd.DataFrame(historical_rows),
        feature_semantics_id=SHARED_SEMANTICS_ID,
    )
    historical_row = historical_frame.loc[
        historical_frame["tourney_id"] == target["id"]
    ].iloc[0]
    historical = {
        name: float(historical_row[name]) for name in EXACT_141_FEATURES
    }

    players = FIXTURE["players"]

    class TiedDayScraper(FixtureScraper):
        def __init__(self, *, reverse: bool):
            super().__init__()
            self.reverse = reverse

        def get_player_matches(self, slug, years=None, **_kwargs):
            del years
            player_key = self.by_slug[slug]
            day_rows = list(reversed(tied)) if self.reverse else list(tied)
            ledger = day_rows + [prior]
            rows = []
            for match in ledger:
                if player_key not in (match["player1"], match["player2"]):
                    continue
                own_is_p1 = match["player1"] == player_key
                opponent_key = match["player2"] if own_is_p1 else match["player1"]
                opponent = players[opponent_key]
                won = (
                    bool(match["player1_wins"])
                    if own_is_p1 else not bool(match["player1_wins"])
                )
                rows.append({
                    "date": pd.Timestamp(match["date"]),
                    "event": f"Ambiguity {match['id']}",
                    "surface": match["surface"],
                    "round": "R32",
                    "level": "A",
                    "rank": float(
                        match["p1_rank"] if own_is_p1 else match["p2_rank"]
                    ),
                    "opp_name": opponent["name"],
                    "opp_rank": float(
                        match["p2_rank"] if own_is_p1 else match["p1_rank"]
                    ),
                    "opp_hand": opponent["hand"],
                    "opp_country": opponent["country"],
                    "score": match["score"],
                    "result": "W" if won else "L",
                    "source": "date_only_fixture",
                })
            return pd.DataFrame(rows)

        @staticmethod
        def get_upcoming_match(*_args, **_kwargs):
            return {"date": "20260707", "round": "R32", "surface": "Clay"}

    monkeypatch.setattr(ta_feature_module, "needs_stitching", lambda *_args: False)

    def _build_live(reverse):
        calc = TAFeatureCalculator(
            TiedDayScraper(reverse=reverse),
            feature_semantics_id=SHARED_SEMANTICS_ID,
        )
        calc.use_store = False
        calc._atp_rankings = pd.DataFrame([
            {
                "player_name": player["name"],
                "rank": player["current_rank"],
                "points": player["rank_points"],
            }
            for player in players.values()
        ])
        features = calc.build_141_features_from_slugs(
            slug1=players["alpha"]["slug"],
            slug2=players["beta"]["slug"],
            match_date=datetime(2026, 7, 7, 23, 30),
            surface="Clay",
            tournament_level="A",
            draw_size=32,
            round_code="R32",
            expected_event_title="Ambiguity Cup",
            force_refresh=False,
            persist=False,
            session_cache={},
            match_date_is_explicit=True,
        )
        return {name: float(features[name]) for name in EXACT_141_FEATURES}

    live_forward = _build_live(False)
    live_reverse = _build_live(True)
    _assert_structurally_valid(historical)
    _assert_structurally_valid(live_forward)
    assert live_forward == live_reverse == historical
    assert historical["P1_WinStreak_Current"] == 0.0
    assert historical["P2_WinStreak_Current"] == 0.0
    assert historical["Surface_Transition_Flag"] == 0.0
    assert historical["H2H_Total_Matches"] == 4.0
    assert historical["H2H_Recent_P1_Advantage"] == 0.0


def test_half_open_windows_score_fallback_and_population_volatility():
    frame = pd.DataFrame([
        {"date": pd.Timestamp(AS_OF) - pd.Timedelta(days=14), "result": "W", "score": "6-4 6-4", "rank": 10},
        {"date": pd.Timestamp(AS_OF) - pd.Timedelta(days=2), "result": "L", "score": "W/O", "rank": 20},
        {"date": pd.Timestamp(AS_OF) - pd.Timedelta(days=1), "result": "W", "score": None, "rank": 30},
        {"date": pd.Timestamp(AS_OF), "result": "W", "score": "6-0 6-0", "rank": 1},
        {"date": pd.Timestamp(AS_OF) - pd.Timedelta(days=15), "result": "L", "score": "6-4 6-4", "rank": 999},
    ])
    history = observations_from_records(frame)
    # Exact cutoff is included; exact ref and older rows are excluded.
    assert count_matches(history, AS_OF, 14) == 3
    # Two observed sets + zero-set walkover + explicit 2.5 unscored fallback.
    assert sets_played(history, AS_OF, 14) == 4.5
    assert form_trend(history, AS_OF, 1) == 1.0
    assert rank_volatility(history, AS_OF, 14) == pytest.approx(
        math.sqrt(200.0 / 3.0)
    )
    assert rank_volatility(history[:2], AS_OF, 14) == 0.0
