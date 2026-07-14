"""Chronological golden contract for ``base_141_shared@1.0.0``.

The same source ledger is adapted through the historical preprocessing state
machine and the TA live builder.  The target match is always evaluated as of a
hard cutoff; a deliberately impossible future result is then appended to both
sources to prove it cannot alter any of the 141 ordered fields.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import sys

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
    SEMANTICS_ID as SHARED_SEMANTICS_ID,
    count_matches,
    form_trend,
    h2h_features_from_counts,
    observations_from_records,
    player_temporal_features,
    rank_volatility,
    sets_played,
    surface_transition_flag,
)
from features.ta_feature_calculator import (  # noqa: E402
    EXACT_141_FEATURES,
    TAFeatureCalculator,
)
import features.ta_feature_calculator as ta_feature_module  # noqa: E402
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
        "date": FIXTURE["as_of"],
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


def _live_vector(*, include_future: bool = False) -> dict[str, float]:
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
        match_date=AS_OF,
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

    historical = calculate_temporal_features(
        pd.DataFrame([_target_row(0)]),
        feature_semantics_id=SHARED_SEMANTICS_ID,
    )
    assert float(historical.iloc[0]["Surface_Transition_Flag"]) == 0.0

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
