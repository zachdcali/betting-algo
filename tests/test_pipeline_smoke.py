"""Offline end-to-end smoke test for the live prediction + staking path.

Exercises the highest-value, most-likely-to-silently-break path identified in
the pipeline audit: all three promoted models (NN v1.2.1, XGB v1.2.0,
RF v1.2.0) must actually load from disk and produce probabilities, and the
edge/Kelly-stake path must run without raising. No Bovada, no Tennis Abstract,
no network, and no writes to the real prediction logs (we call the inference
and staking methods directly rather than run_full_pipeline).

Settlement is validated separately by the live run, since enriching from
Tennis Abstract requires network access.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pandas as pd
import pytest

from main import LiveBettingOrchestrator
from models.inference import EXACT_141_FEATURES


def _canned_features() -> pd.DataFrame:
    """Two matches with the full 141-feature schema (varied so rows differ)."""
    rows = []
    for i, (p1, p2) in enumerate([("Player A", "Player B"), ("Player C", "Player D")]):
        feat = {f: 0.1 for f in EXACT_141_FEATURES}
        # vary a couple of features so the two matches are not identical
        feat[EXACT_141_FEATURES[0]] = 0.2 + 0.1 * i
        feat[EXACT_141_FEATURES[1]] = -0.1 * i
        feat.update({
            "player1_raw": p1, "player2_raw": p2,
            "event": "Smoke Open", "match_time": "2026-06-22 10:00",
            "status": "ok",
        })
        rows.append(feat)
    return pd.DataFrame(rows)


def _canned_odds() -> pd.DataFrame:
    """Market lists player1 as the underdog in both matches (decimal 3.0 / 1.4)."""
    rows = []
    for p1, p2 in [("Player A", "Player B"), ("Player C", "Player D")]:
        rows.append({
            "player1_raw": p1, "player2_raw": p2,
            "player1_odds_decimal": 3.0, "player2_odds_decimal": 1.4,
            "player1_implied_prob": 1 / 3.0, "player2_implied_prob": 1 / 1.4,
            "event": "Smoke Open", "match_time": "2026-06-22 10:00",
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def orchestrator(tmp_path_factory):
    """Keep constructor-time schema migration away from operational logs."""
    temp_dir = tmp_path_factory.mktemp("pipeline_smoke")
    config_path = temp_dir / "config.json"
    config_path.write_text(
        json.dumps({
            "logs_dir": str(temp_dir / "logs"),
            "data_dir": str(REPO_ROOT / "data"),
            "performance_shadow_enabled": False,
        }),
        encoding="utf-8",
    )
    return LiveBettingOrchestrator(str(config_path))


def test_all_three_models_load_and_predict(orchestrator):
    preds = orchestrator.generate_predictions(_canned_features())
    assert len(preds) == 2
    # NN primary probability present and valid
    assert preds["player1_win_prob"].notna().all()
    assert ((preds["player1_win_prob"] >= 0) & (preds["player1_win_prob"] <= 1)).all()
    # XGB and RF probabilities present (guards silent model-load failure)
    for col in ["xgb_p1_prob", "rf_p1_prob"]:
        assert col in preds.columns, f"{col} missing -> a model failed to load"
        assert preds[col].notna().all(), f"{col} is null -> model load/predict failed"
        assert ((preds[col] >= 0) & (preds[col] <= 1)).all()


def test_edge_and_stake_path_runs(orchestrator):
    preds = orchestrator.generate_predictions(_canned_features())
    bet_slips = orchestrator.calculate_edges_and_stakes(preds, _canned_odds())
    assert isinstance(bet_slips, pd.DataFrame)
    # If any bet qualified, it must carry a positive stake and the bet metadata.
    if not bet_slips.empty:
        assert "stake" in bet_slips.columns
        assert (bet_slips["stake"] > 0).all()


def test_incomplete_feature_matches_are_never_bet(orchestrator):
    """A match with defaulted/missing features must be logged but never staked,
    even when it has a strong edge."""
    common = dict(player1_win_prob=0.80, player2_win_prob=0.20, prediction_status="success",
                  match_date="2026-06-30", surface="Grass", tournament="Test", round="R128",
                  match_time="2026-06-30 10:00", event="Test Open")
    preds = pd.DataFrame([
        {**common, "player1_raw": "Complete A", "player2_raw": "Opp Alpha", "_has_defaulted_features": False},
        {**common, "player1_raw": "Incomplete B", "player2_raw": "Opp Beta", "_has_defaulted_features": True},
    ])
    odds = pd.DataFrame([
        dict(player1_raw="Complete A", player2_raw="Opp Alpha", player1_odds_decimal=2.0, player2_odds_decimal=2.0,
             player1_implied_prob=0.5, player2_implied_prob=0.5, event="Test Open", match_time="2026-06-30 10:00"),
        dict(player1_raw="Incomplete B", player2_raw="Opp Beta", player1_odds_decimal=2.0, player2_odds_decimal=2.0,
             player1_implied_prob=0.5, player2_implied_prob=0.5, event="Test Open", match_time="2026-06-30 10:00"),
    ])
    # Isolate feature eligibility from the real local paper-account backlog.
    slips = orchestrator.calculate_edges_and_stakes(
        preds, odds, bankroll=1000.0, available_bankroll=1000.0,
    )
    blob = slips.to_csv(index=False) if not slips.empty else ""
    # complete match (strong edge) is bet; incomplete match is excluded entirely
    assert "Complete A" in blob, "complete-feature match should be bettable"
    assert "Incomplete B" not in blob and "Opp Beta" not in blob, "incomplete-feature match must not be staked"
