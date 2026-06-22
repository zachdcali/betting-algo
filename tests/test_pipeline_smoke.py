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
def orchestrator():
    return LiveBettingOrchestrator()


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
