import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import cohorts


def _pred_log():
    return pd.DataFrame([
        # settled gold row, all models present
        dict(match_uid="m1", actual_winner=1, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.7, xgb_p1_prob=0.65, rf_p1_prob=0.6, market_p1_prob=0.62,
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        # settled complete-but-legacy row, NN+market only (xgb/rf null)
        dict(match_uid="m2", actual_winner=2, features_complete=True,
             logging_quality="legacy_backfilled", rescore_quality="legacy_fallback_match",
             model_p1_prob=0.4, xgb_p1_prob=np.nan, rf_p1_prob=np.nan, market_p1_prob=0.45,
             p1_odds_decimal=2.3, p2_odds_decimal=1.6),
        # unsettled row -> excluded
        dict(match_uid="m3", actual_winner=np.nan, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.5, xgb_p1_prob=0.5, rf_p1_prob=0.5, market_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=1.8),
    ])


def _shadow_log():
    # shadow rows for m1 (settled) and m3 (unsettled -> excluded)
    return pd.DataFrame([
        dict(match_uid="m1", model_family="catboost", shadow_p1_prob=0.68,
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        dict(match_uid="m1", model_family="lightgbm", shadow_p1_prob=0.66,
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        dict(match_uid="m3", model_family="catboost", shadow_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=1.8),
    ])


def test_ground_truth_orientation_and_dedup():
    gt = cohorts.build_ground_truth(_pred_log())
    assert gt.loc["m1"] == 1   # player1 won
    assert gt.loc["m2"] == 0   # player2 won
    assert "m3" not in gt.index  # unsettled excluded


def test_scored_frame_models_and_tiers():
    scored = cohorts.build_scored_frame(_pred_log(), None)
    m1 = scored[scored.match_uid == "m1"]
    assert set(m1.model) == {"nn", "xgb", "rf", "market"}
    assert bool(m1.is_gold.iloc[0]) is True
    m2 = scored[scored.match_uid == "m2"]
    assert set(m2.model) == {"nn", "market"}
    assert bool(m2.is_gold.iloc[0]) is False
    assert bool(m2.is_complete.iloc[0]) is True
    assert scored[(scored.match_uid == "m1") & (scored.model == "nn")].y1.iloc[0] == 1


def test_shadow_models_join_to_ground_truth():
    scored = cohorts.build_scored_frame(_pred_log(), _shadow_log())
    sh = scored[scored.model.str.startswith("shadow_")]
    # only m1 shadows survive (m3 unsettled)
    assert set(sh.match_uid) == {"m1"}
    assert set(sh.model) == {"shadow_catboost", "shadow_lightgbm"}
    assert sh.y1.unique().tolist() == [1]


def test_intersection_excludes_partial_coverage():
    scored = cohorts.build_scored_frame(_pred_log(), None)
    inter = cohorts.intersection_uids(scored, ["nn", "xgb", "rf", "market"], "is_complete")
    assert inter == {"m1"}   # m2 lacks xgb/rf
