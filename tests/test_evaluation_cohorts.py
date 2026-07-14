import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
import pytest
from evaluation import cohorts
from models.inference import EXACT_141_FEATURES


def _valid_features() -> dict:
    row = {name: 0.0 for name in EXACT_141_FEATURES}
    row.update({
        "Surface_Hard": 1.0, "Level_A": 1.0, "Round_R32": 1.0,
        "P1_Hand_U": 1.0, "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0, "P2_Country_Other": 1.0,
    })
    return row


def _pred_log():
    return pd.DataFrame([
        # settled gold row, all models present
        dict(match_uid="m1", actual_winner=1, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             feature_snapshot_verified=True,
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
        # historical void sentinel -> excluded, never scored as a P2 win
        dict(match_uid="m4", actual_winner=-1, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.5, xgb_p1_prob=0.5, rf_p1_prob=0.5, market_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=2.0),
    ])


def _shadow_log():
    # shadow rows for m1 (settled) and m3 (unsettled -> excluded)
    return pd.DataFrame([
        dict(match_uid="m1", model_family="catboost", shadow_p1_prob=0.68,
             model_version="cat_v1", feature_snapshot_id="feat_m1",
             logged_at="2026-07-01T10:00:00Z",
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        # Hourly repeat of the same match/version must not inflate n.
        dict(match_uid="m1", model_family="catboost", shadow_p1_prob=0.99,
             model_version="cat_v1", feature_snapshot_id="feat_m1",
             logged_at="2026-07-01T11:00:00Z",
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        dict(match_uid="m1", model_family="lightgbm", shadow_p1_prob=0.66,
             model_version="lgbm_v1", feature_snapshot_id="feat_m1",
             logged_at="2026-07-01T10:00:00Z",
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        dict(match_uid="m3", model_family="catboost", shadow_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=1.8),
    ])


def test_ground_truth_orientation_and_dedup():
    gt = cohorts.build_ground_truth(_pred_log())
    assert gt.loc["m1"] == 1   # player1 won
    assert gt.loc["m2"] == 0   # player2 won
    assert "m3" not in gt.index  # unsettled excluded
    assert "m4" not in gt.index  # void sentinel excluded


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
    assert set(sh.model) == {"shadow_cat_v1", "shadow_lgbm_v1"}
    assert sh.y1.unique().tolist() == [1]
    assert len(sh) == 2
    assert sh.loc[sh.model == "shadow_cat_v1", "p1_prob"].iloc[0] == 0.68


def test_load_prediction_log_verifies_feature_snapshot_foreign_key(tmp_path):
    log = _pred_log().iloc[:2].copy()
    log["feature_snapshot_id"] = ["feat_present", "feat_missing"]
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    feature_row = _valid_features()
    feature_row.update({"feature_snapshot_id": "feat_present", "status": "ok"})
    pd.DataFrame([feature_row]).to_csv(
        logs / "features_20260701.csv", index=False,
    )

    loaded = cohorts.load_prediction_log(str(tmp_path)).set_index("match_uid")
    assert bool(loaded.loc["m1", "feature_snapshot_verified"])
    assert not bool(loaded.loc["m2", "feature_snapshot_verified"])
    scored = cohorts.build_scored_frame(loaded.reset_index(), None)
    assert bool(scored.loc[scored.match_uid == "m1", "is_gold"].iloc[0])
    assert not bool(scored.loc[scored.match_uid == "m2", "is_gold"].iloc[0])


def test_load_prediction_log_fails_on_corrupt_feature_lineage_file(tmp_path):
    log = _pred_log().iloc[:1].copy()
    log["feature_snapshot_id"] = ["feat_present"]
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "features_corrupt.csv").write_bytes(b'feature_snapshot_id,broken\n"unterminated')

    with pytest.raises(RuntimeError, match="feature snapshot verification"):
        cohorts.load_prediction_log(str(tmp_path))


def test_feature_snapshot_verification_rejects_tampered_content_hash(tmp_path):
    log = _pred_log().iloc[:1].copy()
    log["feature_snapshot_id"] = ["feat_tampered"]
    log["feature_vector_sha256"] = ["0" * 64]
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    feature_row = _valid_features()
    feature_row.update({"feature_snapshot_id": "feat_tampered", "status": "ok"})
    pd.DataFrame([feature_row]).to_csv(logs / "features_20260701.csv", index=False)

    loaded = cohorts.load_prediction_log(str(tmp_path))
    assert not bool(loaded.iloc[0]["feature_snapshot_verified"])


def test_intersection_excludes_partial_coverage():
    scored = cohorts.build_scored_frame(_pred_log(), None)
    inter = cohorts.intersection_uids(scored, ["nn", "xgb", "rf", "market"], "is_complete")
    assert inter == {"m1"}   # m2 lacks xgb/rf
