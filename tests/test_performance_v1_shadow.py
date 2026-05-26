import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from shadow.performance_v1_shadow import (  # noqa: E402
    log_shadow_predictions,
    shadow_row_from_prediction,
    sync_shadow_settlements,
)


def test_shadow_row_devigs_market_and_uses_model_version():
    pred_row = pd.Series(
        {
            "run_id": "run_test",
            "match_uid": "match_test",
            "feature_snapshot_id": "feat_test",
            "meta_match_date": "2026-04-26",
            "player1_raw": "Player A",
            "player2_raw": "Player B",
            "event": "Example Open",
            "meta_surface_input": "Clay",
            "meta_level_input": "A",
            "meta_round_input": "QF",
            "performance_v1_features_available": True,
            "P1_Score_Matches_Last10": 10,
            "P2_Score_Matches_Last10": 9,
            "P1_Stat_Matches_Last10": 5,
            "P2_Stat_Matches_Last10": 4,
        }
    )
    odds_row = pd.Series(
        {
            "player1_implied_prob": "0.44",
            "player2_implied_prob": "0.66",
            "player1_odds_decimal": 2.2,
            "player2_odds_decimal": 1.6,
            "event": "Example Open",
            "match_time": "4/26/26 10:00 AM",
        }
    )

    row = shadow_row_from_prediction(
        pred_row,
        odds_row,
        {"shadow_p1_prob": 0.57, "shadow_p2_prob": 0.43},
        model_version="custom_shadow_v1",
        model_family="catboost",
    )

    assert row["model_version"] == "custom_shadow_v1"
    assert row["model_family"] == "catboost"
    assert row["shadow_status"] == "success"
    assert row["market_p1_prob"] == 0.4
    assert row["market_p2_prob"] == 0.6
    assert row["performance_features_available"] is True


def test_log_shadow_predictions_is_append_only_by_uid(tmp_path):
    path = tmp_path / "shadow.csv"
    row = {
        "shadow_prediction_uid": "shadow_1",
        "run_id": "run_test",
        "match_uid": "match_test",
    }

    assert log_shadow_predictions(path, [row]) == 1
    assert log_shadow_predictions(path, [row]) == 0

    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["shadow_prediction_uid"] == "shadow_1"


def test_sync_shadow_settlements_scores_shadow_rows(tmp_path):
    shadow_path = tmp_path / "shadow.csv"
    prediction_log_path = tmp_path / "prediction_log.csv"
    pd.DataFrame(
        [
            {
                "shadow_prediction_uid": "shadow_1",
                "run_id": "run_test",
                "match_uid": "match_test",
                "feature_snapshot_id": "feat_test",
                "p1": "Player A",
                "p2": "Player B",
                "shadow_p1_prob": 0.57,
                "market_p1_prob": 0.48,
            }
        ]
    ).to_csv(shadow_path, index=False)
    pd.DataFrame(
        [
            {
                "match_uid": "match_test",
                "feature_snapshot_id": "feat_test",
                "actual_winner": 1,
                "score": "6-4 6-4",
                "settled_at": "2026-04-27T12:00:00",
                "p1": "Player A",
                "p2": "Player B",
            }
        ]
    ).to_csv(prediction_log_path, index=False)

    assert sync_shadow_settlements(shadow_path, prediction_log_path) == 1
    synced = pd.read_csv(shadow_path)
    assert synced.loc[0, "actual_winner"] == 1
    assert synced.loc[0, "shadow_pick"] == "Player A"
    assert synced.loc[0, "shadow_correct"] == 1
    assert synced.loc[0, "market_correct"] == 0
