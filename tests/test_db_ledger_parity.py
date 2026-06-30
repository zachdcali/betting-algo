import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pandas as pd
import db
from evaluation import cohorts


def _prediction_log():
    return pd.DataFrame([
        dict(match_uid="m1", actual_winner=1, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.70, xgb_p1_prob=0.65, rf_p1_prob=0.60, market_p1_prob=0.62,
             p1_odds_decimal=1.70, p2_odds_decimal=2.20),
        dict(match_uid="m2", actual_winner=2, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.40, xgb_p1_prob=0.45, rf_p1_prob=0.50, market_p1_prob=0.48,
             p1_odds_decimal=2.30, p2_odds_decimal=1.60),
    ])


def test_scored_frame_parity_csv_vs_db(tmp_path):
    prod = tmp_path
    _prediction_log().to_csv(prod / "prediction_log.csv", index=False)
    db_path = tmp_path / "t.db"
    db.build_database(str(prod), str(db_path))

    csv_scored = cohorts.build_scored_frame(cohorts.load_prediction_log(str(prod)), None)
    db_scored = cohorts.build_scored_frame(db.read_table(str(db_path), "predictions"), None)

    key = ["match_uid", "model"]
    csv_k = csv_scored.set_index(key).sort_index()
    db_k = db_scored.set_index(key).sort_index()

    assert list(csv_k.index) == list(db_k.index)
    for col in ["p1_prob", "y1", "is_gold", "is_complete"]:
        a = csv_k[col].astype(float).round(9).tolist()
        b = db_k[col].astype(float).round(9).tolist()
        assert a == b, f"mismatch in {col}: {a} vs {b}"
