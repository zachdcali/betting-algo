import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pandas as pd
import db


def test_importer_creates_tables_and_dedups_by_key(tmp_path):
    prod = tmp_path
    # operational log with a duplicate match_uid -> last write wins (operational latest)
    pd.DataFrame([
        {"match_uid": "m1", "model_p1_prob": 0.6, "actual_winner": 1},
        {"match_uid": "m1", "model_p1_prob": 0.7, "actual_winner": 1},
        {"match_uid": "m2", "model_p1_prob": 0.4, "actual_winner": 2},
    ]).to_csv(prod / "prediction_log.csv", index=False)

    db_path = tmp_path / "test.db"
    summary = db.build_database(str(prod), str(db_path))
    assert summary["predictions"] == 2  # m1 deduped

    conn = db.connect(str(db_path))
    got = pd.read_sql("SELECT * FROM predictions ORDER BY match_uid", conn)
    conn.close()
    assert list(got["match_uid"]) == ["m1", "m2"]
    # last row for m1 wins, and numeric value round-trips as a number
    assert float(got[got.match_uid == "m1"]["model_p1_prob"].iloc[0]) == 0.7


def test_rebuild_is_idempotent(tmp_path):
    prod = tmp_path
    pd.DataFrame([{"match_uid": "m1", "model_p1_prob": 0.6}]).to_csv(prod / "prediction_log.csv", index=False)
    db_path = tmp_path / "test.db"
    db.build_database(str(prod), str(db_path))
    summary = db.build_database(str(prod), str(db_path))  # second run must not duplicate
    assert summary["predictions"] == 1


def test_missing_csv_is_skipped(tmp_path):
    summary = db.build_database(str(tmp_path), str(tmp_path / "empty.db"))
    assert all(v == 0 for v in summary.values())  # no CSVs -> all zero, no crash
