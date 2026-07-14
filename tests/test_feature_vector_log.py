from pathlib import Path
import sys

import pandas as pd

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

import feature_vector_log as feature_log  # noqa: E402
from models.inference import EXACT_141_FEATURES  # noqa: E402


def _features(**updates):
    values = {name: 0.0 for name in EXACT_141_FEATURES}
    values.update({
        "Surface_Hard": 1.0,
        "Level_A": 1.0,
        "Round_R32": 1.0,
        "P1_Hand_U": 1.0,
        "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0,
        "P2_Country_Other": 1.0,
    })
    values.update(updates)
    return values


def test_feature_vectors_are_immutable_and_keyed_by_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "feature_vectors.csv"
    monkeypatch.setattr(feature_log, "PATH", str(path))
    features = _features(
        Player1_Rank=10, P1_Hand_U=0, P1_Hand_R=1,
        _defaulted_features="",
    )

    feature_log.save_feature_vector(
        "A", "B", "2026-07-13", "run_1", features, True,
        match_uid="match_1", feature_snapshot_id="feat_1",
    )
    feature_log.save_feature_vector(
        "A", "B", "2026-07-13", "run_1", {**features, "Player1_Rank": 99}, True,
        match_uid="match_1", feature_snapshot_id="feat_1",
    )
    feature_log.save_feature_vector(
        "A", "B", "2026-07-13", "run_2", {**features, "Player1_Rank": 11}, True,
        match_uid="match_1", feature_snapshot_id="feat_2",
    )

    rows = pd.read_csv(path)
    assert list(rows["feature_snapshot_id"]) == ["feat_1", "feat_2"]
    assert list(rows["run_id"]) == ["run_1", "run_2"]
    assert rows.loc[0, "match_uid"] == "match_1"
    assert rows.loc[0, "feature_count"] == 141
    assert len(rows.loc[0, "feature_schema_sha256"]) == 64
    assert len(rows.loc[0, "feature_vector_sha256"]) == 64


def test_skipped_feature_build_cannot_be_marked_complete(tmp_path, monkeypatch):
    path = tmp_path / "feature_vectors.csv"
    monkeypatch.setattr(feature_log, "PATH", str(path))

    feature_log.save_feature_vector(
        "A", "B", "2026-07-13", "run_skip", {}, True,
        match_uid="match_skip", feature_snapshot_id="feat_skip",
        build_status="skip",
    )

    row = pd.read_csv(path).iloc[0]
    assert row["build_status"] == "skip"
    assert str(row["features_complete"]).lower() == "false"


def test_incomplete_or_non_finite_vector_cannot_claim_exact_lineage(tmp_path, monkeypatch):
    path = tmp_path / "feature_vectors.csv"
    monkeypatch.setattr(feature_log, "PATH", str(path))

    feature_log.save_feature_vector(
        "A", "B", "2026-07-13", "run_bad", {"Player1_Rank": float("nan")}, True,
        match_uid="match_bad", feature_snapshot_id="feat_bad",
    )

    row = pd.read_csv(path).iloc[0]
    assert str(row["features_complete"]).lower() == "false"
    assert pd.isna(row["feature_vector_sha256"])


def test_invalid_one_hot_cardinality_cannot_claim_exact_lineage():
    invalid = _features(Surface_Hard=0.0, Surface_Clay=0.0)
    _, vector_hash, _ = feature_log.feature_fingerprint(invalid)
    assert vector_hash == ""
    assert "one_hot_cardinality:surface:0" in feature_log.feature_validation_issues(invalid)
