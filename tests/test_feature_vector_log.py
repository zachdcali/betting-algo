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
        P2_Hand_U=0, P2_Hand_A=1,
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
    assert rows.loc[0, "p1_hand"] == "R"
    assert rows.loc[0, "p2_hand"] == "A"
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


def test_blank_id_diagnostic_never_replaces_exact_id_incomplete_row(tmp_path, monkeypatch):
    path = tmp_path / "feature_vectors.csv"
    monkeypatch.setattr(feature_log, "PATH", str(path))
    incomplete = _features(_defaulted_features="Player1_Height")

    feature_log.save_feature_vector(
        "A", "B", "2026-07-13", "run_exact", incomplete, False,
        match_uid="match_exact", feature_snapshot_id="feat_exact",
        build_status="ok",
    )
    feature_log.save_feature_vector(
        "A", "B", "2026-07-13", "run_skip", {}, False,
        match_uid="match_exact", feature_snapshot_id="",
        build_status="skip",
    )

    rows = pd.read_csv(path, keep_default_na=False)
    assert len(rows) == 2
    exact = rows.loc[rows["feature_snapshot_id"].eq("feat_exact")].iloc[0]
    diagnostic = rows.loc[rows["feature_snapshot_id"].eq("")].iloc[0]
    assert exact["build_status"] == "ok"
    assert str(exact["features_complete"]).lower() == "false"
    assert diagnostic["build_status"] == "skip"


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
