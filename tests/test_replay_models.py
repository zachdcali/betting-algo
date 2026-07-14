import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from evaluation import replay_manifest, replay_models
from feature_vector_log import feature_fingerprint
from logging_utils import build_feature_snapshot_id
from models.inference import EXACT_141_FEATURES
from versioning import FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256


def _valid_features(rank: float) -> dict:
    row = {name: 0.0 for name in EXACT_141_FEATURES}
    row.update(
        {
            "Player1_Rank": rank,
            "Surface_Hard": 1.0,
            "Level_A": 1.0,
            "Round_R32": 1.0,
            "P1_Hand_U": 1.0,
            "P2_Hand_U": 1.0,
            "P1_Country_Other": 1.0,
            "P2_Country_Other": 1.0,
        }
    )
    return row


def _feature(snapshot_id: str, uid: str, p1: str, p2: str, rank: float) -> dict:
    snapshot_label = snapshot_id
    run_id = f"run_{snapshot_label}" if snapshot_label else ""
    snapshot_id = (
        build_feature_snapshot_id(uid, run_id, p1, p2)
        if snapshot_label else ""
    )
    row = _valid_features(rank)
    schema_hash, vector_hash, _ = feature_fingerprint(row)
    row.update(
        {
            "feature_snapshot_id": snapshot_id,
            "match_uid": uid,
            "run_id": run_id,
            "player1_raw": p1,
            "player2_raw": p2,
            "meta_match_date": "2026-07-10",
            "timestamp": "2026-07-10T10:01:00Z",
            "status": "ok",
            "_has_defaulted_features": False,
            "feature_schema_sha256": schema_hash,
            "feature_vector_sha256": vector_hash,
        }
    )
    return row


def _prediction(uid: str, p1: str, p2: str, **overrides) -> dict:
    row = {
        "match_uid": uid,
        "match_date": "2026-07-10",
        "tournament": "Test Open",
        "p1": p1,
        "p2": p2,
        "logged_at": "2026-07-10T10:05:00Z",
        "odds_scraped_at": "2026-07-10T10:00:00Z",
        "match_start_time": "7/10/26 9:00 AM",
        "feature_snapshot_id": "",
        "latest_feature_snapshot_id": "",
        "feature_vector_sha256": "",
        "features_complete": True,
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "p1_odds_decimal": 1.8,
        "p2_odds_decimal": 2.1,
        "market_p1_prob": 0.54,
        "market_p2_prob": 0.46,
        "actual_winner": 1,
    }
    snapshot_label = str(overrides.get("feature_snapshot_id", "") or "")
    if snapshot_label:
        run_id = overrides.get("run_id") or f"run_{snapshot_label}"
        overrides["run_id"] = run_id
        overrides["feature_snapshot_id"] = build_feature_snapshot_id(
            uid, run_id, p1, p2
        )
    row.update(overrides)
    return row


def _fixture_production(tmp_path: Path) -> tuple[Path, replay_manifest.ReplayManifest]:
    production = tmp_path / "production"
    logs = production / "logs"
    logs.mkdir(parents=True)
    gold = _feature("feat_gold", "gold", "Gold P1", "Gold P2", 80.0)
    incomplete = _feature(
        "feat_incomplete", "exact_incomplete", "Incomplete P1", "Incomplete P2", 20.0
    )
    legacy = _feature("", "", "Legacy P1", "Legacy P2", 60.0)
    pd.DataFrame([gold, incomplete, legacy]).to_csv(
        logs / "features_20260710_100100.csv", index=False
    )
    predictions = [
        _prediction(
            "gold",
            "Gold P1",
            "Gold P2",
            feature_snapshot_id="feat_gold",
            feature_vector_sha256=gold["feature_vector_sha256"],
            actual_winner=1,
        ),
        _prediction(
            "exact_incomplete",
            "Incomplete P1",
            "Incomplete P2",
            feature_snapshot_id="feat_incomplete",
            feature_vector_sha256=incomplete["feature_vector_sha256"],
            p1_odds_decimal="",
            p2_odds_decimal="",
            actual_winner=2,
        ),
        _prediction("legacy", "Legacy P1", "Legacy P2", actual_winner=1),
        _prediction("none", "No P1", "No P2", actual_winner=1),
    ]
    pd.DataFrame(predictions).to_csv(production / "prediction_log.csv", index=False)
    evidence = replay_manifest.build_replay_manifest(production)
    return production, evidence


def _fake_predictor(family: str, transform) -> replay_models.ReplayPredictor:
    return replay_models.ReplayPredictor(
        family=family,
        version="v1.0.0",
        model_name=f"Fake {family}",
        model_sha256=("a" if family == "nn" else "b") * 64,
        feature_schema_id=FEATURE_SCHEMA_ID,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        training_feature_semantics_id="test_training@1.0.0",
        live_feature_semantics_id="test_live@1.0.0",
        probability_source="fake",
        predict_fn=lambda frame: transform(frame["Player1_Rank"].to_numpy(dtype=float)),
        artifact_files=[],
    )


def _predictors() -> list[replay_models.ReplayPredictor]:
    return [
        _fake_predictor("nn", lambda rank: rank / 100.0),
        _fake_predictor("xgboost", lambda rank: 1.0 - rank / 100.0),
    ]


def test_replays_verified_vectors_and_reports_metrics_by_model_and_tier(tmp_path):
    production, evidence = _fixture_production(tmp_path)

    result = replay_models.build_model_replay(
        production, evidence=evidence, predictors=_predictors()
    )

    assert len(result.frame) == 6
    assert set(result.frame["replay_tier"]) == {
        "GOLD_REPLAY",
        "EXACT_INCOMPLETE",
        "LEGACY_MATCHED",
    }
    assert "none" not in set(result.frame["match_uid"])
    assert result.frame.groupby("model_family")["match_uid"].nunique().to_dict() == {
        "nn": 3,
        "xgboost": 3,
    }
    assert result.frame["scorable"].all()
    assert len(result.metric_frame) == 6
    assert set(result.metric_frame["n"]) == {1}
    nn_metrics = result.metric_frame[result.metric_frame["model_family"] == "nn"]
    xgb_metrics = result.metric_frame[result.metric_frame["model_family"] == "xgboost"]
    assert set(nn_metrics["accuracy"]) == {1.0}
    assert set(xgb_metrics["accuracy"]) == {0.0}


def test_selected_vector_source_hash_change_fails_closed(tmp_path):
    production, evidence = _fixture_production(tmp_path)
    feature_path = production / "logs" / "features_20260710_100100.csv"
    feature_path.write_text(feature_path.read_text() + "\n")

    with pytest.raises(RuntimeError, match="changed after manifest build"):
        replay_models.build_model_replay(
            production, evidence=evidence, predictors=_predictors()[:1]
        )


def test_predictor_nonfinite_probability_fails_closed(tmp_path):
    production, evidence = _fixture_production(tmp_path)
    bad = _fake_predictor("nn", lambda rank: np.full(len(rank), np.nan))

    with pytest.raises(RuntimeError, match="non-finite probabilities"):
        replay_models.build_model_replay(
            production, evidence=evidence, predictors=[bad]
        )


def test_default_cli_is_read_only_with_injected_predictors(tmp_path, monkeypatch, capsys):
    production, _ = _fixture_production(tmp_path)
    before = sorted(path.relative_to(production) for path in production.rglob("*"))
    monkeypatch.setattr(replay_models, "load_promoted_predictors", _predictors)

    assert replay_models.main(["--prod-dir", str(production)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["replayable_match_counts"] == {
        "EXACT_INCOMPLETE": 1,
        "GOLD_REPLAY": 1,
        "LEGACY_MATCHED": 1,
    }
    assert sorted(path.relative_to(production) for path in production.rglob("*")) == before


def test_explicit_write_is_versioned_non_overwriting_and_hashed(tmp_path):
    production, evidence = _fixture_production(tmp_path)
    result = replay_models.build_model_replay(
        production, evidence=evidence, predictors=_predictors()[:1]
    )
    generated_at = datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc)

    destination = replay_models.write_model_replay(
        result, tmp_path / "outputs", generated_at=generated_at
    )

    assert destination.name == "model_replay_20260713T130000Z"
    metadata = json.loads((destination / replay_models.OUTPUT_MANIFEST_JSON).read_text())
    assert metadata["schema_version"] == "1.0.0"
    assert metadata["feature_schema_id"] == "base_141@1.0.0"
    assert metadata["artifacts"][replay_models.OUTPUT_REPLAY_CSV]["rows"] == 3
    assert len(metadata["artifacts"][replay_models.OUTPUT_REPLAY_CSV]["sha256"]) == 64
    assert metadata["artifacts"][replay_models.OUTPUT_METRICS_CSV]["rows"] == 3
    with pytest.raises(FileExistsError):
        replay_models.write_model_replay(
            result, tmp_path / "outputs", generated_at=generated_at
        )
