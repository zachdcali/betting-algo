import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import pytest

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

import storage.import_csv as import_csv  # noqa: E402
from storage.import_csv import _parser, _resolve_database_url, build_plan, main  # noqa: E402
from versioning import FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256  # noqa: E402


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fixture_prod(tmp_path: Path) -> tuple[Path, dict[str, float]]:
    prod = tmp_path / "production"
    (prod / "features").mkdir(parents=True)
    shutil.copy(PRODUCTION / "features" / "schema_141.json", prod / "features/schema_141.json")
    schema = json.loads((prod / "features/schema_141.json").read_text())
    feature_names = [item["name"] for item in sorted(schema["features"], key=lambda item: item["index"])]
    vector = {name: 0.0 for name in feature_names}
    vector.update({
        "Surface_Hard": 1.0,
        "Level_A": 1.0,
        "Round_R32": 1.0,
        "P1_Hand_U": 1.0,
        "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0,
        "P2_Country_Other": 1.0,
    })
    _write_csv(prod / "logs/audit/run_history.csv", [{
        "run_id": "run_1", "run_kind": "prediction_pipeline",
        "started_at": "2026-07-13T12:00:00Z", "completed_at": "2026-07-13T12:05:00Z",
        "status": "success", "prediction_rows_success": "1", "error_message": "",
    }])
    _write_csv(prod / "odds_history.csv", [{
        "odds_snapshot_uid": "odds_1", "logged_at": "2026-07-13T12:01:00Z",
        "run_id": "run_1", "match_uid": "match_1", "match_date": "2026-07-13",
        "tournament": "Test Open", "surface": "Hard", "level": "A", "round": "R32",
        "p1": "One", "p2": "Two", "odds_scraped_at": "2026-07-13T12:01:00Z",
        "market_p1_prob": "0.55", "market_p2_prob": "0.45",
        "p1_odds_american": "-120", "p2_odds_american": "110",
        "p1_odds_decimal": "1.83", "p2_odds_decimal": "2.10",
    }])
    _write_csv(prod / "logs/feature_vectors.csv", [{
        "p1": "One", "p2": "Two", "match_date": "2026-07-13",
        "logged_at": "2026-07-13T12:02:00Z", "run_id": "run_1",
        "match_uid": "match_1", "feature_snapshot_id": "feature_1",
        "build_status": "ok", "features_complete": "True",
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_vector_sha256": "", "feature_count": "141",
        "features_json": json.dumps(vector),
    }])
    prediction = {
        "prediction_uid": "prediction_1", "logged_at": "2026-07-13T12:03:00Z",
        "run_id": "run_1", "match_uid": "match_1", "feature_snapshot_id": "feature_1",
        "model_p1_prob": "0.61", "model_p2_prob": "0.39",
        "xgb_p1_prob": "0.58", "xgb_p2_prob": "0.42",
        "rf_p1_prob": "", "rf_p2_prob": "",
        "primary_model_family": "nn", "model_version": "1.2.1",
        "nn_model_version": "1.2.1", "xgb_model_version": "1.0.0",
        "rf_model_version": "", "logging_quality": "snapshot_v2",
        "actual_winner": "", "score": "", "settled_at": "",
    }
    _write_csv(prod / "prediction_snapshots.csv", [prediction])
    settled = dict(prediction)
    settled.update({"actual_winner": "1", "score": "6-4 6-4", "settled_at": "2026-07-13T15:00:00Z"})
    _write_csv(prod / "prediction_log.csv", [settled])
    _write_csv(prod / "logs/performance_v1_shadow_predictions.csv", [{
        "shadow_prediction_uid": "shadow_forward_1",
        "logged_at": "2026-07-13T12:04:00Z", "run_id": "run_1",
        "match_uid": "match_1", "feature_snapshot_id": "feature_1",
        "model_family": "catboost", "model_version": "cat_exact_version",
        "feature_set": "performance_v1", "n_features": "198",
        "shadow_p1_prob": "0.57", "shadow_p2_prob": "0.43",
        "performance_features_available": "True", "shadow_status": "success",
        "shadow_error": "",
    }])
    _write_csv(prod / "logs/performance_v1_shadow_backfill.csv", [{
        "shadow_prediction_uid": "shadow_backfill_1",
        "logged_at": "2026-07-13T13:00:00Z", "run_id": "run_legacy",
        "match_uid": "match_legacy", "feature_snapshot_id": "feature_legacy",
        "model_family": "xgboost", "model_version": "xgb_exact_version",
        "feature_set": "performance_v1", "n_features": "198",
        "shadow_p1_prob": "", "shadow_p2_prob": "",
        "performance_features_available": "False", "shadow_status": "error",
        "shadow_error": "missing historical stats", "backfill_source": "historical_replay",
        "backfill_quality": "legacy_backfilled", "source_feature_file": "features_old.csv",
        "prediction_logged_at": "2026-07-01T10:00:00Z",
    }])
    _write_csv(prod / "logs/betting_sessions.csv", [{
        "session_id": "session_1", "start_time": "2026-07-13T12:00:00Z",
        "end_time": "2026-07-13T15:00:00Z", "initial_bankroll": "1000.00",
        "final_bankroll": "1012.50", "total_bets_placed": "2",
        "total_staked": "25.50", "total_profit_loss": "12.50",
        "win_rate": "0.5", "avg_odds": "2.1", "avg_edge": "0.04",
        "kelly_multiplier_used": "0.18", "notes": "fixture session",
    }])
    _write_csv(prod / "logs/bankroll_history.csv", [{
        "timestamp": "2026-07-13T15:00:00Z", "session_id": "session_1",
        "bankroll": "1012.50", "change_amount": "12.50",
        "change_reason": "bet_settlement", "total_staked": "25.50",
        "num_pending_bets": "0", "num_settled_bets": "2",
    }])
    return prod, vector


def _all_records(plan, table):
    return [record for batch in plan.batches if batch.table == table for record in batch.unique_records]


def _write_model_registry(prod: Path):
    complete_contract = {
        "features": 141,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "training_feature_semantics_id": "sackmann_historical_legacy@1.0.0",
        "live_feature_semantics_id": "ta_live_legacy@3.0.0",
        "training_dataset_id": "fixture_dataset@1.0.0",
    }
    payload = {
        "registry_schema_version": "2.0.0",
        "registry_generation": 7,
        "registry_effective_at": "2026-07-14T00:00:00Z",
        "current_version": "v1.2.1",
        "models": {
            "v1.1.0": {
                "name": "archived-incomplete",
                "archived": True,
                "model_file": "archive/nn/v1.1.0/model.pth",
            },
            "v1.2.1": {
                **complete_contract,
                "name": "promoted-nn",
                "model_file": "releases/nn/v1.2.1/model.pth",
                "model_sha256": "a" * 64,
                "scaler_file": "releases/nn/v1.2.1/scaler.pkl",
                "scaler_sha256": "b" * 64,
            },
        },
        "candidate_version": "v1.3.0",
        "candidates": {
            "v1.3.0": {
                "name": "candidate-incomplete",
                "model_file": "candidates/nn/v1.3.0/model.pth",
            },
        },
        "xgboost": {
            "current_version": "v1.0.0",
            "models": {
                "v1.0.0": {
                    **complete_contract,
                    "name": "promoted-xgb",
                    "model_file": "releases/xgboost/v1.0.0/model.json",
                    "model_sha256": "c" * 64,
                },
            },
        },
        "random_forest": {
            "current_version": "v1.0.0",
            "models": {
                "v1.0.0": {
                    **complete_contract,
                    "name": "promoted-rf",
                    "model_file": "releases/random_forest/v1.0.0/model.pkl",
                    "model_sha256": "d" * 64,
                },
            },
        },
    }
    path = prod / "models/model_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def test_plan_is_deterministic_typed_and_never_mutates_sources(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    before = {
        path.relative_to(prod).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in prod.rglob("*.csv")
    }

    first = build_plan(prod, include_run_feature_files=False)
    second = build_plan(prod, include_run_feature_files=False)
    after = {
        path.relative_to(prod).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in prod.rglob("*.csv")
    }

    assert first.import_batch_id == second.import_batch_id
    assert first.row_counts == second.row_counts
    assert before == after
    assert first.row_counts["ops.pipeline_runs"] == 1
    assert first.row_counts["ops.odds_observations"] == 1
    assert first.row_counts["raw.source_fetches"] == 1
    assert first.row_counts["ml.feature_snapshots"] == 1
    assert first.row_counts["ml.prediction_observations"] == 4
    assert first.row_counts["ops.settlement_events"] == 1
    assert first.row_counts["ops.paper_sessions"] == 1
    assert first.row_counts["ops.account_ledger"] == 1

    odds = _all_records(first, "ops.odds_observations")[0]
    assert str(odds["player1_decimal_odds"]) == "1.83"
    assert odds["source_file"] == "odds_history.csv"
    assert odds["source_row_number"] == 2
    assert len(odds["source_row_sha256"]) == 64
    assert json.loads(odds["source_row_json"])["match_uid"] == "match_1"


def test_feature_import_uses_stable_version_contract_and_exact_vector_hash(tmp_path):
    prod, vector = _fixture_prod(tmp_path)
    plan = build_plan(prod, include_run_feature_files=False)
    schema = _all_records(plan, "ml.feature_schemas")[0]
    snapshot = _all_records(plan, "ml.feature_snapshots")[0]
    names = json.loads(schema["feature_names"])
    expected_payload = json.dumps([[name, float(vector[name])] for name in names], separators=(",", ":"))

    assert schema["schema_identifier"] == FEATURE_SCHEMA_ID
    assert schema["schema_sha256"] == FEATURE_SCHEMA_SHA256
    assert snapshot["feature_schema_identifier"] == FEATURE_SCHEMA_ID
    assert snapshot["features_complete"] is True
    assert snapshot["feature_count"] == 141
    assert snapshot["lineage_quality"] == "exact_feature_snapshot_id"
    assert snapshot["feature_vector_sha256"] == sha256(expected_payload.encode()).hexdigest()


def test_feature_import_marks_generated_legacy_source_row_identity(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    path = prod / "logs/feature_vectors.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        original = next(csv.DictReader(handle))
    legacy = dict(original)
    legacy.update({
        "feature_snapshot_id": "", "match_uid": "", "run_id": "run_legacy",
        "p1": "Legacy One", "p2": "Legacy Two",
    })
    _write_csv(path, [original, legacy])

    snapshots = _all_records(
        build_plan(prod, include_run_feature_files=False), "ml.feature_snapshots"
    )
    assert {row["lineage_quality"] for row in snapshots} == {
        "exact_feature_snapshot_id", "legacy_source_row_snapshot",
    }
    generated = next(
        row for row in snapshots if row["lineage_quality"] == "legacy_source_row_snapshot"
    )
    assert generated["external_feature_snapshot_id"].startswith("feature_snapshot:")


def test_structurally_invalid_declared_complete_feature_is_demoted(tmp_path):
    prod, vector = _fixture_prod(tmp_path)
    for name in tuple(vector):
        if name.startswith("Round_"):
            vector[name] = 0.0
    path = prod / "logs/feature_vectors.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["features_json"] = json.dumps(vector)
    _write_csv(path, [row])

    snapshot = _all_records(
        build_plan(prod, include_run_feature_files=False), "ml.feature_snapshots"
    )[0]

    assert snapshot["features_complete"] is False
    assert snapshot["feature_vector_sha256"] is None
    assert "structural:one_hot_cardinality:round:0" in json.loads(
        snapshot["defaulted_features"]
    )


def test_prediction_import_is_one_observation_per_available_model_family(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    predictions = _all_records(
        build_plan(prod, include_run_feature_files=False),
        "ml.prediction_observations",
    )

    operational = [row for row in predictions if row["model_role"] != "shadow"]
    assert {(row["model_family"], row["model_role"]) for row in operational} == {
        ("nn", "promoted"), ("xgboost", "companion")
    }
    assert len({row["idempotency_key"] for row in predictions}) == 4


def test_model_registry_imports_versioned_releases_with_contract_status(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    registry = _write_model_registry(prod)
    plan = build_plan(prod, include_run_feature_files=False)
    releases = _all_records(plan, "ml.model_releases")
    generations = _all_records(plan, "ml.model_registry_generations")
    status_events = _all_records(plan, "ml.model_release_status_events")
    by_identity = {
        (row["model_family"], row["model_version"]): row for row in releases
    }
    statuses = {
        row["model_release_key"]: row["release_status"] for row in status_events
    }

    assert "models/model_registry.json" in plan.file_sha256
    assert len(generations) == 1
    assert generations[0]["generation_sequence"] == 7
    assert generations[0]["registry_generation_sha256"] == (
        plan.file_sha256["models/model_registry.json"]
    )
    assert len(releases) == 5
    assert {row["release_status"] for row in releases} == {"registered"}
    assert statuses["model_release:nn:1.1.0"] == "archived"
    assert statuses["model_release:nn:1.2.1"] == "promoted"
    assert statuses["model_release:nn:1.3.0"] == "candidate"
    assert statuses["model_release:xgboost:1.0.0"] == "promoted"
    assert statuses["model_release:random_forest:1.0.0"] == "promoted"
    assert {
        (row["model_release_key"], row["model_family"])
        for row in status_events
    } == {
        ("model_release:nn:1.1.0", "nn"),
        ("model_release:nn:1.2.1", "nn"),
        ("model_release:nn:1.3.0", "nn"),
        ("model_release:xgboost:1.0.0", "xgboost"),
        ("model_release:random_forest:1.0.0", "random_forest"),
    }
    assert by_identity[("nn", "1.2.1")]["contract_complete"] is True
    assert by_identity[("xgboost", "1.0.0")]["contract_complete"] is True
    assert by_identity[("random_forest", "1.0.0")]["contract_complete"] is True
    assert by_identity[("nn", "1.1.0")]["contract_complete"] is False
    assert json.loads(by_identity[("nn", "1.2.1")]["registry_entry"]) == (
        registry["models"]["v1.2.1"]
    )
    source = json.loads(by_identity[("nn", "1.2.1")]["source_row_json"])
    assert source["family"] == "nn"
    assert source["registry_version"] == "v1.2.1"

    artifact = next(
        row for row in _all_records(plan, "raw.source_artifacts")
        if row["source_file"] == "models/model_registry.json"
    )
    assert artifact["artifact_kind"] == "model_registry_import_source"
    assert artifact["storage_uri"] == "repository-file://models/model_registry.json"


def _complete_contract_entry(*, family: str = "nn") -> dict:
    entry = {
        "model_file": "releases/model.bin",
        "model_sha256": "a" * 64,
        "features": 141,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "training_feature_semantics_id": "historical@1.0.0",
        "live_feature_semantics_id": "live@1.0.0",
        "training_dataset_id": "dataset@1.0.0",
        "probability_mode": "raw",
    }
    if family == "nn":
        entry.update({
            "scaler_file": "releases/scaler.pkl",
            "scaler_sha256": "b" * 64,
        })
    return entry


def test_importer_model_contract_completeness_is_fail_closed():
    raw_nn = _complete_contract_entry()
    assert import_csv._model_contract_complete("nn", "2.0.0", raw_nn)

    calibrated_nn = {
        **raw_nn,
        "probability_mode": "calibrated",
        "calibrated_model_file": "releases/calibrated.pkl",
        "calibrated_model_sha256": "c" * 64,
        "calibration_version": "v1.0.0",
    }
    assert import_csv._model_contract_complete(
        "nn", "2.0.0", calibrated_nn
    )

    malformed = []
    for field in ("model_file", "model_sha256", "scaler_file", "scaler_sha256"):
        entry = dict(raw_nn)
        entry.pop(field)
        malformed.append(entry)
    malformed.extend((
        {**raw_nn, "features": 141.0},
        {**raw_nn, "features": True},
        {**raw_nn, "artifact_available": False},
        {**raw_nn, "artifact_available": "true"},
        {**raw_nn, "probability_mode": "automatic"},
        {
            **raw_nn,
            "probability_mode": "calibrated",
            "calibrated_model_sha256": "c" * 64,
            "calibration_version": "v1.0.0",
        },
        {
            **raw_nn,
            "probability_mode": "calibrated",
            "calibrated_model_file": "releases/calibrated.pkl",
            "calibration_version": "v1.0.0",
        },
        {
            **raw_nn,
            "probability_mode": "calibrated",
            "calibrated_model_file": "releases/calibrated.pkl",
            "calibrated_model_sha256": "c" * 64,
        },
        {
            **raw_nn,
            "probability_mode": "calibrated",
            "calibrated_model_file": "releases/calibrated.pkl",
            "calibrated_model_sha256": "c" * 64,
            "calibration_version": "final",
        },
    ))

    assert all(
        not import_csv._model_contract_complete("nn", "2.0.0", entry)
        for entry in malformed
    )
    assert not import_csv._model_contract_complete("nn", "final", raw_nn)
    assert not import_csv._model_contract_complete(
        "bogus", "2.0.0", raw_nn
    )
    assert not import_csv._model_contract_complete(
        "xgboost",
        "2.0.0",
        {**_complete_contract_entry(family="xgboost"), "probability_mode": "calibrated"},
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "registry_schema_version",
        "current_version",
        "candidate_version",
        "release_key",
    ),
)
def test_registry_import_rejects_ad_hoc_versions(tmp_path, mutation):
    prod, _ = _fixture_prod(tmp_path)
    registry = _write_model_registry(prod)
    if mutation == "registry_schema_version":
        registry["registry_schema_version"] = "final"
    elif mutation == "current_version":
        registry["current_version"] = "final"
    elif mutation == "candidate_version":
        registry["candidate_version"] = "next"
    else:
        entry = registry["models"].pop("v1.2.1")
        registry["models"]["final"] = entry
        registry["current_version"] = "final"
    (prod / "models/model_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="valid SemVer"):
        build_plan(prod, include_run_feature_files=False)


def test_model_registry_rejects_current_version_omitted_from_family(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    registry = _write_model_registry(prod)
    registry["current_version"] = "v9.9.9"
    path = prod / "models/model_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "model registry nn.current_version '9.9.9' must identify "
            "exactly one models release"
        ),
    ):
        build_plan(prod, include_run_feature_files=False)


def test_model_registry_requires_current_version_for_every_live_family(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    registry = _write_model_registry(prod)
    registry["random_forest"].pop("current_version")
    path = prod / "models/model_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="model registry random_forest.current_version is required",
    ):
        build_plan(prod, include_run_feature_files=False)


def test_operational_predictions_bind_registry_release_but_shadows_stay_null(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    _write_model_registry(prod)
    predictions = _all_records(
        build_plan(prod, include_run_feature_files=False),
        "ml.prediction_observations",
    )
    operational = {
        row["model_family"]: row
        for row in predictions
        if row["model_role"] != "shadow"
    }
    shadows = [row for row in predictions if row["model_role"] == "shadow"]

    assert operational["nn"]["model_release_key"] == "model_release:nn:1.2.1"
    assert operational["xgboost"]["model_release_key"] == (
        "model_release:xgboost:1.0.0"
    )
    assert shadows
    assert all(row["model_release_key"] is None for row in shadows)


def test_conflicting_external_ids_are_quarantined_not_first_wins(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    path = prod / "prediction_log.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["match_uid"] = "shifted_match_identity"
    row["logged_at"] = "2026-07-14T12:03:00Z"
    _write_csv(path, [row])

    plan = build_plan(prod, include_run_feature_files=False)
    conflicts = _all_records(plan, "ops.import_conflicts")
    accepted = _all_records(plan, "ml.prediction_observations")

    assert {row["conflict_type"] for row in conflicts} == {
        "conflicting_external_prediction"
    }
    assert len(conflicts) == 4
    assert not any(
        row["external_prediction_id"] == "prediction_1"
        and row["model_role"] != "shadow"
        for row in accepted
    )


def test_batch_identity_includes_normalizer_contract_and_target_manifest(
    tmp_path, monkeypatch,
):
    prod, _ = _fixture_prod(tmp_path)
    first = build_plan(prod, include_run_feature_files=False)
    monkeypatch.setattr(import_csv, "OPERATIONAL_NORMALIZER_VERSION", "1.0.1")
    second = build_plan(prod, include_run_feature_files=False)

    assert first.file_sha256 == second.file_sha256
    assert first.import_batch_id != second.import_batch_id
    first_control = _all_records(first, "ops.import_batches")[0]
    manifest = json.loads(first_control["source_manifest"])
    assert manifest["normalizer_contract_version"] == "1.0.0"
    assert len(manifest["target_manifest_sha256"]) == 64


def test_unambiguous_legacy_foreign_keys_are_materialized(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    plan = build_plan(prod, include_run_feature_files=False)
    predictions = [
        row for row in _all_records(plan, "ml.prediction_observations")
        if row["model_role"] != "shadow"
    ]
    account = _all_records(plan, "ops.paper_accounts")[0]

    assert predictions
    assert all(row.get("feature_snapshot_id") for row in predictions)
    assert all(row.get("run_id") for row in predictions)
    assert str(account["starting_capital"]) == "1000.00"


def test_shadow_import_preserves_family_version_failure_and_backfill_metadata(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    first = _all_records(
        build_plan(prod, include_run_feature_files=False),
        "ml.prediction_observations",
    )
    second = _all_records(
        build_plan(prod, include_run_feature_files=False),
        "ml.prediction_observations",
    )
    shadows = [row for row in first if row["model_role"] == "shadow"]

    assert len(shadows) == 2
    assert len({row["idempotency_key"] for row in shadows}) == 2
    assert [row["idempotency_key"] for row in first] == [
        row["idempotency_key"] for row in second
    ]
    by_source = {json.loads(row["metadata"])["shadow_source"]: row for row in shadows}

    forward = by_source["forward"]
    assert forward["external_prediction_id"] == "shadow_forward_1"
    assert forward["model_family"] == "catboost"
    assert forward["model_version"] == "cat_exact_version"
    assert str(forward["player1_probability"]) == "0.57"
    assert json.loads(forward["metadata"]) == {
        "backfill_quality": None,
        "backfill_source": None,
        "feature_count": 198,
        "feature_set": "performance_v1",
        "performance_features_available": True,
        "prediction_logged_at": None,
        "shadow_error": None,
        "shadow_source": "forward",
        "shadow_status": "success",
        "source_feature_file": None,
    }

    backfill = by_source["backfill"]
    assert backfill["external_prediction_id"] == "shadow_backfill_1"
    assert backfill["model_family"] == "xgboost"
    assert backfill["model_version"] == "xgb_exact_version"
    assert backfill["player1_probability"] is None
    assert backfill["player2_probability"] is None
    assert backfill["logging_quality"] == "legacy_backfilled"
    metadata = json.loads(backfill["metadata"])
    assert metadata["shadow_status"] == "error"
    assert metadata["shadow_error"] == "missing historical stats"
    assert metadata["backfill_quality"] == "legacy_backfilled"


def test_paper_sessions_are_typed_operational_records(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    sessions = _all_records(
        build_plan(prod, include_run_feature_files=False), "ops.paper_sessions"
    )

    assert len(sessions) == 1
    session = sessions[0]
    assert session["idempotency_key"] == "paper_session:session_1"
    assert session["external_session_id"] == "session_1"
    assert session["started_at"].isoformat() == "2026-07-13T12:00:00+00:00"
    assert session["completed_at"].isoformat() == "2026-07-13T15:00:00+00:00"
    assert session["total_bets"] == 2
    assert str(session["initial_balance"]) == "1000.00"
    assert str(session["final_balance"]) == "1012.50"
    assert str(session["kelly_multiplier"]) == "0.18"


def test_account_ledger_comes_from_bankroll_history_not_session_projection(tmp_path):
    prod, _ = _fixture_prod(tmp_path)
    ledger = _all_records(
        build_plan(prod, include_run_feature_files=False), "ops.account_ledger"
    )

    assert len(ledger) == 1
    assert ledger[0]["source_file"] == "logs/bankroll_history.csv"
    assert ledger[0]["reason"] == "bet_settlement"
    assert str(ledger[0]["amount"]) == "12.50"
    assert str(ledger[0]["balance_after"]) == "1012.50"


def test_cli_defaults_to_read_only_plan_and_apply_requires_url(tmp_path, capsys):
    prod, _ = _fixture_prod(tmp_path)

    assert main(["--prod-dir", str(prod), "--skip-run-feature-files"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["mode"] == "plan"
    assert planned["total_rows"] > 0

    assert main([
        "--prod-dir", str(prod), "--skip-run-feature-files", "--apply",
    ]) == 2
    assert "--database-url or --database-url-env" in capsys.readouterr().err


def test_database_url_environment_lookup_is_explicit_and_secret_safe(monkeypatch):
    parser = _parser()
    monkeypatch.setenv("OPERATIONAL_DATABASE_URL", "postgresql://example/test")

    args = parser.parse_args([
        "--apply", "--database-url-env", "OPERATIONAL_DATABASE_URL",
    ])
    assert _resolve_database_url(args) == ("postgresql://example/test", None)

    both = parser.parse_args([
        "--apply", "--database-url", "postgresql://direct/test",
        "--database-url-env", "OPERATIONAL_DATABASE_URL",
    ])
    _, error = _resolve_database_url(both)
    assert "only one" in error
