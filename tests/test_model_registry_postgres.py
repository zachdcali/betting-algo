from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from storage.records import RecordBatch, canonical_json  # noqa: E402
from storage.repository import OperationalRepository  # noqa: E402


DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
MIGRATIONS = (
    ROOT / "supabase/migrations/20260714010000_operational_schema_v1.sql",
    ROOT / "supabase/migrations/20260714020000_operational_integrity_v1_1.sql",
)


def _test_url() -> str:
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not configured")
    parsed = urlparse(DATABASE_URL)
    if (
        parsed.hostname not in {"127.0.0.1", "localhost", "postgres"}
        and os.environ.get("ALLOW_REMOTE_DB_TEST") != "1"
    ):
        pytest.fail("refusing a registry integration test against a remote database")
    return DATABASE_URL


def _ensure_migrations(connection) -> None:
    version_table = connection.execute(
        "SELECT to_regclass('ops.schema_versions')"
    ).fetchone()[0]
    applied = set()
    if version_table is not None:
        applied = {
            row[0]
            for row in connection.execute(
                "SELECT version FROM ops.schema_versions"
            ).fetchall()
        }
    for path in MIGRATIONS:
        version = "1.0.0" if path.name.endswith("schema_v1.sql") else "1.1.0"
        if version not in applied:
            connection.execute(path.read_text(encoding="utf-8"), prepare=False)
            applied.add(version)


def _release(
    *, family: str, version: str, schema_identifier: str | None = None,
    schema_sha256: str | None = None, semantics_identifier: str | None = None,
    feature_count: int | None = None,
) -> RecordBatch:
    contract_complete = all((
        schema_identifier, schema_sha256, semantics_identifier, feature_count,
    ))
    model_sha256 = "a" * 64 if contract_complete else None
    scaler_sha256 = "b" * 64 if contract_complete and family == "nn" else None
    registry_entry = {
        "model_file": f"releases/{family}/{version}/model.bin",
        "model_sha256": model_sha256,
        "feature_schema_id": schema_identifier,
        "feature_schema_sha256": schema_sha256,
        "features": feature_count,
        "training_feature_semantics_id": "registry_fixture_training@1.0.0",
        "live_feature_semantics_id": semantics_identifier,
        "training_dataset_id": "registry_fixture_dataset@1.0.0",
        "probability_mode": "raw",
    }
    if family == "nn":
        registry_entry.update({
            "scaler_file": f"releases/{family}/{version}/scaler.pkl",
            "scaler_sha256": scaler_sha256,
        })
    return RecordBatch.from_records("ml.model_releases", [{
        "idempotency_key": f"model_release:{family}:{version}",
        "model_family": family,
        "model_version": version,
        "release_status": "registered",
        "registry_schema_version": "2.0.0",
        "feature_schema_identifier": schema_identifier,
        "feature_schema_sha256": schema_sha256,
        "feature_count": feature_count,
        "training_feature_semantics_id": "registry_fixture_training@1.0.0",
        "live_feature_semantics_id": semantics_identifier,
        "training_dataset_id": "registry_fixture_dataset@1.0.0",
        "model_sha256": model_sha256,
        "scaler_sha256": scaler_sha256,
        "registry_entry": canonical_json(registry_entry),
        "contract_complete": contract_complete,
    }])


def _generation(*, sequence: int, digest: str) -> RecordBatch:
    return RecordBatch.from_records("ml.model_registry_generations", [{
        "idempotency_key": f"model_registry_generation:{digest}",
        "registry_generation_sha256": digest,
        "generation_sequence": sequence,
        "registry_schema_version": "2.0.0",
        "effective_at": "2026-07-14T00:00:00Z",
    }])


def _status(
    *, family: str, version: str, digest: str, status: str = "promoted"
) -> RecordBatch:
    return RecordBatch.from_records("ml.model_release_status_events", [{
        "idempotency_key": (
            f"model_release_status:{digest}:{family}:{version}"
        ),
        "model_release_key": f"model_release:{family}:{version}",
        "model_family": family,
        "registry_generation_sha256": digest,
        "release_status": status,
    }])


def test_database_enforces_model_release_contract_completeness_and_semver():
    import psycopg

    url = _test_url()
    with psycopg.connect(url, autocommit=True) as connection:
        _ensure_migrations(connection)

    schema_identifier = "db_contract_fixture@1.0.0"
    schema_sha256 = sha256(schema_identifier.encode("utf-8")).hexdigest()

    def _row(*, family="xgboost", version="1.0.0"):
        model_sha256 = "a" * 64
        scaler_sha256 = "b" * 64 if family == "nn" else None
        entry = {
            "model_file": "releases/model.bin",
            "model_sha256": model_sha256,
            "feature_schema_id": schema_identifier,
            "feature_schema_sha256": schema_sha256,
            "features": 141,
            "training_feature_semantics_id": "historical@1.0.0",
            "live_feature_semantics_id": "live@1.0.0",
            "training_dataset_id": "dataset@1.0.0",
            "probability_mode": "raw",
        }
        if family == "nn":
            entry.update({
                "scaler_file": "releases/scaler.pkl",
                "scaler_sha256": scaler_sha256,
            })
        return {
            "idempotency_key": f"model_release:db_contract:{uuid4()}",
            "model_family": family,
            "model_version": version,
            "release_status": "registered",
            "registry_schema_version": "2.0.0",
            "feature_schema_identifier": schema_identifier,
            "feature_schema_sha256": schema_sha256,
            "feature_count": 141,
            "training_feature_semantics_id": "historical@1.0.0",
            "live_feature_semantics_id": "live@1.0.0",
            "training_dataset_id": "dataset@1.0.0",
            "model_sha256": model_sha256,
            "scaler_sha256": scaler_sha256,
            "calibrator_sha256": None,
            "registry_entry": entry,
            "contract_complete": True,
        }

    invalid_rows = []
    invalid_rows.append(_row(version="final"))

    missing_path = _row()
    missing_path["registry_entry"].pop("model_file")
    invalid_rows.append(missing_path)

    missing_json_binding = _row()
    missing_json_binding["registry_entry"].pop("model_sha256")
    invalid_rows.append(missing_json_binding)

    non_boolean_availability = _row()
    non_boolean_availability["registry_entry"]["artifact_available"] = "true"
    invalid_rows.append(non_boolean_availability)

    non_integer_width = _row()
    non_integer_width["registry_entry"]["features"] = 141.0
    invalid_rows.append(non_integer_width)

    unknown_family = _row(family="bogus")
    invalid_rows.append(unknown_family)

    missing_nn_scaler = _row(family="nn")
    missing_nn_scaler["registry_entry"].pop("scaler_file")
    invalid_rows.append(missing_nn_scaler)

    incomplete_calibration = _row(family="nn")
    incomplete_calibration["registry_entry"]["probability_mode"] = "calibrated"
    invalid_rows.append(incomplete_calibration)

    missing_calibration_version = _row(family="nn", version="1.0.1")
    missing_calibration_version["registry_entry"].update({
        "probability_mode": "calibrated",
        "calibrated_model_file": "releases/calibrated.pkl",
        "calibrated_model_sha256": "c" * 64,
    })
    missing_calibration_version["calibrator_sha256"] = "c" * 64
    invalid_rows.append(missing_calibration_version)

    with psycopg.connect(url) as connection:
        repository = OperationalRepository(connection)
        valid_calibration = _row(family="nn", version="9.9.9")
        valid_calibration["registry_entry"].update({
            "probability_mode": "calibrated",
            "calibrated_model_file": "releases/calibrated.pkl",
            "calibrated_model_sha256": "c" * 64,
            "calibration_version": "v1.0.0",
        })
        valid_calibration["calibrator_sha256"] = "c" * 64
        valid_calibration["calibration_version"] = "1.0.0"
        valid_calibration["registry_entry"] = canonical_json(
            valid_calibration["registry_entry"]
        )
        latest_sequence = connection.execute(
            "SELECT coalesce(max(generation_sequence), 0) "
            "FROM ml.model_registry_generations"
        ).fetchone()[0]
        calibration_sequence = int(latest_sequence) + 1
        calibration_digest = sha256(
            f"calibration-block:{uuid4()}".encode("utf-8")
        ).hexdigest()
        repository.write_batch(_generation(
            sequence=calibration_sequence,
            digest=calibration_digest,
        ))
        with pytest.raises(
            psycopg.Error,
            match="calibrated NN promotion is blocked",
        ):
            with connection.transaction():
                repository.write_batch(RecordBatch.from_records(
                    "ml.model_release_status_events",
                    [{
                        "idempotency_key": (
                            "model_release_status:calibration_block:"
                            f"{uuid4()}"
                        ),
                        "model_release_key": valid_calibration[
                            "idempotency_key"
                        ],
                        "model_family": "nn",
                        "registry_generation_sha256": calibration_digest,
                        "release_status": "promoted",
                    }],
                ))
                # Import batches write status rows before release rows. The
                # deferred promotion guard must inspect the release only after
                # both facts exist, not silently pass on the first INSERT.
                repository.write_batch(RecordBatch.from_records(
                    "ml.model_releases", [valid_calibration]
                ))
                connection.execute(
                    "SET CONSTRAINTS "
                    "model_release_calibrated_promotion_guard IMMEDIATE"
                )
        for row in invalid_rows:
            db_row = deepcopy(row)
            db_row["registry_entry"] = canonical_json(db_row["registry_entry"])
            with pytest.raises(
                psycopg.Error,
                match=(
                    "model_releases_model_version_semver_check|"
                    "model_releases_family_check|"
                    "model_releases_contract_complete_check"
                ),
            ):
                with connection.transaction():
                    repository.write_batch(RecordBatch.from_records(
                        "ml.model_releases", [db_row]
                    ))
        connection.rollback()


def test_new_generation_omission_revokes_current_release_and_family_is_unique():
    import psycopg

    url = _test_url()
    with psycopg.connect(url, autocommit=True) as connection:
        _ensure_migrations(connection)

    # Keep every fixture fact in one rollback-only transaction. Registry
    # generation authority is global, so leaking a higher test generation
    # would make an otherwise isolated second suite run order-dependent.
    with psycopg.connect(url) as connection:
        latest = connection.execute(
            "SELECT coalesce(max(generation_sequence), 0) "
            "FROM ml.model_registry_generations"
        ).fetchone()[0]
        first_sequence = int(latest) + 1
        second_sequence = first_sequence + 1
        first_digest = sha256(
            f"registry-omission:{first_sequence}".encode("utf-8")
        ).hexdigest()
        second_digest = sha256(
            f"registry-omission:{second_sequence}".encode("utf-8")
        ).hexdigest()
        omitted_family = "xgboost"
        other_family = "random_forest"
        omitted_version = f"0.0.{first_sequence}"
        duplicate_version = f"0.1.{first_sequence}"
        other_version = f"0.2.{first_sequence}"
        schema_identifier = f"registry_omission@{first_sequence}"
        schema_sha256 = sha256(schema_identifier.encode("utf-8")).hexdigest()
        semantics_identifier = f"registry_omission_live@{first_sequence}"

        repository = OperationalRepository(connection)
        repository.write_batch(RecordBatch.from_records(
            "ml.feature_schemas",
            [{
                "idempotency_key": f"feature_schema:{schema_identifier}",
                "schema_name": "registry_omission_fixture",
                "schema_version": "1.0.0",
                "schema_identifier": schema_identifier,
                "schema_sha256": schema_sha256,
                "feature_count": 1,
                "feature_names": canonical_json(["fixture_feature"]),
                "feature_contract": canonical_json({}),
            }],
        ))
        repository.write_batch(_release(
            family=omitted_family,
            version=omitted_version,
            schema_identifier=schema_identifier,
            schema_sha256=schema_sha256,
            semantics_identifier=semantics_identifier,
            feature_count=1,
        ))
        repository.write_batch(_release(
            family=omitted_family, version=duplicate_version
        ))
        repository.write_batch(_release(
            family=other_family, version=other_version
        ))
        repository.write_batch(
            _generation(sequence=first_sequence, digest=first_digest)
        )
        repository.write_batch(_status(
            family=omitted_family,
            version=omitted_version,
            digest=first_digest,
        ))
        repository.write_batch(RecordBatch.from_records(
            "ml.feature_snapshots",
            [{
                "idempotency_key": (
                    f"feature_snapshot:registry_omission:{first_sequence}"
                ),
                "external_feature_snapshot_id": (
                    f"registry_omission_{first_sequence}"
                ),
                "feature_schema_identifier": schema_identifier,
                "feature_schema_sha256": schema_sha256,
                "feature_semantics_identifier": semantics_identifier,
                "captured_at": "2026-07-14T00:00:01Z",
                "build_status": "ok",
                "features_complete": True,
                "lineage_quality": "exact_feature_snapshot_id",
                "feature_count": 1,
                "feature_vector_sha256": "b" * 64,
                "feature_vector": canonical_json({"fixture_feature": 0}),
                "defaulted_features": canonical_json([]),
            }],
        ))
        feature_snapshot_id = connection.execute(
            "SELECT feature_snapshot_id FROM ml.feature_snapshots "
            "WHERE idempotency_key = %s",
            (f"feature_snapshot:registry_omission:{first_sequence}",),
        ).fetchone()[0]
        feature_snapshot_id = str(feature_snapshot_id)

        def eligible_prediction(suffix: str) -> RecordBatch:
            return RecordBatch.from_records("ml.prediction_observations", [{
                "idempotency_key": (
                    f"prediction:registry_omission:{first_sequence}:{suffix}"
                ),
                "external_prediction_id": (
                    f"registry_omission_{first_sequence}_{suffix}"
                ),
                "feature_snapshot_id": feature_snapshot_id,
                "external_feature_snapshot_id": (
                    f"registry_omission_{first_sequence}"
                ),
                "predicted_at": "2026-07-14T00:00:02Z",
                "model_family": omitted_family,
                "model_version": omitted_version,
                "model_role": "promoted",
                "player1_probability": "0.60",
                "player2_probability": "0.40",
                "logging_schema_version": "prediction_log_v2",
                "logging_quality": "snapshot_v2",
                "metadata": canonical_json({}),
                "model_release_key": (
                    f"model_release:{omitted_family}:{omitted_version}"
                ),
                "decision_eligible": True,
            }])

        repository.write_batch(eligible_prediction("before_omission"))
        assert connection.execute(
            "SELECT release_status FROM api.current_model_releases "
            "WHERE model_release_key = %s",
            (f"model_release:{omitted_family}:{omitted_version}",),
        ).fetchone() == ("promoted",)

        # Expected constraint failures run inside savepoints so the enclosing
        # fixture transaction remains usable for the omission assertions.
        with pytest.raises(
            psycopg.Error,
            match=(
                "model_release_one_promoted_per_family_generation_idx|"
                "duplicate key value"
            ),
        ):
            with connection.transaction():
                repository.write_batch(_status(
                    family=omitted_family,
                    version=duplicate_version,
                    digest=first_digest,
                ))

        # The next registry generation mentions a different family only. The
        # previously promoted release remains immutable history, but it is no
        # longer current because current authority is global to the generation.
        repository.write_batch(
            _generation(sequence=second_sequence, digest=second_digest)
        )
        repository.write_batch(_status(
            family=other_family,
            version=other_version,
            digest=second_digest,
        ))

        with pytest.raises(
            psycopg.Error,
            match="contract-complete promoted release",
        ):
            with connection.transaction():
                repository.write_batch(eligible_prediction("after_omission"))

        omitted_current = connection.execute(
            "SELECT release_status FROM api.current_model_releases "
            "WHERE model_release_key = %s",
            (f"model_release:{omitted_family}:{omitted_version}",),
        ).fetchone()
        other_current = connection.execute(
            "SELECT release_status, generation_sequence "
            "FROM api.current_model_releases WHERE model_release_key = %s",
            (f"model_release:{other_family}:{other_version}",),
        ).fetchone()
        historical_status = connection.execute(
            "SELECT release_status FROM ml.model_release_status_events "
            "WHERE model_release_key = %s",
            (f"model_release:{omitted_family}:{omitted_version}",),
        ).fetchone()

        assert omitted_current is None
        assert other_current == ("promoted", second_sequence)
        assert historical_status == ("promoted",)
        connection.rollback()
