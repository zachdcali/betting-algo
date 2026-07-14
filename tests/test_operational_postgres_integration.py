from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from feature_contract import vector_sha256  # noqa: E402
from storage.live_records import (  # noqa: E402
    LiveRecordBuilder,
    match_identity,
    match_metadata_observation,
)
from storage.import_csv import apply_plan  # noqa: E402
from storage.parity import compare_memberships, compare_plan  # noqa: E402
from storage.records import (  # noqa: E402
    ImportPlan, RecordBatch, canonical_json, content_sha256,
)
from storage.repository import OperationalRepository  # noqa: E402
from storage.runtime import OperationalRuntimeSink  # noqa: E402
from versioning import (  # noqa: E402
    FEATURE_SCHEMA_ID, FEATURE_SCHEMA_NAME, FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION, LIVE_SEMANTICS_ID, LOGGING_SCHEMA_VERSION,
    OPERATIONAL_SCHEMA_VERSION,
)


DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
MIGRATIONS = (
    (
        "1.0.0",
        ROOT / "supabase" / "migrations" /
        "20260714010000_operational_schema_v1.sql",
    ),
    (
        "1.1.0",
        ROOT / "supabase" / "migrations" /
        "20260714020000_operational_integrity_v1_1.sql",
    ),
)


def _test_url() -> str:
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not configured")
    parsed = urlparse(DATABASE_URL)
    if (
        parsed.hostname not in {"127.0.0.1", "localhost", "postgres"}
        and os.environ.get("ALLOW_REMOTE_DB_TEST") != "1"
    ):
        pytest.fail(
            "refusing an operational integration test against a remote database"
        )
    return DATABASE_URL


def _ensure_migrations(connection) -> None:
    """Apply only missing versions, proving each new migration is replay-safe."""
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

    for version, migration in MIGRATIONS:
        if version in applied:
            continue
        sql = migration.read_text(encoding="utf-8")
        connection.execute(sql, prepare=False)
        connection.execute(sql, prepare=False)
        applied.add(version)


def _feature_contract_records():
    payload = json.loads(
        (PRODUCTION / "features/schema_141.json").read_text(encoding="utf-8")
    )
    names = tuple(
        item["name"]
        for item in sorted(payload["features"], key=lambda item: item["index"])
    )
    vector = {name: 0.0 for name in names}
    vector.update({
        "Surface_Clay": 1.0,
        "Level_A": 1.0,
        "Round_R32": 1.0,
        "P1_Hand_U": 1.0,
        "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0,
        "P2_Country_Other": 1.0,
    })
    schema = RecordBatch.from_records("ml.feature_schemas", [{
        "idempotency_key": (
            f"feature_schema:{FEATURE_SCHEMA_ID}:{FEATURE_SCHEMA_SHA256}"
        ),
        "schema_name": FEATURE_SCHEMA_NAME,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "schema_identifier": FEATURE_SCHEMA_ID,
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_count": len(names),
        "feature_names": canonical_json(names),
        "feature_contract": canonical_json({
            "semantics_id": LIVE_SEMANTICS_ID,
            "logging_schema_version": LOGGING_SCHEMA_VERSION,
        }),
    }])
    release = RecordBatch.from_records("ml.model_releases", [{
        "idempotency_key": "model_release:nn:1.2.3",
        "model_family": "nn",
        "model_version": "1.2.3",
        "release_status": "registered",
        "registry_schema_version": "2.0.0",
        "feature_schema_identifier": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_count": len(names),
        "training_feature_semantics_id": "sackmann_historical_legacy@1.0.0",
        "live_feature_semantics_id": LIVE_SEMANTICS_ID,
        "training_dataset_id": "integration_fixture@1.0.0",
        "model_sha256": "a" * 64,
        "scaler_sha256": "b" * 64,
        "registry_entry": canonical_json({
            "model_file": "model.pth",
            "model_sha256": "a" * 64,
            "scaler_file": "scaler.pkl",
            "scaler_sha256": "b" * 64,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "features": len(names),
            "training_feature_semantics_id": (
                "sackmann_historical_legacy@1.0.0"
            ),
            "live_feature_semantics_id": LIVE_SEMANTICS_ID,
            "training_dataset_id": "integration_fixture@1.0.0",
            "probability_mode": "raw",
        }),
        "contract_complete": True,
    }])
    registry_generation = RecordBatch.from_records(
        "ml.model_registry_generations", [{
            "idempotency_key": "model_registry_generation:integration:high",
            "registry_generation_sha256": "c" * 64,
            "generation_sequence": 900000000000000002,
            "registry_schema_version": "2.0.0",
            "effective_at": datetime(2026, 7, 14, 6, 0, tzinfo=timezone.utc),
        }],
    )
    status = RecordBatch.from_records("ml.model_release_status_events", [{
        "idempotency_key": "model_release_status:integration:nn:1.2.3",
        "model_release_key": "model_release:nn:1.2.3",
        "model_family": "nn",
        "registry_generation_sha256": "c" * 64,
        "release_status": "promoted",
    }])
    return names, vector, schema, release, registry_generation, status


def test_migration_is_idempotent_and_live_batches_match_postgres_contract():
    import psycopg

    url = _test_url()
    with psycopg.connect(url, autocommit=True) as connection:
        # Replaying an older migration after a newer view contract is not a
        # valid migrator order. A fresh database still exercises every
        # migration twice at its own point; an existing database applies only
        # missing versions.
        _ensure_migrations(connection)

    (
        names, vector, schema, release, registry_generation, release_status,
    ) = _feature_contract_records()
    now = datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc)
    builder = LiveRecordBuilder("run_operational_contract_ci", now)
    identity = match_identity(
        player1="Contract One",
        player2="Contract Two",
        match_date=date(2026, 7, 14),
        tournament="Contract Open",
        source_event_key="contract-event-ci",
    )
    run = builder.pipeline_run()
    stage = builder.pipeline_stage(
        stage_name="odds_fetch",
        status="success",
        started_at=now,
        completed_at=now + timedelta(seconds=2),
        metrics={"rows": 1},
    )
    fetch = builder.source_fetch(
        source_name="bovada",
        fetch_kind="odds",
        attempt=1,
        started_at=now,
        completed_at=now + timedelta(seconds=2),
        status="success",
        rows_observed=1,
    )
    fetch_id = fetch.unique_records[0]["source_fetch_id"]
    metadata = match_metadata_observation(
        identity=identity,
        run_id=builder.run_id,
        source_fetch_id=fetch_id,
        observed_at=now + timedelta(seconds=2),
        source_name="tournament_resolver",
        field_provenance={
            "match_date": "bovada",
            "match_start_at_utc": "bovada",
            "tournament": "official",
            "round_code": "official",
            "surface": "official",
            "level": "official",
        },
        match_start_at_utc=now + timedelta(hours=2),
        tournament="Contract Open",
        round_code="R32",
        surface="Clay",
        level="A",
    )
    odds = builder.odds_observation(
        identity=identity,
        source_fetch_id=fetch_id,
        observed_at=now + timedelta(seconds=2),
        match_start_at_utc=now + timedelta(hours=2),
        player1_decimal_odds="1.80",
        player2_decimal_odds="2.10",
        player1_market_probability="0.54",
        player2_market_probability="0.46",
        tournament="Contract Open",
        round_code="R32",
        surface="Clay",
        level="A",
    )
    feature = builder.feature_snapshot(
        identity=identity,
        player1=identity.player1,
        player2=identity.player2,
        captured_at=now + timedelta(seconds=3),
        build_status="ok",
        features_complete=True,
        feature_vector=vector,
        feature_vector_sha256=vector_sha256(vector, names),
        feature_count=len(names),
    )
    feature_id = feature.unique_records[0]["feature_snapshot_id"]
    predictions = builder.prediction_observations(
        identity=identity,
        feature_snapshot_id=feature_id,
        predicted_at=now + timedelta(seconds=4),
        external_prediction_id="prediction_operational_contract_ci",
        predictions=[{
            "model_family": "nn",
            "model_version": "1.2.3",
            "model_role": "promoted",
            "model_release_key": "model_release:nn:1.2.3",
            "decision_eligible": True,
            "player1_probability": "0.60",
            "player2_probability": "0.40",
        }],
    )
    prediction = predictions.unique_records[0]
    odds_record = odds.unique_records[0]
    account = RecordBatch.from_records("ops.paper_accounts", [{
        "idempotency_key": "paper_account:integration",
        "account_code": "integration",
        "display_name": "Integration paper account",
        "currency": "USD",
        "status": "active",
        "starting_capital": Decimal("1000"),
    }])
    market_probability = Decimal("1") / Decimal("1.80")
    bet = RecordBatch.from_records("ops.bet_recommendations", [{
        "idempotency_key": "bet_recommendation:integration",
        "external_bet_id": "bet_integration",
        "account_code": "integration",
        "match_uid": identity.match_uid,
        "feature_snapshot_id": feature_id,
        "prediction_observation_id": prediction["prediction_observation_id"],
        "odds_observation_id": odds_record["odds_observation_id"],
        "recommended_at": now + timedelta(seconds=5),
        "bet_side": identity.player1,
        "bet_on_player1": True,
        "decimal_odds": Decimal("1.80"),
        "stake": Decimal("10"),
        "stake_fraction": Decimal("0.01"),
        "model_probability": Decimal("0.60"),
        "market_probability": market_probability,
        "edge": Decimal("0.60") - market_probability,
        "kelly_fraction": Decimal("0.10"),
        "model_version": "1.2.3",
        "evidence_quality": "decision_grade",
    }])
    terminal = builder.pipeline_run(
        status="success",
        completed_at=now + timedelta(seconds=6),
        metrics={"predictions": 1},
    )

    ordered = (
        run, stage, fetch, metadata, odds, schema, release,
        registry_generation, release_status, feature, predictions, account,
        bet, terminal,
    )
    with psycopg.connect(url) as connection:
        existing_run = connection.execute(
            "SELECT status FROM ops.pipeline_runs WHERE run_id = %s",
            (builder.run_id,),
        ).fetchone()
        repository = OperationalRepository(connection)
        # A repeated test invocation sees the already-terminal run. Replaying
        # its earlier `running` snapshot is contradictory, so only replay the
        # exact terminal record and the other exact facts in that case.
        first_pass = ordered if existing_run is None else ordered[1:]
        for batch in (*first_pass, *ordered[1:]):
            repository.write_batch(batch)

    with psycopg.connect(url) as connection:
        run_status = connection.execute(
            "SELECT status FROM ops.pipeline_runs WHERE run_id = %s",
            (builder.run_id,),
        ).fetchone()
        metadata_row = connection.execute(
            "SELECT tournament, round_code, surface, level "
            "FROM api.current_match_metadata WHERE match_anchor_key = %s",
            (identity.match_anchor_key,),
        ).fetchone()
        prediction_row = connection.execute(
            "SELECT decision_eligible, model_release_key "
            "FROM ml.prediction_observations WHERE external_prediction_id = %s",
            ("prediction_operational_contract_ci",),
        ).fetchone()
        bet_quality = connection.execute(
            "SELECT evidence_quality FROM ops.bet_recommendations "
            "WHERE external_bet_id = 'bet_integration'"
        ).fetchone()
        release_row = connection.execute(
            "SELECT release_status, contract_complete, generation_sequence "
            "FROM api.current_model_releases "
            "WHERE model_release_key = 'model_release:nn:1.2.3'"
        ).fetchone()
        rls_count = connection.execute(
            "SELECT count(*) FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname IN ('raw', 'ops', 'ml') "
            "AND c.relkind = 'r' AND c.relrowsecurity",
        ).fetchone()[0]

    assert run_status == ("success",)
    assert metadata_row == ("Contract Open", "R32", "Clay", "A")
    assert prediction_row == (True, "model_release:nn:1.2.3")
    assert bet_quality == ("decision_grade",)
    assert release_row == ("promoted", True, 900000000000000002)
    assert rls_count >= 24

    # Once terminal, only the exact semantic hash is retryable. An earlier
    # lifecycle snapshot sharing the idempotency key must fail instead of being
    # silently ignored and reported as a durable write.
    with pytest.raises(
        psycopg.Error,
        match="idempotency conflict on ops.pipeline_runs.*record_sha256 mismatch",
    ):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batch(run)

    exact_result = OperationalRuntimeSink(
        mode="required", database_url=url,
    ).write([terminal])
    assert exact_result.succeeded is True
    assert exact_result.record_counts == {"ops.pipeline_runs": 1}

    reports = []
    conflict_result = OperationalRuntimeSink(
        mode="shadow", database_url=url, reporter=reports.append,
    ).write([run])
    assert conflict_result.succeeded is False
    assert conflict_result.record_counts == {}
    assert conflict_result.submitted_rows == 0
    assert conflict_result.error_type in {"RaiseException", "DatabaseError"}
    assert "record_sha256 mismatch" in conflict_result.error_message
    assert reports and "record_sha256 mismatch" in reports[0]

    lower_generation = RecordBatch.from_records(
        "ml.model_registry_generations", [{
            "idempotency_key": "model_registry_generation:integration:low",
            "registry_generation_sha256": "d" * 64,
            "generation_sequence": 900000000000000001,
            "registry_schema_version": "2.0.0",
            "effective_at": now - timedelta(days=1),
        }],
    )
    lower_status = RecordBatch.from_records(
        "ml.model_release_status_events", [{
            "idempotency_key": "model_release_status:integration:nn:1.2.3:low",
            "model_release_key": "model_release:nn:1.2.3",
            "model_family": "nn",
            "registry_generation_sha256": "d" * 64,
            "release_status": "archived",
        }],
    )
    with psycopg.connect(url) as connection:
        repository = OperationalRepository(connection)
        repository.write_batch(lower_generation)
        repository.write_batch(lower_status)

    with psycopg.connect(url) as connection:
        current_status = connection.execute(
            "SELECT release_status, registry_generation_sha256, "
            "generation_sequence FROM api.current_model_releases "
            "WHERE model_release_key = 'model_release:nn:1.2.3'"
        ).fetchone()
    assert current_status == ("promoted", "c" * 64, 900000000000000002)

    duplicate_generation_status = RecordBatch.from_records(
        "ml.model_release_status_events", [{
            "idempotency_key": (
                "model_release_status:integration:nn:1.2.3:duplicate"
            ),
            "model_release_key": "model_release:nn:1.2.3",
            "model_family": "nn",
            "registry_generation_sha256": "c" * 64,
            "release_status": "archived",
        }],
    )
    with pytest.raises(psycopg.Error, match=(
        "model_release_status_one_per_generation_idx|duplicate key value"
    )):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batch(
                duplicate_generation_status
            )

    wrong_identity = builder.prediction_observations(
        identity=identity,
        feature_snapshot_id=feature_id,
        predicted_at=now + timedelta(seconds=7),
        external_prediction_id="prediction_wrong_release",
        predictions=[{
            "model_family": "xgboost",
            "model_version": "1.2.3",
            "model_role": "promoted",
            "model_release_key": "model_release:nn:1.2.3",
            "decision_eligible": True,
            "player1_probability": "0.60",
            "player2_probability": "0.40",
        }],
    )
    with pytest.raises(psycopg.Error, match="contract-complete promoted release"):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batch(wrong_identity)

    with pytest.raises(psycopg.Error, match="lifecycle row cannot be deleted"):
        with psycopg.connect(url) as connection:
            connection.execute(
                "DELETE FROM ops.pipeline_runs WHERE run_id = %s",
                (builder.run_id,),
            )


def test_changed_manifest_preserves_prior_memberships_and_dedupes_facts():
    import psycopg

    url = _test_url()
    with psycopg.connect(url, autocommit=True) as connection:
        _ensure_migrations(connection)

    # Keep fixture lifecycle timestamps safely before the database clock: the
    # import finalizer records its own real completion timestamp.
    now = datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc)
    first_batch_id = "6fd89b77-7aae-51f9-b5f0-7cd722d22401"
    second_batch_id = "6fd89b77-7aae-51f9-b5f0-7cd722d22402"
    shared_key = "skip_event:integration_multibatch_v1:shared"
    new_key = "skip_event:integration_multibatch_v1:new"

    def make_skip(
        *, batch_id: str, key: str, external_id: str, row_number: int
    ) -> dict:
        source_row = {
            "fixture": "changed_manifest_v1",
            "external_skip_event_id": external_id,
            "row_number": row_number,
        }
        return {
            "idempotency_key": key,
            "external_skip_event_id": external_id,
            "skipped_at": now + timedelta(seconds=row_number),
            "stage_name": "integration_manifest",
            "reason_code": "integration_fixture",
            "reason_detail": "changed-manifest membership proof",
            "context": canonical_json({"fixture": "changed_manifest_v1"}),
            "import_batch_id": batch_id,
            "source_file": "integration/changed_manifest.csv",
            "source_row_number": row_number,
            "source_row_sha256": content_sha256(source_row),
            "source_row_json": canonical_json(source_row),
        }

    def make_plan(batch_id: str, manifest: str, rows: list[dict]) -> ImportPlan:
        batch_control = RecordBatch.from_records("ops.import_batches", [{
            "idempotency_key": f"import_batch:{batch_id}",
            "batch_id": batch_id,
            "schema_version": OPERATIONAL_SCHEMA_VERSION,
            "manifest_sha256": manifest,
            "source_manifest": canonical_json({
                "fixture": "changed_manifest_v1",
                "target_row_count": len(rows),
            }),
            "status": "planned",
            "planned_at": now,
        }])
        facts = RecordBatch.from_records("ops.skip_events", rows)
        return ImportPlan(
            import_batch_id=batch_id,
            production_dir=PRODUCTION,
            batches=(batch_control, facts),
            file_sha256={
                "integration/changed_manifest.csv": manifest,
            },
        )

    first_shared = make_skip(
        batch_id=first_batch_id,
        key=shared_key,
        external_id="integration_multibatch_v1_shared",
        row_number=1,
    )
    second_shared = make_skip(
        batch_id=second_batch_id,
        key=shared_key,
        external_id="integration_multibatch_v1_shared",
        row_number=1,
    )
    second_new = make_skip(
        batch_id=second_batch_id,
        key=new_key,
        external_id="integration_multibatch_v1_new",
        row_number=2,
    )
    first_plan = make_plan(first_batch_id, "d" * 64, [first_shared])
    second_plan = make_plan(
        second_batch_id, "e" * 64, [second_shared, second_new]
    )

    # Import provenance is deliberately different, but it must not change the
    # semantic identity of the shared immutable fact.
    first_hash = first_plan.batches[1].unique_records[0]["record_sha256"]
    second_hash = second_plan.batches[1].unique_records[0]["record_sha256"]
    assert first_hash == second_hash

    with psycopg.connect(url) as connection:
        first_counts, first_result = apply_plan(connection, first_plan)
    assert first_result["matches"] is True
    assert first_result["membership_matches"] is True
    assert first_counts["ops.skip_events"] == 1
    assert first_counts["ops.import_batch_memberships"] == 1

    with psycopg.connect(url) as connection:
        second_counts, second_result = apply_plan(connection, second_plan)
    assert second_result["matches"] is True
    assert second_result["membership_matches"] is True
    assert second_counts["ops.skip_events"] == 2
    assert second_counts["ops.import_batch_memberships"] == 2

    # A byte-for-byte semantic retry is a no-op, but a reused key with changed
    # content must fail rather than silently retaining a contradictory row.
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batch(first_plan.batches[1])
    contradictory_shared = dict(first_shared)
    contradictory_shared["reason_detail"] = "contradictory fixture content"
    with pytest.raises(
        psycopg.Error,
        match="idempotency conflict on ops.skip_events",
    ):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batch(
                RecordBatch.from_records(
                    "ops.skip_events", [contradictory_shared]
                )
            )

    with psycopg.connect(url) as connection:
        repository = OperationalRepository(connection)
        assert compare_plan(repository, first_plan).matches is True
        assert compare_memberships(repository, first_plan).matches is True
        assert compare_plan(repository, second_plan).matches is True
        assert compare_memberships(repository, second_plan).matches is True

        fact_count = connection.execute(
            "SELECT count(*) FROM ops.skip_events "
            "WHERE idempotency_key = ANY(%s)",
            ([shared_key, new_key],),
        ).fetchone()[0]
        shared_count = connection.execute(
            "SELECT count(*) FROM ops.skip_events WHERE idempotency_key = %s",
            (shared_key,),
        ).fetchone()[0]
        membership_counts = dict(connection.execute(
            "SELECT import_batch_id::text, count(*) "
            "FROM ops.import_batch_memberships "
            "WHERE import_batch_id = ANY(%s::uuid[]) "
            "GROUP BY import_batch_id",
            ([first_batch_id, second_batch_id],),
        ).fetchall())

    assert fact_count == 2
    assert shared_count == 1
    assert membership_counts == {
        first_batch_id: 1,
        second_batch_id: 2,
    }
