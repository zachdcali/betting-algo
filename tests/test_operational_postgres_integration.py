from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from feature_contract import vector_sha256  # noqa: E402
from storage.live_records import (  # noqa: E402
    LiveRecordBuilder,
    eligibility_match_round_observation,
    match_identity,
    match_metadata_observation,
)
from storage.import_csv import apply_plan  # noqa: E402
from storage.eligibility import (  # noqa: E402
    EvidenceSource, eligibility_generation, eligibility_generation_sha256,
    eligibility_generation_status_event, eligibility_review_event,
    player_entity, player_identity_observation, projection_seal_from_batches,
)
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
    (
        "1.2.0",
        ROOT / "supabase" / "migrations" /
        "20260714030000_eligibility_provenance_v1_2.sql",
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


def test_feature_profile_write_through_commits_after_read_and_connection_closes(
    monkeypatch,
):
    """Exercise the run-owned connection with real psycopg transaction state."""
    import psycopg
    from psycopg.pq import TransactionStatus

    import canonical_store
    from features.ta_feature_calculator import TAFeatureCalculator

    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    connection = psycopg.connect(_test_url())
    connection.execute(
        """CREATE TEMP TABLE players (
               player_id BIGINT PRIMARY KEY,
               height_cm REAL,
               hand TEXT,
               updated_at TIMESTAMPTZ
           )"""
    )
    connection.execute(
        "INSERT INTO players (player_id, height_cm, hand) VALUES (42, NULL, 'U')"
    )
    connection.commit()
    monkeypatch.setattr(canonical_store, "connect", lambda: connection)
    calculator = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calculator._store_conn = None
    calculator.use_store = True

    owned = calculator._store()
    assert owned.autocommit is True
    owned.execute("SELECT player_id FROM players WHERE player_id = 42").fetchone()
    assert owned.info.transaction_status is TransactionStatus.IDLE

    calculator._persist_player_field({"player_id": 42}, "height_cm", 188)
    calculator._persist_player_field({"player_id": 42}, "hand", "L")
    assert owned.execute(
        "SELECT height_cm, hand FROM players WHERE player_id = 42"
    ).fetchone() == (188.0, "L")
    assert owned.info.transaction_status is TransactionStatus.IDLE

    calculator.close_store()
    assert connection.closed is True


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
    # Contract 1.2 leaves the established metadata projection untouched.
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


def test_eligibility_database_rejects_noncanonical_direct_facts():
    import psycopg

    url = _test_url()
    with psycopg.connect(url) as connection:
        _ensure_migrations(connection)

    sequence = time.time_ns() // 100
    observed = datetime.now(timezone.utc) - timedelta(minutes=2)
    manifest = {"canonical_storage": str(sequence)}
    generation_sha = eligibility_generation_sha256(manifest)
    builder = LiveRecordBuilder(f"canonical_storage_{sequence}", observed)
    run = builder.pipeline_run()
    fetch = builder.source_fetch(
        source_name="atp_official",
        fetch_kind="draw",
        attempt=1,
        started_at=observed,
        completed_at=observed + timedelta(seconds=1),
        status="success",
        rows_observed=1,
    )
    fetch_id = str(fetch.unique_records[0]["source_fetch_id"])
    source_uri = "https://www.atptour.com/en/scores/canonical/draws"
    artifact = builder.source_artifact(
        source_fetch_id=fetch_id,
        artifact_kind="atp_draw_html",
        storage_uri=f"s3://private-test/canonical/{sequence}.html",
        content_sha256="e" * 64,
        captured_at=observed + timedelta(seconds=1),
        metadata={"source_uri": source_uri},
    )
    artifact_id = str(artifact.unique_records[0]["source_artifact_id"])
    identity = match_identity(
        player1=f"Canonical Alpha {sequence}",
        player2=f"Canonical Beta {sequence}",
        match_date=observed.date(),
        tournament="Canonical Storage Open",
    )
    round_batch = eligibility_match_round_observation(
        identity=identity,
        run_id=builder.run_id,
        source_fetch_id=fetch_id,
        observed_at=observed,
        source_name="atp_official",
        round_code="QF",
        eligibility_generation_sha256=generation_sha,
        source_artifact_id=artifact_id,
        source_uri=source_uri,
        source_content_sha256="e" * 64,
        confidence=1,
        initial_review_state="unreviewed",
        expires_at=observed + timedelta(days=1),
    )
    entity = player_entity(
        generation_sha256=generation_sha,
        legacy_player_id=sequence,
        canonical_name=identity.player1,
    )
    source = EvidenceSource.validated(
        source_name="atp_official",
        source_uri=source_uri,
        source_content_sha256="e" * 64,
        source_artifact_id=artifact_id,
        observed_at=observed,
        confidence=1,
        expires_at=observed + timedelta(days=1),
    )
    identity_batch = player_identity_observation(
        generation_sha256=generation_sha,
        canonical_player_id=sequence,
        observed_name=identity.player1,
        source_player_key=f"atp-{sequence}",
        source=source,
    )
    seal = projection_seal_from_batches(
        (entity, round_batch), generation_sha256=generation_sha,
    )
    generation = eligibility_generation(
        generation_sequence=sequence,
        effective_at=observed,
        source_manifest=manifest,
        expected_projection_seal_sha256=seal.projection_seal_sha256,
        expected_projection_row_count=seal.projection_row_count,
    )
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batches(
            (generation, run, fetch, artifact, entity),
        )

    def malformed(batch, id_column, field, value):
        row = dict(batch.unique_records[0])
        row[id_column] = str(uuid4())
        row["idempotency_key"] = f"{row['idempotency_key']}:bad:{field}"
        row[field] = value
        row.pop("record_sha256", None)
        return RecordBatch.from_records(batch.table, [row])

    invalid_facts = (
        malformed(
            round_batch, "eligibility_match_round_observation_id",
            "round_code", "qf",
        ),
        malformed(
            round_batch, "eligibility_match_round_observation_id",
            "match_anchor_key", f"\t{identity.match_anchor_key}",
        ),
        malformed(
            round_batch, "eligibility_match_round_observation_id",
            "source_uri", "HTTPS://www.atptour.com/en/scores/canonical/draws",
        ),
        malformed(
            round_batch, "eligibility_match_round_observation_id",
            "source_uri", f"{source_uri}\n",
        ),
        malformed(
            identity_batch, "player_identity_observation_id",
            "source_name", "\tatp_official",
        ),
        malformed(
            identity_batch, "player_identity_observation_id",
            "source_player_key", f"atp-{sequence}\n",
        ),
    )
    for invalid in invalid_facts:
        with pytest.raises(psycopg.Error):
            with psycopg.connect(url) as connection:
                OperationalRepository(connection).write_batch(invalid)


def test_eligibility_generation_seal_rollback_and_round_isolation():
    import psycopg

    url = _test_url()
    with psycopg.connect(url) as connection:
        _ensure_migrations(connection)

    base_sequence = time.time_ns() // 100
    t0 = datetime.now(timezone.utc) - timedelta(minutes=20)
    identity = match_identity(
        player1=f"Legacy Alpha {base_sequence}",
        player2=f"Legacy Beta {base_sequence}",
        match_date=t0.date(),
        tournament="Eligibility Integration Open",
    )
    legacy_round = match_metadata_observation(
        identity=identity,
        run_id=None,
        observed_at=t0,
        source_name="legacy_import",
        field_provenance={"match_date": "inferred", "round_code": "inferred"},
        round_code="R32",
    )
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batch(legacy_round)
    with psycopg.connect(url) as connection:
        legacy_before = connection.execute(
            "SELECT to_jsonb(m)::text FROM api.current_match_metadata m "
            "WHERE match_anchor_key = %s",
            (identity.match_anchor_key,),
        ).fetchone()
    assert legacy_before is not None

    lower_manifest = {"integration": f"lower-{base_sequence}"}
    lower_sha = eligibility_generation_sha256(lower_manifest)
    builder = LiveRecordBuilder(
        f"eligibility_contract_{base_sequence}", t0 + timedelta(minutes=1),
    )
    run = builder.pipeline_run()
    fetch = builder.source_fetch(
        source_name="atp_official",
        fetch_kind="draw",
        attempt=1,
        started_at=t0 + timedelta(minutes=1),
        completed_at=t0 + timedelta(minutes=1, seconds=1),
        status="success",
        rows_observed=1,
    )
    fetch_id = str(fetch.unique_records[0]["source_fetch_id"])
    source_uri = "https://www.atptour.com/en/scores/integration/draws"
    artifact = builder.source_artifact(
        source_fetch_id=fetch_id,
        artifact_kind="atp_draw_html",
        storage_uri=f"s3://private-test/eligibility/{base_sequence}.html",
        content_sha256="a" * 64,
        captured_at=t0 + timedelta(minutes=1, seconds=1),
        content_type="text/html",
        byte_size=123,
        metadata={"source_uri": source_uri},
    )
    artifact_id = str(artifact.unique_records[0]["source_artifact_id"])
    compatibility_source = EvidenceSource.validated(
        source_name="public_compatibility_projection",
        source_uri=(
            "compatibility://public.players-player_aliases/"
            f"integration-{base_sequence}"
        ),
        source_content_sha256="b" * 64,
        observed_at=t0 + timedelta(minutes=1),
        confidence=1,
        compatibility_import=True,
    )
    lower_entity = player_entity(
        generation_sha256=lower_sha,
        legacy_player_id=base_sequence,
        canonical_name=identity.player1,
    )
    lower_identity = player_identity_observation(
        generation_sha256=lower_sha,
        canonical_player_id=base_sequence,
        observed_name=identity.player1,
        source_player_key=str(base_sequence),
        source=compatibility_source,
    )
    candidate_round = eligibility_match_round_observation(
        identity=identity,
        run_id=builder.run_id,
        source_fetch_id=fetch_id,
        observed_at=t0 + timedelta(minutes=1),
        source_name="atp_official",
        round_code="QF",
        eligibility_generation_sha256=lower_sha,
        source_artifact_id=artifact_id,
        source_uri=source_uri,
        source_content_sha256="a" * 64,
        confidence=1,
        initial_review_state="unreviewed",
        expires_at=t0 + timedelta(days=2),
    )
    identity_review = eligibility_review_event(
        generation_sha256=lower_sha,
        target_table="ops.player_identity_observations",
        target_idempotency_key=str(lower_identity.unique_records[0]["idempotency_key"]),
        review_state="accepted",
        reviewed_at=t0 + timedelta(minutes=2),
        reviewed_by="integration-test",
        reason="compatibility identity row reviewed",
    )
    round_review = eligibility_review_event(
        generation_sha256=lower_sha,
        target_table="ops.eligibility_match_round_observations",
        target_idempotency_key=str(candidate_round.unique_records[0]["idempotency_key"]),
        review_state="accepted",
        reviewed_at=t0 + timedelta(minutes=2),
        reviewed_by="integration-test",
        reason="official draw checksum and exact player pair reviewed",
    )
    lower_content = (
        lower_entity, lower_identity, candidate_round, identity_review, round_review,
    )
    lower_seal = projection_seal_from_batches(
        lower_content, generation_sha256=lower_sha,
    )
    lower_generation = eligibility_generation(
        generation_sequence=base_sequence,
        effective_at=t0,
        source_manifest=lower_manifest,
        expected_projection_seal_sha256=lower_seal.projection_seal_sha256,
        expected_projection_row_count=lower_seal.projection_row_count,
    )
    lower_candidate = eligibility_generation_status_event(
        generation_sha256=lower_sha,
        generation_sequence=base_sequence,
        status="candidate",
        effective_at=t0 + timedelta(minutes=3),
        reviewed_by="integration-test",
        reason="sealed candidate with complete reviews",
        projection_seal_sha256=lower_seal.projection_seal_sha256,
        projection_row_count=lower_seal.projection_row_count,
    )
    lower_accepted = eligibility_generation_status_event(
        generation_sha256=lower_sha,
        generation_sequence=base_sequence,
        status="accepted",
        effective_at=t0 + timedelta(minutes=4),
        reviewed_by="integration-test",
        reason="sealed lower generation accepted",
        projection_seal_sha256=lower_seal.projection_seal_sha256,
        projection_row_count=lower_seal.projection_row_count,
    )
    # Deliberately scramble the input; repository ordering must write
    # generation -> artifacts/content/reviews -> candidate -> accepted.
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batches((
            lower_accepted, round_review, candidate_round, artifact,
            lower_generation, lower_identity, fetch, identity_review,
            lower_candidate, run, lower_entity,
        ))

    with psycopg.connect(url) as connection:
        sql_seal = connection.execute(
            "SELECT projection_seal_sha256, projection_row_count "
            "FROM ops.compute_eligibility_projection_seal(%s)",
            (lower_sha,),
        ).fetchone()
        legacy_after = connection.execute(
            "SELECT to_jsonb(m)::text FROM api.current_match_metadata m "
            "WHERE match_anchor_key = %s",
            (identity.match_anchor_key,),
        ).fetchone()
        candidate_projection = connection.execute(
            "SELECT legacy_round_code, candidate_round_code "
            "FROM api.candidate_eligibility_match_metadata "
            "WHERE match_anchor_key = %s",
            (identity.match_anchor_key,),
        ).fetchone()
    assert sql_seal == (
        lower_seal.projection_seal_sha256, lower_seal.projection_row_count,
    )
    assert legacy_after == legacy_before
    assert candidate_projection == ("R32", "QF")

    other_builder = LiveRecordBuilder(
        f"eligibility_other_run_{base_sequence}", t0 + timedelta(minutes=5),
    )
    other_run = other_builder.pipeline_run()
    other_fetch = other_builder.source_fetch(
        source_name="atp_official",
        fetch_kind="draw",
        attempt=1,
        started_at=t0 + timedelta(minutes=5),
        completed_at=t0 + timedelta(minutes=5, seconds=1),
        status="success",
        rows_observed=1,
    )
    other_fetch_id = str(other_fetch.unique_records[0]["source_fetch_id"])
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batches((other_run, other_fetch))

    for offset, mismatched_fetch_id, mismatched_run_id in (
        (100, fetch_id, other_builder.run_id),
        (101, other_fetch_id, other_builder.run_id),
    ):
        mismatch_manifest = {"lineage_mismatch": f"{base_sequence}-{offset}"}
        mismatch_sha = eligibility_generation_sha256(mismatch_manifest)
        mismatched_round = eligibility_match_round_observation(
            identity=identity,
            run_id=mismatched_run_id,
            source_fetch_id=mismatched_fetch_id,
            observed_at=t0 + timedelta(minutes=5),
            source_name="atp_official",
            round_code="SF",
            eligibility_generation_sha256=mismatch_sha,
            source_artifact_id=artifact_id,
            source_uri=source_uri,
            source_content_sha256="a" * 64,
            confidence=1,
            initial_review_state="unreviewed",
            expires_at=t0 + timedelta(days=2),
        )
        mismatch_seal = projection_seal_from_batches(
            (mismatched_round,), generation_sha256=mismatch_sha,
        )
        mismatch_generation = eligibility_generation(
            generation_sequence=base_sequence + offset,
            effective_at=t0,
            source_manifest=mismatch_manifest,
            expected_projection_seal_sha256=mismatch_seal.projection_seal_sha256,
            expected_projection_row_count=mismatch_seal.projection_row_count,
        )
        with pytest.raises(psycopg.Error, match="artifact fetch/run lineage mismatch"):
            with psycopg.connect(url) as connection:
                OperationalRepository(connection).write_batches((
                    mismatched_round, mismatch_generation,
                ))

    # Exact retries remain no-ops after sealing; new facts and every physical
    # update/delete are rejected.
    with psycopg.connect(url) as connection:
        assert OperationalRepository(connection).write_batch(candidate_round) == 1
    late_entity = player_entity(
        generation_sha256=lower_sha,
        legacy_player_id=base_sequence + 999,
        canonical_name=f"Late Player {base_sequence}",
    )
    with pytest.raises(psycopg.Error, match="projection is sealed"):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batch(late_entity)
    for statement in (
        "UPDATE ops.eligibility_match_round_observations SET round_code = 'SF' "
        "WHERE idempotency_key = %s",
        "DELETE FROM ops.eligibility_match_round_observations "
        "WHERE idempotency_key = %s",
    ):
        with pytest.raises(psycopg.Error, match="immutable operational fact"):
            with psycopg.connect(url) as connection:
                connection.execute(
                    statement,
                    (candidate_round.unique_records[0]["idempotency_key"],),
                )

    # A higher accepted generation wins, and explicit retirement falls back to
    # the exact lower generation seal.
    higher_manifest = {"integration": f"higher-{base_sequence}"}
    higher_sha = eligibility_generation_sha256(higher_manifest)
    higher_entity = player_entity(
        generation_sha256=higher_sha,
        legacy_player_id=base_sequence + 1,
        canonical_name=f"Higher Identity {base_sequence}",
    )
    higher_identity = player_identity_observation(
        generation_sha256=higher_sha,
        canonical_player_id=base_sequence + 1,
        observed_name=f"Higher Identity {base_sequence}",
        source_player_key=str(base_sequence + 1),
        source=compatibility_source,
    )
    higher_review = eligibility_review_event(
        generation_sha256=higher_sha,
        target_table="ops.player_identity_observations",
        target_idempotency_key=str(higher_identity.unique_records[0]["idempotency_key"]),
        review_state="accepted",
        reviewed_at=t0 + timedelta(minutes=5),
        reviewed_by="integration-test",
        reason="higher identity reviewed",
    )
    higher_seal = projection_seal_from_batches(
        (higher_entity, higher_identity, higher_review),
        generation_sha256=higher_sha,
    )
    higher_generation = eligibility_generation(
        generation_sequence=base_sequence + 1,
        effective_at=t0,
        source_manifest=higher_manifest,
        expected_projection_seal_sha256=higher_seal.projection_seal_sha256,
        expected_projection_row_count=higher_seal.projection_row_count,
    )
    higher_candidate = eligibility_generation_status_event(
        generation_sha256=higher_sha,
        generation_sequence=base_sequence + 1,
        status="candidate",
        effective_at=t0 + timedelta(minutes=6),
        reviewed_by="integration-test",
        reason="higher candidate sealed",
        projection_seal_sha256=higher_seal.projection_seal_sha256,
        projection_row_count=higher_seal.projection_row_count,
    )
    higher_accepted = eligibility_generation_status_event(
        generation_sha256=higher_sha,
        generation_sequence=base_sequence + 1,
        status="accepted",
        effective_at=t0 + timedelta(minutes=7),
        reviewed_by="integration-test",
        reason="higher generation accepted",
        projection_seal_sha256=higher_seal.projection_seal_sha256,
        projection_row_count=higher_seal.projection_row_count,
    )
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batches((
            higher_accepted, higher_identity, higher_generation,
            higher_review, higher_candidate, higher_entity,
        ))
    with psycopg.connect(url) as connection:
        assert connection.execute(
            "SELECT generation_sha256 FROM api.current_eligibility_generation"
        ).fetchone() == (higher_sha,)

    invalid_regression = eligibility_generation_status_event(
        generation_sha256=higher_sha,
        generation_sequence=base_sequence + 1,
        status="candidate",
        effective_at=t0 + timedelta(minutes=8),
        reviewed_by="integration-test",
        reason="accepted cannot regress to candidate",
        projection_seal_sha256=higher_seal.projection_seal_sha256,
        projection_row_count=higher_seal.projection_row_count,
    )
    with pytest.raises(psycopg.Error, match="invalid eligibility status transition"):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batch(invalid_regression)
    explicit_rollback = eligibility_generation_status_event(
        generation_sha256=higher_sha,
        generation_sequence=base_sequence + 1,
        status="retired",
        effective_at=t0 + timedelta(minutes=9),
        reviewed_by="integration-test",
        reason="explicit rollback to lower accepted generation",
    )
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batch(explicit_rollback)
    with psycopg.connect(url) as connection:
        assert connection.execute(
            "SELECT generation_sha256, projection_seal_sha256 "
            "FROM api.current_eligibility_generation"
        ).fetchone() == (lower_sha, lower_seal.projection_seal_sha256)


def test_eligibility_candidate_rejects_missing_reviews_and_identity_conflicts():
    import psycopg

    url = _test_url()
    with psycopg.connect(url) as connection:
        _ensure_migrations(connection)
    sequence = time.time_ns() // 100
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    source = EvidenceSource.validated(
        source_name="public_compatibility_projection",
        source_uri=(
            "compatibility://public.players-player_aliases/"
            f"readiness-{sequence}"
        ),
        source_content_sha256="d" * 64,
        observed_at=t0,
        confidence=1,
        compatibility_import=True,
    )

    missing_manifest = {"readiness": f"missing-review-{sequence}"}
    missing_sha = eligibility_generation_sha256(missing_manifest)
    missing_entity = player_entity(
        generation_sha256=missing_sha,
        legacy_player_id=sequence,
        canonical_name=f"Missing Review {sequence}",
    )
    missing_identity = player_identity_observation(
        generation_sha256=missing_sha,
        canonical_player_id=sequence,
        observed_name=f"Missing Review {sequence}",
        source_player_key=str(sequence),
        source=source,
    )
    missing_seal = projection_seal_from_batches(
        (missing_entity, missing_identity), generation_sha256=missing_sha,
    )
    missing_generation = eligibility_generation(
        generation_sequence=sequence,
        effective_at=t0,
        source_manifest=missing_manifest,
        expected_projection_seal_sha256=missing_seal.projection_seal_sha256,
        expected_projection_row_count=missing_seal.projection_row_count,
    )
    missing_candidate = eligibility_generation_status_event(
        generation_sha256=missing_sha,
        generation_sequence=sequence,
        status="candidate",
        effective_at=t0 + timedelta(minutes=1),
        reviewed_by="integration-test",
        reason="must fail without explicit review",
        projection_seal_sha256=missing_seal.projection_seal_sha256,
        projection_row_count=missing_seal.projection_row_count,
    )
    with pytest.raises(psycopg.Error, match="without explicit terminal review"):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batches((
                missing_candidate, missing_identity,
                missing_generation, missing_entity,
            ))

    conflict_manifest = {"readiness": f"identity-conflict-{sequence}"}
    conflict_sha = eligibility_generation_sha256(conflict_manifest)
    entities = tuple(
        player_entity(
            generation_sha256=conflict_sha,
            legacy_player_id=sequence + offset,
            canonical_name=f"Conflict Entity {offset} {sequence}",
        )
        for offset in (10, 11)
    )
    identities = tuple(
        player_identity_observation(
            generation_sha256=conflict_sha,
            canonical_player_id=sequence + offset,
            observed_name=f"Shared Conflict Name {sequence}",
            source_player_key=f"source-{offset}",
            source=source,
        )
        for offset in (10, 11)
    )
    reviews = tuple(
        eligibility_review_event(
            generation_sha256=conflict_sha,
            target_table="ops.player_identity_observations",
            target_idempotency_key=str(identity.unique_records[0]["idempotency_key"]),
            review_state="accepted",
            reviewed_at=t0 + timedelta(minutes=1),
            reviewed_by="integration-test",
            reason="accepted to prove conflict gate",
        )
        for identity in identities
    )
    conflict_content = (*entities, *identities, *reviews)
    conflict_seal = projection_seal_from_batches(
        conflict_content, generation_sha256=conflict_sha,
    )
    conflict_generation = eligibility_generation(
        generation_sequence=sequence + 1,
        effective_at=t0,
        source_manifest=conflict_manifest,
        expected_projection_seal_sha256=conflict_seal.projection_seal_sha256,
        expected_projection_row_count=conflict_seal.projection_row_count,
    )
    conflict_candidate = eligibility_generation_status_event(
        generation_sha256=conflict_sha,
        generation_sequence=sequence + 1,
        status="candidate",
        effective_at=t0 + timedelta(minutes=2),
        reviewed_by="integration-test",
        reason="must fail on accepted name conflict",
        projection_seal_sha256=conflict_seal.projection_seal_sha256,
        projection_row_count=conflict_seal.projection_row_count,
    )
    with pytest.raises(psycopg.Error, match="accepted projection conflicts"):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batches((
                conflict_candidate, *reviews, *identities,
                conflict_generation, *entities,
            ))


def test_eligibility_acceptance_rechecks_identity_expiry():
    import psycopg

    url = _test_url()
    with psycopg.connect(url) as connection:
        _ensure_migrations(connection)
    sequence = time.time_ns() // 100
    t0 = datetime.now(timezone.utc) - timedelta(minutes=10)
    manifest = {"readiness": f"expires-between-statuses-{sequence}"}
    generation_sha = eligibility_generation_sha256(manifest)
    source = EvidenceSource.validated(
        source_name="public_compatibility_projection",
        source_uri=(
            "compatibility://public.players-player_aliases/"
            f"expiry-{sequence}"
        ),
        source_content_sha256="e" * 64,
        observed_at=t0,
        confidence=1,
        expires_at=t0 + timedelta(minutes=3),
        compatibility_import=True,
    )
    entity = player_entity(
        generation_sha256=generation_sha,
        legacy_player_id=sequence,
        canonical_name=f"Expiring Identity {sequence}",
    )
    identity = player_identity_observation(
        generation_sha256=generation_sha,
        canonical_player_id=sequence,
        observed_name=f"Expiring Identity {sequence}",
        source_player_key=str(sequence),
        source=source,
    )
    review = eligibility_review_event(
        generation_sha256=generation_sha,
        target_table="ops.player_identity_observations",
        target_idempotency_key=str(identity.unique_records[0]["idempotency_key"]),
        review_state="accepted",
        reviewed_at=t0 + timedelta(minutes=1),
        reviewed_by="integration-test",
        reason="identity valid at candidate time",
    )
    seal = projection_seal_from_batches(
        (entity, identity, review), generation_sha256=generation_sha,
    )
    generation = eligibility_generation(
        generation_sequence=sequence,
        effective_at=t0,
        source_manifest=manifest,
        expected_projection_seal_sha256=seal.projection_seal_sha256,
        expected_projection_row_count=seal.projection_row_count,
    )
    candidate = eligibility_generation_status_event(
        generation_sha256=generation_sha,
        generation_sequence=sequence,
        status="candidate",
        effective_at=t0 + timedelta(minutes=2),
        reviewed_by="integration-test",
        reason="identity remains active at candidate time",
        projection_seal_sha256=seal.projection_seal_sha256,
        projection_row_count=seal.projection_row_count,
    )
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batches((
            candidate, review, identity, generation, entity,
        ))
    accepted = eligibility_generation_status_event(
        generation_sha256=generation_sha,
        generation_sequence=sequence,
        status="accepted",
        effective_at=t0 + timedelta(minutes=4),
        reviewed_by="integration-test",
        reason="must fail because accepted identity expired",
        projection_seal_sha256=seal.projection_seal_sha256,
        projection_row_count=seal.projection_row_count,
    )
    with pytest.raises(psycopg.Error, match="no active accepted identity"):
        with psycopg.connect(url) as connection:
            OperationalRepository(connection).write_batch(accepted)


def test_eligibility_candidate_and_late_content_race_is_serialized():
    import psycopg

    url = _test_url()
    with psycopg.connect(url) as connection:
        _ensure_migrations(connection)
    sequence = time.time_ns() // 100
    t0 = datetime.now(timezone.utc) - timedelta(minutes=5)
    manifest = {"race": f"candidate-vs-content-{sequence}"}
    generation_sha = eligibility_generation_sha256(manifest)
    source = EvidenceSource.validated(
        source_name="public_compatibility_projection",
        source_uri=(
            "compatibility://public.players-player_aliases/"
            f"race-{sequence}"
        ),
        source_content_sha256="f" * 64,
        observed_at=t0,
        confidence=1,
        compatibility_import=True,
    )
    entity = player_entity(
        generation_sha256=generation_sha,
        legacy_player_id=sequence,
        canonical_name=f"Race Identity {sequence}",
    )
    identity = player_identity_observation(
        generation_sha256=generation_sha,
        canonical_player_id=sequence,
        observed_name=f"Race Identity {sequence}",
        source_player_key=str(sequence),
        source=source,
    )
    review = eligibility_review_event(
        generation_sha256=generation_sha,
        target_table="ops.player_identity_observations",
        target_idempotency_key=str(identity.unique_records[0]["idempotency_key"]),
        review_state="accepted",
        reviewed_at=t0 + timedelta(minutes=1),
        reviewed_by="integration-test",
        reason="race fixture identity reviewed",
    )
    seal = projection_seal_from_batches(
        (entity, identity, review), generation_sha256=generation_sha,
    )
    generation = eligibility_generation(
        generation_sequence=sequence,
        effective_at=t0,
        source_manifest=manifest,
        expected_projection_seal_sha256=seal.projection_seal_sha256,
        expected_projection_row_count=seal.projection_row_count,
    )
    with psycopg.connect(url) as connection:
        OperationalRepository(connection).write_batches((
            review, identity, generation, entity,
        ))
    candidate = eligibility_generation_status_event(
        generation_sha256=generation_sha,
        generation_sequence=sequence,
        status="candidate",
        effective_at=t0 + timedelta(minutes=2),
        reviewed_by="integration-test",
        reason="candidate holds generation lock during race",
        projection_seal_sha256=seal.projection_seal_sha256,
        projection_row_count=seal.projection_row_count,
    )
    late = player_entity(
        generation_sha256=generation_sha,
        legacy_player_id=sequence + 1,
        canonical_name=f"Late Race Identity {sequence}",
    )

    status_connection = psycopg.connect(url)
    errors: list[str] = []
    attempted = threading.Event()

    def insert_late_content() -> None:
        try:
            with psycopg.connect(url) as connection:
                attempted.set()
                OperationalRepository(connection).write_batch(late)
        except psycopg.Error as exc:
            errors.append(str(exc))

    try:
        OperationalRepository(status_connection).write_batch(candidate)
        worker = threading.Thread(target=insert_late_content, daemon=True)
        worker.start()
        assert attempted.wait(timeout=5)
        time.sleep(0.2)
        status_connection.commit()
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        status_connection.close()
    assert errors and "projection is sealed at status candidate" in errors[0]
    with psycopg.connect(url) as connection:
        assert connection.execute(
            "SELECT count(*) FROM ops.player_entities "
            "WHERE eligibility_generation_sha256 = %s",
            (generation_sha,),
        ).fetchone() == (1,)

    # Stronger snapshot isolation is deliberately unsupported because it can
    # retain a pre-lock snapshot; both content and status guards fail closed.
    with pytest.raises(psycopg.Error, match="requires READ COMMITTED"):
        with psycopg.connect(url) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            OperationalRepository(connection).write_batch(late)


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
