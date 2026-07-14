from pathlib import Path
import sys

import pytest

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

from storage.parity import compare_plan  # noqa: E402
from storage.records import ImportPlan, RecordBatch  # noqa: E402
from storage.repository import OperationalRepository  # noqa: E402


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []
        self.closed = False

    def execute(self, query, params=()):
        self.connection.execute_calls.append((query, params))
        for table, rows in self.connection.inventories.items():
            quoted = ".".join(f'"{part}"' for part in table.split("."))
            if f"FROM {quoted}" in query:
                self.rows = list(rows)
                break

    def executemany(self, query, params_seq):
        self.connection.executemany_calls.append((query, list(params_seq)))

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True
        self.connection.closed_cursors += 1


class FakeConnection:
    def __init__(self, inventories=None):
        self.execute_calls = []
        self.executemany_calls = []
        self.inventories = inventories or {}
        self.closed_cursors = 0
        self.commit_calls = 0
        self.rollback_calls = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1


def test_immutable_batch_is_parameterized_idempotent_and_caller_owned():
    connection = FakeConnection()
    repository = OperationalRepository(connection)
    batch = RecordBatch.from_records("ops.odds_observations", [
        {
            "idempotency_key": "odds:1",
            "match_uid": "m1'; DROP TABLE ops.odds_observations; --",
            "source_row_json": '{"match":"m1"}',
        },
        {
            "idempotency_key": "odds:1",
            "match_uid": "m1'; DROP TABLE ops.odds_observations; --",
            "source_row_json": '{"match":"m1"}',
        },
    ])

    assert repository.write_batch(batch) == 1
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 0
    assert connection.closed_cursors == 1
    query, parameters = connection.executemany_calls[0]
    assert 'INSERT INTO "ops"."odds_observations"' in query
    assert 'ON CONFLICT ("idempotency_key") DO UPDATE SET' in query
    assert "ops.require_matching_record_sha256" in query
    assert "'ops.odds_observations'" in query
    assert "%s::jsonb" in query
    assert "DROP TABLE" not in query
    assert any("DROP TABLE" in str(value) for value in parameters[0])


def test_terminal_lifecycle_upsert_cannot_replace_a_terminal_status():
    connection = FakeConnection()
    repository = OperationalRepository(connection)
    repository.write_batch(RecordBatch.from_records("ops.pipeline_runs", [{
        "idempotency_key": "run:1",
        "external_run_id": "run_1",
        "status": "running",
    }]))

    query, parameters = connection.executemany_calls[0]
    assert 'ON CONFLICT ("idempotency_key") DO UPDATE SET' in query
    assert 'CASE WHEN lower(COALESCE("existing"."status", \'\')) IN' in query
    assert "ops.require_matching_record_sha256" in query
    assert "'ops.pipeline_runs'" in query
    assert ' WHERE lower(COALESCE("existing"."status", \'\')) NOT IN' not in query
    for status in ("success", "partial", "failed", "cancelled", "canceled"):
        assert f"'{status}'" in query
        assert status not in parameters[0]
    assert connection.commit_calls == 0


def test_terminal_import_batch_uses_stable_manifest_hash_for_retry_conflicts():
    connection = FakeConnection()
    repository = OperationalRepository(connection)

    repository.write_batch(RecordBatch.from_records("ops.import_batches", [{
        "idempotency_key": "import_batch:fixture",
        "batch_id": "11111111-1111-1111-1111-111111111111",
        "schema_version": "1.1.0",
        "manifest_sha256": "a" * 64,
        "source_manifest": "{}",
        "status": "verified",
    }]))

    query, _parameters = connection.executemany_calls[0]
    assert '"existing"."manifest_sha256"' in query
    assert "ops.require_matching_record_sha256" in query
    assert "'ops.import_batches'" in query


def test_repository_rejects_unapproved_tables_and_unsafe_columns():
    repository = OperationalRepository(FakeConnection())
    with pytest.raises(ValueError, match="operational contract"):
        repository.write_batch(RecordBatch.from_records("public.users", [{
            "idempotency_key": "bad",
        }]))
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        repository.write_batch(RecordBatch.from_records("ops.skip_events", [{
            "idempotency_key": "bad",
            "reason; DROP TABLE x": "bad",
        }]))


def test_paper_sessions_and_shadow_metadata_are_allowlisted_and_json_typed():
    connection = FakeConnection()
    repository = OperationalRepository(connection)

    assert repository.write_batch(RecordBatch.from_records("ops.paper_sessions", [{
        "idempotency_key": "session:1",
        "external_session_id": "session_1",
        "completed_at": "2026-07-13T15:00:00Z",
    }])) == 1
    assert repository.write_batch(RecordBatch.from_records(
        "ml.prediction_observations", [{
            "idempotency_key": "shadow:1",
            "model_role": "shadow",
            "metadata": '{"shadow_status":"error"}',
        }],
    )) == 1
    assert repository.write_batch(RecordBatch.from_records(
        "ml.model_releases", [{
            "idempotency_key": "model_release:nn:1.2.3",
            "model_family": "nn",
            "model_version": "1.2.3",
            "registry_entry": '{"model_file":"model.pth"}',
        }],
    )) == 1

    session_sql = connection.executemany_calls[0][0]
    shadow_sql = connection.executemany_calls[1][0]
    release_sql = connection.executemany_calls[2][0]
    assert 'INSERT INTO "ops"."paper_sessions"' in session_sql
    assert '"metadata"' in shadow_sql
    assert "%s::jsonb" in shadow_sql
    assert 'INSERT INTO "ml"."model_releases"' in release_sql
    assert '"registry_entry"' in release_sql
    assert "%s::jsonb" in release_sql


def test_parity_compares_canonical_keys_counts_and_source_hashes(tmp_path):
    batch_id = "11111111-1111-1111-1111-111111111111"
    batch = RecordBatch.from_records("ops.skip_events", [
        {"idempotency_key": "skip:1", "source_row_sha256": "hash-1"},
        {"idempotency_key": "skip:2", "source_row_sha256": "hash-2"},
    ])
    plan = ImportPlan(batch_id, tmp_path, (batch,))
    hashes = {
        row["idempotency_key"]: row["record_sha256"]
        for row in batch.unique_records
    }
    connection = FakeConnection({
        "ops.skip_events": [("skip:1", hashes["skip:1"]), ("skip:2", "wrong")],
    })

    report = compare_plan(OperationalRepository(connection), plan)

    assert not report.matches
    assert report.tables[0].source_count == 2
    assert report.tables[0].repository_count == 2
    assert report.tables[0].hash_mismatches == ("skip:2",)


def test_parity_reports_missing_and_extra_keys(tmp_path):
    batch_id = "22222222-2222-2222-2222-222222222222"
    plan = ImportPlan(batch_id, tmp_path, (RecordBatch.from_records(
        "ml.feature_snapshots",
        [{"idempotency_key": "feature:expected", "source_row_sha256": "same"}],
    ),))
    connection = FakeConnection({
        "ml.feature_snapshots": [("feature:unexpected", "same")],
    })

    table = compare_plan(OperationalRepository(connection), plan).tables[0]

    assert table.missing_keys == ("feature:expected",)
    assert table.extra_keys == ("feature:unexpected",)


def test_record_hash_ignores_import_provenance_but_detects_fact_changes():
    first = RecordBatch.from_records("ops.skip_events", [{
        "idempotency_key": "skip:stable",
        "reason_code": "missing_round",
        "source_file": "first.csv",
        "source_row_number": 2,
        "source_row_sha256": "a" * 64,
        "source_row_json": '{"first":true}',
        "import_batch_id": "11111111-1111-1111-1111-111111111111",
    }]).unique_records[0]
    second = RecordBatch.from_records("ops.skip_events", [{
        "idempotency_key": "skip:stable",
        "reason_code": "missing_round",
        "source_file": "second.csv",
        "source_row_number": 99,
        "source_row_sha256": "b" * 64,
        "source_row_json": '{"second":true}',
        "import_batch_id": "22222222-2222-2222-2222-222222222222",
    }]).unique_records[0]
    changed = RecordBatch.from_records("ops.skip_events", [{
        **second,
        "record_sha256": "",
        "reason_code": "missing_surface",
    }]).unique_records[0]

    assert first["record_sha256"] == second["record_sha256"]
    assert changed["record_sha256"] != first["record_sha256"]


def test_record_batch_rejects_conflicting_duplicate_idempotency_keys():
    batch = RecordBatch.from_records("ops.skip_events", [
        {"idempotency_key": "skip:collision", "reason_code": "missing_round"},
        {"idempotency_key": "skip:collision", "reason_code": "missing_surface"},
    ])

    with pytest.raises(ValueError, match="contradictory normalized facts"):
        _ = batch.unique_records
