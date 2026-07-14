from pathlib import Path
import sys

import pytest

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

import storage.runtime as runtime  # noqa: E402
from storage.records import RecordBatch  # noqa: E402
from versioning import OPERATIONAL_SCHEMA_VERSION  # noqa: E402


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self._schema_row = None

    def execute(self, query, params=()):
        self.connection.execute_calls.append((query, params))
        if "ops.schema_versions" in query:
            if self.connection.schema_versions:
                latest = max(
                    self.connection.schema_versions,
                    key=lambda value: tuple(int(part) for part in value.split(".")),
                )
                self._schema_row = (latest,)

    def executemany(self, query, params_seq):
        parameters = list(params_seq)
        self.connection.executemany_calls.append((query, parameters))
        if self.connection.write_error is not None:
            raise self.connection.write_error

    def fetchone(self):
        return self._schema_row

    def close(self):
        self.connection.closed_cursors += 1


class FakeConnection:
    def __init__(
        self,
        *,
        schema_version=OPERATIONAL_SCHEMA_VERSION,
        schema_versions=None,
        write_error=None,
    ):
        self.schema_versions = (
            list(schema_versions)
            if schema_versions is not None
            else ([] if schema_version is None else [schema_version])
        )
        self.write_error = write_error
        self.execute_calls = []
        self.executemany_calls = []
        self.closed_cursors = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def cursor(self):
        return FakeCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        if exc_type is None:
            self.commits += 1
        else:
            self.rollbacks += 1
        self.closed += 1
        return False


class FakeFactory:
    def __init__(self, connection):
        self.connection = connection
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        return self.connection


def _batch():
    return RecordBatch.from_records("ops.skip_events", [{
        "idempotency_key": "skip:1",
        "external_skip_event_id": "skip_1",
        "stage_name": "feature_extraction",
        "reason_code": "missing_start",
        "context": "{}",
    }])


def test_default_off_mode_does_not_consume_batches_or_connect(monkeypatch):
    consumed = []

    def batches():
        consumed.append(True)
        yield _batch()

    def forbidden_connect(_url):
        raise AssertionError("off mode must not connect")

    sink = runtime.OperationalRuntimeSink(connection_factory=forbidden_connect)
    result = sink.write(batches())

    assert sink.mode is runtime.SinkMode.OFF
    assert not result.attempted
    assert result.succeeded
    assert consumed == []


def test_environment_off_ignores_both_database_urls():
    factory = FakeFactory(FakeConnection())
    sink = runtime.OperationalRuntimeSink.from_environment({
        "DATABASE_URL": "postgresql://canonical-secret",
        "OPERATIONAL_DATABASE_URL": "postgresql://operational-secret",
    }, connection_factory=factory)

    sink.write([_batch()])

    assert factory.urls == []
    assert sink._database_url == ""


@pytest.mark.parametrize("mode", ["shadow", "required"])
def test_enabled_mode_requires_dedicated_url_and_never_falls_back(mode):
    with pytest.raises(runtime.SinkConfigurationError, match="OPERATIONAL_DATABASE_URL"):
        runtime.OperationalRuntimeSink.from_environment({
            "OPERATIONAL_SINK_MODE": mode,
            "DATABASE_URL": "postgresql://must-not-be-used",
        })


def test_invalid_mode_fails_closed():
    with pytest.raises(runtime.SinkConfigurationError, match="expected one of"):
        runtime.OperationalRuntimeSink.from_environment({
            "OPERATIONAL_SINK_MODE": "best_effort",
        })


def test_required_write_verifies_schema_and_commits_all_batches_atomically():
    connection = FakeConnection(schema_versions=["1.0.0", OPERATIONAL_SCHEMA_VERSION])
    factory = FakeFactory(connection)
    sink = runtime.OperationalRuntimeSink(
        mode="required", database_url="postgresql://operational-only",
        connection_factory=factory,
    )

    result = sink.write([_batch(), _batch()])

    assert result.succeeded
    assert result.schema_verified
    assert result.record_counts == {"ops.skip_events": 2}
    assert factory.urls == ["postgresql://operational-only"]
    schema_query, schema_params = connection.execute_calls[0]
    assert "ORDER BY string_to_array(version, '.')::integer[] DESC" in schema_query
    assert "LIMIT 1" in schema_query
    assert schema_params == ()
    assert len(connection.executemany_calls) == 2
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed == 1


def test_shadow_schema_mismatch_reports_and_rolls_back_nonfatally():
    connection = FakeConnection(schema_version=None)
    reports = []
    sink = runtime.OperationalRuntimeSink(
        mode="shadow", database_url="postgresql://operational-only",
        connection_factory=FakeFactory(connection), reporter=reports.append,
    )

    result = sink.write([_batch()])

    assert not result.succeeded
    assert result.error_type == "OperationalSchemaError"
    assert not result.schema_verified
    assert connection.executemany_calls == []
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert len(reports) == 1
    assert "CSV authority continues" in reports[0]
    assert "postgresql://" not in reports[0]


@pytest.mark.parametrize(
    ("schema_versions", "reported_latest"),
    [
        (["1.0.0"], "1.0.0"),
        (["1.0.0", OPERATIONAL_SCHEMA_VERSION, "1.2.0"], "1.2.0"),
    ],
)
def test_required_schema_gate_rejects_any_noncurrent_latest_version(
    schema_versions, reported_latest
):
    connection = FakeConnection(schema_versions=schema_versions)
    sink = runtime.OperationalRuntimeSink(
        mode="required",
        database_url="postgresql://operational-only",
        connection_factory=FakeFactory(connection),
    )

    with pytest.raises(
        runtime.OperationalSchemaError,
        match=(
            rf"required latest version {OPERATIONAL_SCHEMA_VERSION}.*"
            rf"database latest is {reported_latest}"
        ),
    ):
        sink.write([_batch()])

    assert connection.executemany_calls == []
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed == 1


def test_shadow_write_error_rolls_back_whole_sequence_and_returns_error():
    connection = FakeConnection(write_error=RuntimeError("simulated write failure"))
    reports = []
    sink = runtime.OperationalRuntimeSink(
        mode="shadow", database_url="postgresql://operational-only",
        connection_factory=FakeFactory(connection), reporter=reports.append,
    )

    result = sink.write([_batch(), _batch()])

    assert not result.succeeded
    assert result.error_type == "RuntimeError"
    assert result.schema_verified
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed == 1
    assert reports and "simulated write failure" in reports[0]


def test_shadow_terminal_hash_conflict_cannot_report_durable_success():
    conflict = RuntimeError(
        "idempotency conflict on ops.pipeline_runs for key pipeline_run:1: "
        "record_sha256 mismatch"
    )
    connection = FakeConnection(write_error=conflict)
    reports = []
    sink = runtime.OperationalRuntimeSink(
        mode="shadow", database_url="postgresql://operational-only",
        connection_factory=FakeFactory(connection), reporter=reports.append,
    )

    result = sink.write([_batch()])

    assert not result.succeeded
    assert result.record_counts == {}
    assert result.submitted_rows == 0
    assert result.error_type == "RuntimeError"
    assert "record_sha256 mismatch" in result.error_message
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert reports and "record_sha256 mismatch" in reports[0]


def test_shadow_redacts_database_url_and_password_from_result_and_report():
    secret_url = "postgresql://operator:super-secret@db.example.test/operations"
    reports = []

    def failing_factory(url):
        raise RuntimeError(f"could not connect with {url}; password=super-secret")

    sink = runtime.OperationalRuntimeSink(
        mode="shadow", database_url=secret_url,
        connection_factory=failing_factory, reporter=reports.append,
    )

    result = sink.write([_batch()])

    assert not result.succeeded
    assert "super-secret" not in result.error_message
    assert secret_url not in result.error_message
    assert "<redacted operational database URL>" in result.error_message
    assert reports
    assert "super-secret" not in reports[0]
    assert secret_url not in reports[0]


def test_shadow_captures_batch_iterator_errors_without_connecting():
    reports = []
    factory = FakeFactory(FakeConnection())

    def broken_batches():
        raise RuntimeError("batch construction failed")
        yield _batch()

    sink = runtime.OperationalRuntimeSink(
        mode="shadow", database_url="postgresql://operational-only",
        connection_factory=factory, reporter=reports.append,
    )

    result = sink.write(broken_batches())

    assert not result.succeeded
    assert result.error_type == "RuntimeError"
    assert factory.urls == []
    assert reports and "batch construction failed" in reports[0]


def test_required_error_propagates_after_atomic_rollback():
    connection = FakeConnection(write_error=RuntimeError("durable write failed"))
    sink = runtime.OperationalRuntimeSink(
        mode="required", database_url="postgresql://operational-only",
        connection_factory=FakeFactory(connection),
    )

    with pytest.raises(RuntimeError, match="durable write failed"):
        sink.write([_batch()])

    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed == 1


def test_psycopg_loader_is_not_touched_while_off(monkeypatch):
    monkeypatch.setattr(
        runtime, "_psycopg_connect",
        lambda _url: (_ for _ in ()).throw(AssertionError("must stay lazy")),
    )
    sink = runtime.OperationalRuntimeSink(mode="off")

    assert sink.write([_batch()]).succeeded
