"""Opt-in runtime sink for the normalized operational Postgres contract.

This module is intentionally not wired into ``main.py`` yet.  The default mode
is ``off`` and does not import psycopg, inspect credentials, consume batches, or
open a connection.  Enabled modes use only ``OPERATIONAL_DATABASE_URL``:

``shadow``
    Attempt one atomic write, report an error, and let the CSV pipeline continue.

``required``
    Attempt one atomic write and propagate any configuration, schema, or write
    error so the caller cannot claim a successful durable run.

Every enabled connection verifies the exact operational schema version before
writing.  ``DATABASE_URL`` is deliberately never read; the canonical store and
the normalized operational store have separate rollout and safety boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import urlsplit

try:
    from versioning import OPERATIONAL_SCHEMA_VERSION
except ImportError:  # pragma: no cover - package-style execution
    from production.versioning import OPERATIONAL_SCHEMA_VERSION  # type: ignore

from .records import RecordBatch
from .repository import OperationalRepository


MODE_ENV = "OPERATIONAL_SINK_MODE"
DATABASE_URL_ENV = "OPERATIONAL_DATABASE_URL"


class SinkMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    REQUIRED = "required"

    @classmethod
    def parse(cls, value: str | "SinkMode" | None) -> "SinkMode":
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.OFF.value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in cls)
            raise SinkConfigurationError(
                f"invalid {MODE_ENV}={value!r}; expected one of: {allowed}"
            ) from exc


class SinkConfigurationError(ValueError):
    """The opt-in sink is enabled without a safe, complete configuration."""


class OperationalSchemaError(RuntimeError):
    """The connected database does not contain the required schema contract."""


class ConnectionFactory(Protocol):
    def __call__(self, database_url: str) -> Any: ...


Reporter = Callable[[str], None]


@dataclass(frozen=True)
class SinkResult:
    mode: SinkMode
    attempted: bool
    schema_verified: bool
    record_counts: Mapping[str, int] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None

    @property
    def submitted_rows(self) -> int:
        return sum(int(count) for count in self.record_counts.values())


def _default_reporter(message: str) -> None:
    logging.getLogger(__name__).warning(message)


def _psycopg_connect(database_url: str) -> Any:
    """Import psycopg only after an enabled sink is asked to write."""
    import psycopg

    return psycopg.connect(database_url)


def _redacted_error_message(exc: Exception, database_url: str) -> str:
    """Keep connection strings and embedded passwords out of results/logs."""
    message = str(exc)
    if database_url:
        message = message.replace(database_url, "<redacted operational database URL>")
        try:
            password = urlsplit(database_url).password
        except ValueError:
            password = None
        if password:
            message = message.replace(password, "<redacted>")
    return message


class OperationalRuntimeSink:
    """Atomically submit caller-built records to the reviewed schema.

    The connection context owns commit/rollback. ``OperationalRepository`` does
    neither, so schema verification and every batch in a call are one database
    transaction.  The sink never retries: idempotency is a repository contract,
    while retry timing remains an orchestration concern.
    """

    def __init__(
        self,
        *,
        mode: SinkMode | str = SinkMode.OFF,
        database_url: str | None = None,
        connection_factory: ConnectionFactory | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.mode = SinkMode.parse(mode)
        self._database_url = str(database_url or "").strip()
        if self.mode is not SinkMode.OFF and not self._database_url:
            raise SinkConfigurationError(
                f"{DATABASE_URL_ENV} is required when {MODE_ENV}={self.mode.value}"
            )
        self._connection_factory = connection_factory or _psycopg_connect
        self._reporter = reporter or _default_reporter

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        connection_factory: ConnectionFactory | None = None,
        reporter: Reporter | None = None,
    ) -> "OperationalRuntimeSink":
        environment = os.environ if environ is None else environ
        mode = SinkMode.parse(environment.get(MODE_ENV, SinkMode.OFF.value))
        # Never inspect or fall back to DATABASE_URL. Disabled mode also avoids
        # reading an unrelated credential into this object's state.
        database_url = (
            environment.get(DATABASE_URL_ENV) if mode is not SinkMode.OFF else None
        )
        return cls(
            mode=mode,
            database_url=database_url,
            connection_factory=connection_factory,
            reporter=reporter,
        )

    def _verify_schema(self, connection: Any) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute(
                """SELECT version
                   FROM ops.schema_versions
                   ORDER BY string_to_array(version, '.')::integer[] DESC
                   LIMIT 1""",
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
        actual_version = None if row is None else str(row[0])
        if actual_version != OPERATIONAL_SCHEMA_VERSION:
            raise OperationalSchemaError(
                "operational database schema mismatch: required latest version "
                f"{OPERATIONAL_SCHEMA_VERSION}, database latest is "
                f"{actual_version or '<missing>'}"
            )

    def write(self, batches: Iterable[RecordBatch]) -> SinkResult:
        """Submit all batches atomically according to the configured mode."""
        if self.mode is SinkMode.OFF:
            # Do not even tuple() a generator: off really means no sink work.
            return SinkResult(
                mode=self.mode,
                attempted=False,
                schema_verified=False,
            )

        schema_verified = False
        try:
            planned_batches = tuple(batches)
            # Psycopg connection contexts commit on clean exit, roll back on an
            # exception, and close the connection. Test doubles exercise the
            # same ownership boundary.
            with self._connection_factory(self._database_url) as connection:
                self._verify_schema(connection)
                schema_verified = True
                counts = OperationalRepository(connection).write_batches(planned_batches)
            return SinkResult(
                mode=self.mode,
                attempted=True,
                schema_verified=True,
                record_counts=counts,
            )
        except Exception as exc:
            if self.mode is SinkMode.REQUIRED:
                raise
            safe_error = _redacted_error_message(exc, self._database_url)
            message = (
                "operational shadow sink failed; CSV authority continues: "
                f"{type(exc).__name__}: {safe_error}"
            )
            try:
                self._reporter(message)
            except Exception as report_exc:  # reporting must not break shadow mode
                _default_reporter(
                    "operational shadow sink reporter also failed: "
                    f"{type(report_exc).__name__}"
                )
            return SinkResult(
                mode=self.mode,
                attempted=True,
                schema_verified=schema_verified,
                error_type=type(exc).__name__,
                error_message=safe_error,
            )
