"""Value objects and canonical hashing for operational imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

RECORD_HASH_EXCLUDED_COLUMNS = frozenset({
    "record_sha256",
    "import_batch_id",
    "source_file",
    "source_row_number",
    "source_row_sha256",
    "source_row_json",
    # Full registry prose/metrics are generation evidence. Immutable artifact
    # identity is carried by the normalized release columns and hashes.
    "registry_entry",
})


def json_compatible(value: Any) -> JsonValue:
    """Convert supported typed values to deterministic JSON primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        normalized = value
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return byte-stable JSON used by row hashes and idempotency keys."""
    return json.dumps(
        json_compatible(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_key(namespace: str, *parts: Any) -> str:
    """Build a readable deterministic idempotency key.

    A hash keeps potentially sensitive names and free text out of unique keys.
    The namespace remains visible for database and incident triage.
    """
    digest = content_sha256([json_compatible(part) for part in parts])
    return f"{namespace}:{digest}"


def record_sha256(record: Mapping[str, Any]) -> str:
    """Hash the normalized target fact, excluding import provenance.

    Source rows can gain settlement or projection columns without changing the
    immutable target fact. This fingerprint lets a later import batch prove
    that an existing idempotency key still represents the same normalized
    record while preserving each batch's distinct source-row evidence.
    """
    return content_sha256({
        str(key): value
        for key, value in record.items()
        if str(key) not in RECORD_HASH_EXCLUDED_COLUMNS
    })


@dataclass(frozen=True)
class RecordBatch:
    """Typed rows destined for exactly one allow-listed database table."""

    table: str
    records: tuple[Mapping[str, Any], ...]
    source_files: tuple[str, ...] = ()

    @classmethod
    def from_records(
        cls,
        table: str,
        records: Iterable[Mapping[str, Any]],
        *,
        source_files: Iterable[str] = (),
    ) -> "RecordBatch":
        prepared: list[dict[str, Any]] = []
        for source in records:
            record = dict(source)
            if table != "ops.import_batches":
                computed = record_sha256(record)
                supplied = str(record.get("record_sha256") or "").strip()
                if supplied and supplied != computed:
                    raise ValueError(
                        f"{table} record_sha256 does not match normalized content"
                    )
                record["record_sha256"] = computed
            prepared.append(record)
        return cls(table, tuple(prepared), tuple(source_files))

    @property
    def unique_records(self) -> tuple[Mapping[str, Any], ...]:
        """Deduplicate identical facts and reject contradictory key collisions."""
        seen: dict[str, str] = {}
        result: list[Mapping[str, Any]] = []
        for record in self.records:
            key = str(record.get("idempotency_key", ""))
            if not key:
                raise ValueError(f"{self.table} row is missing idempotency_key")
            digest = str(record.get("record_sha256") or content_sha256(record))
            if key not in seen:
                result.append(record)
                seen[key] = digest
            elif seen[key] != digest:
                raise ValueError(
                    f"{self.table} idempotency key {key!r} maps to "
                    "contradictory normalized facts"
                )
        return tuple(result)


@dataclass(frozen=True)
class ImportPlan:
    """A read-only import plan; construction never changes a source or DB."""

    import_batch_id: str
    production_dir: Path
    batches: tuple[RecordBatch, ...]
    file_sha256: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for batch in self.batches:
            counts[batch.table] = counts.get(batch.table, 0) + len(batch.unique_records)
        return counts

    @property
    def total_rows(self) -> int:
        return sum(self.row_counts.values())

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "plan",
            "import_batch_id": self.import_batch_id,
            "production_dir": str(self.production_dir),
            "row_counts": dict(sorted(self.row_counts.items())),
            "total_rows": self.total_rows,
            "source_files": dict(sorted(self.file_sha256.items())),
            "warnings": list(self.warnings),
        }
