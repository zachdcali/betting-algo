"""Exact source-to-Postgres parity reporting for one import batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .records import ImportPlan
from .repository import OperationalRepository


@dataclass(frozen=True)
class TableParity:
    table: str
    source_count: int
    repository_count: int
    missing_keys: tuple[str, ...]
    extra_keys: tuple[str, ...]
    hash_mismatches: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.missing_keys and not self.extra_keys and not self.hash_mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "source_count": self.source_count,
            "repository_count": self.repository_count,
            "matches": self.matches,
            "missing_keys": list(self.missing_keys),
            "extra_keys": list(self.extra_keys),
            "hash_mismatches": list(self.hash_mismatches),
        }


@dataclass(frozen=True)
class ParityReport:
    import_batch_id: str
    tables: tuple[TableParity, ...]

    @property
    def matches(self) -> bool:
        return all(table.matches for table in self.tables)

    def as_dict(self) -> dict[str, Any]:
        return {
            "import_batch_id": self.import_batch_id,
            "matches": self.matches,
            "tables": [table.as_dict() for table in self.tables],
        }


def source_inventory(plan: ImportPlan) -> dict[str, dict[str, str | None]]:
    """Canonical semantic target hashes for every planned fact."""
    result: dict[str, dict[str, str | None]] = {}
    for batch in plan.batches:
        # The batch control row describes the import rather than a source CSV
        # row and therefore has no legacy source hash to compare.
        if batch.table == "ops.import_batches":
            continue
        table = result.setdefault(batch.table, {})
        for record in batch.unique_records:
            key = str(record["idempotency_key"])
            if key not in table:
                digest = record.get("record_sha256")
                table[key] = None if digest is None else str(digest)
    return result


def compare_plan(repository: OperationalRepository, plan: ImportPlan) -> ParityReport:
    tables: list[TableParity] = []
    for table, source_rows in sorted(source_inventory(plan).items()):
        stored = repository.inventory(
            table,
            plan.import_batch_id,
            idempotency_keys=source_rows,
        ).rows_by_key
        source_keys = set(source_rows)
        stored_keys = set(stored)
        shared = source_keys & stored_keys
        tables.append(TableParity(
            table=table,
            source_count=len(source_rows),
            repository_count=len(stored),
            missing_keys=tuple(sorted(source_keys - stored_keys)),
            extra_keys=tuple(sorted(stored_keys - source_keys)),
            hash_mismatches=tuple(sorted(
                key for key in shared if source_rows[key] != stored[key]
            )),
        ))
    return ParityReport(plan.import_batch_id, tuple(tables))


def compare_memberships(
    repository: OperationalRepository, plan: ImportPlan
) -> ParityReport:
    """Prove every planned target is linked to this exact import batch."""
    tables: list[TableParity] = []
    for table, source_rows in sorted(source_inventory(plan).items()):
        stored = repository.membership_inventory(
            table, plan.import_batch_id
        ).rows_by_key
        source_keys = set(source_rows)
        stored_keys = set(stored)
        shared = source_keys & stored_keys
        tables.append(TableParity(
            table=table,
            source_count=len(source_rows),
            repository_count=len(stored),
            missing_keys=tuple(sorted(source_keys - stored_keys)),
            extra_keys=tuple(sorted(stored_keys - source_keys)),
            hash_mismatches=tuple(sorted(
                key for key in shared if source_rows[key] != stored[key]
            )),
        ))
    return ParityReport(plan.import_batch_id, tuple(tables))
