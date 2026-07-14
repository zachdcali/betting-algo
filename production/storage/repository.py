"""Psycopg-compatible repository for the operational Postgres contract.

The repository never opens, commits, rolls back, or closes a connection.  The
caller owns the transaction so a complete run (or import batch) can atomically
span all raw/ops/ml writes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Protocol

from .records import RecordBatch


class Cursor(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any: ...
    def executemany(self, query: str, params_seq: Iterable[tuple[Any, ...]]) -> Any: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...
    def close(self) -> Any: ...


class Connection(Protocol):
    def cursor(self) -> Cursor: ...


@dataclass(frozen=True)
class TablePolicy:
    table: str
    mode: str = "immutable"
    status_column: str | None = None
    terminal_statuses: tuple[str, ...] = ()
    terminal_hash_column: str = "record_sha256"
    json_columns: frozenset[str] = frozenset({
        "source_row_json", "metrics", "request",
        # Shadow prediction feature/status/error/backfill detail is structured
        # evidence, not an opaque text blob.
        "metadata",
        "feature_names",
        "feature_contract", "feature_vector", "defaulted_features", "evidence",
        "context", "market_payload", "source_manifest", "row_counts",
        "field_provenance", "registry_entry", "candidate_record",
        "resolution_mapping",
    })


# This allow-list is the security boundary for dynamic identifiers. Values are
# always bound with psycopg's %s parameters; no source value enters SQL text.
TABLE_POLICIES: dict[str, TablePolicy] = {
    "ops.import_batches": TablePolicy(
        "ops.import_batches", "terminal", "status",
        ("applied", "verified", "failed", "cancelled", "canceled"),
        # Import-batch identity is the normalized target manifest. Its
        # completion timestamp is expected to differ on an exact re-apply, so
        # the manifest digest is the stable semantic conflict key.
        "manifest_sha256",
    ),
    "ops.import_batch_memberships": TablePolicy(
        "ops.import_batch_memberships"
    ),
    "ops.import_conflicts": TablePolicy("ops.import_conflicts"),
    "ops.import_conflict_resolutions": TablePolicy(
        "ops.import_conflict_resolutions"
    ),
    "ops.pipeline_runs": TablePolicy(
        "ops.pipeline_runs", "terminal", "status",
        ("success", "partial", "failed", "cancelled", "canceled"),
    ),
    "ops.pipeline_run_stages": TablePolicy(
        "ops.pipeline_run_stages", "terminal", "status",
        ("success", "partial", "failed", "skipped", "cancelled", "canceled"),
    ),
    "raw.source_fetches": TablePolicy(
        "raw.source_fetches", "terminal", "status",
        ("success", "partial", "failed", "blocked", "cancelled", "canceled"),
    ),
    "raw.source_artifacts": TablePolicy("raw.source_artifacts"),
    "ops.odds_observations": TablePolicy("ops.odds_observations"),
    "ops.match_metadata_observations": TablePolicy(
        "ops.match_metadata_observations"
    ),
    "ops.eligibility_match_round_observations": TablePolicy(
        "ops.eligibility_match_round_observations"
    ),
    "ops.eligibility_generations": TablePolicy("ops.eligibility_generations"),
    "ops.eligibility_generation_status_events": TablePolicy(
        "ops.eligibility_generation_status_events"
    ),
    "ops.player_entities": TablePolicy("ops.player_entities"),
    "ops.player_identity_observations": TablePolicy(
        "ops.player_identity_observations"
    ),
    "ops.player_alias_observations": TablePolicy(
        "ops.player_alias_observations"
    ),
    "ops.player_profile_observations": TablePolicy(
        "ops.player_profile_observations"
    ),
    "ops.eligibility_review_events": TablePolicy(
        "ops.eligibility_review_events"
    ),
    "ml.feature_schemas": TablePolicy("ml.feature_schemas"),
    "ml.feature_snapshots": TablePolicy("ml.feature_snapshots"),
    "ml.model_registry_generations": TablePolicy(
        "ml.model_registry_generations"
    ),
    "ml.model_releases": TablePolicy("ml.model_releases"),
    "ml.model_release_status_events": TablePolicy(
        "ml.model_release_status_events"
    ),
    "ml.prediction_observations": TablePolicy("ml.prediction_observations"),
    "ops.paper_accounts": TablePolicy("ops.paper_accounts"),
    "ops.paper_sessions": TablePolicy("ops.paper_sessions"),
    "ops.account_ledger": TablePolicy("ops.account_ledger"),
    "ops.bet_recommendations": TablePolicy("ops.bet_recommendations"),
    "ops.bet_state_events": TablePolicy("ops.bet_state_events"),
    "ops.settlement_attempts": TablePolicy("ops.settlement_attempts"),
    "ops.settlement_events": TablePolicy("ops.settlement_events"),
    "ops.skip_events": TablePolicy("ops.skip_events"),
}

# Generation sealing makes write order part of the contract. Stable sorting
# preserves candidate->accepted event order while preventing lexicographic
# import plans from sealing before their facts and review decisions exist.
ELIGIBILITY_BATCH_WRITE_PRIORITY: dict[str, int] = {
    "ops.eligibility_generations": 10,
    "raw.source_fetches": 20,
    "raw.source_artifacts": 30,
    "ops.player_entities": 40,
    "ops.player_identity_observations": 50,
    "ops.player_alias_observations": 50,
    "ops.player_profile_observations": 50,
    "ops.eligibility_match_round_observations": 50,
    "ops.eligibility_review_events": 60,
    "ops.eligibility_generation_status_events": 70,
    "ops.import_batch_memberships": 100,
}
ELIGIBILITY_STATUS_PRIORITY = {
    "candidate": 10,
    "accepted": 20,
    "rejected": 20,
    "retired": 30,
}

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
PROVENANCE_COLUMNS = frozenset({
    "import_batch_id", "source_file", "source_row_number",
    "source_row_sha256", "source_row_json",
})


@dataclass(frozen=True)
class RepositoryInventory:
    table: str
    import_batch_id: str
    rows_by_key: Mapping[str, str | None]

    @property
    def count(self) -> int:
        return len(self.rows_by_key)


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _quote_literal(value: str) -> str:
    """Quote a repository-owned SQL literal.

    Terminal statuses are static allow-list policy, never source data. Keeping
    them in the statement lets every terminal assignment share one predicate
    without duplicating bound parameters for every column.
    """
    return "'" + value.replace("'", "''") + "'"


def _quoted_table(table: str) -> str:
    if table not in TABLE_POLICIES:
        raise ValueError(f"table is not in the operational contract: {table}")
    schema, name = table.split(".", 1)
    return f"{_quote_identifier(schema)}.{_quote_identifier(name)}"


def _insert_sql(policy: TablePolicy, columns: tuple[str, ...]) -> str:
    if "idempotency_key" not in columns:
        raise ValueError(f"{policy.table} records require idempotency_key")
    table = _quoted_table(policy.table)
    rendered_columns = ", ".join(_quote_identifier(column) for column in columns)
    placeholders = ", ".join(
        "%s::jsonb" if column in policy.json_columns else "%s" for column in columns
    )
    base = f'INSERT INTO {table} AS "existing" ({rendered_columns}) VALUES ({placeholders})'
    if policy.mode == "immutable":
        # Exact retries are accepted without mutating the stored fact. Reusing
        # an idempotency key for different normalized content raises inside
        # PostgreSQL, so the runtime sink cannot silently claim durability
        # after retaining a contradictory first write.
        return (
            base
            + ' ON CONFLICT ("idempotency_key") DO UPDATE SET '
            + '"record_sha256" = ops.require_matching_record_sha256('
            + '"existing"."record_sha256", EXCLUDED."record_sha256", '
            + f"'{policy.table}', EXCLUDED.\"idempotency_key\")"
        )

    updates = [column for column in columns if column != "idempotency_key"]
    if not updates:
        return base + ' ON CONFLICT ("idempotency_key") DO NOTHING'
    if policy.mode == "terminal":
        status = _quote_identifier(policy.status_column or "status")
        hash_column = policy.terminal_hash_column
        if hash_column not in columns:
            raise ValueError(
                f"{policy.table} terminal records require {hash_column}"
            )
        quoted_hash = _quote_identifier(hash_column)
        terminals = ", ".join(
            _quote_literal(value) for value in policy.terminal_statuses
        )
        is_terminal = (
            f"lower(COALESCE(\"existing\".{status}, '')) IN ({terminals})"
        )
        assignments = []
        for column in updates:
            quoted = _quote_identifier(column)
            if column == hash_column:
                terminal_value = (
                    "ops.require_matching_record_sha256("
                    f'"existing".{quoted_hash}, EXCLUDED.{quoted_hash}, '
                    f"'{policy.table}', EXCLUDED.\"idempotency_key\")"
                )
            else:
                # Once terminal, preserve the exact stored row. This makes an
                # equal-hash retry a semantic no-op even when excluded import
                # provenance or completion timestamps differ.
                terminal_value = f'"existing".{quoted}'
            assignments.append(
                f"{quoted} = CASE WHEN {is_terminal} "
                f"THEN {terminal_value} ELSE EXCLUDED.{quoted} END"
            )
        update_sql = ", ".join(assignments)
    else:
        update_sql = ", ".join(
            f'{_quote_identifier(column)} = EXCLUDED.{_quote_identifier(column)}'
            for column in updates
        )
    return base + ' ON CONFLICT ("idempotency_key") DO UPDATE SET ' + update_sql


class OperationalRepository:
    """Write/query the contract using a caller-owned psycopg connection."""

    def __init__(self, connection: Connection):
        self.connection = connection

    def write_batch(self, batch: RecordBatch) -> int:
        policy = TABLE_POLICIES.get(batch.table)
        if policy is None:
            raise ValueError(f"table is not in the operational contract: {batch.table}")
        unique = batch.unique_records
        if not unique:
            return 0
        if batch.table == "ops.eligibility_generation_status_events":
            unique = tuple(sorted(unique, key=lambda record: (
                str(record.get("effective_at") or ""),
                ELIGIBILITY_STATUS_PRIORITY.get(
                    str(record.get("status") or "").lower(), 99,
                ),
                str(record.get("idempotency_key") or ""),
            )))

        # Different legacy sources can expose optional fields. Grouping by the
        # exact column tuple keeps every statement typed without filling absent
        # columns with NULL and accidentally defeating database defaults.
        grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for record in unique:
            columns = tuple(sorted(str(column) for column in record))
            for column in columns:
                _quote_identifier(column)
            grouped[columns].append(record)

        cursor = self.connection.cursor()
        try:
            for columns, records in grouped.items():
                statement = _insert_sql(policy, columns)
                params = [tuple(record[column] for column in columns) for record in records]
                cursor.executemany(statement, params)
        finally:
            cursor.close()
        return len(unique)

    def write_batches(self, batches: Iterable[RecordBatch]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        expanded: list[RecordBatch] = []
        for batch in batches:
            if batch.table == "ops.eligibility_generation_status_events":
                expanded.extend(
                    RecordBatch(batch.table, (record,), batch.source_files)
                    for record in batch.unique_records
                )
            else:
                expanded.append(batch)
        indexed = list(enumerate(expanded))
        if any(
            batch.table in ELIGIBILITY_BATCH_WRITE_PRIORITY
            and batch.table.startswith(("ops.eligibility", "ops.player_"))
            for _, batch in indexed
        ):
            def write_order(item: tuple[int, RecordBatch]) -> tuple[Any, ...]:
                index, batch = item
                priority = ELIGIBILITY_BATCH_WRITE_PRIORITY.get(batch.table, 90)
                if batch.table != "ops.eligibility_generation_status_events":
                    return (priority, "", 0, index)
                records = batch.unique_records
                first = min(records, key=lambda record: (
                    str(record.get("effective_at") or ""),
                    ELIGIBILITY_STATUS_PRIORITY.get(
                        str(record.get("status") or "").lower(), 99,
                    ),
                )) if records else {}
                return (
                    priority,
                    str(first.get("effective_at") or ""),
                    ELIGIBILITY_STATUS_PRIORITY.get(
                        str(first.get("status") or "").lower(), 99,
                    ),
                    index,
                )

            indexed.sort(key=write_order)
        for _, batch in indexed:
            counts[batch.table] += self.write_batch(batch)
        return dict(counts)

    def inventory(
        self,
        table: str,
        import_batch_id: str,
        *,
        idempotency_keys: Iterable[str] | None = None,
    ) -> RepositoryInventory:
        """Return semantic target hashes for parity proof.

        When explicit keys are supplied, inventory is global rather than tied
        to the row's first import batch. This is what makes an additive second
        manifest safe: an unchanged target fact can participate in many import
        batches without rewriting its original provenance.
        """
        quoted = _quoted_table(table)
        cursor = self.connection.cursor()
        try:
            if idempotency_keys is None:
                cursor.execute(
                    f'SELECT "idempotency_key", "record_sha256" '
                    f'FROM {quoted} WHERE "import_batch_id" = %s',
                    (import_batch_id,),
                )
            else:
                keys = list(idempotency_keys)
                cursor.execute(
                    f'SELECT "idempotency_key", "record_sha256" '
                    f'FROM {quoted} WHERE "idempotency_key" = ANY(%s)',
                    (keys,),
                )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return RepositoryInventory(
            table=table,
            import_batch_id=import_batch_id,
            rows_by_key={str(key): (None if digest is None else str(digest)) for key, digest in rows},
        )

    def membership_inventory(
        self, table: str, import_batch_id: str
    ) -> RepositoryInventory:
        """Return the batch-to-target membership ledger for one table."""
        _quoted_table(table)  # validate target table through the allow-list
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                'SELECT "target_idempotency_key", "target_record_sha256" '
                'FROM "ops"."import_batch_memberships" '
                'WHERE "import_batch_id" = %s AND "target_table" = %s',
                (import_batch_id, table),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return RepositoryInventory(
            table=table,
            import_batch_id=import_batch_id,
            rows_by_key={
                str(key): (None if digest is None else str(digest))
                for key, digest in rows
            },
        )
