from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from storage.repository import TABLE_POLICIES  # noqa: E402
from versioning import OPERATIONAL_SCHEMA_VERSION  # noqa: E402


MIGRATIONS = (
    ROOT / "supabase" / "migrations" / "20260714010000_operational_schema_v1.sql",
    ROOT / "supabase" / "migrations" / "20260714020000_operational_integrity_v1_1.sql",
)


def _sql() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in MIGRATIONS)


def _created_tables(sql: str) -> set[str]:
    return {
        f"{schema}.{table}"
        for schema, table in re.findall(
            r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+"
            r"(raw|ops|ml)\.([a-z][a-z0-9_]*)",
            sql,
            flags=re.IGNORECASE,
        )
    }


def test_operational_migration_is_additive_versioned_and_idempotent():
    for path in MIGRATIONS:
        sql = path.read_text(encoding="utf-8")
        upper = sql.upper()

        assert "BEGIN;" in upper and "COMMIT;" in upper
        assert "DROP TABLE" not in upper
        assert "TRUNCATE" not in upper
        assert "CREATE TABLE IF NOT EXISTS" in upper
        assert "CREATE INDEX IF NOT EXISTS" in upper
        assert "ON CONFLICT (VERSION) DO NOTHING" in upper
    assert "VALUES ('1.0.0'" in MIGRATIONS[0].read_text(encoding="utf-8")
    assert (
        f"VALUES ('{OPERATIONAL_SCHEMA_VERSION}'"
        in MIGRATIONS[-1].read_text(encoding="utf-8")
    )


def test_every_repository_table_exists_in_the_migration():
    created = _created_tables(_sql())
    assert set(TABLE_POLICIES).issubset(created)


def test_contract_includes_durable_metadata_sessions_and_private_projections():
    sql = _sql()

    for table in (
        "ops.match_metadata_observations",
        "ops.paper_sessions",
        "raw.source_fetches",
        "raw.source_artifacts",
        "ops.odds_observations",
        "ml.feature_snapshots",
        "ml.prediction_observations",
        "ops.bet_state_events",
        "ops.settlement_events",
    ):
        assert table in sql

    assert "CREATE OR REPLACE VIEW api.current_match_metadata" in sql
    assert "CREATE OR REPLACE VIEW api.paper_account_state" in sql
    assert "REVOKE ALL ON SCHEMA raw, ops, ml, api FROM PUBLIC" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "no browser role receives" in sql


def test_integrity_migration_enforces_multibatch_and_decision_contracts():
    sql = MIGRATIONS[-1].read_text(encoding="utf-8")

    for marker in (
        "ops.import_batch_memberships",
        "ml.model_releases",
        "ml.model_registry_generations",
        "model_releases_model_version_semver_check",
        "model_releases_calibration_version_semver_check",
        "model_releases_family_check",
        "model_releases_feature_schema_fkey",
        "model_releases_contract_complete_check",
        "block_unlogged_calibrated_promotion",
        "record_sha256",
        "require_matching_record_sha256",
        "guard_terminal_status_update",
        "validate_feature_snapshot",
        "validate_prediction_eligibility",
        "validate_bet_evidence",
        "validate_bet_state_transition",
        "validate_settlement_correction",
        "decision_eligible",
        "inference_eligible",
        "supersedes_settlement_event_id",
    ):
        assert marker in sql


def test_registry_status_uses_explicit_monotonic_generation_contract():
    sql = MIGRATIONS[-1].read_text(encoding="utf-8")

    generation_start = sql.index(
        "CREATE TABLE IF NOT EXISTS ml.model_registry_generations ("
    )
    generation_end = sql.index("\n);", generation_start)
    generation_block = sql[generation_start:generation_end]

    assert "registry_generation_sha256 text NOT NULL UNIQUE" in generation_block
    assert "generation_sequence bigint NOT NULL UNIQUE" in generation_block
    assert "CHECK (generation_sequence > 0)" in generation_block
    assert "effective_at timestamptz NOT NULL" in generation_block
    assert "record_sha256 text NOT NULL" in generation_block
    assert "model_release_status_events_generation_fkey" in sql
    assert "REFERENCES ml.model_registry_generations(registry_generation_sha256)" in sql
    assert "model_release_status_one_per_generation_idx" in sql
    assert "model_release_key, registry_generation_sha256" in sql
    assert "model_release_status_events_release_family_fkey" in sql
    assert "FOREIGN KEY (model_release_key, model_family)" in sql
    assert "model_release_one_promoted_per_family_generation_idx" in sql
    assert "registry_generation_sha256, model_family" in sql
    assert "WHERE release_status = 'promoted'" in sql
    assert sql.count("SELECT max(latest.generation_sequence)") == 2
    assert "LEFT JOIN LATERAL" not in sql
    assert (
        "ALTER TABLE ml.model_registry_generations ENABLE ROW LEVEL SECURITY"
        in sql
    )
    assert "'ml.model_registry_generations'::regclass" in sql


def test_immutable_retry_hash_guard_fails_closed_in_postgres():
    sql = MIGRATIONS[-1].read_text(encoding="utf-8")

    assert (
        "CREATE OR REPLACE FUNCTION ops.require_matching_record_sha256("
        in sql
    )
    assert "existing_hash IS DISTINCT FROM incoming_hash" in sql
    assert "idempotency conflict on % for key %" in sql
    assert "IF TG_OP = 'UPDATE' AND NEW IS NOT DISTINCT FROM OLD" in sql
    assert (
        "REVOKE ALL ON FUNCTION "
        "ops.require_matching_record_sha256(text, text, text, text)"
        in sql
    )


def test_correctness_constraints_fail_closed_without_erasing_error_evidence():
    sql = _sql()

    assert "player1_decimal_odds IS NULL OR player1_decimal_odds > 1" in sql
    assert "actual_winner IS NULL OR actual_winner IN (1, 2)" in sql
    assert "(player1_probability IS NULL) = (player2_probability IS NULL)" in sql
    assert "NOT features_complete OR" in sql
    assert "lower(build_status) = 'ok'" in sql
    assert "num_nonnulls(match_date, match_start_at_utc, tournament, event_title" in sql


def test_imported_tables_retain_row_level_source_provenance():
    sql = _sql()
    # Import controls and later human conflict-resolution decisions are not
    # normalized source facts; every other repository table retains row-level
    # source evidence.
    non_imported = {"ops.import_batches", "ops.import_conflict_resolutions"}
    for table in sorted(set(TABLE_POLICIES) - non_imported):
        schema, name = table.split(".")
        marker = f"CREATE TABLE IF NOT EXISTS {schema}.{name} ("
        start = sql.index(marker)
        end = sql.index("\n);", start)
        block = sql[start:end]
        for column in (
            "idempotency_key",
            "import_batch_id",
            "source_file",
            "source_row_number",
            "source_row_sha256",
            "source_row_json",
        ):
            assert column in block, f"{table} missing {column}"
