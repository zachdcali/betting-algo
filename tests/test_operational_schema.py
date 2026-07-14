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
    ROOT / "supabase" / "migrations" / "20260714030000_eligibility_provenance_v1_2.sql",
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
    sql = MIGRATIONS[1].read_text(encoding="utf-8")

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
    sql = MIGRATIONS[1].read_text(encoding="utf-8")

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
    sql = MIGRATIONS[1].read_text(encoding="utf-8")

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


def test_eligibility_migration_reuses_normalized_authority_and_fails_closed():
    sql = MIGRATIONS[2].read_text(encoding="utf-8")

    for marker in (
        "ops.eligibility_generations",
        "ops.eligibility_generation_status_events",
        "ops.player_entities",
        "ops.player_identity_observations",
        "ops.player_alias_observations",
        "ops.player_profile_observations",
        "ops.eligibility_match_round_observations",
        "ops.eligibility_review_events",
        "api.current_player_identities",
        "api.current_player_aliases",
        "api.current_player_profiles",
        "api.current_match_rounds",
        "api.eligibility_conflicts",
        "validate_eligibility_review_target",
        "validate_eligibility_source_artifact",
        "source_artifact_id",
        "source_uri",
        "source_content_sha256",
        "observed_at",
        "confidence",
        "initial_review_state",
        "expires_at",
    ):
        assert marker in sql

    assert "CREATE TABLE IF NOT EXISTS players" not in sql
    assert "CREATE TABLE IF NOT EXISTS player_aliases" not in sql
    assert "ALTER TABLE ops.match_metadata_observations" not in sql
    assert "CREATE OR REPLACE VIEW api.current_match_metadata" not in sql
    assert "CREATE OR REPLACE VIEW api.candidate_eligibility_match_metadata" in sql
    assert "REFERENCES ops.player_entities" in sql
    assert "num_nonnulls(height_cm, hand, country, birthdate, ta_slug, atp_url) = 1" in sql
    assert "field_value text" not in sql
    assert "HAVING count(DISTINCT canonical_player_id) = 1" in sql
    assert "HAVING count(DISTINCT upper(round_code)) = 1" in sql
    assert "e.confidence," in sql
    assert "minimum_confidence" not in sql
    assert "eligibility review target does not exist in generation" in sql
    assert "eligibility source artifact checksum mismatch" in sql
    assert "hand IN ('L', 'R', 'A')" in sql
    assert "'R16', 'RR', 'QF', 'SF', 'F', 'ER', 'BR'" in sql
    assert "initial_review_state IN ('unreviewed', 'quarantined')" in sql
    assert "CREATE CONSTRAINT TRIGGER eligibility_review_target_guard" in sql
    assert "DEFERRABLE INITIALLY DEFERRED" in sql


def test_eligibility_generation_cutover_is_append_only_and_highest_accepted():
    sql = MIGRATIONS[2].read_text(encoding="utf-8")

    assert "eligibility_generations_one_accepted_idx" not in sql
    assert "eligibility_generation_status_events" in sql
    assert "SELECT max(generation_sequence) FROM accepted" in sql
    assert "ops.eligibility_generation_status_events'::regclass" in sql
    assert "ops.eligibility_generations'::regclass" in sql
    assert "ops.reject_immutable_fact_mutation()" in sql
    assert "validate_eligibility_generation_status_event" in sql
    assert "first eligibility generation status must be candidate" in sql
    assert "NEW.status NOT IN ('accepted', 'rejected')" in sql
    assert "previous_status = 'accepted' AND NEW.status <> 'retired'" in sql
    assert "terminal eligibility status cannot transition" in sql
    assert "eligibility status effective_at must advance monotonically" in sql
    assert "existing_record_sha256 = NEW.record_sha256" in sql
    assert "reviewed_by text NOT NULL" in sql
    assert "reason text NOT NULL" in sql
    assert "expected_projection_seal_sha256 text NOT NULL" in sql
    assert "expected_projection_row_count bigint NOT NULL" in sql
    assert "ops.compute_eligibility_projection_seal" in sql
    assert "ops.assert_eligibility_candidate_ready" in sql
    assert "observations without explicit terminal review" in sql
    assert "accepted projection conflicts" in sql
    assert "projection is sealed at status" in sql
    assert "eligibility sealing requires READ COMMITTED" in sql
    assert "REVOKE ALL ON FUNCTION ops.compute_eligibility_projection_seal(text)" in sql


def test_eligibility_sql_requires_canonical_semantic_storage():
    sql = MIGRATIONS[2].read_text(encoding="utf-8")

    assert "CHECK (round_code IN" in sql
    assert "upper(round_code) IN" not in sql[
        sql.index("CREATE TABLE IF NOT EXISTS ops.eligibility_match_round_observations"):
        sql.index("CREATE INDEX IF NOT EXISTS eligibility_match_round_lookup_idx")
    ]
    for marker in (
        "source_name !~ '^[[:space:]]'",
        "source_player_key !~ '^[[:space:]]'",
        "source_player_key !~ '[[:space:]]$'",
        "match_uid !~ '^[[:space:]]'",
        "match_anchor_key !~ '[[:space:]]$'",
        "source_uri !~ '[[:space:]]$'",
        "atp_url !~ '[[:space:]]$'",
    ):
        assert marker in sql
    assert "source_uri ~ '^[a-z][a-z0-9+.-]*://'" in sql


def test_candidate_rounds_are_isolated_and_never_enter_legacy_metadata_table():
    sql = MIGRATIONS[2].read_text(encoding="utf-8")
    start = sql.index("CREATE OR REPLACE VIEW api.current_match_rounds AS")
    end = sql.index("CREATE OR REPLACE VIEW api.eligibility_conflicts AS", start)
    view = sql[start:end]

    assert "FROM ops.eligibility_match_round_observations o" in view
    assert "ops.match_metadata_observations" not in view
    assert "JOIN api.current_eligibility_generation g" in view
    assert "g.generation_sha256 = o.eligibility_generation_sha256" in view
    assert "o.round_code IS NOT NULL" in view
    assert "o.expires_at > now()" in view
    assert "HAVING count(DISTINCT upper(round_code)) = 1" in view
