-- Operational eligibility provenance contract 1.2.0
--
-- This is an additive, staging-only contract.  It does not replace the
-- existing public.players/player_aliases compatibility projection and it must
-- not be applied to the live Supabase project before the documented cutover
-- gates pass.

BEGIN;

DO $require_v1_1$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ops.schema_versions WHERE version = '1.1.0'
    ) THEN
        RAISE EXCEPTION
            'operational 1.2.0 requires operational contract 1.1.0 first';
    END IF;
END
$require_v1_1$;

-- This finalized file may be replayed only after this exact contract.  The
-- rejected pre-seal draft also used version 1.2.0; fail loudly instead of
-- pretending CREATE TABLE IF NOT EXISTS upgraded that incompatible history.
DO $reject_incompatible_v1_2_history$
BEGIN
    IF EXISTS (
        SELECT 1 FROM ops.schema_versions WHERE version = '1.2.0'
    ) AND (
        to_regclass('ops.eligibility_match_round_observations') IS NULL
        OR EXISTS (
            SELECT 1
              FROM information_schema.columns
             WHERE table_schema = 'ops'
               AND table_name = 'match_metadata_observations'
               AND column_name = 'eligibility_generation_sha256'
        )
        OR NOT EXISTS (
            SELECT 1
              FROM information_schema.columns
             WHERE table_schema = 'ops'
               AND table_name = 'eligibility_generations'
               AND column_name = 'expected_projection_seal_sha256'
               AND is_nullable = 'NO'
        )
        OR NOT EXISTS (
            SELECT 1
              FROM information_schema.columns
             WHERE table_schema = 'ops'
               AND table_name = 'eligibility_match_round_observations'
               AND column_name = 'source_fetch_id'
               AND is_nullable = 'NO'
        )
    ) THEN
        RAISE EXCEPTION
            'incompatible rejected eligibility 1.2 history detected; recreate disposable staging or apply a reviewed forward migration';
    END IF;
END
$reject_incompatible_v1_2_history$;

CREATE TABLE IF NOT EXISTS ops.eligibility_generations (
    eligibility_generation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    generation_sequence bigint NOT NULL UNIQUE,
    generation_sha256 text NOT NULL UNIQUE,
    contract_version text NOT NULL,
    effective_at timestamptz NOT NULL,
    source_manifest jsonb NOT NULL,
    expected_projection_seal_sha256 text NOT NULL,
    expected_projection_row_count bigint NOT NULL,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (generation_sequence > 0),
    CHECK (generation_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (contract_version = '1.0.0'),
    CHECK (jsonb_typeof(source_manifest) = 'object'),
    CHECK (expected_projection_seal_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (expected_projection_row_count > 0),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$'),
    UNIQUE (generation_sha256, generation_sequence)
);

CREATE TABLE IF NOT EXISTS ops.eligibility_generation_status_events (
    eligibility_generation_status_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    eligibility_generation_sha256 text NOT NULL
        REFERENCES ops.eligibility_generations(generation_sha256)
        DEFERRABLE INITIALLY DEFERRED,
    generation_sequence bigint NOT NULL,
    status text NOT NULL,
    effective_at timestamptz NOT NULL,
    reviewed_by text NOT NULL,
    reason text NOT NULL,
    projection_seal_sha256 text,
    projection_row_count bigint,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (eligibility_generation_sha256, status, effective_at),
    FOREIGN KEY (eligibility_generation_sha256, generation_sequence)
        REFERENCES ops.eligibility_generations(generation_sha256, generation_sequence)
        DEFERRABLE INITIALLY DEFERRED,
    CHECK (generation_sequence > 0),
    CHECK (status IN ('candidate', 'accepted', 'rejected', 'retired')),
    CHECK (length(trim(reviewed_by)) > 0),
    CHECK (length(trim(reason)) > 0),
    CHECK (
        (status IN ('candidate', 'accepted')
         AND projection_seal_sha256 IS NOT NULL
         AND projection_seal_sha256 ~ '^[0-9a-f]{64}$'
         AND projection_row_count IS NOT NULL
         AND projection_row_count > 0)
        OR (status NOT IN ('candidate', 'accepted')
            AND projection_seal_sha256 IS NULL
            AND projection_row_count IS NULL)
    ),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

-- Final normalized identity target.  During migration each row is an exact,
-- generation-bound import of public.players; it is not a second mutable live
-- registry.  Observations below cannot reference a bare legacy integer.
CREATE TABLE IF NOT EXISTS ops.player_entities (
    player_entity_id uuid PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    eligibility_generation_sha256 text NOT NULL
        REFERENCES ops.eligibility_generations(generation_sha256)
        DEFERRABLE INITIALLY DEFERRED,
    legacy_player_id bigint NOT NULL,
    canonical_name text NOT NULL,
    canonical_name_norm text NOT NULL,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (eligibility_generation_sha256, legacy_player_id),
    CHECK (legacy_player_id > 0),
    CHECK (length(canonical_name) > 0
           AND canonical_name !~ '^[[:space:]]'
           AND canonical_name !~ '[[:space:]]$'),
    CHECK (canonical_name_norm ~ '^[a-z0-9]+$'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.player_identity_observations (
    player_identity_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    eligibility_generation_sha256 text NOT NULL
        REFERENCES ops.eligibility_generations(generation_sha256)
        DEFERRABLE INITIALLY DEFERRED,
    canonical_player_id bigint NOT NULL,
    observed_name text NOT NULL,
    observed_name_norm text NOT NULL,
    source_name text NOT NULL,
    source_player_key text NOT NULL,
    source_uri text NOT NULL,
    source_content_sha256 text NOT NULL,
    source_artifact_id uuid REFERENCES raw.source_artifacts(source_artifact_id)
        DEFERRABLE INITIALLY DEFERRED,
    observed_at timestamptz NOT NULL,
    confidence numeric(6,5) NOT NULL,
    initial_review_state text NOT NULL DEFAULT 'unreviewed',
    expires_at timestamptz,
    compatibility_import boolean NOT NULL DEFAULT false,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (canonical_player_id > 0),
    CHECK (length(observed_name) > 0
           AND observed_name !~ '^[[:space:]]'
           AND observed_name !~ '[[:space:]]$'),
    CHECK (observed_name_norm ~ '^[a-z0-9]+$'),
    CHECK (length(source_name) > 0
           AND source_name !~ '^[[:space:]]'
           AND source_name !~ '[[:space:]]$'),
    CHECK (length(source_player_key) > 0
           AND source_player_key !~ '^[[:space:]]'
           AND source_player_key !~ '[[:space:]]$'),
    CHECK (source_uri ~ '^[a-z][a-z0-9+.-]*://'
           AND source_uri !~ '[[:space:]]$'),
    CHECK (source_content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (confidence BETWEEN 0 AND 1),
    CHECK (initial_review_state IN ('unreviewed', 'quarantined')),
    CHECK (
        (compatibility_import
         AND source_name = 'public_compatibility_projection'
         AND source_uri ~ '^compatibility://public\.players-player_aliases/'
         AND source_artifact_id IS NULL)
        OR (NOT compatibility_import AND source_artifact_id IS NOT NULL)
    ),
    CHECK (expires_at IS NULL OR expires_at > observed_at),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

DO $player_identity_entity_fkey$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'player_identity_entity_fkey'
           AND conrelid = 'ops.player_identity_observations'::regclass
    ) THEN
        ALTER TABLE ops.player_identity_observations
            ADD CONSTRAINT player_identity_entity_fkey
            FOREIGN KEY (eligibility_generation_sha256, canonical_player_id)
            REFERENCES ops.player_entities(eligibility_generation_sha256, legacy_player_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$player_identity_entity_fkey$;

CREATE TABLE IF NOT EXISTS ops.player_alias_observations (
    player_alias_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    eligibility_generation_sha256 text NOT NULL
        REFERENCES ops.eligibility_generations(generation_sha256)
        DEFERRABLE INITIALLY DEFERRED,
    canonical_player_id bigint NOT NULL,
    alias text NOT NULL,
    alias_norm text NOT NULL,
    source_name text NOT NULL,
    source_uri text NOT NULL,
    source_content_sha256 text NOT NULL,
    source_artifact_id uuid REFERENCES raw.source_artifacts(source_artifact_id)
        DEFERRABLE INITIALLY DEFERRED,
    observed_at timestamptz NOT NULL,
    confidence numeric(6,5) NOT NULL,
    initial_review_state text NOT NULL DEFAULT 'unreviewed',
    expires_at timestamptz,
    compatibility_import boolean NOT NULL DEFAULT false,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (canonical_player_id > 0),
    CHECK (length(alias) > 0
           AND alias !~ '^[[:space:]]'
           AND alias !~ '[[:space:]]$'),
    CHECK (alias_norm ~ '^[a-z0-9]+$'),
    CHECK (length(source_name) > 0
           AND source_name !~ '^[[:space:]]'
           AND source_name !~ '[[:space:]]$'),
    CHECK (source_uri ~ '^[a-z][a-z0-9+.-]*://'
           AND source_uri !~ '[[:space:]]$'),
    CHECK (source_content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (confidence BETWEEN 0 AND 1),
    CHECK (initial_review_state IN ('unreviewed', 'quarantined')),
    CHECK (
        (compatibility_import
         AND source_name = 'public_compatibility_projection'
         AND source_uri ~ '^compatibility://public\.players-player_aliases/'
         AND source_artifact_id IS NULL)
        OR (NOT compatibility_import AND source_artifact_id IS NOT NULL)
    ),
    CHECK (expires_at IS NULL OR expires_at > observed_at),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

DO $player_alias_entity_fkey$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'player_alias_entity_fkey'
           AND conrelid = 'ops.player_alias_observations'::regclass
    ) THEN
        ALTER TABLE ops.player_alias_observations
            ADD CONSTRAINT player_alias_entity_fkey
            FOREIGN KEY (eligibility_generation_sha256, canonical_player_id)
            REFERENCES ops.player_entities(eligibility_generation_sha256, legacy_player_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$player_alias_entity_fkey$;

CREATE TABLE IF NOT EXISTS ops.player_profile_observations (
    player_profile_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    eligibility_generation_sha256 text NOT NULL
        REFERENCES ops.eligibility_generations(generation_sha256)
        DEFERRABLE INITIALLY DEFERRED,
    canonical_player_id bigint NOT NULL,
    field_name text NOT NULL,
    height_cm numeric(6,2),
    hand text,
    country text,
    birthdate date,
    ta_slug text,
    atp_url text,
    source_name text NOT NULL,
    source_uri text NOT NULL,
    source_content_sha256 text NOT NULL,
    source_artifact_id uuid REFERENCES raw.source_artifacts(source_artifact_id)
        DEFERRABLE INITIALLY DEFERRED,
    observed_at timestamptz NOT NULL,
    confidence numeric(6,5) NOT NULL,
    initial_review_state text NOT NULL DEFAULT 'unreviewed',
    expires_at timestamptz,
    compatibility_import boolean NOT NULL DEFAULT false,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (canonical_player_id > 0),
    CHECK (field_name IN ('height_cm', 'hand', 'country', 'birthdate', 'ta_slug', 'atp_url')),
    CHECK (num_nonnulls(height_cm, hand, country, birthdate, ta_slug, atp_url) = 1),
    CHECK (
        (field_name = 'height_cm' AND height_cm IS NOT NULL AND height_cm BETWEEN 150 AND 230)
        OR (field_name = 'hand' AND hand IN ('L', 'R', 'A'))
        OR (field_name = 'country' AND country IS NOT NULL
            AND length(country) > 0
            AND country !~ '^[[:space:]]' AND country !~ '[[:space:]]$')
        OR (field_name = 'birthdate' AND birthdate IS NOT NULL)
        OR (field_name = 'ta_slug' AND ta_slug IS NOT NULL
            AND length(ta_slug) > 0
            AND ta_slug !~ '^[[:space:]]' AND ta_slug !~ '[[:space:]]$')
        OR (field_name = 'atp_url' AND atp_url IS NOT NULL
            AND atp_url ~ '^[a-z][a-z0-9+.-]*://'
            AND atp_url !~ '[[:space:]]$')
    ),
    CHECK (length(source_name) > 0
           AND source_name !~ '^[[:space:]]'
           AND source_name !~ '[[:space:]]$'),
    CHECK (source_uri ~ '^[a-z][a-z0-9+.-]*://'
           AND source_uri !~ '[[:space:]]$'),
    CHECK (source_content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (confidence BETWEEN 0 AND 1),
    CHECK (initial_review_state IN ('unreviewed', 'quarantined')),
    CHECK (
        (compatibility_import
         AND source_name = 'public_compatibility_projection'
         AND source_uri ~ '^compatibility://public\.players-player_aliases/'
         AND source_artifact_id IS NULL)
        OR (NOT compatibility_import AND source_artifact_id IS NOT NULL)
    ),
    CHECK (expires_at IS NULL OR expires_at > observed_at),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

DO $player_profile_entity_fkey$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'player_profile_entity_fkey'
           AND conrelid = 'ops.player_profile_observations'::regclass
    ) THEN
        ALTER TABLE ops.player_profile_observations
            ADD CONSTRAINT player_profile_entity_fkey
            FOREIGN KEY (eligibility_generation_sha256, canonical_player_id)
            REFERENCES ops.player_entities(eligibility_generation_sha256, legacy_player_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$player_profile_entity_fkey$;

CREATE TABLE IF NOT EXISTS ops.eligibility_review_events (
    eligibility_review_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    eligibility_generation_sha256 text NOT NULL
        REFERENCES ops.eligibility_generations(generation_sha256)
        DEFERRABLE INITIALLY DEFERRED,
    target_table text NOT NULL,
    target_idempotency_key text NOT NULL,
    review_state text NOT NULL,
    reviewed_at timestamptz NOT NULL,
    reviewed_by text NOT NULL,
    reason text NOT NULL,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (
        eligibility_generation_sha256, target_table,
        target_idempotency_key, reviewed_at
    ),
    CHECK (target_table IN (
        'ops.player_identity_observations',
        'ops.player_alias_observations',
        'ops.player_profile_observations',
        'ops.eligibility_match_round_observations'
    )),
    CHECK (length(target_idempotency_key) > 0
           AND target_idempotency_key !~ '^[[:space:]]'
           AND target_idempotency_key !~ '[[:space:]]$'),
    CHECK (review_state IN ('accepted', 'rejected', 'quarantined', 'superseded')),
    CHECK (length(reviewed_by) > 0
           AND reviewed_by !~ '^[[:space:]]'
           AND reviewed_by !~ '[[:space:]]$'),
    CHECK (length(reason) > 0
           AND reason !~ '^[[:space:]]'
           AND reason !~ '[[:space:]]$'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

-- Candidate round evidence is intentionally isolated from the established
-- ops.match_metadata_observations -> api.current_match_metadata path.  Merely
-- adding provenance columns to that legacy table would let candidate evidence
-- change the live projection before an explicit cutover migration.
CREATE TABLE IF NOT EXISTS ops.eligibility_match_round_observations (
    eligibility_match_round_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    eligibility_generation_sha256 text NOT NULL
        REFERENCES ops.eligibility_generations(generation_sha256)
        DEFERRABLE INITIALLY DEFERRED,
    run_id uuid NOT NULL REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_fetch_id uuid NOT NULL REFERENCES raw.source_fetches(source_fetch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_artifact_id uuid NOT NULL REFERENCES raw.source_artifacts(source_artifact_id)
        DEFERRABLE INITIALLY DEFERRED,
    match_uid text NOT NULL,
    match_anchor_key text NOT NULL,
    observed_at timestamptz NOT NULL,
    source_name text NOT NULL,
    round_code text NOT NULL,
    round_provenance text NOT NULL,
    source_uri text NOT NULL,
    source_content_sha256 text NOT NULL,
    confidence numeric(6,5) NOT NULL,
    initial_review_state text NOT NULL DEFAULT 'unreviewed',
    expires_at timestamptz NOT NULL,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (length(match_uid) > 0
           AND match_uid !~ '^[[:space:]]'
           AND match_uid !~ '[[:space:]]$'),
    CHECK (length(match_anchor_key) > 0
           AND match_anchor_key !~ '^[[:space:]]'
           AND match_anchor_key !~ '[[:space:]]$'),
    CHECK (length(source_name) > 0
           AND source_name !~ '^[[:space:]]'
           AND source_name !~ '[[:space:]]$'),
    CHECK (round_code IN (
        'Q1', 'Q2', 'Q3', 'Q4', 'R128', 'R64', 'R32',
        'R16', 'RR', 'QF', 'SF', 'F', 'ER', 'BR'
    )),
    CHECK (round_provenance IN (
        'default', 'inferred', 'bovada', 'tournament_registry',
        'canonical_store', 'official'
    )),
    CHECK (source_uri ~ '^[a-z][a-z0-9+.-]*://'
           AND source_uri !~ '[[:space:]]$'),
    CHECK (source_content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (confidence BETWEEN 0 AND 1),
    CHECK (initial_review_state IN ('unreviewed', 'quarantined')),
    CHECK (expires_at > observed_at),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS player_identity_observations_lookup_idx
    ON ops.player_identity_observations
       (eligibility_generation_sha256, observed_name_norm, observed_at DESC);
CREATE INDEX IF NOT EXISTS player_alias_observations_lookup_idx
    ON ops.player_alias_observations
       (eligibility_generation_sha256, alias_norm, observed_at DESC);
CREATE INDEX IF NOT EXISTS player_profile_observations_lookup_idx
    ON ops.player_profile_observations
       (eligibility_generation_sha256, canonical_player_id, field_name, observed_at DESC);
CREATE INDEX IF NOT EXISTS eligibility_review_events_target_idx
    ON ops.eligibility_review_events
       (eligibility_generation_sha256, target_table, target_idempotency_key, reviewed_at DESC);
CREATE INDEX IF NOT EXISTS eligibility_match_round_lookup_idx
    ON ops.eligibility_match_round_observations
       (eligibility_generation_sha256, match_anchor_key, observed_at DESC)
    ;
CREATE INDEX IF NOT EXISTS eligibility_generation_status_current_idx
    ON ops.eligibility_generation_status_events
       (eligibility_generation_sha256, effective_at DESC, created_at DESC);

CREATE OR REPLACE FUNCTION ops.compute_eligibility_projection_seal(
    requested_generation_sha256 text
)
RETURNS TABLE (projection_seal_sha256 text, projection_row_count bigint)
LANGUAGE sql
STABLE
AS $$
    WITH projection_rows AS (
        SELECT 'ops.player_entities'::text AS table_name,
               idempotency_key, record_sha256
          FROM ops.player_entities
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.player_identity_observations', idempotency_key, record_sha256
          FROM ops.player_identity_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.player_alias_observations', idempotency_key, record_sha256
          FROM ops.player_alias_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.player_profile_observations', idempotency_key, record_sha256
          FROM ops.player_profile_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.eligibility_match_round_observations', idempotency_key, record_sha256
          FROM ops.eligibility_match_round_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.eligibility_review_events', idempotency_key, record_sha256
          FROM ops.eligibility_review_events
         WHERE eligibility_generation_sha256 = requested_generation_sha256
    ), packed AS (
        SELECT table_name, idempotency_key, record_sha256,
               octet_length(table_name)::text || ':' || table_name
               || octet_length(idempotency_key)::text || ':' || idempotency_key
               || octet_length(record_sha256)::text || ':' || record_sha256
                   AS packed_row
          FROM projection_rows
    )
    SELECT encode(
               sha256(convert_to(
                   'eligibility-projection-seal-v1' || chr(10)
                   || count(*)::text || chr(10)
                   || coalesce(string_agg(
                       packed_row, '' ORDER BY
                           table_name COLLATE "C",
                           idempotency_key COLLATE "C",
                           record_sha256 COLLATE "C"
                   ), ''),
                   'UTF8'
               )),
               'hex'
           ),
           count(*)::bigint
      FROM packed
$$;

-- Imported facts may participate in multiple deterministic batches.  Extend
-- the existing allow-list rather than bypassing membership parity.
ALTER TABLE ops.import_batch_memberships
    DROP CONSTRAINT IF EXISTS import_batch_memberships_target_table_check;
ALTER TABLE ops.import_batch_memberships
    ADD CONSTRAINT import_batch_memberships_target_table_check CHECK (
        target_table IN (
            'raw.source_fetches', 'raw.source_artifacts',
            'ops.pipeline_runs', 'ops.pipeline_run_stages',
            'ops.odds_observations', 'ops.match_metadata_observations',
            'ops.eligibility_match_round_observations',
            'ops.paper_accounts', 'ops.paper_sessions', 'ops.account_ledger',
            'ops.account_journal_entries', 'ops.bet_recommendations',
            'ops.bet_state_events', 'ops.settlement_attempts',
            'ops.settlement_events', 'ops.skip_events', 'ops.import_conflicts',
            'ops.eligibility_generations',
            'ops.eligibility_generation_status_events',
            'ops.player_entities',
            'ops.player_identity_observations',
            'ops.player_alias_observations',
            'ops.player_profile_observations',
            'ops.eligibility_review_events',
            'ml.feature_schemas', 'ml.feature_snapshots', 'ml.model_releases',
            'ml.model_registry_generations',
            'ml.model_release_status_events',
            'ml.prediction_observations'
        )
    );

CREATE OR REPLACE VIEW api.eligibility_review_state AS
SELECT DISTINCT ON (
    eligibility_generation_sha256, target_table, target_idempotency_key
)
    eligibility_generation_sha256,
    target_table,
    target_idempotency_key,
    review_state,
    reviewed_at,
    reviewed_by,
    reason
FROM ops.eligibility_review_events
WHERE reviewed_at <= now()
ORDER BY eligibility_generation_sha256, target_table, target_idempotency_key,
         reviewed_at DESC, created_at DESC, idempotency_key DESC;

CREATE OR REPLACE VIEW api.current_eligibility_generation AS
WITH latest_status AS (
    SELECT DISTINCT ON (eligibility_generation_sha256)
        eligibility_generation_sha256,
        generation_sequence,
        status,
        effective_at AS status_effective_at,
        projection_seal_sha256,
        projection_row_count
    FROM ops.eligibility_generation_status_events
    WHERE effective_at <= now()
    ORDER BY eligibility_generation_sha256, effective_at DESC, created_at DESC
), accepted AS (
    SELECT g.*, s.status_effective_at,
           s.projection_seal_sha256,
           s.projection_row_count
      FROM ops.eligibility_generations g
      JOIN latest_status s
        ON s.eligibility_generation_sha256 = g.generation_sha256
       AND s.generation_sequence = g.generation_sequence
     WHERE s.status = 'accepted'
       AND s.projection_seal_sha256 = g.expected_projection_seal_sha256
       AND s.projection_row_count = g.expected_projection_row_count
       AND EXISTS (
           SELECT 1
             FROM ops.eligibility_generation_status_events candidate
            WHERE candidate.eligibility_generation_sha256 = g.generation_sha256
              AND candidate.status = 'candidate'
              AND candidate.projection_seal_sha256 = s.projection_seal_sha256
              AND candidate.projection_row_count = s.projection_row_count
       )
)
SELECT
    generation_sequence,
    generation_sha256,
    contract_version,
    effective_at,
    source_manifest,
    status_effective_at,
    projection_seal_sha256,
    projection_row_count
FROM accepted
WHERE generation_sequence = (SELECT max(generation_sequence) FROM accepted);

CREATE OR REPLACE VIEW api.current_player_identities AS
WITH reviewed AS (
    SELECT o.*
      FROM ops.player_identity_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.player_identity_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE (o.expires_at IS NULL OR o.expires_at > now())
       AND r.review_state = 'accepted'
), source_conflicts AS (
    SELECT eligibility_generation_sha256, source_name, source_player_key
      FROM reviewed
     GROUP BY eligibility_generation_sha256, source_name, source_player_key
    HAVING count(DISTINCT canonical_player_id) > 1
), eligible AS (
    SELECT reviewed.*
      FROM reviewed
      LEFT JOIN source_conflicts USING (
          eligibility_generation_sha256, source_name, source_player_key
      )
     WHERE source_conflicts.source_player_key IS NULL
)
SELECT
    eligibility_generation_sha256,
    observed_name_norm,
    min(canonical_player_id) AS canonical_player_id,
    max(observed_at) AS last_observed_at,
    count(*) AS observation_count,
    CASE WHEN bool_or(expires_at IS NULL) THEN NULL
         ELSE max(expires_at) END AS valid_until
FROM eligible
GROUP BY eligibility_generation_sha256, observed_name_norm
HAVING count(DISTINCT canonical_player_id) = 1;

CREATE OR REPLACE VIEW api.current_player_aliases AS
WITH eligible AS (
    SELECT o.*
      FROM ops.player_alias_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.player_alias_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE (o.expires_at IS NULL OR o.expires_at > now())
       AND r.review_state = 'accepted'
)
SELECT
    eligibility_generation_sha256,
    alias_norm,
    min(canonical_player_id) AS canonical_player_id,
    max(observed_at) AS last_observed_at,
    count(*) AS observation_count,
    CASE WHEN bool_or(expires_at IS NULL) THEN NULL
         ELSE max(expires_at) END AS valid_until
FROM eligible
GROUP BY eligibility_generation_sha256, alias_norm
HAVING count(DISTINCT canonical_player_id) = 1;

CREATE OR REPLACE VIEW api.current_player_profiles AS
WITH eligible AS (
    SELECT o.*,
           coalesce(
               o.height_cm::text, o.hand, o.country, o.birthdate::text,
               o.ta_slug, o.atp_url
           ) AS field_value
      FROM ops.player_profile_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.player_profile_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE (o.expires_at IS NULL OR o.expires_at > now())
       AND r.review_state = 'accepted'
), resolved AS (
    SELECT
        eligibility_generation_sha256,
        canonical_player_id,
        field_name,
        min(field_value) AS field_value,
        max(observed_at) AS last_observed_at,
        count(*) AS observation_count,
        CASE WHEN bool_or(expires_at IS NULL) THEN NULL
             ELSE max(expires_at) END AS valid_until
    FROM eligible
    GROUP BY eligibility_generation_sha256, canonical_player_id, field_name
    HAVING count(DISTINCT field_value) = 1
)
SELECT * FROM resolved;

CREATE OR REPLACE VIEW api.current_player_name_bindings AS
WITH bindings AS (
    SELECT eligibility_generation_sha256,
           observed_name_norm AS player_name_norm,
           canonical_player_id, valid_until
      FROM api.current_player_identities
    UNION ALL
    SELECT eligibility_generation_sha256,
           alias_norm AS player_name_norm,
           canonical_player_id, valid_until
      FROM api.current_player_aliases
)
SELECT eligibility_generation_sha256,
       player_name_norm,
       min(canonical_player_id) AS canonical_player_id,
       CASE WHEN bool_or(valid_until IS NULL) THEN NULL
            ELSE max(valid_until) END AS valid_until,
       count(*) AS binding_count
  FROM bindings
 GROUP BY eligibility_generation_sha256, player_name_norm
HAVING count(DISTINCT canonical_player_id) = 1;

CREATE OR REPLACE VIEW api.current_match_rounds AS
WITH eligible AS (
    SELECT o.*
      FROM ops.eligibility_match_round_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.eligibility_match_round_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE o.round_code IS NOT NULL
       AND o.expires_at > now()
       AND r.review_state = 'accepted'
), resolved_keys AS (
    SELECT
        eligibility_generation_sha256,
        match_anchor_key,
        count(*) AS observation_count
    FROM eligible
    GROUP BY eligibility_generation_sha256, match_anchor_key
    HAVING count(DISTINCT upper(round_code)) = 1
)
SELECT DISTINCT ON (e.eligibility_generation_sha256, e.match_anchor_key)
    e.eligibility_generation_sha256,
    e.match_anchor_key,
    upper(e.round_code) AS round_code,
    e.observed_at AS last_observed_at,
    e.expires_at,
    e.source_uri,
    e.source_content_sha256,
    e.confidence,
    k.observation_count
FROM eligible e
JOIN resolved_keys k USING (eligibility_generation_sha256, match_anchor_key)
ORDER BY e.eligibility_generation_sha256, e.match_anchor_key,
         e.observed_at DESC, e.created_at DESC, e.idempotency_key DESC;

CREATE OR REPLACE VIEW api.eligibility_conflicts AS
WITH accepted_identity AS (
    SELECT o.*
      FROM ops.player_identity_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.player_identity_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE r.review_state = 'accepted'
       AND (o.expires_at IS NULL OR o.expires_at > now())
), accepted_alias AS (
    SELECT o.*
      FROM ops.player_alias_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.player_alias_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE r.review_state = 'accepted'
       AND (o.expires_at IS NULL OR o.expires_at > now())
), source_key_conflicts AS (
    SELECT eligibility_generation_sha256, source_name, source_player_key,
           count(DISTINCT canonical_player_id::text) AS distinct_values
      FROM accepted_identity
     GROUP BY eligibility_generation_sha256, source_name, source_player_key
    HAVING count(DISTINCT canonical_player_id::text) > 1
), source_key_audit AS (
    SELECT 'player_source_key'::text AS evidence_kind,
           eligibility_generation_sha256,
           source_name || ':' || source_player_key AS subject_key,
           distinct_values
      FROM source_key_conflicts
), source_identity_names AS (
    SELECT 'player_source_identity'::text AS evidence_kind,
           i.eligibility_generation_sha256,
           i.observed_name_norm AS subject_key,
           max(c.distinct_values) AS distinct_values
      FROM accepted_identity i
      JOIN source_key_conflicts c USING (
          eligibility_generation_sha256, source_name, source_player_key
      )
     GROUP BY i.eligibility_generation_sha256, i.observed_name_norm
), identity_conflicts AS (
    SELECT
        'player_identity'::text AS evidence_kind,
        o.eligibility_generation_sha256,
        o.observed_name_norm AS subject_key,
        count(DISTINCT o.canonical_player_id::text) AS distinct_values
      FROM accepted_identity o
     GROUP BY o.eligibility_generation_sha256, o.observed_name_norm
    HAVING count(DISTINCT o.canonical_player_id::text) > 1
), alias_conflicts AS (
    SELECT
        'player_alias'::text,
        o.eligibility_generation_sha256,
        o.alias_norm,
        count(DISTINCT o.canonical_player_id::text)
      FROM accepted_alias o
     GROUP BY o.eligibility_generation_sha256, o.alias_norm
    HAVING count(DISTINCT o.canonical_player_id::text) > 1
), name_binding_conflicts AS (
    SELECT 'player_name_binding'::text,
           eligibility_generation_sha256,
           player_name_norm,
           count(DISTINCT canonical_player_id::text)
      FROM (
          SELECT eligibility_generation_sha256,
                 observed_name_norm AS player_name_norm,
                 canonical_player_id
            FROM accepted_identity
          UNION ALL
          SELECT eligibility_generation_sha256,
                 alias_norm AS player_name_norm,
                 canonical_player_id
            FROM accepted_alias
      ) bindings
     GROUP BY eligibility_generation_sha256, player_name_norm
    HAVING count(DISTINCT canonical_player_id::text) > 1
), profile_conflicts AS (
    SELECT
        'player_profile'::text,
        o.eligibility_generation_sha256,
        o.canonical_player_id::text || ':' || o.field_name,
        count(DISTINCT coalesce(
            o.height_cm::text, o.hand, o.country, o.birthdate::text,
            o.ta_slug, o.atp_url
        ))
      FROM ops.player_profile_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.player_profile_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE r.review_state = 'accepted'
       AND (o.expires_at IS NULL OR o.expires_at > now())
     GROUP BY o.eligibility_generation_sha256, o.canonical_player_id, o.field_name
    HAVING count(DISTINCT coalesce(
        o.height_cm::text, o.hand, o.country, o.birthdate::text,
        o.ta_slug, o.atp_url
    )) > 1
), round_conflicts AS (
    SELECT
        'match_round'::text,
        o.eligibility_generation_sha256,
        o.match_anchor_key,
        count(DISTINCT upper(o.round_code))
      FROM ops.eligibility_match_round_observations o
      JOIN api.current_eligibility_generation g
        ON g.generation_sha256 = o.eligibility_generation_sha256
      JOIN api.eligibility_review_state r
        ON r.eligibility_generation_sha256 = o.eligibility_generation_sha256
       AND r.target_table = 'ops.eligibility_match_round_observations'
       AND r.target_idempotency_key = o.idempotency_key
     WHERE o.round_code IS NOT NULL
       AND r.review_state = 'accepted'
       AND o.expires_at > now()
     GROUP BY o.eligibility_generation_sha256, o.match_anchor_key
    HAVING count(DISTINCT upper(o.round_code)) > 1
)
SELECT * FROM identity_conflicts
UNION ALL SELECT * FROM alias_conflicts
UNION ALL SELECT * FROM name_binding_conflicts
UNION ALL SELECT * FROM source_key_audit
UNION ALL SELECT * FROM source_identity_names
UNION ALL SELECT * FROM profile_conflicts
UNION ALL SELECT * FROM round_conflicts;

-- Contract 1.2 is candidate-only: it must not change the established
-- api.current_match_metadata semantics. A future explicit cutover migration
-- may promote this separate comparison projection after parity acceptance.
CREATE OR REPLACE VIEW api.candidate_eligibility_match_metadata AS
SELECT
    m.match_anchor_key,
    m.match_uid,
    m.match_date,
    m.match_start_at_utc,
    m.tournament,
    m.event_title,
    m.round_code AS legacy_round_code,
    r.round_code AS candidate_round_code,
    m.surface,
    m.level,
    m.last_observed_at,
    m.observation_count,
    r.eligibility_generation_sha256 AS round_generation_sha256,
    r.source_uri AS round_source_uri,
    r.source_content_sha256 AS round_source_content_sha256,
    r.confidence AS round_confidence,
    r.expires_at AS round_expires_at
FROM api.current_match_metadata m
LEFT JOIN api.current_match_rounds r USING (match_anchor_key);

CREATE OR REPLACE FUNCTION ops.validate_eligibility_source_artifact()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    artifact_sha256 text;
    artifact_source_uri text;
    artifact_source_fetch_id uuid;
    artifact_run_id uuid;
BEGIN
    IF NEW.source_artifact_id IS NULL THEN
        RETURN NEW;
    END IF;
    SELECT artifact.content_sha256,
           artifact.metadata->>'source_uri',
           artifact.source_fetch_id,
           source_fetch.run_id
      INTO artifact_sha256, artifact_source_uri,
           artifact_source_fetch_id, artifact_run_id
      FROM raw.source_artifacts artifact
      JOIN raw.source_fetches source_fetch
        ON source_fetch.source_fetch_id = artifact.source_fetch_id
     WHERE artifact.source_artifact_id = NEW.source_artifact_id;
    IF artifact_sha256 IS NULL THEN
        RAISE EXCEPTION 'eligibility source artifact does not exist: %',
            NEW.source_artifact_id;
    END IF;
    IF artifact_sha256 IS DISTINCT FROM NEW.source_content_sha256 THEN
        RAISE EXCEPTION 'eligibility source artifact checksum mismatch: %',
            NEW.source_artifact_id;
    END IF;
    IF artifact_source_uri IS NULL
       OR artifact_source_uri IS DISTINCT FROM NEW.source_uri THEN
        RAISE EXCEPTION 'eligibility source artifact URI mismatch: %',
            NEW.source_artifact_id;
    END IF;
    IF TG_TABLE_SCHEMA = 'ops'
       AND TG_TABLE_NAME = 'eligibility_match_round_observations'
       AND (artifact_source_fetch_id IS DISTINCT FROM NEW.source_fetch_id
            OR artifact_run_id IS DISTINCT FROM NEW.run_id) THEN
        RAISE EXCEPTION
            'eligibility round artifact fetch/run lineage mismatch: %',
            NEW.source_artifact_id;
    END IF;
    RETURN NEW;
END
$$;

DO $eligibility_source_artifact_triggers$
DECLARE
    relation regclass;
    trigger_name text;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.player_identity_observations'::regclass,
        'ops.player_alias_observations'::regclass,
        'ops.player_profile_observations'::regclass,
        'ops.eligibility_match_round_observations'::regclass
    ]
    LOOP
        trigger_name := 'eligibility_' || relation::oid::text
                        || '_source_artifact_guard';
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %s', trigger_name, relation);
        EXECUTE format(
            'CREATE CONSTRAINT TRIGGER %I AFTER INSERT ON %s '
            'DEFERRABLE INITIALLY DEFERRED '
            'FOR EACH ROW EXECUTE FUNCTION ops.validate_eligibility_source_artifact()',
            trigger_name, relation
        );
    END LOOP;
END
$eligibility_source_artifact_triggers$;

CREATE OR REPLACE FUNCTION ops.validate_eligibility_review_target()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    target_observed_at timestamptz;
BEGIN
    CASE NEW.target_table
        WHEN 'ops.player_identity_observations' THEN
            SELECT observed_at INTO target_observed_at
              FROM ops.player_identity_observations
             WHERE idempotency_key = NEW.target_idempotency_key
               AND eligibility_generation_sha256 = NEW.eligibility_generation_sha256;
        WHEN 'ops.player_alias_observations' THEN
            SELECT observed_at INTO target_observed_at
              FROM ops.player_alias_observations
             WHERE idempotency_key = NEW.target_idempotency_key
               AND eligibility_generation_sha256 = NEW.eligibility_generation_sha256;
        WHEN 'ops.player_profile_observations' THEN
            SELECT observed_at INTO target_observed_at
              FROM ops.player_profile_observations
             WHERE idempotency_key = NEW.target_idempotency_key
               AND eligibility_generation_sha256 = NEW.eligibility_generation_sha256;
        WHEN 'ops.eligibility_match_round_observations' THEN
            SELECT observed_at INTO target_observed_at
              FROM ops.eligibility_match_round_observations
             WHERE idempotency_key = NEW.target_idempotency_key
               AND eligibility_generation_sha256 = NEW.eligibility_generation_sha256;
        ELSE
            target_observed_at := NULL;
    END CASE;
    IF target_observed_at IS NULL THEN
        RAISE EXCEPTION 'eligibility review target does not exist in generation: %.%',
            NEW.target_table, NEW.target_idempotency_key;
    END IF;
    IF NEW.reviewed_at < target_observed_at THEN
        RAISE EXCEPTION 'eligibility review predates target observation: %.%',
            NEW.target_table, NEW.target_idempotency_key;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS eligibility_review_target_guard
    ON ops.eligibility_review_events;
CREATE CONSTRAINT TRIGGER eligibility_review_target_guard
AFTER INSERT ON ops.eligibility_review_events
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION ops.validate_eligibility_review_target();

CREATE OR REPLACE FUNCTION ops.assert_eligibility_candidate_ready(
    requested_generation_sha256 text,
    candidate_effective_at timestamptz
)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    unreviewed_count bigint;
    accepted_identity_count bigint;
    conflict_count bigint;
BEGIN
    IF EXISTS (
        SELECT 1
          FROM ops.eligibility_review_events
         WHERE eligibility_generation_sha256 = requested_generation_sha256
           AND reviewed_at > candidate_effective_at
    ) THEN
        RAISE EXCEPTION
            'eligibility candidate contains a review after candidate effective_at';
    END IF;
    -- Every observation must have an explicit review event. Initial source
    -- state is never an acceptance decision, including compatibility imports.
    WITH observation_targets AS (
        SELECT 'ops.player_identity_observations'::text AS target_table,
               idempotency_key
          FROM ops.player_identity_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.player_alias_observations', idempotency_key
          FROM ops.player_alias_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.player_profile_observations', idempotency_key
          FROM ops.player_profile_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
        UNION ALL
        SELECT 'ops.eligibility_match_round_observations', idempotency_key
          FROM ops.eligibility_match_round_observations
         WHERE eligibility_generation_sha256 = requested_generation_sha256
    )
    SELECT count(*)
      INTO unreviewed_count
      FROM observation_targets target
     WHERE NOT EXISTS (
         SELECT 1
           FROM ops.eligibility_review_events review
          WHERE review.eligibility_generation_sha256 = requested_generation_sha256
            AND review.target_table = target.target_table
            AND review.target_idempotency_key = target.idempotency_key
            AND review.reviewed_at <= candidate_effective_at
     );
    IF unreviewed_count > 0 THEN
        RAISE EXCEPTION
            'eligibility candidate has % observations without explicit terminal review',
            unreviewed_count;
    END IF;

    -- Candidate sealing rejects every ambiguity among accepted evidence that
    -- is active at the decision time. Read views must never be responsible for
    -- silently hiding a conflict that the generation gate should have caught.
    WITH latest_review AS (
        SELECT DISTINCT ON (target_table, target_idempotency_key)
               target_table, target_idempotency_key, review_state
         FROM ops.eligibility_review_events
         WHERE eligibility_generation_sha256 = requested_generation_sha256
           AND reviewed_at <= candidate_effective_at
         ORDER BY target_table, target_idempotency_key,
                  reviewed_at DESC, created_at DESC, idempotency_key DESC
    ), accepted_identity AS (
        SELECT o.*
          FROM ops.player_identity_observations o
          JOIN latest_review r
            ON r.target_table = 'ops.player_identity_observations'
           AND r.target_idempotency_key = o.idempotency_key
         WHERE o.eligibility_generation_sha256 = requested_generation_sha256
           AND r.review_state = 'accepted'
           AND (o.expires_at IS NULL OR o.expires_at > candidate_effective_at)
    ), accepted_alias AS (
        SELECT o.*
          FROM ops.player_alias_observations o
          JOIN latest_review r
            ON r.target_table = 'ops.player_alias_observations'
           AND r.target_idempotency_key = o.idempotency_key
         WHERE o.eligibility_generation_sha256 = requested_generation_sha256
           AND r.review_state = 'accepted'
           AND (o.expires_at IS NULL OR o.expires_at > candidate_effective_at)
    ), accepted_profile AS (
        SELECT o.*,
               coalesce(o.height_cm::text, o.hand, o.country,
                        o.birthdate::text, o.ta_slug, o.atp_url) AS field_value
          FROM ops.player_profile_observations o
          JOIN latest_review r
            ON r.target_table = 'ops.player_profile_observations'
           AND r.target_idempotency_key = o.idempotency_key
         WHERE o.eligibility_generation_sha256 = requested_generation_sha256
           AND r.review_state = 'accepted'
           AND (o.expires_at IS NULL OR o.expires_at > candidate_effective_at)
    ), accepted_round AS (
        SELECT o.*
          FROM ops.eligibility_match_round_observations o
          JOIN latest_review r
            ON r.target_table = 'ops.eligibility_match_round_observations'
           AND r.target_idempotency_key = o.idempotency_key
         WHERE o.eligibility_generation_sha256 = requested_generation_sha256
           AND r.review_state = 'accepted'
           AND o.expires_at > candidate_effective_at
    ), conflicts AS (
        SELECT 1
          FROM accepted_identity
         GROUP BY observed_name_norm
        HAVING count(DISTINCT canonical_player_id) > 1
        UNION ALL
        SELECT 1
          FROM accepted_alias
         GROUP BY alias_norm
        HAVING count(DISTINCT canonical_player_id) > 1
        UNION ALL
        SELECT 1
          FROM (
              SELECT observed_name_norm AS name_norm, canonical_player_id
                FROM accepted_identity
              UNION ALL
              SELECT alias_norm, canonical_player_id FROM accepted_alias
          ) binding
         GROUP BY name_norm
        HAVING count(DISTINCT canonical_player_id) > 1
        UNION ALL
        SELECT 1
          FROM accepted_identity
         GROUP BY source_name, source_player_key
        HAVING count(DISTINCT canonical_player_id) > 1
        UNION ALL
        SELECT 1
          FROM accepted_profile
         GROUP BY canonical_player_id, field_name
        HAVING count(DISTINCT field_value) > 1
        UNION ALL
        SELECT 1
          FROM accepted_round
         GROUP BY match_anchor_key
        HAVING count(DISTINCT upper(round_code)) > 1
    )
    SELECT (SELECT count(*) FROM accepted_identity), count(*)
      INTO accepted_identity_count, conflict_count
      FROM conflicts;
    IF accepted_identity_count <= 0 THEN
        RAISE EXCEPTION
            'eligibility generation has no active accepted identity binding';
    END IF;
    IF conflict_count > 0 THEN
        RAISE EXCEPTION
            'eligibility candidate has % accepted projection conflicts',
            conflict_count;
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION ops.guard_eligibility_projection_content()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    current_status text;
    exact_retry boolean := false;
BEGIN
    IF current_setting('transaction_isolation') <> 'read committed' THEN
        RAISE EXCEPTION
            'eligibility sealing requires READ COMMITTED transaction isolation';
    END IF;
    IF NEW.eligibility_generation_sha256 IS NULL THEN
        RETURN NEW;
    END IF;
    PERFORM 1
      FROM ops.eligibility_generations
     WHERE generation_sha256 = NEW.eligibility_generation_sha256
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown eligibility generation: %',
            NEW.eligibility_generation_sha256;
    END IF;
    SELECT status
      INTO current_status
      FROM ops.eligibility_generation_status_events
     WHERE eligibility_generation_sha256 = NEW.eligibility_generation_sha256
     ORDER BY effective_at DESC, created_at DESC, idempotency_key DESC
     LIMIT 1;
    IF current_status IS NULL THEN
        RETURN NEW;
    END IF;
    EXECUTE format(
        'SELECT EXISTS (SELECT 1 FROM %I.%I '
        'WHERE idempotency_key = $1 AND eligibility_generation_sha256 = $2 '
        'AND record_sha256 = $3)',
        TG_TABLE_SCHEMA, TG_TABLE_NAME
    ) INTO exact_retry USING
        NEW.idempotency_key, NEW.eligibility_generation_sha256,
        NEW.record_sha256;
    IF exact_retry THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION
        'eligibility projection is sealed at status %; new content rejected on %.%',
        current_status, TG_TABLE_SCHEMA, TG_TABLE_NAME;
END
$$;

DO $eligibility_projection_content_triggers$
DECLARE
    relation regclass;
    trigger_name text;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.player_entities'::regclass,
        'ops.player_identity_observations'::regclass,
        'ops.player_alias_observations'::regclass,
        'ops.player_profile_observations'::regclass,
        'ops.eligibility_match_round_observations'::regclass,
        'ops.eligibility_review_events'::regclass
    ]
    LOOP
        trigger_name := 'eligibility_' || relation::oid::text || '_seal_guard';
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %s', trigger_name, relation);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE INSERT ON %s '
            'FOR EACH ROW EXECUTE FUNCTION ops.guard_eligibility_projection_content()',
            trigger_name, relation
        );
    END LOOP;
END
$eligibility_projection_content_triggers$;

CREATE OR REPLACE FUNCTION ops.validate_eligibility_generation_status_event()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    previous_status text;
    previous_effective_at timestamptz;
    previous_projection_seal_sha256 text;
    previous_projection_row_count bigint;
    existing_record_sha256 text;
    generation_effective_at timestamptz;
    expected_projection_seal_sha256 text;
    expected_projection_row_count bigint;
    actual_projection_seal_sha256 text;
    actual_projection_row_count bigint;
BEGIN
    IF current_setting('transaction_isolation') <> 'read committed' THEN
        RAISE EXCEPTION
            'eligibility sealing requires READ COMMITTED transaction isolation';
    END IF;
    -- Serialize status decisions for one immutable generation.  A rollback is
    -- explicit: accepted -> retired, with reviewer/reason on the new event.
    -- The current projection then falls back to the highest older generation
    -- whose latest state is still accepted.
    SELECT generation.effective_at,
           generation.expected_projection_seal_sha256,
           generation.expected_projection_row_count
      INTO generation_effective_at, expected_projection_seal_sha256,
           expected_projection_row_count
      FROM ops.eligibility_generations generation
     WHERE generation.generation_sha256 = NEW.eligibility_generation_sha256
       AND generation.generation_sequence = NEW.generation_sequence
     FOR UPDATE;
    IF generation_effective_at IS NULL THEN
        RAISE EXCEPTION 'unknown eligibility generation % sequence %',
            NEW.eligibility_generation_sha256, NEW.generation_sequence;
    END IF;

    -- BEFORE INSERT runs before ON CONFLICT. Lock first so a concurrent exact
    -- retry sees the committed winner, then preserve its semantic no-op path.
    SELECT record_sha256
      INTO existing_record_sha256
      FROM ops.eligibility_generation_status_events
     WHERE idempotency_key = NEW.idempotency_key;
    IF FOUND AND existing_record_sha256 = NEW.record_sha256 THEN
        RETURN NEW;
    END IF;

    SELECT status, effective_at, projection_seal_sha256, projection_row_count
      INTO previous_status, previous_effective_at,
           previous_projection_seal_sha256, previous_projection_row_count
      FROM ops.eligibility_generation_status_events
     WHERE eligibility_generation_sha256 = NEW.eligibility_generation_sha256
     ORDER BY effective_at DESC, created_at DESC, idempotency_key DESC
     LIMIT 1;

    IF previous_status IS NULL THEN
        IF NEW.status <> 'candidate' THEN
            RAISE EXCEPTION 'first eligibility generation status must be candidate';
        END IF;
        IF NEW.effective_at < generation_effective_at THEN
            RAISE EXCEPTION
                'first eligibility status effective_at precedes generation effective_at';
        END IF;
    ELSIF NEW.effective_at <= previous_effective_at THEN
        RAISE EXCEPTION 'eligibility status effective_at must advance monotonically';
    ELSIF previous_status = 'candidate'
          AND NEW.status NOT IN ('accepted', 'rejected') THEN
        RAISE EXCEPTION 'invalid eligibility status transition: % -> %',
            previous_status, NEW.status;
    ELSIF previous_status = 'accepted' AND NEW.status <> 'retired' THEN
        RAISE EXCEPTION 'invalid eligibility status transition: % -> %',
            previous_status, NEW.status;
    ELSIF previous_status IN ('rejected', 'retired') THEN
        RAISE EXCEPTION 'terminal eligibility status cannot transition: %',
            previous_status;
    END IF;
    IF NEW.status IN ('candidate', 'accepted') THEN
        SELECT seal.projection_seal_sha256, seal.projection_row_count
          INTO actual_projection_seal_sha256, actual_projection_row_count
          FROM ops.compute_eligibility_projection_seal(
              NEW.eligibility_generation_sha256
          ) seal;
        IF actual_projection_row_count <= 0 THEN
            RAISE EXCEPTION 'eligibility projection cannot seal an empty generation';
        END IF;
        IF actual_projection_seal_sha256 IS DISTINCT FROM
               expected_projection_seal_sha256
           OR actual_projection_row_count IS DISTINCT FROM
               expected_projection_row_count
           OR NEW.projection_seal_sha256 IS DISTINCT FROM
               expected_projection_seal_sha256
           OR NEW.projection_row_count IS DISTINCT FROM
               expected_projection_row_count THEN
            RAISE EXCEPTION 'eligibility projection seal/count mismatch for generation %',
                NEW.eligibility_generation_sha256;
        END IF;
        IF NEW.status = 'accepted' AND (
            previous_projection_seal_sha256 IS DISTINCT FROM
                NEW.projection_seal_sha256
            OR previous_projection_row_count IS DISTINCT FROM
                NEW.projection_row_count
        ) THEN
            RAISE EXCEPTION 'accepted eligibility seal must equal sealed candidate';
        END IF;
        IF NEW.status IN ('candidate', 'accepted') THEN
            PERFORM ops.assert_eligibility_candidate_ready(
                NEW.eligibility_generation_sha256, NEW.effective_at
            );
        END IF;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS eligibility_generation_status_transition_guard
    ON ops.eligibility_generation_status_events;
CREATE TRIGGER eligibility_generation_status_transition_guard
BEFORE INSERT ON ops.eligibility_generation_status_events
FOR EACH ROW EXECUTE FUNCTION ops.validate_eligibility_generation_status_event();

DO $eligibility_record_hash_triggers$
DECLARE
    relation regclass;
    trigger_name text;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.eligibility_generations'::regclass,
        'ops.eligibility_generation_status_events'::regclass,
        'ops.player_entities'::regclass,
        'ops.player_identity_observations'::regclass,
        'ops.player_alias_observations'::regclass,
        'ops.player_profile_observations'::regclass,
        'ops.eligibility_match_round_observations'::regclass,
        'ops.eligibility_review_events'::regclass
    ]
    LOOP
        trigger_name := 'eligibility_' || relation::oid::text
                        || '_record_hash_guard';
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %s', trigger_name, relation);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE INSERT OR UPDATE ON %s '
            'FOR EACH ROW EXECUTE FUNCTION ops.require_record_sha256()',
            trigger_name, relation
        );
    END LOOP;
END
$eligibility_record_hash_triggers$;

DO $eligibility_immutable_triggers$
DECLARE
    relation regclass;
    trigger_name text;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.eligibility_generations'::regclass,
        'ops.eligibility_generation_status_events'::regclass,
        'ops.player_entities'::regclass,
        'ops.player_identity_observations'::regclass,
        'ops.player_alias_observations'::regclass,
        'ops.player_profile_observations'::regclass,
        'ops.eligibility_match_round_observations'::regclass,
        'ops.eligibility_review_events'::regclass
    ]
    LOOP
        trigger_name := 'eligibility_' || relation::oid::text
                        || '_immutable_guard';
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %s', trigger_name, relation);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %s '
            'FOR EACH ROW EXECUTE FUNCTION ops.reject_immutable_fact_mutation()',
            trigger_name, relation
        );
    END LOOP;
END
$eligibility_immutable_triggers$;

REVOKE ALL ON ALL TABLES IN SCHEMA raw, ops, ml, api FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.validate_eligibility_source_artifact() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.validate_eligibility_review_target() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.compute_eligibility_projection_seal(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.assert_eligibility_candidate_ready(text, timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.guard_eligibility_projection_content() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.validate_eligibility_generation_status_event() FROM PUBLIC;

ALTER TABLE ops.eligibility_generations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.eligibility_generation_status_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.player_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.player_identity_observations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.player_alias_observations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.player_profile_observations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.eligibility_match_round_observations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.eligibility_review_events ENABLE ROW LEVEL SECURITY;

INSERT INTO ops.schema_versions (version, migration_name)
VALUES ('1.2.0', '20260714030000_eligibility_provenance_v1_2.sql')
ON CONFLICT (version) DO NOTHING;

COMMIT;
