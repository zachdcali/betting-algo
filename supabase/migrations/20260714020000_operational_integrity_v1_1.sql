-- Operational integrity contract 1.1.0
-- Closes the first staging review's multi-batch parity, terminal lifecycle,
-- correction lineage, feature validation, and exact inference-link gaps.

BEGIN;

-- Contract 1.0.0 was staging-only and did not persist semantic row hashes or
-- validate every already-loaded feature vector. There is no lossless SQL-only
-- way to reconstruct those hashes. Refuse a path-dependent in-place upgrade;
-- operators must rebuild staging from the immutable sources under 1.1.0.
DO $require_clean_v1_upgrade$
DECLARE
    relation regclass;
    populated boolean;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ops.schema_versions WHERE version = '1.1.0'
    ) THEN
        FOREACH relation IN ARRAY ARRAY[
            'ops.pipeline_runs'::regclass,
            'ops.pipeline_run_stages'::regclass,
            'raw.source_fetches'::regclass,
            'raw.source_artifacts'::regclass,
            'ops.odds_observations'::regclass,
            'ops.match_metadata_observations'::regclass,
            'ml.feature_schemas'::regclass,
            'ml.feature_snapshots'::regclass,
            'ml.prediction_observations'::regclass,
            'ops.paper_accounts'::regclass,
            'ops.paper_sessions'::regclass,
            'ops.account_ledger'::regclass,
            'ops.bet_recommendations'::regclass,
            'ops.bet_state_events'::regclass,
            'ops.settlement_attempts'::regclass,
            'ops.settlement_events'::regclass,
            'ops.skip_events'::regclass
        ]
        LOOP
            EXECUTE format('SELECT EXISTS (SELECT 1 FROM %s LIMIT 1)', relation)
               INTO populated;
            IF populated THEN
                RAISE EXCEPTION
                    'operational 1.1.0 requires a clean staging rebuild; %.% contains 1.0.0 rows',
                    split_part(relation::text, '.', 1),
                    split_part(relation::text, '.', 2);
            END IF;
        END LOOP;
    END IF;
END
$require_clean_v1_upgrade$;

DO $record_hash_columns$
DECLARE
    relation regclass;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.pipeline_runs'::regclass,
        'ops.pipeline_run_stages'::regclass,
        'raw.source_fetches'::regclass,
        'raw.source_artifacts'::regclass,
        'ops.odds_observations'::regclass,
        'ops.match_metadata_observations'::regclass,
        'ml.feature_schemas'::regclass,
        'ml.feature_snapshots'::regclass,
        'ml.prediction_observations'::regclass,
        'ops.paper_accounts'::regclass,
        'ops.paper_sessions'::regclass,
        'ops.account_ledger'::regclass,
        'ops.bet_recommendations'::regclass,
        'ops.bet_state_events'::regclass,
        'ops.settlement_attempts'::regclass,
        'ops.settlement_events'::regclass,
        'ops.skip_events'::regclass
    ]
    LOOP
        EXECUTE format(
            'ALTER TABLE %s ADD COLUMN IF NOT EXISTS record_sha256 text',
            relation
        );
    END LOOP;
END
$record_hash_columns$;

CREATE TABLE IF NOT EXISTS ops.import_batch_memberships (
    import_batch_membership_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    import_batch_id uuid NOT NULL REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    target_table text NOT NULL,
    target_idempotency_key text NOT NULL,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    target_record_sha256 text NOT NULL,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (import_batch_id, target_table, target_idempotency_key),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (target_record_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (target_table IN (
        'raw.source_fetches', 'raw.source_artifacts',
        'ops.pipeline_runs', 'ops.pipeline_run_stages',
        'ops.odds_observations', 'ops.match_metadata_observations',
        'ops.paper_accounts', 'ops.paper_sessions', 'ops.account_ledger',
        'ops.account_journal_entries', 'ops.bet_recommendations',
        'ops.bet_state_events', 'ops.settlement_attempts',
        'ops.settlement_events', 'ops.skip_events', 'ops.import_conflicts',
        'ml.feature_schemas', 'ml.feature_snapshots', 'ml.model_releases',
        'ml.model_registry_generations',
        'ml.model_release_status_events',
        'ml.prediction_observations'
    ))
);

CREATE INDEX IF NOT EXISTS import_batch_memberships_target_idx
    ON ops.import_batch_memberships (target_table, target_idempotency_key);

CREATE TABLE IF NOT EXISTS ops.import_conflicts (
    import_conflict_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    import_batch_id uuid NOT NULL REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    target_table text NOT NULL,
    target_idempotency_key text NOT NULL,
    conflict_type text NOT NULL,
    candidate_record_sha256 text NOT NULL,
    candidate_record jsonb NOT NULL,
    review_status text NOT NULL DEFAULT 'open',
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (candidate_record_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (jsonb_typeof(candidate_record) = 'object'),
    CHECK (review_status = 'open'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.import_conflict_resolutions (
    import_conflict_resolution_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    import_conflict_key text NOT NULL REFERENCES ops.import_conflicts(idempotency_key)
        DEFERRABLE INITIALLY DEFERRED,
    disposition text NOT NULL,
    resolved_at timestamptz NOT NULL,
    resolved_by text NOT NULL,
    reason text NOT NULL,
    resolution_mapping jsonb NOT NULL DEFAULT '{}'::jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (disposition IN ('accept', 'reject', 'supersede', 'remap', 'quarantine')),
    CHECK (jsonb_typeof(resolution_mapping) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS import_conflicts_target_idx
    ON ops.import_conflicts (target_table, target_idempotency_key, review_status);

CREATE TABLE IF NOT EXISTS ml.model_releases (
    model_release_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    model_family text NOT NULL,
    model_version text NOT NULL,
    release_status text NOT NULL,
    registry_schema_version text,
    feature_schema_identifier text,
    feature_schema_sha256 text,
    feature_count integer,
    training_feature_semantics_id text,
    live_feature_semantics_id text,
    training_dataset_id text,
    model_sha256 text,
    scaler_sha256 text,
    calibrator_sha256 text,
    calibration_version text,
    registry_entry jsonb NOT NULL,
    contract_complete boolean NOT NULL DEFAULT false,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_family, model_version),
    CONSTRAINT model_releases_key_family_unique
        UNIQUE (idempotency_key, model_family),
    CHECK (jsonb_typeof(registry_entry) = 'object'),
    CHECK (feature_schema_sha256 IS NULL OR feature_schema_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (model_sha256 IS NULL OR model_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (scaler_sha256 IS NULL OR scaler_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (calibrator_sha256 IS NULL OR calibrator_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (record_sha256 IS NULL OR record_sha256 ~ '^[0-9a-f]{64}$')
);

ALTER TABLE ml.model_releases
    ADD COLUMN IF NOT EXISTS calibration_version text,
    ADD COLUMN IF NOT EXISTS feature_count integer;

-- `contract_complete` is consumed by the decision-eligibility trigger, so it
-- cannot be a caller assertion.  Enforce normalized SemVer and bind every
-- complete row back to the artifact metadata carried in registry_entry.
DO $model_release_contract_checks$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_family_check'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_family_check
            CHECK (model_family IN ('nn', 'xgboost', 'random_forest'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_feature_count_check'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_feature_count_check
            CHECK (feature_count IS NULL OR feature_count > 0);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_model_version_semver_check'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_model_version_semver_check
            CHECK (
                model_version ~
                '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$'
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_calibration_version_semver_check'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_calibration_version_semver_check
            CHECK (
                calibration_version IS NULL
                OR calibration_version ~
                   '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$'
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_registry_schema_semver_check'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_registry_schema_semver_check
            CHECK (
                registry_schema_version IS NULL
                OR registry_schema_version ~
                   '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$'
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_contract_complete_check'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_contract_complete_check
            CHECK (
                NOT contract_complete
                OR (
                    registry_schema_version IS NOT NULL
                    AND feature_schema_identifier IS NOT NULL
                    AND feature_schema_sha256 IS NOT NULL
                    AND feature_count IS NOT NULL
                    AND training_feature_semantics_id IS NOT NULL
                    AND live_feature_semantics_id IS NOT NULL
                    AND training_dataset_id IS NOT NULL
                    AND model_sha256 IS NOT NULL
                    AND nullif(btrim(registry_entry ->> 'model_file'), '')
                        IS NOT NULL
                    AND (
                        NOT registry_entry ? 'artifact_available'
                        OR registry_entry -> 'artifact_available' =
                            'true'::jsonb
                    )
                    AND registry_entry ->> 'model_sha256'
                        IS NOT DISTINCT FROM model_sha256
                    AND registry_entry ->> 'feature_schema_id'
                        IS NOT DISTINCT FROM
                        feature_schema_identifier
                    AND registry_entry ->> 'feature_schema_sha256'
                        IS NOT DISTINCT FROM
                        feature_schema_sha256
                    AND jsonb_typeof(registry_entry -> 'features') = 'number'
                    AND CASE
                        WHEN registry_entry ->> 'features'
                            ~ '^[1-9][0-9]*$'
                        THEN (registry_entry ->> 'features')::bigint =
                            feature_count
                        ELSE false
                    END
                    AND registry_entry ->> 'training_feature_semantics_id'
                        IS NOT DISTINCT FROM
                        training_feature_semantics_id
                    AND registry_entry ->> 'live_feature_semantics_id'
                        IS NOT DISTINCT FROM
                        live_feature_semantics_id
                    AND registry_entry ->> 'training_dataset_id'
                        IS NOT DISTINCT FROM
                        training_dataset_id
                    AND coalesce(
                        nullif(lower(btrim(
                            registry_entry ->> 'probability_mode'
                        )), ''),
                        'raw'
                    ) IN ('raw', 'calibrated')
                    AND (
                        (
                            model_family <> 'nn'
                            AND coalesce(
                                nullif(lower(btrim(
                                    registry_entry ->> 'probability_mode'
                                )), ''),
                                'raw'
                            ) = 'raw'
                        )
                        OR (
                            model_family = 'nn'
                            AND scaler_sha256 IS NOT NULL
                            AND nullif(btrim(
                                registry_entry ->> 'scaler_file'
                            ), '') IS NOT NULL
                            AND registry_entry ->> 'scaler_sha256'
                                IS NOT DISTINCT FROM
                                scaler_sha256
                            AND (
                                coalesce(
                                    nullif(lower(btrim(
                                        registry_entry ->> 'probability_mode'
                                    )), ''),
                                    'raw'
                                ) = 'raw'
                                OR (
                                    coalesce(
                                        nullif(lower(btrim(
                                            registry_entry ->>
                                                'probability_mode'
                                        )), ''),
                                        'raw'
                                    ) = 'calibrated'
                                    AND calibrator_sha256 IS NOT NULL
                                    AND calibration_version IS NOT NULL
                                    AND nullif(btrim(
                                        registry_entry ->>
                                            'calibrated_model_file'
                                    ), '') IS NOT NULL
                                    AND registry_entry ->>
                                        'calibrated_model_sha256'
                                        IS NOT DISTINCT FROM
                                        calibrator_sha256
                                    AND (
                                        CASE
                                            WHEN registry_entry ->>
                                                'calibration_version'
                                                ~ '^v[0-9]'
                                            THEN substr(
                                                registry_entry ->>
                                                    'calibration_version',
                                                2
                                            )
                                            ELSE registry_entry ->>
                                                'calibration_version'
                                        END
                                    ) IS NOT DISTINCT FROM
                                        calibration_version
                                )
                            )
                        )
                    )
                )
            );
    END IF;
END
$model_release_contract_checks$;

DO $model_release_feature_schema_fk$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'feature_schemas_identifier_hash_count_unique'
          AND conrelid = 'ml.feature_schemas'::regclass
    ) THEN
        ALTER TABLE ml.feature_schemas
            ADD CONSTRAINT feature_schemas_identifier_hash_count_unique
            UNIQUE (schema_identifier, schema_sha256, feature_count);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_feature_schema_fkey'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_feature_schema_fkey
            FOREIGN KEY (
                feature_schema_identifier,
                feature_schema_sha256,
                feature_count
            ) REFERENCES ml.feature_schemas (
                schema_identifier,
                schema_sha256,
                feature_count
            ) DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$model_release_feature_schema_fk$;

-- A registry file hash identifies its bytes, but hashes have no temporal
-- ordering. The source-owned sequence is therefore the promotion authority:
-- importing an older, previously unseen generation later cannot make its
-- statuses current merely because its database row was created later.
CREATE TABLE IF NOT EXISTS ml.model_registry_generations (
    model_registry_generation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    registry_generation_sha256 text NOT NULL UNIQUE,
    generation_sequence bigint NOT NULL UNIQUE,
    registry_schema_version text NOT NULL,
    effective_at timestamptz NOT NULL,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (registry_generation_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (generation_sequence > 0),
    CHECK (registry_schema_version <> ''),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (source_row_json IS NULL OR jsonb_typeof(source_row_json) = 'object'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ml.model_release_status_events (
    model_release_status_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    model_release_key text NOT NULL REFERENCES ml.model_releases(idempotency_key)
        DEFERRABLE INITIALLY DEFERRED,
    model_family text NOT NULL,
    registry_generation_sha256 text NOT NULL,
    release_status text NOT NULL,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (registry_generation_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (release_status IN ('promoted', 'candidate', 'archived', 'superseded')),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (record_sha256 ~ '^[0-9a-f]{64}$')
);

-- Keep the family on each status event so Postgres can enforce the registry
-- invariant without a race-prone trigger lookup. Refuse to infer it into an
-- already-populated pre-release schema: staging must be rebuilt from the
-- immutable registry source under the complete 1.1.0 contract.
ALTER TABLE ml.model_release_status_events
    ADD COLUMN IF NOT EXISTS model_family text;
DO $require_status_family_rebuild$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM ml.model_release_status_events
         WHERE model_family IS NULL
    ) THEN
        RAISE EXCEPTION
            'operational 1.1.0 requires a clean staging rebuild; model release status family is missing';
    END IF;
END
$require_status_family_rebuild$;
ALTER TABLE ml.model_release_status_events
    ALTER COLUMN model_family SET NOT NULL;

DO $model_release_status_generation_contract$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_releases_key_family_unique'
          AND conrelid = 'ml.model_releases'::regclass
    ) THEN
        ALTER TABLE ml.model_releases
            ADD CONSTRAINT model_releases_key_family_unique
            UNIQUE (idempotency_key, model_family);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_release_status_events_generation_fkey'
          AND conrelid = 'ml.model_release_status_events'::regclass
    ) THEN
        ALTER TABLE ml.model_release_status_events
            ADD CONSTRAINT model_release_status_events_generation_fkey
            FOREIGN KEY (registry_generation_sha256)
            REFERENCES ml.model_registry_generations(registry_generation_sha256)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'model_release_status_events_release_family_fkey'
          AND conrelid = 'ml.model_release_status_events'::regclass
    ) THEN
        ALTER TABLE ml.model_release_status_events
            ADD CONSTRAINT model_release_status_events_release_family_fkey
            FOREIGN KEY (model_release_key, model_family)
            REFERENCES ml.model_releases(idempotency_key, model_family)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$model_release_status_generation_contract$;

CREATE UNIQUE INDEX IF NOT EXISTS model_release_status_one_per_generation_idx
    ON ml.model_release_status_events (
        model_release_key, registry_generation_sha256
    );
CREATE INDEX IF NOT EXISTS model_release_status_generation_idx
    ON ml.model_release_status_events (registry_generation_sha256, model_release_key);
CREATE UNIQUE INDEX IF NOT EXISTS model_release_one_promoted_per_family_generation_idx
    ON ml.model_release_status_events (
        registry_generation_sha256, model_family
    )
    WHERE release_status = 'promoted';

CREATE OR REPLACE FUNCTION ml.block_unlogged_calibrated_promotion()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    probability_mode text;
BEGIN
    IF NEW.release_status = 'promoted' AND NEW.model_family = 'nn' THEN
        SELECT coalesce(
                   nullif(lower(btrim(
                       release.registry_entry ->> 'probability_mode'
                   )), ''),
                   'raw'
               )
          INTO probability_mode
          FROM ml.model_releases AS release
         WHERE release.idempotency_key = NEW.model_release_key
           AND release.model_family = NEW.model_family;
        IF probability_mode = 'calibrated' THEN
            RAISE EXCEPTION
                'calibrated NN promotion is blocked until nn_calibration_version lineage is persisted';
        END IF;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS model_release_calibrated_promotion_guard
    ON ml.model_release_status_events;
CREATE CONSTRAINT TRIGGER model_release_calibrated_promotion_guard
AFTER INSERT ON ml.model_release_status_events
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION ml.block_unlogged_calibrated_promotion();

ALTER TABLE ml.prediction_observations
    ADD COLUMN IF NOT EXISTS model_release_key text,
    ADD COLUMN IF NOT EXISTS decision_eligible boolean NOT NULL DEFAULT false;

DO $prediction_release_fk$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'prediction_observations_model_release_key_fkey'
          AND conrelid = 'ml.prediction_observations'::regclass
    ) THEN
        ALTER TABLE ml.prediction_observations
            ADD CONSTRAINT prediction_observations_model_release_key_fkey
            FOREIGN KEY (model_release_key)
            REFERENCES ml.model_releases(idempotency_key)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'prediction_observations_probability_pair_check'
          AND conrelid = 'ml.prediction_observations'::regclass
    ) THEN
        ALTER TABLE ml.prediction_observations
            ADD CONSTRAINT prediction_observations_probability_pair_check
            CHECK (
                player1_probability IS NULL
                OR abs(player1_probability + player2_probability - 1) <= 0.0001
            );
    END IF;
END
$prediction_release_fk$;

ALTER TABLE ops.odds_observations
    ADD COLUMN IF NOT EXISTS validation_status text NOT NULL DEFAULT 'legacy_unverified',
    ADD COLUMN IF NOT EXISTS inference_eligible boolean NOT NULL DEFAULT false;

DO $odds_validation_checks$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'odds_observations_probability_pair_check'
          AND conrelid = 'ops.odds_observations'::regclass
    ) THEN
        ALTER TABLE ops.odds_observations
            ADD CONSTRAINT odds_observations_probability_pair_check
            CHECK (
                (player1_market_probability IS NULL AND player2_market_probability IS NULL)
                OR (
                    player1_market_probability IS NOT NULL
                    AND player2_market_probability IS NOT NULL
                    AND abs(player1_market_probability + player2_market_probability - 1) <= 0.0001
                )
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'odds_observations_inference_evidence_check'
          AND conrelid = 'ops.odds_observations'::regclass
    ) THEN
        ALTER TABLE ops.odds_observations
            ADD CONSTRAINT odds_observations_inference_evidence_check
            CHECK (
                NOT inference_eligible
                OR (
                    validation_status = 'valid_two_sided_prestart'
                    AND match_uid IS NOT NULL
                    AND observed_at IS NOT NULL
                    AND match_start_at_utc IS NOT NULL
                    AND observed_at < match_start_at_utc
                    AND player1_decimal_odds > 1
                    AND player2_decimal_odds > 1
                )
            );
    END IF;
END
$odds_validation_checks$;

ALTER TABLE ops.bet_recommendations
    ADD COLUMN IF NOT EXISTS prediction_observation_id uuid,
    ADD COLUMN IF NOT EXISTS odds_observation_id uuid,
    ADD COLUMN IF NOT EXISTS evidence_quality text NOT NULL DEFAULT 'legacy_unlinked';

DO $bet_evidence_links$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'bet_recommendations_prediction_observation_id_fkey'
          AND conrelid = 'ops.bet_recommendations'::regclass
    ) THEN
        ALTER TABLE ops.bet_recommendations
            ADD CONSTRAINT bet_recommendations_prediction_observation_id_fkey
            FOREIGN KEY (prediction_observation_id)
            REFERENCES ml.prediction_observations(prediction_observation_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'bet_recommendations_odds_observation_id_fkey'
          AND conrelid = 'ops.bet_recommendations'::regclass
    ) THEN
        ALTER TABLE ops.bet_recommendations
            ADD CONSTRAINT bet_recommendations_odds_observation_id_fkey
            FOREIGN KEY (odds_observation_id)
            REFERENCES ops.odds_observations(odds_observation_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'bet_recommendations_decision_grade_links_check'
          AND conrelid = 'ops.bet_recommendations'::regclass
    ) THEN
        ALTER TABLE ops.bet_recommendations
            ADD CONSTRAINT bet_recommendations_decision_grade_links_check
            CHECK (
                evidence_quality <> 'decision_grade'
                OR (
                    prediction_observation_id IS NOT NULL
                    AND odds_observation_id IS NOT NULL
                    AND feature_snapshot_id IS NOT NULL
                )
            );
    END IF;
END
$bet_evidence_links$;

ALTER TABLE ops.bet_state_events
    ADD COLUMN IF NOT EXISTS supersedes_bet_state_event_id uuid,
    ADD COLUMN IF NOT EXISTS correction_reason text;

ALTER TABLE ops.settlement_events
    ADD COLUMN IF NOT EXISTS supersedes_settlement_event_id uuid,
    ADD COLUMN IF NOT EXISTS correction_reason text;

DO $correction_fks$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'bet_state_events_supersedes_fkey'
          AND conrelid = 'ops.bet_state_events'::regclass
    ) THEN
        ALTER TABLE ops.bet_state_events
            ADD CONSTRAINT bet_state_events_supersedes_fkey
            FOREIGN KEY (supersedes_bet_state_event_id)
            REFERENCES ops.bet_state_events(bet_state_event_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'settlement_events_supersedes_fkey'
          AND conrelid = 'ops.settlement_events'::regclass
    ) THEN
        ALTER TABLE ops.settlement_events
            ADD CONSTRAINT settlement_events_supersedes_fkey
            FOREIGN KEY (supersedes_settlement_event_id)
            REFERENCES ops.settlement_events(settlement_event_id)
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END
$correction_fks$;

DO $state_contract_checks$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'bet_state_events_state_check'
          AND conrelid = 'ops.bet_state_events'::regclass
    ) THEN
        ALTER TABLE ops.bet_state_events
            ADD CONSTRAINT bet_state_events_state_check
            CHECK (lower(state) IN (
                'pending', 'placed', 'settled', 'won', 'lost',
                'void', 'voided', 'cancelled', 'canceled'
            ));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'bet_state_events_correction_pair_check'
          AND conrelid = 'ops.bet_state_events'::regclass
    ) THEN
        ALTER TABLE ops.bet_state_events
            ADD CONSTRAINT bet_state_events_correction_pair_check
            CHECK (
                (supersedes_bet_state_event_id IS NULL)
                = (correction_reason IS NULL)
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'settlement_events_correction_pair_check'
          AND conrelid = 'ops.settlement_events'::regclass
    ) THEN
        ALTER TABLE ops.settlement_events
            ADD CONSTRAINT settlement_events_correction_pair_check
            CHECK (
                (supersedes_settlement_event_id IS NULL)
                = (correction_reason IS NULL)
            );
    END IF;
END
$state_contract_checks$;

CREATE TABLE IF NOT EXISTS ops.account_journal_entries (
    account_journal_entry_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    account_code text NOT NULL REFERENCES ops.paper_accounts(account_code)
        DEFERRABLE INITIALLY DEFERRED,
    external_bet_id text,
    entry_type text NOT NULL,
    occurred_at timestamptz NOT NULL,
    amount numeric NOT NULL,
    balance_after numeric,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    record_sha256 text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (entry_type IN (
        'opening_balance', 'stake_reserve', 'stake_release', 'win_profit',
        'loss', 'void_refund', 'deposit', 'withdrawal', 'adjustment'
    )),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (record_sha256 IS NULL OR record_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS model_releases_family_version_idx
    ON ml.model_releases (model_family, model_version);
CREATE INDEX IF NOT EXISTS account_journal_account_time_idx
    ON ops.account_journal_entries (account_code, occurred_at, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS bet_state_events_one_child_idx
    ON ops.bet_state_events (supersedes_bet_state_event_id)
    WHERE supersedes_bet_state_event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS settlement_events_one_child_idx
    ON ops.settlement_events (supersedes_settlement_event_id)
    WHERE supersedes_settlement_event_id IS NOT NULL;

CREATE OR REPLACE FUNCTION ops.guard_terminal_status_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    old_status text := lower(coalesce(OLD.status, ''));
    terminal boolean := false;
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'lifecycle row cannot be deleted from %.%',
            TG_TABLE_SCHEMA, TG_TABLE_NAME;
    END IF;
    terminal := CASE TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME
        WHEN 'ops.import_batches' THEN old_status IN ('verified', 'failed', 'cancelled', 'canceled')
        WHEN 'ops.pipeline_runs' THEN old_status IN ('success', 'partial', 'failed', 'cancelled', 'canceled')
        WHEN 'ops.pipeline_run_stages' THEN old_status IN ('success', 'partial', 'failed', 'skipped', 'cancelled', 'canceled')
        WHEN 'raw.source_fetches' THEN old_status IN ('success', 'partial', 'failed', 'blocked', 'cancelled', 'canceled')
        ELSE false
    END;
    IF terminal AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'terminal row is immutable on %.%: status %',
            TG_TABLE_SCHEMA, TG_TABLE_NAME, OLD.status;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS import_batches_terminal_guard ON ops.import_batches;
CREATE TRIGGER import_batches_terminal_guard
BEFORE UPDATE OR DELETE ON ops.import_batches
FOR EACH ROW EXECUTE FUNCTION ops.guard_terminal_status_update();
DROP TRIGGER IF EXISTS pipeline_runs_terminal_guard ON ops.pipeline_runs;
CREATE TRIGGER pipeline_runs_terminal_guard
BEFORE UPDATE OR DELETE ON ops.pipeline_runs
FOR EACH ROW EXECUTE FUNCTION ops.guard_terminal_status_update();
DROP TRIGGER IF EXISTS pipeline_run_stages_terminal_guard ON ops.pipeline_run_stages;
CREATE TRIGGER pipeline_run_stages_terminal_guard
BEFORE UPDATE OR DELETE ON ops.pipeline_run_stages
FOR EACH ROW EXECUTE FUNCTION ops.guard_terminal_status_update();
DROP TRIGGER IF EXISTS source_fetches_terminal_guard ON raw.source_fetches;
CREATE TRIGGER source_fetches_terminal_guard
BEFORE UPDATE OR DELETE ON raw.source_fetches
FOR EACH ROW EXECUTE FUNCTION ops.guard_terminal_status_update();

CREATE OR REPLACE FUNCTION ops.reject_immutable_fact_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    -- Immutable retries use an ON CONFLICT DO UPDATE expression solely to call
    -- require_matching_record_sha256(). Once that helper proves equality,
    -- suppress the physical no-op update. Any value change still fails.
    IF TG_OP = 'UPDATE' AND NEW IS NOT DISTINCT FROM OLD THEN
        RETURN NULL;
    END IF;
    RAISE EXCEPTION 'immutable operational fact cannot be %: %.%',
        lower(TG_OP), TG_TABLE_SCHEMA, TG_TABLE_NAME;
END
$$;

CREATE OR REPLACE FUNCTION ops.require_record_sha256()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.record_sha256 IS NULL
       OR NEW.record_sha256 !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION '%.% requires a valid semantic record_sha256',
            TG_TABLE_SCHEMA, TG_TABLE_NAME;
    END IF;
    RETURN NEW;
END
$$;

CREATE OR REPLACE FUNCTION ops.require_matching_record_sha256(
    existing_hash text,
    incoming_hash text,
    target_table text,
    target_key text
)
RETURNS text
LANGUAGE plpgsql
AS $$
BEGIN
    IF existing_hash IS NULL
       OR incoming_hash IS NULL
       OR existing_hash IS DISTINCT FROM incoming_hash THEN
        RAISE EXCEPTION
            'idempotency conflict on % for key %: record_sha256 mismatch',
            target_table, target_key;
    END IF;
    RETURN existing_hash;
END
$$;

DO $record_hash_triggers$
DECLARE
    relation regclass;
    trigger_name text;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.pipeline_runs'::regclass,
        'ops.pipeline_run_stages'::regclass,
        'raw.source_fetches'::regclass,
        'raw.source_artifacts'::regclass,
        'ops.odds_observations'::regclass,
        'ops.match_metadata_observations'::regclass,
        'ml.feature_schemas'::regclass,
        'ml.feature_snapshots'::regclass,
        'ml.model_releases'::regclass,
        'ml.model_registry_generations'::regclass,
        'ml.model_release_status_events'::regclass,
        'ml.prediction_observations'::regclass,
        'ops.paper_accounts'::regclass,
        'ops.paper_sessions'::regclass,
        'ops.account_ledger'::regclass,
        'ops.account_journal_entries'::regclass,
        'ops.bet_recommendations'::regclass,
        'ops.bet_state_events'::regclass,
        'ops.settlement_attempts'::regclass,
        'ops.settlement_events'::regclass,
        'ops.skip_events'::regclass,
        'ops.import_batch_memberships'::regclass,
        'ops.import_conflicts'::regclass,
        'ops.import_conflict_resolutions'::regclass
    ]
    LOOP
        trigger_name := replace(relation::text, '.', '_') || '_record_hash_guard';
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %s', trigger_name, relation);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE INSERT OR UPDATE ON %s '
            'FOR EACH ROW EXECUTE FUNCTION ops.require_record_sha256()',
            trigger_name, relation
        );
    END LOOP;
END
$record_hash_triggers$;

DO $immutable_triggers$
DECLARE
    relation regclass;
    trigger_name text;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.import_batch_memberships'::regclass,
        'raw.source_artifacts'::regclass,
        'ops.odds_observations'::regclass,
        'ops.match_metadata_observations'::regclass,
        'ml.feature_schemas'::regclass,
        'ml.feature_snapshots'::regclass,
        'ml.model_releases'::regclass,
        'ml.model_registry_generations'::regclass,
        'ml.model_release_status_events'::regclass,
        'ml.prediction_observations'::regclass,
        'ops.paper_accounts'::regclass,
        'ops.paper_sessions'::regclass,
        'ops.account_ledger'::regclass,
        'ops.account_journal_entries'::regclass,
        'ops.bet_recommendations'::regclass,
        'ops.bet_state_events'::regclass,
        'ops.settlement_attempts'::regclass,
        'ops.settlement_events'::regclass,
        'ops.skip_events'::regclass,
        'ops.import_conflicts'::regclass,
        'ops.import_conflict_resolutions'::regclass
    ]
    LOOP
        trigger_name := replace(relation::text, '.', '_') || '_immutable_guard';
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %s', trigger_name, relation);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %s '
            'FOR EACH ROW EXECUTE FUNCTION ops.reject_immutable_fact_mutation()',
            trigger_name, relation
        );
    END LOOP;
END
$immutable_triggers$;

CREATE OR REPLACE FUNCTION ml.validate_feature_snapshot()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    expected_count integer;
    expected_hash text;
    expected_names jsonb;
BEGIN
    SELECT feature_count, schema_sha256, feature_names
      INTO expected_count, expected_hash, expected_names
      FROM ml.feature_schemas
     WHERE schema_identifier = NEW.feature_schema_identifier;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown feature schema: %', NEW.feature_schema_identifier;
    END IF;
    IF NEW.feature_schema_sha256 IS NULL THEN
        NEW.feature_schema_sha256 := expected_hash;
    END IF;
    IF NEW.features_complete THEN
        IF NEW.feature_count <> expected_count
           OR NEW.feature_schema_sha256 <> expected_hash
           OR NEW.feature_vector_sha256 IS NULL
           OR jsonb_typeof(NEW.feature_vector) <> 'object'
           OR jsonb_array_length(NEW.defaulted_features) <> 0
           OR EXISTS (
                SELECT expected.feature_name
                  FROM jsonb_array_elements_text(expected_names)
                       AS expected(feature_name)
                EXCEPT
                SELECT actual.feature_name
                  FROM jsonb_object_keys(NEW.feature_vector)
                       AS actual(feature_name)
           )
           OR EXISTS (
                SELECT actual.feature_name
                  FROM jsonb_object_keys(NEW.feature_vector)
                       AS actual(feature_name)
                EXCEPT
                SELECT expected.feature_name
                  FROM jsonb_array_elements_text(expected_names)
                       AS expected(feature_name)
           )
           OR EXISTS (
                SELECT 1
                  FROM jsonb_each(NEW.feature_vector) AS item
                 WHERE jsonb_typeof(item.value) <> 'number'
           ) THEN
            RAISE EXCEPTION 'complete feature snapshot violates registered schema %',
                NEW.feature_schema_identifier;
        END IF;
        IF NEW.feature_schema_identifier = 'base_141@1.0.0'
           AND EXISTS (
                WITH grouped AS (
                    SELECT
                        CASE
                            WHEN item.key LIKE 'Surface\_%' ESCAPE '\'
                                 AND item.key <> 'Surface_Transition_Flag' THEN 'surface'
                            WHEN item.key LIKE 'Level\_%' ESCAPE '\' THEN 'level'
                            WHEN item.key LIKE 'Round\_%' ESCAPE '\' THEN 'round'
                            WHEN item.key LIKE 'P1\_Hand\_%' ESCAPE '\' THEN 'p1_hand'
                            WHEN item.key LIKE 'P2\_Hand\_%' ESCAPE '\' THEN 'p2_hand'
                            WHEN item.key LIKE 'P1\_Country\_%' ESCAPE '\' THEN 'p1_country'
                            WHEN item.key LIKE 'P2\_Country\_%' ESCAPE '\' THEN 'p2_country'
                            WHEN item.key LIKE 'Handedness\_Matchup\_%' ESCAPE '\'
                                THEN 'hand_matchup'
                        END AS group_name,
                        item.value::numeric AS value
                    FROM jsonb_each_text(NEW.feature_vector) AS item
                ), invalid AS (
                    SELECT group_name
                    FROM grouped
                    WHERE group_name IS NOT NULL
                    GROUP BY group_name
                    HAVING bool_or(value NOT IN (0, 1))
                       OR (
                            group_name = 'hand_matchup'
                            AND sum(value) NOT IN (0, 1)
                       )
                       OR (
                            group_name <> 'hand_matchup'
                            AND sum(value) <> 1
                       )
                )
                SELECT 1 FROM invalid
           ) THEN
            RAISE EXCEPTION 'complete base_141 snapshot violates one-hot cardinality';
        END IF;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS feature_snapshots_contract_guard ON ml.feature_snapshots;
CREATE TRIGGER feature_snapshots_contract_guard
BEFORE INSERT ON ml.feature_snapshots
FOR EACH ROW EXECUTE FUNCTION ml.validate_feature_snapshot();

CREATE OR REPLACE FUNCTION ml.validate_prediction_eligibility()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    snapshot_complete boolean;
    snapshot_lineage text;
    snapshot_schema text;
    snapshot_schema_hash text;
    snapshot_semantics text;
    snapshot_feature_count integer;
    release_complete boolean;
    release_status_value text;
    release_family text;
    release_version text;
    release_schema text;
    release_schema_hash text;
    release_live_semantics text;
    release_feature_count integer;
BEGIN
    IF NEW.decision_eligible THEN
        IF NEW.feature_snapshot_id IS NULL
           OR NEW.model_release_key IS NULL
           OR NEW.player1_probability IS NULL
           OR NEW.player2_probability IS NULL
           OR NEW.predicted_at IS NULL
           OR NEW.model_role <> 'promoted'
           OR NEW.logging_schema_version <> 'prediction_log_v2'
           OR NEW.logging_quality <> 'snapshot_v2' THEN
            RAISE EXCEPTION 'decision-eligible prediction requires exact feature and model release';
        END IF;
        SELECT features_complete, lineage_quality, feature_schema_identifier,
               feature_schema_sha256, feature_semantics_identifier,
               feature_count
          INTO snapshot_complete, snapshot_lineage, snapshot_schema,
               snapshot_schema_hash, snapshot_semantics,
               snapshot_feature_count
          FROM ml.feature_snapshots
         WHERE feature_snapshot_id = NEW.feature_snapshot_id;
        SELECT release.contract_complete, status.release_status,
               release.model_family, release.model_version,
               release.feature_schema_identifier,
               release.feature_schema_sha256,
               release.live_feature_semantics_id,
               release.feature_count
          INTO release_complete, release_status_value,
               release_family, release_version, release_schema,
               release_schema_hash, release_live_semantics,
               release_feature_count
          FROM ml.model_releases AS release
          JOIN ml.model_release_status_events AS status
            ON status.model_release_key = release.idempotency_key
           AND status.model_family = release.model_family
          JOIN ml.model_registry_generations AS generation
            ON generation.registry_generation_sha256 =
               status.registry_generation_sha256
         WHERE release.idempotency_key = NEW.model_release_key
           AND generation.generation_sequence = (
               SELECT max(latest.generation_sequence)
                 FROM ml.model_registry_generations AS latest
           );
        IF coalesce(snapshot_complete, false) IS NOT TRUE
           OR snapshot_lineage <> 'exact_feature_snapshot_id' THEN
            RAISE EXCEPTION 'decision-eligible prediction requires exact complete feature lineage';
        END IF;
        IF coalesce(release_complete, false) IS NOT TRUE
           OR release_status_value <> 'promoted'
           OR NEW.model_family IS DISTINCT FROM release_family
           OR NEW.model_version IS DISTINCT FROM release_version THEN
            RAISE EXCEPTION 'decision-eligible prediction requires a contract-complete promoted release';
        END IF;
        IF snapshot_schema IS DISTINCT FROM release_schema
           OR snapshot_schema_hash IS DISTINCT FROM release_schema_hash
           OR snapshot_feature_count IS DISTINCT FROM release_feature_count
           OR snapshot_semantics IS DISTINCT FROM release_live_semantics THEN
            RAISE EXCEPTION 'prediction release and feature snapshot contracts do not match';
        END IF;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS prediction_observations_eligibility_guard
    ON ml.prediction_observations;
CREATE TRIGGER prediction_observations_eligibility_guard
BEFORE INSERT ON ml.prediction_observations
FOR EACH ROW EXECUTE FUNCTION ml.validate_prediction_eligibility();

CREATE OR REPLACE FUNCTION ops.validate_bet_evidence()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    prediction ml.prediction_observations%ROWTYPE;
    odds ops.odds_observations%ROWTYPE;
    expected_model_probability numeric;
    expected_decimal_odds numeric;
    expected_market_probability numeric;
    expected_bet_side text;
    expected_kelly_fraction numeric;
BEGIN
    IF NEW.evidence_quality = 'decision_grade' THEN
        SELECT * INTO prediction
          FROM ml.prediction_observations
         WHERE prediction_observation_id = NEW.prediction_observation_id;
        IF NOT FOUND OR prediction.decision_eligible IS NOT TRUE THEN
            RAISE EXCEPTION 'decision-grade bet requires a decision-eligible prediction';
        END IF;
        SELECT * INTO odds
          FROM ops.odds_observations
         WHERE odds_observation_id = NEW.odds_observation_id;
        IF NOT FOUND OR odds.inference_eligible IS NOT TRUE THEN
            RAISE EXCEPTION 'decision-grade bet requires inference-eligible odds';
        END IF;
        IF NEW.match_uid IS NULL
           OR prediction.match_uid IS DISTINCT FROM NEW.match_uid
           OR odds.match_uid IS DISTINCT FROM NEW.match_uid
           OR prediction.feature_snapshot_id IS DISTINCT FROM NEW.feature_snapshot_id THEN
            RAISE EXCEPTION 'decision-grade bet evidence does not identify one exact decision';
        END IF;
        IF NEW.recommended_at IS NULL
           OR prediction.predicted_at IS NULL
           OR odds.observed_at IS NULL
           OR odds.match_start_at_utc IS NULL
           OR prediction.predicted_at > NEW.recommended_at
           OR odds.observed_at > NEW.recommended_at
           OR NEW.recommended_at >= odds.match_start_at_utc THEN
            RAISE EXCEPTION 'decision-grade bet evidence violates the pre-start timeline';
        END IF;
        IF NEW.bet_on_player1 IS NULL THEN
            RAISE EXCEPTION 'decision-grade bet requires an explicit selected side';
        END IF;
        expected_model_probability := CASE WHEN NEW.bet_on_player1
            THEN prediction.player1_probability ELSE prediction.player2_probability END;
        expected_decimal_odds := CASE WHEN NEW.bet_on_player1
            THEN odds.player1_decimal_odds ELSE odds.player2_decimal_odds END;
        expected_bet_side := CASE WHEN NEW.bet_on_player1
            THEN odds.player1 ELSE odds.player2 END;
        IF expected_model_probability IS NULL
           OR expected_decimal_odds IS NULL
           OR expected_decimal_odds <= 1
           OR expected_bet_side IS NULL THEN
            RAISE EXCEPTION 'decision-grade bet linked side has incomplete evidence';
        END IF;
        expected_market_probability := 1 / expected_decimal_odds;
        expected_kelly_fraction := greatest(
            0,
            (expected_model_probability * expected_decimal_odds - 1)
            / (expected_decimal_odds - 1)
        );
        IF NEW.model_probability IS NULL
           OR NEW.market_probability IS NULL
           OR NEW.decimal_odds IS NULL
           OR NEW.edge IS NULL
           OR NEW.kelly_fraction IS NULL
           OR NEW.stake IS NULL OR NEW.stake <= 0
           OR NEW.stake_fraction IS NULL
           OR NEW.stake_fraction <= 0 OR NEW.stake_fraction > 0.05
           OR abs(NEW.model_probability - expected_model_probability) > 0.000001
           OR abs(NEW.decimal_odds - expected_decimal_odds) > 0.000001
           OR abs(NEW.market_probability - expected_market_probability) > 0.000001
           OR abs(NEW.edge - (NEW.model_probability - NEW.market_probability)) > 0.000001
           OR abs(NEW.kelly_fraction - expected_kelly_fraction) > 0.000001
           OR NEW.model_version IS DISTINCT FROM prediction.model_version
           OR lower(btrim(NEW.bet_side)) IS DISTINCT FROM lower(btrim(expected_bet_side)) THEN
            RAISE EXCEPTION 'decision-grade bet calculations do not match linked evidence';
        END IF;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS bet_recommendations_evidence_guard
    ON ops.bet_recommendations;
CREATE TRIGGER bet_recommendations_evidence_guard
BEFORE INSERT ON ops.bet_recommendations
FOR EACH ROW EXECUTE FUNCTION ops.validate_bet_evidence();

CREATE OR REPLACE FUNCTION ops.validate_bet_state_transition()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    previous ops.bet_state_events%ROWTYPE;
    duplicate ops.bet_state_events%ROWTYPE;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(NEW.external_bet_id, 0));
    SELECT * INTO duplicate
      FROM ops.bet_state_events
     WHERE idempotency_key = NEW.idempotency_key;
    IF FOUND THEN
        IF duplicate.record_sha256 IS DISTINCT FROM NEW.record_sha256 THEN
            RAISE EXCEPTION 'idempotency conflict for bet state %', NEW.idempotency_key;
        END IF;
        RETURN NEW;
    END IF;
    SELECT * INTO previous
      FROM ops.bet_state_events
     WHERE external_bet_id = NEW.external_bet_id
       AND lower(state) IN ('settled', 'won', 'lost', 'void', 'voided', 'cancelled', 'canceled')
     ORDER BY occurred_at DESC NULLS LAST, created_at DESC
     LIMIT 1;
    IF FOUND THEN
        IF NEW.supersedes_bet_state_event_id IS DISTINCT FROM previous.bet_state_event_id
           OR coalesce(btrim(NEW.correction_reason), '') = ''
           OR lower(NEW.state) NOT IN (
                'settled', 'won', 'lost', 'void', 'voided',
                'cancelled', 'canceled'
           )
           OR NEW.occurred_at IS NULL
           OR previous.occurred_at IS NULL
           OR NEW.occurred_at < previous.occurred_at THEN
            RAISE EXCEPTION 'terminal bet state requires explicit correction chain for %',
                NEW.external_bet_id;
        END IF;
    ELSIF NEW.supersedes_bet_state_event_id IS NOT NULL THEN
        RAISE EXCEPTION 'bet correction does not supersede a terminal state for %',
            NEW.external_bet_id;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS bet_state_events_transition_guard ON ops.bet_state_events;
CREATE TRIGGER bet_state_events_transition_guard
BEFORE INSERT ON ops.bet_state_events
FOR EACH ROW EXECUTE FUNCTION ops.validate_bet_state_transition();

CREATE OR REPLACE FUNCTION ops.validate_settlement_correction()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    previous ops.settlement_events%ROWTYPE;
    duplicate ops.settlement_events%ROWTYPE;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(NEW.match_uid, 0));
    SELECT * INTO duplicate
      FROM ops.settlement_events
     WHERE idempotency_key = NEW.idempotency_key;
    IF FOUND THEN
        IF duplicate.record_sha256 IS DISTINCT FROM NEW.record_sha256 THEN
            RAISE EXCEPTION 'idempotency conflict for settlement %', NEW.idempotency_key;
        END IF;
        RETURN NEW;
    END IF;
    SELECT * INTO previous
      FROM ops.settlement_events
     WHERE match_uid = NEW.match_uid
     ORDER BY settled_at DESC NULLS LAST, created_at DESC
     LIMIT 1;
    IF FOUND THEN
        IF NEW.supersedes_settlement_event_id IS DISTINCT FROM previous.settlement_event_id
           OR coalesce(btrim(NEW.correction_reason), '') = ''
           OR NEW.settled_at IS NULL
           OR previous.settled_at IS NULL
           OR NEW.settled_at < previous.settled_at THEN
            RAISE EXCEPTION 'settlement correction requires explicit latest-event chain for %',
                NEW.match_uid;
        END IF;
    ELSIF NEW.supersedes_settlement_event_id IS NOT NULL THEN
        RAISE EXCEPTION 'settlement correction has no prior event for %', NEW.match_uid;
    END IF;
    RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS settlement_events_correction_guard ON ops.settlement_events;
CREATE TRIGGER settlement_events_correction_guard
BEFORE INSERT ON ops.settlement_events
FOR EACH ROW EXECUTE FUNCTION ops.validate_settlement_correction();

CREATE OR REPLACE VIEW api.current_predictions AS
SELECT DISTINCT ON (match_uid, model_family, model_version)
    prediction_observation_id, external_prediction_id, match_uid,
    external_feature_snapshot_id, predicted_at, model_family, model_version,
    model_role, player1_probability, player2_probability,
    logging_schema_version, logging_quality, metadata,
    model_release_key, decision_eligible
FROM ml.prediction_observations
ORDER BY match_uid, model_family, model_version, predicted_at DESC NULLS LAST,
         created_at DESC;

CREATE OR REPLACE VIEW api.current_model_releases AS
SELECT
    release.model_release_id,
    release.idempotency_key AS model_release_key,
    release.model_family,
    release.model_version,
    status.release_status,
    release.feature_schema_identifier,
    release.feature_schema_sha256,
    release.training_feature_semantics_id,
    release.live_feature_semantics_id,
    release.training_dataset_id,
    release.model_sha256,
    release.scaler_sha256,
    release.calibrator_sha256,
    release.contract_complete,
    status.registry_generation_sha256,
    generation.generation_sequence,
    generation.effective_at AS generation_effective_at,
    release.feature_count,
    release.calibration_version
FROM ml.model_releases AS release
JOIN ml.model_release_status_events AS status
  ON status.model_release_key = release.idempotency_key
 AND status.model_family = release.model_family
JOIN ml.model_registry_generations AS generation
  ON generation.registry_generation_sha256 =
     status.registry_generation_sha256
WHERE generation.generation_sequence = (
    SELECT max(latest.generation_sequence)
      FROM ml.model_registry_generations AS latest
);

CREATE OR REPLACE VIEW api.current_bet_states AS
SELECT DISTINCT ON (external_bet_id)
    bet_state_event_id, external_bet_id, occurred_at, state, outcome,
    actual_profit, balance_after, notes,
    supersedes_bet_state_event_id, correction_reason
FROM ops.bet_state_events
ORDER BY external_bet_id, occurred_at DESC NULLS LAST, created_at DESC;

CREATE OR REPLACE VIEW api.current_settlements AS
SELECT DISTINCT ON (match_uid)
    settlement_event_id, match_uid, settled_at, result_status,
    actual_winner, score, evidence,
    supersedes_settlement_event_id, correction_reason
FROM ops.settlement_events
ORDER BY match_uid, settled_at DESC NULLS LAST, created_at DESC;

REVOKE ALL ON ALL TABLES IN SCHEMA raw, ops, ml, api FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.guard_terminal_status_update() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.reject_immutable_fact_mutation() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.require_record_sha256() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.require_matching_record_sha256(text, text, text, text)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION ml.validate_feature_snapshot() FROM PUBLIC;
REVOKE ALL ON FUNCTION ml.validate_prediction_eligibility() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.validate_bet_evidence() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.validate_bet_state_transition() FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.validate_settlement_correction() FROM PUBLIC;

ALTER TABLE ops.import_batch_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.import_conflicts ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.import_conflict_resolutions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml.model_releases ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml.model_registry_generations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml.model_release_status_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE ops.account_journal_entries ENABLE ROW LEVEL SECURITY;

INSERT INTO ops.schema_versions (version, migration_name)
VALUES ('1.1.0', '20260714020000_operational_integrity_v1_1.sql')
ON CONFLICT (version) DO NOTHING;

COMMIT;
