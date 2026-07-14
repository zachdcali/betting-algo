-- Operational Postgres contract 1.0.0
--
-- This migration is additive. It does not alter the existing canonical
-- players/matches store or the transitional dash_* recovery bridge. The raw,
-- ops, and ml schemas are private write models; api contains reviewed read
-- projections. CSV remains authoritative until parity and staging gates pass.

BEGIN;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS api;

REVOKE ALL ON SCHEMA raw, ops, ml, api FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA ops REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA api REVOKE ALL ON TABLES FROM PUBLIC;

CREATE TABLE IF NOT EXISTS ops.schema_versions (
    version text PRIMARY KEY,
    migration_name text NOT NULL UNIQUE,
    applied_at timestamptz NOT NULL DEFAULT now(),
    CHECK (version ~ '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')
);

CREATE TABLE IF NOT EXISTS ops.import_batches (
    import_batch_pk bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id uuid NOT NULL UNIQUE,
    idempotency_key text NOT NULL UNIQUE,
    schema_version text NOT NULL,
    manifest_sha256 text,
    source_manifest jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL,
    planned_at timestamptz,
    completed_at timestamptz,
    row_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (manifest_sha256 IS NULL OR manifest_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (jsonb_typeof(source_manifest) = 'object'),
    CHECK (jsonb_typeof(row_counts) = 'object'),
    CHECK (completed_at IS NULL OR planned_at IS NULL OR completed_at >= planned_at)
);

CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_run_id text NOT NULL UNIQUE,
    run_kind text NOT NULL,
    status text NOT NULL,
    started_at timestamptz,
    completed_at timestamptz,
    error_message text,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (jsonb_typeof(metrics) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE IF NOT EXISTS ops.pipeline_run_stages (
    stage_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    stage_name text NOT NULL,
    attempt integer NOT NULL DEFAULT 1,
    status text NOT NULL,
    started_at timestamptz,
    completed_at timestamptz,
    error_message text,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (attempt >= 1),
    CHECK (jsonb_typeof(metrics) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE IF NOT EXISTS raw.source_fetches (
    source_fetch_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    source_name text NOT NULL,
    fetch_kind text NOT NULL,
    attempt integer NOT NULL DEFAULT 1,
    status text NOT NULL,
    started_at timestamptz,
    completed_at timestamptz,
    rows_observed integer,
    http_status integer,
    request jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_message text,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (attempt >= 1),
    CHECK (rows_observed IS NULL OR rows_observed >= 0),
    CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
    CHECK (jsonb_typeof(request) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at)
);

CREATE TABLE IF NOT EXISTS raw.source_artifacts (
    source_artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    source_fetch_id uuid REFERENCES raw.source_fetches(source_fetch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_fetch_key text REFERENCES raw.source_fetches(idempotency_key)
        DEFERRABLE INITIALLY DEFERRED,
    artifact_kind text NOT NULL,
    storage_uri text NOT NULL,
    content_sha256 text NOT NULL,
    content_type text,
    byte_size bigint,
    captured_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (byte_size IS NULL OR byte_size >= 0),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.odds_observations (
    odds_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_observation_id text NOT NULL,
    source_fetch_id uuid REFERENCES raw.source_fetches(source_fetch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_fetch_key text REFERENCES raw.source_fetches(idempotency_key)
        DEFERRABLE INITIALLY DEFERRED,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    match_uid text,
    match_anchor_key text,
    observed_at timestamptz,
    match_date date,
    match_start_at_utc timestamptz,
    tournament text,
    event_title text,
    surface text,
    level text,
    round text,
    round_code text,
    player1 text,
    player2 text,
    bookmaker text NOT NULL DEFAULT 'bovada',
    market_type text NOT NULL DEFAULT 'moneyline',
    player1_decimal_odds numeric,
    player2_decimal_odds numeric,
    player1_american_odds integer,
    player2_american_odds integer,
    player1_market_probability numeric,
    player2_market_probability numeric,
    market_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (player1_decimal_odds IS NULL OR player1_decimal_odds > 1),
    CHECK (player2_decimal_odds IS NULL OR player2_decimal_odds > 1),
    CHECK (player1_market_probability IS NULL OR player1_market_probability BETWEEN 0 AND 1),
    CHECK (player2_market_probability IS NULL OR player2_market_probability BETWEEN 0 AND 1),
    CHECK (jsonb_typeof(market_payload) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.match_metadata_observations (
    match_metadata_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    source_fetch_id uuid REFERENCES raw.source_fetches(source_fetch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_fetch_key text REFERENCES raw.source_fetches(idempotency_key)
        DEFERRABLE INITIALLY DEFERRED,
    match_uid text NOT NULL,
    match_anchor_key text NOT NULL,
    observed_at timestamptz NOT NULL,
    source_name text NOT NULL,
    match_date date,
    match_start_at_utc timestamptz,
    tournament text,
    event_title text,
    round_code text,
    surface text,
    level text,
    field_provenance jsonb NOT NULL,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (jsonb_typeof(field_provenance) = 'object'),
    CHECK (num_nonnulls(match_date, match_start_at_utc, tournament, event_title,
                        round_code, surface, level) > 0),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ml.feature_schemas (
    feature_schema_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    schema_name text NOT NULL,
    schema_version text NOT NULL,
    schema_identifier text NOT NULL UNIQUE,
    schema_sha256 text NOT NULL,
    feature_count integer NOT NULL,
    feature_names jsonb NOT NULL,
    feature_contract jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (schema_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (feature_count > 0),
    CHECK (jsonb_typeof(feature_names) = 'array'),
    CHECK (jsonb_array_length(feature_names) = feature_count),
    CHECK (jsonb_typeof(feature_contract) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ml.feature_snapshots (
    feature_snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_feature_snapshot_id text NOT NULL,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    match_uid text,
    feature_schema_identifier text NOT NULL REFERENCES ml.feature_schemas(schema_identifier)
        DEFERRABLE INITIALLY DEFERRED,
    feature_schema_sha256 text,
    feature_semantics_identifier text,
    captured_at timestamptz,
    build_status text NOT NULL,
    features_complete boolean NOT NULL DEFAULT false,
    lineage_quality text NOT NULL DEFAULT 'unknown',
    feature_count integer NOT NULL DEFAULT 0,
    feature_vector_sha256 text,
    feature_vector jsonb NOT NULL DEFAULT '{}'::jsonb,
    defaulted_features jsonb NOT NULL DEFAULT '[]'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (feature_schema_sha256 IS NULL OR feature_schema_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (feature_vector_sha256 IS NULL OR feature_vector_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (feature_count >= 0),
    CHECK (jsonb_typeof(feature_vector) IN ('object', 'array')),
    CHECK (jsonb_typeof(defaulted_features) = 'array'),
    CHECK (NOT features_complete OR
           (lower(build_status) = 'ok' AND feature_vector_sha256 IS NOT NULL)),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ml.prediction_observations (
    prediction_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_prediction_id text NOT NULL,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    match_uid text,
    feature_snapshot_id uuid REFERENCES ml.feature_snapshots(feature_snapshot_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_feature_snapshot_id text,
    predicted_at timestamptz,
    model_family text NOT NULL,
    model_version text NOT NULL,
    model_role text NOT NULL,
    player1_probability numeric,
    player2_probability numeric,
    logging_schema_version text,
    logging_quality text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (player1_probability IS NULL OR player1_probability BETWEEN 0 AND 1),
    CHECK (player2_probability IS NULL OR player2_probability BETWEEN 0 AND 1),
    CHECK ((player1_probability IS NULL) = (player2_probability IS NULL)),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.paper_accounts (
    paper_account_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    account_code text NOT NULL UNIQUE,
    display_name text NOT NULL,
    currency text NOT NULL DEFAULT 'USD',
    status text NOT NULL,
    starting_capital numeric,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (starting_capital IS NULL OR starting_capital >= 0),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.paper_sessions (
    paper_session_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_session_id text NOT NULL UNIQUE,
    account_code text NOT NULL REFERENCES ops.paper_accounts(account_code)
        DEFERRABLE INITIALLY DEFERRED,
    started_at timestamptz,
    completed_at timestamptz,
    initial_balance numeric,
    final_balance numeric,
    total_bets integer,
    total_staked numeric,
    total_profit_loss numeric,
    win_rate numeric,
    average_odds numeric,
    average_edge numeric,
    kelly_multiplier numeric,
    notes text,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at),
    CHECK (total_bets IS NULL OR total_bets >= 0),
    CHECK (total_staked IS NULL OR total_staked >= 0),
    CHECK (win_rate IS NULL OR win_rate BETWEEN 0 AND 1),
    CHECK (average_odds IS NULL OR average_odds > 1),
    CHECK (kelly_multiplier IS NULL OR kelly_multiplier >= 0),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.account_ledger (
    account_ledger_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    account_code text NOT NULL REFERENCES ops.paper_accounts(account_code)
        DEFERRABLE INITIALLY DEFERRED,
    external_session_id text,
    occurred_at timestamptz,
    amount numeric,
    balance_after numeric,
    reason text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.bet_recommendations (
    bet_recommendation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_bet_id text NOT NULL UNIQUE,
    account_code text NOT NULL REFERENCES ops.paper_accounts(account_code)
        DEFERRABLE INITIALLY DEFERRED,
    external_session_id text,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    match_uid text,
    feature_snapshot_id uuid REFERENCES ml.feature_snapshots(feature_snapshot_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_feature_snapshot_id text,
    recommended_at timestamptz,
    bet_side text NOT NULL,
    bet_on_player1 boolean,
    decimal_odds numeric,
    stake numeric,
    stake_fraction numeric,
    model_probability numeric,
    market_probability numeric,
    edge numeric,
    kelly_fraction numeric,
    model_version text,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (decimal_odds IS NULL OR decimal_odds > 1),
    CHECK (stake IS NULL OR stake >= 0),
    CHECK (stake_fraction IS NULL OR stake_fraction BETWEEN 0 AND 1),
    CHECK (model_probability IS NULL OR model_probability BETWEEN 0 AND 1),
    CHECK (market_probability IS NULL OR market_probability BETWEEN 0 AND 1),
    CHECK (kelly_fraction IS NULL OR kelly_fraction >= 0),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.bet_state_events (
    bet_state_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    bet_recommendation_id uuid REFERENCES ops.bet_recommendations(bet_recommendation_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_bet_id text NOT NULL REFERENCES ops.bet_recommendations(external_bet_id)
        DEFERRABLE INITIALLY DEFERRED,
    occurred_at timestamptz,
    state text NOT NULL,
    outcome text,
    actual_profit numeric,
    balance_after numeric,
    notes text,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.settlement_attempts (
    settlement_attempt_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_attempt_id text NOT NULL,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    match_uid text,
    attempted_at timestamptz,
    dry_run boolean NOT NULL DEFAULT false,
    outcome_code text NOT NULL,
    outcome_detail text,
    confidence_score numeric,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (jsonb_typeof(evidence) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.settlement_events (
    settlement_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    match_uid text NOT NULL,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    settled_at timestamptz,
    result_status text NOT NULL,
    actual_winner smallint,
    score text,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (actual_winner IS NULL OR actual_winner IN (1, 2)),
    CHECK (jsonb_typeof(evidence) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS ops.skip_events (
    skip_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key text NOT NULL UNIQUE,
    external_skip_event_id text NOT NULL,
    run_id uuid REFERENCES ops.pipeline_runs(run_id)
        DEFERRABLE INITIALLY DEFERRED,
    external_run_id text,
    match_uid text,
    external_feature_snapshot_id text,
    external_prediction_id text,
    skipped_at timestamptz,
    stage_name text NOT NULL,
    reason_code text NOT NULL,
    reason_detail text,
    context jsonb NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id uuid REFERENCES ops.import_batches(batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    source_file text,
    source_row_number bigint,
    source_row_sha256 text,
    source_row_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (jsonb_typeof(context) = 'object'),
    CHECK (source_row_number IS NULL OR source_row_number >= 0),
    CHECK (source_row_sha256 IS NULL OR source_row_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS pipeline_runs_started_idx
    ON ops.pipeline_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS pipeline_run_stages_run_idx
    ON ops.pipeline_run_stages (run_id, started_at, stage_name);
CREATE INDEX IF NOT EXISTS source_fetches_source_started_idx
    ON raw.source_fetches (source_name, fetch_kind, started_at DESC);
CREATE INDEX IF NOT EXISTS source_artifacts_fetch_idx
    ON raw.source_artifacts (source_fetch_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS odds_match_observed_idx
    ON ops.odds_observations (match_uid, observed_at DESC);
CREATE INDEX IF NOT EXISTS odds_anchor_observed_idx
    ON ops.odds_observations (match_anchor_key, observed_at DESC);
CREATE INDEX IF NOT EXISTS match_metadata_anchor_observed_idx
    ON ops.match_metadata_observations (match_anchor_key, observed_at DESC);
CREATE INDEX IF NOT EXISTS match_metadata_uid_observed_idx
    ON ops.match_metadata_observations (match_uid, observed_at DESC);
CREATE INDEX IF NOT EXISTS feature_snapshots_match_captured_idx
    ON ml.feature_snapshots (match_uid, captured_at DESC);
CREATE INDEX IF NOT EXISTS prediction_observations_match_predicted_idx
    ON ml.prediction_observations (match_uid, model_family, model_version, predicted_at DESC);
CREATE INDEX IF NOT EXISTS paper_sessions_started_idx
    ON ops.paper_sessions (account_code, started_at DESC);
CREATE INDEX IF NOT EXISTS account_ledger_account_time_idx
    ON ops.account_ledger (account_code, occurred_at DESC);
CREATE INDEX IF NOT EXISTS bet_recommendations_match_idx
    ON ops.bet_recommendations (match_uid, recommended_at DESC);
CREATE INDEX IF NOT EXISTS bet_state_events_bet_time_idx
    ON ops.bet_state_events (external_bet_id, occurred_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS settlement_attempts_match_time_idx
    ON ops.settlement_attempts (match_uid, attempted_at DESC);
CREATE INDEX IF NOT EXISTS settlement_events_match_idx
    ON ops.settlement_events (match_uid, settled_at DESC);
CREATE INDEX IF NOT EXISTS skip_events_run_stage_idx
    ON ops.skip_events (external_run_id, stage_name, skipped_at DESC);

CREATE OR REPLACE FUNCTION ops.metadata_provenance_rank(source_name text)
RETURNS integer
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT CASE lower(coalesce(source_name, ''))
        WHEN 'official' THEN 400
        WHEN 'canonical_store' THEN 350
        WHEN 'tournament_registry' THEN 300
        WHEN 'bovada' THEN 200
        WHEN 'inferred' THEN 100
        WHEN 'default' THEN 0
        ELSE -1
    END
$$;

CREATE OR REPLACE VIEW api.current_match_metadata AS
SELECT
    match_anchor_key,
    (array_agg(match_uid ORDER BY observed_at DESC, created_at DESC))[1] AS match_uid,
    (array_agg(match_date ORDER BY
        ops.metadata_provenance_rank(field_provenance->>'match_date') DESC,
        observed_at DESC, created_at DESC)
        FILTER (WHERE match_date IS NOT NULL))[1] AS match_date,
    (array_agg(match_start_at_utc ORDER BY
        ops.metadata_provenance_rank(field_provenance->>'match_start_at_utc') DESC,
        observed_at DESC, created_at DESC)
        FILTER (WHERE match_start_at_utc IS NOT NULL))[1] AS match_start_at_utc,
    (array_agg(tournament ORDER BY
        ops.metadata_provenance_rank(field_provenance->>'tournament') DESC,
        observed_at DESC, created_at DESC)
        FILTER (WHERE tournament IS NOT NULL))[1] AS tournament,
    (array_agg(event_title ORDER BY
        ops.metadata_provenance_rank(field_provenance->>'event_title') DESC,
        observed_at DESC, created_at DESC)
        FILTER (WHERE event_title IS NOT NULL))[1] AS event_title,
    (array_agg(round_code ORDER BY
        ops.metadata_provenance_rank(field_provenance->>'round_code') DESC,
        observed_at DESC, created_at DESC)
        FILTER (WHERE round_code IS NOT NULL))[1] AS round_code,
    (array_agg(surface ORDER BY
        ops.metadata_provenance_rank(field_provenance->>'surface') DESC,
        observed_at DESC, created_at DESC)
        FILTER (WHERE surface IS NOT NULL))[1] AS surface,
    (array_agg(level ORDER BY
        ops.metadata_provenance_rank(field_provenance->>'level') DESC,
        observed_at DESC, created_at DESC)
        FILTER (WHERE level IS NOT NULL))[1] AS level,
    max(observed_at) AS last_observed_at,
    count(*) AS observation_count
FROM ops.match_metadata_observations
GROUP BY match_anchor_key;

CREATE OR REPLACE VIEW api.latest_pipeline_runs AS
SELECT
    run_id, external_run_id, run_kind, status, started_at, completed_at,
    extract(epoch FROM (coalesce(completed_at, now()) - started_at)) AS duration_seconds,
    error_message, metrics
FROM ops.pipeline_runs;

CREATE OR REPLACE VIEW api.source_freshness AS
SELECT DISTINCT ON (source_name, fetch_kind)
    source_name, fetch_kind, status AS latest_status, started_at AS latest_started_at,
    completed_at AS latest_completed_at, rows_observed, http_status, error_message
FROM raw.source_fetches
ORDER BY source_name, fetch_kind, started_at DESC NULLS LAST, created_at DESC;

CREATE OR REPLACE VIEW api.current_predictions AS
SELECT DISTINCT ON (match_uid, model_family, model_version)
    prediction_observation_id, external_prediction_id, match_uid,
    external_feature_snapshot_id, predicted_at, model_family, model_version,
    model_role, player1_probability, player2_probability,
    logging_schema_version, logging_quality, metadata
FROM ml.prediction_observations
ORDER BY match_uid, model_family, model_version, predicted_at DESC NULLS LAST,
         created_at DESC;

CREATE OR REPLACE VIEW api.current_bet_states AS
SELECT DISTINCT ON (external_bet_id)
    bet_state_event_id, external_bet_id, occurred_at, state, outcome,
    actual_profit, balance_after, notes
FROM ops.bet_state_events
ORDER BY external_bet_id, occurred_at DESC NULLS LAST, created_at DESC;

CREATE OR REPLACE VIEW api.current_settlements AS
SELECT DISTINCT ON (match_uid)
    settlement_event_id, match_uid, settled_at, result_status,
    actual_winner, score, evidence
FROM ops.settlement_events
ORDER BY match_uid, settled_at DESC NULLS LAST, created_at DESC;

CREATE OR REPLACE VIEW api.paper_account_state AS
WITH latest_states AS (
    SELECT * FROM api.current_bet_states
), account_facts AS (
    SELECT
        account.account_code,
        account.currency,
        coalesce(account.starting_capital, 0) AS starting_capital,
        coalesce(sum(state.actual_profit)
            FILTER (WHERE lower(state.state) IN ('settled', 'won', 'lost')), 0)
            AS realized_profit_loss,
        coalesce(sum(bet.stake)
            FILTER (WHERE lower(coalesce(state.state, 'pending')) = 'pending'), 0)
            AS pending_stake,
        count(*) FILTER (WHERE lower(coalesce(state.state, 'pending')) = 'pending')
            AS pending_bets
    FROM ops.paper_accounts account
    LEFT JOIN ops.bet_recommendations bet ON bet.account_code = account.account_code
    LEFT JOIN latest_states state ON state.external_bet_id = bet.external_bet_id
    GROUP BY account.account_code, account.currency, account.starting_capital
)
SELECT
    account_code,
    currency,
    starting_capital,
    realized_profit_loss,
    starting_capital + realized_profit_loss AS equity,
    pending_stake,
    greatest(0, starting_capital + realized_profit_loss - pending_stake)
        AS available_capital,
    pending_bets
FROM account_facts;

-- Private by default. A later reviewed dashboard migration may grant SELECT on
-- specific api views to a dedicated read-only role; no browser role receives
-- raw/ops/ml access here.
REVOKE ALL ON ALL TABLES IN SCHEMA raw, ops, ml, api FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.metadata_provenance_rank(text) FROM PUBLIC;

DO $rls$
DECLARE
    relation regclass;
BEGIN
    FOREACH relation IN ARRAY ARRAY[
        'ops.import_batches'::regclass,
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
        EXECUTE format('ALTER TABLE %s ENABLE ROW LEVEL SECURITY', relation);
    END LOOP;
END
$rls$;

INSERT INTO ops.schema_versions (version, migration_name)
VALUES ('1.0.0', '20260714010000_operational_schema_v1.sql')
ON CONFLICT (version) DO NOTHING;

COMMIT;
