(function () {
  "use strict";

  const Logic = window.DashboardLogic;
  if (!Logic) throw new Error("dashboard_logic.js did not load");

  const API_ROOT = "https://nwcayyusigznreygjlxl.supabase.co/rest/v1";
  const BUILD_ID = "2026-07-14.3";
  // Supabase publishable keys are intentionally public. RLS must remain read-only.
  const API_KEY = "sb_publishable_3GMmWx4Zws9G_tCbU5faXw_X_0SdrHq";
  const PAGE_SIZE = 1000;
  const REQUEST_TIMEOUT_MS = 18000;
  const SETTLEMENT_SLA_HOURS = 18;
  const CONSERVATIVE_UNZONED_PENDING_HOURS = 72;

  const PREDICTION_COLUMNS = [
    "match_uid", "run_id", "latest_run_id", "logged_at", "latest_logged_at",
    "match_date", "latest_match_date", "match_start_time", "latest_match_start_time",
    "tournament", "surface", "level", "round", "p1", "p2", "p1_rank", "p2_rank",
    "p1_hand", "p2_hand", "p1_odds_decimal", "p2_odds_decimal", "market_p1_prob",
    "market_p2_prob", "model_p1_prob", "model_p2_prob", "xgb_p1_prob", "xgb_p2_prob",
    "rf_p1_prob", "rf_p2_prob", "features_complete", "defaulted_features", "actual_winner",
    "score", "settled_at", "record_status", "logging_quality", "rescore_quality",
    "model_version", "nn_model_version", "xgb_model_version", "rf_model_version",
    "latest_model_version_seen", "latest_nn_model_version_seen", "latest_xgb_model_version_seen",
    "latest_rf_model_version_seen", "feature_snapshot_id", "latest_feature_snapshot_id",
    "prediction_uid", "latest_prediction_uid", "odds_scraped_at", "latest_odds_scraped_at",
    "match_start_at_utc", "latest_match_start_at_utc",
  ].join(",");

  const RUN_COLUMNS = [
    "run_id", "run_kind", "started_at", "completed_at", "status", "odds_rows_fetched",
    "odds_rows_candidate", "feature_rows_total", "feature_rows_ok", "feature_rows_skipped",
    "prediction_rows_total", "prediction_rows_success", "prediction_rows_error",
    "bet_opportunities", "bets_logged", "settlement_candidates", "settlement_newly_settled",
    "auto_settle_status", "auto_settle_error", "canonical_ingest_status",
    "canonical_ingest_rows", "canonical_ingest_error", "reconcile_status",
    "reconcile_error", "account_equity", "pending_exposure", "available_bankroll",
    "exposure_gate_status", "error_message",
  ].join(",");

  const BET_COLUMNS = [
    "bet_id", "timestamp", "event", "match", "match_uid", "feature_snapshot_id", "run_id",
    "bet_on", "odds_decimal", "stake", "model_prob", "market_prob", "edge", "kelly_fraction",
    "status", "outcome", "actual_profit", "bankroll_before", "bankroll_after",
    "settled_timestamp", "model_version", "match_date", "match_start_time",
  ].join(",");

  // Current-run tables are small after run_id filtering. Selecting all columns
  // lets the UI tolerate additive logger migrations without inventing fallbacks.
  const SNAPSHOT_COLUMNS = "*";
  const SKIPPED_COLUMNS = "*";

  const MODEL_METRIC_COLUMNS = [
    "model", "tier", "n", "accuracy", "auc", "log_loss", "brier", "ece",
    "cal_slope", "cal_intercept", "roi_flat", "n_bets_flat", "pnl_flat",
    "win_rate_flat", "roi_kelly", "n_bets_kelly", "pnl_kelly",
    "max_drawdown_kelly", "generated_at", "metric_source", "dashboard_row_key",
    "sync_id",
  ].join(",");

  const store = {
    predictions: [],
    snapshots: [],
    skipped: [],
    acceptedFeatures: { syncId: "", runId: "", ids: [], seenIds: [] },
    runs: [],
    bets: [],
    metrics: [],
    meta: {
      odds: null,
      shadows: null,
      features: null,
      snapshots_total: null,
      skipped_total: null,
      settlement: null,
      bankroll: null,
      sessions: null,
    },
    manifest: null,
    loadedSyncId: "",
    errors: {},
    browserFetchedAt: null,
  };

  let health = null;
  let generationCounts = { ok: false, issues: ["generation counts not loaded"], expected: {}, actual: {} };
  let currentSlate = { eligible: [], blocked: [], expired: [] };
  let resultsLimit = 100;

  const byId = (id) => document.getElementById(id);

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function setText(id, value) {
    const target = byId(id);
    if (target) target.textContent = value === null || value === undefined ? "—" : String(value);
  }

  async function checkForDeployment() {
    try {
      const response = await fetch(`dashboard_version.json?ts=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(String(response.status));
      const payload = await response.json();
      const deployedBuildId = Logic.clean(payload.build_id);
      if (!deployedBuildId) throw new Error("build ID missing");
      setText("dashboard-build", `Build ${BUILD_ID}`);
      if (deployedBuildId !== BUILD_ID) {
        const target = new URL(window.location.href);
        if (target.searchParams.get("build") === deployedBuildId) {
          setText("dashboard-build", `Build ${BUILD_ID} · deploy ${deployedBuildId} pending; refresh once`);
          return;
        }
        target.searchParams.set("build", deployedBuildId);
        window.location.replace(target.toString());
      }
    } catch (_) {
      setText("dashboard-build", `Build ${BUILD_ID} · update check unavailable`);
    }
  }

  function clear(node) {
    if (node) node.replaceChildren();
  }

  function normalizedPrediction(row) {
    return {
      ...row,
      match_date: Logic.clean(row.latest_match_date) || row.match_date,
      match_start_time: Logic.clean(row.latest_match_start_time) || row.match_start_time,
      match_start_at_utc: Logic.clean(row.latest_match_start_at_utc) || row.match_start_at_utc,
      latest_logged_at: Logic.clean(row.latest_logged_at) || row.logged_at,
      latest_odds_scraped_at: Logic.clean(row.latest_odds_scraped_at) || row.odds_scraped_at,
    };
  }

  function formatNumber(value) {
    const number = Logic.numberOrNull(value);
    return number === null ? "—" : new Intl.NumberFormat().format(number);
  }

  function formatPercent(value, digits = 1) {
    const number = Logic.numberOrNull(value);
    return number === null ? "—" : `${(number * 100).toFixed(digits)}%`;
  }

  function formatPoints(value) {
    const number = Logic.numberOrNull(value);
    return number === null ? "—" : `${number >= 0 ? "+" : ""}${(number * 100).toFixed(1)} pt`;
  }

  function formatMoney(value, digits = 2) {
    const number = Logic.numberOrNull(value);
    if (number === null) return "—";
    return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", minimumFractionDigits: digits, maximumFractionDigits: digits }).format(number);
  }

  function formatDateTime(value) {
    const timestamp = typeof value === "number" ? value : Logic.parseTimestamp(value);
    if (timestamp === null) return "—";
    return new Date(timestamp).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  }

  function formatDate(value) {
    const text = Logic.clean(value);
    if (!text) return "—";
    const timestamp = Logic.parseTimestamp(text.length === 10 ? `${text}T12:00:00Z` : text);
    return timestamp === null ? text : new Date(timestamp).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
  }

  function formatAge(minutes) {
    if (minutes === null || minutes === undefined || !Number.isFinite(minutes)) return "age unknown";
    if (minutes < 2) return "just now";
    if (minutes < 60) return `${Math.round(minutes)}m ago`;
    if (minutes < 1440) return `${Math.round(minutes / 60)}h ago`;
    return `${Math.round(minutes / 1440)}d ago`;
  }

  function americanOdds(decimalValue) {
    const decimal = Logic.numberOrNull(decimalValue);
    if (decimal === null || decimal <= 1) return "—";
    return decimal >= 2 ? `+${Math.round((decimal - 1) * 100)}` : `−${Math.round(100 / (decimal - 1))}`;
  }

  function lastName(name) {
    const parts = Logic.clean(name).split(/\s+/).filter(Boolean);
    return parts[parts.length - 1] || "P1";
  }

  function statusClass(status) {
    const value = Logic.clean(status).toLowerCase();
    if (Logic.SUCCESS_STATUSES.has(value)) return "success";
    if (Logic.FAILURE_STATUSES.has(value)) return "failed";
    if (["running", "pending"].includes(value) || Logic.DEGRADED_STATUSES.has(value)) return "warning";
    return "neutral";
  }

  function statusChip(status, extraClass, displayLabel) {
    const value = Logic.clean(status) || "unknown";
    const label = Logic.clean(displayLabel) || value.replaceAll("_", " ");
    return element("span", `status-chip ${extraClass || statusClass(value)}`, label);
  }

  function emptyState(message) {
    return element("div", "empty-state", message);
  }

  async function apiFetch(path, options = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(`${API_ROOT}/${path}`, {
        ...options,
        headers: { apikey: API_KEY, ...(options.headers || {}) },
        signal: controller.signal,
      });
      if (!response.ok) {
        let detail = "";
        try {
          const payload = await response.json();
          detail = Logic.clean(payload.message || payload.details || payload.hint);
        } catch (_) {
          detail = "";
        }
        throw new Error(`${response.status}${detail ? `: ${detail}` : ""}`);
      }
      return response;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function queryPath(table, params) {
    return `${table}?${new URLSearchParams(params).toString()}`;
  }

  async function fetchAll(table, columns, order, filters = {}) {
    const rows = [];
    let from = 0;
    for (;;) {
      const params = { select: columns };
      if (order) params.order = order;
      Object.entries(filters).forEach(([key, value]) => { params[key] = `eq.${value}`; });
      const response = await apiFetch(queryPath(table, params), { headers: { Range: `${from}-${from + PAGE_SIZE - 1}` } });
      const page = await response.json();
      rows.push(...page);
      if (page.length < PAGE_SIZE) return rows;
      from += PAGE_SIZE;
    }
  }

  async function fetchTableMeta(table, columns, order, filters = {}) {
    const params = { select: columns, order, limit: "1" };
    Object.entries(filters).forEach(([key, value]) => { params[key] = `eq.${value}`; });
    const response = await apiFetch(queryPath(table, params), {
      headers: { Prefer: "count=exact", Range: "0-0" },
    });
    const rows = await response.json();
    const range = response.headers.get("content-range") || "";
    const totalText = range.split("/")[1];
    return { count: totalText && totalText !== "*" ? Number(totalText) : rows.length, latest: rows[0] || null };
  }

  async function fetchOne(table, order) {
    const params = { select: "*", limit: "1" };
    if (order) params.order = order;
    const response = await apiFetch(queryPath(table, params));
    const rows = await response.json();
    return rows[0] || null;
  }

  async function fetchFiltered(table, columns, filters, order) {
    const params = { select: columns, limit: "250" };
    const syncId = Logic.clean(store.loadedSyncId);
    if (!syncId) throw new Error("accepted dashboard sync is unavailable");
    params.sync_id = `eq.${syncId}`;
    Object.entries(filters || {}).forEach(([key, value]) => { params[key] = `eq.${value}`; });
    if (order) params.order = order;
    const response = await apiFetch(queryPath(table, params));
    return response.json();
  }

  async function refreshResource(name, loader, transform) {
    try {
      const value = await loader();
      store[name] = transform ? transform(value) : value;
      delete store.errors[name];
    } catch (error) {
      store.errors[name] = error && error.name === "AbortError" ? "request timed out" : Logic.clean(error.message) || "request failed";
    }
  }

  async function refreshMeta(name, loader) {
    try {
      store.meta[name] = await loader();
      delete store.errors[name];
    } catch (error) {
      store.errors[name] = error && error.name === "AbortError" ? "request timed out" : Logic.clean(error.message) || "request failed";
    }
  }

  async function loadGeneration(manifest, retryCount = 0) {
    const syncId = Logic.clean(manifest && manifest.sync_id);
    const acceptedRunId = Logic.clean(manifest && manifest.accepted_prediction_run_id);
    if (!syncId) throw new Error("latest sync manifest has no sync_id");
    store.manifest = manifest;
    const generationFilter = { sync_id: syncId };
    const currentRunFilter = acceptedRunId ? { sync_id: syncId, run_id: acceptedRunId } : null;

    await Promise.all([
      refreshResource("predictions", () => fetchAll("dash_predictions", PREDICTION_COLUMNS, "match_date.asc,p1.asc,p2.asc", generationFilter), (rows) => rows.map(normalizedPrediction)),
      refreshResource("snapshots", () => currentRunFilter ? fetchAll("dash_snapshots", SNAPSHOT_COLUMNS, "logged_at.desc.nullslast,prediction_uid.asc", currentRunFilter) : Promise.resolve([])),
      refreshResource("skipped", () => currentRunFilter ? fetchAll("dash_skipped_live_matches", SKIPPED_COLUMNS, "logged_at.desc.nullslast,skip_event_id.asc", currentRunFilter) : Promise.resolve([])),
      refreshResource(
        "acceptedFeatures",
        () => currentRunFilter ? fetchAll("dash_features", "feature_snapshot_id,run_id,sync_id,build_status,features_complete,feature_schema_sha256,feature_vector_sha256,feature_count", "feature_snapshot_id.asc", currentRunFilter) : Promise.resolve([]),
        (rows) => ({
          syncId,
          runId: acceptedRunId,
          seenIds: rows.map((row) => Logic.clean(row.feature_snapshot_id)).filter(Boolean),
          ids: rows.filter((row) => (
            Logic.clean(row.build_status).toLowerCase() === "ok"
            && Logic.isTrue(row.features_complete)
            && Boolean(Logic.clean(row.feature_schema_sha256))
            && Boolean(Logic.clean(row.feature_vector_sha256))
            && Logic.numberOrNull(row.feature_count) === 141
          )).map((row) => Logic.clean(row.feature_snapshot_id)).filter(Boolean),
        }),
      ),
      refreshResource("runs", () => fetchAll("dash_runs", RUN_COLUMNS, "started_at.desc.nullslast,run_id.desc", generationFilter)),
      refreshResource("bets", () => fetchAll("dash_bets", BET_COLUMNS, "timestamp.desc.nullslast,bet_id.desc", generationFilter)),
      refreshResource("metrics", () => fetchAll("dash_model_metrics", MODEL_METRIC_COLUMNS, "tier.asc,log_loss.asc.nullslast,model.asc", generationFilter)),
      refreshMeta("odds", () => fetchTableMeta("dash_odds_history", "logged_at,odds_scraped_at,run_id,match_uid", "logged_at.desc.nullslast", generationFilter)),
      refreshMeta("shadows", () => fetchTableMeta("dash_shadow", "logged_at,run_id,match_uid,model_version", "logged_at.desc.nullslast", generationFilter)),
      refreshMeta("features", () => fetchTableMeta("dash_features", "logged_at,run_id", "logged_at.desc.nullslast", generationFilter)),
      refreshMeta("snapshots_total", () => fetchTableMeta("dash_snapshots", "logged_at,run_id,prediction_uid", "logged_at.desc.nullslast", generationFilter)),
      refreshMeta("skipped_total", () => fetchTableMeta("dash_skipped_live_matches", "logged_at,run_id,skip_event_id", "logged_at.desc.nullslast", generationFilter)),
      refreshMeta("settlement", () => fetchTableMeta("dash_settlement_audit", "logged_at,run_id,settlement_event_id", "logged_at.desc.nullslast", generationFilter)),
      refreshMeta("bankroll", () => fetchTableMeta("dash_bankroll", "timestamp,session_id", "timestamp.desc.nullslast", generationFilter)),
      refreshMeta("sessions", () => fetchTableMeta("dash_sessions", "start_time,session_id", "start_time.desc.nullslast", generationFilter)),
    ]);

    const confirmedManifest = await fetchOne("dash_sync_manifest", "published_at.desc,sync_id.desc");
    const confirmedSyncId = Logic.clean(confirmedManifest && confirmedManifest.sync_id);
    if (!confirmedSyncId) throw new Error("manifest disappeared while dashboard data loaded");
    if (confirmedSyncId !== syncId) {
      if (retryCount === 0) return loadGeneration(confirmedManifest, 1);
      store.manifest = confirmedManifest;
      store.loadedSyncId = syncId;
      store.errors.generation = `manifest advanced twice while loading (${syncId} → ${confirmedSyncId})`;
      return;
    }

    store.manifest = confirmedManifest;
    store.loadedSyncId = syncId;
    delete store.errors.generation;
  }

  async function loadAll() {
    try {
      const manifest = await fetchOne("dash_sync_manifest", "published_at.desc,sync_id.desc");
      delete store.errors.manifest;
      await loadGeneration(manifest);
    } catch (error) {
      store.errors.manifest = error && error.name === "AbortError" ? "request timed out" : Logic.clean(error.message) || "request failed";
    }
    store.browserFetchedAt = Date.now();
    renderAll();
  }

  function predictionRunId(row) {
    return Logic.clean(row.latest_run_id || row.run_id);
  }

  function buildCurrentSlate() {
    const buckets = { eligible: [], blocked: [], expired: [] };
    const acceptedRunId = Logic.clean(store.manifest && store.manifest.accepted_prediction_run_id);
    if (!acceptedRunId || !store.loadedSyncId) return buckets;
    const featureReferenceLoaded = !store.errors.acceptedFeatures
      && store.acceptedFeatures.syncId === store.loadedSyncId
      && store.acceptedFeatures.runId === acceptedRunId;
    const acceptedFeatureIds = new Set(store.acceptedFeatures.ids || []);
    const seenFeatureIds = new Set(store.acceptedFeatures.seenIds || []);

    const byMatch = new Map();
    store.snapshots
      .filter((row) => Logic.clean(row.run_id) === acceptedRunId && !Logic.validWinner(row.actual_winner))
      .forEach((row, index) => {
        const key = Logic.clean(row.match_uid) || `${Logic.clean(row.p1)}|${Logic.clean(row.p2)}|${Logic.clean(row.match_date)}|${index}`;
        const at = Logic.parseTimestamp(row.logged_at) || 0;
        const previous = byMatch.get(key);
        if (!previous || at >= previous.at) byMatch.set(key, { row, at });
      });

    [...byMatch.values()].forEach(({ row }) => {
      let classification = Logic.classifySlateRow(row, Date.now());
      const featureReference = Logic.featureReferenceStatus(row, acceptedFeatureIds, featureReferenceLoaded, seenFeatureIds);
      if (!["expired", "settled"].includes(classification.state) && ["unverified", "not_found", "invalid"].includes(featureReference)) {
        const referenceReason = featureReference === "not_found"
          ? "Feature snapshot ID is not present in the published immutable feature store"
          : featureReference === "invalid"
            ? "Feature row exists but failed status, completeness, hash, or 141-feature integrity checks"
            : "Feature snapshot ID has not been referentially verified";
        classification = { ...classification, state: "blocked", reasons: [...classification.reasons, referenceReason] };
      }
      const entry = { row, classification, source: "snapshot", featureReference };
      if (classification.state === "eligible") buckets.eligible.push(entry);
      else if (classification.state === "blocked") buckets.blocked.push(entry);
      else if (classification.state === "expired") buckets.expired.push(entry);
    });

    const predictedMatchIds = new Set([...byMatch.keys()]);
    const skippedByEvent = new Map();
    store.skipped
      .filter((row) => Logic.clean(row.run_id) === acceptedRunId)
      .forEach((row, index) => {
        const matchKey = Logic.clean(row.match_uid) || `${Logic.clean(row.p1)}|${Logic.clean(row.p2)}|${Logic.clean(row.match_date)}`;
        if (predictedMatchIds.has(matchKey)) return;
        const eventKey = Logic.clean(row.skip_event_id) || `${matchKey}|${Logic.clean(row.stage)}|${index}`;
        skippedByEvent.set(eventKey, row);
      });
    [...skippedByEvent.values()].forEach((row) => {
      const classification = Logic.classifySkippedRow(row, Date.now());
      buckets[classification.state].push({ row, classification, source: "skipped", featureReference: "not_applicable" });
    });

    const sorter = (a, b) => {
      const aStart = a.classification.startAt === null ? Number.MAX_SAFE_INTEGER : a.classification.startAt;
      const bStart = b.classification.startAt === null ? Number.MAX_SAFE_INTEGER : b.classification.startAt;
      return aStart - bStart || Logic.clean(a.row.tournament).localeCompare(Logic.clean(b.row.tournament));
    };
    Object.values(buckets).forEach((rows) => rows.sort(sorter));
    return buckets;
  }

  function parseManifestCounts(manifest) {
    const raw = manifest && manifest.table_counts_json;
    if (raw && typeof raw === "object" && !Array.isArray(raw)) return raw;
    const text = Logic.clean(raw);
    if (!text) throw new Error("manifest table_counts_json is missing");
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("manifest table_counts_json is not an object");
    return parsed;
  }

  function validateGenerationCounts() {
    let expected;
    try {
      expected = parseManifestCounts(store.manifest);
    } catch (error) {
      return { ok: false, issues: [Logic.clean(error.message) || "manifest counts are invalid"], expected: {}, actual: {} };
    }

    const arrayCount = (name) => store.errors[name] ? null : store[name].length;
    const metaCount = (name) => store.errors[name] || !store.meta[name] ? null : Logic.numberOrNull(store.meta[name].count);
    const actual = {
      dash_predictions: arrayCount("predictions"),
      dash_odds_history: metaCount("odds"),
      dash_shadow: metaCount("shadows"),
      dash_runs: arrayCount("runs"),
      dash_bets: arrayCount("bets"),
      dash_snapshots: metaCount("snapshots_total"),
      dash_skipped_live_matches: metaCount("skipped_total"),
      dash_settlement_audit: metaCount("settlement"),
      dash_features: metaCount("features"),
      dash_bankroll: metaCount("bankroll"),
      dash_sessions: metaCount("sessions"),
      dash_model_metrics: arrayCount("metrics"),
    };
    const comparison = Logic.compareManifestCounts(expected, actual);
    return { ...comparison, expected, actual };
  }

  function renderHealth() {
    const effectiveErrors = { ...store.errors };
    const manifestStatus = Logic.clean(store.manifest && store.manifest.status).toLowerCase();
    if (store.manifest && manifestStatus !== "success") {
      effectiveErrors.manifest_status = `latest dashboard generation is ${manifestStatus || "unknown"}`;
    }
    if (store.manifest && !Logic.clean(store.manifest.accepted_prediction_run_id)) {
      effectiveErrors.accepted_prediction_run = "no successful snapshot-bearing prediction run has been accepted yet";
    }
    if (store.manifest && !Logic.clean(store.manifest.latest_attempt_run_id)) {
      effectiveErrors.latest_attempt_run = "no pipeline attempt is identified by the sync manifest";
    }
    const metricSyncIds = new Set(store.metrics.map((row) => Logic.clean(row.sync_id)).filter(Boolean));
    const manifestSyncId = Logic.clean(store.manifest && store.manifest.sync_id);
    if (metricSyncIds.size && manifestSyncId && (metricSyncIds.size !== 1 || !metricSyncIds.has(manifestSyncId))) {
      effectiveErrors.metrics_generation = "ledger metrics do not match the accepted dashboard generation";
    }
    generationCounts = validateGenerationCounts();
    if (!generationCounts.ok) effectiveErrors.generation_counts = generationCounts.issues.join("; ");
    health = Logic.computeHealth({
      runs: store.runs,
      predictions: store.predictions,
      errors: effectiveErrors,
      latestAttemptRunId: Logic.clean(store.manifest && store.manifest.latest_attempt_run_id),
      acceptedPredictionRunId: Logic.clean(store.manifest && store.manifest.accepted_prediction_run_id),
      now: Date.now(),
    });
    const pendingBacklog = pendingBetDiagnostics();
    if (pendingBacklog.overdue.length) {
      health.reasons.push(`${pendingBacklog.overdue.length} pending paper bets (${formatMoney(pendingBacklog.overdueExposure, 0)}) are past the settlement SLA.`);
      if (health.state === "healthy") health.state = "degraded";
    }
    if (pendingBacklog.unverified.length) {
      health.reasons.push(`${pendingBacklog.unverified.length} pending paper bets lack enough time lineage to classify their settlement SLA.`);
      if (health.state === "healthy") health.state = "degraded";
    }
    currentSlate = generationCounts.ok ? buildCurrentSlate() : { eligible: [], blocked: [], expired: [] };
    const banner = byId("health-banner");
    banner.className = `health-banner health-${health.state}`;

    const titles = {
      healthy: "Pipeline and dashboard projection are current",
      degraded: "Pipeline is running with a degraded signal",
      stale: "Pipeline state is stale",
      failed: "Pipeline health cannot be trusted",
    };
    setText("health-title", titles[health.state]);
    setText("health-message", health.reasons.length ? health.reasons.join(" ") : "Latest attempt and accepted predictions are inside the expected :17 / :47 cadence.");

    if (health.latestAttempt) {
      const run = health.latestAttempt.run;
      setText("latest-attempt", `${Logic.clean(run.run_id)} · ${Logic.clean(run.status) || "unknown"} · ${formatAge(health.attemptAgeMinutes)}`);
    } else setText("latest-attempt", "not visible");

    const acceptedRunId = Logic.clean(store.manifest && store.manifest.accepted_prediction_run_id);
    if (acceptedRunId && health.predictionRun) setText("last-prediction-run", `${health.predictionRun.id} · ${formatAge(health.predictionAgeMinutes)}`);
    else setText("last-prediction-run", "none accepted in manifest");

    const acceptedSyncId = Logic.clean(store.manifest && store.manifest.sync_id);
    const acceptedSyncStatus = Logic.clean(store.manifest && store.manifest.status) || "unknown";
    setText("accepted-generation", acceptedSyncId
      ? `${acceptedSyncId} · ${acceptedSyncStatus} · counts ${generationCounts.ok ? "verified" : "unverified"}`
      : "manifest unavailable");

    const nextRun = Logic.nextScheduledRun(Date.now());
    setText("next-run", nextRun ? `${nextRun.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })} · :17 / :47 schedule` : ":17 / :47 schedule");
    setText("page-fetch-time", store.browserFetchedAt ? `Browser fetched ${new Date(store.browserFetchedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : "Loading data…");
  }

  function pendingBetDiagnostics(now = Date.now()) {
    const predictionsByMatch = new Map();
    store.predictions.forEach((row) => {
      const uid = Logic.clean(row.match_uid);
      if (!uid) return;
      const previous = predictionsByMatch.get(uid);
      const at = Logic.parseTimestamp(row.latest_logged_at || row.logged_at) || 0;
      if (!previous || at >= previous.at) predictionsByMatch.set(uid, { row, at });
    });
    const groups = { active: [], overdue: [], unverified: [] };
    store.bets.forEach((bet) => {
      if (Logic.normalizeBetOutcome(bet) !== "pending") return;
      const prediction = predictionsByMatch.get(Logic.clean(bet.match_uid));
      const sla = Logic.pendingBetSlaStatus(
        bet,
        prediction && prediction.row,
        now,
        SETTLEMENT_SLA_HOURS,
        CONSERVATIVE_UNZONED_PENDING_HOURS,
      );
      groups[sla.state].push({ bet, sla });
    });
    const exposure = (rows) => rows.reduce((sum, item) => sum + (Logic.numberOrNull(item.bet.stake) || 0), 0);
    return {
      ...groups,
      activeExposure: exposure(groups.active),
      overdueExposure: exposure(groups.overdue),
      unverifiedExposure: exposure(groups.unverified),
    };
  }

  function renderHeadlines() {
    setText("metric-eligible", currentSlate.eligible.length);
    setText("metric-blocked", currentSlate.blocked.length);
    setText("metric-expired", currentSlate.expired.length);
    setText("metric-settled", store.predictions.filter((row) => Logic.validWinner(row.actual_winner)).length.toLocaleString());
    const pending = pendingBetDiagnostics();
    setText("metric-exposure", formatMoney(pending.activeExposure, 0));
    setText("metric-overdue", `${pending.overdue.length} · ${formatMoney(pending.overdueExposure, 0)}`);
  }

  function addDefinition(list, term, description) {
    const wrapper = element("div");
    wrapper.append(element("dt", null, term), element("dd", null, description));
    list.appendChild(wrapper);
  }

  function runCount(run, numerator, denominator) {
    const top = Logic.numberOrNull(run && run[numerator]);
    const bottom = denominator ? Logic.numberOrNull(run && run[denominator]) : null;
    if (top === null && bottom === null) return "—";
    return bottom !== null ? `${formatNumber(top || 0)} / ${formatNumber(bottom)}` : formatNumber(top);
  }

  function renderOverview() {
    const run = health.latestAttempt && health.latestAttempt.run;
    const chipHost = byId("overview-run-status");
    chipHost.className = `status-chip ${run ? statusClass(run.status) : "neutral"}`;
    chipHost.textContent = run ? Logic.clean(run.status).replaceAll("_", " ") : "unavailable";

    const funnel = byId("run-funnel");
    clear(funnel);
    funnel.classList.remove("skeleton");
    const stages = [
      ["Odds", runCount(run, "odds_rows_fetched", "odds_rows_candidate")],
      ["Features", runCount(run, "feature_rows_ok", "feature_rows_total")],
      ["Skipped audit", store.errors.skipped ? "unavailable" : formatNumber(store.skipped.length)],
      ["Predictions", runCount(run, "prediction_rows_success", "prediction_rows_total")],
      ["Paper bets", runCount(run, "bets_logged", "bet_opportunities")],
      ["Settled", runCount(run, "settlement_newly_settled", "settlement_candidates")],
    ];
    stages.forEach(([label, value]) => {
      const step = element("div", "funnel-step");
      step.append(element("span", null, label), element("strong", null, value));
      funnel.appendChild(step);
    });

    const errorBox = byId("run-error");
    const errorMessage = run ? runErrorSummary(run).replace(/^—$/, "") : "";
    errorBox.textContent = errorMessage;
    errorBox.classList.toggle("hidden", !errorMessage);

    const freshness = byId("freshness-list");
    clear(freshness);
    addDefinition(freshness, "Latest attempt", health.latestAttempt ? formatAge(health.attemptAgeMinutes) : "unavailable");
    addDefinition(freshness, "Accepted predictions", Logic.clean(store.manifest && store.manifest.accepted_prediction_run_id) && health.predictionRun ? formatAge(health.predictionAgeMinutes) : "none accepted");
    const oddsAt = store.meta.odds && store.meta.odds.latest && (store.meta.odds.latest.odds_scraped_at || store.meta.odds.latest.logged_at);
    const oddsAge = oddsAt ? (Date.now() - Logic.parseTimestamp(oddsAt)) / 60000 : null;
    addDefinition(freshness, "Latest odds row", oddsAt ? formatAge(oddsAge) : "unavailable");
    const manifestAt = store.manifest && (store.manifest.published_at || store.manifest.synced_at || store.manifest.created_at);
    addDefinition(freshness, "Sync manifest", store.manifest ? (manifestAt ? formatDateTime(manifestAt) : "present; timestamp absent") : "unavailable / unverified");
    addDefinition(freshness, "Browser delivery", store.browserFetchedAt ? formatDateTime(store.browserFetchedAt) : "not loaded");

    const reasonCounts = new Map();
    currentSlate.blocked.forEach(({ classification }) => classification.reasons.forEach((reason) => reasonCounts.set(reason, (reasonCounts.get(reason) || 0) + 1)));
    const reasonHost = byId("blocked-reasons");
    clear(reasonHost);
    if (!reasonCounts.size) reasonHost.appendChild(emptyState("No blocked rows in the accepted slate."));
    else [...reasonCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 7).forEach(([reason, count]) => {
      const row = element("div", "reason-row");
      row.append(element("span", null, reason), element("strong", null, count));
      reasonHost.appendChild(row);
    });

    const pending = pendingBetDiagnostics();
    const account = byId("account-state");
    clear(account);
    addDefinition(account, "Account equity", formatMoney(run && run.account_equity));
    addDefinition(account, "Run-reported pending", formatMoney(run && run.pending_exposure));
    addDefinition(account, "Active exposure", `${formatMoney(pending.activeExposure)} · ${pending.active.length} bets`);
    addDefinition(account, "Overdue backlog", `${formatMoney(pending.overdueExposure)} · ${pending.overdue.length} bets`);
    addDefinition(account, "Time unverified", `${formatMoney(pending.unverifiedExposure)} · ${pending.unverified.length} bets`);
    addDefinition(account, "Available capital", formatMoney(run && run.available_bankroll));
    addDefinition(account, "Exposure gate", Logic.clean(run && run.exposure_gate_status).replaceAll("_", " ") || "unreported");
    addDefinition(account, "Settlement SLA", `${SETTLEMENT_SLA_HOURS}h after exact UTC start; ${CONSERVATIVE_UNZONED_PENDING_HOURS}h conservative fallback`);
  }

  function matchValueSummary(row) {
    const probability = Logic.numberOrNull(row.model_p1_prob);
    const p1Odds = Logic.numberOrNull(row.p1_odds_decimal);
    const p2Odds = Logic.numberOrNull(row.p2_odds_decimal);
    if (probability === null || !p1Odds || !p2Odds) return "No diagnostic value comparison";
    const edges = [
      { player: Logic.clean(row.p1), edge: probability - 1 / p1Odds },
      { player: Logic.clean(row.p2), edge: (1 - probability) - 1 / p2Odds },
    ].sort((a, b) => b.edge - a.edge);
    return `${edges[0].player || "Model side"}: ${formatPoints(edges[0].edge)} vs raw break-even`;
  }

  function addProbabilityCell(host, label, value) {
    const cell = element("div", "prob-cell");
    cell.append(element("span", null, label), element("strong", null, formatPercent(value)));
    host.appendChild(cell);
  }

  function addLineageValue(host, label, value) {
    const wrapper = element("div");
    wrapper.append(element("span", null, label), element("code", null, Logic.clean(value) || "—"));
    host.appendChild(wrapper);
  }

  function pendingBetFor(row) {
    const uid = Logic.clean(row.match_uid);
    return store.bets.find((bet) => uid && Logic.clean(bet.match_uid) === uid && Logic.normalizeBetOutcome(bet) === "pending") || null;
  }

  function renderMatchCard(entry, state) {
    const { row, classification, source, featureReference } = entry;
    const isSkipped = source === "skipped";
    const card = element("article", `match-card ${state}`);
    const header = element("div", "match-card-header");
    const titleWrap = element("div");
    titleWrap.append(
      element("h4", null, `${Logic.clean(row.p1) || "Unknown"} vs ${Logic.clean(row.p2) || "Unknown"}`),
      element("div", "match-meta", [Logic.clean(row.tournament) || "Event unknown", Logic.clean(row.round) || "round pending", Logic.clean(row.surface) || "surface pending", Logic.clean(row.level), isSkipped ? "skipped audit" : "immutable snapshot"].filter(Boolean).join(" · ")),
    );
    const time = element("div", "match-time");
    time.append(element("span", null, classification.startAt === null ? (Logic.clean(row.match_start_time) || "Time pending") : formatDateTime(classification.startAt)));
    if (classification.startAt !== null) {
      const minutes = (classification.startAt - Date.now()) / 60000;
      time.append(element("small", null, minutes > 0 ? `starts in ${formatAge(minutes).replace(" ago", "")}` : `started ${formatAge(Math.abs(minutes))}`));
    } else if (Logic.clean(row.match_start_time)) time.append(element("small", null, "feed-local time; UTC unavailable"));
    header.append(titleWrap, time);
    card.appendChild(header);

    if (!isSkipped) {
      const probabilities = element("div", "match-probs");
      addProbabilityCell(probabilities, `Market ${lastName(row.p1)}`, row.market_p1_prob);
      addProbabilityCell(probabilities, `NN ${lastName(row.p1)}`, row.model_p1_prob);
      addProbabilityCell(probabilities, `XGB ${lastName(row.p1)}`, row.xgb_p1_prob);
      addProbabilityCell(probabilities, `RF ${lastName(row.p1)}`, row.rf_p1_prob);
      card.appendChild(probabilities);
    }

    if (classification.reasons.length) {
      const list = element("ul", "match-reasons");
      classification.reasons.forEach((reason) => list.appendChild(element("li", null, reason)));
      card.appendChild(list);
    }

    const footer = element("div", "match-footer");
    const pendingBet = pendingBetFor(row);
    const summary = element("div");
    if (isSkipped) {
      summary.append(element("div", "match-meta", `Skipped before prediction · ${Logic.clean(row.stage).replaceAll("_", " ") || "pipeline stage unknown"}`));
      summary.append(element("div", "match-meta", Logic.clean(row.skip_reason_code).replaceAll("_", " ") || "reason unavailable"));
    } else {
      summary.append(element("div", "match-meta", `Line ${americanOdds(row.p1_odds_decimal)} / ${americanOdds(row.p2_odds_decimal)}`));
      summary.append(element("div", "match-meta", pendingBet ? `Paper bet pending: ${Logic.clean(pendingBet.bet_on)} · ${formatMoney(pendingBet.stake, 0)}` : matchValueSummary(row)));
    }
    const allocationGate = Logic.clean(
      health && health.latestAttempt && health.latestAttempt.run.exposure_gate_status,
    ).toLowerCase();
    const stateLabel = state === "eligible"
      ? (allocationGate.startsWith("blocked") ? "data valid · capital blocked" : "data valid")
      : state;
    footer.append(summary, statusChip(state, state, stateLabel));
    card.appendChild(footer);

    const details = element("details", "match-lineage");
    details.appendChild(element("summary", null, "Lineage and secondary models"));
    const grid = element("div", "lineage-grid");
    addLineageValue(grid, "match_uid", row.match_uid);
    addLineageValue(grid, isSkipped ? "skip_event_id" : "prediction_uid", isSkipped ? row.skip_event_id : row.prediction_uid);
    addLineageValue(grid, "feature_snapshot_id", Logic.exactFeatureId(row));
    const featureReferenceLabels = {
      verified: "verified in immutable feature store",
      not_found: "ID present; immutable feature row missing",
      invalid: "feature row failed integrity verification",
      unverified: "ID present; reference not verified",
      missing_id: "immutable feature ID missing",
      not_applicable: "not applicable; skipped before prediction",
    };
    addLineageValue(grid, "feature reference", featureReferenceLabels[featureReference] || "unverified");
    addLineageValue(grid, "accepted run", predictionRunId(row));
    addLineageValue(grid, "logged", row.logged_at);
    addLineageValue(grid, "odds scraped", row.odds_scraped_at);
    if (isSkipped) {
      addLineageValue(grid, "stage", row.stage);
      addLineageValue(grid, "resolver", row.resolver_source);
    } else {
      addLineageValue(grid, "NN version", row.nn_model_version || row.model_version);
      addLineageValue(grid, "XGB / RF", `${Logic.clean(row.xgb_model_version) || "—"} / ${Logic.clean(row.rf_model_version) || "—"}`);
    }
    details.appendChild(grid);

    const actions = element("div", "match-footer");
    const featureId = Logic.exactFeatureId(row);
    const featureVerified = featureReference === "verified";
    const featureButton = element("button", "lineage-button", featureVerified ? "Open verified feature vector" : featureId ? "Feature ID is not verified" : "Exact feature vector unavailable");
    featureButton.type = "button";
    featureButton.disabled = !featureVerified;
    if (featureVerified) featureButton.addEventListener("click", () => openFeatureSnapshot(row));
    const shadowButton = element("button", "secondary-button", "Load exact-run shadows");
    shadowButton.type = "button";
    const shadowHost = element("div");
    shadowButton.addEventListener("click", () => loadShadowsForMatch(row, shadowHost, shadowButton));
    actions.append(featureButton);
    if (!isSkipped) actions.append(shadowButton);
    details.append(actions, shadowHost);
    card.appendChild(details);
    return card;
  }

  function renderSlateBucket(hostId, entries, state, emptyMessage) {
    const host = byId(hostId);
    clear(host);
    if (!entries.length) host.appendChild(emptyState(emptyMessage));
    else entries.forEach((entry) => host.appendChild(renderMatchCard(entry, state)));
  }

  function renderSlate() {
    setText("eligible-count", currentSlate.eligible.length);
    setText("blocked-count", currentSlate.blocked.length);
    setText("expired-count", currentSlate.expired.length);
    const acceptedRunId = Logic.clean(store.manifest && store.manifest.accepted_prediction_run_id);
    setText("slate-cohort-note", acceptedRunId ? `Accepted prediction-bearing run ${acceptedRunId}. Cards come from immutable prediction snapshots plus skipped-live audit rows; canonical predictions are history-only.` : "No accepted prediction-bearing run is identified by the sync manifest.");
    renderSlateBucket("eligible-slate", currentSlate.eligible, "eligible", "No matches currently satisfy every eligibility precondition.");
    renderSlateBucket("blocked-slate", currentSlate.blocked, "blocked", "No blocked rows in the accepted slate.");
    renderSlateBucket("expired-slate", currentSlate.expired, "expired", "No started or expired rows in the accepted slate.");
  }

  async function loadShadowsForMatch(row, host, button) {
    button.disabled = true;
    button.textContent = "Loading…";
    clear(host);
    try {
      const uid = Logic.clean(row.match_uid);
      const runId = predictionRunId(row);
      if (!uid || !runId) throw new Error("immutable match/run lineage is missing");
      const rows = await fetchFiltered(
        "dash_shadow",
        "match_uid,run_id,model_version,model_family,shadow_p1_prob,logged_at,shadow_status,shadow_error,feature_snapshot_id",
        { match_uid: uid, run_id: runId },
        "logged_at.desc.nullslast",
      );
      if (!rows.length) {
        host.appendChild(emptyState("No shadow observations match this exact match_uid and run_id. Name/date fallback is intentionally disabled."));
        return;
      }
      const byVersion = new Map();
      rows.forEach((item) => { if (!byVersion.has(Logic.clean(item.model_version))) byVersion.set(Logic.clean(item.model_version), item); });
      const shell = element("div", "table-shell");
      const table = element("table");
      const caption = element("caption", null, "Shadow observations from the exact accepted run");
      const head = element("thead");
      const headRow = element("tr");
      ["Variant", `P(${lastName(row.p1)})`, "Status", "Feature snapshot"].forEach((label) => { const th = element("th", null, label); th.scope = "col"; headRow.appendChild(th); });
      head.appendChild(headRow);
      const body = element("tbody");
      [...byVersion.values()].forEach((item) => {
        const tr = element("tr");
        [Logic.clean(item.model_version) || Logic.clean(item.model_family), formatPercent(item.shadow_p1_prob), Logic.clean(item.shadow_status) || "unknown", Logic.clean(item.feature_snapshot_id) || "—"].forEach((value) => tr.appendChild(element("td", null, value)));
        body.appendChild(tr);
      });
      table.append(caption, head, body);
      shell.appendChild(table);
      host.appendChild(shell);
    } catch (error) {
      host.appendChild(emptyState(`Shadow lookup failed: ${Logic.clean(error.message)}`));
    } finally {
      button.disabled = false;
      button.textContent = "Reload exact-run shadows";
    }
  }

  function featureGroupName(key) {
    const text = key.toLowerCase();
    if (/rank|point|seed/.test(text)) return "Ranking";
    if (/surface|clay|grass|hard|carpet/.test(text)) return "Surface";
    if (/h2h/.test(text)) return "Head-to-head";
    if (/win|form|streak|matches|days|fatigue|set/.test(text)) return "Form and schedule";
    if (/serve|return|ace|double_fault|break_point|tiebreak/.test(text)) return "Serve and return";
    if (/age|height|hand|country/.test(text)) return "Player profile";
    return "Context";
  }

  async function openFeatureSnapshot(row) {
    const dialog = byId("feature-dialog");
    const body = byId("feature-dialog-body");
    const featureId = Logic.exactFeatureId(row);
    setText("feature-dialog-title", `Feature snapshot · ${featureId || "unavailable"}`);
    clear(body);
    body.appendChild(emptyState("Loading exact feature_snapshot_id…"));
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");

    try {
      const rows = await fetchFiltered(
        "dash_features",
        "feature_snapshot_id,match_uid,run_id,logged_at,features_complete,features_json",
        { feature_snapshot_id: featureId, run_id: predictionRunId(row) },
        "logged_at.desc.nullslast",
      );
      clear(body);
      if (!rows.length) {
        body.appendChild(emptyState("No feature row matches this exact feature_snapshot_id. Player/date fallback is intentionally disabled because it can show the wrong build."));
        return;
      }
      const featureRow = rows[0];
      let features;
      try { features = JSON.parse(featureRow.features_json || "{}"); }
      catch (_) { throw new Error("features_json is not valid JSON"); }

      const meta = element("div", "feature-meta");
      [["feature_snapshot_id", featureRow.feature_snapshot_id], ["match_uid", featureRow.match_uid], ["run_id", featureRow.run_id], ["logged_at", featureRow.logged_at], ["features_complete", featureRow.features_complete]].forEach(([label, value]) => {
        const item = element("div");
        item.append(element("span", null, label), element("code", null, Logic.clean(value) || "—"));
        meta.appendChild(item);
      });
      body.appendChild(meta);

      const groups = new Map();
      Object.entries(features).filter(([key]) => !key.startsWith("_")).forEach(([key, value]) => {
        const group = featureGroupName(key);
        if (!groups.has(group)) groups.set(group, []);
        groups.get(group).push([key, value]);
      });
      const groupHost = element("div", "feature-groups");
      [...groups.entries()].forEach(([groupName, items]) => {
        const group = element("section", "feature-group");
        group.appendChild(element("h3", null, `${groupName} · ${items.length}`));
        items.sort((a, b) => a[0].localeCompare(b[0])).forEach(([key, value]) => {
          const feature = element("div", "feature-row");
          feature.append(element("code", null, key), element("code", null, Logic.clean(value) || "0"));
          group.appendChild(feature);
        });
        groupHost.appendChild(group);
      });
      body.appendChild(groupHost);
    } catch (error) {
      clear(body);
      const message = Logic.clean(error.message);
      const schemaMissing = /feature_snapshot_id|column/i.test(message);
      body.appendChild(emptyState(schemaMissing
        ? "The public dash_features projection does not yet expose feature_snapshot_id. Exact lookup is unavailable until that lineage column is published; unsafe player/date fallback remains disabled."
        : `Exact feature lookup failed: ${message}`));
    }
  }

  function addCohortFact(host, label, value) {
    const wrapper = element("span");
    wrapper.append(document.createTextNode(`${label} `), element("strong", null, value));
    host.appendChild(wrapper);
  }

  function metricValue(value, digits = 3) {
    const number = Logic.numberOrNull(value);
    return number === null ? "—" : number.toFixed(digits);
  }

  function clearPerformanceTable(message) {
    const body = byId("performance-table").tBodies[0];
    clear(body);
    const row = element("tr");
    const cell = element("td", null, message);
    cell.colSpan = 13;
    row.appendChild(cell);
    body.appendChild(row);
  }

  function renderPerformance() {
    const tier = byId("cohort-select").value;
    const tierLabels = {
      gold_intersection: "GOLD core intersection",
      complete_intersection: "COMPLETE core intersection",
      gold: "GOLD per-model coverage",
      complete: "COMPLETE per-model coverage",
    };
    const selectedTierRows = store.metrics.filter((row) => Logic.clean(row.tier).toLowerCase() === tier);
    const shadowRowsWithheld = selectedTierRows.filter((row) => Logic.clean(row.model).toLowerCase().startsWith("shadow_"));
    const sourceRows = selectedTierRows
      .filter((row) => !Logic.clean(row.model).toLowerCase().startsWith("shadow_"))
      .sort((a, b) => {
        const aScore = Logic.numberOrNull(a.log_loss);
        const bScore = Logic.numberOrNull(b.log_loss);
        if (aScore === null && bScore !== null) return 1;
        if (aScore !== null && bScore === null) return -1;
        if (aScore !== null && bScore !== null && aScore !== bScore) return aScore - bScore;
        return Logic.clean(a.model).localeCompare(Logic.clean(b.model));
      });
    const allMetricSyncIds = new Set(store.metrics.map((row) => Logic.clean(row.sync_id)).filter(Boolean));
    const manifestSyncId = Logic.clean(store.manifest && store.manifest.sync_id);
    const generationMatches = Boolean(
      manifestSyncId && allMetricSyncIds.size === 1 && allMetricSyncIds.has(manifestSyncId),
    );
    const definition = byId("cohort-definition");
    clear(definition);
    addCohortFact(definition, "Cohort", tierLabels[tier] || tier);
    addCohortFact(definition, "Comparison", tier.endsWith("_intersection") ? "same match_uids across market + NN + XGB + RF" : "each model's own settled coverage");

    const state = byId("ledger-publication-state");
    if (!generationCounts.ok) {
      state.className = "notice failed";
      state.textContent = `Dashboard generation counts are not verified (${generationCounts.issues.join("; ")}). Ledger rows are withheld; no browser-calculated fallback is shown.`;
      addCohortFact(definition, "Count verification", "failed");
      clearPerformanceTable("Authoritative rows are withheld until every published table count matches the sync manifest.");
      return;
    }
    const metricsError = Logic.clean(store.errors.metrics);
    if (metricsError || !store.metrics.length) {
      state.className = "notice failed";
      state.textContent = `Authoritative ledger metrics unavailable${metricsError ? ` (${metricsError})` : ""}. No browser-calculated fallback is shown.`;
      addCohortFact(definition, "Source", "production.evaluation.ledger");
      addCohortFact(definition, "State", "unavailable");
      clearPerformanceTable("Ledger metrics unavailable. The dashboard will not substitute locally calculated scores.");
      return;
    }

    if (!generationMatches) {
      state.className = "notice failed";
      state.textContent = manifestSyncId
        ? "Ledger metrics do not match the accepted dashboard sync generation. The comparison is withheld to avoid mixing data eras."
        : "Ledger metrics loaded, but no accepted sync manifest is visible. The comparison is withheld because its generation cannot be verified.";
      addCohortFact(definition, "Metric sync", [...allMetricSyncIds].join(", ") || "missing");
      addCohortFact(definition, "Manifest sync", manifestSyncId || "unavailable");
      clearPerformanceTable("Authoritative rows are withheld until metrics and manifest belong to one accepted generation.");
      return;
    }

    if (!sourceRows.length) {
      state.className = "notice warning";
      state.textContent = `The accepted ledger generation contains no ${tierLabels[tier] || tier} rows. No browser-calculated fallback is shown.`;
      addCohortFact(definition, "Sync", manifestSyncId);
      addCohortFact(definition, "State", "no rows for selected cohort");
      clearPerformanceTable("No authoritative rows are available for this cohort.");
      return;
    }

    const cohortSizes = sourceRows.map((row) => Logic.numberOrNull(row.n)).filter((value) => value !== null);
    const minN = cohortSizes.length ? Math.min(...cohortSizes) : null;
    const maxN = cohortSizes.length ? Math.max(...cohortSizes) : null;
    const coverage = minN === null ? "—" : minN === maxN ? formatNumber(minN) : `${formatNumber(minN)}–${formatNumber(maxN)}`;
    const generatedAt = sourceRows.map((row) => Logic.parseTimestamp(row.generated_at)).filter((value) => value !== null).sort((a, b) => b - a)[0] || null;
    const metricSource = Logic.clean(sourceRows[0].metric_source) || "production.evaluation.ledger";
    addCohortFact(definition, "Models", sourceRows.length.toLocaleString());
    if (shadowRowsWithheld.length) addCohortFact(definition, "Shadow rows withheld", shadowRowsWithheld.length.toLocaleString());
    addCohortFact(definition, "Cohort n", coverage);
    addCohortFact(definition, "Generated", generatedAt === null ? "—" : formatDateTime(generatedAt));
    addCohortFact(definition, "Source", metricSource);
    addCohortFact(definition, "Sync", manifestSyncId);

    const manifestStatus = Logic.clean(store.manifest.status).toLowerCase();
    state.className = `notice ${manifestStatus === "success" ? "success" : "warning"}`;
    state.textContent = manifestStatus === "success"
      ? "Authoritative ledger metrics match the accepted dashboard generation. Rows are ranked by log loss; lower is better."
      : `Metrics match a ${manifestStatus || "non-success"} dashboard generation. Review the System tab before interpreting them.`;

    const body = byId("performance-table").tBodies[0];
    clear(body);
    sourceRows.forEach((metric) => {
      const row = element("tr");
      [
        Logic.clean(metric.model) || "Unknown model",
        formatNumber(metric.n),
        metricValue(metric.log_loss, 4),
        metricValue(metric.brier, 4),
        metricValue(metric.ece, 4),
        metricValue(metric.cal_slope, 3),
        formatPercent(metric.accuracy),
        formatPercent(metric.auc),
        formatPercent(metric.roi_flat),
        formatNumber(metric.n_bets_flat),
        formatPercent(metric.roi_kelly),
        formatNumber(metric.n_bets_kelly),
        formatMoney(metric.max_drawdown_kelly),
      ].forEach((value) => row.appendChild(element("td", null, value)));
      body.appendChild(row);
    });
  }

  function renderMetricCard(host, label, value, subtext) {
    const card = element("article", "metric-card");
    card.append(element("span", null, label), element("strong", null, value), element("small", null, subtext));
    host.appendChild(card);
  }

  function renderBets() {
    const pendingDiagnostics = pendingBetDiagnostics();
    const pendingSlaByBet = new Map(
      [...pendingDiagnostics.active, ...pendingDiagnostics.overdue, ...pendingDiagnostics.unverified]
        .map((item) => [item.bet, item.sla]),
    );
    const categories = store.bets.map((bet) => ({
      bet,
      outcome: Logic.normalizeBetOutcome(bet),
      pendingSla: pendingSlaByBet.get(bet) || null,
    }));
    const wins = categories.filter((item) => item.outcome === "win");
    const losses = categories.filter((item) => item.outcome === "loss");
    const voided = categories.filter((item) => ["void", "cancelled"].includes(item.outcome));
    const decided = [...wins, ...losses];
    const realizedProfit = categories.filter((item) => ["win", "loss", "void", "cancelled"].includes(item.outcome)).reduce((sum, item) => sum + (Logic.numberOrNull(item.bet.actual_profit) || 0), 0);
    const decidedStake = decided.reduce((sum, item) => sum + (Logic.numberOrNull(item.bet.stake) || 0), 0);
    const roi = decidedStake ? realizedProfit / decidedStake : null;
    const metrics = byId("bet-metrics");
    clear(metrics);
    renderMetricCard(metrics, "Settled decisions", decided.length.toLocaleString(), `${wins.length} wins · ${losses.length} losses`);
    renderMetricCard(metrics, "Active pending", pendingDiagnostics.active.length.toLocaleString(), `${formatMoney(pendingDiagnostics.activeExposure, 0)} exposure`);
    renderMetricCard(metrics, "Overdue backlog", pendingDiagnostics.overdue.length.toLocaleString(), `${formatMoney(pendingDiagnostics.overdueExposure, 0)} past settlement SLA`);
    renderMetricCard(metrics, "Time unverified", pendingDiagnostics.unverified.length.toLocaleString(), `${formatMoney(pendingDiagnostics.unverifiedExposure, 0)} excluded from active`);
    renderMetricCard(metrics, "Void / cancelled", voided.length.toLocaleString(), "excluded from decision ROI");
    renderMetricCard(metrics, "Realized P&L", formatMoney(realizedProfit), "paper ledger actual_profit");
    renderMetricCard(metrics, "Realized ROI", roi === null ? "—" : formatPercent(roi), `${formatMoney(decidedStake, 0)} decided stake`);

    const body = byId("bets-table").tBodies[0];
    clear(body);
    categories
      .sort((a, b) => (Logic.parseTimestamp(b.bet.timestamp) || 0) - (Logic.parseTimestamp(a.bet.timestamp) || 0))
      .slice(0, 400)
      .forEach(({ bet, outcome, pendingSla }) => {
        const row = element("tr");
        row.appendChild(element("td", null, formatDateTime(bet.timestamp)));
        const matchCell = element("td");
        matchCell.append(element("span", "cell-primary", Logic.clean(bet.bet_on) || "Unknown side"), element("span", "cell-secondary", Logic.clean(bet.match || bet.event) || "Match unavailable"));
        row.appendChild(matchCell);
        row.appendChild(element("td", null, americanOdds(bet.odds_decimal)));
        row.appendChild(element("td", null, formatMoney(bet.stake, 0)));
        row.appendChild(element("td", null, formatPoints(bet.edge)));
        const statusCell = element("td");
        if (outcome === "pending" && pendingSla) {
          const pendingLabels = { active: "active pending", overdue: "overdue", unverified: "pending time unverified" };
          const pendingClasses = { active: "pending", overdue: "overdue", unverified: "warning" };
          statusCell.appendChild(statusChip(pendingLabels[pendingSla.state], pendingClasses[pendingSla.state]));
          statusCell.appendChild(element("span", "cell-secondary", pendingSla.deadline ? `deadline ${formatDateTime(pendingSla.deadline)} · ${pendingSla.basis}` : pendingSla.basis));
        } else statusCell.appendChild(statusChip(outcome, outcome));
        row.appendChild(statusCell);
        row.appendChild(element("td", outcome === "pending" ? "" : outcome === "settled_unknown" ? "unavailable" : formatMoney(bet.actual_profit)));
        body.appendChild(row);
      });
    if (!categories.length) {
      const row = element("tr");
      const cell = element("td", null, "No paper bet rows are available.");
      cell.colSpan = 7;
      row.appendChild(cell);
      body.appendChild(row);
    }
  }

  function resultsRows() {
    const query = Logic.clean(byId("results-search").value).toLowerCase();
    return store.predictions
      .filter((row) => Logic.validWinner(row.actual_winner))
      .filter((row) => !query || [row.p1, row.p2, row.tournament, row.round].some((value) => Logic.clean(value).toLowerCase().includes(query)))
      .sort((a, b) => (Logic.parseTimestamp(b.settled_at) || Logic.parseTimestamp(b.match_date) || 0) - (Logic.parseTimestamp(a.settled_at) || Logic.parseTimestamp(a.match_date) || 0));
  }

  function renderResults() {
    const allSettled = store.predictions.filter((row) => Logic.validWinner(row.actual_winner));
    const filtered = resultsRows();
    const shown = filtered.slice(0, resultsLimit);
    setText("results-summary", `Showing ${shown.length.toLocaleString()} of ${filtered.length.toLocaleString()} matching rows · ${allSettled.length.toLocaleString()} valid settled predictions total.`);
    const body = byId("results-table").tBodies[0];
    clear(body);
    shown.forEach((item) => {
      const row = element("tr");
      const winner = Logic.numberOrNull(item.actual_winner) === 1 ? Logic.clean(item.p1) : Logic.clean(item.p2);
      row.appendChild(element("td", null, formatDate(item.match_date)));
      const event = element("td");
      event.append(element("span", "cell-primary", Logic.clean(item.tournament) || "Unknown event"), element("span", "cell-secondary", [Logic.clean(item.round), Logic.clean(item.surface)].filter(Boolean).join(" · ")));
      row.appendChild(event);
      row.appendChild(element("td", "cell-primary", `${Logic.clean(item.p1)} vs ${Logic.clean(item.p2)}`));
      row.appendChild(element("td", "numeric-good", winner || "—"));
      row.appendChild(element("td", null, Logic.clean(item.score) || "—"));
      row.appendChild(element("td", null, formatPercent(item.model_p1_prob)));
      row.appendChild(element("td", null, formatPercent(item.xgb_p1_prob)));
      row.appendChild(element("td", null, formatPercent(item.market_p1_prob)));
      body.appendChild(row);
    });
    if (!shown.length) {
      const row = element("tr");
      const cell = element("td", null, "No settled results match this search.");
      cell.colSpan = 8;
      row.appendChild(cell);
      body.appendChild(row);
    }
    byId("results-more").hidden = shown.length >= filtered.length;
  }

  function appendRunCell(row, value) {
    row.appendChild(element("td", null, value));
  }

  function runStageSummary(run) {
    const stages = [
      ["settle", run.auto_settle_status, null],
      ["ingest", run.canonical_ingest_status, Logic.numberOrNull(run.canonical_ingest_rows)],
      ["reconcile", run.reconcile_status, null],
    ];
    return stages.map(([label, status, rows]) => {
      const normalized = Logic.clean(status) || "unreported";
      return `${label}: ${normalized.replaceAll("_", " ")}${rows === null ? "" : ` (${formatNumber(rows)} rows)`}`;
    }).join(" · ");
  }

  function runAccountSummary(run) {
    const equity = formatMoney(run.account_equity, 0);
    const pending = formatMoney(run.pending_exposure, 0);
    const available = formatMoney(run.available_bankroll, 0);
    const gate = Logic.clean(run.exposure_gate_status).replaceAll("_", " ") || "unreported";
    return `equity ${equity} · pending ${pending} · available ${available} · gate ${gate}`;
  }

  function runErrorSummary(run) {
    return [run.error_message, run.auto_settle_error, run.canonical_ingest_error, run.reconcile_error]
      .map(Logic.clean)
      .filter(Boolean)
      .filter((value, index, values) => values.indexOf(value) === index)
      .join(" · ") || "—";
  }

  function renderSystem() {
    const loadState = byId("load-state");
    const errors = Object.entries(store.errors);
    const integrityIssues = generationCounts.ok ? [] : generationCounts.issues;
    loadState.className = `notice ${errors.length || integrityIssues.length ? "failed" : "success"}`;
    loadState.textContent = errors.length || integrityIssues.length
      ? `Dashboard generation is not accepted. Last-known-good resource values were retained where available. ${errors.map(([name, message]) => `${name}: ${message}`).concat(integrityIssues.map((issue) => `count: ${issue}`)).join(" · ")}`
      : `All requested public projections match manifest ${Logic.clean(store.manifest && store.manifest.sync_id)} exactly.`;

    const body = byId("runs-table").tBodies[0];
    clear(body);
    store.runs
      .slice()
      .sort((a, b) => (Logic.runTimestamp(b) || 0) - (Logic.runTimestamp(a) || 0))
      .slice(0, 120)
      .forEach((run) => {
        const row = element("tr");
        appendRunCell(row, Logic.clean(run.run_id) || "—");
        appendRunCell(row, formatDateTime(run.started_at || Logic.parseRunId(run.run_id)));
        const statusCell = element("td");
        statusCell.appendChild(statusChip(run.status));
        row.appendChild(statusCell);
        appendRunCell(row, runCount(run, "odds_rows_fetched", "odds_rows_candidate"));
        appendRunCell(row, runCount(run, "feature_rows_ok", "feature_rows_total"));
        appendRunCell(row, runCount(run, "prediction_rows_success", "prediction_rows_total"));
        appendRunCell(row, runCount(run, "bets_logged", "bet_opportunities"));
        appendRunCell(row, runCount(run, "settlement_newly_settled", "settlement_candidates"));
        appendRunCell(row, runAccountSummary(run));
        appendRunCell(row, runStageSummary(run));
        appendRunCell(row, runErrorSummary(run));
        body.appendChild(row);
      });

    const inventory = byId("data-inventory");
    clear(inventory);
    const pending = pendingBetDiagnostics();
    const cards = [
      ["Prediction history", store.predictions.length, "canonical settlement/history rows; never current-slate cards"],
      ["Current snapshots", store.snapshots.length, `${store.meta.snapshots_total ? `${formatNumber(store.meta.snapshots_total.count)} generation-total · ` : "total unverified · "}${Logic.clean(store.manifest && store.manifest.accepted_prediction_run_id) || "accepted prediction run unavailable"}`],
      ["Skipped-live audit", store.skipped.length, store.errors.skipped ? `unavailable: ${store.errors.skipped}` : `${store.meta.skipped_total ? `${formatNumber(store.meta.skipped_total.count)} generation-total · ` : "total unverified · "}included in blocked / expired slate`],
      ["Run audit", store.runs.length, health.latestAttempt ? Logic.clean(health.latestAttempt.run.status) : "latest attempt unavailable"],
      ["Paper bets", store.bets.length, `${pending.active.length} active · ${pending.overdue.length} overdue · ${pending.unverified.length} time unverified`],
      ["Odds observations", store.meta.odds && store.meta.odds.count, store.meta.odds && store.meta.odds.latest ? formatDateTime(store.meta.odds.latest.logged_at) : "unavailable"],
      ["Shadow observations", store.meta.shadows && store.meta.shadows.count, "secondary; loaded per match"],
      ["Feature vectors", store.meta.features && store.meta.features.count, "exact ID lookup only"],
      ["Accepted feature references", store.acceptedFeatures.ids.length, store.errors.acceptedFeatures ? `unverified: ${store.errors.acceptedFeatures}` : `${store.acceptedFeatures.seenIds.length} IDs present · verified status + complete + schema/vector hashes + 141 features · ${Logic.clean(store.acceptedFeatures.runId) || "no accepted run"}`],
      ["Settlement audit", store.meta.settlement && store.meta.settlement.count, "full generation count verified"],
      ["Bankroll history", store.meta.bankroll && store.meta.bankroll.count, "full generation count verified"],
      ["Betting sessions", store.meta.sessions && store.meta.sessions.count, "full generation count verified"],
      ["Dashboard build", 1, `${BUILD_ID} · auto-checks deployed version every minute`],
      ["Sync manifest", store.manifest ? 1 : 0, store.manifest ? `${Logic.clean(store.manifest.status) || "unknown"} · ${Logic.clean(store.manifest.sync_id) || "ID missing"}` : "generation unverified"],
    ];
    cards.forEach(([label, value, detail]) => {
      const card = element("article", "inventory-card");
      card.append(element("span", null, label), element("strong", null, value === null || value === undefined ? "—" : formatNumber(value)), element("small", null, detail));
      inventory.appendChild(card);
    });
  }

  function renderAll() {
    renderHealth();
    renderHeadlines();
    renderOverview();
    renderSlate();
    renderPerformance();
    renderBets();
    renderResults();
    renderSystem();
  }

  function selectTab(button, focus = true) {
    const buttons = [...document.querySelectorAll('[role="tab"]')];
    buttons.forEach((item) => {
      const selected = item === button;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      const panel = byId(item.getAttribute("aria-controls"));
      if (panel) panel.hidden = !selected;
    });
    if (focus) button.focus();
    const view = button.dataset.view;
    if (view) history.replaceState(null, "", `#${view}`);
  }

  function installTabs() {
    const tabList = byId("tabs");
    const buttons = [...tabList.querySelectorAll('[role="tab"]')];
    buttons.forEach((button) => button.addEventListener("click", () => selectTab(button, false)));
    tabList.addEventListener("keydown", (event) => {
      const current = document.activeElement;
      const index = buttons.indexOf(current);
      if (index < 0) return;
      let nextIndex = index;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
      else if (event.key === "ArrowLeft") nextIndex = (index - 1 + buttons.length) % buttons.length;
      else if (event.key === "Home") nextIndex = 0;
      else if (event.key === "End") nextIndex = buttons.length - 1;
      else return;
      event.preventDefault();
      selectTab(buttons[nextIndex]);
    });
    const initial = location.hash.replace("#", "");
    const initialButton = buttons.find((button) => button.dataset.view === initial);
    if (initialButton) selectTab(initialButton, false);
  }

  function installInteractions() {
    installTabs();
    byId("cohort-select").addEventListener("change", renderPerformance);
    byId("results-search").addEventListener("input", () => { resultsLimit = 100; renderResults(); });
    byId("results-more").addEventListener("click", () => { resultsLimit += 100; renderResults(); });
    const dialog = byId("feature-dialog");
    byId("feature-dialog-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  }

  function refreshTimeSensitiveViews() {
    renderHealth();
    renderHeadlines();
    renderOverview();
    renderSlate();
  }

  function init() {
    installInteractions();
    checkForDeployment();
    loadAll();
    window.setInterval(refreshTimeSensitiveViews, 60000);
    window.setInterval(checkForDeployment, 60000);
    window.setInterval(loadAll, 300000);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && (!store.browserFetchedAt || Date.now() - store.browserFetchedAt > 120000)) loadAll();
    });
  }

  init();
})();
