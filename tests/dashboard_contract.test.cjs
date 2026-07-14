"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const client = fs.readFileSync(path.join(ROOT, "docs", "dashboard.js"), "utf8");
const html = fs.readFileSync(path.join(ROOT, "docs", "index.html"), "utf8");
const deployedVersion = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "dashboard_version.json"), "utf8"));

test("dashboard loads one manifest-pinned generation and retries a moving manifest", () => {
  assert.match(client, /fetchOne\("dash_sync_manifest", "published_at\.desc,sync_id\.desc"\)/);
  assert.match(client, /if \(confirmedSyncId !== syncId\)/);
  assert.match(client, /retryCount === 0/);
  assert.match(client, /const generationFilter = \{ sync_id: syncId \}/);
  assert.match(client, /const acceptedRunId = Logic\.clean\(manifest && manifest\.accepted_prediction_run_id\)/);
  assert.match(client, /const currentRunFilter = acceptedRunId \? \{ sync_id: syncId, run_id: acceptedRunId \} : null/);
  assert.match(client, /latestAttemptRunId: Logic\.clean\(store\.manifest && store\.manifest\.latest_attempt_run_id\)/);
  assert.match(client, /params\.sync_id = `eq\.\$\{syncId\}`/);
});

test("deployed build changes self-heal already-open tabs", () => {
  const buildMatch = client.match(/const BUILD_ID = "([^"]+)"/);
  assert.ok(buildMatch, "client build ID must exist");
  assert.equal(buildMatch[1], deployedVersion.build_id);
  for (const asset of ["dashboard.css", "dashboard_logic.js", "dashboard.js"]) {
    assert.match(html, new RegExp(`${asset.replace(".", "\\.")}\\?v=${deployedVersion.build_id.replaceAll(".", "\\.")}`));
  }
  assert.match(client, /dashboard_version\.json\?ts=\$\{Date\.now\(\)\}/);
  assert.match(client, /cache: "no-store"/);
  assert.match(client, /window\.location\.replace\(target\.toString\(\)\)/);
  assert.match(client, /window\.setInterval\(checkForDeployment, 60000\)/);
  assert.match(html, /id="dashboard-build"/);
});

test("deployment-version checks are allowed by the dashboard CSP", () => {
  assert.match(
    html,
    /connect-src 'self' https:\/\/nwcayyusigznreygjlxl\.supabase\.co/,
  );
});

test("current slate is snapshot and skipped-audit based, never canonical-latest based", () => {
  assert.match(client, /fetchAll\("dash_snapshots"[^\n]+currentRunFilter\)/);
  assert.match(client, /fetchAll\("dash_skipped_live_matches"[^\n]+currentRunFilter\)/);
  assert.match(client, /store\.snapshots/);
  assert.match(client, /store\.skipped/);
  assert.match(client, /fetchAll\("dash_features", "feature_snapshot_id,run_id,sync_id,build_status,features_complete,feature_schema_sha256,feature_vector_sha256,feature_count"[^\n]+currentRunFilter\)/);
  assert.match(client, /Logic\.featureReferenceStatus\(row, acceptedFeatureIds, featureReferenceLoaded, seenFeatureIds\)/);
  assert.match(client, /Feature snapshot ID is not present in the published immutable feature store/);
  assert.match(client, /Logic\.numberOrNull\(row\.feature_count\) === 141/);
  assert.match(client, /Feature row exists but failed status, completeness, hash, or 141-feature integrity checks/);
  assert.doesNotMatch(client, /store\.predictions\s*\n\s*\.filter\(\(row\) => predictionRunId\(row\) === selectedRun\.id/);
});

test("operations UI exposes accepted generation, capital gate, and settlement backlog", () => {
  for (const field of ["account_equity", "pending_exposure", "available_bankroll", "exposure_gate_status"]) {
    assert.match(client, new RegExp(`"${field}"`));
  }
  assert.match(client, /pendingBetDiagnostics\(\)/);
  assert.match(client, /past the settlement SLA/);
  assert.match(html, /id="accepted-generation"/);
  assert.match(html, /id="account-state"/);
  assert.match(html, /id="metric-overdue"/);
  assert.match(html, /Capital \/ exposure/);
});

test("slate wording separates valid decision inputs from available capital", () => {
  assert.match(html, /Data-valid now/);
  assert.match(html, /Decision inputs valid/);
  assert.doesNotMatch(html, /Eligible for paper decision/);
  assert.match(client, /data valid · capital blocked/);
  assert.match(client, /function statusChip\(status, extraClass, displayLabel\)/);
  assert.match(client, /statusChip\(state, state, stateLabel\)/);
});

test("manifest counts cover every published operational projection", () => {
  for (const table of [
    "dash_predictions", "dash_odds_history", "dash_shadow", "dash_runs", "dash_bets",
    "dash_snapshots", "dash_skipped_live_matches", "dash_settlement_audit", "dash_features",
    "dash_bankroll", "dash_sessions", "dash_model_metrics",
  ]) {
    assert.match(client, new RegExp(`${table}:`));
  }
  assert.match(client, /Logic\.compareManifestCounts\(expected, actual\)/);
  assert.match(client, /generationCounts\.ok \? buildCurrentSlate\(\)/);
});

test("performance UI consumes ledger rows without client metric math", () => {
  assert.match(client, /fetchAll\("dash_model_metrics"[^\n]+generationFilter\)/);
  assert.doesNotMatch(client, /Math\.log|expectedCalibrationError|scoreDiagnosticCohort|buildCommonCohort/);
  assert.match(client, /formatMoney\(metric\.max_drawdown_kelly\)/);
  assert.match(client, /startsWith\("shadow_"\)/);
  for (const tier of ["gold_intersection", "complete_intersection", "gold", "complete"]) {
    assert.match(html, new RegExp(`<option value="${tier}"`));
  }
  const performanceHead = html.match(/<table id="performance-table">[\s\S]*?<thead><tr>([\s\S]*?)<\/tr><\/thead>/);
  assert.ok(performanceHead, "performance table header must exist");
  assert.equal((performanceHead[1].match(/<th\b/g) || []).length, 13);
});
