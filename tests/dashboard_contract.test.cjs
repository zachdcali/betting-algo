"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const client = fs.readFileSync(path.join(ROOT, "docs", "dashboard.js"), "utf8");
const logic = fs.readFileSync(path.join(ROOT, "docs", "dashboard_logic.js"), "utf8");
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
  assert.match(client, /return fetchAll\(table, columns, order, \{ sync_id: syncId/);
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
  assert.match(client, /fetchAll\("dash_features", "feature_snapshot_id,run_id,sync_id,build_status,features_complete,p1_hand,p2_hand,feature_schema_sha256,feature_vector_sha256,feature_count"[^\n]+currentRunFilter\)/);
  assert.match(client, /Logic\.isStructurallyValidFeatureRow\(row\)/);
  assert.match(client, /Logic\.hydrateSnapshotHands\(/);
  assert.match(client, /featureReferenceLoaded \? store\.acceptedFeatures\.profiles \|\| \[\] : \[\]/);
  assert.match(client, /featureProfilesById\.get\(Logic\.exactFeatureId\(row\)\)/);
  assert.match(client, /Logic\.featureReferenceStatus\(row, acceptedFeatureIds, featureReferenceLoaded, seenFeatureIds\)/);
  assert.match(client, /Feature snapshot ID is not present in the published immutable feature store/);
  assert.match(client, /Logic\.numberOrNull\(row\.feature_count\) === 141/);
  assert.match(client, /Feature row exists but failed status, completeness, hash, or 141-feature integrity checks/);
  assert.doesNotMatch(client, /store\.predictions\s*\n\s*\.filter\(\(row\) => predictionRunId\(row\) === selectedRun\.id/);
  assert.match(client, /Logic\.classifySlateEvidence\(row, auditRows, Date\.now\(\)\)/);
  assert.match(client, /auditRows/);
  assert.doesNotMatch(client, /if \(predictedMatchIds\.has\(matchKey\)\) return/);
});

test("accepted slate exposes a monotone eligibility funnel and primary blocker groups", () => {
  for (const label of ["Accepted matches", "Finite NN outputs", "Complete vectors", "Identity-clean"]) {
    assert.match(client, new RegExp(label));
  }
  assert.match(client, /Logic\.acceptedSlateFunnel\(/);
  assert.match(client, /every stage is an explicit subset of the previous stage/);
  assert.match(client, /Started \/ expired retained separately/);
  assert.match(client, /Logic\.primaryBlockerGroup\(entry\)/);
  assert.match(html, /id="slate-funnel"/);
  assert.match(html, /One primary group per blocked row/);
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
  assert.match(html, /scores below are current for the accepted dashboard sync shown on this page/);
  assert.match(html, /Open dated full ledger snapshot/);
  assert.match(html, /This report may lag the accepted sync shown above/);
  assert.match(html, /dated full ledger snapshot \(live metrics plus offline experiments\)/);
  assert.doesNotMatch(html, /Open generated Model Evaluation Ledger/);
  assert.match(html, /Current manifest-pinned model metrics for the selected accepted dashboard generation/);
  assert.match(client, /Current dashboard authority: ledger metrics match accepted sync \$\{manifestSyncId\}/);
  assert.match(client, /The dated Markdown report may lag this generation/);
  assert.match(client, /if \(metric === "max_drawdown_kelly"\) return formatMoney\(value\)/);
  assert.match(client, /metricCell\(\s*"max_drawdown_kelly", metric\.max_drawdown_kelly/);
  assert.match(client, /startsWith\("shadow_"\)/);
  for (const tier of ["gold_intersection", "complete_intersection", "gold", "complete"]) {
    assert.match(html, new RegExp(`<option value="${tier}"`));
  }
  const performanceHead = html.match(/<table id="performance-table">[\s\S]*?<thead><tr>([\s\S]*?)<\/tr><\/thead>/);
  assert.ok(performanceHead, "performance table header must exist");
  assert.equal((performanceHead[1].match(/<th\b/g) || []).length, 13);
});

test("performance UI separates prediction, counterfactual, and placed-bet populations", () => {
  assert.match(html, /id="performance-population-map"/);
  assert.match(html, /Parallel populations—not one sample/);
  for (const label of [
    "Settled prediction rows",
    "Selected model cohort",
    "NN counterfactual bets",
    "Placed bets · exact",
    "Placed bets · accounting only",
    "Placed bets · legacy unknown",
  ]) {
    assert.match(client, new RegExp(label));
  }
  for (const field of ["settlement_quality", "attribution_quality", "metric_eligible"]) {
    assert.match(client, new RegExp(`"${field}"`));
  }
  assert.match(client, /Logic\.isTrue\(bet\.metric_eligible\)/);
  assert.match(client, /withholdPerformancePopulationMap\(/);
  assert.match(client, /renderPerformancePopulationMap\(selectedTierRows, sourceRows, tier\)/);
});

test("slate renders tournament-grouped two-player decision boards with an explicit EV hurdle", () => {
  assert.match(client, /Logic\.groupTournamentEntries\(entries\)/);
  assert.match(client, /Logic\.playerDecisionRows\(row\)\.forEach\(\(playerRow\) =>/);
  assert.match(client, /Logic\.edgeBand\(playerRow\.edge\)/);
  assert.match(client, /\["Player", ""\], \["Price · break-even", ""\]/);
  assert.match(client, /\["NN · live", ""\]/);
  assert.match(client, /\["NN edge", ""\]/);
  assert.match(client, /raw BE/);
  assert.match(client, /qualifies · capital blocked/);
  assert.match(client, /qualifies at 2 pt gate/);
  assert.match(client, /Edge = NN probability − offered price raw break-even/);
});

test("shadow candidates are selectable and presented rather than silently filtered", () => {
  assert.match(html, /<option value="shadow">Shadow candidates<\/option>/);
  assert.match(html, /<option value="all">All available models<\/option>/);
  assert.match(client, /if \(scope === "shadow"\) return isShadow/);
  assert.match(client, /if \(scope === "all"\) return !isTiming/);
  assert.match(client, /Logic\.modelPresentation\(metric\.model\)/);
  assert.match(client, /Shadow candidates/);
  assert.match(html, /Each shadow variant uses one deterministic opening observation per match and model version/);
  assert.doesNotMatch(client, /shadowRowsWithheld/);
  assert.doesNotMatch(html, /Shadow metrics are intentionally withheld/i);
});

test("calibration and market-timing views consume manifest-pinned authoritative rows only", () => {
  assert.match(client, /fetchAll\("dash_model_calibration", CALIBRATION_COLUMNS,[^\n]+generationFilter\)/);
  assert.match(client, /Object\.prototype\.hasOwnProperty\.call\(\s*publishedCounts, "dash_model_calibration"/);
  assert.match(client, /actual\.dash_model_calibration = arrayCount\("calibration"\)/);
  assert.match(client, /store\.calibration\s*\n\s*\.filter\(\(row\) => Logic\.clean\(row\.tier\)/);
  assert.match(client, /Accessible calibration bin table/);
  assert.match(client, /renderMetricExplorer\(sourceRows, tier, selectedTierRows\)/);
  assert.match(client, /\["market_open", "market_close"\]\.includes\(model\)/);
  assert.match(client, /tier\.endsWith\("_market_timing"\) \? "market_open" : "market"/);
  assert.match(html, /<option value="gold_market_timing">/);
  assert.match(html, /<option value="complete_market_timing">/);
  assert.match(html, /<option value="settled_market_timing">All settled · first vs last market<\/option>/);
  assert.match(html, /Reliability diagram/);
  assert.match(html, /GOLD · first vs last market/);
  assert.doesNotMatch(client, /Math\.log|expectedCalibrationError|reliabilityTable|reliability_table|scoreDiagnosticCohort|buildCommonCohort/);
});

test("odds movement is an exact-match lazy projection and delegates strict timing to shared logic", () => {
  assert.match(client, /fetchFiltered\(\s*"dash_odds_history"/);
  assert.match(client, /return fetchAll\(table, columns, order, \{ sync_id: syncId/);
  assert.doesNotMatch(client, /limit: "250"/);
  assert.match(client, /\{ match_uid: Logic\.clean\(row\.match_uid\) \}/);
  assert.match(client, /const series = Logic\.prepareOddsSeries\(rows, row\)/);
  assert.match(logic, /row\.odds_scraped_at \|\| row\.logged_at/);
  assert.match(client, /First observed is the earliest price captured by this pipeline, not necessarily the sportsbook's true opener/);
  assert.match(client, /Last pre-start excludes every observation at or after the exact UTC start/);
  assert.match(client, /Accessible observation table/);
});
