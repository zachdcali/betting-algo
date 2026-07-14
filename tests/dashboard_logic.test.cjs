"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const Logic = require("../docs/dashboard_logic.js");

const NOW = Date.parse("2026-07-13T18:00:00Z");

function completeSlateRow(overrides = {}) {
  return {
    tournament: "Los Cabos",
    surface: "Hard",
    level: "A",
    round: "R32",
    p1: "Player One",
    p2: "Player Two",
    match_uid: "match_exact_1",
    prediction_uid: "pred_exact_1",
    features_complete: true,
    feature_snapshot_id: "feat_exact_1",
    model_p1_prob: 0.61,
    market_p1_prob: 0.55,
    p1_odds_decimal: 1.83,
    p2_odds_decimal: 2.1,
    match_start_at_utc: "2026-07-13T20:00:00Z",
    ...overrides,
  };
}

test("timezone-naive operational timestamps are interpreted as UTC", () => {
  assert.equal(
    Logic.parseTimestamp("2026-07-14T08:26:24.639361"),
    Date.parse("2026-07-14T08:26:24.639361Z"),
  );
  assert.equal(
    Logic.parseTimestamp("2026-07-14 08:08:02"),
    Date.parse("2026-07-14T08:08:02Z"),
  );
  assert.equal(
    Logic.parseTimestamp("2026-07-14T08:08:02-04:00"),
    Date.parse("2026-07-14T08:08:02-04:00"),
  );
});

test("normalizes the operational bet outcome vocabulary", () => {
  assert.equal(Logic.normalizeBetOutcome({ outcome: "win", status: "settled" }), "win");
  assert.equal(Logic.normalizeBetOutcome({ outcome: "won", status: "settled" }), "win");
  assert.equal(Logic.normalizeBetOutcome({ outcome: "loss", status: "settled" }), "loss");
  assert.equal(Logic.normalizeBetOutcome({ outcome: "lost", status: "settled" }), "loss");
  assert.equal(Logic.normalizeBetOutcome({ outcome: "push", status: "settled" }), "void");
  assert.equal(Logic.normalizeBetOutcome({ outcome: "", status: "canceled" }), "cancelled");
  assert.equal(Logic.normalizeBetOutcome({ outcome: "", status: "pending" }), "pending");
  assert.equal(Logic.normalizeBetOutcome({ outcome: "", status: "settled" }), "settled_unknown");
});

test("pending exposure is split by the settlement SLA without guessing local match time", () => {
  const pending = { status: "pending", outcome: "", timestamp: "2026-07-13T12:00:00Z" };
  assert.equal(
    Logic.pendingBetSlaStatus(pending, { match_start_at_utc: "2026-07-13T20:00:00Z" }, NOW).state,
    "active",
  );
  assert.equal(
    Logic.pendingBetSlaStatus(pending, { match_start_at_utc: "2026-07-12T20:00:00Z" }, NOW).state,
    "overdue",
  );
  const conservative = Logic.pendingBetSlaStatus({ ...pending, match_date: "2026-07-10", match_start_time: "7/10/26 2:00 PM" }, null, NOW);
  assert.equal(conservative.state, "overdue");
  assert.equal(conservative.basis, "conservative unzoned match date");
  assert.equal(Logic.pendingBetSlaStatus({ status: "pending" }, null, NOW).state, "unverified");
});

test("current snapshot eligibility requires complete immutable lineage", () => {
  assert.deepEqual(Logic.classifySlateRow(completeSlateRow(), NOW), {
    state: "eligible",
    reasons: [],
    startAt: Date.parse("2026-07-13T20:00:00Z"),
  });

  const blocked = Logic.classifySlateRow(completeSlateRow({ feature_snapshot_id: "", round: "" }), NOW);
  assert.equal(blocked.state, "blocked");
  assert.ok(blocked.reasons.includes("Exact feature snapshot missing"));
  assert.ok(blocked.reasons.includes("Round unresolved"));

  const expired = Logic.classifySlateRow(
    completeSlateRow({ match_start_at_utc: "2026-07-13T17:59:59Z" }),
    NOW,
  );
  assert.equal(expired.state, "expired");
  assert.ok(expired.reasons.includes("Scheduled start has passed"));

  const naiveStart = Logic.classifySlateRow(
    completeSlateRow({ match_start_at_utc: "", match_start_time: "7/13/26 2:30 PM" }),
    NOW,
  );
  assert.equal(naiveStart.state, "blocked");
  assert.ok(naiveStart.reasons.includes("UTC start time unavailable"));
});

test("feature IDs are distinguished from referentially verified feature rows", () => {
  const row = completeSlateRow();
  assert.equal(Logic.featureReferenceStatus(row, new Set(), false), "unverified");
  assert.equal(Logic.featureReferenceStatus(row, new Set(), true), "not_found");
  assert.equal(Logic.featureReferenceStatus(row, new Set(), true, new Set(["feat_exact_1"])), "invalid");
  assert.equal(Logic.featureReferenceStatus(row, new Set(["feat_exact_1"]), true), "verified");
  assert.equal(Logic.featureReferenceStatus({ ...row, feature_snapshot_id: "" }, new Set(), true), "missing_id");
});

test("skipped-live audit is blocked or expired without guessing a naive timezone", () => {
  const blocked = Logic.classifySkippedRow({
    stage: "feature_extraction",
    skip_reason_code: "feature_error",
    skip_reason_detail: "rank lookup failed",
    match_start_time: "7/13/26 2:30 PM",
  }, NOW);
  assert.equal(blocked.state, "blocked");
  assert.equal(blocked.startAt, null);
  assert.match(blocked.reasons[0], /feature extraction: feature error/);

  const expired = Logic.classifySkippedRow({
    stage: "pre_inference",
    skip_reason_code: "scheduled_start_passed",
  }, NOW);
  assert.equal(expired.state, "expired");
});

test("pipeline health distinguishes failure, degraded no-odds, and stale state", () => {
  const previousRun = {
    run_id: "run_20260713T173000Z",
    run_kind: "prediction_pipeline",
    status: "success",
    started_at: "2026-07-13T17:30:00Z",
    prediction_rows_success: 4,
  };
  const predictions = [{
    latest_run_id: previousRun.run_id,
    latest_logged_at: "2026-07-13T17:35:00Z",
  }];

  const healthy = Logic.computeHealth({ runs: [previousRun], predictions, now: NOW });
  assert.equal(healthy.state, "healthy");

  const manifestPinned = Logic.computeHealth({
    runs: [
      { ...previousRun, run_id: "run_20260713T175900Z", started_at: "2026-07-13T17:59:00Z", status: "running" },
      previousRun,
    ],
    predictions,
    latestAttemptRunId: previousRun.run_id,
    now: NOW,
  });
  assert.equal(manifestPinned.latestAttempt.run.run_id, previousRun.run_id);
  assert.equal(manifestPinned.state, "healthy");

  const acceptedPinned = Logic.computeHealth({
    runs: [
      { ...previousRun, run_id: "run_20260713T174500Z", started_at: "2026-07-13T17:45:00Z" },
      previousRun,
    ],
    predictions: [
      { latest_run_id: "run_20260713T174500Z", latest_logged_at: "2026-07-13T17:46:00Z" },
      ...predictions,
    ],
    latestAttemptRunId: "run_20260713T174500Z",
    acceptedPredictionRunId: previousRun.run_id,
    now: NOW,
  });
  assert.equal(acceptedPinned.predictionRun.id, previousRun.run_id);

  const partialAccepted = Logic.computeHealth({
    runs: [{ ...previousRun, status: "partial", reconcile_status: "error", reconcile_error: "settlement join failed", exposure_gate_status: "blocked_pending_exposure" }],
    predictions,
    latestAttemptRunId: previousRun.run_id,
    acceptedPredictionRunId: previousRun.run_id,
    now: NOW,
  });
  assert.equal(partialAccepted.predictionRun.id, previousRun.run_id);
  assert.equal(partialAccepted.state, "degraded");
  assert.match(partialAccepted.reasons.join(" "), /Reconciliation error: settlement join failed/);
  assert.match(partialAccepted.reasons.join(" "), /Portfolio exposure gate is blocked pending exposure/);

  const failed = Logic.computeHealth({
    runs: [{ ...previousRun, run_id: "run_20260713T175500Z", started_at: "2026-07-13T17:55:00Z", status: "no_features", prediction_rows_success: 0 }, previousRun],
    predictions,
    now: NOW,
  });
  assert.equal(failed.state, "failed");
  assert.match(failed.reasons.join(" "), /no features/);

  const degraded = Logic.computeHealth({
    runs: [{ ...previousRun, run_id: "run_20260713T175500Z", started_at: "2026-07-13T17:55:00Z", status: "no_odds", prediction_rows_success: 0 }, previousRun],
    predictions,
    now: NOW,
  });
  assert.equal(degraded.state, "degraded");
  assert.match(degraded.reasons.join(" "), /no odds/);

  const stale = Logic.computeHealth({
    runs: [{ ...previousRun, started_at: "2026-07-13T15:00:00Z" }],
    predictions: [{ ...predictions[0], latest_logged_at: "2026-07-13T15:05:00Z" }],
    now: NOW,
  });
  assert.equal(stale.state, "stale");
});

test("terminal run freshness uses completion time while active runs use start time", () => {
  const completed = {
    run_id: "run_20260713T170000Z",
    run_kind: "prediction_pipeline",
    status: "partial",
    started_at: "2026-07-13T17:00:00Z",
    completed_at: "2026-07-13T17:58:00Z",
  };
  const running = { ...completed, status: "running" };

  assert.equal(Logic.runTimestamp(completed), Date.parse(completed.completed_at));
  assert.equal(Logic.runTimestamp(running), Date.parse(running.started_at));
});

test("next cadence calculation advances across :17 and :47", () => {
  assert.equal(Logic.nextScheduledRun("2026-07-13T18:16:30Z").toISOString(), "2026-07-13T18:17:00.000Z");
  assert.equal(Logic.nextScheduledRun("2026-07-13T18:17:00Z").toISOString(), "2026-07-13T18:47:00.000Z");
  assert.equal(Logic.nextScheduledRun("2026-07-13T18:50:00Z").toISOString(), "2026-07-13T19:17:00.000Z");
});

test("manifest count comparison accepts explicit zero and rejects omissions or truncation", () => {
  assert.deepEqual(
    Logic.compareManifestCounts({ dash_bets: 0, dash_predictions: 4 }, { dash_bets: 0, dash_predictions: 4 }),
    { ok: true, issues: [] },
  );
  const omitted = Logic.compareManifestCounts({ dash_predictions: 4 }, { dash_bets: 0, dash_predictions: 4 });
  assert.equal(omitted.ok, false);
  assert.ok(omitted.issues.includes("manifest omits dash_bets"));
  const truncated = Logic.compareManifestCounts({ dash_predictions: 4 }, { dash_predictions: 3 });
  assert.equal(truncated.ok, false);
  assert.ok(truncated.issues.includes("dash_predictions expected 4, loaded 3"));
});
