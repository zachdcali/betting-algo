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

function assertClose(actual, expected, message = "") {
  assert.ok(
    Math.abs(actual - expected) < 1e-12,
    `${message || "values differ"}: expected ${expected}, received ${actual}`,
  );
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

test("snapshot hands hydrate only from the exact structurally valid feature row", () => {
  const snapshot = completeSlateRow({ p1_hand: "", p2_hand: "" });
  const feature = {
    feature_snapshot_id: "feat_exact_1",
    build_status: "ok",
    features_complete: false,
    feature_schema_sha256: "schema-hash",
    feature_vector_sha256: "vector-hash",
    feature_count: "141",
    p1_hand: "r",
    p2_hand: "U",
  };

  assert.equal(Logic.isStructurallyValidFeatureRow(feature), true);
  assert.deepEqual(
    Logic.hydrateSnapshotHands(snapshot, feature),
    { ...snapshot, p1_hand: "R", p2_hand: "U" },
  );
  assert.deepEqual(
    Logic.hydrateSnapshotHands(snapshot, { ...feature, feature_snapshot_id: "feat_other" }),
    snapshot,
  );
  assert.deepEqual(
    Logic.hydrateSnapshotHands(snapshot, { ...feature, feature_count: 140 }),
    snapshot,
  );
  assert.deepEqual(
    Logic.hydrateSnapshotHands(snapshot, { ...feature, p1_hand: "right", p2_hand: "L" }),
    { ...snapshot, p1_hand: "", p2_hand: "L" },
  );
  assert.deepEqual(
    Logic.hydrateSnapshotHands({ ...snapshot, p1_hand: "R", p2_hand: "invalid" }, null),
    { ...snapshot, p1_hand: "R", p2_hand: "" },
  );
  assert.equal(Logic.normalizedHandCode("a"), "A");
  assert.equal(Logic.handDisplayLabel("R"), "right-handed");
  assert.equal(Logic.handDisplayLabel("L"), "left-handed");
  assert.equal(Logic.handDisplayLabel("A"), "ambidextrous");
  assert.equal(Logic.handDisplayLabel("U"), "hand unknown");
  assert.equal(Logic.handDisplayLabel("invalid"), "hand unknown");
});

test("slate fallback identity requires both players", () => {
  assert.equal(
    Logic.slateEvidenceKey({ p1: "Alpha", p2: "Beta", match_date: "2026-07-13" }, "both"),
    "pair:alpha|beta|2026-07-13",
  );
  assert.equal(
    Logic.slateEvidenceKey({ p1: "Alpha", p2: "", match_date: "2026-07-13" }, "one"),
    "row:one",
  );
  assert.equal(
    Logic.slateEvidenceKey({ p1: "", p2: "", match_date: "2026-07-13" }, "missing"),
    "row:missing",
  );
});

test("accepted slate funnel is built from explicit monotone intersections", () => {
  const entry = (id, state, overrides = {}, extras = {}) => ({
    row: completeSlateRow({
      match_uid: `match_${id}`,
      prediction_uid: `pred_${id}`,
      feature_snapshot_id: `feat_${id}`,
      ...overrides,
    }),
    source: "snapshot",
    classification: { state, reasons: [] },
    auditRows: [],
    ...extras,
  });
  const entries = [
    entry("eligible", "eligible"),
    entry("expired", "expired"),
    entry("conflict", "blocked", { record_status: "identity_conflict" }),
    entry("not_complete", "blocked"),
    {
      row: { p1: "Skipped", p2: "Match", match_date: "2026-07-13" },
      source: "skipped",
      classification: { state: "blocked", reasons: ["feature error"] },
      auditRows: [{ skip_reason_code: "feature_error" }],
    },
  ];
  const funnel = Logic.acceptedSlateFunnel(
    entries,
    new Set(["feat_eligible", "feat_expired", "feat_conflict"]),
  );

  assert.deepEqual(funnel.counts, {
    accepted: 5,
    finite: 4,
    complete: 3,
    identityClean: 2,
    dataValidNow: 1,
  });
  assert.equal(funnel.expired, 1);
  assert.equal(funnel.monotone, true);
  const stages = ["accepted", "finite", "complete", "identityClean", "dataValidNow"];
  stages.slice(1).forEach((stage, index) => {
    const parent = new Set(funnel.stageKeys[stages[index]]);
    assert.ok(funnel.stageKeys[stage].every((key) => parent.has(key)));
    assert.ok(funnel.counts[stage] <= funnel.counts[stages[index]]);
  });
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

test("matching skip evidence remains visible beside a snapshot", () => {
  const classification = Logic.classifySlateEvidence(
    completeSlateRow({
      features_complete: false,
      defaulted_features: "round_code=None,structural_validation",
      model_p1_prob: null,
    }),
    [{
      stage: "feature_extraction",
      skip_reason_code: "feature_schema_invalid",
      skip_reason_detail: "one_hot_cardinality:round:0",
      match_start_at_utc: "2026-07-13T20:00:00Z",
    }],
    NOW,
  );

  assert.equal(classification.state, "blocked");
  assert.ok(classification.reasons.includes(
    "Skipped at feature extraction: feature schema invalid",
  ));
  assert.ok(classification.reasons.includes("one_hot_cardinality:round:0"));
});

test("administrative identity tombstone blocks an accepted immutable snapshot", () => {
  const snapshot = completeSlateRow({
    record_status: "pending",
    features_complete: true,
  });
  const audit = {
    run_id: "run_collision",
    match_uid: snapshot.match_uid,
    feature_snapshot_id: snapshot.feature_snapshot_id,
    prediction_uid: snapshot.prediction_uid,
    p1: snapshot.p1,
    p2: snapshot.p2,
    stage: "administrative_quarantine",
    skip_reason_code: "match_identity_conflict",
    skip_reason_detail: "operational prediction tombstoned; immutable snapshots retained",
    match_start_at_utc: snapshot.match_start_at_utc,
  };

  // The immutable snapshot intentionally remains unchanged. Its exact-run
  // skipped-live audit is the browser-facing tombstone.
  const classification = Logic.classifySlateEvidence(snapshot, [audit], NOW);
  assert.equal(classification.state, "blocked");
  assert.ok(classification.reasons.includes(
    "Skipped at administrative quarantine: match identity conflict",
  ));
  assert.equal(
    Logic.primaryBlockerGroup({
      row: snapshot,
      source: "snapshot",
      featureReference: "verified",
      auditRows: [audit],
      classification,
    }),
    "Identity conflict",
  );
});

test("blocked rows receive one mutually exclusive primary group", () => {
  const featureFailure = {
    row: completeSlateRow({ features_complete: false, model_p1_prob: null }),
    source: "snapshot",
    featureReference: "invalid",
    auditRows: [{ skip_reason_code: "feature_schema_invalid" }],
    classification: { reasons: ["Features incomplete"] },
  };
  const identity = {
    row: completeSlateRow({ record_status: "identity_conflict", features_complete: false }),
    source: "snapshot",
    featureReference: "verified",
    auditRows: [{ skip_reason_code: "match_identity_conflict" }],
    classification: { reasons: ["Match identity conflict"] },
  };
  const incomplete = {
    row: completeSlateRow({ features_complete: false, defaulted_features: "Player1_Height" }),
    source: "snapshot",
    featureReference: "invalid",
    auditRows: [],
    classification: { reasons: ["Incomplete features: Player1_Height"] },
  };

  assert.equal(Logic.primaryBlockerGroup(featureFailure), "Feature / prediction build failed");
  assert.equal(Logic.primaryBlockerGroup(identity), "Identity conflict");
  assert.equal(Logic.primaryBlockerGroup(incomplete), "Feature values incomplete");
  assert.equal(
    Logic.primaryBlockerGroup({
      row: completeSlateRow({ market_p1_prob: null }),
      source: "snapshot",
      featureReference: "verified",
      auditRows: [],
      classification: { reasons: ["Market probability unavailable"] },
    }),
    "Market price unavailable",
  );
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

test("two-player decision rows preserve player orientation and invert every P1 probability", () => {
  const rows = Logic.playerDecisionRows(completeSlateRow({
    p1: "Alpha Server",
    p2: "Beta Returner",
    p1_rank: 12,
    p2_rank: 48,
    p1_hand: "R",
    p2_hand: "L",
    p1_odds_decimal: 2,
    p2_odds_decimal: 2.5,
    market_p1_prob: 0.57,
    model_p1_prob: 0.64,
    xgb_p1_prob: 0.59,
    rf_p1_prob: 0.54,
  }));

  assert.equal(rows.length, 2);
  assert.deepEqual(
    rows.map(({ side, player, rank, hand }) => ({ side, player, rank, hand })),
    [
      { side: 1, player: "Alpha Server", rank: 12, hand: "R" },
      { side: 2, player: "Beta Returner", rank: 48, hand: "L" },
    ],
  );
  const expectedProbabilities = [
    { rawBreakEven: 0.5, marketProbability: 0.57, nnProbability: 0.64, xgbProbability: 0.59, rfProbability: 0.54 },
    { rawBreakEven: 0.4, marketProbability: 0.43, nnProbability: 0.36, xgbProbability: 0.41, rfProbability: 0.46 },
  ];
  rows.forEach((row, index) => {
    Object.entries(expectedProbabilities[index]).forEach(([field, expected]) => {
      assertClose(row[field], expected, `side ${index + 1} ${field}`);
    });
  });
  assertClose(rows[0].edge, 0.14, "P1 edge");
  assertClose(rows[1].edge, -0.04, "P2 edge");

  const missingDriver = Logic.playerDecisionRows(completeSlateRow({ model_p1_prob: null }));
  assert.equal(missingDriver[0].nnProbability, null);
  assert.equal(missingDriver[1].nnProbability, null);
  assert.equal(missingDriver[0].edge, null);
  assert.equal(missingDriver[1].edge, null);
  assert.equal(
    Logic.playerDecisionRows(completeSlateRow({ p1_hand: "right" }))[0].hand,
    "",
  );
});

test("EV edge bands honor the configured two-point gate and increasing intensity", () => {
  const cases = [
    [null, "missing"],
    [-0.000001, "negative"],
    [0, "watch"],
    [0.019999, "watch"],
    [0.02, "positive-low"],
    [0.049999, "positive-low"],
    [0.05, "positive-medium"],
    [0.099999, "positive-medium"],
    [0.1, "positive-strong"],
  ];
  cases.forEach(([edge, expected]) => assert.equal(Logic.edgeBand(edge), expected, String(edge)));
});

test("metric signals respect direction, comparability, and calibration's target of one", () => {
  for (const metric of ["log_loss", "brier", "ece", "max_drawdown_kelly"]) {
    assert.equal(Logic.metricSignal(metric, 0.2, 0.3, true), "good", `${metric} lower`);
    assert.equal(Logic.metricSignal(metric, 0.4, 0.3, true), "bad", `${metric} higher`);
    assert.equal(Logic.metricSignal(metric, 0.2, 0.3, false), "neutral", `${metric} incomparable`);
  }
  for (const metric of ["accuracy", "auc"]) {
    assert.equal(Logic.metricSignal(metric, 0.7, 0.6, true), "good", `${metric} higher`);
    assert.equal(Logic.metricSignal(metric, 0.5, 0.6, true), "bad", `${metric} lower`);
  }
  assert.equal(Logic.metricSignal("cal_slope", 1), "good");
  assert.equal(Logic.metricSignal("cal_slope", 1.15), "good");
  assert.equal(Logic.metricSignal("cal_slope", 0.7), "warning");
  assert.equal(Logic.metricSignal("cal_slope", 1.6), "bad");
  assert.equal(Logic.metricSignal("roi_kelly", 0.001), "good");
  assert.equal(Logic.metricSignal("roi_kelly", -0.001), "bad");
  assert.equal(Logic.metricSignal("roi_kelly", 0), "neutral");
  assert.equal(Logic.metricSignal("n", 249), "warning");
  assert.equal(Logic.metricSignal("n", 250), "neutral");
});

test("shadow model presentation is readable while preserving exact registry identity", () => {
  const xgb = Logic.modelPresentation(
    "shadow_performance_v1_xgb_depth5_recency_hl_8y__2026-04-25",
  );
  assert.deepEqual(xgb, {
    label: "XGB · depth5 recency half-life 8y",
    role: "shadow",
    family: "xgboost",
    exact: "shadow_performance_v1_xgb_depth5_recency_hl_8y__2026-04-25",
  });
  assert.equal(Logic.modelPresentation("shadow_performance_v1_cat_depth6_screening_one_hot__2026-04-25").label, "CatBoost · depth6 screening one hot");
  assert.equal(Logic.modelPresentation("shadow_performance_v1_nn_logits_128_64_robust__2026-04-25").family, "nn");
  assert.deepEqual(Logic.modelPresentation("market_close"), {
    label: "Market · last pre-start",
    role: "benchmark",
    family: "market",
    exact: "market_close",
  });
  assert.equal(xgb.label.includes("performance_v1"), false);
  assert.equal(xgb.label.includes("_"), false);
});

test("odds series is chronological and excludes observations at or after exact start", () => {
  const start = "2026-07-13T20:00:00Z";
  const series = Logic.prepareOddsSeries([
    { logged_at: "2026-07-13T20:00:01Z", market_p1_prob: 0.61, p1_odds_decimal: 1.7 },
    { logged_at: "2026-07-13T19:30:00Z", market_p1_prob: 0.58, market_p2_prob: 0.42, p1_odds_decimal: 1.8, p2_odds_decimal: 2.05 },
    { logged_at: "2026-07-13T18:00:00Z", market_p1_prob: 0.55, p1_odds_decimal: 1.9, p2_odds_decimal: 1.95 },
    { logged_at: "2026-07-13T20:00:00Z", market_p1_prob: 0.6, p1_odds_decimal: 1.75 },
    { logged_at: "2026-07-13T19:30:00Z", market_p1_prob: 0.59, p1_odds_decimal: 1.78, p2_odds_decimal: 2.1 },
    { logged_at: "not-a-time", market_p1_prob: 0.5 },
    { logged_at: "2026-07-13T19:45:00Z", market_p1_prob: 1.2 },
  ], { match_start_at_utc: start });

  assert.equal(series.startAt, Date.parse(start));
  assert.equal(series.lastLabel, "last pre-start");
  assert.deepEqual(series.points.map((point) => point.at), [
    Date.parse("2026-07-13T18:00:00Z"),
    Date.parse("2026-07-13T19:30:00Z"),
  ]);
  assert.equal(series.first.p1, 0.55);
  assertClose(series.first.p2, 0.45, "opening P2 complement");
  assert.equal(series.last.p1, 0.59, "same-timestamp observations deterministically keep the last row");
  assertClose(series.last.p2, 0.41, "closing P2 complement");
  assert.ok(series.points.every((point) => point.at < Date.parse(start)));

  const legacy = Logic.prepareOddsSeries([
    { logged_at: "2026-07-13T18:00:00Z", market_p1_prob: 0.55 },
  ]);
  assert.equal(legacy.lastLabel, "latest observed");
  assert.equal(legacy.startAt, null);

  const explicitUtcClock = Logic.prepareOddsSeries([
    {
      logged_at: "2026-07-13T20:30:00Z",
      odds_scraped_at: "2026-07-13T19:00:00Z",
      market_p1_prob: 0.52,
    },
    {
      logged_at: "2026-07-13T19:00:00Z",
      odds_scraped_at: "2026-07-13T20:00:00Z",
      market_p1_prob: 0.60,
    },
  ], { match_start_at_utc: start });
  assert.deepEqual(explicitUtcClock.points.map((point) => point.at), [
    Date.parse("2026-07-13T19:00:00Z"),
  ]);
  assert.equal(explicitUtcClock.first.p1, 0.52);
});
