(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.DashboardLogic = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SUCCESS_STATUSES = new Set(["success", "completed", "complete", "ok"]);
  const FAILURE_STATUSES = new Set([
    "failed",
    "failure",
    "error",
    "odds_fetch_error",
    "timed_out",
    "timeout",
    "cancelled",
    "canceled",
    "no_features",
    "no_predictions",
  ]);
  const DEGRADED_STATUSES = new Set(["no_odds", "degraded", "partial"]);
  const HAND_CODES = new Set(["R", "L", "A", "U"]);

  function clean(value) {
    if (value === null || value === undefined) return "";
    const text = String(value).trim();
    return /^(nan|none|null)$/i.test(text) ? "" : text;
  }

  function numberOrNull(value) {
    if (value === null || value === undefined || clean(value) === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function isTrue(value) {
    if (value === true || value === 1) return true;
    return ["true", "1", "1.0", "t", "yes"].includes(clean(value).toLowerCase());
  }

  function parseRunId(runId) {
    const match = clean(runId).match(/(?:run|settle)_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/i);
    if (!match) return null;
    const [, year, month, day, hour, minute, second] = match;
    const time = Date.UTC(+year, +month - 1, +day, +hour, +minute, +second);
    return Number.isFinite(time) ? time : null;
  }

  function parseTimestamp(value) {
    const text = clean(value);
    if (!text) return null;
    // Operational CSV/Postgres timestamps are written in UTC, but older rows
    // omit the trailing zone. Date.parse treats those ISO values as browser
    // local time, which can make fresh data appear to come from the future.
    const naiveOperationalIso = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/;
    const normalized = naiveOperationalIso.test(text) ? `${text.replace(" ", "T")}Z` : text;
    const parsed = Date.parse(normalized);
    return Number.isFinite(parsed) ? parsed : parseRunId(text);
  }

  function runTimestamp(run) {
    if (!run) return null;
    const status = clean(run.status).toLowerCase();
    const isActive = !status || ["running", "started"].includes(status);
    if (isActive) {
      return (
        parseTimestamp(run.started_at) ||
        parseTimestamp(run.completed_at) ||
        parseRunId(run.run_id)
      );
    }
    return (
      parseTimestamp(run.completed_at) ||
      parseTimestamp(run.started_at) ||
      parseRunId(run.run_id)
    );
  }

  function isPipelineRun(run) {
    const kind = clean(run && run.run_kind).toLowerCase();
    const id = clean(run && run.run_id).toLowerCase();
    return !kind || kind === "prediction_pipeline" || id.startsWith("run_");
  }

  function latestRunAttempt(runs) {
    return (runs || [])
      .filter(isPipelineRun)
      .map((run) => ({ run, at: runTimestamp(run) }))
      .filter((entry) => entry.at !== null)
      .sort((a, b) => b.at - a.at)[0] || null;
  }

  function predictionRunId(row) {
    return clean(row && (row.latest_run_id || row.run_id));
  }

  function latestSuccessfulPredictionRun(predictions, runs, acceptedRunId = "") {
    const runMap = new Map((runs || []).map((run) => [clean(run.run_id), run]));
    const groups = new Map();

    (predictions || []).forEach((row) => {
      const id = predictionRunId(row);
      if (!id) return;
      const timestamp =
        parseTimestamp(row.latest_logged_at) ||
        parseTimestamp(row.logged_at) ||
        parseRunId(id);
      const previous = groups.get(id) || { id, at: timestamp, rowCount: 0 };
      previous.rowCount += 1;
      if (timestamp !== null && (previous.at === null || timestamp > previous.at)) previous.at = timestamp;
      groups.set(id, previous);
    });

    const preferredId = clean(acceptedRunId);
    if (preferredId) {
      const selected = groups.get(preferredId);
      const run = runMap.get(preferredId) || null;
      const acceptedStatus = clean(run && run.status).toLowerCase();
      if (!selected || (run && !SUCCESS_STATUSES.has(acceptedStatus) && acceptedStatus !== "partial")) return null;
      return { ...selected, run };
    }

    const candidates = [...groups.values()]
      .filter((entry) => {
        const run = runMap.get(entry.id);
        if (!run) return true;
        return SUCCESS_STATUSES.has(clean(run.status).toLowerCase());
      })
      .sort((a, b) => {
        if (a.at !== null && b.at !== null && a.at !== b.at) return b.at - a.at;
        return b.id.localeCompare(a.id);
      });

    const selected = candidates[0] || [...groups.values()].sort((a, b) => b.id.localeCompare(a.id))[0];
    if (!selected) return null;
    return { ...selected, run: runMap.get(selected.id) || null };
  }

  function ageMinutes(timestamp, now) {
    if (timestamp === null || timestamp === undefined) return null;
    return Math.max(0, (now - timestamp) / 60000);
  }

  function computeHealth({ runs = [], predictions = [], errors = {}, latestAttemptRunId = "", acceptedPredictionRunId = "", now = Date.now() } = {}) {
    const nowMs = now instanceof Date ? now.getTime() : Number(now);
    const pinnedAttemptId = clean(latestAttemptRunId);
    const pinnedAttemptRun = pinnedAttemptId ? runs.find((run) => clean(run.run_id) === pinnedAttemptId) : null;
    const pinnedAttemptAt = runTimestamp(pinnedAttemptRun);
    const attempt = pinnedAttemptId
      ? (pinnedAttemptRun && pinnedAttemptAt !== null ? { run: pinnedAttemptRun, at: pinnedAttemptAt } : null)
      : latestRunAttempt(runs);
    const predictionRun = latestSuccessfulPredictionRun(predictions, runs, acceptedPredictionRunId);
    const attemptAgeMinutes = attempt ? ageMinutes(attempt.at, nowMs) : null;
    const predictionAgeMinutes = predictionRun ? ageMinutes(predictionRun.at, nowMs) : null;
    const loadErrors = Object.entries(errors || {}).filter(([, value]) => Boolean(value));
    const reasons = [];
    const attemptStatus = clean(attempt && attempt.run.status).toLowerCase();

    if (!attempt) reasons.push("No pipeline attempt is visible in run history.");
    if (!predictionRun) reasons.push("No successful prediction-bearing run is visible.");
    if (attempt && FAILURE_STATUSES.has(attemptStatus)) {
      reasons.push(`Latest pipeline attempt ended ${attemptStatus.replaceAll("_", " ")}.`);
    } else if (attempt && DEGRADED_STATUSES.has(attemptStatus)) {
      reasons.push(`Latest pipeline attempt ended ${attemptStatus.replaceAll("_", " ")}.`);
    } else if (attemptStatus === "running") {
      reasons.push("Latest pipeline attempt is still running.");
    }
    if (attemptAgeMinutes !== null && attemptAgeMinutes > 90) {
      reasons.push(`No pipeline attempt has appeared for ${Math.round(attemptAgeMinutes)} minutes.`);
    }
    if (predictionAgeMinutes !== null && predictionAgeMinutes > 90) {
      reasons.push(`Last successful prediction run is ${Math.round(predictionAgeMinutes)} minutes old.`);
    }
    if (loadErrors.length) {
      reasons.push(`Partial dashboard load: ${loadErrors.map(([name]) => name).join(", ")}.`);
    }
    if (attempt) {
      [
        ["Auto-settlement", "auto_settle_status", "auto_settle_error"],
        ["Canonical ingest", "canonical_ingest_status", "canonical_ingest_error"],
        ["Reconciliation", "reconcile_status", "reconcile_error"],
      ].forEach(([label, statusField, errorField]) => {
        const stageStatus = clean(attempt.run[statusField]).toLowerCase();
        const stageError = clean(attempt.run[errorField]);
        if (stageError || FAILURE_STATUSES.has(stageStatus) || stageStatus === "partial") {
          const detail = stageError ? `: ${stageError.slice(0, 180)}` : "";
          reasons.push(`${label} ${stageStatus.replaceAll("_", " ") || "failed"}${detail}.`);
        }
      });
      const exposureGate = clean(attempt.run.exposure_gate_status).toLowerCase();
      if (exposureGate.startsWith("blocked")) {
        reasons.push(`Portfolio exposure gate is ${exposureGate.replaceAll("_", " ")}.`);
      }
    }
    if (
      attempt &&
      predictionRun &&
      SUCCESS_STATUSES.has(attemptStatus) &&
      numberOrNull(attempt.run.prediction_rows_success) > 0 &&
      clean(attempt.run.run_id) !== predictionRun.id
    ) {
      reasons.push("Latest successful run is newer than the accepted prediction mirror.");
    }

    let state = "healthy";
    if (
      !attempt ||
      !predictionRun ||
      FAILURE_STATUSES.has(attemptStatus) ||
      (predictionAgeMinutes !== null && predictionAgeMinutes > 180)
    ) {
      state = "failed";
    } else if (
      (attemptAgeMinutes !== null && attemptAgeMinutes > 90) ||
      (predictionAgeMinutes !== null && predictionAgeMinutes > 90)
    ) {
      state = "stale";
    } else if (reasons.length) {
      state = "degraded";
    }

    return {
      state,
      reasons,
      latestAttempt: attempt,
      predictionRun,
      attemptAgeMinutes,
      predictionAgeMinutes,
      loadErrors,
    };
  }

  function parseMatchStart(row) {
    if (!row) return null;
    for (const field of ["match_start_at_utc", "scheduled_start_at", "match_start_time_utc"]) {
      const parsed = parseTimestamp(row[field]);
      if (parsed !== null) return parsed;
    }

    const raw = clean(row.match_start_time);
    if (!raw) return null;
    // Bovada's legacy display value is Eastern-local with no offset. Never
    // guess its timezone in the browser. A zoned ISO value remains safe.
    if (!/(?:Z|[+-]\d{2}:?\d{2})$/i.test(raw)) return null;
    return parseTimestamp(raw);
  }

  function validWinner(value) {
    const winner = numberOrNull(value);
    return winner === 1 || winner === 2;
  }

  function exactFeatureId(row) {
    return clean(row && (row.feature_snapshot_id || row.latest_feature_snapshot_id));
  }

  function normalizedHandCode(value) {
    const code = clean(value).toUpperCase();
    return HAND_CODES.has(code) ? code : "";
  }

  function handDisplayLabel(value) {
    const code = normalizedHandCode(value);
    if (code === "R") return "right-handed";
    if (code === "L") return "left-handed";
    if (code === "A") return "ambidextrous";
    return "hand unknown";
  }

  function isStructurallyValidFeatureRow(row) {
    return Boolean(
      exactFeatureId(row)
      && clean(row && row.build_status).toLowerCase() === "ok"
      && clean(row && row.feature_schema_sha256)
      && clean(row && row.feature_vector_sha256)
      && numberOrNull(row && row.feature_count) === 141
    );
  }

  function hydrateSnapshotHands(snapshotRow, featureRow) {
    const snapshot = {
      ...(snapshotRow || {}),
      p1_hand: normalizedHandCode(snapshotRow && snapshotRow.p1_hand),
      p2_hand: normalizedHandCode(snapshotRow && snapshotRow.p2_hand),
    };
    const snapshotFeatureId = exactFeatureId(snapshot);
    if (
      !snapshotFeatureId
      || !isStructurallyValidFeatureRow(featureRow)
      || exactFeatureId(featureRow) !== snapshotFeatureId
    ) {
      return snapshot;
    }
    return {
      ...snapshot,
      p1_hand: normalizedHandCode(featureRow.p1_hand),
      p2_hand: normalizedHandCode(featureRow.p2_hand),
    };
  }

  function slateEvidenceKey(row, fallback = "") {
    const uid = clean(row && row.match_uid);
    if (uid) return `match:${uid}`;
    const featureId = exactFeatureId(row);
    if (featureId) return `feature:${featureId}`;
    const players = [clean(row && row.p1), clean(row && row.p2)]
      .map((value) => value.toLowerCase());
    if (players.length === 2 && players.every(Boolean)) {
      return `pair:${players.sort().join("|")}|${clean(row && row.match_date)}`;
    }
    return `row:${clean(fallback)}`;
  }

  function acceptedSlateFunnel(entries = [], acceptedFeatureIds = new Set()) {
    const featureIds = acceptedFeatureIds instanceof Set
      ? acceptedFeatureIds
      : new Set(acceptedFeatureIds || []);
    const unique = new Map();
    entries.forEach((entry, index) => {
      const key = slateEvidenceKey(entry && entry.row, `accepted:${index}`);
      if (!unique.has(key)) unique.set(key, entry);
    });
    const accepted = [...unique.entries()].map(([key, entry]) => ({ key, entry }));
    const finite = accepted.filter(({ entry }) => (
      clean(entry && entry.source).toLowerCase() === "snapshot"
      && numberOrNull(entry && entry.row && entry.row.model_p1_prob) !== null
    ));
    const complete = finite.filter(({ entry }) => {
      const featureId = exactFeatureId(entry && entry.row);
      return Boolean(featureId && featureIds.has(featureId));
    });
    const identityClean = complete.filter(({ entry }) => {
      const row = entry && entry.row ? entry.row : {};
      const audits = entry && Array.isArray(entry.auditRows) ? entry.auditRows : [];
      const reasons = entry && entry.classification && Array.isArray(entry.classification.reasons)
        ? entry.classification.reasons
        : [];
      const evidence = [
        clean(row.record_status),
        ...audits.map((audit) => clean(audit.skip_reason_code)),
        ...reasons,
      ].join(" | ").toLowerCase();
      return !evidence.includes("identity_conflict") && !evidence.includes("identity conflict");
    });
    const dataValidNow = identityClean.filter(({ entry }) => (
      clean(entry && entry.classification && entry.classification.state).toLowerCase() === "eligible"
    ));
    const stageKeys = {
      accepted: accepted.map(({ key }) => key),
      finite: finite.map(({ key }) => key),
      complete: complete.map(({ key }) => key),
      identityClean: identityClean.map(({ key }) => key),
      dataValidNow: dataValidNow.map(({ key }) => key),
    };
    const subset = (left, right) => {
      const parent = new Set(right);
      return left.every((key) => parent.has(key));
    };
    const monotone = (
      subset(stageKeys.finite, stageKeys.accepted)
      && subset(stageKeys.complete, stageKeys.finite)
      && subset(stageKeys.identityClean, stageKeys.complete)
      && subset(stageKeys.dataValidNow, stageKeys.identityClean)
    );
    return {
      counts: Object.fromEntries(
        Object.entries(stageKeys).map(([name, keys]) => [name, keys.length]),
      ),
      stageKeys,
      monotone,
      expired: accepted.filter(({ entry }) => (
        clean(entry && entry.classification && entry.classification.state).toLowerCase() === "expired"
      )).length,
    };
  }

  function featureReferenceStatus(row, featureIds, referenceLoaded = false, seenFeatureIds = null) {
    const featureId = exactFeatureId(row);
    if (!featureId) return "missing_id";
    if (!referenceLoaded) return "unverified";
    if (featureIds instanceof Set && featureIds.has(featureId)) return "verified";
    if (seenFeatureIds instanceof Set && seenFeatureIds.has(featureId)) return "invalid";
    return "not_found";
  }

  function blockedReasons(row) {
    const reasons = [];
    const defaults = clean(row.defaulted_features);
    const recordStatus = clean(row.record_status).toLowerCase();
    if (recordStatus === "identity_conflict") reasons.push("Match identity conflict");
    if (!clean(row.match_uid)) reasons.push("Immutable match ID missing");
    if (!clean(row.prediction_uid)) reasons.push("Immutable prediction snapshot ID missing");
    if (!clean(row.tournament)) reasons.push("Tournament unresolved");
    if (!clean(row.surface)) reasons.push("Surface unresolved");
    if (!clean(row.level)) reasons.push("Level unresolved");
    if (!clean(row.round)) reasons.push("Round unresolved");
    if (!isTrue(row.features_complete)) reasons.push(defaults ? `Incomplete features: ${defaults}` : "Features incomplete");
    if (!exactFeatureId(row)) reasons.push("Exact feature snapshot missing");
    if (numberOrNull(row.model_p1_prob) === null) reasons.push("NN prediction unavailable");
    if (numberOrNull(row.market_p1_prob) === null) reasons.push("Market probability unavailable");
    if (numberOrNull(row.p1_odds_decimal) === null || numberOrNull(row.p2_odds_decimal) === null) {
      reasons.push("Two-sided price unavailable");
    }
    if (parseMatchStart(row) === null) reasons.push("UTC start time unavailable");
    return reasons;
  }

  function classifySlateRow(row, now = Date.now()) {
    const nowMs = now instanceof Date ? now.getTime() : Number(now);
    if (validWinner(row && row.actual_winner)) return { state: "settled", reasons: [], startAt: parseMatchStart(row) };

    const status = clean(row && row.record_status).toLowerCase();
    const startAt = parseMatchStart(row);
    const explicitExpired = status.includes("expired") || status.includes("stale") || status === "cancelled" || status === "canceled";
    if (explicitExpired || (startAt !== null && startAt <= nowMs)) {
      const reason = explicitExpired ? `Record status: ${status.replaceAll("_", " ")}` : "Scheduled start has passed";
      return { state: "expired", reasons: [reason], startAt };
    }

    const reasons = blockedReasons(row || {});
    if (reasons.length) return { state: "blocked", reasons, startAt };
    return { state: "eligible", reasons: [], startAt };
  }

  function classifySkippedRow(row, now = Date.now()) {
    const nowMs = now instanceof Date ? now.getTime() : Number(now);
    // Skip audit historically stored a timezone-naive Bovada-local string.
    // Only use an explicit UTC field for expiry math; the skip reason remains
    // visible when that migration has not landed yet.
    const startAt = parseTimestamp(row && row.match_start_at_utc);
    const code = clean(row && row.skip_reason_code).toLowerCase();
    const detail = clean(row && row.skip_reason_detail);
    const stage = clean(row && row.stage).replaceAll("_", " ") || "pipeline";
    const reasons = [`Skipped at ${stage}: ${(code || "reason unavailable").replaceAll("_", " ")}`];
    if (detail && detail.toLowerCase() !== code) reasons.push(detail);
    const explicitlyExpired = ["scheduled_start_passed", "match_already_completed"].includes(code);
    if (explicitlyExpired || (startAt !== null && startAt <= nowMs)) {
      reasons.unshift(explicitlyExpired ? "Skip audit marks this match as already started" : "Scheduled start has passed");
      return { state: "expired", reasons, startAt };
    }
    return { state: "blocked", reasons, startAt };
  }

  function mergeReasons(...groups) {
    const seen = new Set();
    const merged = [];
    groups.flat().forEach((reason) => {
      const text = clean(reason);
      if (!text || seen.has(text)) return;
      seen.add(text);
      merged.push(text);
    });
    return merged;
  }

  function classifySlateEvidence(row, auditRows = [], now = Date.now()) {
    const snapshot = classifySlateRow(row, now);
    if (!auditRows.length || snapshot.state === "settled") return snapshot;
    const audits = auditRows.map((audit) => classifySkippedRow(audit, now));
    const auditExpired = audits.some((audit) => audit.state === "expired");
    return {
      state: snapshot.state === "expired" || auditExpired ? "expired" : "blocked",
      reasons: mergeReasons(
        snapshot.reasons,
        audits.flatMap((audit) => audit.reasons),
      ),
      startAt: snapshot.startAt ?? audits.find((audit) => audit.startAt !== null)?.startAt ?? null,
    };
  }

  function primaryBlockerGroup(entry) {
    const row = entry && entry.row ? entry.row : {};
    const auditRows = entry && Array.isArray(entry.auditRows) ? entry.auditRows : [];
    const classification = entry && entry.classification ? entry.classification : { reasons: [] };
    const source = clean(entry && entry.source).toLowerCase();
    const auditCodes = auditRows.map((audit) => clean(audit.skip_reason_code).toLowerCase());
    const evidence = [
      clean(row.record_status),
      clean(row.record_note),
      clean(row.defaulted_features),
      ...(classification.reasons || []),
      ...auditRows.flatMap((audit) => [
        clean(audit.stage), clean(audit.skip_reason_code), clean(audit.skip_reason_detail),
      ]),
    ].join(" | ").toLowerCase();

    if (auditCodes.some((code) => [
      "scheduled_start_passed", "match_already_completed",
      "inside_pre_match_buffer_5m", "match_start_time_missing",
    ].includes(code))) return "Start time / pre-start guard";
    if (
      auditCodes.some((code) => (
        code.startsWith("feature_")
        || code.startsWith("prediction_")
        || code === "no_features"
        || code === "no_predictions"
      ))
    ) return "Feature / prediction build failed";
    if (
      clean(row.record_status).toLowerCase() === "identity_conflict"
      || auditCodes.includes("match_identity_conflict")
      || evidence.includes("identity conflict")
      || evidence.includes("identity_conflict")
    ) return "Identity conflict";
    if (!isTrue(row.features_complete)) return "Feature values incomplete";

    const featureReference = clean(entry && entry.featureReference).toLowerCase();
    if (["unverified", "not_found", "invalid", "missing_id"].includes(featureReference)) {
      return "Feature lineage / integrity";
    }
    if (numberOrNull(row.model_p1_prob) === null) return "Model output unavailable";
    if (
      numberOrNull(row.market_p1_prob) === null
      || numberOrNull(row.p1_odds_decimal) === null
      || numberOrNull(row.p2_odds_decimal) === null
    ) return "Market price unavailable";
    if (parseMatchStart(row) === null) return "Start time unavailable";
    if (source === "skipped" || auditRows.length) return "Other pipeline skip";
    return "Other required input";
  }

  function normalizeBetOutcome(bet) {
    const outcome = clean(bet && bet.outcome).toLowerCase();
    const status = clean(bet && bet.status).toLowerCase();
    if (["win", "won"].includes(outcome)) return "win";
    if (["loss", "lost"].includes(outcome)) return "loss";
    if (["void", "push", "pushed", "refund", "refunded"].includes(outcome)) return "void";
    if (["cancelled", "canceled"].includes(outcome) || ["cancelled", "canceled"].includes(status)) return "cancelled";
    if (status === "pending" || !status) return "pending";
    if (status === "settled") return "settled_unknown";
    return status || "unknown";
  }

  function currentAccountState(bankrollRows, bets, minStake = 1, trusted = true) {
    const statuses = (bets || []).map((bet) => clean(bet && bet.status).toLowerCase());
    const pendingCount = statuses.filter((status) => status === "pending").length;
    const settledCount = statuses.filter((status) => status === "settled").length;
    const pendingExposure = (bets || []).reduce((total, bet, index) => {
      if (statuses[index] !== "pending") return total;
      const stake = numberOrNull(bet && bet.stake);
      return total + (stake !== null && stake >= 0 ? stake : 0);
    }, 0);
    if (!trusted) {
      return {
        verified: false,
        equity: null,
        pendingExposure: null,
        available: null,
        gate: "unverified",
        observedAt: null,
        reason: "manifest generation or account resources are unverified",
      };
    }
    const tolerance = 1e-6;
    const candidates = (bankrollRows || [])
      .map((row) => ({
        row,
        at: parseTimestamp(row && row.timestamp),
        equity: numberOrNull(row && (row.account_equity ?? row.bankroll)),
        reportedPending: numberOrNull(row && row.pending_exposure),
        reportedAvailable: numberOrNull(row && row.available_bankroll),
        pendingCount: numberOrNull(row && row.num_pending_bets),
        settledCount: numberOrNull(row && row.num_settled_bets),
        rowKey: clean(row && row.dashboard_row_key),
      }))
      .filter((entry) => (
        entry.equity !== null
        && entry.reportedPending !== null
        && Math.abs(entry.reportedPending - pendingExposure) <= tolerance
        && entry.pendingCount === pendingCount
        && entry.settledCount === settledCount
      ))
      .sort((a, b) => (
        (b.at ?? Number.NEGATIVE_INFINITY) - (a.at ?? Number.NEGATIVE_INFINITY)
        || b.rowKey.localeCompare(a.rowKey)
      ));
    const authority = candidates[0] || null;
    if (!authority) {
      return {
        verified: false,
        equity: null,
        pendingExposure,
        available: null,
        gate: "unverified",
        observedAt: null,
        reason: "no bankroll row agrees with the manifest-pinned bet counts and pending stakes",
      };
    }
    const available = Math.max(0, authority.equity - pendingExposure);
    const availableAgrees = (
      authority.reportedAvailable === null
      || Math.abs(authority.reportedAvailable - available) <= tolerance
    );
    if (!availableAgrees) {
      return {
        verified: false,
        equity: authority.equity,
        pendingExposure,
        available: null,
        gate: "unverified",
        observedAt: authority.at,
        reason: "bankroll row available capital disagrees with equity minus pending exposure",
      };
    }
    return {
      verified: true,
      equity: authority.equity,
      pendingExposure,
      available,
      gate: available < minStake ? "blocked pending exposure" : "open",
      observedAt: authority.at,
      reason: "manifest-pinned bankroll row reconciled to bet counts and pending stakes",
    };
  }

  function generationTrustIssue(state, countsOk, resourceNames = []) {
    const current = state || {};
    const errors = current.errors || {};
    if (clean(errors.manifest)) return `manifest refresh failed: ${clean(errors.manifest)}`;
    if (clean(errors.generation)) return `generation refresh failed: ${clean(errors.generation)}`;
    const manifest = current.manifest || null;
    if (!manifest) return "sync manifest is unavailable";
    const status = clean(manifest.status).toLowerCase();
    if (status !== "success") return `sync manifest status is ${status || "missing"}`;
    const manifestSyncId = clean(manifest.sync_id);
    const loadedSyncId = clean(current.loadedSyncId);
    if (!manifestSyncId) return "sync manifest ID is missing";
    if (!loadedSyncId || loadedSyncId !== manifestSyncId) {
      return `loaded generation does not match manifest (${loadedSyncId || "none"} vs ${manifestSyncId})`;
    }
    if (!countsOk) return "manifest generation counts are unverified";
    for (const name of resourceNames || []) {
      if (clean(errors[name])) return `${name} refresh failed: ${clean(errors[name])}`;
    }
    return "";
  }

  function pendingBetSlaStatus(bet, prediction, now = Date.now(), settlementHours = 18, unzonedHours = 72) {
    if (normalizeBetOutcome(bet) !== "pending") return { state: "not_pending", deadline: null, basis: "not pending" };
    const nowMs = now instanceof Date ? now.getTime() : Number(now);
    const startAt = parseMatchStart(prediction) ?? parseMatchStart(bet);
    if (startAt !== null) {
      const deadline = startAt + settlementHours * 3600000;
      return { state: nowMs > deadline ? "overdue" : "active", deadline, basis: "exact UTC start" };
    }

    const matchDate = clean((bet && bet.match_date) || (prediction && prediction.match_date));
    if (/^\d{4}-\d{2}-\d{2}$/.test(matchDate)) {
      const dateStart = Date.parse(`${matchDate}T00:00:00Z`);
      if (Number.isFinite(dateStart)) {
        const deadline = dateStart + unzonedHours * 3600000;
        return { state: nowMs > deadline ? "overdue" : "active", deadline, basis: "conservative unzoned match date" };
      }
    }

    const placedAt = parseTimestamp(bet && bet.timestamp);
    if (placedAt !== null) {
      const deadline = placedAt + unzonedHours * 3600000;
      return { state: nowMs > deadline ? "overdue" : "active", deadline, basis: "conservative placement age" };
    }
    return { state: "unverified", deadline: null, basis: "missing match and placement time" };
  }

  function nextScheduledRun(now = Date.now()) {
    const nowDate = now instanceof Date ? new Date(now.getTime()) : new Date(now);
    const candidates = [];
    for (const offsetHours of [0, 1]) {
      for (const minute of [17, 47]) {
        const candidate = new Date(nowDate.getTime());
        candidate.setSeconds(0, 0);
        candidate.setMinutes(minute);
        candidate.setHours(nowDate.getHours() + offsetHours);
        if (candidate.getTime() > nowDate.getTime()) candidates.push(candidate);
      }
    }
    return candidates.sort((a, b) => a - b)[0] || null;
  }

  function compareManifestCounts(expected, actual) {
    const expectedCounts = expected && typeof expected === "object" && !Array.isArray(expected) ? expected : {};
    const actualCounts = actual && typeof actual === "object" && !Array.isArray(actual) ? actual : {};
    const issues = [];
    Object.entries(actualCounts).forEach(([table, count]) => {
      if (!Object.prototype.hasOwnProperty.call(expectedCounts, table)) {
        issues.push(`manifest omits ${table}`);
        return;
      }
      const expectedCount = numberOrNull(expectedCounts[table]);
      const actualCount = numberOrNull(count);
      if (expectedCount === null) issues.push(`${table} manifest count is invalid`);
      else if (actualCount === null) issues.push(`${table} count is unverified`);
      else if (actualCount !== expectedCount) issues.push(`${table} expected ${expectedCount}, loaded ${actualCount}`);
    });
    Object.keys(expectedCounts).forEach((table) => {
      if (!Object.prototype.hasOwnProperty.call(actualCounts, table)) issues.push(`${table} has no client count verifier`);
    });
    return { ok: issues.length === 0, issues };
  }

  function playerDecisionRows(row) {
    const p1Probability = numberOrNull(row && row.model_p1_prob);
    return [1, 2].map((side) => {
      const isPlayerOne = side === 1;
      const oddsDecimal = numberOrNull(row && row[`p${side}_odds_decimal`]);
      const rawBreakEven = oddsDecimal !== null && oddsDecimal > 1
        ? 1 / oddsDecimal
        : null;
      const invert = (value) => {
        const probability = numberOrNull(value);
        return probability === null ? null : isPlayerOne ? probability : 1 - probability;
      };
      const nnProbability = p1Probability === null
        ? null
        : isPlayerOne ? p1Probability : 1 - p1Probability;
      return {
        side,
        player: clean(row && row[`p${side}`]),
        rank: numberOrNull(row && row[`p${side}_rank`]),
        hand: normalizedHandCode(row && row[`p${side}_hand`]),
        oddsDecimal,
        rawBreakEven,
        marketProbability: invert(row && row.market_p1_prob),
        nnProbability,
        xgbProbability: invert(row && row.xgb_p1_prob),
        rfProbability: invert(row && row.rf_p1_prob),
        edge: nnProbability === null || rawBreakEven === null
          ? null
          : nnProbability - rawBreakEven,
      };
    });
  }

  function edgeBand(edge) {
    const value = numberOrNull(edge);
    if (value === null) return "missing";
    if (value < 0) return "negative";
    if (value < 0.02) return "watch";
    if (value < 0.05) return "positive-low";
    if (value < 0.10) return "positive-medium";
    return "positive-strong";
  }

  function metricSignal(metric, value, baseline = null, comparable = true) {
    const number = numberOrNull(value);
    if (number === null) return "neutral";
    const key = clean(metric).toLowerCase();
    if (["roi_flat", "roi_kelly", "pnl_flat", "pnl_kelly"].includes(key)) {
      if (number > 1e-12) return "good";
      if (number < -1e-12) return "bad";
      return "neutral";
    }
    if (key === "cal_slope") {
      const distance = Math.abs(number - 1);
      if (distance <= 0.2) return "good";
      if (distance <= 0.5) return "warning";
      return "bad";
    }
    if (key === "n") return number < 250 ? "warning" : "neutral";
    if (!comparable) return "neutral";
    const benchmark = numberOrNull(baseline);
    if (benchmark === null || Math.abs(number - benchmark) <= 1e-12) return "neutral";
    if (["log_loss", "brier", "ece", "max_drawdown_kelly"].includes(key)) {
      return number < benchmark ? "good" : "bad";
    }
    if (["accuracy", "auc"].includes(key)) {
      return number > benchmark ? "good" : "bad";
    }
    return "neutral";
  }

  function modelPresentation(model) {
    const exact = clean(model);
    const shadow = exact.startsWith("shadow_");
    const raw = shadow ? exact.slice("shadow_".length) : exact;
    if (raw === "market") return { label: "Market at prediction", role: "benchmark", family: "market", exact };
    if (raw === "market_open") return { label: "Market · first observed", role: "benchmark", family: "market", exact };
    if (raw === "market_close") return { label: "Market · last pre-start", role: "benchmark", family: "market", exact };
    if (!shadow) {
      const core = { nn: "NN", xgb: "XGB", rf: "Random forest" };
      return { label: core[raw] || raw, role: "promoted", family: raw, exact };
    }
    let label = raw
      .replace(/^performance_v1_/, "")
      .replace(/__\d{4}-\d{2}-\d{2}$/, "")
      .replace(/^xgb_/, "XGB · ")
      .replace(/^cat_/, "CatBoost · ")
      .replace(/^lgbm_/, "LightGBM · ")
      .replace(/^nn_/, "NN · ")
      .replaceAll("_", " ")
      .replace(/\bhl (\d+)y\b/i, "half-life $1y")
      .replace(/\breg\b/i, "regularized")
      .replace(/\s+/g, " ")
      .trim();
    label = label.replace(/^xgb\b/i, "XGB").replace(/^catboost\b/i, "CatBoost").replace(/^lightgbm\b/i, "LightGBM").replace(/^nn\b/i, "NN");
    const family = /^XGB/i.test(label) ? "xgboost" : /^CatBoost/i.test(label) ? "catboost" : /^LightGBM/i.test(label) ? "lightgbm" : /^NN/i.test(label) ? "nn" : "shadow";
    return { label, role: "shadow", family, exact };
  }

  function groupTournamentEntries(entries) {
    const groups = new Map();
    (entries || []).forEach((entry) => {
      const row = entry && entry.row ? entry.row : entry || {};
      const tournament = clean(row.tournament) || "Tournament unresolved";
      if (!groups.has(tournament)) groups.set(tournament, { tournament, entries: [] });
      groups.get(tournament).entries.push(entry);
    });
    return [...groups.values()];
  }

  function prepareOddsSeries(rows, context = {}) {
    const contextStart = parseMatchStart(context);
    const points = [];
    (rows || []).forEach((row) => {
      // odds_scraped_at is explicitly UTC. Older logged_at values can be
      // timezone-naive host-local timestamps, so they are fallback evidence.
      const at = parseTimestamp(row && (row.odds_scraped_at || row.logged_at));
      const p1 = numberOrNull(row && row.market_p1_prob);
      const rowStart = parseMatchStart(row);
      const startAt = contextStart === null ? rowStart : contextStart;
      if (at === null || p1 === null || p1 < 0 || p1 > 1) return;
      if (startAt !== null && at >= startAt) return;
      const p2Value = numberOrNull(row && row.market_p2_prob);
      points.push({
        at,
        p1,
        p2: p2Value === null ? 1 - p1 : p2Value,
        p1OddsDecimal: numberOrNull(row && row.p1_odds_decimal),
        p2OddsDecimal: numberOrNull(row && row.p2_odds_decimal),
      });
    });
    points.sort((a, b) => a.at - b.at);
    const deduped = [];
    points.forEach((point) => {
      if (deduped.length && deduped[deduped.length - 1].at === point.at) deduped[deduped.length - 1] = point;
      else deduped.push(point);
    });
    return {
      points: deduped,
      startAt: contextStart,
      first: deduped[0] || null,
      last: deduped[deduped.length - 1] || null,
      lastLabel: contextStart === null ? "latest observed" : "last pre-start",
    };
  }

  return {
    SUCCESS_STATUSES,
    FAILURE_STATUSES,
    DEGRADED_STATUSES,
    clean,
    numberOrNull,
    isTrue,
    parseRunId,
    parseTimestamp,
    runTimestamp,
    latestRunAttempt,
    latestSuccessfulPredictionRun,
    computeHealth,
    parseMatchStart,
    validWinner,
    exactFeatureId,
    normalizedHandCode,
    handDisplayLabel,
    isStructurallyValidFeatureRow,
    hydrateSnapshotHands,
    slateEvidenceKey,
    acceptedSlateFunnel,
    featureReferenceStatus,
    blockedReasons,
    classifySlateRow,
    classifySkippedRow,
    mergeReasons,
    classifySlateEvidence,
    primaryBlockerGroup,
    normalizeBetOutcome,
    currentAccountState,
    generationTrustIssue,
    pendingBetSlaStatus,
    nextScheduledRun,
    compareManifestCounts,
    playerDecisionRows,
    edgeBand,
    metricSignal,
    modelPresentation,
    groupTournamentEntries,
    prepareOddsSeries,
  };
});
