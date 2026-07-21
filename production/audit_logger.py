#!/usr/bin/env python3
"""Append-only audit logs and per-run summaries for the live production pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from logging_utils import append_unique_row, ensure_csv_columns, stable_hash, utc_now_iso
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .logging_utils import append_unique_row, ensure_csv_columns, stable_hash, utc_now_iso


BASE_DIR = Path(__file__).parent
AUDIT_DIR = BASE_DIR / "logs" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SKIPPED_MATCHES_LOG_PATH = AUDIT_DIR / "skipped_live_matches.csv"
SETTLEMENT_AUDIT_LOG_PATH = AUDIT_DIR / "settlement_audit.csv"
RUN_HISTORY_LOG_PATH = AUDIT_DIR / "run_history.csv"

SKIPPED_MATCH_COLUMNS = [
    "skip_event_id",
    "logged_at",
    "run_id",
    "run_started_at",
    "stage",
    "skip_reason_code",
    "skip_reason_detail",
    "match_uid",
    "feature_snapshot_id",
    "prediction_uid",
    "match_date",
    "match_start_time",
    "match_start_dt_local",
    "match_start_at_utc",
    "odds_scraped_at",
    "tournament",
    "event_title",
    "surface",
    "level",
    "round",
    "resolver_source",
    "p1",
    "p2",
    "defaulted_features",
]

SETTLEMENT_AUDIT_COLUMNS = [
    "settlement_event_id",
    "logged_at",
    "run_id",
    "dry_run",
    "row_index",
    "record_status_before",
    "match_uid",
    "prediction_uid",
    "match_date",
    "match_start_time",
    "tournament",
    "round",
    "surface",
    "p1",
    "p2",
    "model_version",
    "ta_player_slug",
    "outcome_code",
    "outcome_detail",
    "ta_match_date_found",
    "ta_event_found",
    "ta_round_found",
    "ta_surface_found",
    "settlement_score",
    "settlement_evidence",
    "actual_winner",
    "score",
]

RUN_HISTORY_COLUMNS = [
    "run_id",
    "run_kind",
    "started_at",
    "completed_at",
    "status",
    "auto_settle_enabled",
    "rankings_refresh_enabled",
    "odds_rows_fetched",
    "odds_rows_candidate",
    "event_discovery_status",
    "atp_events_discovered",
    "atp_events_prefetched",
    "event_discovery_error",
    "kalshi_fetch_status",
    "kalshi_market_rows_fetched",
    "kalshi_two_sided_events",
    "kalshi_observations_logged",
    "kalshi_matches_matched",
    "kalshi_match_status_summary",
    "kalshi_fetch_error",
    "feature_rows_total",
    "feature_rows_ok",
    "feature_rows_skipped",
    "feature_skip_reason_summary",
    "prediction_rows_total",
    "prediction_rows_success",
    "prediction_rows_error",
    "prediction_log_attempts",
    "prediction_log_created",
    "prediction_log_updated",
    "prediction_log_skipped_incomplete",
    "prediction_identity_aliases",
    "prediction_identity_conflicts",
    "bet_opportunities",
    "bets_logged",
    "settlement_candidates",
    "settlement_newly_settled",
    "settlement_auto_settled_bets",
    "settlement_reason_summary",
    "auto_settle_status",
    "auto_settle_error",
    "canonical_ingest_status",
    "canonical_ingest_rows",
    "canonical_ingest_error",
    "reconcile_status",
    "reconcile_error",
    "account_equity",
    "pending_exposure",
    "available_bankroll",
    "exposure_gate_status",
    "performance_shadow_attempts",
    "performance_shadow_logged",
    "performance_shadow_models_loaded",
    "error_message",
]


def _serialize(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def log_skipped_live_match(
    *,
    run_id: str,
    stage: str,
    skip_reason_code: str,
    skip_reason_detail: str = "",
    run_started_at: str = "",
    match_uid: str = "",
    feature_snapshot_id: str = "",
    prediction_uid: str = "",
    match_date: str = "",
    match_start_time: str = "",
    match_start_dt_local: str = "",
    match_start_at_utc: str = "",
    odds_scraped_at: str = "",
    tournament: str = "",
    event_title: str = "",
    surface: str = "",
    level: str = "",
    round_code: str = "",
    resolver_source: str = "",
    p1: str = "",
    p2: str = "",
    defaulted_features: str = "",
) -> bool:
    """Append an audit row for a match skipped from live prediction lineage."""
    skip_event_id = "skip_" + stable_hash(
        run_id,
        stage,
        match_uid,
        feature_snapshot_id,
        prediction_uid,
        skip_reason_code,
        p1,
        p2,
    )
    row = {
        "skip_event_id": skip_event_id,
        "logged_at": utc_now_iso(),
        "run_id": run_id,
        "run_started_at": run_started_at,
        "stage": stage,
        "skip_reason_code": skip_reason_code,
        "skip_reason_detail": skip_reason_detail,
        "match_uid": match_uid,
        "feature_snapshot_id": feature_snapshot_id,
        "prediction_uid": prediction_uid,
        "match_date": match_date,
        "match_start_time": match_start_time,
        "match_start_dt_local": match_start_dt_local,
        "match_start_at_utc": match_start_at_utc,
        "odds_scraped_at": odds_scraped_at,
        "tournament": tournament,
        "event_title": event_title or tournament,
        "surface": surface,
        "level": level,
        "round": round_code,
        "resolver_source": resolver_source,
        "p1": p1,
        "p2": p2,
        "defaulted_features": defaulted_features,
    }
    return append_unique_row(
        SKIPPED_MATCHES_LOG_PATH,
        row,
        SKIPPED_MATCH_COLUMNS,
        unique_key="skip_event_id",
    )


def log_settlement_event(
    *,
    run_id: str,
    dry_run: bool,
    row_index,
    record_status_before: str = "",
    match_uid: str = "",
    prediction_uid: str = "",
    match_date: str = "",
    match_start_time: str = "",
    tournament: str = "",
    round_code: str = "",
    surface: str = "",
    p1: str = "",
    p2: str = "",
    model_version: str = "",
    ta_player_slug: str = "",
    outcome_code: str = "",
    outcome_detail: str = "",
    ta_match_date_found: str = "",
    ta_event_found: str = "",
    ta_round_found: str = "",
    ta_surface_found: str = "",
    settlement_score=None,
    settlement_evidence=None,
    actual_winner=None,
    score: str = "",
) -> bool:
    """Append an audit row for one settlement attempt."""
    settlement_event_id = "settle_" + stable_hash(
        run_id,
        row_index,
        match_uid,
        prediction_uid,
        outcome_code,
        score,
    )
    row = {
        "settlement_event_id": settlement_event_id,
        "logged_at": utc_now_iso(),
        "run_id": run_id,
        "dry_run": bool(dry_run),
        "row_index": row_index,
        "record_status_before": record_status_before,
        "match_uid": match_uid,
        "prediction_uid": prediction_uid,
        "match_date": match_date,
        "match_start_time": match_start_time,
        "tournament": tournament,
        "round": round_code,
        "surface": surface,
        "p1": p1,
        "p2": p2,
        "model_version": model_version,
        "ta_player_slug": ta_player_slug,
        "outcome_code": outcome_code,
        "outcome_detail": outcome_detail,
        "ta_match_date_found": ta_match_date_found,
        "ta_event_found": ta_event_found,
        "ta_round_found": ta_round_found,
        "ta_surface_found": ta_surface_found,
        "settlement_score": settlement_score,
        "settlement_evidence": _serialize(settlement_evidence or ""),
        "actual_winner": actual_winner,
        "score": score,
    }
    return append_unique_row(
        SETTLEMENT_AUDIT_LOG_PATH,
        row,
        SETTLEMENT_AUDIT_COLUMNS,
        unique_key="settlement_event_id",
    )


def upsert_run_history(row: dict) -> None:
    """Insert or update a per-run summary row keyed by `run_id`."""
    run_id = str(row.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run_history rows require run_id")

    df = ensure_csv_columns(RUN_HISTORY_LOG_PATH, RUN_HISTORY_COLUMNS)
    normalized = {col: _serialize(row.get(col, "")) for col in RUN_HISTORY_COLUMNS}
    mask = df["run_id"].astype(str) == run_id
    if mask.any():
        # values are serialized strings; the CSV round-trip infers numeric dtypes
        # and newer pandas raises LossySetitemError on lossy in-place sets —
        # write into object columns so string-serialized values always fit
        df = df.astype(object)
        idx = df[mask].index[0]
        for col, value in normalized.items():
            df.at[idx, col] = value
    else:
        df = pd.concat([df, pd.DataFrame([normalized])], ignore_index=True)
    df.to_csv(RUN_HISTORY_LOG_PATH, index=False)


def finalize_latest_running_run(
    *, status: str = "cancelled", error_message: str = "pipeline process interrupted"
) -> str:
    """Close the newest still-running pipeline row after its process exits.

    GitHub can terminate the Python process at the job deadline before the
    orchestrator's exception handler runs. The workflow calls this helper in
    the following always-capable step, before checkpointing the CSVs. It only
    closes the newest running prediction attempt; it never rewrites a terminal
    row or guesses at prediction/settlement counters.
    """
    terminal_status = str(status or "").strip().lower()
    if terminal_status in {"", "running", "started"}:
        raise ValueError("interrupted run status must be terminal")

    df = ensure_csv_columns(RUN_HISTORY_LOG_PATH, RUN_HISTORY_COLUMNS)
    if df.empty:
        return ""
    kinds = df["run_kind"].fillna("").astype(str).str.strip().str.lower()
    run_ids = df["run_id"].fillna("").astype(str).str.strip()
    statuses = df["status"].fillna("").astype(str).str.strip().str.lower()
    candidates = df[
        run_ids.str.startswith("run_")
        & kinds.isin(["", "prediction_pipeline"])
        & statuses.isin(["", "running", "started"])
    ].copy()
    if candidates.empty:
        return ""

    candidates["_started"] = pd.to_datetime(
        candidates["started_at"], errors="coerce", utc=True, format="mixed"
    )
    candidates = candidates.sort_values(
        ["_started", "run_id"], kind="stable", na_position="first"
    )
    idx = candidates.index[-1]
    run_id = str(df.at[idx, "run_id"]).strip()
    df = df.astype(object)
    df.at[idx, "status"] = terminal_status
    df.at[idx, "completed_at"] = utc_now_iso()
    df.at[idx, "error_message"] = str(error_message or "").strip()
    df.to_csv(RUN_HISTORY_LOG_PATH, index=False)
    return run_id
