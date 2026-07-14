"""Merge and hydrate durable operational state.

The hourly runner is ephemeral and git pushes can race.  These helpers make the
Supabase ``dash_*`` tables an additive recovery bridge: immutable rows are
unioned, terminal states outrank pending/running states, and newer rows win only
when they are at the same quality tier.  The CSVs remain the application-facing
format during this migration; model evaluation continues to read them exactly
as documented.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import tempfile

import pandas as pd


BASE = Path(__file__).resolve().parent
CONTROL_COLUMNS = {"dashboard_row_key", "sync_id"}


@dataclass(frozen=True)
class StateSpec:
    table: str
    relative_path: str
    primary_fields: tuple[str, ...]
    fallback_fields: tuple[str, ...]
    quality_mode: str = "immutable"

    def path(self, base: Path = BASE) -> Path:
        return base / self.relative_path


STATE_SPECS = (
    StateSpec("dash_predictions", "prediction_log.csv", ("match_uid",),
              ("p1", "p2", "match_date"), "prediction"),
    StateSpec("dash_odds_history", "odds_history.csv", ("odds_snapshot_uid",),
              ("match_uid", "odds_scraped_at", "p1_odds_decimal", "p2_odds_decimal")),
    StateSpec("dash_shadow", "logs/performance_v1_shadow_predictions.csv",
              ("shadow_prediction_uid",),
              ("match_uid", "model_version", "feature_snapshot_id")),
    StateSpec("dash_runs", "logs/audit/run_history.csv", ("run_id",),
              ("started_at", "run_kind"), "run"),
    StateSpec("dash_bets", "logs/all_bets.csv", ("bet_id",),
              ("match_uid", "bet_on", "match_date", "timestamp"), "bet"),
    StateSpec("dash_snapshots", "prediction_snapshots.csv", ("prediction_uid",),
              ("match_uid", "feature_snapshot_id", "logged_at")),
    StateSpec("dash_settlement_audit", "logs/audit/settlement_audit.csv",
              ("settlement_event_id",), ("run_id", "match_uid", "logged_at", "outcome_code")),
    StateSpec("dash_skipped_live_matches", "logs/audit/skipped_live_matches.csv",
              ("skip_event_id",), ("run_id", "match_uid", "stage", "skip_reason_code")),
    StateSpec("dash_features", "logs/feature_vectors.csv", ("feature_snapshot_id",),
              ("p1", "p2", "match_date", "run_id")),
    StateSpec("dash_bankroll", "logs/bankroll_history.csv", (),
              ("timestamp", "session_id", "change_reason", "change_amount")),
    StateSpec("dash_sessions", "logs/betting_sessions.csv", ("session_id",),
              ("start_time",), "session"),
)


def _clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(c).strip().lower().replace(" ", "_") for c in result.columns]
    return result.drop(columns=[c for c in CONTROL_COLUMNS if c in result.columns], errors="ignore")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return _normalized_columns(
        pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    )


def row_key(row: pd.Series, spec: StateSpec) -> str:
    for field in spec.primary_fields:
        value = _clean(row.get(field, ""))
        if value:
            return f"{spec.table}:{field}:{value}"
    values = [_clean(row.get(field, "")) for field in spec.fallback_fields]
    if not any(values):
        # Last-resort identity is the full row.  It preserves evidence rather
        # than collapsing unrelated malformed records into one blank key.
        values = [f"{column}={_clean(row.get(column, ''))}" for column in sorted(row.index)]
    digest = sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:24]
    return f"{spec.table}:fallback:{digest}"


def add_row_keys(frame: pd.DataFrame, spec: StateSpec) -> pd.DataFrame:
    result = _normalized_columns(frame)
    if result.empty:
        result["dashboard_row_key"] = pd.Series(dtype=str)
        return result
    result["dashboard_row_key"] = result.apply(lambda row: row_key(row, spec), axis=1)
    return result


def _true(value) -> bool:
    return _clean(value).lower() in {"true", "1", "yes", "y"}


def _quality(frame: pd.DataFrame, mode: str) -> pd.Series:
    score = pd.Series(0, index=frame.index, dtype="int64")
    if mode == "prediction":
        winner = pd.to_numeric(
            frame.get("actual_winner", pd.Series("", index=frame.index)),
            errors="coerce",
        )
        settled = winner.isin([1, 2])
        complete = frame.get("features_complete", pd.Series("", index=frame.index)).map(_true)
        has_probability = frame.get("model_p1_prob", pd.Series("", index=frame.index)).map(_clean).ne("")
        score = settled.astype(int) * 100 + (complete & has_probability).astype(int) * 10
    elif mode == "bet":
        status = frame.get("status", pd.Series("", index=frame.index)).map(_clean).str.lower()
        outcome = frame.get("outcome", pd.Series("", index=frame.index)).map(_clean).str.lower()
        score = (status.isin({"settled", "void", "cancelled", "canceled"})
                 | outcome.isin({"win", "loss", "void", "cancelled", "canceled"})).astype(int) * 100
    elif mode == "run":
        status = frame.get("status", pd.Series("", index=frame.index)).map(_clean).str.lower()
        score = (~status.isin({"", "running", "started"})).astype(int) * 100
    elif mode == "session":
        ended = frame.get("end_time", pd.Series("", index=frame.index)).map(_clean).ne("")
        score = ended.astype(int) * 100
    return score


def _reconcile_prediction_groups(combined: pd.DataFrame,
                                 ordered_columns: list[str]) -> pd.DataFrame:
    """Join opening inference evidence with independently-settled outcomes."""
    probability = combined.get(
        "model_p1_prob", pd.Series("", index=combined.index)
    ).map(_clean).ne("")
    complete = combined.get(
        "features_complete", pd.Series("", index=combined.index)
    ).map(_true)
    feature_id = combined.get(
        "feature_snapshot_id", pd.Series("", index=combined.index)
    ).map(_clean).ne("")
    combined = combined.copy()
    combined["_inference_quality"] = (
        probability.astype(int) * 10
        + complete.astype(int) * 20
        + feature_id.astype(int) * 5
    )
    winner = pd.to_numeric(
        combined.get("actual_winner", pd.Series("", index=combined.index)),
        errors="coerce",
    )
    status = combined.get(
        "record_status", pd.Series("", index=combined.index)
    ).map(_clean).str.lower()
    combined["_is_terminal"] = winner.isin([1, 2]) | status.isin(
        {"settled", "void", "cancelled", "canceled"}
    )
    terminal_fields = (
        "actual_winner", "score", "settled_at", "model_correct",
        "market_correct", "xgb_correct", "rf_correct", "record_status",
        "record_note",
    )
    rows: list[pd.Series] = []
    for _, group in combined.groupby("_row_key", sort=False):
        inference = group.sort_values(
            ["_inference_quality", "_freshness", "_source_order"], kind="stable"
        ).iloc[-1].copy()
        terminal = group[group["_is_terminal"]]
        if not terminal.empty:
            terminal_row = terminal.sort_values(
                ["_freshness", "_source_order"], kind="stable"
            ).iloc[-1]
            for field in terminal_fields:
                if field not in group.columns:
                    continue
                value = terminal_row.get(field, "")
                if _clean(value) or field in {"record_status", "record_note"}:
                    inference[field] = value

        freshest = group.sort_values(
            ["_freshness", "_source_order"], kind="stable"
        ).iloc[-1]
        for field in (c for c in ordered_columns if c.startswith("latest_")):
            value = freshest.get(field, "")
            if _clean(value):
                inference[field] = value
        inference["_freshness"] = group["_freshness"].max()
        inference["_source_order"] = group["_source_order"].max()
        rows.append(inference)
    return pd.DataFrame(rows)


def _freshness(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(0, index=frame.index, dtype="int64")
    for column in (
        "completed_at", "settled_timestamp", "settled_at", "latest_logged_at",
        "logged_at", "timestamp", "started_at", "start_time",
    ):
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True, format="mixed")
        numeric = pd.Series(parsed.astype("int64", copy=False), index=frame.index)
        numeric = numeric.where(parsed.notna(), 0)
        result = pd.concat([result, numeric], axis=1).max(axis=1).astype("int64")
    return result


def merge_state_frames(existing: pd.DataFrame, incoming: pd.DataFrame,
                       spec: StateSpec) -> pd.DataFrame:
    """Return an additive, monotonic merge for one operational table."""
    left = _normalized_columns(existing)
    right = _normalized_columns(incoming)
    ordered_columns = list(dict.fromkeys([*left.columns, *right.columns]))
    if not ordered_columns:
        return pd.DataFrame()
    left = left.reindex(columns=ordered_columns, fill_value="")
    right = right.reindex(columns=ordered_columns, fill_value="")
    combined = pd.concat([left, right], ignore_index=True)
    combined = combined.fillna("")
    if combined.empty:
        return combined[ordered_columns].reset_index(drop=True)
    combined["_row_key"] = combined.apply(lambda row: row_key(row, spec), axis=1)
    combined["_quality"] = _quality(combined, spec.quality_mode)
    combined["_freshness"] = _freshness(combined)
    combined["_source_order"] = range(len(combined))
    if spec.quality_mode == "prediction":
        combined = _reconcile_prediction_groups(combined, ordered_columns)
    else:
        combined = combined.sort_values(
            ["_row_key", "_quality", "_freshness", "_source_order"],
            kind="stable",
        ).drop_duplicates("_row_key", keep="last")
    combined = combined.sort_values(["_freshness", "_source_order"], kind="stable")
    return combined[ordered_columns].reset_index(drop=True)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def hydrate_operational_state(base: Path = BASE, verbose: bool = True) -> dict[str, int]:
    """Merge the durable Supabase snapshot into local CSVs before a cloud run."""
    from canonical_store import connect
    from dashboard_sync import _table_exists, read_table

    counts: dict[str, int] = {}
    planned: dict[StateSpec, pd.DataFrame] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            # Serialize against publishers so hydration reads one accepted
            # generation rather than a set of tables moving underneath it.
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                        ("betting-algo-operational-state",))
            manifest = None
            if _table_exists(cur, "dash_sync_manifest"):
                cur.execute(
                    """SELECT sync_id, status, table_counts_json
                       FROM dash_sync_manifest
                       ORDER BY published_at DESC, sync_id DESC LIMIT 1"""
                )
                manifest = cur.fetchone()
            expected_counts = json.loads(manifest[2]) if manifest and manifest[2] else {}
            accepted_sync_id = str(manifest[0]) if manifest else ""
            if manifest and str(manifest[1]) not in {"success", "degraded"}:
                raise RuntimeError(f"latest dashboard generation is not accepted: {manifest[1]}")

            for spec in STATE_SPECS:
                remote = read_table(cur, spec.table)
                if remote.empty:
                    expected = expected_counts.get(spec.table)
                    if accepted_sync_id and expected is not None and int(expected) != 0:
                        raise RuntimeError(
                            f"{spec.table} count mismatch for {accepted_sync_id}: "
                            f"manifest={expected}, rows=0"
                        )
                    continue
                if accepted_sync_id:
                    sync_ids = set(remote.get("sync_id", pd.Series(dtype=str)).astype(str))
                    if sync_ids != {accepted_sync_id}:
                        raise RuntimeError(
                            f"{spec.table} is not pinned to accepted generation {accepted_sync_id}: {sync_ids}"
                        )
                    expected = expected_counts.get(spec.table)
                    if expected is not None and int(expected) != len(remote):
                        raise RuntimeError(
                            f"{spec.table} count mismatch for {accepted_sync_id}: "
                            f"manifest={expected}, rows={len(remote)}"
                        )
                path = spec.path(base)
                local = load_csv(path)
                merged = merge_state_frames(remote, local, spec)
                planned[spec] = merged
                counts[spec.table] = len(merged)
                if verbose:
                    print(f"   ↩️ hydrated {path.name}: {len(local)} local + "
                          f"{len(remote)} durable → {len(merged)} merged")

    # Do not mutate any local file until every remote table has been read,
    # generation-checked, and merged. Stage all outputs first; if an os.replace
    # fails, restore every file already swapped from local backups.
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hydrate-state-", dir=base) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged: dict[StateSpec, Path] = {}
        backups: dict[StateSpec, Path | None] = {}
        for spec, frame in planned.items():
            staged_path = temp_dir / f"{spec.table}.csv"
            frame.to_csv(staged_path, index=False)
            staged[spec] = staged_path
            original = spec.path(base)
            if original.exists():
                backup_path = temp_dir / f"{spec.table}.backup"
                shutil.copy2(original, backup_path)
                backups[spec] = backup_path
            else:
                backups[spec] = None

        replaced: list[StateSpec] = []
        try:
            for spec, staged_path in staged.items():
                destination = spec.path(base)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, destination)
                replaced.append(spec)
        except Exception:
            for spec in reversed(replaced):
                destination = spec.path(base)
                backup = backups[spec]
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    shutil.copy2(backup, destination)
            raise
    return counts
