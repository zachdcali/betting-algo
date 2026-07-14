"""Atomically publish additive operational state to Supabase.

Every dashboard table is planned first, then replaced inside one transaction.
Remote-only rows are merged back before replacement, so a stale runner cannot
delete settlements or immutable lineage.  A manifest identifies the accepted
generation and its exact row counts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from glob import glob
import io
import json
import math
import os
import sys
import uuid

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canonical_store import connect  # noqa: E402
from operational_state import (  # noqa: E402
    STATE_SPECS, add_row_keys, hydrate_operational_state, load_csv,
    merge_state_frames,
)


def _json_scalar(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, bool)):
        if not isinstance(value, bool) and not math.isfinite(float(value)):
            return None
        return value
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        numeric = float(text_value)
        return numeric if math.isfinite(numeric) else None
    except ValueError:
        return text_value


def _load_feature_state(aggregate_path) -> pd.DataFrame:
    """Build exact drill-down lineage from aggregate and immutable run files."""
    from feature_vector_log import COLS, feature_fingerprint
    from features.performance_v1 import PERFORMANCE_FEATURES
    from models.inference import EXACT_141_FEATURES

    aggregate = load_csv(aggregate_path).reindex(columns=COLS, fill_value="")
    existing_ids = set(
        value for value in aggregate["feature_snapshot_id"].fillna("").astype(str) if value
    )
    rows: list[dict] = []
    feature_names = [*EXACT_141_FEATURES, *PERFORMANCE_FEATURES]
    for path in sorted(glob(str(aggregate_path.parent / "features_*.csv"))):
        run_frame = pd.read_csv(path, low_memory=False)
        if "feature_snapshot_id" not in run_frame.columns:
            continue
        run_frame = run_frame[
            run_frame["feature_snapshot_id"].fillna("").astype(str).ne("")
        ]
        for _, source in run_frame.iterrows():
            feature_id = str(source.get("feature_snapshot_id", "")).strip()
            if not feature_id or feature_id in existing_ids:
                continue
            payload = {
                name: _json_scalar(source.get(name))
                for name in feature_names if name in source.index and pd.notna(source.get(name))
            }
            defaults = source.get("meta_defaulted_features", "")
            payload["_defaulted_features"] = "" if pd.isna(defaults) else str(defaults)
            build_status = str(source.get("status", "unknown") or "unknown")
            schema_hash, vector_hash, feature_count = feature_fingerprint(payload)

            def hand(prefix: str) -> str:
                for label in ("U", "L", "R"):
                    value = pd.to_numeric(source.get(f"{prefix}_Hand_{label}"), errors="coerce")
                    if pd.notna(value) and float(value) == 1.0:
                        return label
                return ""

            rows.append({
                "p1": source.get("player1_raw", ""),
                "p2": source.get("player2_raw", ""),
                "match_date": source.get("meta_match_date", ""),
                "logged_at": source.get("timestamp", source.get("run_started_at", "")),
                "run_id": source.get("run_id", ""),
                "match_uid": source.get("match_uid", ""),
                "feature_snapshot_id": feature_id,
                "build_status": build_status,
                "features_complete": (
                    build_status == "ok"
                    and not payload["_defaulted_features"]
                    and bool(vector_hash)
                ),
                "p1_hand": hand("P1"),
                "p2_hand": hand("P2"),
                "feature_schema_sha256": schema_hash,
                "feature_vector_sha256": vector_hash,
                "feature_count": feature_count,
                "features_json": json.dumps(payload, separators=(",", ":"), default=str),
            })
            existing_ids.add(feature_id)
    if rows:
        aggregate = pd.concat([aggregate, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
    return aggregate.reindex(columns=COLS, fill_value="")


def _table_exists(cur, table: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (f'public."{table}"',))
    return cur.fetchone()[0] is not None


def read_table(cur, table: str) -> pd.DataFrame:
    if not _table_exists(cur, table):
        return pd.DataFrame()
    cur.execute(f'SELECT * FROM "{table}"')
    rows = cur.fetchall()
    columns = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=columns).fillna("")


def _ensure_schema(cur, table: str, frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    if not _table_exists(cur, table):
        definitions = ", ".join(f'"{column}" text' for column in columns)
        cur.execute(f'CREATE TABLE "{table}" ({definitions})')
        return
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s",
        (table,),
    )
    existing = {row[0] for row in cur.fetchall()}
    for column in columns:
        if column not in existing:
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" text')


def _replace_table(cur, table: str, frame: pd.DataFrame) -> None:
    _ensure_schema(cur, table, frame)
    cur.execute(f'TRUNCATE "{table}"')
    if frame.empty:
        return
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    column_list = ", ".join(f'"{column}"' for column in frame.columns)
    with cur.copy(
        f'COPY "{table}" ({column_list}) FROM STDIN WITH (FORMAT csv, NULL \'\')'
    ) as copy:
        while chunk := buffer.read(65536):
            copy.write(chunk)


def _enable_public_read(cur, table: str) -> None:
    cur.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')
    cur.execute(
        "SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename=%s AND policyname='dash_read'",
        (table,),
    )
    if not cur.fetchone():
        cur.execute(f'CREATE POLICY dash_read ON "{table}" FOR SELECT USING (true)')
    cur.execute(f'GRANT SELECT ON "{table}" TO anon')


def _latest_run_id(frame: pd.DataFrame) -> str:
    if frame.empty or "run_id" not in frame.columns:
        return ""
    candidates = frame.copy()
    values = candidates["run_id"].fillna("").astype(str)
    candidates = candidates[values.str.startswith("run_")]
    if "run_kind" in candidates.columns:
        kinds = candidates["run_kind"].fillna("").astype(str)
        candidates = candidates[kinds.isin(["", "prediction_pipeline"])]
    if candidates.empty:
        return ""
    if "started_at" in candidates.columns:
        candidates = candidates.assign(
            _started=pd.to_datetime(
                candidates["started_at"], errors="coerce", utc=True, format="mixed"
            )
        ).sort_values(["_started", "run_id"], kind="stable", na_position="first")
        return str(candidates.iloc[-1]["run_id"])
    return str(candidates["run_id"].max())


def _accepted_prediction_run_id(run_frame: pd.DataFrame,
                                snapshot_frame: pd.DataFrame) -> str:
    """Newest terminal prediction-bearing run, distinct from latest attempt."""
    if snapshot_frame.empty or "run_id" not in snapshot_frame.columns:
        return ""
    snapshot_ids = set(
        value for value in snapshot_frame["run_id"].fillna("").astype(str) if value
    )
    if not run_frame.empty and "run_id" in run_frame.columns:
        candidates = run_frame[
            run_frame["run_id"].fillna("").astype(str).isin(snapshot_ids)
        ].copy()
        if "run_kind" in candidates.columns:
            kinds = candidates["run_kind"].fillna("").astype(str)
            candidates = candidates[kinds.isin(["", "prediction_pipeline"])]
        if "status" in candidates.columns:
            statuses = candidates["status"].fillna("").astype(str).str.lower()
            candidates = candidates[statuses.isin(["success", "partial"])]
        if not candidates.empty:
            return _latest_run_id(candidates)

    # Bootstrap old mirrors whose lifecycle rows were never terminally updated.
    # Explicit failed/no-data terminal rows are never prediction sources merely
    # because they happened to write a snapshot before failing.
    fallback = snapshot_frame[
        snapshot_frame["run_id"].fillna("").astype(str).isin(snapshot_ids)
    ].copy()
    if (not run_frame.empty and "run_id" in run_frame.columns
            and "status" in run_frame.columns):
        lifecycle = run_frame[
            run_frame["run_id"].fillna("").astype(str).isin(snapshot_ids)
        ].copy()
        known_ids = set(lifecycle["run_id"].fillna("").astype(str))
        bootstrap_ids = set(
            lifecycle.loc[
                lifecycle["status"].fillna("").astype(str).str.lower().isin(["", "running"]),
                "run_id",
            ].fillna("").astype(str)
        )
        bootstrap_ids.update(snapshot_ids - known_ids)
        fallback = fallback[
            fallback["run_id"].fillna("").astype(str).isin(bootstrap_ids)
        ]
    if "logged_at" in fallback.columns:
        fallback["_logged"] = pd.to_datetime(
            fallback["logged_at"], errors="coerce", utc=True, format="mixed"
        )
        fallback = fallback.sort_values(["_logged", "run_id"], kind="stable")
    return str(fallback.iloc[-1]["run_id"]) if not fallback.empty else ""


def _build_model_metrics(sync_id: str, pred_log: pd.DataFrame | None = None,
                         shadow_log: pd.DataFrame | None = None,
                         feature_log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Materialize authoritative live metrics through evaluation's one math path."""
    from evaluation import cohorts
    from evaluation.ledger import LIVE_COLUMNS, build_live_ledger

    # Scan local immutable lineage even when the scoring cohort comes from the
    # remote+local merge. This is fail-closed on corrupt lineage and contributes
    # IDs from per-run feature files not duplicated in feature_vectors.csv.
    local_pred = cohorts.load_prediction_log(os.path.dirname(__file__))
    verified_mask = local_pred.get(
        "feature_snapshot_verified", pd.Series(False, index=local_pred.index)
    ).fillna(False).astype(bool)
    verified_ids = set(
        local_pred.loc[verified_mask, "feature_snapshot_id"].fillna("").astype(str)
    ) if "feature_snapshot_id" in local_pred.columns else set()
    remote_evidence: dict[str, str] = {}
    remote_invalid: set[str] = set()
    if feature_log is not None and "feature_snapshot_id" in feature_log.columns:
        remote_evidence, remote_invalid = cohorts.verify_feature_frame(feature_log)
        verified_ids.update(remote_evidence)
        verified_ids.difference_update(remote_invalid)

    if pred_log is None:
        pred_log = local_pred
    else:
        pred_log = pred_log.copy()
        ids = pred_log.get(
            "feature_snapshot_id", pd.Series("", index=pred_log.index)
        ).fillna("").astype(str)
        expected_hash = pred_log.get(
            "feature_vector_sha256", pd.Series("", index=pred_log.index)
        ).fillna("").astype(str).str.strip()
        evidence_hash = ids.map(remote_evidence if feature_log is not None else {})
        local_verified = ids.isin(verified_ids)
        pred_log["feature_snapshot_verified"] = (
            ids.ne("") & local_verified
            & (expected_hash.eq("") | evidence_hash.fillna("").eq(expected_hash)
               | evidence_hash.isna())
        )
    if shadow_log is None:
        shadow_log = cohorts.load_shadow_log(os.path.dirname(__file__))
    scored = cohorts.build_scored_frame(pred_log, shadow_log)
    frame = build_live_ledger(scored)
    if frame.empty:
        frame = pd.DataFrame(columns=LIVE_COLUMNS)
    frame["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    frame["metric_source"] = "production.evaluation.ledger"
    frame["dashboard_row_key"] = (
        frame.get("tier", pd.Series("", index=frame.index)).astype(str)
        + ":"
        + frame.get("model", pd.Series("", index=frame.index)).astype(str)
    )
    frame["sync_id"] = sync_id
    return frame


def _write_manifest(cur, *, sync_id: str, status: str,
                    latest_attempt_run_id: str, accepted_prediction_run_id: str,
                    counts: dict[str, int], missing_files: list[str], error: str = "") -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dash_sync_manifest (
            sync_id text PRIMARY KEY,
            published_at text NOT NULL,
            status text NOT NULL,
            pipeline_run_id text,
            table_counts_json text NOT NULL,
            missing_files_json text NOT NULL,
            error_message text
        )
    """)
    cur.execute("ALTER TABLE dash_sync_manifest ADD COLUMN IF NOT EXISTS latest_attempt_run_id text")
    cur.execute("ALTER TABLE dash_sync_manifest ADD COLUMN IF NOT EXISTS accepted_prediction_run_id text")
    cur.execute(
        """INSERT INTO dash_sync_manifest
           (sync_id,published_at,status,pipeline_run_id,latest_attempt_run_id,
            accepted_prediction_run_id,table_counts_json,missing_files_json,error_message)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            sync_id, datetime.now(timezone.utc).isoformat(timespec="microseconds"), status,
            accepted_prediction_run_id, latest_attempt_run_id,
            accepted_prediction_run_id, json.dumps(counts, sort_keys=True),
            json.dumps(missing_files), error,
        ),
    )
    _enable_public_read(cur, "dash_sync_manifest")


def sync_dashboard_tables(verbose: bool = True) -> dict[str, int]:
    """Publish one additive, all-or-nothing dashboard generation."""
    sync_id = f"sync_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    planned: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    counts: dict[str, int] = {}
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                # Two overlapping runners may both plan from the same old
                # snapshot. Serialize the read/merge/replace transaction so the
                # second publisher includes the first publisher's new rows.
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                            ("betting-algo-operational-state",))
                # Plan every table before mutating any of them. Any exception
                # rolls back the complete publication, preventing mixed eras.
                for spec in STATE_SPECS:
                    path = spec.path()
                    incoming = (
                        _load_feature_state(path)
                        if spec.table == "dash_features"
                        else load_csv(path)
                    )
                    existing = read_table(cur, spec.table)
                    if not path.exists():
                        missing_files.append(str(path.relative_to(path.parents[1])))
                    merged = merge_state_frames(existing, incoming, spec)
                    publish = add_row_keys(merged, spec)
                    publish["sync_id"] = sync_id
                    planned[spec.table] = publish
                    counts[spec.table] = len(publish)

                # Performance is derived, not independently calculated in the
                # browser. This table uses evaluation.metrics/roi via the ledger.
                model_metrics = _build_model_metrics(
                    sync_id,
                    pred_log=planned.get("dash_predictions"),
                    shadow_log=planned.get("dash_shadow"),
                    feature_log=planned.get("dash_features"),
                )
                planned["dash_model_metrics"] = model_metrics
                counts["dash_model_metrics"] = len(model_metrics)

                for table, frame in planned.items():
                    _replace_table(cur, table, frame)
                    _enable_public_read(cur, table)

                run_frame = planned.get("dash_runs", pd.DataFrame())
                snapshot_frame = planned.get("dash_snapshots", pd.DataFrame())
                status = "degraded" if missing_files else "success"
                _write_manifest(
                    cur, sync_id=sync_id, status=status,
                    latest_attempt_run_id=_latest_run_id(run_frame),
                    accepted_prediction_run_id=_accepted_prediction_run_id(
                        run_frame, snapshot_frame
                    ),
                    counts=counts,
                    missing_files=missing_files,
                )
            conn.commit()
    except Exception:
        # The accepted generation remains unchanged because the transaction is
        # atomic. Absence of a new manifest is itself an honest stale signal.
        raise

    if verbose:
        summary = ", ".join(f"{table}={count}" for table, count in counts.items())
        state = "degraded" if missing_files else "accepted"
        print(f"   📊 Dashboard sync {state} ({sync_id}): {summary}")
        if missing_files:
            print(f"   ⚠️ missing local state files: {', '.join(missing_files)}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydrate", action="store_true",
                        help="merge durable dashboard state into local CSVs before a runner starts")
    args = parser.parse_args()
    if args.hydrate:
        hydrate_operational_state()
    else:
        sync_dashboard_tables()


if __name__ == "__main__":
    main()
