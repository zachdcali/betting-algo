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
from time import sleep
import uuid

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canonical_store import connect  # noqa: E402
from feature_lineage import (  # noqa: E402
    expected_feature_hash_matches,
    load_production_feature_lineage,
    parse_feature_occurrence,
    read_feature_csv,
    resolve_feature_lineage,
    validate_derived_feature_copy,
    validate_derived_projection_authority,
)
from operational_state import (  # noqa: E402
    STATE_SPECS, add_row_keys, hydrate_operational_state, load_csv,
    merge_state_frames,
)


RETRYABLE_PUBLICATION_SQLSTATES = frozenset({"40P01", "40001"})
PUBLICATION_RETRY_DELAYS_SECONDS = (1.0, 3.0)

# The public dashboard filters every projection by its manifest sync_id and
# then reads in a stable order. These indexes survive TRUNCATE/COPY generations
# and prevent the browser's exact-count probes from competing through repeated
# full scans and sorts as the immutable shadow/ROC histories grow.
DASHBOARD_QUERY_INDEX_SPECS = {
    "dash_predictions": (
        ("sync_id", ""), ("match_date", ""), ("p1", ""), ("p2", "")
    ),
    "dash_odds_history": (("sync_id", ""), ("logged_at", "DESC NULLS LAST")),
    "dash_kalshi_odds_history": (
        ("sync_id", ""), ("polled_at", "DESC NULLS LAST")
    ),
    "dash_shadow": (("sync_id", ""), ("logged_at", "DESC NULLS LAST")),
    "dash_runs": (("sync_id", ""), ("started_at", "DESC NULLS LAST")),
    "dash_bets": (("sync_id", ""), ("timestamp", "DESC NULLS LAST")),
    "dash_snapshots": (("sync_id", ""), ("logged_at", "DESC NULLS LAST")),
    "dash_skipped_live_matches": (
        ("sync_id", ""), ("logged_at", "DESC NULLS LAST")
    ),
    "dash_settlement_audit": (
        ("sync_id", ""), ("logged_at", "DESC NULLS LAST")
    ),
    "dash_features": (("sync_id", ""), ("logged_at", "DESC NULLS LAST")),
    "dash_bankroll": (("sync_id", ""), ("timestamp", "DESC NULLS LAST")),
    "dash_sessions": (("sync_id", ""), ("start_time", "DESC NULLS LAST")),
    "dash_model_metrics": (
        ("sync_id", ""), ("tier", ""), ("log_loss", "NULLS LAST"),
        ("model", "")
    ),
    "dash_model_calibration": (
        ("sync_id", ""), ("tier", ""), ("model", ""), ("bin_index", "")
    ),
    "dash_model_roc": (("sync_id", ""), ("roc_row_key", "")),
}


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
    """Rebuild derived drill-down rows from immutable per-run authorities."""
    from feature_vector_log import COLS
    from features.performance_v1 import PERFORMANCE_FEATURES
    from models.inference import EXACT_141_FEATURES

    aggregate = load_csv(aggregate_path).reindex(columns=COLS, fill_value="")
    immutable_sources: list[tuple[str, pd.DataFrame]] = []
    for path in sorted(glob(str(aggregate_path.parent / "features_*.csv"))):
        run_frame = read_feature_csv(path)
        if "feature_snapshot_id" in run_frame.columns:
            immutable_sources.append((os.path.basename(path), run_frame))
    resolution = resolve_feature_lineage(
        ordered_names=EXACT_141_FEATURES,
        immutable_sources=immutable_sources,
        derived_sources=[(aggregate_path.name, aggregate)],
    )

    # Blank-ID legacy rows have no immutable join key. Preserve them exactly;
    # every exact-ID row is reconstructed from the resolved authority below.
    blank_ids = aggregate["feature_snapshot_id"].fillna("").astype(str).str.strip().eq("")
    output = aggregate.loc[blank_ids, COLS].copy()
    rows: list[dict[str, object]] = []
    feature_names = [*EXACT_141_FEATURES, *PERFORMANCE_FEATURES]
    for feature_id, authority in sorted(resolution.canonical_by_id.items()):
        source = authority.row
        if authority.source_kind == "derived":
            validate_derived_projection_authority(authority)
            rows.append({column: source.get(column, "") for column in COLS})
            continue

        payload = {
            name: _json_scalar(source.get(name))
            for name in feature_names if name in source and pd.notna(source.get(name))
        }
        defaults = source.get("meta_defaulted_features", "")
        payload["_defaulted_features"] = "" if pd.isna(defaults) else str(defaults)
        build_status = authority.status or "unknown"

        def hand(prefix: str) -> str:
            for label in ("U", "L", "R", "A"):
                value = pd.to_numeric(source.get(f"{prefix}_Hand_{label}"), errors="coerce")
                if pd.notna(value) and float(value) == 1.0:
                    return label
            return ""

        rows.append({
            "p1": source.get("player1_raw", ""),
            "p2": source.get("player2_raw", ""),
            "match_date": source.get("meta_match_date", ""),
            "logged_at": source.get("timestamp", source.get("run_started_at", "")),
            "run_id": authority.run_id,
            "match_uid": authority.match_uid,
            "feature_snapshot_id": feature_id,
            "build_status": build_status,
            "features_complete": (
                build_status == "ok"
                and not payload["_defaulted_features"]
                and authority.structurally_verified
            ),
            "p1_hand": hand("P1"),
            "p2_hand": hand("P2"),
            "feature_schema_sha256": authority.schema_sha256,
            "feature_vector_sha256": authority.verified_vector_sha256,
            "feature_count": len(EXACT_141_FEATURES),
            "features_json": json.dumps(
                payload, separators=(",", ":"), default=str, allow_nan=False
            ),
        })
    if rows:
        output = pd.concat([output, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
    output = output.reindex(columns=COLS, fill_value="")
    output.attrs["immutable_feature_ids"] = set(resolution.immutable_ids)
    return output


def _merge_feature_state(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge rows with immutable-local, then durable-remote precedence.

    A clean clone may contain an older aggregate but no additional private run
    file.  In that case the accepted durable projection is the repair source
    and must not be overwritten merely because the local frame was appended
    second.  A locally tracked immutable row outranks both after compatibility
    validation.
    """
    from feature_vector_log import COLS
    from models.inference import EXACT_141_FEATURES

    authoritative_ids = set(incoming.attrs.get("immutable_feature_ids", set()))
    existing_ids = existing.get(
        "feature_snapshot_id", pd.Series("", index=existing.index)
    ).fillna("").astype(str).str.strip()
    incoming_ids = incoming.get(
        "feature_snapshot_id", pd.Series("", index=incoming.index)
    ).fillna("").astype(str).str.strip()

    # Accepted durable state is an authority during clean-clone recovery, so
    # validate it before a generic merge can collapse duplicate IDs or carry an
    # invalid exactness claim into the hydrated CSV.
    durable_authorities: dict[str, pd.Series] = {}
    for snapshot_id in sorted(set(existing_ids) - {""}):
        durable_rows = existing.loc[existing_ids.eq(snapshot_id)]
        if len(durable_rows) != 1:
            raise RuntimeError(
                f"durable feature {snapshot_id!r} has {len(durable_rows)} projection rows"
            )
        durable_row = durable_rows.iloc[0]
        parsed = parse_feature_occurrence(
            durable_row,
            EXACT_141_FEATURES,
            source_kind="derived",
            source_file="durable dash_features",
            source_row=2,
        )
        if parsed is None:  # pragma: no cover - guarded by nonblank ID filter
            raise RuntimeError(
                f"durable feature {snapshot_id!r} lost its projection identity"
            )
        validate_derived_projection_authority(parsed)
        durable_authorities[snapshot_id] = durable_row

    for snapshot_id in sorted(authoritative_ids):
        authorities = incoming.loc[incoming_ids.eq(snapshot_id)]
        if len(authorities) != 1:
            raise RuntimeError(
                f"resolved immutable feature {snapshot_id!r} has {len(authorities)} projection rows"
            )
        authority = authorities.iloc[0]
        for _, candidate in existing.loc[existing_ids.eq(snapshot_id)].iterrows():
            validate_derived_feature_copy(
                authority,
                candidate,
                EXACT_141_FEATURES,
                snapshot_id=snapshot_id,
                authority_source="local immutable reconstruction",
                candidate_source="durable dash_features",
                require_valid_authority=False,
            )

    # For IDs without a local immutable authority, a previously accepted
    # durable row is the repair authority. Validate a stale local aggregate as
    # an equivalent copy, then retain the durable vector/hash.
    durable_ids: set[str] = set()
    for snapshot_id in sorted(set(durable_authorities) - authoritative_ids):
        local_copies = incoming.loc[incoming_ids.eq(snapshot_id)]
        authority = durable_authorities[snapshot_id]
        for _, candidate in local_copies.iterrows():
            validate_derived_feature_copy(
                authority,
                candidate,
                EXACT_141_FEATURES,
                snapshot_id=snapshot_id,
                authority_source="durable dash_features",
                candidate_source="local aggregate",
            )
        durable_ids.add(snapshot_id)

    spec = next(spec for spec in STATE_SPECS if spec.table == "dash_features")
    merged = merge_state_frames(existing, incoming, spec)
    merged_ids = merged.get(
        "feature_snapshot_id", pd.Series("", index=merged.index)
    ).fillna("").astype(str).str.strip()
    replacement_ids = authoritative_ids | durable_ids
    merged = merged.loc[~merged_ids.isin(replacement_ids)]
    authorities = pd.concat([
        incoming.loc[incoming_ids.isin(authoritative_ids)],
        existing.loc[existing_ids.isin(durable_ids)],
    ], ignore_index=True)
    columns = list(dict.fromkeys([*merged.columns, *authorities.columns]))
    result = pd.concat([
        merged.reindex(columns=columns, fill_value=""),
        authorities.reindex(columns=columns, fill_value=""),
    ], ignore_index=True)
    result = result.reindex(columns=COLS, fill_value="")
    result.attrs["immutable_feature_ids"] = authoritative_ids
    return result


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


def _ensure_query_index(cur, table: str, frame: pd.DataFrame) -> None:
    index_spec = DASHBOARD_QUERY_INDEX_SPECS.get(table)
    index_columns = {column for column, _order in index_spec or ()}
    if not index_spec or not index_columns.issubset(frame.columns):
        return
    index_name = f"{table}_generation_read_v2_idx"
    columns = ", ".join(
        f'"{column}"{f" {order}" if order else ""}'
        for column, order in index_spec
    )
    cur.execute(
        f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}" ({columns})'
    )


def _replace_table(cur, table: str, frame: pd.DataFrame) -> None:
    _ensure_schema(cur, table, frame)
    _ensure_query_index(cur, table, frame)
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
                         feature_log: pd.DataFrame | None = None,
                         odds_history: pd.DataFrame | None = None,
                         kalshi_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Materialize authoritative live metrics through evaluation's one math path."""
    from evaluation import cohorts
    from evaluation.ledger import LIVE_COLUMNS, build_live_ledger
    from models.inference import EXACT_141_FEATURES

    # Scan local immutable lineage even when the scoring cohort comes from the
    # remote+local merge. This is fail-closed on corrupt lineage and contributes
    # IDs from per-run feature files not duplicated in feature_vectors.csv.
    local_pred = cohorts.load_prediction_log(os.path.dirname(__file__))
    local_lineage = load_production_feature_lineage(
        os.path.dirname(__file__), EXACT_141_FEATURES
    )
    referential_hashes = {
        snapshot_id: set(local_lineage.referential_vector_hashes(snapshot_id))
        for snapshot_id in local_lineage.canonical_by_id
        if snapshot_id not in local_lineage.invalid_ids
    }
    authority_contracts = {
        snapshot_id: cohorts.feature_identity_contract(
            occurrence.row,
            match_uid=occurrence.match_uid,
            run_id=occurrence.run_id,
        )
        for snapshot_id, occurrence in local_lineage.canonical_by_id.items()
        if snapshot_id not in local_lineage.invalid_ids
    }
    identity_invalid = set(local_lineage.invalid_ids)
    remote_evidence: dict[str, str] = {}
    remote_invalid: set[str] = set()
    if feature_log is not None and "feature_snapshot_id" in feature_log.columns:
        remote_evidence, remote_invalid = cohorts.verify_feature_frame(feature_log)
        for snapshot_id, vector_hash in remote_evidence.items():
            if snapshot_id not in remote_invalid:
                referential_hashes.setdefault(snapshot_id, set()).add(vector_hash)
        for _, feature_row in feature_log.iterrows():
            raw_snapshot_id = feature_row.get("feature_snapshot_id", "")
            snapshot_id = (
                "" if raw_snapshot_id is None or pd.isna(raw_snapshot_id)
                else str(raw_snapshot_id).strip()
            )
            if not snapshot_id or snapshot_id not in remote_evidence:
                continue
            contract = cohorts.feature_identity_contract(feature_row)
            if not all(contract.values()):
                continue
            existing = authority_contracts.get(snapshot_id)
            if existing is not None and existing != contract:
                identity_invalid.add(snapshot_id)
            else:
                authority_contracts[snapshot_id] = contract
        for snapshot_id in remote_invalid:
            referential_hashes.pop(snapshot_id, None)
            identity_invalid.add(snapshot_id)
    for snapshot_id in identity_invalid:
        authority_contracts.pop(snapshot_id, None)

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
        hash_verified = pd.Series([
            bool(referential_hashes.get(snapshot_id))
            and expected_feature_hash_matches(
                expected, referential_hashes.get(snapshot_id, set())
            )
            for snapshot_id, expected in zip(ids, expected_hash)
        ], index=pred_log.index)
        identity_verified = cohorts.prediction_feature_identity_matches(
            pred_log, authority_contracts
        )
        pred_log["feature_snapshot_verified"] = (
            ids.ne("") & hash_verified & identity_verified
        )
    if shadow_log is None:
        shadow_log = cohorts.load_shadow_log(os.path.dirname(__file__))
    if kalshi_history is None:
        kalshi_history = cohorts.load_kalshi_history(os.path.dirname(__file__))
    scored = cohorts.build_scored_frame(
        pred_log, shadow_log, odds_history, kalshi_history,
    )
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
    frame.attrs["scored_frame"] = scored
    return frame


def _build_model_calibration(
    sync_id: str,
    pred_log: pd.DataFrame,
    shadow_log: pd.DataFrame | None,
    odds_history: pd.DataFrame | None,
    metrics_frame: pd.DataFrame,
    kalshi_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Materialize manifest-pinned reliability bins without browser math."""
    from evaluation import cohorts
    from evaluation.ledger import CALIBRATION_COLUMNS, build_calibration_ledger

    scored = metrics_frame.attrs.get("scored_frame")
    if not isinstance(scored, pd.DataFrame):
        scored = cohorts.build_scored_frame(
            pred_log, shadow_log, odds_history, kalshi_history,
        )
    live_columns = [column for column in metrics_frame.columns if column in {
        "model", "tier", "n", "accuracy", "auc", "log_loss", "brier", "ece",
        "cal_slope", "cal_intercept", "roi_flat", "n_bets_flat", "pnl_flat",
        "win_rate_flat", "roi_kelly", "n_bets_kelly", "pnl_kelly",
        "max_drawdown_kelly",
    }]
    calibration = build_calibration_ledger(scored, metrics_frame[live_columns])
    if calibration.empty:
        calibration = pd.DataFrame(columns=CALIBRATION_COLUMNS)
    calibration["generated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )
    calibration["calibration_row_key"] = (
        calibration.get("tier", pd.Series("", index=calibration.index)).astype(str)
        + ":" + calibration.get("model", pd.Series("", index=calibration.index)).astype(str)
        + ":" + calibration.get("bin_index", pd.Series("", index=calibration.index)).astype(str)
    )
    calibration["sync_id"] = sync_id
    return calibration


def _build_model_roc(
    sync_id: str,
    pred_log: pd.DataFrame,
    shadow_log: pd.DataFrame | None,
    odds_history: pd.DataFrame | None,
    metrics_frame: pd.DataFrame,
    kalshi_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Materialize manifest-pinned ROC points from the verified scored frame."""
    from evaluation import cohorts
    from evaluation.ledger import ROC_COLUMNS, build_roc_ledger

    scored = metrics_frame.attrs.get("scored_frame")
    if not isinstance(scored, pd.DataFrame):
        scored = cohorts.build_scored_frame(
            pred_log, shadow_log, odds_history, kalshi_history,
        )
    roc = build_roc_ledger(scored, metrics_frame)
    if roc.empty:
        roc = pd.DataFrame(columns=ROC_COLUMNS)
    roc["generated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )
    roc["roc_row_key"] = (
        roc.get("tier", pd.Series("", index=roc.index)).astype(str)
        + ":" + roc.get("model", pd.Series("", index=roc.index)).astype(str)
        + ":" + roc.get("point_index", pd.Series("", index=roc.index)).astype(str)
    )
    roc["sync_id"] = sync_id
    return roc


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


def _sync_dashboard_tables_once(verbose: bool = True) -> dict[str, int]:
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
                    merged = (
                        _merge_feature_state(existing, incoming)
                        if spec.table == "dash_features"
                        else merge_state_frames(existing, incoming, spec)
                    )
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
                    odds_history=planned.get("dash_odds_history"),
                    kalshi_history=planned.get("dash_kalshi_odds_history"),
                )
                planned["dash_model_metrics"] = model_metrics
                counts["dash_model_metrics"] = len(model_metrics)
                model_calibration = _build_model_calibration(
                    sync_id,
                    pred_log=planned.get("dash_predictions", pd.DataFrame()),
                    shadow_log=planned.get("dash_shadow"),
                    odds_history=planned.get("dash_odds_history"),
                    kalshi_history=planned.get("dash_kalshi_odds_history"),
                    metrics_frame=model_metrics,
                )
                planned["dash_model_calibration"] = model_calibration
                counts["dash_model_calibration"] = len(model_calibration)
                model_roc = _build_model_roc(
                    sync_id,
                    pred_log=planned.get("dash_predictions", pd.DataFrame()),
                    shadow_log=planned.get("dash_shadow"),
                    odds_history=planned.get("dash_odds_history"),
                    kalshi_history=planned.get("dash_kalshi_odds_history"),
                    metrics_frame=model_metrics,
                )
                planned["dash_model_roc"] = model_roc
                counts["dash_model_roc"] = len(model_roc)

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


def _publication_sqlstate(error: BaseException) -> str:
    """Return a retryable SQLSTATE from a wrapped database exception, if any."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        sqlstate = str(getattr(current, "sqlstate", "") or "").strip()
        if sqlstate:
            return sqlstate
        current = current.__cause__ or current.__context__
    return ""


def sync_dashboard_tables(verbose: bool = True) -> dict[str, int]:
    """Publish atomically, retrying only safe transaction-abort failures.

    PostgreSQL aborts the full transaction on a deadlock or serialization
    failure, so replaying the complete read/merge/replace operation is safe.
    All other failures still fail closed immediately.
    """
    attempts = len(PUBLICATION_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, attempts + 1):
        try:
            return _sync_dashboard_tables_once(verbose=verbose)
        except Exception as error:
            sqlstate = _publication_sqlstate(error)
            if (
                sqlstate not in RETRYABLE_PUBLICATION_SQLSTATES
                or attempt >= attempts
            ):
                raise
            delay = PUBLICATION_RETRY_DELAYS_SECONDS[attempt - 1]
            if verbose:
                print(
                    "   ⚠️ Durable dashboard publication transaction "
                    f"aborted (SQLSTATE {sqlstate}); retrying full publication "
                    f"in {delay:g}s ({attempt + 1}/{attempts})"
                )
            sleep(delay)

    raise AssertionError("unreachable publication retry state")


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
