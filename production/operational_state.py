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
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shutil
import tempfile

import pandas as pd

from operations.operational_lock import operational_csv_lock
from settlement_attribution import (
    ATTRIBUTION_QUALITY_EXACT_UID,
    ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED,
    SETTLEMENT_QUALITY_EXACT_UID,
    bet_accounting_matches_outcome,
)


BASE = Path(__file__).resolve().parent
CONTROL_COLUMNS = {"dashboard_row_key", "sync_id"}
IDENTITY_TERMINAL_STATUSES = {"identity_conflict", "superseded_identity"}
BET_TERMINAL_STATUSES = {
    "settled", "void", "voided", "cancel", "cancelled", "canceled",
}
BET_TERMINAL_OUTCOMES = {
    "win", "loss", "void", "voided", "cancel", "cancelled", "canceled",
}
BET_CANONICAL_STATUSES = {"pending", "settled", "void", "cancelled"}
BET_ATTRIBUTION_FIELDS = (
    "settlement_quality", "attribution_quality", "metric_eligible",
    "result_evidence_kind", "result_evidence_sha256",
)
BET_EXPOSURE_FIELDS = (
    "match", "match_uid", "feature_snapshot_id", "run_id", "bet_on",
    "bet_on_player1", "stake", "odds_decimal",
)
BET_TERMINAL_FIELDS = (
    "status", "outcome", "actual_profit", "bankroll_after",
    "settled_timestamp",
)
BET_NUMERIC_FIELDS = frozenset({
    "stake", "odds_decimal", "actual_profit", "bankroll_after",
})
# Pandas' CSV parser and Postgres text round trips can choose adjacent binary64
# representations for the same source value.  Preserve the fail-closed contract
# for meaningful changes while treating only sub-trillionth transport noise as
# equivalent.  At ordinary bankroll sizes this is several orders of magnitude
# tighter than one cent.
BET_NUMERIC_ABS_TOLERANCE = Decimal("1e-12")
BET_NUMERIC_REL_TOLERANCE = Decimal("1e-15")
BET_NUMERIC_MAX_TOLERANCE = Decimal("1e-9")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    StateSpec("dash_kalshi_odds_history", "kalshi_odds_history.csv",
              ("kalshi_observation_uid",),
              ("run_id", "polled_at", "market_ticker"), "strict_immutable"),
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


def _canonical_bet_value(field: str, value) -> str:
    """Normalize syntax; numeric transport tolerance is applied separately."""
    text = _clean(value)
    if not text:
        return ""
    lower = text.lower()
    if field in {"metric_eligible", "bet_on_player1"}:
        if lower in {"true", "1", "1.0", "yes", "y"}:
            return "true"
        if lower in {"false", "0", "0.0", "no", "n"}:
            return "false"
        return lower
    if field in {"stake", "odds_decimal", "actual_profit", "bankroll_after"}:
        try:
            number = Decimal(text)
        except InvalidOperation:
            return lower
        if not number.is_finite():
            return lower
        if number == 0:
            return "0"
        return format(number.normalize(), "f")
    if field == "settled_timestamp":
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed.isoformat()
    if field in {"status", "outcome"}:
        aliases = {
            "cancel": "cancelled",
            "canceled": "cancelled",
            "voided": "void",
        }
        return aliases.get(lower, lower)
    if field in {
        "settlement_quality", "attribution_quality", "result_evidence_kind",
        "result_evidence_sha256", "match_uid", "feature_snapshot_id", "run_id",
    }:
        return lower
    return lower


def _explicit_bet_values(group: pd.DataFrame, field: str) -> set[str]:
    if field not in group.columns:
        return set()
    return {
        canonical
        for canonical in group[field].map(
            lambda value: _canonical_bet_value(field, value)
        )
        if canonical
    }


def _bet_values_conflict(field: str, values: set[str]) -> bool:
    """Return whether explicit values disagree beyond transport precision."""
    if len(values) <= 1:
        return False
    if field not in BET_NUMERIC_FIELDS:
        return True
    try:
        numbers = [Decimal(value) for value in values]
    except InvalidOperation:
        return True
    if any(not number.is_finite() for number in numbers):
        return True
    magnitude = max((abs(number) for number in numbers), default=Decimal("0"))
    tolerance = min(
        BET_NUMERIC_MAX_TOLERANCE,
        max(
            BET_NUMERIC_ABS_TOLERANCE,
            BET_NUMERIC_REL_TOLERANCE * magnitude,
        ),
    )
    return max(numbers) - min(numbers) > tolerance


def _finite_decimal(value) -> Decimal | None:
    text = _clean(value)
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _validate_bet_row(row: pd.Series, row_key_value: str) -> None:
    """Validate one exposure/accounting row before any quality reconciliation."""
    status = _canonical_bet_value("status", row.get("status", ""))
    outcome = _canonical_bet_value("outcome", row.get("outcome", ""))
    metric = _canonical_bet_value(
        "metric_eligible", row.get("metric_eligible", "")
    )
    settlement = _canonical_bet_value(
        "settlement_quality", row.get("settlement_quality", "")
    )
    attribution = _canonical_bet_value(
        "attribution_quality", row.get("attribution_quality", "")
    )
    kind = _canonical_bet_value(
        "result_evidence_kind", row.get("result_evidence_kind", "")
    )
    digest = _canonical_bet_value(
        "result_evidence_sha256", row.get("result_evidence_sha256", "")
    )
    bundle_present = any((metric, settlement, attribution, kind, digest))

    if status not in BET_CANONICAL_STATUSES:
        raise RuntimeError(
            f"invalid canonical bet status for {row_key_value}: "
            f"{status or '<blank>'}"
        )

    if status == "pending":
        if bundle_present:
            raise RuntimeError(
                f"nonterminal bet carries attribution bundle for {row_key_value}"
            )
        if any(
            _clean(row.get(field, ""))
            for field in (
                "outcome", "actual_profit", "bankroll_after", "settled_timestamp",
            )
        ):
            raise RuntimeError(
                f"pending bet carries terminal state for {row_key_value}"
            )
        if "stake" in row.index or "odds_decimal" in row.index:
            stake = _finite_decimal(row.get("stake"))
            odds = _finite_decimal(row.get("odds_decimal"))
            if stake is None or stake <= 0 or odds is None or odds <= 1:
                raise RuntimeError(
                    f"pending bet has invalid exposure arithmetic for {row_key_value}"
                )
        return

    if status == "settled":
        if outcome not in {"win", "loss"}:
            raise RuntimeError(
                f"settled bet has invalid outcome for {row_key_value}"
            )
        accounting_fields = {"stake", "odds_decimal", "actual_profit"}
        if accounting_fields & set(row.index) and not bet_accounting_matches_outcome(row):
            raise RuntimeError(
                f"settled bet has invalid P&L arithmetic for {row_key_value}"
            )
        if (
            "bankroll_after" in row.index
            and _finite_decimal(row.get("bankroll_after")) is None
        ):
            raise RuntimeError(
                f"settled bet has invalid bankroll for {row_key_value}"
            )
        if "settled_timestamp" in row.index:
            timestamp = pd.to_datetime(
                _clean(row.get("settled_timestamp")), errors="coerce", utc=True,
            )
            if pd.isna(timestamp):
                raise RuntimeError(
                    f"settled bet has invalid timestamp for {row_key_value}"
                )
    elif status in {"void", "cancelled"}:
        if outcome != status:
            raise RuntimeError(
                f"void/cancelled bet has incoherent outcome for {row_key_value}"
            )
        profit = _finite_decimal(row.get("actual_profit"))
        if "actual_profit" in row.index and (profit is None or profit != 0):
            raise RuntimeError(
                f"void/cancelled bet is not refunded for {row_key_value}"
            )
        if (
            "bankroll_after" in row.index
            and _finite_decimal(row.get("bankroll_after")) is None
        ):
            raise RuntimeError(
                f"void/cancelled bet has invalid bankroll for {row_key_value}"
            )
        if "settled_timestamp" in row.index:
            timestamp = pd.to_datetime(
                _clean(row.get("settled_timestamp")), errors="coerce", utc=True,
            )
            if pd.isna(timestamp):
                raise RuntimeError(
                f"void/cancelled bet has invalid timestamp for {row_key_value}"
            )
        if bundle_present and metric != "false":
            raise RuntimeError(
                f"void/cancelled bet cannot be metric eligible for "
                f"{row_key_value}"
            )
    if not bundle_present:
        # Historical settled rows without any assertion remain repairable legacy
        # unknowns; every partial assertion below fails closed.
        return
    if bool(kind) != bool(digest) or (
        digest and not _SHA256_RE.fullmatch(digest)
    ):
        raise RuntimeError(
            f"invalid terminal bet result evidence bundle for {row_key_value}"
        )
    if metric == "":
        if (
            settlement != SETTLEMENT_QUALITY_EXACT_UID
            or attribution != ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED
        ):
            raise RuntimeError(
                f"invalid repairable bet attribution state for {row_key_value}"
            )
        return
    if metric not in {"true", "false"} or not settlement or not attribution:
        raise RuntimeError(
            f"incomplete terminal bet attribution bundle for {row_key_value}"
        )
    if metric == "true" and (
        settlement != SETTLEMENT_QUALITY_EXACT_UID
        or attribution != ATTRIBUTION_QUALITY_EXACT_UID
        or not kind
        or not digest
    ):
        raise RuntimeError(
            f"unproven terminal bet metric eligibility for {row_key_value}"
        )


def _validate_bet_group(group: pd.DataFrame, row_key_value: str) -> None:
    """Reject contradictory immutable exposure, attribution, or P&L facts."""
    for _, row in group.iterrows():
        _validate_bet_row(row, row_key_value)

    for field in (
        *BET_EXPOSURE_FIELDS,
        "settlement_quality", "result_evidence_kind", "result_evidence_sha256",
    ):
        values = _explicit_bet_values(group, field)
        if _bet_values_conflict(field, values):
            raise RuntimeError(
                f"conflicting immutable bet {field} for {row_key_value}: "
                f"{sorted(values)}"
            )

    metrics = _explicit_bet_values(group, "metric_eligible")
    if len(metrics) > 1:
        raise RuntimeError(
            f"conflicting immutable bet metric_eligible for {row_key_value}: "
            f"{sorted(metrics)}"
        )
    attributions = _explicit_bet_values(group, "attribution_quality")
    if len(attributions) > 1:
        allowed_upgrade = attributions == {
            ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED,
            ATTRIBUTION_QUALITY_EXACT_UID,
        }
        settlements = _explicit_bet_values(group, "settlement_quality")
        if (
            not allowed_upgrade
            or "false" in metrics
            or settlements != {SETTLEMENT_QUALITY_EXACT_UID}
        ):
            raise RuntimeError(
                f"conflicting immutable bet attribution_quality for "
                f"{row_key_value}: {sorted(attributions)}"
            )

    status = group.get(
        "status", pd.Series("", index=group.index)
    ).map(_clean).str.lower()
    outcome = group.get(
        "outcome", pd.Series("", index=group.index)
    ).map(_clean).str.lower()
    terminal = group[
        status.isin(BET_TERMINAL_STATUSES)
        | outcome.isin(BET_TERMINAL_OUTCOMES)
    ]
    for field in BET_TERMINAL_FIELDS:
        values = _explicit_bet_values(terminal, field)
        if _bet_values_conflict(field, values):
            raise RuntimeError(
                f"conflicting terminal bet {field} for {row_key_value}: "
                f"{sorted(values)}"
            )

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
        terminal = (
            status.isin(BET_TERMINAL_STATUSES)
            | outcome.isin(BET_TERMINAL_OUTCOMES)
        )
        metric = frame.get(
            "metric_eligible", pd.Series("", index=frame.index)
        ).map(_clean).str.lower()
        classified = metric.isin({
            "true", "1", "yes", "y", "false", "0", "no", "n"
        })
        has_attribution = frame.get(
            "attribution_quality", pd.Series("", index=frame.index)
        ).map(_clean).ne("")
        has_settlement = frame.get(
            "settlement_quality", pd.Series("", index=frame.index)
        ).map(_clean).ne("")
        has_evidence_kind = frame.get(
            "result_evidence_kind", pd.Series("", index=frame.index)
        ).map(_clean).ne("")
        has_evidence_hash = frame.get(
            "result_evidence_sha256", pd.Series("", index=frame.index)
        ).map(_clean).str.fullmatch(r"[0-9a-fA-F]{64}")
        # Attribution enrichment is monotonic durable state. A stale local
        # settled copy with blank classification must not erase a repaired
        # Supabase row merely because both share the same terminal outcome.
        score = (
            terminal.astype(int) * 100
            + classified.astype(int) * 40
            + has_settlement.astype(int) * 8
            + has_attribution.astype(int) * 4
            + has_evidence_kind.astype(int) * 2
            + has_evidence_hash.astype(int)
        )
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
        {
            "settled", "void", "cancelled", "canceled",
            *IDENTITY_TERMINAL_STATUSES,
        }
    )
    # Identity terminals are safety tombstones, not ordinary freshness-based
    # outcomes.  Once any durable copy records an identity conflict or
    # supersession, a later/staler settlement copy must not make the match
    # decision-eligible again.  Only an explicit identity-resolution workflow
    # may clear these states.
    combined["_terminal_priority"] = (
        combined["_is_terminal"].astype(int) * 100
        + status.isin(IDENTITY_TERMINAL_STATUSES).astype(int) * 100
    )
    terminal_fields = (
        "actual_winner", "score", "settled_at", "model_correct",
        "market_correct", "xgb_correct", "rf_correct", "record_status",
        "record_note",
    )
    identity_terminal_fields = (
        "identity_status", "identity_event_key",
        "identity_related_match_uid", "identity_conflict_fields",
        "features_complete", "defaulted_features",
    )
    rows: list[pd.Series] = []
    for _, group in combined.groupby("_row_key", sort=False):
        inference = group.sort_values(
            ["_inference_quality", "_freshness", "_source_order"], kind="stable"
        ).iloc[-1].copy()
        terminal = group[group["_is_terminal"]]
        if not terminal.empty:
            terminal_row = terminal.sort_values(
                ["_terminal_priority", "_freshness", "_source_order"],
                kind="stable",
            ).iloc[-1]
            terminal_status = _clean(terminal_row.get("record_status", "")).lower()
            fields = terminal_fields
            if terminal_status in IDENTITY_TERMINAL_STATUSES:
                fields = (*fields, *identity_terminal_fields)
            for field in fields:
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
        "logged_at", "polled_at", "timestamp", "started_at", "start_time",
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
    if spec.quality_mode == "bet":
        for row_key_value, group in combined.groupby("_row_key", sort=False):
            _validate_bet_group(group, row_key_value)
    if spec.quality_mode == "strict_immutable":
        for row_key_value, group in combined.groupby("_row_key", sort=False):
            for field in ordered_columns:
                values = {
                    _clean(value) for value in group[field] if _clean(value)
                }
                if len(values) > 1:
                    raise RuntimeError(
                        f"conflicting immutable {spec.table} {field} for "
                        f"{row_key_value}: {sorted(values)}"
                    )
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


def _hydrate_operational_state_unlocked(
    base: Path = BASE, verbose: bool = True
) -> dict[str, int]:
    """Merge the durable Supabase snapshot into local CSVs before a cloud run."""
    from canonical_store import connect
    from dashboard_sync import (
        _load_feature_state, _merge_feature_state, _table_exists, read_table,
    )

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
                local = (
                    _load_feature_state(path)
                    if spec.table == "dash_features"
                    else load_csv(path)
                )
                merged = (
                    _merge_feature_state(remote, local)
                    if spec.table == "dash_features"
                    else merge_state_frames(remote, local, spec)
                )
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


def hydrate_operational_state(base: Path = BASE, verbose: bool = True) -> dict[str, int]:
    """Hydrate while excluding paper-account writers and manual reconciliation."""
    with operational_csv_lock(base / "logs"):
        return _hydrate_operational_state_unlocked(base=base, verbose=verbose)
