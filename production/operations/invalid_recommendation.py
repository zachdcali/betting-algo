"""Fail-closed quarantine and refund for one reviewed bad-input recommendation.

This is deliberately not a generic operator override.  The only supported
case is a ranking identity collision proven against one exact prediction and
paper-bet lineage.  The operation tombstones the mutable operational
prediction, appends an audit row that blocks the accepted immutable Slate, and
only then administratively refunds the paper exposure.  Immutable prediction
and feature snapshots are evidence and are never rewritten.

Every write is replay-safe.  A crash after the prediction tombstone but before
the audit/refund is repaired by rerunning the exact same command; a different
terminal state or different operator detail fails closed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from audit_logger import SKIPPED_MATCH_COLUMNS
from logging_utils import atomic_write_csv, stable_hash, utc_now_iso
from operations.operational_lock import operational_csv_lock
from prediction_logger import COLUMNS as PREDICTION_COLUMNS
from utils.bet_tracker import (
    INVALID_RECOMMENDATION_REASON_CODES,
    INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
    BetTracker,
)


IDENTITY_TERMINAL_STATUSES = frozenset({
    "identity_conflict",
    "superseded_identity",
})
QUARANTINE_STAGE = "administrative_quarantine"
QUARANTINE_SKIP_REASON = "match_identity_conflict"
QUARANTINE_CONFLICT_FIELDS = (
    "p1_rank,p1_rank_points,ranking_player_identity"
)
QUARANTINE_DEFAULT_MARKERS = (
    "Player1_Rank=rank_identity_collision",
    "Player1_Rank_Points=rank_identity_collision",
)


@dataclass(frozen=True)
class QuarantineResult:
    match_uid: str
    prediction_uid: str
    prediction_changed: bool
    audit_appended: bool
    skip_event_id: str


@dataclass(frozen=True)
class RemediationResult:
    match_uid: str
    prediction_uid: str
    bet_id: str
    prediction_changed: bool
    audit_appended: bool
    bet_refunded: bool
    skip_event_id: str


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _false(value: Any) -> bool:
    return _clean(value).casefold() in {"false", "0", "0.0", "no", "n"}


def _normalize_detail(value: str) -> str:
    return " ".join(str(value or "").split())


def _append_default_markers(value: Any) -> str:
    tokens = [token.strip() for token in _clean(value).split(",") if token.strip()]
    for marker in QUARANTINE_DEFAULT_MARKERS:
        if marker not in tokens:
            tokens.append(marker)
    return ",".join(tokens)


def _quarantine_note(reason_code: str, detail: str) -> str:
    note = (
        "Administrative identity quarantine; underlying match result not "
        f"asserted; reason_code={reason_code}; immutable_snapshots_retained=true"
    )
    if detail:
        note += f"; detail={detail}"
    return note


def _audit_detail(reason_code: str, detail: str) -> str:
    value = (
        f"reason_code={reason_code}; bad ranking-row player identity; "
        "operational prediction tombstoned; immutable snapshots retained"
    )
    if detail:
        value += f"; detail={detail}"
    return value


def _read_csv_strings(path: Path, *, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if frame.columns.duplicated().any():
        raise RuntimeError(f"duplicate CSV columns are not safe to mutate: {path}")
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    ordered = [*columns, *(column for column in frame.columns if column not in columns)]
    return frame.loc[:, ordered]


def _read_optional_csv_strings(path: Path, *, columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    return _read_csv_strings(path, columns=columns)


def _exact_prediction_index(
    predictions: pd.DataFrame,
    *,
    expected_match_uid: str,
    expected_feature_snapshot_id: str,
    expected_run_id: str,
    expected_p1: str,
    expected_p2: str,
) -> int:
    expected = {
        "match_uid": _clean(expected_match_uid),
        "feature_snapshot_id": _clean(expected_feature_snapshot_id),
        "run_id": _clean(expected_run_id),
        "p1": _clean(expected_p1),
        "p2": _clean(expected_p2),
    }
    if any(not value for value in expected.values()):
        raise ValueError(
            "quarantine requires match_uid, feature_snapshot_id, run_id, p1, and p2"
        )

    mask = (
        predictions["match_uid"].map(_clean).eq(expected["match_uid"])
        & predictions["feature_snapshot_id"].map(_clean).eq(
            expected["feature_snapshot_id"]
        )
        & predictions["run_id"].map(_clean).eq(expected["run_id"])
        & predictions["p1"].map(_clean).eq(expected["p1"])
        & predictions["p2"].map(_clean).eq(expected["p2"])
    )
    matches = list(predictions.index[mask])
    if len(matches) != 1:
        uid_count = int(
            predictions["match_uid"].map(_clean).eq(expected["match_uid"]).sum()
        )
        raise RuntimeError(
            "quarantine requires exactly one row matching match, feature, run, "
            f"and oriented player pair ({len(matches)} exact; {uid_count} match_uid)"
        )
    return matches[0]


def _skip_event_id(row: pd.Series, reason_code: str) -> str:
    return "skip_" + stable_hash(
        row.get("run_id", ""),
        QUARANTINE_STAGE,
        row.get("match_uid", ""),
        row.get("feature_snapshot_id", ""),
        row.get("prediction_uid", ""),
        reason_code,
        row.get("p1", ""),
        row.get("p2", ""),
    )


def _build_skip_row(
    row: pd.Series,
    *,
    reason_code: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "skip_event_id": _skip_event_id(row, QUARANTINE_SKIP_REASON),
        "logged_at": utc_now_iso(),
        "run_id": _clean(row.get("run_id")),
        "run_started_at": "",
        "stage": QUARANTINE_STAGE,
        "skip_reason_code": QUARANTINE_SKIP_REASON,
        "skip_reason_detail": _audit_detail(reason_code, detail),
        "match_uid": _clean(row.get("match_uid")),
        "feature_snapshot_id": _clean(row.get("feature_snapshot_id")),
        "prediction_uid": _clean(row.get("prediction_uid")),
        "match_date": _clean(row.get("match_date")),
        "match_start_time": _clean(row.get("match_start_time")),
        "match_start_dt_local": "",
        "match_start_at_utc": _clean(row.get("match_start_at_utc")),
        "odds_scraped_at": _clean(row.get("odds_scraped_at")),
        "tournament": _clean(row.get("tournament")),
        "event_title": _clean(row.get("tournament")),
        "surface": _clean(row.get("surface")),
        "level": _clean(row.get("level")),
        "round": _clean(row.get("round")),
        "resolver_source": "administrative_review",
        "p1": _clean(row.get("p1")),
        "p2": _clean(row.get("p2")),
        "defaulted_features": _clean(row.get("defaulted_features")),
    }


def _append_or_verify_skip_audit(
    audit_path: Path,
    audit_row: dict[str, Any],
) -> bool:
    audit = _read_optional_csv_strings(audit_path, columns=SKIPPED_MATCH_COLUMNS)
    event_id = audit_row["skip_event_id"]
    matches = audit.index[audit["skip_event_id"].map(_clean).eq(event_id)]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate quarantine skip audit: {event_id}")
    if len(matches) == 1:
        existing = audit.loc[matches[0]]
        immutable_fields = (
            "run_id", "stage", "skip_reason_code", "skip_reason_detail",
            "match_uid", "feature_snapshot_id", "prediction_uid", "p1", "p2",
            "defaulted_features",
        )
        conflicts = [
            field for field in immutable_fields
            if _clean(existing.get(field)) != _clean(audit_row.get(field))
        ]
        if conflicts:
            raise RuntimeError(
                f"quarantine audit {event_id} has competing content: "
                f"{','.join(conflicts)}"
            )
        return False

    audit = pd.concat(
        [audit, pd.DataFrame([audit_row], columns=SKIPPED_MATCH_COLUMNS)],
        ignore_index=True,
    )
    atomic_write_csv(audit, audit_path)
    return True


def _quarantine_under_lock(
    production_dir: Path,
    *,
    reason_code: str,
    expected_match_uid: str,
    expected_feature_snapshot_id: str,
    expected_run_id: str,
    expected_p1: str,
    expected_p2: str,
    detail: str,
) -> QuarantineResult:
    prediction_path = production_dir / "prediction_log.csv"
    audit_path = production_dir / "logs" / "audit" / "skipped_live_matches.csv"
    predictions = _read_csv_strings(
        prediction_path,
        columns=PREDICTION_COLUMNS,
    )
    index = _exact_prediction_index(
        predictions,
        expected_match_uid=expected_match_uid,
        expected_feature_snapshot_id=expected_feature_snapshot_id,
        expected_run_id=expected_run_id,
        expected_p1=expected_p1,
        expected_p2=expected_p2,
    )
    row = predictions.loc[index]
    prediction_uid = _clean(row.get("prediction_uid"))
    if not prediction_uid:
        raise RuntimeError("quarantine target is missing immutable prediction_uid")

    winner = pd.to_numeric(pd.Series([row.get("actual_winner")]), errors="coerce").iloc[0]
    status = _clean(row.get("record_status")).casefold()
    note = _quarantine_note(reason_code, detail)
    defaulted = _append_default_markers(row.get("defaulted_features"))

    exact_replay = bool(
        status == "identity_conflict"
        and _clean(row.get("record_note")) == note
        and _clean(row.get("identity_status")).casefold() == "conflict"
        and _clean(row.get("identity_conflict_fields"))
        == QUARANTINE_CONFLICT_FIELDS
        and _false(row.get("features_complete"))
        and all(
            marker in [
                token.strip()
                for token in _clean(row.get("defaulted_features")).split(",")
            ]
            for marker in QUARANTINE_DEFAULT_MARKERS
        )
        and pd.isna(winner)
    )
    if status in IDENTITY_TERMINAL_STATUSES:
        if not exact_replay:
            raise RuntimeError(
                "prediction already has a competing identity terminal state: "
                f"{status}"
            )
        prediction_changed = False
    else:
        if status != "pending" or pd.notna(winner):
            raise RuntimeError(
                "prediction already has a competing terminal/non-pending state: "
                f"{status or '<blank>'}"
            )
        if (
            _clean(row.get("identity_status")).casefold() == "conflict"
            or _clean(row.get("identity_conflict_fields"))
        ):
            raise RuntimeError("pending prediction already carries competing identity state")

        for column in (
            "record_status", "record_note", "identity_status",
            "identity_conflict_fields", "features_complete", "defaulted_features",
        ):
            predictions[column] = predictions[column].astype(object)
        predictions.at[index, "record_status"] = "identity_conflict"
        predictions.at[index, "record_note"] = note
        predictions.at[index, "identity_status"] = "conflict"
        predictions.at[index, "identity_conflict_fields"] = (
            QUARANTINE_CONFLICT_FIELDS
        )
        predictions.at[index, "features_complete"] = "False"
        predictions.at[index, "defaulted_features"] = defaulted
        atomic_write_csv(predictions, prediction_path)
        prediction_changed = True
        row = predictions.loc[index]

    # Prediction first is intentional: after any crash, auto-settlement and
    # evaluation already fail closed.  An exact replay repairs the audit row
    # needed to block the accepted immutable browser Slate.
    audit_row = _build_skip_row(row, reason_code=reason_code, detail=detail)
    audit_appended = _append_or_verify_skip_audit(audit_path, audit_row)
    return QuarantineResult(
        match_uid=_clean(row.get("match_uid")),
        prediction_uid=prediction_uid,
        prediction_changed=prediction_changed,
        audit_appended=audit_appended,
        skip_event_id=str(audit_row["skip_event_id"]),
    )


def quarantine_invalid_recommendation(
    production_dir: Path | str,
    *,
    pipeline_paused: bool,
    reason_code: str,
    expected_match_uid: str,
    expected_feature_snapshot_id: str,
    expected_run_id: str,
    expected_p1: str,
    expected_p2: str,
    detail: str = "",
) -> QuarantineResult:
    """Tombstone one exact operational prediction and append Slate audit."""
    if pipeline_paused is not True:
        raise RuntimeError(
            "quarantine requires the hourly pipeline and auto-settlement to be paused"
        )
    normalized_reason = _clean(reason_code).casefold()
    if normalized_reason not in INVALID_RECOMMENDATION_REASON_CODES:
        raise ValueError(
            "unsupported invalid-recommendation quarantine reason: "
            f"{normalized_reason or '<blank>'}"
        )
    base = Path(production_dir).resolve()
    normalized_detail = _normalize_detail(detail)
    with operational_csv_lock(base / "logs"):
        return _quarantine_under_lock(
            base,
            reason_code=normalized_reason,
            expected_match_uid=expected_match_uid,
            expected_feature_snapshot_id=expected_feature_snapshot_id,
            expected_run_id=expected_run_id,
            expected_p1=expected_p1,
            expected_p2=expected_p2,
            detail=normalized_detail,
        )


def remediate_invalid_recommendation(
    production_dir: Path | str,
    *,
    pipeline_paused: bool,
    bet_id: str,
    reason_code: str,
    expected_match_uid: str,
    expected_feature_snapshot_id: str,
    expected_run_id: str,
    expected_p1: str,
    expected_p2: str,
    detail: str = "",
) -> RemediationResult:
    """Quarantine first, then release only the exact bad paper exposure."""
    if pipeline_paused is not True:
        raise RuntimeError(
            "remediation requires the hourly pipeline and auto-settlement to be paused"
        )
    normalized_reason = _clean(reason_code).casefold()
    if normalized_reason not in INVALID_RECOMMENDATION_REASON_CODES:
        raise ValueError(
            "unsupported invalid-recommendation remediation reason: "
            f"{normalized_reason or '<blank>'}"
        )
    base = Path(production_dir).resolve()
    normalized_detail = _normalize_detail(detail)
    with operational_csv_lock(base / "logs"):
        quarantine = _quarantine_under_lock(
            base,
            reason_code=normalized_reason,
            expected_match_uid=expected_match_uid,
            expected_feature_snapshot_id=expected_feature_snapshot_id,
            expected_run_id=expected_run_id,
            expected_p1=expected_p1,
            expected_p2=expected_p2,
            detail=normalized_detail,
        )
        tracker = BetTracker(str(base / "logs"))
        refunded = tracker.void_invalid_recommendation(
            _clean(bet_id),
            reason_code=normalized_reason,
            expected_match_uid=_clean(expected_match_uid),
            expected_feature_snapshot_id=_clean(expected_feature_snapshot_id),
            expected_run_id=_clean(expected_run_id),
            detail=normalized_detail,
        )
    return RemediationResult(
        match_uid=quarantine.match_uid,
        prediction_uid=quarantine.prediction_uid,
        bet_id=_clean(bet_id),
        prediction_changed=quarantine.prediction_changed,
        audit_appended=quarantine.audit_appended,
        bet_refunded=refunded,
        skip_event_id=quarantine.skip_event_id,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quarantine one exact bad-input prediction, append a Slate-blocking "
            "audit tombstone, then refund its exact paper bet"
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-pipeline-paused",
        action="store_true",
        help=(
            "required concurrency acknowledgement: the scheduled pipeline and "
            "standalone auto-settlement must both be paused"
        ),
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--bet-id", required=True)
    parser.add_argument("--match-uid", required=True)
    parser.add_argument("--feature-snapshot-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--p1", required=True)
    parser.add_argument("--p2", required=True)
    parser.add_argument(
        "--reason-code",
        default=INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
        choices=sorted(INVALID_RECOMMENDATION_REASON_CODES),
    )
    parser.add_argument("--detail", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.apply:
        parser.error("refusing mutation without explicit --apply")
    if not args.confirm_pipeline_paused:
        parser.error(
            "refusing mutation until --confirm-pipeline-paused is supplied"
        )

    result = remediate_invalid_recommendation(
        args.production_dir,
        pipeline_paused=args.confirm_pipeline_paused,
        bet_id=args.bet_id,
        reason_code=args.reason_code,
        expected_match_uid=args.match_uid,
        expected_feature_snapshot_id=args.feature_snapshot_id,
        expected_run_id=args.run_id,
        expected_p1=args.p1,
        expected_p2=args.p2,
        detail=args.detail,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
