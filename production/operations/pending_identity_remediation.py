"""Evidence-bound review registry for pending paper-bet identity remediation.

This module intentionally does *not* mutate ``all_bets.csv`` and does not
settle anything.  It has three responsibilities:

1. produce a deterministic report of orphan UID and duplicate-intent cases;
2. validate explicit human decisions against locally retained source evidence;
3. append approved decisions to a hash-bound immutable registry.

Normalized player/date joins and feature-snapshot joins are candidate discovery
signals only.  They never authorize a remap or a duplicate disposition.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from logging_utils import normalize_name
from operations.pending_reconciliation import (
    ORPHAN_MATCH_UID,
    UNRESOLVED,
    build_pending_review,
)


REVIEW_SCHEMA_VERSION = "1.0.0"
DECISION_SCHEMA_VERSION = "1.0.0"
PLAN_SCHEMA_VERSION = "1.0.0"
REGISTRY_SCHEMA_VERSION = "1.0.0"
EVIDENCE_ENVELOPE_SCHEMA_VERSION = "1.0.0"

REMAP_MATCH_UID = "remap_match_uid"
DUPLICATE_OF = "duplicate_of"
RETAIN_DISTINCT = "retain_distinct_exposure"
DEFER = "defer"
DECISION_TYPES = {REMAP_MATCH_UID, DUPLICATE_OF, RETAIN_DISTINCT, DEFER}

EVIDENCE_KINDS = {
    "bookmaker_event_record",
    "identity_capture_record",
    "official_match_record",
    "raw_source_artifact",
    "operator_intent_record",
}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_NAMES = ("bets", "predictions", "prediction_snapshots", "odds_history")

REQUIRED_BET_COLUMNS = {
    "bet_id",
    "status",
    "match",
    "match_uid",
    "bet_on",
    "match_date",
    "stake",
    "feature_snapshot_id",
    "run_id",
}
REQUIRED_PREDICTION_COLUMNS = {
    "match_uid",
    "p1",
    "p2",
    "match_date",
    "actual_winner",
    "feature_snapshot_id",
    "run_id",
}
REQUIRED_LINEAGE_COLUMNS = {
    "match_uid",
    "p1",
    "p2",
    "match_date",
    "feature_snapshot_id",
    "run_id",
}


class RemediationConflict(RuntimeError):
    """The reviewed evidence or registry no longer matches the plan."""


@dataclass(frozen=True)
class RemediationPaths:
    bets: Path
    predictions: Path
    prediction_snapshots: Path
    odds_history: Path

    @classmethod
    def from_prod_dir(cls, prod_dir: Path | str) -> "RemediationPaths":
        root = Path(prod_dir).resolve()
        return cls(
            bets=root / "logs" / "all_bets.csv",
            predictions=root / "prediction_log.csv",
            prediction_snapshots=root / "prediction_snapshots.csv",
            odds_history=root / "odds_history.csv",
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "bets": self.bets.resolve(),
            "predictions": self.predictions.resolve(),
            "prediction_snapshots": self.prediction_snapshots.resolve(),
            "odds_history": self.odds_history.resolve(),
        }


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_descriptor(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    exists = resolved.is_file()
    descriptor: dict[str, Any] = {
        "path": str(resolved),
        "exists": exists,
        "sha256": _file_sha256(resolved) if exists else None,
        "byte_size": resolved.stat().st_size if exists else 0,
    }
    if rows is not None:
        descriptor["rows"] = int(rows)
    return descriptor


def _read_csv(path: Path, *, label: str, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path.resolve()}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")
    return frame


def _row_payload(row: pd.Series, columns: Sequence[str]) -> dict[str, str]:
    return {column: _clean(row.get(column)) for column in columns}


def _row_sha256(row: pd.Series, columns: Sequence[str]) -> str:
    return _payload_sha256(_row_payload(row, columns))


def _strict_utc(value: Any, *, label: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{label} is required")
    parsed = pd.to_datetime(text, errors="coerce", utc=False)
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return parsed.tz_convert("UTC").isoformat()


def _match_players(value: Any) -> tuple[str, str] | None:
    parts = re.split(r"\s+vs\.?\s+", _clean(value), flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    players = tuple(normalize_name(part) for part in parts)
    if not all(players) or players[0] == players[1]:
        return None
    return players


def _prediction_pair(row: pd.Series) -> tuple[str, str] | None:
    p1 = normalize_name(_clean(row.get("p1")))
    p2 = normalize_name(_clean(row.get("p2")))
    if not p1 or not p2 or p1 == p2:
        return None
    return tuple(sorted((p1, p2)))


def _bet_pair(row: pd.Series) -> tuple[str, str] | None:
    players = _match_players(row.get("match"))
    return tuple(sorted(players)) if players else None


def _stable_case_id(case_type: str, identity: Mapping[str, Any]) -> str:
    return "pending_identity_case_" + _payload_sha256(
        {"case_type": case_type, "identity": identity}
    )[:24]


def _stable_subject_key(subject_type: str, identity: str) -> str:
    """Return a cross-generation subject key independent of row membership."""
    return "pending_identity_subject_" + _payload_sha256(
        {"subject_type": subject_type, "identity": identity}
    )[:24]


def _source_rows(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in frame[mask].iterrows():
        rows.append(
            {
                "source_row": int(index) + 2,
                "row_sha256": _row_sha256(row, columns),
            }
        )
    return rows


def _bound_source_rows(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    columns: Sequence[str],
    operational_source: str,
) -> list[dict[str, Any]]:
    return [
        {
            "operational_source": operational_source,
            "source_row": row["source_row"],
            "source_row_sha256": row["row_sha256"],
        }
        for row in _source_rows(frame, mask, columns=columns)
    ]


def _load_frames(
    paths: RemediationPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bets = _read_csv(paths.bets, label="bet log", required=REQUIRED_BET_COLUMNS)
    predictions = _read_csv(
        paths.predictions,
        label="prediction log",
        required=REQUIRED_PREDICTION_COLUMNS,
    )
    snapshots = _read_csv(
        paths.prediction_snapshots,
        label="prediction snapshots",
        required=REQUIRED_LINEAGE_COLUMNS,
    )
    odds = _read_csv(
        paths.odds_history,
        label="odds history",
        required=REQUIRED_LINEAGE_COLUMNS - {"feature_snapshot_id"},
    )
    return bets, predictions, snapshots, odds


def build_review_report(paths: RemediationPaths) -> dict[str, Any]:
    """Build a deterministic candidate report without approving any case."""
    bets, predictions, snapshots, odds = _load_frames(paths)
    normalized_bet_ids = bets["bet_id"].map(_clean)
    blank_bet_rows = [int(index) + 2 for index in bets.index[normalized_bet_ids.eq("")]]
    if blank_bet_rows:
        raise ValueError(
            "bet log has blank bet_id at source rows: "
            + ", ".join(str(value) for value in blank_bet_rows)
        )
    duplicate_bet_ids = sorted(
        value
        for value, count in normalized_bet_ids.value_counts().items()
        if count > 1
    )
    if duplicate_bet_ids:
        raise ValueError(
            "bet log has duplicate bet_id values: " + ", ".join(duplicate_bet_ids)
        )
    review = build_pending_review(bets, predictions)

    bets_work = bets.reset_index(drop=True).copy()
    bets_work["_source_row"] = bets_work.index + 2
    bet_columns = list(bets.columns)
    prediction_columns = list(predictions.columns)
    snapshot_columns = list(snapshots.columns)
    odds_columns = list(odds.columns)

    by_bet_id = {
        _clean(row.get("bet_id")): row for _, row in bets_work.iterrows()
    }
    prediction_index: dict[tuple[tuple[str, str], str], set[str]] = {}
    uid_identity_rows: dict[str, list[tuple[tuple[str, str] | None, str]]] = {}
    for _, row in predictions.iterrows():
        pair = _prediction_pair(row)
        date = _clean(row.get("match_date"))
        match_uid = _clean(row.get("match_uid"))
        if match_uid:
            uid_identity_rows.setdefault(match_uid, []).append((pair, date))
        if pair and date and match_uid:
            prediction_index.setdefault((pair, date), set()).add(match_uid)

    uid_contracts: dict[str, dict[str, Any]] = {}
    for match_uid, identities in uid_identity_rows.items():
        complete = all(pair is not None and bool(date) for pair, date in identities)
        identity_set = {
            (pair, date) for pair, date in identities if pair is not None and date
        }
        uid_contracts[match_uid] = {
            "prediction_rows": len(identities),
            "all_rows_identity_complete": complete,
            "identity_count": len(identity_set),
            "identities": [
                {"match_pair": list(pair), "match_date": date}
                for pair, date in sorted(identity_set)
            ],
            "semantically_unique": complete and len(identity_set) == 1,
        }

    cases: list[dict[str, Any]] = []
    orphan_rows = review[review["outcome_classification"].eq(ORPHAN_MATCH_UID)]
    ambiguous_identity_rows = review[
        review["authoritative_outcome_status"].eq("ambiguous_match_identity")
    ]
    uid_review_rows = pd.concat(
        [orphan_rows, ambiguous_identity_rows], ignore_index=True
    ).drop_duplicates(subset=["bet_id"], keep="first")
    for _, review_row in uid_review_rows.sort_values("bet_id").iterrows():
        bet_id = _clean(review_row.get("bet_id"))
        row = by_bet_id[bet_id]
        source_row = int(row["_source_row"])
        row_hash = _row_sha256(row, bet_columns)
        original_uid = _clean(row.get("match_uid"))
        pair = _bet_pair(row)
        date = _clean(row.get("match_date"))
        raw_candidates = sorted(
            set(prediction_index.get((pair, date), set())) - {original_uid}
        ) if pair else []
        candidates = [
            match_uid
            for match_uid in raw_candidates
            if uid_contracts.get(match_uid, {}).get("semantically_unique")
            and uid_contracts[match_uid]["identities"]
            == [{"match_pair": list(pair or ()), "match_date": date}]
        ]
        rejected_candidates = [
            match_uid for match_uid in raw_candidates if match_uid not in candidates
        ]
        feature_id = _clean(row.get("feature_snapshot_id"))
        run_id = _clean(row.get("run_id"))

        feature_targets = sorted(
            set(
                _clean(value)
                for value in predictions.loc[
                    predictions["feature_snapshot_id"].eq(feature_id)
                    & predictions["run_id"].eq(run_id),
                    "match_uid",
                ]
                if _clean(value)
            )
        ) if feature_id and run_id else []
        snapshot_rows = _source_rows(
            snapshots,
            snapshots["match_uid"].eq(original_uid),
            columns=snapshot_columns,
        )
        odds_rows = _source_rows(
            odds,
            odds["match_uid"].eq(original_uid),
            columns=odds_columns,
        )
        original_binding_sources = [
            {
                "operational_source": "bets",
                "source_row": source_row,
                "source_row_sha256": row_hash,
            }
        ]
        original_binding_sources.extend(
            _bound_source_rows(
                snapshots,
                snapshots["match_uid"].eq(original_uid),
                columns=snapshot_columns,
                operational_source="prediction_snapshots",
            )
        )
        original_binding_sources.extend(
            _bound_source_rows(
                odds,
                odds["match_uid"].eq(original_uid),
                columns=odds_columns,
                operational_source="odds_history",
            )
        )
        target_binding_sources = {
            match_uid: _bound_source_rows(
                predictions,
                predictions["match_uid"].eq(match_uid),
                columns=prediction_columns,
                operational_source="predictions",
            )
            for match_uid in candidates
        }
        identity = {
            "bet_id": bet_id,
            "bet_row_sha256": row_hash,
            "original_match_uid": original_uid,
        }
        subject_key = _stable_subject_key("bet", bet_id)
        cases.append(
            {
                "case_id": _stable_case_id("match_uid_remap", identity),
                "subject_key": subject_key,
                "case_type": "match_uid_remap",
                "automatic_disposition": "requires_source_review",
                "candidate_is_authority": False,
                "identity": {
                    **identity,
                    "source_row": source_row,
                    "pending_identity_key": _clean(
                        review_row.get("pending_identity_key")
                    ),
                    "match_pair": list(pair or ()),
                    "match_date": date,
                    "bet_on": normalize_name(_clean(row.get("bet_on"))),
                    "feature_snapshot_id": feature_id,
                    "run_id": run_id,
                },
                "candidate_target_match_uids": candidates,
                "candidate_target_identity_contracts": {
                    match_uid: uid_contracts[match_uid] for match_uid in candidates
                },
                "rejected_semantically_ambiguous_target_match_uids": {
                    match_uid: uid_contracts[match_uid]
                    for match_uid in rejected_candidates
                },
                "candidate_basis": (
                    "exact normalized player pair and date; discovery only, "
                    "never remap authority"
                ),
                "lineage_signals": {
                    "same_run_feature_snapshot_target_match_uids": feature_targets,
                    "original_uid_prediction_snapshot_rows": snapshot_rows,
                    "original_uid_odds_history_rows": odds_rows,
                },
                "uid_binding_sources": {
                    "original": original_binding_sources,
                    "targets": target_binding_sources,
                },
                "blocked_reasons": [
                    (
                        "no_authoritative_uid_alias_record"
                        if _clean(review_row.get("outcome_classification"))
                        == ORPHAN_MATCH_UID
                        else "reused_uid_requires_authoritative_replacement"
                    ),
                    "name_date_or_feature_lineage_cannot_authorize_remap",
                ],
            }
        )

    duplicate_rows = review[review["is_duplicate_pending_identity"].astype(bool)]
    for identity_key, group in duplicate_rows.groupby(
        "pending_identity_key", sort=True, dropna=False
    ):
        bet_ids = sorted(_clean(value) for value in group["bet_id"])
        members: list[dict[str, Any]] = []
        for bet_id in bet_ids:
            row = by_bet_id[bet_id]
            members.append(
                {
                    "bet_id": bet_id,
                    "source_row": int(row["_source_row"]),
                    "bet_row_sha256": _row_sha256(row, bet_columns),
                    "match_uid": _clean(row.get("match_uid")),
                    "timestamp": _clean(row.get("timestamp")),
                    "stake": _clean(row.get("stake")),
                }
            )
        identity = {
            "pending_identity_key": _clean(identity_key),
            "members": [
                {"bet_id": member["bet_id"], "bet_row_sha256": member["bet_row_sha256"]}
                for member in members
            ],
        }
        subject_key = _stable_subject_key("pending_identity", _clean(identity_key))
        cases.append(
            {
                "case_id": _stable_case_id("duplicate_intent", identity),
                "subject_key": subject_key,
                "case_type": "duplicate_intent",
                "automatic_disposition": "requires_source_review",
                "candidate_is_authority": False,
                "identity": {
                    "pending_identity_key": _clean(identity_key),
                    "members": members,
                },
                "candidate_basis": (
                    "same normalized pair, selected side, and date; duplicate "
                    "label only, not evidence of operator intent"
                ),
                "blocked_reasons": [
                    "intent_not_observable_from_recommendation_rows",
                    "explicit_duplicate_or_distinct_exposure_evidence_required",
                ],
            }
        )

    cases.sort(key=lambda case: (case["case_type"], case["case_id"]))
    inputs = {
        name: _file_descriptor(path, rows=len(frame))
        for (name, path), frame in zip(
            paths.as_dict().items(),
            (bets, predictions, snapshots, odds),
        )
    }
    unresolved_outcome_only = review[
        review["outcome_classification"].eq(UNRESOLVED)
        & ~review["is_duplicate_pending_identity"].astype(bool)
        & ~review["authoritative_outcome_status"].eq("ambiguous_match_identity")
    ]
    report_id = "pending_identity_review_" + _payload_sha256(
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "input_hashes": {name: value["sha256"] for name, value in inputs.items()},
            "case_ids": [case["case_id"] for case in cases],
        }
    )[:24]
    without_digest: dict[str, Any] = {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "report_id": report_id,
        "operation": "pending_identity_candidate_review",
        "read_only": True,
        "automatic_approvals": 0,
        "candidate_authority_warning": (
            "Candidate joins are investigation leads only. No normalized-name, "
            "date, feature-vector, feature-snapshot, or odds similarity can "
            "authorize a UID remap or duplicate-intent disposition."
        ),
        "inputs": inputs,
        "summary": {
            "pending_rows": int(len(review)),
            "orphan_uid_rows": int(len(orphan_rows)),
            "orphan_uid_stake": round(
                float(
                    pd.to_numeric(orphan_rows["stake"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                6,
            ),
            "ambiguous_identity_uid_rows": int(len(ambiguous_identity_rows)),
            "duplicate_labeled_rows": int(len(duplicate_rows)),
            "duplicate_identity_groups": int(
                duplicate_rows["pending_identity_key"].nunique()
            ),
            "duplicate_labeled_stake": round(
                float(
                    pd.to_numeric(duplicate_rows["stake"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                6,
            ),
            "outcome_pending_not_identity_cases": int(len(unresolved_outcome_only)),
            "cases": len(cases),
            "automatically_proven_cases": 0,
        },
        "cases": cases,
    }
    return {**without_digest, "report_digest": _payload_sha256(without_digest)}


def _write_immutable_json(payload: Mapping[str, Any], path: Path) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    if resolved.is_file():
        if resolved.read_text(encoding="utf-8") == body:
            return
        raise ValueError(f"refusing to overwrite different content: {resolved}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        _fsync_directory(resolved.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_review_report(report: Mapping[str, Any], path: Path) -> None:
    _write_immutable_json(report, path)


def build_decision_template(report: Mapping[str, Any]) -> dict[str, Any]:
    _validate_report_digest(report)
    return {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "report_id": report["report_id"],
        "report_digest": report["report_digest"],
        "reviewer": "",
        "reviewed_at_utc": "",
        "evidence": [],
        "decisions": [
            {
                "case_id": case["case_id"],
                "decision": DEFER,
                "reason": "",
                "evidence_ids": [],
                "supersedes_decision_id": "",
            }
            for case in report["cases"]
        ],
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path.resolve()}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_report_digest(report: Mapping[str, Any]) -> None:
    if report.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported review schema version")
    digest = _clean(report.get("report_digest"))
    without = {key: value for key, value in report.items() if key != "report_digest"}
    if not HEX64_RE.fullmatch(digest) or _payload_sha256(without) != digest:
        raise RemediationConflict("review report digest mismatch")


def _report_paths(report: Mapping[str, Any]) -> RemediationPaths:
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(SOURCE_NAMES):
        raise RemediationConflict(
            "review report must contain exactly four named source descriptors"
        )
    resolved_paths: dict[str, Path] = {}
    for label in SOURCE_NAMES:
        descriptor = inputs[label]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "exists",
            "sha256",
            "byte_size",
            "rows",
        }:
            raise RemediationConflict(f"invalid source descriptor shape: {label}")
        path_text = _clean(descriptor.get("path"))
        if not path_text:
            raise RemediationConflict(f"blank source descriptor path: {label}")
        path = Path(path_text).resolve()
        if descriptor.get("exists") is not True:
            raise RemediationConflict(f"source descriptor must bind an existing file: {label}")
        if not HEX64_RE.fullmatch(_clean(descriptor.get("sha256"))):
            raise RemediationConflict(f"invalid source descriptor hash: {label}")
        if not isinstance(descriptor.get("byte_size"), int) or descriptor["byte_size"] < 0:
            raise RemediationConflict(f"invalid source descriptor byte size: {label}")
        if not isinstance(descriptor.get("rows"), int) or descriptor["rows"] < 0:
            raise RemediationConflict(f"invalid source descriptor row count: {label}")
        resolved_paths[label] = path
    if len(set(resolved_paths.values())) != len(SOURCE_NAMES):
        raise RemediationConflict("review report source paths must be distinct")
    return RemediationPaths(
        bets=resolved_paths["bets"],
        predictions=resolved_paths["predictions"],
        prediction_snapshots=resolved_paths["prediction_snapshots"],
        odds_history=resolved_paths["odds_history"],
    )


def _verify_report_inputs(report: Mapping[str, Any]) -> RemediationPaths:
    paths = _report_paths(report)
    for label, path in paths.as_dict().items():
        descriptor = report["inputs"][label]
        if not path.is_file():
            raise RemediationConflict(f"stale report input missing: {label}")
        if _file_sha256(path) != descriptor.get("sha256"):
            raise RemediationConflict(f"stale report input hash: {label}")
        if path.stat().st_size != descriptor.get("byte_size"):
            raise RemediationConflict(f"stale report input byte size: {label}")
    return paths


def _verify_report_exact(report: Mapping[str, Any]) -> RemediationPaths:
    """Regenerate the report from its four sources and require exact equality."""
    _validate_report_digest(report)
    paths = _verify_report_inputs(report)
    rebuilt = build_review_report(paths)
    if rebuilt != report:
        raise RemediationConflict(
            "review report differs from deterministic source regeneration"
        )
    return paths


def _validate_source_uri(value: Any) -> str:
    text = _clean(value)
    parsed = urlparse(text)
    if parsed.scheme not in {"https", "http", "s3", "gs", "file"}:
        raise ValueError("evidence source_uri must use https/http/s3/gs/file")
    if parsed.scheme in {"https", "http", "s3", "gs"} and not parsed.netloc:
        raise ValueError("evidence source_uri is missing an authority")
    if parsed.scheme == "file" and not parsed.path:
        raise ValueError("file evidence source_uri is missing a path")
    return text


def _resolve_evidence_artifact(decision_path: Path, value: Any) -> Path:
    relative = Path(_clean(value))
    if not _clean(value) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence artifact_path must be a safe relative path")
    resolved = (decision_path.resolve().parent / relative).resolve()
    try:
        resolved.relative_to(decision_path.resolve().parent)
    except ValueError:
        raise ValueError("evidence artifact_path escapes the decision directory") from None
    if not resolved.is_file():
        raise FileNotFoundError(f"evidence artifact not found: {resolved}")
    return resolved


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{label} has invalid fields; missing={missing}, extra={extra}"
        )


def _load_evidence_record(
    item: Mapping[str, Any], *, decision_path: Path
) -> dict[str, Any]:
    _require_exact_keys(
        item,
        {
            "evidence_id",
            "evidence_kind",
            "source_uri",
            "source_system",
            "external_record_id",
            "observed_at_utc",
            "artifact_path",
            "artifact_sha256",
            "envelope_path",
            "envelope_sha256",
            "claim",
        },
        label="evidence declaration",
    )
    evidence_id = _clean(item.get("evidence_id"))
    kind = _clean(item.get("evidence_kind"))
    if not evidence_id:
        raise ValueError("evidence_id is required")
    if kind not in EVIDENCE_KINDS:
        raise ValueError(f"unsupported evidence_kind for {evidence_id}")
    source_uri = _validate_source_uri(item.get("source_uri"))
    source_system = _clean(item.get("source_system"))
    external_record_id = _clean(item.get("external_record_id"))
    if not source_system or not external_record_id:
        raise ValueError(
            f"source_system and external_record_id are required: {evidence_id}"
        )
    observed_at = _strict_utc(
        item.get("observed_at_utc"), label=f"evidence {evidence_id} observed_at_utc"
    )
    claim = _clean(item.get("claim"))
    if len(claim) < 12:
        raise ValueError(f"evidence claim is too short: {evidence_id}")

    expected_artifact_hash = _clean(item.get("artifact_sha256"))
    if not HEX64_RE.fullmatch(expected_artifact_hash):
        raise ValueError(f"invalid artifact_sha256 for {evidence_id}")
    artifact = _resolve_evidence_artifact(decision_path, item.get("artifact_path"))
    artifact_hash = _file_sha256(artifact)
    if artifact_hash != expected_artifact_hash:
        raise RemediationConflict(f"evidence artifact hash mismatch: {evidence_id}")
    parsed_artifact: dict[str, Any] | None = None
    if kind in {
        "bookmaker_event_record",
        "identity_capture_record",
        "operator_intent_record",
    }:
        parsed_artifact = _load_json(
            artifact, label=f"structured evidence artifact {evidence_id}"
        )

    expected_envelope_hash = _clean(item.get("envelope_sha256"))
    if not HEX64_RE.fullmatch(expected_envelope_hash):
        raise ValueError(f"invalid envelope_sha256 for {evidence_id}")
    envelope_path = _resolve_evidence_artifact(
        decision_path, item.get("envelope_path")
    )
    if envelope_path == artifact:
        raise ValueError(
            f"evidence source artifact and typed envelope must be distinct: {evidence_id}"
        )
    envelope_hash = _file_sha256(envelope_path)
    if envelope_hash != expected_envelope_hash:
        raise RemediationConflict(f"evidence envelope hash mismatch: {evidence_id}")
    envelope = _load_json(envelope_path, label=f"evidence envelope {evidence_id}")
    _require_exact_keys(
        envelope,
        {
            "evidence_envelope_schema_version",
            "evidence_id",
            "evidence_kind",
            "source",
            "claim",
            "assertion",
            "binding",
        },
        label=f"evidence envelope {evidence_id}",
    )
    if envelope.get("evidence_envelope_schema_version") != EVIDENCE_ENVELOPE_SCHEMA_VERSION:
        raise ValueError(f"unsupported evidence envelope schema: {evidence_id}")
    source = envelope.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"evidence envelope source must be an object: {evidence_id}")
    _require_exact_keys(
        source,
        {
            "source_uri",
            "source_system",
            "external_record_id",
            "observed_at_utc",
            "artifact_sha256",
        },
        label=f"evidence envelope source {evidence_id}",
    )
    expected_source = {
        "source_uri": source_uri,
        "source_system": source_system,
        "external_record_id": external_record_id,
        "observed_at_utc": observed_at,
        "artifact_sha256": artifact_hash,
    }
    if (
        envelope.get("evidence_id") != evidence_id
        or envelope.get("evidence_kind") != kind
        or envelope.get("claim") != claim
        or source != expected_source
    ):
        raise RemediationConflict(
            f"evidence declaration/envelope identity mismatch: {evidence_id}"
        )
    if not isinstance(envelope.get("binding"), dict):
        raise ValueError(f"evidence binding must be an object: {evidence_id}")
    return {
        "evidence_id": evidence_id,
        "evidence_kind": kind,
        "source_uri": source_uri,
        "source_system": source_system,
        "external_record_id": external_record_id,
        "observed_at_utc": observed_at,
        "artifact_path": _clean(item.get("artifact_path")),
        "artifact_sha256": artifact_hash,
        "artifact_byte_size": artifact.stat().st_size,
        "envelope_path": _clean(item.get("envelope_path")),
        "envelope_sha256": envelope_hash,
        "envelope_byte_size": envelope_path.stat().st_size,
        "claim": claim,
        "envelope": envelope,
        "parsed_artifact": parsed_artifact,
    }


def _validate_evidence_binding(
    evidence: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    decision: str,
    resolution: Mapping[str, Any],
    supersedes_decision_id: str,
) -> None:
    case_type = _clean(case.get("case_type"))
    if case_type == "match_uid_remap":
        assertion = "same_external_match_identity"
        expected_binding = {
            "case_type": case_type,
            "subject_key": case["subject_key"],
            "case_id": case["case_id"],
            "bet_id": case["identity"]["bet_id"],
            "bet_row_sha256": case["identity"]["bet_row_sha256"],
            "original_match_uid": resolution["original_match_uid"],
            "target_match_uid": resolution["target_match_uid"],
            "match_pair": case["identity"]["match_pair"],
            "match_date": case["identity"]["match_date"],
            "supersedes_decision_id": supersedes_decision_id,
        }
    elif case_type == "duplicate_intent":
        assertion = (
            "same_operator_decision_identity"
            if decision == DUPLICATE_OF
            else "distinct_operator_decision_identities"
        )
        expected_binding = {
            "case_type": case_type,
            "subject_key": case["subject_key"],
            "case_id": case["case_id"],
            "pending_identity_key": case["identity"]["pending_identity_key"],
            "members": [
                {
                    "bet_id": member["bet_id"],
                    "bet_row_sha256": member["bet_row_sha256"],
                }
                for member in case["identity"]["members"]
            ],
            "decision": decision,
            "resolution": dict(resolution),
            "supersedes_decision_id": supersedes_decision_id,
        }
    else:
        raise ValueError(f"unsupported report case type: {case_type}")
    envelope = evidence["envelope"]
    if envelope.get("assertion") != assertion or envelope.get("binding") != expected_binding:
        raise RemediationConflict(
            f"evidence envelope is not bound to decision subject: {evidence['evidence_id']}"
        )


def _validate_structured_source_identity(
    artifact: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    expected = {
        "source_system": evidence["source_system"],
        "external_record_id": evidence["external_record_id"],
        "source_uri": evidence["source_uri"],
        "observed_at_utc": evidence["observed_at_utc"],
    }
    actual = {key: artifact.get(key) for key in expected}
    if actual != expected:
        raise RemediationConflict(
            f"structured source identity does not match declaration: "
            f"{evidence['evidence_id']}"
        )


def _validate_identity_capture_artifact(
    evidence: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> None:
    artifact = evidence.get("parsed_artifact")
    if not isinstance(artifact, dict):
        raise ValueError(
            f"identity approval requires structured JSON: {evidence['evidence_id']}"
        )
    _require_exact_keys(
        artifact,
        {
            "identity_capture_schema_version",
            "record_type",
            "source_system",
            "external_record_id",
            "source_uri",
            "observed_at_utc",
            "match_pair",
            "match_date",
            "uid_bindings",
        },
        label=f"identity capture {evidence['evidence_id']}",
    )
    if (
        artifact.get("identity_capture_schema_version") != "1.0.0"
        or artifact.get("record_type") != "bookmaker_event_identity_capture"
    ):
        raise ValueError(
            f"unsupported identity capture schema/type: {evidence['evidence_id']}"
        )
    _validate_structured_source_identity(artifact, evidence)
    if (
        artifact.get("match_pair") != case["identity"]["match_pair"]
        or artifact.get("match_date") != case["identity"]["match_date"]
    ):
        raise RemediationConflict(
            f"identity capture pair/date does not match case: {evidence['evidence_id']}"
        )
    raw_bindings = artifact.get("uid_bindings")
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 2:
        raise ValueError(
            f"identity capture must contain exactly two UID bindings: "
            f"{evidence['evidence_id']}"
        )
    by_role: dict[str, dict[str, Any]] = {}
    for binding in raw_bindings:
        if not isinstance(binding, dict):
            raise ValueError(
                f"identity capture UID bindings must be objects: "
                f"{evidence['evidence_id']}"
            )
        _require_exact_keys(
            binding,
            {
                "role",
                "match_uid",
                "operational_source",
                "source_row",
                "source_row_sha256",
            },
            label=f"identity capture UID binding {evidence['evidence_id']}",
        )
        role = _clean(binding.get("role"))
        if role not in {"original", "target"} or role in by_role:
            raise ValueError(
                f"identity capture roles must be unique original/target: "
                f"{evidence['evidence_id']}"
            )
        if (
            not isinstance(binding.get("source_row"), int)
            or binding["source_row"] < 2
            or not HEX64_RE.fullmatch(_clean(binding.get("source_row_sha256")))
        ):
            raise ValueError(
                f"identity capture source-row binding is invalid: "
                f"{evidence['evidence_id']}"
            )
        by_role[role] = binding
    if set(by_role) != {"original", "target"}:
        raise ValueError(
            f"identity capture requires original and target roles: "
            f"{evidence['evidence_id']}"
        )
    if by_role["original"]["match_uid"] != resolution["original_match_uid"]:
        raise RemediationConflict(
            f"identity capture original UID mismatch: {evidence['evidence_id']}"
        )
    if by_role["target"]["match_uid"] != resolution["target_match_uid"]:
        raise RemediationConflict(
            f"identity capture target UID mismatch: {evidence['evidence_id']}"
        )
    original_source = {
        key: value
        for key, value in by_role["original"].items()
        if key not in {"role", "match_uid"}
    }
    target_source = {
        key: value
        for key, value in by_role["target"].items()
        if key not in {"role", "match_uid"}
    }
    allowed = case["uid_binding_sources"]
    if original_source not in allowed["original"]:
        raise RemediationConflict(
            f"identity capture original source-row hash is not report-bound: "
            f"{evidence['evidence_id']}"
        )
    if target_source not in allowed["targets"].get(
        resolution["target_match_uid"], []
    ):
        raise RemediationConflict(
            f"identity capture target source-row hash is not report-bound: "
            f"{evidence['evidence_id']}"
        )


def _validate_operator_intent_artifact(
    evidence: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    decision: str,
    resolution: Mapping[str, Any],
) -> None:
    artifact = evidence.get("parsed_artifact")
    if not isinstance(artifact, dict):
        raise ValueError(
            f"duplicate approval requires structured JSON: {evidence['evidence_id']}"
        )
    _require_exact_keys(
        artifact,
        {
            "intent_record_schema_version",
            "record_type",
            "source_system",
            "external_record_id",
            "source_uri",
            "observed_at_utc",
            "subject_key",
            "pending_identity_key",
            "disposition",
            "members",
            "resolution",
        },
        label=f"operator intent record {evidence['evidence_id']}",
    )
    if (
        artifact.get("intent_record_schema_version") != "1.0.0"
        or artifact.get("record_type") != "operator_intent_record"
    ):
        raise ValueError(
            f"unsupported operator intent schema/type: {evidence['evidence_id']}"
        )
    _validate_structured_source_identity(artifact, evidence)
    if (
        artifact.get("subject_key") != case["subject_key"]
        or artifact.get("pending_identity_key")
        != case["identity"]["pending_identity_key"]
        or artifact.get("resolution") != dict(resolution)
    ):
        raise RemediationConflict(
            f"operator intent record is not bound to decision subject: "
            f"{evidence['evidence_id']}"
        )
    members = artifact.get("members")
    if not isinstance(members, list):
        raise ValueError(
            f"operator intent members must be a list: {evidence['evidence_id']}"
        )
    by_bet: dict[str, dict[str, Any]] = {}
    for member in members:
        if not isinstance(member, dict):
            raise ValueError(
                f"operator intent members must be objects: {evidence['evidence_id']}"
            )
        _require_exact_keys(
            member,
            {"bet_id", "bet_row_sha256", "intent_id"},
            label=f"operator intent member {evidence['evidence_id']}",
        )
        bet_id = _clean(member.get("bet_id"))
        intent_id = _clean(member.get("intent_id"))
        if (
            not bet_id
            or bet_id in by_bet
            or not intent_id
            or not HEX64_RE.fullmatch(_clean(member.get("bet_row_sha256")))
        ):
            raise ValueError(
                f"operator intent member identity is invalid: {evidence['evidence_id']}"
            )
        by_bet[bet_id] = member
    expected_members = {
        member["bet_id"]: member["bet_row_sha256"]
        for member in case["identity"]["members"]
    }
    actual_members = {
        bet_id: member["bet_row_sha256"] for bet_id, member in by_bet.items()
    }
    if actual_members != expected_members:
        raise RemediationConflict(
            f"operator intent members do not match report rows: "
            f"{evidence['evidence_id']}"
        )
    intent_ids = [by_bet[bet_id]["intent_id"] for bet_id in sorted(by_bet)]
    if decision == DUPLICATE_OF:
        if artifact.get("disposition") != "same_intent" or len(set(intent_ids)) != 1:
            raise RemediationConflict(
                f"operator intent record does not prove one shared intent: "
                f"{evidence['evidence_id']}"
            )
    elif (
        artifact.get("disposition") != "distinct_intents"
        or len(set(intent_ids)) != len(intent_ids)
    ):
        raise RemediationConflict(
            f"operator intent record does not prove distinct intents: "
            f"{evidence['evidence_id']}"
        )


def _case_index(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("review report cases must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    subjects: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("review case must be an object")
        case_id = _clean(case.get("case_id"))
        subject_key = _clean(case.get("subject_key"))
        if not case_id or case_id in result:
            raise ValueError("review case IDs must be nonblank and unique")
        if not subject_key or subject_key in subjects:
            raise ValueError("review subject keys must be nonblank and unique")
        subjects.add(subject_key)
        result[case_id] = case
    return result


def validate_decisions(
    report: Mapping[str, Any],
    decisions: Mapping[str, Any],
    *,
    decision_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return approved registry entries and explicitly deferred case records."""
    _verify_report_exact(report)
    decision_path = decision_path.resolve()
    if _load_json(decision_path, label="decision document") != decisions:
        raise RemediationConflict(
            "decision object differs from the bound decision document"
        )
    _require_exact_keys(
        decisions,
        {
            "decision_schema_version",
            "report_id",
            "report_digest",
            "reviewer",
            "reviewed_at_utc",
            "evidence",
            "decisions",
        },
        label="decision document",
    )
    if decisions.get("decision_schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError("unsupported decision schema version")
    if decisions.get("report_id") != report.get("report_id"):
        raise RemediationConflict("decision report_id mismatch")
    if decisions.get("report_digest") != report.get("report_digest"):
        raise RemediationConflict("decision report_digest mismatch")

    reviewer = _clean(decisions.get("reviewer"))
    if not reviewer:
        raise ValueError("reviewer is required")
    reviewed_at = _strict_utc(decisions.get("reviewed_at_utc"), label="reviewed_at_utc")

    raw_evidence = decisions.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("evidence must be a list")
    evidence: dict[str, dict[str, Any]] = {}
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise ValueError("evidence entries must be objects")
        record = _load_evidence_record(item, decision_path=decision_path)
        evidence_id = record["evidence_id"]
        if evidence_id in evidence:
            raise ValueError("evidence_id values must be nonblank and unique")
        evidence[evidence_id] = record

    case_by_id = _case_index(report)
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("decisions must be a list")
    seen_cases: set[str] = set()
    seen_subjects: set[str] = set()
    used_evidence_ids: set[str] = set()
    approved: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("decision entries must be objects")
        case_id = _clean(raw.get("case_id"))
        if case_id not in case_by_id:
            raise ValueError(f"unknown decision case_id: {case_id}")
        if case_id in seen_cases:
            raise RemediationConflict(f"conflicting duplicate decision: {case_id}")
        seen_cases.add(case_id)
        decision = _clean(raw.get("decision"))
        if decision not in DECISION_TYPES:
            raise ValueError(f"unsupported decision for {case_id}")
        expected_decision_fields = {
            "case_id",
            "decision",
            "reason",
            "evidence_ids",
            "supersedes_decision_id",
        }
        if decision == REMAP_MATCH_UID:
            expected_decision_fields.add("target_match_uid")
        elif decision == DUPLICATE_OF:
            expected_decision_fields.update(
                {"canonical_bet_id", "duplicate_bet_ids"}
            )
        _require_exact_keys(
            raw,
            expected_decision_fields,
            label=f"decision {case_id}",
        )
        reason = _clean(raw.get("reason"))
        if len(reason) < 12:
            raise ValueError(f"decision reason is too short: {case_id}")
        evidence_ids = raw.get("evidence_ids")
        if not isinstance(evidence_ids, list) or any(
            not isinstance(value, str) for value in evidence_ids
        ):
            raise ValueError(f"evidence_ids must be a string list: {case_id}")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError(f"duplicate evidence_ids: {case_id}")
        missing_evidence = sorted(set(evidence_ids) - set(evidence))
        if missing_evidence:
            raise ValueError(
                f"unknown evidence_ids for {case_id}: {', '.join(missing_evidence)}"
            )

        case = case_by_id[case_id]
        case_type = _clean(case.get("case_type"))
        if decision == DEFER:
            if evidence_ids or _clean(raw.get("supersedes_decision_id")):
                raise ValueError(f"deferred decision cannot bind evidence/supersession: {case_id}")
            deferred.append({"case_id": case_id, "decision": decision, "reason": reason})
            continue
        if not evidence_ids:
            raise ValueError(f"approved decision requires evidence: {case_id}")
        repeated_evidence = sorted(set(evidence_ids) & used_evidence_ids)
        if repeated_evidence:
            raise RemediationConflict(
                f"evidence cannot bind multiple decisions: {', '.join(repeated_evidence)}"
            )
        used_evidence_ids.update(evidence_ids)
        subject_key = _clean(case.get("subject_key"))
        if not subject_key:
            raise RemediationConflict(f"review case lacks stable subject_key: {case_id}")
        if subject_key in seen_subjects:
            raise RemediationConflict(
                f"decision document repeats stable subject: {subject_key}"
            )
        seen_subjects.add(subject_key)
        supersedes_decision_id = _clean(raw.get("supersedes_decision_id"))
        if supersedes_decision_id and not re.fullmatch(
            r"pending_identity_decision_[0-9a-f]{24}", supersedes_decision_id
        ):
            raise ValueError(f"invalid supersedes_decision_id: {case_id}")

        resolution: dict[str, Any] = {}
        if case_type == "match_uid_remap":
            if decision != REMAP_MATCH_UID:
                raise ValueError(f"remap case requires {REMAP_MATCH_UID}: {case_id}")
            target_uid = _clean(raw.get("target_match_uid"))
            candidates = set(case.get("candidate_target_match_uids", []))
            if not target_uid or target_uid not in candidates:
                raise RemediationConflict(
                    f"target UID is not an exact report candidate: {case_id}"
                )
            target_contracts = case.get("candidate_target_identity_contracts", {})
            target_contract = target_contracts.get(target_uid)
            expected_identity = {
                "match_pair": case["identity"]["match_pair"],
                "match_date": case["identity"]["match_date"],
            }
            if (
                not isinstance(target_contract, dict)
                or target_contract.get("semantically_unique") is not True
                or target_contract.get("all_rows_identity_complete") is not True
                or target_contract.get("identity_count") != 1
                or target_contract.get("identities") != [expected_identity]
            ):
                raise RemediationConflict(
                    f"target UID is semantically reused or incomplete: {case_id}"
                )
            resolution = {
                "original_match_uid": case["identity"]["original_match_uid"],
                "target_match_uid": target_uid,
                "bet_id": case["identity"]["bet_id"],
                "bet_row_sha256": case["identity"]["bet_row_sha256"],
            }
        elif case_type == "duplicate_intent":
            if decision not in {DUPLICATE_OF, RETAIN_DISTINCT}:
                raise ValueError(f"duplicate case has invalid disposition: {case_id}")
            member_ids = sorted(
                member["bet_id"] for member in case["identity"]["members"]
            )
            if decision == DUPLICATE_OF:
                canonical = _clean(raw.get("canonical_bet_id"))
                duplicates = raw.get("duplicate_bet_ids")
                if not isinstance(duplicates, list) or any(
                    not isinstance(value, str) for value in duplicates
                ):
                    raise ValueError(f"duplicate_bet_ids must be a string list: {case_id}")
                if canonical not in member_ids or sorted(duplicates) != sorted(
                    set(member_ids) - {canonical}
                ):
                    raise RemediationConflict(
                        f"duplicate disposition must partition every case member: {case_id}"
                    )
                resolution = {
                    "canonical_bet_id": canonical,
                    "duplicate_bet_ids": sorted(duplicates),
                    "pending_identity_key": case["identity"]["pending_identity_key"],
                }
            else:
                resolution = {
                    "distinct_bet_ids": member_ids,
                    "pending_identity_key": case["identity"]["pending_identity_key"],
                }
        else:
            raise ValueError(f"unsupported report case type: {case_type}")

        evidence_records = [evidence[evidence_id] for evidence_id in sorted(evidence_ids)]
        for evidence_record in evidence_records:
            _validate_evidence_binding(
                evidence_record,
                case=case,
                decision=decision,
                resolution=resolution,
                supersedes_decision_id=supersedes_decision_id,
            )
        if case_type == "match_uid_remap":
            primary_records = [
                record
                for record in evidence_records
                if record["evidence_kind"]
                in {"bookmaker_event_record", "identity_capture_record"}
            ]
            if not primary_records:
                raise ValueError(
                    f"UID remap requires a structured identity capture artifact: "
                    f"{case_id}"
                )
            for primary_record in primary_records:
                _validate_identity_capture_artifact(
                    primary_record,
                    case=case,
                    resolution=resolution,
                )
        else:
            primary_records = [
                record
                for record in evidence_records
                if record["evidence_kind"] == "operator_intent_record"
            ]
            if not primary_records:
                raise ValueError(
                    f"duplicate disposition requires a structured operator intent "
                    f"record: {case_id}"
                )
            for primary_record in primary_records:
                _validate_operator_intent_artifact(
                    primary_record,
                    case=case,
                    decision=decision,
                    resolution=resolution,
                )
        entry_without_id = {
            "subject_key": subject_key,
            "case_id": case_id,
            "case_type": case_type,
            "decision": decision,
            "resolution": resolution,
            "reviewer": reviewer,
            "reviewed_at_utc": reviewed_at,
            "reason": reason,
            "report_id": report["report_id"],
            "report_digest": report["report_digest"],
            "report_inputs": report["inputs"],
            "decision_document": _file_descriptor(decision_path),
            "supersedes_decision_id": supersedes_decision_id,
            "evidence": evidence_records,
        }
        approved.append(
            {
                **entry_without_id,
                "decision_id": "pending_identity_decision_"
                + _payload_sha256(entry_without_id)[:24],
                "record_sha256": _payload_sha256(entry_without_id),
            }
        )

    unused_evidence = sorted(set(evidence) - used_evidence_ids)
    if unused_evidence:
        raise ValueError(
            "decision document contains unreferenced evidence: "
            + ", ".join(unused_evidence)
        )
    approved.sort(key=lambda entry: (entry["case_id"], entry["decision_id"]))
    deferred.sort(key=lambda entry: entry["case_id"])
    return approved, deferred


def _empty_registry() -> dict[str, Any]:
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_generation": 0,
        "entries": [],
    }


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_registry()
    registry = _load_json(path, label="identity registry")
    if registry.get("registry_schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported registry schema version")
    generation = registry.get("registry_generation")
    entries = registry.get("entries")
    if not isinstance(generation, int) or generation < 0:
        raise ValueError("registry_generation must be a nonnegative integer")
    if not isinstance(entries, list):
        raise ValueError("registry entries must be a list")
    seen_decisions: set[str] = set()
    active_by_subject: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("registry entries must be objects")
        case_id = _clean(entry.get("case_id"))
        subject_key = _clean(entry.get("subject_key"))
        decision_id = _clean(entry.get("decision_id"))
        record_hash = _clean(entry.get("record_sha256"))
        supersedes = _clean(entry.get("supersedes_decision_id"))
        if (
            not case_id
            or not re.fullmatch(r"pending_identity_subject_[0-9a-f]{24}", subject_key)
            or not re.fullmatch(r"pending_identity_decision_[0-9a-f]{24}", decision_id)
            or not HEX64_RE.fullmatch(record_hash)
        ):
            raise ValueError("registry entry identity/hash is invalid")
        if decision_id in seen_decisions:
            raise RemediationConflict(f"registry repeats decision_id: {decision_id}")
        seen_decisions.add(decision_id)
        without_hashes = {
            key: value
            for key, value in entry.items()
            if key not in {"decision_id", "record_sha256"}
        }
        expected_id = "pending_identity_decision_" + _payload_sha256(without_hashes)[:24]
        if expected_id != decision_id or _payload_sha256(without_hashes) != record_hash:
            raise RemediationConflict(f"registry entry digest mismatch: {case_id}")
        prior = active_by_subject.get(subject_key)
        if prior is None and supersedes:
            raise RemediationConflict(
                f"registry subject starts with an invalid supersession: {subject_key}"
            )
        if prior is not None and supersedes != prior["decision_id"]:
            raise RemediationConflict(
                f"registry subject supersession chain is invalid: {subject_key}"
            )
        active_by_subject[subject_key] = entry
    if generation != len(entries):
        raise RemediationConflict("registry generation must equal immutable entry count")
    return registry


def _descriptor_matches(expected: Mapping[str, Any], path: Path) -> bool:
    exists = path.is_file()
    return (
        bool(expected.get("exists")) == exists
        and expected.get("sha256") == (_file_sha256(path) if exists else None)
        and int(expected.get("byte_size", 0)) == (path.stat().st_size if exists else 0)
    )


def build_apply_plan(
    *,
    report_path: Path,
    decision_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    report_path = report_path.resolve()
    decision_path = decision_path.resolve()
    registry_path = registry_path.resolve()
    report = _load_json(report_path, label="review report")
    decisions = _load_json(decision_path, label="decision document")
    approved, deferred = validate_decisions(
        report, decisions, decision_path=decision_path
    )
    registry = load_registry(registry_path)
    existing_by_decision = {
        entry["decision_id"]: entry for entry in registry["entries"]
    }
    active_by_subject: dict[str, dict[str, Any]] = {}
    for existing_entry in registry["entries"]:
        active_by_subject[existing_entry["subject_key"]] = existing_entry
    additions: list[dict[str, Any]] = []
    replayed: list[str] = []
    for entry in approved:
        added = False
        exact = existing_by_decision.get(entry["decision_id"])
        active = active_by_subject.get(entry["subject_key"])
        if exact is not None:
            if exact != entry:
                raise RemediationConflict(
                    f"registry decision ID collision: {entry['decision_id']}"
                )
            if active is None or active["decision_id"] != entry["decision_id"]:
                raise RemediationConflict(
                    f"submitted decision has been superseded: {entry['subject_key']}"
                )
            replayed.append(entry["case_id"])
        elif active is None:
            if entry["supersedes_decision_id"]:
                raise RemediationConflict(
                    f"new registry subject cannot supersede another decision: "
                    f"{entry['subject_key']}"
                )
            additions.append(entry)
            added = True
        else:
            if entry["supersedes_decision_id"] != active["decision_id"]:
                raise RemediationConflict(
                    f"registry subject requires explicit active supersession: "
                    f"{entry['subject_key']}"
                )
            additions.append(entry)
            added = True
        if added:
            active_by_subject[entry["subject_key"]] = entry

    additions.sort(key=lambda entry: (entry["case_id"], entry["decision_id"]))
    inputs = {
        "report": _file_descriptor(report_path),
        "decisions": _file_descriptor(decision_path),
        "registry": _file_descriptor(registry_path),
        "operational_sources": report["inputs"],
    }
    next_registry = {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_generation": registry["registry_generation"] + len(additions),
        "entries": registry["entries"] + additions,
    }
    without_digest: dict[str, Any] = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "operation": "append_pending_identity_decisions",
        "canonical_bet_log_mutation": False,
        "settlement_mutation": False,
        "inputs": inputs,
        "summary": {
            "approved_decisions": len(approved),
            "new_registry_entries": len(additions),
            "verified_replays": len(replayed),
            "deferred_cases": len(deferred),
            "registry_generation_before": registry["registry_generation"],
            "registry_generation_after": next_registry["registry_generation"],
        },
        "additions": additions,
        "verified_replay_case_ids": sorted(replayed),
        "deferred": deferred,
        "post_registry_sha256": _payload_sha256(next_registry),
        "post_registry": next_registry,
    }
    return {**without_digest, "plan_digest": _payload_sha256(without_digest)}


def _validate_plan_digest(plan: Mapping[str, Any], expected_digest: str) -> None:
    if plan.get("plan_schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported apply plan schema version")
    digest = _clean(plan.get("plan_digest"))
    without = {key: value for key, value in plan.items() if key != "plan_digest"}
    if not HEX64_RE.fullmatch(expected_digest) or digest != expected_digest:
        raise RemediationConflict("expected plan digest mismatch")
    if _payload_sha256(without) != digest:
        raise RemediationConflict("apply plan digest mismatch")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _registry_lock(registry_path: Path) -> Iterable[None]:
    lock_path = registry_path.resolve().parent / ".pending_identity_registry.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_registry(registry: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        registry,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_plan(
    *,
    plan_path: Path,
    expected_plan_digest: str,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = _load_json(plan_path, label="apply plan")
    _validate_plan_digest(plan, expected_plan_digest)
    inputs = plan.get("inputs", {})
    registry_path = Path(inputs["registry"]["path"]).resolve()

    with _registry_lock(registry_path):
        report_path = Path(inputs["report"]["path"]).resolve()
        decision_path = Path(inputs["decisions"]["path"]).resolve()
        for label, path in (
            ("report", report_path),
            ("decisions", decision_path),
            ("registry", registry_path),
        ):
            if not _descriptor_matches(inputs[label], path):
                raise RemediationConflict(f"stale apply input: {label}")
        report = _load_json(report_path, label="review report")
        _verify_report_exact(report)
        rebuilt = build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=registry_path,
        )
        if rebuilt != plan:
            raise RemediationConflict("rebuilt plan differs from reviewed plan")

        if plan["summary"]["new_registry_entries"] == 0:
            return {
                "status": "verified_noop",
                "plan_digest": expected_plan_digest,
                "registry_generation": plan["summary"]["registry_generation_after"],
            }
        _atomic_write_registry(plan["post_registry"], registry_path)
        written = load_registry(registry_path)
        if _payload_sha256(written) != plan["post_registry_sha256"]:
            raise RemediationConflict("written registry hash mismatch")
        return {
            "status": "applied",
            "plan_digest": expected_plan_digest,
            "new_registry_entries": plan["summary"]["new_registry_entries"],
            "registry_generation": written["registry_generation"],
            "registry_sha256": _file_sha256(registry_path),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidence-bound pending identity review registry"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="write deterministic review cases")
    report.add_argument("--prod-dir", default=".")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--decision-template", type=Path)

    plan = subparsers.add_parser("plan", help="validate decisions and write an apply plan")
    plan.add_argument("--report", type=Path, required=True)
    plan.add_argument("--decisions", type=Path, required=True)
    plan.add_argument("--registry", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    apply = subparsers.add_parser("apply", help="append a reviewed plan to the registry")
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--expected-plan-digest", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "report":
        report = build_review_report(RemediationPaths.from_prod_dir(args.prod_dir))
        write_review_report(report, args.output)
        if args.decision_template:
            _write_immutable_json(build_decision_template(report), args.decision_template)
        print(
            json.dumps(
                {
                    "report_id": report["report_id"],
                    "report_digest": report["report_digest"],
                    "summary": report["summary"],
                    "automatic_approvals": 0,
                    "candidate_is_authority": False,
                    "output": str(args.output.resolve()),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.command == "plan":
        plan = build_apply_plan(
            report_path=args.report,
            decision_path=args.decisions,
            registry_path=args.registry,
        )
        _write_immutable_json(plan, args.output)
        print(
            json.dumps(
                {
                    "plan_digest": plan["plan_digest"],
                    "summary": plan["summary"],
                    "canonical_bet_log_mutation": False,
                    "settlement_mutation": False,
                    "output": str(args.output.resolve()),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    result = apply_plan(
        plan_path=args.plan,
        expected_plan_digest=args.expected_plan_digest,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
