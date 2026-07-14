"""Audit and explicitly reconcile safe pending paper bets.

The default command deliberately stops at evidence assembly.  It does not call
the bet tracker, does not scrape a result source, and never updates an input
CSV.  The prediction log is the only authoritative winner source: a result is
exact only when one ``match_uid`` resolves to one valid normalized winner
identity and no void/cancellation marker conflicts with it.

Run from ``production/``::

    ../tennis_env/bin/python -m operations.pending_reconciliation --prod-dir .

Review files are optional and are written only when an explicit output path is
provided::

    ../tennis_env/bin/python -m operations.pending_reconciliation \
        --prod-dir . --output-csv /tmp/pending-review.csv \
        --output-json /tmp/pending-review.json

Settlement is a separate two-step operation.  A deterministic plan must first
be written to an explicit path, reviewed, and then applied with both that plan
and its exact digest.  Apply takes an exclusive lock and atomically replaces
the bet, bankroll, session, and dedicated apply-audit files as one rollback
unit.  It never uses fuzzy or name-only fallback matching.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from logging_utils import normalize_name, normalize_text
from operations.operational_lock import (
    canonical_pending_reconciliation_transaction_path,
    canonical_pending_reconciliation_targets,
    canonical_operational_lock_path,
    operational_csv_lock,
    PENDING_RECONCILIATION_TARGET_ROLES,
)


RECONCILIATION_SCHEMA_VERSION = "1.1.0"
SETTLEMENT_PLAN_SCHEMA_VERSION = "1.0.0"
APPLY_AUDIT_SCHEMA_VERSION = "1.0.0"

EXACT_WINNER = "exact_authoritative_winner_available"
DUPLICATE_IDENTITY = "duplicate_pending_match_side_date_identity"
ORPHAN_MATCH_UID = "orphan_match_uid_absent_from_prediction_log"
UNRESOLVED = "unresolved_or_ambiguous"

REVIEW_LABEL_ORDER = (
    EXACT_WINNER,
    DUPLICATE_IDENTITY,
    ORPHAN_MATCH_UID,
    UNRESOLVED,
)
OUTCOME_CLASSIFICATIONS = (EXACT_WINNER, ORPHAN_MATCH_UID, UNRESOLVED)
VOID_VALUES = {-1}
VOID_STATUSES = {"void", "voided", "cancel", "cancelled", "canceled"}
APPLY_UNSAFE_TERMINAL_STATUSES = VOID_STATUSES | {
    "abandoned",
    "postponed",
    "retired",
    "retirement",
    "suspended",
    "unfinished",
    "walkover",
    "w/o",
}

REQUIRED_BET_COLUMNS = {
    "bet_id",
    "status",
    "match",
    "match_uid",
    "bet_on",
    "match_date",
    "stake",
}
REQUIRED_PREDICTION_COLUMNS = {"match_uid", "actual_winner", "p1", "p2"}

PLAN_REQUIRED_BET_COLUMNS = REQUIRED_BET_COLUMNS | {
    "bet_on_player1",
    "odds_decimal",
    "session_id",
    "timestamp",
    "outcome",
    "actual_profit",
    "bankroll_after",
    "settled_timestamp",
    "notes",
    "edge",
}
PLAN_REQUIRED_PREDICTION_COLUMNS = REQUIRED_PREDICTION_COLUMNS | {
    "match_date",
    "settled_at",
}
PLAN_REQUIRED_BANKROLL_COLUMNS = {
    "timestamp",
    "session_id",
    "bankroll",
    "change_amount",
    "change_reason",
    "total_staked",
    "num_pending_bets",
    "num_settled_bets",
}
BANKROLL_OUTPUT_COLUMNS = (
    "timestamp",
    "session_id",
    "bankroll",
    "change_amount",
    "change_reason",
    "account_equity",
    "pending_exposure",
    "available_bankroll",
    "total_staked",
    "num_pending_bets",
    "num_settled_bets",
)
PLAN_REQUIRED_SESSION_COLUMNS = {
    "session_id",
    "start_time",
    "end_time",
    "initial_bankroll",
    "final_bankroll",
    "total_bets_placed",
    "total_staked",
    "total_profit_loss",
    "win_rate",
    "avg_odds",
    "avg_edge",
}

APPLY_AUDIT_COLUMNS = (
    "audit_schema_version",
    "event_id",
    "event_payload_sha256",
    "plan_digest",
    "plan_schema_version",
    "event_sequence",
    "event_type",
    "applied_at_utc",
    "bet_id",
    "match_uid",
    "session_id",
    "outcome",
    "actual_profit",
    "authoritative_winner_name",
    "pending_identity_key",
    "bet_row_sha256",
    "prediction_rows_sha256",
    "session_row_sha256",
    "bet_post_row_sha256",
    "bankroll_event_row_sha256",
    "session_post_row_sha256",
    "reviewed_post_state_sha256",
    "bets_input_sha256",
    "predictions_input_sha256",
    "bankroll_input_sha256",
    "sessions_input_sha256",
    "apply_audit_input_sha256",
)

REVIEW_COLUMNS = (
    "source_row",
    "bet_id",
    "pending_identity_key",
    "identity_match",
    "identity_bet_on",
    "identity_match_date",
    "duplicate_group_size",
    "duplicate_group_position",
    "is_duplicate_pending_identity",
    "match_uid",
    "match_uid_in_prediction_log",
    "outcome_classification",
    "review_classifications",
    "authoritative_outcome_status",
    "authoritative_actual_winner",
    "authoritative_winner_name",
    "prediction_source_rows",
    "bet_result_if_applied",
    "profit_if_applied",
    "stake",
    "odds_decimal",
    "match",
    "bet_on",
    "match_date",
    "timestamp",
    "event",
    "session_id",
    "feature_snapshot_id",
    "run_id",
)


@dataclass(frozen=True)
class SettlementPaths:
    """Every input/output path bound into a settlement plan.

    ``apply_audit`` is supplied explicitly but must equal the canonical private
    recovery target. It is not the settlement-source evidence audit.
    """

    bets: Path
    predictions: Path
    bankroll: Path
    sessions: Path
    apply_audit: Path
    lock: Path
    transaction_dir: Path

    def resolved(self) -> "SettlementPaths":
        return SettlementPaths(
            bets=self.bets.resolve(),
            predictions=self.predictions.resolve(),
            bankroll=self.bankroll.resolve(),
            sessions=self.sessions.resolve(),
            apply_audit=self.apply_audit.resolve(),
            lock=self.lock.resolve(),
            transaction_dir=self.transaction_dir.resolve(),
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "bets": self.bets,
            "predictions": self.predictions,
            "bankroll": self.bankroll,
            "sessions": self.sessions,
            "apply_audit": self.apply_audit,
        }


class ReconciliationConflict(RuntimeError):
    """The reviewed plan no longer matches the operational state."""


class AtomicApplyError(RuntimeError):
    """A file-set commit failed and the original files were restored."""


def _validate_settlement_paths(paths: SettlementPaths) -> None:
    paths = paths.resolved()
    named = {
        **paths.as_dict(),
        "lock": paths.lock,
        "transaction_dir": paths.transaction_dir,
    }
    reverse: dict[Path, list[str]] = {}
    for name, path in named.items():
        reverse.setdefault(path, []).append(name)
    collisions = [names for names in reverse.values() if len(names) > 1]
    if collisions:
        raise ValueError(f"settlement paths must be distinct: {collisions}")
    canonical_lock = canonical_operational_lock_path(paths.bets.parent)
    if paths.lock != canonical_lock:
        raise ValueError(
            f"lock path must be the canonical operational lock: {canonical_lock}"
        )
    canonical_transaction = canonical_pending_reconciliation_transaction_path(
        paths.bets.parent
    )
    if paths.transaction_dir != canonical_transaction:
        raise ValueError(
            "transaction directory must be the canonical recovery path: "
            f"{canonical_transaction}"
        )
    canonical_targets = dict(
        zip(
            PENDING_RECONCILIATION_TARGET_ROLES,
            canonical_pending_reconciliation_targets(paths.bets.parent),
        )
    )
    supplied_targets = {
        "bets": paths.bets,
        "bankroll": paths.bankroll,
        "sessions": paths.sessions,
        "apply_audit": paths.apply_audit,
    }
    wrong_targets = [
        name
        for name, path in supplied_targets.items()
        if path != canonical_targets[name]
    ]
    if wrong_targets:
        expected = {
            name: str(canonical_targets[name]) for name in wrong_targets
        }
        raise ValueError(
            "plan/apply operational targets must use the canonical recovery "
            f"allowlist: {expected}"
        )
    nested_targets = [
        name
        for name, path in paths.as_dict().items()
        if paths.transaction_dir in path.parents
    ]
    if nested_targets:
        raise ValueError(
            "operational paths must not be inside the transaction recovery "
            f"directory: {nested_targets}"
        )


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_csv_text(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} file has no CSV header: {path}") from exc


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _row_sha256(row: pd.Series, columns: Sequence[str]) -> str:
    payload = {column: _clean(row.get(column, "")) for column in columns}
    return _payload_sha256(payload)


def _file_descriptor(path: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _file_sha256(path) if path.is_file() else None,
    }
    if frame is not None:
        descriptor["rows"] = int(len(frame))
        descriptor["columns"] = list(frame.columns)
    return descriptor


def _decimal_value(value: Any) -> Decimal | None:
    text = _clean(value)
    if not text:
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _positive_decimal(value: Any) -> Decimal | None:
    number = _decimal_value(value)
    return number if number is not None and number > 0 else None


def _price_decimal(value: Any) -> Decimal | None:
    number = _decimal_value(value)
    return number if number is not None and number > 1 else None


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _strict_date(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    # Match date is a calendar identity, not an instant.  Do not shift an
    # explicitly offset source value across dates by converting it to UTC.
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _strict_timestamp(value: Any) -> pd.Timestamp | None:
    text = _clean(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed


def _strict_bool(value: Any) -> bool | None:
    text = _clean(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _ordered_match_players(value: Any) -> tuple[str, str] | None:
    players = re.split(r"\s+vs\.?\s+", _clean(value), maxsplit=1, flags=re.IGNORECASE)
    if len(players) != 2:
        return None
    normalized = tuple(normalize_name(player) for player in players)
    if not all(normalized) or normalized[0] == normalized[1]:
        return None
    return normalized


def _canonical_match(value: Any) -> str:
    """Return a side/order-insensitive normalized match identity."""
    text = _clean(value)
    players = re.split(r"\s+vs\.?\s+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(players) == 2:
        normalized = sorted(normalize_name(player) for player in players)
        if all(normalized):
            return " vs ".join(normalized)
    return normalize_text(text)


def _canonical_date(value: Any) -> str:
    text = _clean(value)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.date().isoformat()
    return normalize_text(text)


def _identity_parts(row: pd.Series) -> tuple[str, str, str]:
    return (
        _canonical_match(row.get("match")),
        normalize_name(_clean(row.get("bet_on"))),
        _canonical_date(row.get("match_date")),
    )


def _identity_key(parts: Iterable[str]) -> str:
    payload = json.dumps(list(parts), ensure_ascii=True, separators=(",", ":"))
    return "pending_bet_identity_" + sha256(payload.encode("utf-8")).hexdigest()[:20]


def _parse_numeric_winner(value: Any) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    if not number.is_integer():
        return None
    return int(number)


def _prediction_terminal_status(row: pd.Series) -> str:
    for column in ("record_status", "outcome", "status"):
        status = normalize_text(row.get(column, ""))
        if status in VOID_STATUSES:
            return "void_or_cancelled"
    winner = _parse_numeric_winner(row.get("actual_winner"))
    if winner in VOID_VALUES:
        return "void_or_cancelled"
    return ""


@dataclass(frozen=True)
class OutcomeEvidence:
    match_uid: str
    in_prediction_log: bool
    status: str
    actual_winner: int | None = None
    winner_name: str = ""
    player_names: tuple[str, ...] = ()
    player_pairs: tuple[tuple[str, str], ...] = ()
    match_dates: tuple[str, ...] = ()
    source_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class SafeOutcomeResolution:
    """Strict source evidence used by the apply planner."""

    match_uid: str
    reasons: tuple[str, ...]
    winner_name: str = ""
    player_pair: tuple[str, str] = ()
    match_date: str = ""
    settlement_effective_at: str = ""
    winner_source_values: tuple[int, ...] = ()
    source_rows: tuple[int, ...] = ()
    source_row_hashes: tuple[str, ...] = ()


def _dedupe_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reason for reason in reasons if reason))


def _resolve_safe_outcome(
    match_uid: str,
    group: pd.DataFrame,
    *,
    prediction_columns: Sequence[str],
) -> SafeOutcomeResolution:
    """Resolve one UID without any pair, date, or name fallback.

    Repeated hourly observations are allowed only when every row carries the
    same exact normalized pair/date semantic identity.  Numeric P1/P2 labels
    may reverse with row orientation, but every valid label must resolve to one
    normalized winner identity.
    """
    reasons: list[str] = []
    semantic_keys: set[tuple[tuple[str, str], str]] = set()
    winner_names: set[str] = set()
    winner_values: set[int] = set()
    invalid_identity = False
    has_void = False
    has_invalid_nonblank_winner = False
    settlement_timestamps: list[pd.Timestamp] = []
    semantic_metadata: dict[str, set[str]] = {
        column: set() for column in ("tournament", "round", "surface")
        if column in group.columns
    }

    source_rows: list[int] = []
    source_hashes: list[str] = []
    for _, row in group.iterrows():
        source_rows.append(int(row["_source_row"]))
        source_hashes.append(_row_sha256(row, prediction_columns))
        for column, values in semantic_metadata.items():
            value = normalize_text(row.get(column, ""))
            if value:
                values.add(value)

        p1 = normalize_name(_clean(row.get("p1")))
        p2 = normalize_name(_clean(row.get("p2")))
        match_date = _strict_date(row.get("match_date"))
        if not p1 or not p2 or p1 == p2 or not match_date:
            invalid_identity = True
        else:
            semantic_keys.add((tuple(sorted((p1, p2))), match_date))

        terminal = _prediction_terminal_status(row)
        explicit_statuses = {
            normalize_text(row.get(column, ""))
            for column in ("record_status", "outcome", "status")
        }
        has_void = has_void or terminal == "void_or_cancelled" or bool(
            explicit_statuses & APPLY_UNSAFE_TERMINAL_STATUSES
        )
        raw_winner = _clean(row.get("actual_winner"))
        if not raw_winner:
            continue
        winner = _parse_numeric_winner(raw_winner)
        if winner not in (1, 2):
            has_invalid_nonblank_winner = True
            continue
        winner_values.add(winner)
        winner_name = p1 if winner == 1 else p2
        if winner_name:
            winner_names.add(winner_name)
        else:
            has_invalid_nonblank_winner = True
        settled_at = _strict_timestamp(row.get("settled_at"))
        if settled_at is None:
            reasons.append("missing_authoritative_settled_at")
        else:
            settlement_timestamps.append(settled_at)

    if invalid_identity:
        reasons.append("incomplete_prediction_identity")
    if len(semantic_keys) != 1:
        reasons.append("reused_or_ambiguous_match_uid")
    if any(len(values) > 1 for values in semantic_metadata.values()):
        reasons.append("reused_or_ambiguous_match_uid")
    if has_void:
        reasons.append("void_or_cancelled_source_conflict")
    if has_invalid_nonblank_winner:
        reasons.append("invalid_nonblank_winner_value")
    if not winner_values:
        reasons.append("no_valid_winner_value")
    if len(winner_names) != 1:
        reasons.append("conflicting_or_missing_winner_identity")
    if winner_values and not settlement_timestamps:
        reasons.append("missing_authoritative_settled_at")

    pair: tuple[str, str] = ()
    match_date = ""
    if len(semantic_keys) == 1:
        pair, match_date = next(iter(semantic_keys))
    winner_name = next(iter(winner_names)) if len(winner_names) == 1 else ""
    settlement_effective_at = (
        max(settlement_timestamps).isoformat() if settlement_timestamps else ""
    )
    if pair and winner_name and winner_name not in pair:
        reasons.append("winner_not_in_prediction_pair")

    return SafeOutcomeResolution(
        match_uid=match_uid,
        reasons=_dedupe_reasons(reasons),
        winner_name=winner_name,
        player_pair=pair,
        match_date=match_date,
        settlement_effective_at=settlement_effective_at,
        winner_source_values=tuple(sorted(winner_values)),
        source_rows=tuple(source_rows),
        source_row_hashes=tuple(source_hashes),
    )


def build_outcome_evidence(predictions: pd.DataFrame) -> dict[str, OutcomeEvidence]:
    """Build one conservative outcome record per non-empty ``match_uid``."""
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction log missing required columns: {sorted(missing)}")

    frame = predictions.reset_index(drop=True).copy()
    frame["_source_row"] = frame.index + 2  # one-based CSV line, including header
    frame["_match_uid"] = frame["match_uid"].map(_clean)
    evidence: dict[str, OutcomeEvidence] = {}

    for match_uid, group in frame[frame["_match_uid"] != ""].groupby(
        "_match_uid", sort=True, dropna=False
    ):
        valid_rows: list[tuple[int, str, tuple[str, str]]] = []
        has_void = False
        all_players: set[str] = set()
        player_pairs: set[tuple[str, str]] = set()
        match_dates: set[str] = set()
        for _, row in group.iterrows():
            p1 = normalize_name(_clean(row.get("p1")))
            p2 = normalize_name(_clean(row.get("p2")))
            all_players.update(name for name in (p1, p2) if name)
            if p1 and p2:
                player_pairs.add(tuple(sorted((p1, p2))))
            match_date = _canonical_date(row.get("match_date"))
            if match_date:
                match_dates.add(match_date)
            has_void = has_void or _prediction_terminal_status(row) == "void_or_cancelled"
            winner = _parse_numeric_winner(row.get("actual_winner"))
            if winner not in (1, 2):
                continue
            winner_name = p1 if winner == 1 else p2
            valid_rows.append((winner, winner_name, (p1, p2)))

        numeric_winners = {winner for winner, _, _ in valid_rows}
        winner_names = {name for _, name, _ in valid_rows if name}
        source_rows = tuple(int(value) for value in group["_source_row"].tolist())

        if len(player_pairs) > 1 or len(match_dates) > 1:
            status = "ambiguous_match_identity"
            actual_winner = None
            winner_name = ""
        elif has_void and valid_rows:
            status = "ambiguous_terminal_conflict"
            actual_winner = None
            winner_name = ""
        elif has_void:
            status = "void_or_cancelled"
            actual_winner = None
            winner_name = ""
        elif valid_rows and len(winner_names) == 1:
            status = "exact_winner"
            # Numeric winner is orientation-dependent. Preserve it only when
            # every observation uses the same p1/p2 orientation; the stable
            # authority for reconciliation is the single winner identity.
            actual_winner = next(iter(numeric_winners)) if len(numeric_winners) == 1 else None
            winner_name = next(iter(winner_names))
        elif valid_rows:
            status = "ambiguous_conflicting_winners"
            actual_winner = None
            winner_name = ""
        else:
            status = "unresolved"
            actual_winner = None
            winner_name = ""

        evidence[match_uid] = OutcomeEvidence(
            match_uid=match_uid,
            in_prediction_log=True,
            status=status,
            actual_winner=actual_winner,
            winner_name=winner_name,
            player_names=tuple(sorted(all_players)),
            player_pairs=tuple(sorted(player_pairs)),
            match_dates=tuple(sorted(match_dates)),
            source_rows=source_rows,
        )
    return evidence


def _finite_number(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    return number if math.isfinite(number) else None


def _stake_number(value: Any) -> float | None:
    return _finite_number(value)


def _valid_stake(value: Any) -> bool:
    number = _stake_number(value)
    return number is not None and number > 0


def _valid_decimal_odds(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and number > 1


def _bet_result(row: pd.Series, outcome: OutcomeEvidence) -> tuple[str, float | None]:
    if outcome.status != "exact_winner":
        return "", None
    bet_on = normalize_name(_clean(row.get("bet_on")))
    if not bet_on or bet_on not in outcome.player_names:
        return "ambiguous_bet_side", None
    result = "win" if bet_on == outcome.winner_name else "loss"
    stake = _stake_number(row.get("stake"))
    odds = _finite_number(row.get("odds_decimal"))
    if not _valid_stake(stake):
        return result, None
    if result == "loss":
        return result, -stake
    if not _valid_decimal_odds(odds):
        return result, None
    return result, stake * (odds - 1.0)


def build_pending_review(
    bets: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return a deterministic review row for every pending paper bet.

    Outcome state is a partition: exact winner, orphan UID, or unresolved.
    Duplicate identity is an independent review label because a duplicate can
    also have an exact result or an orphan UID.
    """
    missing = REQUIRED_BET_COLUMNS - set(bets.columns)
    if missing:
        raise ValueError(f"bet log missing required columns: {sorted(missing)}")

    outcomes = build_outcome_evidence(predictions)
    bet_rows = bets.reset_index(drop=True).copy()
    bet_rows["source_row"] = bet_rows.index + 2
    pending = bet_rows[
        bet_rows["status"].fillna("").astype(str).str.strip().str.lower().eq("pending")
    ].copy()
    if pending.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    identity_parts = [_identity_parts(row) for _, row in pending.iterrows()]
    pending["identity_match"] = [parts[0] for parts in identity_parts]
    pending["identity_bet_on"] = [parts[1] for parts in identity_parts]
    pending["identity_match_date"] = [parts[2] for parts in identity_parts]
    pending["pending_identity_key"] = [_identity_key(parts) for parts in identity_parts]
    pending["duplicate_group_size"] = pending.groupby(
        "pending_identity_key", sort=False
    )["pending_identity_key"].transform("size")

    pending = pending.sort_values(
        ["pending_identity_key", "timestamp", "bet_id", "source_row"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    pending["duplicate_group_position"] = (
        pending.groupby("pending_identity_key", sort=False).cumcount() + 1
    )

    review_rows: list[dict[str, Any]] = []
    for _, row in pending.iterrows():
        match_uid = _clean(row.get("match_uid"))
        evidence = outcomes.get(
            match_uid,
            OutcomeEvidence(
                match_uid=match_uid,
                in_prediction_log=False,
                status="missing_match_uid" if not match_uid else "absent_from_prediction_log",
            ),
        )
        duplicate = int(row["duplicate_group_size"]) > 1
        if evidence.status == "exact_winner":
            outcome_classification = EXACT_WINNER
        elif not evidence.in_prediction_log:
            outcome_classification = ORPHAN_MATCH_UID
        else:
            outcome_classification = UNRESOLVED

        labels = [outcome_classification]
        if duplicate:
            labels.append(DUPLICATE_IDENTITY)
        labels = [label for label in REVIEW_LABEL_ORDER if label in labels]
        bet_result, profit = _bet_result(row, evidence)

        review_rows.append(
            {
                "source_row": int(row["source_row"]),
                "bet_id": _clean(row.get("bet_id")),
                "pending_identity_key": row["pending_identity_key"],
                "identity_match": row["identity_match"],
                "identity_bet_on": row["identity_bet_on"],
                "identity_match_date": row["identity_match_date"],
                "duplicate_group_size": int(row["duplicate_group_size"]),
                "duplicate_group_position": int(row["duplicate_group_position"]),
                "is_duplicate_pending_identity": duplicate,
                "match_uid": match_uid,
                "match_uid_in_prediction_log": evidence.in_prediction_log,
                "outcome_classification": outcome_classification,
                "review_classifications": "|".join(labels),
                "authoritative_outcome_status": evidence.status,
                "authoritative_actual_winner": evidence.actual_winner,
                "authoritative_winner_name": evidence.winner_name,
                "prediction_source_rows": ";".join(str(value) for value in evidence.source_rows),
                "bet_result_if_applied": bet_result,
                "profit_if_applied": profit,
                "stake": _stake_number(row.get("stake")),
                "odds_decimal": _finite_number(row.get("odds_decimal")),
                "match": _clean(row.get("match")),
                "bet_on": _clean(row.get("bet_on")),
                "match_date": _clean(row.get("match_date")),
                "timestamp": _clean(row.get("timestamp")),
                "event": _clean(row.get("event")),
                "session_id": _clean(row.get("session_id")),
                "feature_snapshot_id": _clean(row.get("feature_snapshot_id")),
                "run_id": _clean(row.get("run_id")),
            }
        )

    return pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)


def _rounded_sum(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.map(_valid_stake)
    return round(float(numeric[valid].sum()), 6)


def _bucket(review: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    subset = review[mask]
    return {
        "rows": int(len(subset)),
        "stake_total": _rounded_sum(subset["stake"]),
        "bet_ids": sorted(subset["bet_id"].astype(str).tolist()),
    }


def build_summary(
    review: pd.DataFrame,
    *,
    bets_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    """Build a stable, JSON-serializable reconciliation summary."""
    if review.empty:
        outcome_counts = {name: {"rows": 0, "stake_total": 0.0, "bet_ids": []}
                          for name in OUTCOME_CLASSIFICATIONS}
        duplicate = {
            "rows": 0,
            "stake_total": 0.0,
            "bet_ids": [],
            "identities": 0,
            "identity_keys": [],
            "by_outcome_classification": {
                name: {"rows": 0, "stake_total": 0.0}
                for name in OUTCOME_CLASSIFICATIONS
            },
        }
        invalid_stake_rows = 0
        invalid_exact_odds = {"rows": 0, "bet_ids": []}
    else:
        outcome_counts = {
            name: _bucket(review, review["outcome_classification"].eq(name))
            for name in OUTCOME_CLASSIFICATIONS
        }
        duplicate_mask = review["is_duplicate_pending_identity"].astype(bool)
        duplicate = _bucket(review, duplicate_mask)
        duplicate["identities"] = int(
            review.loc[duplicate_mask, "pending_identity_key"].nunique()
        )
        duplicate["identity_keys"] = sorted(
            review.loc[duplicate_mask, "pending_identity_key"].unique().tolist()
        )
        duplicate["by_outcome_classification"] = {
            name: {
                "rows": int(
                    (duplicate_mask & review["outcome_classification"].eq(name)).sum()
                ),
                "stake_total": _rounded_sum(
                    review.loc[
                        duplicate_mask & review["outcome_classification"].eq(name),
                        "stake",
                    ]
                ),
            }
            for name in OUTCOME_CLASSIFICATIONS
        }
        invalid_stake_rows = int((~review["stake"].map(_valid_stake)).sum())
        exact_mask = review["outcome_classification"].eq(EXACT_WINNER)
        invalid_exact_odds_mask = exact_mask & ~review["odds_decimal"].map(
            _valid_decimal_odds
        )
        invalid_exact_odds = {
            "rows": int(invalid_exact_odds_mask.sum()),
            "bet_ids": sorted(
                review.loc[invalid_exact_odds_mask, "bet_id"].astype(str).tolist()
            ),
        }

    outcome_total = sum(bucket["rows"] for bucket in outcome_counts.values())
    inputs: dict[str, Any] = {}
    if bets_path is not None:
        inputs["bets"] = {"path": str(bets_path), "sha256": _file_sha256(bets_path)}
    if predictions_path is not None:
        inputs["predictions"] = {
            "path": str(predictions_path),
            "sha256": _file_sha256(predictions_path),
        }

    status_counts = Counter(review["authoritative_outcome_status"].astype(str))
    return {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "read_only": True,
        "inputs": inputs,
        "pending": {
            "rows": int(len(review)),
            "stake_total": _rounded_sum(review["stake"]) if not review.empty else 0.0,
            "invalid_stake_rows": invalid_stake_rows,
        },
        "invalid_exact_outcome_odds": invalid_exact_odds,
        "outcome_classifications": outcome_counts,
        DUPLICATE_IDENTITY: duplicate,
        "authoritative_outcome_status_counts": dict(sorted(status_counts.items())),
        "integrity": {
            "outcome_partition_rows": outcome_total,
            "every_pending_row_outcome_classified": outcome_total == len(review),
            "duplicate_is_orthogonal": True,
        },
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the human terminal view without row IDs or input hashes."""
    outcome_counts = {
        name: {
            "rows": bucket["rows"],
            "stake_total": bucket["stake_total"],
        }
        for name, bucket in summary["outcome_classifications"].items()
    }
    duplicate = summary[DUPLICATE_IDENTITY]
    inputs = {
        name: {"path": details["path"]}
        for name, details in summary.get("inputs", {}).items()
    }
    return {
        "schema_version": summary["schema_version"],
        "read_only": summary["read_only"],
        "inputs": inputs,
        "pending": summary["pending"],
        "outcome_classifications": outcome_counts,
        DUPLICATE_IDENTITY: {
            "rows": duplicate["rows"],
            "stake_total": duplicate["stake_total"],
            "identities": duplicate["identities"],
            "by_outcome_classification": duplicate["by_outcome_classification"],
        },
        "invalid_exact_outcome_odds": {
            "rows": summary["invalid_exact_outcome_odds"]["rows"],
        },
        "authoritative_outcome_status_counts": summary[
            "authoritative_outcome_status_counts"
        ],
        "integrity": summary["integrity"],
    }


def _json_safe(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def write_review_exports(
    review: pd.DataFrame,
    summary: dict[str, Any],
    *,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> None:
    """Write only caller-requested review artifacts; never infer a path."""
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        review.to_csv(output_csv, index=False, lineterminator="\n")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {column: _json_safe(value) for column, value in row.items()}
            for row in review.to_dict(orient="records")
        ]
        payload = {"summary": summary, "rows": rows}
        output_json.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )


def reconcile_paths(
    bets_path: Path,
    predictions_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bets_path = bets_path.resolve()
    predictions_path = predictions_path.resolve()
    bets = pd.read_csv(bets_path, low_memory=False)
    predictions = pd.read_csv(predictions_path, low_memory=False)
    review = build_pending_review(bets, predictions)
    summary = build_summary(
        review, bets_path=bets_path, predictions_path=predictions_path
    )
    return review, summary


def _strict_bet_identity(row: pd.Series) -> tuple[str, tuple[str, str], str, str] | None:
    ordered_players = _ordered_match_players(row.get("match"))
    match_date = _strict_date(row.get("match_date"))
    bet_on = normalize_name(_clean(row.get("bet_on")))
    if (
        ordered_players is None
        or not match_date
        or not bet_on
        or bet_on not in ordered_players
    ):
        return None
    pair = tuple(sorted(ordered_players))
    key = _identity_key((" vs ".join(pair), bet_on, match_date))
    return key, pair, match_date, bet_on


def _load_settlement_frames(
    paths: SettlementPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = paths.resolved()
    bets = _read_csv_text(paths.bets, label="bet log")
    predictions = _read_csv_text(paths.predictions, label="prediction log")
    bankroll = _read_csv_text(paths.bankroll, label="bankroll history")
    sessions = _read_csv_text(paths.sessions, label="betting sessions")
    _require_columns(bets, PLAN_REQUIRED_BET_COLUMNS, label="bet log")
    _require_columns(
        predictions,
        PLAN_REQUIRED_PREDICTION_COLUMNS,
        label="prediction log",
    )
    _require_columns(
        bankroll,
        PLAN_REQUIRED_BANKROLL_COLUMNS,
        label="bankroll history",
    )
    # Older accepted CSV generations predate the three derived account fields.
    # Pad them in memory only; planning stays read-only and a reviewed apply
    # upgrades the file while appending the first fully reconciled event.
    for column in BANKROLL_OUTPUT_COLUMNS:
        if column not in bankroll.columns:
            bankroll[column] = ""
    bankroll = bankroll[
        list(BANKROLL_OUTPUT_COLUMNS)
        + [column for column in bankroll.columns if column not in BANKROLL_OUTPUT_COLUMNS]
    ]
    _require_columns(
        sessions,
        PLAN_REQUIRED_SESSION_COLUMNS,
        label="betting sessions",
    )

    if paths.apply_audit.is_file():
        apply_audit = _read_csv_text(paths.apply_audit, label="apply audit")
        _require_columns(
            apply_audit,
            set(APPLY_AUDIT_COLUMNS),
            label="apply audit",
        )
        if tuple(apply_audit.columns) != APPLY_AUDIT_COLUMNS:
            raise ValueError(
                "apply audit columns/order do not match schema "
                f"{APPLY_AUDIT_SCHEMA_VERSION}"
            )
    else:
        apply_audit = pd.DataFrame(columns=APPLY_AUDIT_COLUMNS)
    return bets, predictions, bankroll, sessions, apply_audit


def _account_state(
    bets: pd.DataFrame,
    *,
    starting_capital: Decimal,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not starting_capital.is_finite() or starting_capital <= 0:
        blockers.append("invalid_starting_capital")

    status = bets["status"].map(lambda value: _clean(value).lower())
    allowed_statuses = {"pending", "settled", "void", "voided", "cancelled", "canceled"}
    settled_profit = Decimal("0")
    pending_exposure = Decimal("0")
    total_staked = Decimal("0")
    for index, row in bets.iterrows():
        stake = _positive_decimal(row.get("stake"))
        if stake is None:
            blockers.append(f"invalid_global_stake_at_source_row_{index + 2}")
        else:
            total_staked += stake

        odds = _price_decimal(row.get("odds_decimal"))
        if odds is None:
            blockers.append(f"invalid_global_odds_at_source_row_{index + 2}")
        if _decimal_value(row.get("edge")) is None:
            blockers.append(f"invalid_global_edge_at_source_row_{index + 2}")

        row_status = status.loc[index]
        if row_status not in allowed_statuses:
            blockers.append(f"unsupported_status_at_source_row_{index + 2}")
            continue
        if row_status == "settled":
            profit = _decimal_value(row.get("actual_profit"))
            outcome = _clean(row.get("outcome")).lower()
            if outcome not in {"win", "loss"}:
                blockers.append(
                    f"invalid_settled_outcome_at_source_row_{index + 2}"
                )
            if profit is None:
                blockers.append(
                    f"invalid_settled_profit_at_source_row_{index + 2}"
                )
            else:
                settled_profit += profit
                if stake is not None and odds is not None and outcome in {"win", "loss"}:
                    expected = stake * (odds - Decimal("1")) if outcome == "win" else -stake
                    if abs(profit - expected) > Decimal("1e-9"):
                        blockers.append(
                            f"settled_profit_semantics_mismatch_at_source_row_{index + 2}"
                        )
            if _strict_timestamp(row.get("settled_timestamp")) is None:
                blockers.append(
                    f"invalid_settled_timestamp_at_source_row_{index + 2}"
                )
            if _decimal_value(row.get("bankroll_after")) is None:
                blockers.append(
                    f"invalid_settled_bankroll_after_at_source_row_{index + 2}"
                )
        elif row_status == "pending":
            if stake is not None:
                pending_exposure += stake
            terminal_fields = (
                "outcome",
                "actual_profit",
                "bankroll_after",
                "settled_timestamp",
                "notes",
            )
            if any(_clean(row.get(column)) for column in terminal_fields):
                blockers.append(
                    f"pending_row_has_terminal_fields_at_source_row_{index + 2}"
                )
        elif row_status in {"void", "voided", "cancelled", "canceled"}:
            profit = _decimal_value(row.get("actual_profit"))
            outcome = _clean(row.get("outcome")).lower()
            if profit != Decimal("0") or outcome not in {
                "void", "voided", "cancel", "cancelled", "canceled"
            }:
                blockers.append(
                    f"invalid_void_semantics_at_source_row_{index + 2}"
                )
            elif profit is not None:
                settled_profit += profit
            if _strict_timestamp(row.get("settled_timestamp")) is None:
                blockers.append(
                    f"invalid_void_timestamp_at_source_row_{index + 2}"
                )

    equity = starting_capital + settled_profit
    if not equity.is_finite():
        blockers.append("nonfinite_account_equity")

    return {
        "starting_capital": _decimal_text(starting_capital),
        "settled_profit_before": _decimal_text(settled_profit),
        "account_equity_before": _decimal_text(equity),
        "pending_exposure_before": _decimal_text(pending_exposure),
        "total_staked": _decimal_text(total_staked),
        "global_blockers": list(_dedupe_reasons(blockers)),
    }


def _plan_digest(plan_without_digest: Mapping[str, Any]) -> str:
    return _payload_sha256(plan_without_digest)


def build_settlement_plan(
    paths: SettlementPaths,
    *,
    starting_capital: Decimal | str | float = Decimal("1000"),
) -> dict[str, Any]:
    """Build a deterministic, reviewable plan for strictly safe exact rows."""
    paths = paths.resolved()
    _validate_settlement_paths(paths)
    capital = _decimal_value(starting_capital)
    if capital is None:
        capital = Decimal("NaN")
    bets, predictions, bankroll, sessions, apply_audit = _load_settlement_frames(paths)

    bet_columns = list(bets.columns)
    prediction_columns = list(predictions.columns)
    bankroll_columns = list(bankroll.columns)
    session_columns = list(sessions.columns)

    bets_work = bets.reset_index(drop=True).copy()
    bets_work["_source_row"] = bets_work.index + 2
    predictions_work = predictions.reset_index(drop=True).copy()
    predictions_work["_source_row"] = predictions_work.index + 2
    predictions_work["_match_uid"] = predictions_work["match_uid"].map(_clean)
    sessions_work = sessions.reset_index(drop=True).copy()
    sessions_work["_source_row"] = sessions_work.index + 2
    bankroll_work = bankroll.reset_index(drop=True).copy()
    bankroll_work["_source_row"] = bankroll_work.index + 2

    bet_id_counts = Counter(bets_work["bet_id"].map(_clean))
    strict_identities = {
        int(row["_source_row"]): _strict_bet_identity(row)
        for _, row in bets_work.iterrows()
    }
    identity_counts = Counter(
        identity[0] for identity in strict_identities.values() if identity is not None
    )

    prediction_outcomes: dict[str, SafeOutcomeResolution] = {}
    for match_uid, group in predictions_work[
        predictions_work["_match_uid"] != ""
    ].groupby("_match_uid", sort=True, dropna=False):
        prediction_outcomes[match_uid] = _resolve_safe_outcome(
            match_uid,
            group,
            prediction_columns=prediction_columns,
        )

    session_ids = sessions_work["session_id"].map(_clean)
    session_counts = Counter(session_ids)
    bankroll_session_counts = Counter(bankroll_work["session_id"].map(_clean))
    account = _account_state(bets_work, starting_capital=capital)
    bankroll_timestamps = [
        _strict_timestamp(value) for value in bankroll_work["timestamp"].tolist()
    ]
    if not bankroll_timestamps or any(
        timestamp is None for timestamp in bankroll_timestamps
    ):
        account["global_blockers"].append("invalid_bankroll_event_timestamp")
    global_blockers = tuple(account["global_blockers"])

    provisional: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pending = bets_work[
        bets_work["status"].map(lambda value: _clean(value).lower()).eq("pending")
    ]
    for _, row in pending.iterrows():
        source_row = int(row["_source_row"])
        bet_id = _clean(row.get("bet_id"))
        match_uid = _clean(row.get("match_uid"))
        session_id = _clean(row.get("session_id"))
        reasons: list[str] = list(global_blockers)

        if any(
            _clean(row.get(column))
            for column in (
                "outcome",
                "actual_profit",
                "bankroll_after",
                "settled_timestamp",
                "notes",
            )
        ):
            reasons.append("pending_settlement_fields_not_blank")

        if not bet_id:
            reasons.append("blank_bet_id")
        elif bet_id_counts[bet_id] != 1:
            reasons.append("nonunique_bet_id")

        strict_identity = strict_identities[source_row]
        if strict_identity is None:
            reasons.append("invalid_bet_pair_date_or_side_identity")
            identity_key = ""
            bet_pair: tuple[str, str] = ()
            bet_date = ""
            bet_on = normalize_name(_clean(row.get("bet_on")))
        else:
            identity_key, bet_pair, bet_date, bet_on = strict_identity
            if identity_counts[identity_key] != 1:
                reasons.append("duplicate_bet_identity_any_status")

        ordered_bet_players = _ordered_match_players(row.get("match"))
        side_flag = _strict_bool(row.get("bet_on_player1"))
        if side_flag is None:
            reasons.append("invalid_bet_side_orientation_flag")
        elif ordered_bet_players is not None and bet_on in ordered_bet_players:
            if side_flag != (bet_on == ordered_bet_players[0]):
                reasons.append("bet_side_orientation_mismatch")

        outcome = prediction_outcomes.get(match_uid)
        if not match_uid:
            reasons.append("blank_match_uid")
        elif outcome is None:
            reasons.append("match_uid_absent_from_prediction_log")
        else:
            reasons.extend(outcome.reasons)
            if outcome.player_pair and bet_pair and outcome.player_pair != bet_pair:
                reasons.append("prediction_pair_mismatch")
            if outcome.match_date and bet_date and outcome.match_date != bet_date:
                reasons.append("prediction_date_mismatch")
            if outcome.winner_name and outcome.player_pair:
                if outcome.winner_name not in outcome.player_pair:
                    reasons.append("winner_not_in_prediction_pair")
            if outcome.player_pair and bet_on and bet_on not in outcome.player_pair:
                reasons.append("bet_side_not_in_prediction_pair")

        stake = _positive_decimal(row.get("stake"))
        if stake is None:
            reasons.append("invalid_positive_stake")
        odds = _price_decimal(row.get("odds_decimal"))
        if odds is None:
            reasons.append("invalid_decimal_odds")

        session_row: pd.Series | None = None
        session_start_event: pd.Series | None = None
        if not session_id:
            reasons.append("blank_session_id")
        elif session_counts[session_id] != 1:
            reasons.append("missing_or_nonunique_session_lineage")
        else:
            session_row = sessions_work[session_ids.eq(session_id)].iloc[0]
            bet_timestamp = _strict_timestamp(row.get("timestamp"))
            session_start = _strict_timestamp(session_row.get("start_time"))
            if bet_timestamp is None or session_start is None:
                reasons.append("invalid_bet_or_session_timestamp")
            elif bet_timestamp < session_start:
                reasons.append("bet_precedes_session_start")
            if bankroll_session_counts[session_id] < 1:
                reasons.append("session_missing_bankroll_lineage")
            else:
                session_bankroll = bankroll_work[
                    bankroll_work["session_id"].map(_clean).eq(session_id)
                ]
                start_mask = (
                    session_bankroll["change_reason"]
                    .map(normalize_text)
                    .eq("session started")
                    & session_bankroll["change_amount"]
                    .map(_decimal_value)
                    .eq(Decimal("0"))
                )
                start_events = session_bankroll[start_mask]
                if len(start_events) != 1:
                    reasons.append("missing_or_nonunique_session_start_event")
                else:
                    session_start_event = start_events.iloc[0]
                    event_timestamp = _strict_timestamp(
                        session_start_event.get("timestamp")
                    )
                    initial_balance = _positive_decimal(
                        session_row.get("initial_bankroll")
                    )
                    event_balance = _positive_decimal(
                        session_start_event.get("bankroll")
                    )
                    if (
                        event_timestamp is None
                        or session_start is None
                        or bet_timestamp is None
                        or event_timestamp != session_start
                        or event_timestamp > bet_timestamp
                        or initial_balance is None
                        or event_balance != initial_balance
                    ):
                        reasons.append("session_start_event_lineage_mismatch")

        reasons_tuple = _dedupe_reasons(reasons)
        if reasons_tuple:
            rejected.append(
                {
                    "source_row": source_row,
                    "bet_id": bet_id,
                    "match_uid": match_uid,
                    "pending_identity_key": identity_key,
                    "reasons": list(reasons_tuple),
                }
            )
            continue

        assert outcome is not None
        assert stake is not None and odds is not None
        assert session_row is not None
        assert session_start_event is not None
        result = "win" if bet_on == outcome.winner_name else "loss"
        profit = stake * (odds - Decimal("1")) if result == "win" else -stake
        provisional.append(
            {
                "source_row": source_row,
                "bet_id": bet_id,
                "bet_row_sha256": _row_sha256(row, bet_columns),
                "pending_identity_key": identity_key,
                "match_uid": match_uid,
                "match_pair": list(bet_pair),
                "match_date": bet_date,
                "settlement_effective_at": outcome.settlement_effective_at,
                "bet_on": bet_on,
                "bet_on_player1": side_flag,
                "stake": _decimal_text(stake),
                "odds_decimal": _decimal_text(odds),
                "authoritative_winner_name": outcome.winner_name,
                "winner_source_values": list(outcome.winner_source_values),
                "prediction_source_rows": list(outcome.source_rows),
                "prediction_row_hashes": list(outcome.source_row_hashes),
                "session_id": session_id,
                "session_source_row": int(session_row["_source_row"]),
                "session_row_sha256": _row_sha256(session_row, session_columns),
                "session_start_event_source_row": int(
                    session_start_event["_source_row"]
                ),
                "session_start_event_row_sha256": _row_sha256(
                    session_start_event, bankroll_columns
                ),
                "outcome": result,
                "actual_profit": _decimal_text(profit),
            }
        )

    candidates = sorted(
        provisional,
        key=lambda row: (
            row["settlement_effective_at"],
            row["match_uid"],
            row["bet_id"],
            row["source_row"],
        ),
    )
    equity_cursor = Decimal(account["account_equity_before"])
    latest_bankroll_timestamp = max(
        timestamp for timestamp in bankroll_timestamps if timestamp is not None
    ) if bankroll_timestamps and all(
        timestamp is not None for timestamp in bankroll_timestamps
    ) else None
    if latest_bankroll_timestamp is not None and candidates:
        latest_bankroll_timestamp = max(
            latest_bankroll_timestamp,
            *(
                _strict_timestamp(candidate["settlement_effective_at"])
                for candidate in candidates
            ),
        )
    for sequence, candidate in enumerate(candidates, start=1):
        equity_cursor += Decimal(candidate["actual_profit"])
        candidate["event_sequence"] = sequence
        candidate["expected_bankroll_after"] = _decimal_text(equity_cursor)
        if latest_bankroll_timestamp is not None:
            candidate["accounting_recorded_at"] = (
                latest_bankroll_timestamp + pd.Timedelta(microseconds=sequence)
            ).isoformat()

    rejected = sorted(rejected, key=lambda row: (row["source_row"], row["bet_id"]))
    inputs = {
        "bets": _file_descriptor(paths.bets, bets),
        "predictions": _file_descriptor(paths.predictions, predictions),
        "bankroll": _file_descriptor(paths.bankroll, bankroll),
        "sessions": _file_descriptor(paths.sessions, sessions),
        "apply_audit": _file_descriptor(
            paths.apply_audit,
            apply_audit if paths.apply_audit.is_file() else None,
        ),
    }
    input_fingerprint = _payload_sha256(
        {
            name: {
                "path": details["path"],
                "exists": details["exists"],
                "sha256": details["sha256"],
            }
            for name, details in inputs.items()
        }
    )
    operation_id = "pending_reconcile_" + _payload_sha256(
        {
            "input_fingerprint": input_fingerprint,
            "candidates": candidates,
            "starting_capital": account["starting_capital"],
        }
    )[:24]
    eligible_profit = sum(
        (Decimal(row["actual_profit"]) for row in candidates), Decimal("0")
    )
    preview_plan = {
        "plan_digest": "0" * 64,
        "operation_id": operation_id,
        "account": account,
        "inputs": inputs,
        "candidates": candidates,
    }
    preview_bets, preview_bankroll, preview_sessions, _ = _build_applied_frames(
        preview_plan,
        bets=bets,
        bankroll=bankroll,
        sessions=sessions,
        apply_audit=apply_audit,
        applied_at="1970-01-01T00:00:00+00:00",
    )
    post_state = _post_state_manifest(
        original_bet_rows=len(bets),
        original_bankroll_rows=len(bankroll),
        candidates=candidates,
        bets=preview_bets,
        bankroll=preview_bankroll,
        sessions=preview_sessions,
    )

    plan_without_digest: dict[str, Any] = {
        "plan_schema_version": SETTLEMENT_PLAN_SCHEMA_VERSION,
        "operation": "safe_exact_pending_bet_settlement",
        "read_only_plan": True,
        "automatic_startup_wiring": False,
        "operation_id": operation_id,
        "input_fingerprint": input_fingerprint,
        "coordination": {
            "exclusive_lock_path": str(paths.lock),
            "transaction_recovery_dir": str(paths.transaction_dir),
        },
        "inputs": inputs,
        "account": account,
        "post_state": post_state,
        "summary": {
            "pending_rows": int(len(pending)),
            "eligible_rows": len(candidates),
            "rejected_rows": len(rejected),
            "eligible_profit": _decimal_text(eligible_profit),
            "account_equity_after": _decimal_text(equity_cursor),
        },
        "candidates": candidates,
        "rejected": rejected,
    }
    return {
        **plan_without_digest,
        "plan_digest": _plan_digest(plan_without_digest),
    }


def write_settlement_plan(plan: Mapping[str, Any], output_path: Path) -> None:
    """Write a plan only to the operator-supplied path."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        plan,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    if output_path.is_file():
        existing = output_path.read_text(encoding="utf-8")
        if existing == payload:
            return
        raise ValueError(
            "refusing to overwrite an existing plan with different content"
        )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_settlement_plan(
    plan_path: Path,
    *,
    expected_digest: str,
) -> dict[str, Any]:
    """Load a reviewed plan and verify both embedded and operator digests."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest or ""):
        raise ReconciliationConflict("expected plan digest must be 64 lowercase hex chars")
    try:
        plan = json.loads(plan_path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconciliationConflict(f"unable to load settlement plan: {exc}") from exc
    if not isinstance(plan, dict):
        raise ReconciliationConflict("settlement plan must be a JSON object")
    if plan.get("plan_schema_version") != SETTLEMENT_PLAN_SCHEMA_VERSION:
        raise ReconciliationConflict("unsupported settlement plan schema version")
    embedded = _clean(plan.get("plan_digest"))
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    computed = _plan_digest(unsigned)
    if embedded != computed:
        raise ReconciliationConflict("settlement plan content does not match embedded digest")
    if expected_digest != embedded:
        raise ReconciliationConflict("expected digest does not match settlement plan")
    if plan.get("operation") != "safe_exact_pending_bet_settlement":
        raise ReconciliationConflict("unexpected settlement plan operation")
    return plan


def _validate_bound_paths(plan: Mapping[str, Any], paths: SettlementPaths) -> None:
    paths = paths.resolved()
    for name, expected_path in paths.as_dict().items():
        planned = _clean(plan.get("inputs", {}).get(name, {}).get("path"))
        if planned != str(expected_path):
            raise ReconciliationConflict(
                f"plan {name} path does not match the explicitly supplied path"
            )
    planned_lock = _clean(
        plan.get("coordination", {}).get("exclusive_lock_path")
    )
    if planned_lock != str(paths.lock):
        raise ReconciliationConflict(
            "plan lock path does not match the explicitly supplied lock path"
        )
    planned_transaction = _clean(
        plan.get("coordination", {}).get("transaction_recovery_dir")
    )
    if planned_transaction != str(paths.transaction_dir):
        raise ReconciliationConflict(
            "plan transaction directory does not match the explicit recovery path"
        )


def _validate_input_hashes(plan: Mapping[str, Any], paths: SettlementPaths) -> None:
    for name, path in paths.resolved().as_dict().items():
        descriptor = plan["inputs"][name]
        exists = path.is_file()
        if bool(descriptor.get("exists")) != exists:
            raise ReconciliationConflict(f"{name} existence changed after planning")
        current_hash = _file_sha256(path) if exists else None
        if descriptor.get("sha256") != current_hash:
            raise ReconciliationConflict(f"{name} input hash changed after planning")


def _event_id(plan_digest: str, sequence: int) -> str:
    return f"pending_apply_{plan_digest[:20]}_{sequence:04d}"


def _settlement_note(operation_id: str) -> str:
    return f"Exact-UID pending reconciliation; operation_id={operation_id}"


def _bankroll_reason(operation_id: str, candidate: Mapping[str, Any]) -> str:
    return (
        f"Pending reconciliation {operation_id} bet "
        f"{candidate['bet_id']} ({str(candidate['outcome']).upper()})"
    )


def _verify_replay_or_raise(
    plan: Mapping[str, Any],
    *,
    bets: pd.DataFrame,
    bankroll: pd.DataFrame,
    sessions: pd.DataFrame,
    apply_audit: pd.DataFrame,
) -> bool:
    """Return true for a complete prior apply; reject every partial replay."""
    candidates = list(plan.get("candidates", []))
    if not candidates:
        return False
    digest = str(plan["plan_digest"])
    events = apply_audit[
        apply_audit.get(
            "plan_digest", pd.Series("", index=apply_audit.index)
        ).map(_clean).eq(digest)
    ]
    if events.empty:
        return False
    if len(events) != len(candidates):
        raise ReconciliationConflict("partial or duplicate apply-audit replay detected")

    expected_ids = {
        _event_id(digest, int(candidate["event_sequence"]))
        for candidate in candidates
    }
    actual_ids = set(events["event_id"].map(_clean))
    if actual_ids != expected_ids or len(actual_ids) != len(events):
        raise ReconciliationConflict("apply-audit event identity conflict")

    applied_timestamps: set[str] = set()
    post_bets = {
        row["bet_id"]: row for row in plan["post_state"]["bet_rows"]
    }
    post_bankroll = {
        int(row["append_position"]): row
        for row in plan["post_state"]["bankroll_appends"]
    }
    post_sessions = {
        row["session_id"]: row for row in plan["post_state"]["session_rows"]
    }
    for candidate in candidates:
        sequence = int(candidate["event_sequence"])
        event = events[
            events["event_id"].map(_clean).eq(_event_id(digest, sequence))
        ].iloc[0]
        event_payload = {
            column: _clean(event.get(column))
            for column in APPLY_AUDIT_COLUMNS
            if column != "event_payload_sha256"
        }
        if _clean(event.get("event_payload_sha256")) != _payload_sha256(
            event_payload
        ):
            raise ReconciliationConflict(
                "apply-audit replay conflict: event payload hash"
            )
        expected_static = {
            "audit_schema_version": APPLY_AUDIT_SCHEMA_VERSION,
            "plan_digest": digest,
            "plan_schema_version": SETTLEMENT_PLAN_SCHEMA_VERSION,
            "event_sequence": str(sequence),
            "event_type": "settled_from_prediction_log_exact_uid",
            "bet_id": str(candidate["bet_id"]),
            "match_uid": str(candidate["match_uid"]),
            "session_id": str(candidate["session_id"]),
            "outcome": str(candidate["outcome"]),
            "actual_profit": str(candidate["actual_profit"]),
            "authoritative_winner_name": str(
                candidate["authoritative_winner_name"]
            ),
            "pending_identity_key": str(candidate["pending_identity_key"]),
            "bet_row_sha256": str(candidate["bet_row_sha256"]),
            "prediction_rows_sha256": _payload_sha256(
                candidate["prediction_row_hashes"]
            ),
            "session_row_sha256": str(candidate["session_row_sha256"]),
            "bet_post_row_sha256": str(
                post_bets[str(candidate["bet_id"])]["row_sha256"]
            ),
            "bankroll_event_row_sha256": str(
                post_bankroll[sequence]["row_sha256"]
            ),
            "session_post_row_sha256": str(
                post_sessions[str(candidate["session_id"])]["row_sha256"]
            ),
            "reviewed_post_state_sha256": _payload_sha256(plan["post_state"]),
            "bets_input_sha256": str(plan["inputs"]["bets"]["sha256"]),
            "predictions_input_sha256": str(
                plan["inputs"]["predictions"]["sha256"]
            ),
            "bankroll_input_sha256": str(
                plan["inputs"]["bankroll"]["sha256"]
            ),
            "sessions_input_sha256": str(
                plan["inputs"]["sessions"]["sha256"]
            ),
            "apply_audit_input_sha256": str(
                plan["inputs"]["apply_audit"]["sha256"] or ""
            ),
        }
        for column, expected in expected_static.items():
            if _clean(event.get(column)) != expected:
                raise ReconciliationConflict(
                    f"apply-audit replay conflict for {candidate['bet_id']}:{column}"
                )
        applied_at = _clean(event.get("applied_at_utc"))
        if _strict_timestamp(applied_at) is None:
            raise ReconciliationConflict("apply-audit replay has invalid timestamp")
        applied_timestamps.add(applied_at)

        matching_bets = bets[bets["bet_id"].map(_clean).eq(str(candidate["bet_id"]))]
        if len(matching_bets) != 1:
            raise ReconciliationConflict("replayed bet identity is missing or nonunique")
        bet = matching_bets.iloc[0]
        if _row_sha256(bet, list(bets.columns)) != event["bet_post_row_sha256"]:
            raise ReconciliationConflict("replayed bet post-row hash conflicts with audit")
        if _clean(bet.get("status")).lower() != "settled":
            raise ReconciliationConflict("apply audit exists but bet is not settled")
        if _clean(bet.get("outcome")).lower() != str(candidate["outcome"]):
            raise ReconciliationConflict("replayed bet outcome conflicts with plan")
        profit = _decimal_value(bet.get("actual_profit"))
        if profit != Decimal(str(candidate["actual_profit"])):
            raise ReconciliationConflict("replayed bet profit conflicts with plan")
        if _clean(bet.get("notes")) != _settlement_note(str(plan["operation_id"])):
            raise ReconciliationConflict("replayed bet note does not bind plan digest")
        if _clean(bet.get("settled_timestamp")) != str(
            candidate["settlement_effective_at"]
        ):
            raise ReconciliationConflict("replayed settlement timestamp conflicts with plan")

        reason = _bankroll_reason(str(plan["operation_id"]), candidate)
        matching_bankroll = bankroll[
            bankroll["change_reason"].map(_clean).eq(reason)
        ]
        if len(matching_bankroll) != 1:
            raise ReconciliationConflict("replayed bankroll event is missing or duplicated")
        if _decimal_value(matching_bankroll.iloc[0].get("change_amount")) != Decimal(
            str(candidate["actual_profit"])
        ):
            raise ReconciliationConflict("replayed bankroll amount conflicts with plan")
        if _row_sha256(
            matching_bankroll.iloc[0], list(bankroll.columns)
        ) != event["bankroll_event_row_sha256"]:
            raise ReconciliationConflict(
                "replayed bankroll event row conflicts with audit"
            )

    if len(applied_timestamps) != 1:
        raise ReconciliationConflict("apply-audit events do not share one transaction timestamp")
    _verify_current_replay_state(
        plan,
        bets=bets,
        bankroll=bankroll,
        sessions=sessions,
    )
    return True


def _mean_decimal(values: Iterable[Any]) -> Decimal | None:
    numbers = [number for number in (_decimal_value(value) for value in values)
               if number is not None]
    if not numbers:
        return None
    return sum(numbers, Decimal("0")) / Decimal(len(numbers))


def _require_decimal_equal(
    row: pd.Series,
    column: str,
    expected: Decimal,
    *,
    context: str,
) -> None:
    if _decimal_value(row.get(column)) != expected:
        raise ReconciliationConflict(f"{context} {column} does not reconcile")


def _verify_current_replay_state(
    plan: Mapping[str, Any],
    *,
    bets: pd.DataFrame,
    bankroll: pd.DataFrame,
    sessions: pd.DataFrame,
) -> None:
    """Recompute current account/session invariants before a replay no-op."""
    starting_capital = Decimal(str(plan["account"]["starting_capital"]))
    account = _account_state(bets, starting_capital=starting_capital)
    if account["global_blockers"]:
        raise ReconciliationConflict("current replay ledger fails account integrity")
    if bankroll.empty:
        raise ReconciliationConflict("current replay ledger has no bankroll event")
    latest = bankroll.iloc[-1]
    equity = Decimal(account["account_equity_before"])
    pending_exposure = Decimal(account["pending_exposure_before"])
    total_staked = Decimal(account["total_staked"])
    available = max(Decimal("0"), equity - pending_exposure)
    for column, expected in (
        ("bankroll", equity),
        ("account_equity", equity),
        ("pending_exposure", pending_exposure),
        ("available_bankroll", available),
        ("total_staked", total_staked),
    ):
        _require_decimal_equal(latest, column, expected, context="latest bankroll")
    statuses = bets["status"].map(lambda value: _clean(value).lower())
    pending_count = _decimal_value(latest.get("num_pending_bets"))
    if pending_count != Decimal(int(statuses.eq("pending").sum())):
        raise ReconciliationConflict("latest bankroll pending count does not reconcile")
    settled_count = _decimal_value(latest.get("num_settled_bets"))
    if settled_count != Decimal(int(statuses.eq("settled").sum())):
        raise ReconciliationConflict("latest bankroll settled count does not reconcile")

    for planned in plan["post_state"]["session_rows"]:
        session_id = str(planned["session_id"])
        matching = sessions[sessions["session_id"].map(_clean).eq(session_id)]
        if len(matching) != 1:
            raise ReconciliationConflict("replayed session is missing or nonunique")
        current = matching.iloc[0]
        if _row_sha256(current, list(sessions.columns)) != planned["row_sha256"]:
            raise ReconciliationConflict("replayed session post-row hash conflicts with plan")
        session_bets = bets[bets["session_id"].map(_clean).eq(session_id)]
        session_status = session_bets["status"].map(
            lambda value: _clean(value).lower()
        )
        settled = session_bets[session_status.eq("settled")]
        stakes = [_positive_decimal(value) for value in session_bets["stake"]]
        profits = [_decimal_value(value) for value in settled["actual_profit"]]
        if any(value is None for value in stakes + profits):
            raise ReconciliationConflict("replayed session arithmetic is invalid")
        _require_decimal_equal(
            current,
            "total_staked",
            sum((value for value in stakes if value is not None), Decimal("0")),
            context="replayed session",
        )
        _require_decimal_equal(
            current,
            "total_profit_loss",
            sum((value for value in profits if value is not None), Decimal("0")),
            context="replayed session",
        )
        total_bets_value = _decimal_value(current.get("total_bets_placed"))
        if total_bets_value != Decimal(len(session_bets)):
            raise ReconciliationConflict("replayed session bet count does not reconcile")
        wins = int(
            settled["outcome"].map(lambda value: _clean(value).lower()).eq("win").sum()
        )
        expected_win_rate = (
            Decimal(wins) / Decimal(len(settled)) if len(settled) else None
        )
        if expected_win_rate is None:
            if _clean(current.get("win_rate")):
                raise ReconciliationConflict("replayed empty session has a win rate")
        else:
            _require_decimal_equal(
                current,
                "win_rate",
                expected_win_rate,
                context="replayed session",
            )
        expected_odds = _mean_decimal(session_bets["odds_decimal"])
        expected_edge = _mean_decimal(session_bets["edge"])
        if expected_odds is None or expected_edge is None:
            raise ReconciliationConflict("replayed session averages are invalid")
        _require_decimal_equal(
            current, "avg_odds", expected_odds, context="replayed session"
        )
        _require_decimal_equal(
            current, "avg_edge", expected_edge, context="replayed session"
        )


def _build_applied_frames(
    plan: Mapping[str, Any],
    *,
    bets: pd.DataFrame,
    bankroll: pd.DataFrame,
    sessions: pd.DataFrame,
    apply_audit: pd.DataFrame,
    applied_at: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    digest = str(plan["plan_digest"])
    candidates = list(plan["candidates"])
    bets_out = bets.copy()
    bankroll_out = bankroll.copy()
    sessions_out = sessions.copy()
    audit_out = apply_audit.copy()

    equity = Decimal(str(plan["account"]["account_equity_before"]))
    pending_exposure = Decimal(str(plan["account"]["pending_exposure_before"]))
    total_staked = Decimal(str(plan["account"]["total_staked"]))
    statuses_before = bets_out["status"].map(lambda value: _clean(value).lower())
    num_pending = int(statuses_before.eq("pending").sum())
    num_settled = int(statuses_before.eq("settled").sum())
    bankroll_events: list[dict[str, str]] = []
    audit_events: list[dict[str, str]] = []

    for candidate in candidates:
        sequence = int(candidate["event_sequence"])
        source_index = int(candidate["source_row"]) - 2
        if source_index < 0 or source_index >= len(bets_out):
            raise ReconciliationConflict("planned bet source row is out of range")
        current = bets_out.iloc[source_index]
        if _row_sha256(current, list(bets_out.columns)) != candidate["bet_row_sha256"]:
            raise ReconciliationConflict("planned bet row hash changed")
        if _clean(current.get("bet_id")) != candidate["bet_id"]:
            raise ReconciliationConflict("planned bet row identity changed")
        if _clean(current.get("status")).lower() != "pending":
            raise ReconciliationConflict("planned bet is no longer pending")

        stake = Decimal(str(candidate["stake"]))
        profit = Decimal(str(candidate["actual_profit"]))
        settlement_effective_at = str(candidate["settlement_effective_at"])
        if _strict_timestamp(settlement_effective_at) is None:
            raise ReconciliationConflict("planned settlement timestamp is invalid")
        equity += profit
        pending_exposure -= stake
        num_pending -= 1
        num_settled += 1
        expected_equity = Decimal(str(candidate["expected_bankroll_after"]))
        if equity != expected_equity or pending_exposure < 0:
            raise ReconciliationConflict("planned account arithmetic no longer reconciles")
        available = max(Decimal("0"), equity - pending_exposure)

        bets_out.at[source_index, "status"] = "settled"
        bets_out.at[source_index, "outcome"] = str(candidate["outcome"])
        bets_out.at[source_index, "actual_profit"] = _decimal_text(profit)
        bets_out.at[source_index, "bankroll_after"] = _decimal_text(equity)
        bets_out.at[source_index, "settled_timestamp"] = settlement_effective_at
        bets_out.at[source_index, "notes"] = _settlement_note(
            str(plan["operation_id"])
        )

        bankroll_events.append(
            {
                "timestamp": str(candidate["accounting_recorded_at"]),
                "session_id": str(candidate["session_id"]),
                "bankroll": _decimal_text(equity),
                "change_amount": _decimal_text(profit),
                "change_reason": _bankroll_reason(
                    str(plan["operation_id"]), candidate
                ),
                "account_equity": _decimal_text(equity),
                "pending_exposure": _decimal_text(pending_exposure),
                "available_bankroll": _decimal_text(available),
                "total_staked": _decimal_text(total_staked),
                "num_pending_bets": str(num_pending),
                "num_settled_bets": str(num_settled),
            }
        )
        inputs = plan["inputs"]
        audit_events.append(
            {
                "audit_schema_version": APPLY_AUDIT_SCHEMA_VERSION,
                "event_id": _event_id(digest, sequence),
                "event_payload_sha256": "",
                "plan_digest": digest,
                "plan_schema_version": SETTLEMENT_PLAN_SCHEMA_VERSION,
                "event_sequence": str(sequence),
                "event_type": "settled_from_prediction_log_exact_uid",
                "applied_at_utc": applied_at,
                "bet_id": str(candidate["bet_id"]),
                "match_uid": str(candidate["match_uid"]),
                "session_id": str(candidate["session_id"]),
                "outcome": str(candidate["outcome"]),
                "actual_profit": str(candidate["actual_profit"]),
                "authoritative_winner_name": str(
                    candidate["authoritative_winner_name"]
                ),
                "pending_identity_key": str(candidate["pending_identity_key"]),
                "bet_row_sha256": str(candidate["bet_row_sha256"]),
                "prediction_rows_sha256": _payload_sha256(
                    candidate["prediction_row_hashes"]
                ),
                "session_row_sha256": str(candidate["session_row_sha256"]),
                "bet_post_row_sha256": "",
                "bankroll_event_row_sha256": "",
                "session_post_row_sha256": "",
                "reviewed_post_state_sha256": _payload_sha256(
                    plan.get("post_state", {})
                ),
                "bets_input_sha256": str(inputs["bets"]["sha256"]),
                "predictions_input_sha256": str(inputs["predictions"]["sha256"]),
                "bankroll_input_sha256": str(inputs["bankroll"]["sha256"]),
                "sessions_input_sha256": str(inputs["sessions"]["sha256"]),
                "apply_audit_input_sha256": str(
                    inputs["apply_audit"]["sha256"] or ""
                ),
            }
        )

    if bankroll_events:
        bankroll_out = pd.concat(
            [bankroll_out, pd.DataFrame(bankroll_events, columns=bankroll.columns)],
            ignore_index=True,
        )

    affected_sessions = sorted({str(row["session_id"]) for row in candidates})
    final_equity = equity
    for session_id in affected_sessions:
        session_indexes = sessions_out[
            sessions_out["session_id"].map(_clean).eq(session_id)
        ].index
        if len(session_indexes) != 1:
            raise ReconciliationConflict("affected session lineage is no longer unique")
        session_bets = bets_out[
            bets_out["session_id"].map(_clean).eq(session_id)
        ]
        session_status = session_bets["status"].map(
            lambda value: _clean(value).lower()
        )
        settled = session_bets[session_status.eq("settled")]
        pending_count = int(session_status.eq("pending").sum())
        session_stakes = [
            _positive_decimal(value) for value in session_bets["stake"].tolist()
        ]
        if any(value is None for value in session_stakes):
            raise ReconciliationConflict("affected session contains an invalid stake")
        settled_profits = [
            _decimal_value(value) for value in settled["actual_profit"].tolist()
        ]
        if any(value is None for value in settled_profits):
            raise ReconciliationConflict("affected session contains invalid settled profit")
        wins = int(
            settled["outcome"].map(lambda value: _clean(value).lower()).eq("win").sum()
        )
        idx = session_indexes[0]
        sessions_out.at[idx, "total_bets_placed"] = str(len(session_bets))
        sessions_out.at[idx, "total_staked"] = _decimal_text(
            sum((value for value in session_stakes if value is not None), Decimal("0"))
        )
        sessions_out.at[idx, "total_profit_loss"] = _decimal_text(
            sum((value for value in settled_profits if value is not None), Decimal("0"))
        )
        sessions_out.at[idx, "win_rate"] = (
            _decimal_text(Decimal(wins) / Decimal(len(settled)))
            if len(settled)
            else ""
        )
        avg_odds = _mean_decimal(session_bets["odds_decimal"].tolist())
        avg_edge = _mean_decimal(session_bets["edge"].tolist()) if "edge" in session_bets else None
        sessions_out.at[idx, "avg_odds"] = _decimal_text(avg_odds) if avg_odds is not None else ""
        sessions_out.at[idx, "avg_edge"] = _decimal_text(avg_edge) if avg_edge is not None else ""
        if pending_count == 0 and len(settled):
            settlement_times = [
                timestamp
                for timestamp in (
                    _strict_timestamp(value)
                    for value in settled["settled_timestamp"].tolist()
                )
                if timestamp is not None
            ]
            if len(settlement_times) != len(settled):
                raise ReconciliationConflict(
                    "affected session contains an invalid settlement timestamp"
                )
            sessions_out.at[idx, "end_time"] = max(settlement_times).isoformat()
            sessions_out.at[idx, "final_bankroll"] = _decimal_text(final_equity)

    for event, candidate in zip(audit_events, candidates):
        bet = bets_out.iloc[int(candidate["source_row"]) - 2]
        bankroll_event = bankroll_out.iloc[
            len(bankroll_out) - len(candidates) + int(candidate["event_sequence"]) - 1
        ]
        session = sessions_out[
            sessions_out["session_id"].map(_clean).eq(str(candidate["session_id"]))
        ].iloc[0]
        event["bet_post_row_sha256"] = _row_sha256(bet, list(bets_out.columns))
        event["bankroll_event_row_sha256"] = _row_sha256(
            bankroll_event, list(bankroll_out.columns)
        )
        event["session_post_row_sha256"] = _row_sha256(
            session, list(sessions_out.columns)
        )
        event["event_payload_sha256"] = _payload_sha256(
            {
                column: event[column]
                for column in APPLY_AUDIT_COLUMNS
                if column != "event_payload_sha256"
            }
        )

    if audit_events:
        audit_out = pd.concat(
            [audit_out, pd.DataFrame(audit_events, columns=APPLY_AUDIT_COLUMNS)],
            ignore_index=True,
        )
    return bets_out, bankroll_out, sessions_out, audit_out


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _post_state_manifest(
    *,
    original_bet_rows: int,
    original_bankroll_rows: int,
    candidates: Sequence[Mapping[str, Any]],
    bets: pd.DataFrame,
    bankroll: pd.DataFrame,
    sessions: pd.DataFrame,
) -> dict[str, Any]:
    bet_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        source_index = int(candidate["source_row"]) - 2
        row = bets.iloc[source_index]
        bet_rows.append(
            {
                "bet_id": str(candidate["bet_id"]),
                "source_row": int(candidate["source_row"]),
                "row_sha256": _row_sha256(row, list(bets.columns)),
                "row": {column: _clean(row.get(column)) for column in bets.columns},
            }
        )
    bankroll_rows = []
    for offset, (_, row) in enumerate(
        bankroll.iloc[original_bankroll_rows:].iterrows(), start=1
    ):
        bankroll_rows.append(
            {
                "append_position": offset,
                "row_sha256": _row_sha256(row, list(bankroll.columns)),
                "row": {
                    column: _clean(row.get(column)) for column in bankroll.columns
                },
            }
        )
    session_rows = []
    for session_id in sorted({str(row["session_id"]) for row in candidates}):
        matching = sessions[sessions["session_id"].map(_clean).eq(session_id)]
        if len(matching) != 1:
            raise ReconciliationConflict("post-state session is missing or nonunique")
        row = matching.iloc[0]
        session_rows.append(
            {
                "session_id": session_id,
                "row_sha256": _row_sha256(row, list(sessions.columns)),
                "row": {
                    column: _clean(row.get(column)) for column in sessions.columns
                },
            }
        )
    return {
        "bet_rows": bet_rows,
        "bankroll_appends": bankroll_rows,
        "session_rows": session_rows,
        "row_counts_after": {
            "bets": original_bet_rows,
            "bankroll": len(bankroll),
            "sessions": len(sessions),
        },
    }


def _write_fsynced_temp(
    path: Path,
    payload: bytes,
    *,
    prefix: str,
    mode: int = 0o600,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.chmod(temporary, mode)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_apply_audit_directory(path: Path) -> None:
    """Create and durably anchor the canonical private audit directory."""
    directory = path.resolve().parent
    parent = directory.parent
    if not parent.is_dir():
        raise AtomicApplyError(
            f"apply-audit parent root does not exist: {parent}"
        )
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise AtomicApplyError(
                "canonical apply-audit parent is not a regular directory"
            )
    else:
        directory.mkdir(mode=0o700)
    directory_stat = directory.stat()
    mode = stat.S_IMODE(directory_stat.st_mode)
    if mode & 0o077:
        raise AtomicApplyError(
            "canonical apply-audit directory must not grant group/other access"
        )
    if directory_stat.st_uid != os.geteuid():
        raise AtomicApplyError(
            "canonical apply-audit directory must be owned by this process user"
        )
    # Persist the directory entry in the production root before any journal or
    # target replacement can begin. Fsyncing only the new directory would not
    # make its name durable across a power loss.
    _fsync_directory(parent)


TRANSACTION_SCHEMA_VERSION = "1.0.0"


class _SimulatedProcessCrash(BaseException):
    """Test-only stand-in for SIGKILL after one or more replacements."""


@dataclass(frozen=True)
class _RollbackEntry:
    target: Path
    original_exists: bool
    original_mode: int
    original_payload: bytes | None


def _persist_transaction_manifest(
    transaction_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    payload = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = _write_fsynced_temp(
        transaction_dir / "manifest.json",
        payload,
        prefix=".manifest.",
    )
    os.replace(temporary, transaction_dir / "manifest.json")
    _fsync_directory(transaction_dir)


def _remove_transaction_dir(transaction_dir: Path) -> None:
    if transaction_dir.exists():
        shutil.rmtree(transaction_dir)
        _fsync_directory(transaction_dir.parent)


def _journal_child_path(
    transaction_dir: Path,
    value: Any,
    *,
    label: str,
    allow_blank: bool = False,
) -> Path | None:
    text = value if isinstance(value, str) else ""
    if allow_blank and not text:
        return None
    relative = Path(text)
    candidate = transaction_dir / relative
    if (
        not text
        or relative.is_absolute()
        or relative.name != text
        or candidate.resolve().parent != transaction_dir.resolve()
    ):
        raise AtomicApplyError(
            f"transaction manifest {label} escapes the durable journal"
        )
    return candidate


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _validate_transaction_entries(
    transaction_dir: Path,
    manifest: Any,
    *,
    expected_targets: Sequence[Path],
) -> tuple[str, list[Mapping[str, Any]], list[Path]]:
    if not isinstance(manifest, dict):
        raise AtomicApplyError("transaction manifest must be a JSON object")
    if manifest.get("transaction_schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise AtomicApplyError("unsupported transaction recovery schema")
    state = manifest.get("state")
    if state not in {"prepared", "committed"}:
        raise AtomicApplyError("unknown transaction recovery state")
    transaction_id = manifest.get("transaction_id")
    if not isinstance(transaction_id, str) or not re.fullmatch(
        r"[0-9a-f]{20}", transaction_id
    ):
        raise AtomicApplyError("transaction manifest has an invalid transaction ID")

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise AtomicApplyError("transaction manifest entries are malformed")
    roles = tuple(
        entry.get("role") if isinstance(entry, dict) else None
        for entry in entries
    )
    if roles != PENDING_RECONCILIATION_TARGET_ROLES:
        raise AtomicApplyError(
            "transaction manifest target roles do not match the recovery contract"
        )

    canonical_targets = list(
        canonical_pending_reconciliation_targets(transaction_dir.parent.resolve())
    )
    allowed_targets = [path.resolve() for path in expected_targets]
    if allowed_targets != canonical_targets:
        raise AtomicApplyError(
            "transaction recovery targets are outside the canonical allowlist"
        )

    planned_targets: list[Path] = []
    for entry, allowed_target in zip(entries, allowed_targets):
        target_text = entry.get("target")
        if not isinstance(target_text, str):
            raise AtomicApplyError("transaction manifest target is malformed")
        target = Path(target_text)
        if not target.is_absolute() or target != allowed_target:
            raise AtomicApplyError(
                "transaction recovery targets do not match the canonical allowlist"
            )
        planned_targets.append(target)

        original_exists = entry.get("original_exists")
        original_mode = entry.get("original_mode")
        original_sha = entry.get("original_sha256")
        new_sha = entry.get("new_sha256")
        if not isinstance(original_exists, bool):
            raise AtomicApplyError(
                "transaction manifest original-existence flag is malformed"
            )
        if (
            isinstance(original_mode, bool)
            or not isinstance(original_mode, int)
            or not 0 <= original_mode <= 0o7777
        ):
            raise AtomicApplyError("transaction manifest file mode is malformed")
        if not _is_sha256(new_sha):
            raise AtomicApplyError("transaction manifest new hash is malformed")
        if original_exists:
            if not _is_sha256(original_sha):
                raise AtomicApplyError(
                    "transaction manifest original hash is malformed"
                )
            _journal_child_path(
                transaction_dir,
                entry.get("backup_file"),
                label="backup path",
            )
        elif original_sha is not None or entry.get("backup_file") != "":
            raise AtomicApplyError(
                "transaction manifest has a backup for a nonexistent original"
            )
        _journal_child_path(
            transaction_dir,
            entry.get("new_file"),
            label="staged-new path",
        )

        if target.is_symlink():
            raise AtomicApplyError(
                "transaction recovery target became a symbolic link"
            )
        current_exists = target.is_file()
        current_sha = _file_sha256(target) if current_exists else None
        allowed_hashes = {new_sha}
        if original_exists:
            allowed_hashes.add(original_sha)
            if not current_exists:
                raise AtomicApplyError(
                    "transaction recovery target unexpectedly disappeared"
                )
        elif target.exists() and not current_exists:
            raise AtomicApplyError(
                "transaction recovery target is not a regular file"
            )
        if current_sha is not None and current_sha not in allowed_hashes:
            raise AtomicApplyError(
                "transaction recovery target hash is neither old nor new"
            )

    return state, entries, planned_targets


def _prepare_transaction_rollback(
    transaction_dir: Path,
    entries: Sequence[Mapping[str, Any]],
    planned_targets: Sequence[Path],
) -> list[_RollbackEntry]:
    """Load and validate every rollback byte before mutating any target."""
    rollback: list[_RollbackEntry] = []
    for entry, target in zip(entries, planned_targets):
        original_exists = bool(entry["original_exists"])
        original_payload: bytes | None = None
        if original_exists:
            backup = _journal_child_path(
                transaction_dir,
                entry["backup_file"],
                label="backup path",
            )
            assert backup is not None
            if backup.is_symlink() or not backup.is_file():
                raise AtomicApplyError(
                    "durable transaction backup is missing or not a regular file"
                )
            original_payload = backup.read_bytes()
            if sha256(original_payload).hexdigest() != entry["original_sha256"]:
                raise AtomicApplyError("durable transaction backup is corrupt")
        rollback.append(
            _RollbackEntry(
                target=target,
                original_exists=original_exists,
                original_mode=int(entry["original_mode"]),
                original_payload=original_payload,
            )
        )
    return rollback


def _restore_transaction(rollback: Sequence[_RollbackEntry]) -> None:
    restore_errors: list[str] = []
    target_directories: set[Path] = set()
    for entry in rollback:
        target = entry.target
        target_directories.add(target.parent)
        try:
            if not entry.original_exists:
                target.unlink(missing_ok=True)
                continue
            assert entry.original_payload is not None
            restore = _write_fsynced_temp(
                target,
                entry.original_payload,
                prefix=f".{target.name}.recovery.",
                mode=entry.original_mode,
            )
            os.replace(restore, target)
        except Exception as exc:  # pragma: no cover - catastrophic disk failure
            restore_errors.append(f"{target}: {exc}")
    for directory in target_directories:
        if not directory.exists():
            continue
        try:
            _fsync_directory(directory)
        except Exception as exc:  # pragma: no cover - catastrophic disk failure
            restore_errors.append(f"{directory}: {exc}")
    if restore_errors:
        raise AtomicApplyError(
            "durable transaction recovery was incomplete: "
            + "; ".join(restore_errors)
        )


def _recover_file_set_transaction(
    transaction_dir: Path,
    *,
    expected_targets: Sequence[Path],
) -> str:
    """Recover a prepared crash or finish cleanup of a committed transaction."""
    if transaction_dir.is_symlink():
        raise AtomicApplyError(
            "transaction recovery directory must not be a symbolic link"
        )
    if not transaction_dir.exists() and not transaction_dir.is_symlink():
        return "none"
    if not transaction_dir.is_dir():
        raise AtomicApplyError(
            "transaction recovery path is not a directory"
        )
    transaction_stat = transaction_dir.stat()
    if stat.S_IMODE(transaction_stat.st_mode) & 0o077:
        raise AtomicApplyError(
            "transaction recovery directory is not owner-private"
        )
    if transaction_stat.st_uid != os.geteuid():
        raise AtomicApplyError(
            "transaction recovery directory is not owned by this process user"
        )
    manifest_path = transaction_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise AtomicApplyError(
            "transaction manifest must not be a symbolic link"
        )
    if not manifest_path.is_file():
        # Replacement never begins before the prepared manifest is durable.
        _remove_transaction_dir(transaction_dir)
        return "discarded_unprepared"
    manifest_stat = manifest_path.stat()
    if stat.S_IMODE(manifest_stat.st_mode) & 0o077:
        raise AtomicApplyError("transaction manifest is not owner-private")
    if manifest_stat.st_uid != os.geteuid():
        raise AtomicApplyError(
            "transaction manifest is not owned by this process user"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomicApplyError("transaction manifest is unreadable; manual recovery required") from exc
    state, entries, planned_targets = _validate_transaction_entries(
        transaction_dir,
        manifest,
        expected_targets=expected_targets,
    )
    if state == "committed":
        complete = all(
            path.is_file() and _file_sha256(path) == entry["new_sha256"]
            for path, entry in zip(planned_targets, entries)
        )
        if complete:
            _remove_transaction_dir(transaction_dir)
            return "completed_commit_cleanup"

    rollback = _prepare_transaction_rollback(
        transaction_dir,
        entries,
        planned_targets,
    )
    _restore_transaction(rollback)
    _remove_transaction_dir(transaction_dir)
    return "restored_pre_transaction_state"


def recover_pending_reconciliation_transaction(logs_dir: Path | str) -> str:
    """Recover the canonical journal while its operational lock is held.

    This entry point is called by :func:`operational_csv_lock` before yielding
    to any first-level lock holder.  Target order is part of the transaction
    schema: bets, bankroll, sessions, then the canonical private apply audit.
    """
    logs_dir = Path(logs_dir).resolve()
    transaction_dir = canonical_pending_reconciliation_transaction_path(logs_dir)
    if not transaction_dir.exists() and not transaction_dir.is_symlink():
        return "none"

    return _recover_file_set_transaction(
        transaction_dir,
        expected_targets=canonical_pending_reconciliation_targets(logs_dir),
    )


def _atomic_replace_file_set(
    payloads: Sequence[tuple[str, Path, bytes]],
    *,
    transaction_id: str,
    transaction_dir: Path,
    _fail_after_replace: int | None = None,
    _simulate_crash_after_replace: int | None = None,
) -> None:
    """Durably journal, replace, and recover a locked operational file set."""
    roles = tuple(role for role, _, _ in payloads)
    if roles != PENDING_RECONCILIATION_TARGET_ROLES:
        raise AtomicApplyError(
            "file-set commit roles do not match the recovery contract"
        )
    targets = [path.resolve() for _, path, _ in payloads]
    if targets != list(
        canonical_pending_reconciliation_targets(transaction_dir.resolve().parent)
    ):
        raise AtomicApplyError(
            "file-set commit targets are outside the canonical allowlist"
        )
    missing_parents = [
        str(target.parent) for target in targets if not target.parent.is_dir()
    ]
    if missing_parents:
        raise AtomicApplyError(
            "file-set target directories must be durable before journaling: "
            + ", ".join(missing_parents)
        )
    _recover_file_set_transaction(transaction_dir, expected_targets=targets)
    transaction_dir.mkdir(parents=False, mode=0o700)
    entries: list[dict[str, Any]] = []
    staged: list[Path] = []
    for index, ((role, path, payload), target) in enumerate(zip(payloads, targets)):
        original_exists = target.is_file()
        original_mode = stat.S_IMODE(target.stat().st_mode) if original_exists else 0o600
        backup_file = ""
        original_sha = None
        if original_exists:
            original = target.read_bytes()
            original_sha = sha256(original).hexdigest()
            backup = _write_fsynced_temp(
                transaction_dir / f"backup-{index}",
                original,
                prefix=f".backup-{index}.",
                mode=original_mode,
            )
            backup_file = backup.name
        new_file = _write_fsynced_temp(
            transaction_dir / f"new-{index}",
            payload,
            prefix=f".new-{index}.",
            mode=original_mode,
        )
        staged.append(new_file)
        entries.append(
            {
                "role": role,
                "target": str(target),
                "original_exists": original_exists,
                "original_sha256": original_sha,
                "original_mode": original_mode,
                "backup_file": backup_file,
                "new_file": new_file.name,
                "new_sha256": sha256(payload).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "state": "prepared",
        "entries": entries,
    }
    _persist_transaction_manifest(transaction_dir, manifest)
    _fsync_directory(transaction_dir.parent)

    try:
        for replaced, (target, staged_path) in enumerate(
            zip(targets, staged), start=1
        ):
            os.replace(staged_path, target)
            if _simulate_crash_after_replace == replaced:
                raise _SimulatedProcessCrash(
                    "simulated process death with durable recovery journal"
                )
            if _fail_after_replace == replaced:
                raise OSError("injected file-set commit failure")
        for directory in {path.parent for path in targets}:
            _fsync_directory(directory)
    except Exception as exc:
        try:
            _recover_file_set_transaction(
                transaction_dir,
                expected_targets=targets,
            )
        except Exception as recovery_exc:
            raise AtomicApplyError(
                "file-set commit failed and durable rollback was incomplete"
            ) from recovery_exc
        raise AtomicApplyError(
            "file-set commit failed; durable backups restored every original file"
        ) from exc

    manifest["state"] = "committed"
    _persist_transaction_manifest(transaction_dir, manifest)
    try:
        _remove_transaction_dir(transaction_dir)
    except OSError:
        # The committed manifest lets the next lock holder verify the new hashes
        # and finish cleanup without rolling back a complete transaction.
        pass


def apply_settlement_plan(
    plan_path: Path,
    *,
    expected_digest: str,
    paths: SettlementPaths,
    _fail_after_replace: int | None = None,
    _simulate_crash_after_replace: int | None = None,
) -> dict[str, Any]:
    """Apply a reviewed plan once, or return a verified replay no-op."""
    paths = paths.resolved()
    _validate_settlement_paths(paths)
    plan = load_settlement_plan(plan_path, expected_digest=expected_digest)
    _validate_bound_paths(plan, paths)
    with operational_csv_lock(paths.bets.parent):
        transaction_targets = (
            paths.bets,
            paths.bankroll,
            paths.sessions,
            paths.apply_audit,
        )
        _recover_file_set_transaction(
            paths.transaction_dir,
            expected_targets=transaction_targets,
        )
        bets, predictions, bankroll, sessions, apply_audit = _load_settlement_frames(paths)
        if _verify_replay_or_raise(
            plan,
            bets=bets,
            bankroll=bankroll,
            sessions=sessions,
            apply_audit=apply_audit,
        ):
            return {
                "status": "replay_noop",
                "plan_digest": plan["plan_digest"],
                "applied_rows": 0,
                "previously_applied_rows": len(plan["candidates"]),
            }

        _validate_input_hashes(plan, paths)
        rebuilt = build_settlement_plan(
            paths,
            starting_capital=plan["account"]["starting_capital"],
        )
        if rebuilt["plan_digest"] != plan["plan_digest"] or rebuilt != plan:
            raise ReconciliationConflict(
                "current rows no longer reproduce the reviewed settlement plan"
            )
        if plan["account"].get("global_blockers"):
            raise ReconciliationConflict("plan contains global account-state blockers")
        if not plan["candidates"]:
            return {
                "status": "nothing_to_apply",
                "plan_digest": plan["plan_digest"],
                "applied_rows": 0,
                "previously_applied_rows": 0,
            }

        applied_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        bets_out, bankroll_out, sessions_out, audit_out = _build_applied_frames(
            plan,
            bets=bets,
            bankroll=bankroll,
            sessions=sessions,
            apply_audit=apply_audit,
            applied_at=applied_at,
        )
        actual_post_state = _post_state_manifest(
            original_bet_rows=len(bets),
            original_bankroll_rows=len(bankroll),
            candidates=plan["candidates"],
            bets=bets_out,
            bankroll=bankroll_out,
            sessions=sessions_out,
        )
        if actual_post_state != plan["post_state"]:
            raise ReconciliationConflict(
                "computed output rows do not match the reviewed post-state manifest"
            )
        _ensure_private_apply_audit_directory(paths.apply_audit)
        payloads = (
            ("bets", paths.bets, _csv_bytes(bets_out)),
            ("bankroll", paths.bankroll, _csv_bytes(bankroll_out)),
            ("sessions", paths.sessions, _csv_bytes(sessions_out)),
            ("apply_audit", paths.apply_audit, _csv_bytes(audit_out)),
        )
        _atomic_replace_file_set(
            payloads,
            transaction_id=str(plan["plan_digest"])[:20],
            transaction_dir=paths.transaction_dir,
            _fail_after_replace=_fail_after_replace,
            _simulate_crash_after_replace=_simulate_crash_after_replace,
        )
        return {
            "status": "applied",
            "plan_digest": plan["plan_digest"],
            "applied_rows": len(plan["candidates"]),
            "previously_applied_rows": 0,
            "applied_at_utc": applied_at,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only pending paper-bet audit by default; explicit "
            "digest-gated plan/apply is optional"
        )
    )
    parser.add_argument(
        "--prod-dir", type=Path, default=Path("."),
        help="production directory (default: current directory)",
    )
    parser.add_argument("--bets", type=Path, help="override all_bets.csv path")
    parser.add_argument(
        "--predictions", type=Path, help="override prediction_log.csv path"
    )
    parser.add_argument(
        "--bankroll-history", type=Path,
        help="override bankroll_history.csv path for plan/apply",
    )
    parser.add_argument(
        "--sessions", type=Path,
        help="override betting_sessions.csv path for plan/apply",
    )
    parser.add_argument("--output-csv", type=Path, help="explicit CSV review output")
    parser.add_argument("--output-json", type=Path, help="explicit JSON review output")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--plan-output", type=Path,
        help="write a deterministic settlement plan to this explicit path",
    )
    operation.add_argument(
        "--apply-plan", type=Path,
        help="apply this reviewed settlement plan",
    )
    parser.add_argument(
        "--expected-plan-digest",
        help="required exact SHA-256 digest when --apply-plan is used",
    )
    parser.add_argument(
        "--apply-audit", type=Path,
        help=(
            "explicit canonical .private/pending_reconciliation_apply_audit.csv "
            "path for plan/apply"
        ),
    )
    parser.add_argument(
        "--lock-file", type=Path,
        help="explicit canonical logs/.operational_csv.lock path for plan/apply",
    )
    parser.add_argument(
        "--transaction-dir", type=Path,
        help="explicit canonical durable transaction-recovery directory",
    )
    parser.add_argument(
        "--starting-capital", default="1000",
        help="paper-account starting capital bound into a plan (default: 1000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print full input hashes, bet IDs, and duplicate identity keys",
    )
    args = parser.parse_args(argv)
    if args.apply_plan and not args.expected_plan_digest:
        parser.error("--apply-plan requires --expected-plan-digest")
    if (args.plan_output or args.apply_plan) and not args.apply_audit:
        parser.error("plan/apply requires explicit --apply-audit")
    if (args.plan_output or args.apply_plan) and not args.lock_file:
        parser.error("plan/apply requires explicit canonical --lock-file")
    if (args.plan_output or args.apply_plan) and not args.transaction_dir:
        parser.error("plan/apply requires explicit canonical --transaction-dir")
    if not args.apply_plan and args.expected_plan_digest:
        parser.error("--expected-plan-digest is valid only with --apply-plan")
    if not (args.plan_output or args.apply_plan) and (
        args.apply_audit or args.lock_file or args.transaction_dir
    ):
        parser.error(
            "--apply-audit/--lock-file/--transaction-dir require plan/apply"
        )
    if (args.plan_output or args.apply_plan) and (
        args.output_csv or args.output_json
    ):
        parser.error("review exports and plan/apply are separate operations")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prod_dir = args.prod_dir.resolve()
    bets_path = (args.bets or prod_dir / "logs" / "all_bets.csv").resolve()
    predictions_path = (
        args.predictions or prod_dir / "prediction_log.csv"
    ).resolve()
    if args.plan_output or args.apply_plan:
        paths = SettlementPaths(
            bets=bets_path,
            predictions=predictions_path,
            bankroll=(
                args.bankroll_history or prod_dir / "logs" / "bankroll_history.csv"
            ).resolve(),
            sessions=(
                args.sessions or prod_dir / "logs" / "betting_sessions.csv"
            ).resolve(),
            apply_audit=args.apply_audit.resolve(),
            lock=args.lock_file.resolve(),
            transaction_dir=args.transaction_dir.resolve(),
        )
        try:
            if args.plan_output:
                plan_output = args.plan_output.resolve()
                if plan_output in {
                    **paths.as_dict(),
                    "lock": paths.lock,
                    "transaction_dir": paths.transaction_dir,
                }.values():
                    raise ValueError("plan output must not overlap an operational path")
                plan = build_settlement_plan(
                    paths,
                    starting_capital=args.starting_capital,
                )
                write_settlement_plan(plan, plan_output)
                result = {
                    "status": "plan_written",
                    "plan_schema_version": plan["plan_schema_version"],
                    "plan_digest": plan["plan_digest"],
                    "plan_path": str(plan_output),
                    "summary": plan["summary"],
                    "global_blockers": plan["account"]["global_blockers"],
                }
            else:
                result = apply_settlement_plan(
                    args.apply_plan.resolve(),
                    expected_digest=args.expected_plan_digest,
                    paths=paths,
                )
        except (ValueError, ReconciliationConflict, AtomicApplyError) as exc:
            print(
                json.dumps(
                    {"status": "refused", "error": str(exc)},
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
            )
            return 2
        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return 0

    review, summary = reconcile_paths(bets_path, predictions_path)
    write_review_exports(
        review,
        summary,
        output_csv=args.output_csv.resolve() if args.output_csv else None,
        output_json=args.output_json.resolve() if args.output_json else None,
    )
    console_summary = summary if args.verbose else compact_summary(summary)
    print(
        json.dumps(
            console_summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
