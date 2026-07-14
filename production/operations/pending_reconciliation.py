"""Audit pending paper bets without changing operational state.

This module deliberately stops at evidence assembly.  It does not call the bet
tracker, does not scrape a result source, and never updates either input CSV.
The prediction log is the only authoritative winner source: a result is exact
only when one ``match_uid`` resolves to one valid winner identity and no
void/cancellation marker conflicts with it.

Run from ``production/``::

    ../tennis_env/bin/python -m operations.pending_reconciliation --prod-dir .

Review files are optional and are written only when an explicit output path is
provided::

    ../tennis_env/bin/python -m operations.pending_reconciliation \
        --prod-dir . --output-csv /tmp/pending-review.csv \
        --output-json /tmp/pending-review.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import pandas as pd

from logging_utils import normalize_name, normalize_text


RECONCILIATION_SCHEMA_VERSION = "1.1.0"

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only pending paper-bet reconciliation audit"
    )
    parser.add_argument(
        "--prod-dir", type=Path, default=Path("."),
        help="production directory (default: current directory)",
    )
    parser.add_argument("--bets", type=Path, help="override all_bets.csv path")
    parser.add_argument(
        "--predictions", type=Path, help="override prediction_log.csv path"
    )
    parser.add_argument("--output-csv", type=Path, help="explicit CSV review output")
    parser.add_argument("--output-json", type=Path, help="explicit JSON review output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print full input hashes, bet IDs, and duplicate identity keys",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prod_dir = args.prod_dir.resolve()
    bets_path = (args.bets or prod_dir / "logs" / "all_bets.csv").resolve()
    predictions_path = (
        args.predictions or prod_dir / "prediction_log.csv"
    ).resolve()
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
