#!/usr/bin/env python3
"""Backfill pending bet settlements from the settled prediction log."""

from pathlib import Path

import pandas as pd

from utils.bet_tracker import BetTracker
from settlement_attribution import (
    build_prediction_result_consensus,
    build_prediction_result_evidence,
    load_feature_attribution_evidence,
    load_verified_prediction_log,
    parse_actual_winner,
    prediction_match_supports_exact_attribution,
)


BASE_DIR = Path(__file__).parent
PREDICTION_LOG_PATH = BASE_DIR / "prediction_log.csv"
LOGS_DIR = BASE_DIR / "logs"


def main():
    if not PREDICTION_LOG_PATH.exists():
        print("No prediction_log.csv found.")
        return 0

    tracker = BetTracker(str(LOGS_DIR))
    df = pd.read_csv(PREDICTION_LOG_PATH)
    verified_predictions = pd.DataFrame()
    exact_feature_evidence = {}
    repaired_count = 0
    try:
        verified_predictions = load_verified_prediction_log(BASE_DIR)
        exact_feature_evidence = load_feature_attribution_evidence(BASE_DIR)
        repaired_count = tracker.repair_settled_bet_attribution(
            verified_predictions, exact_feature_evidence
        )
    except Exception as exc:
        print(
            "Exact attribution unavailable; settlement will remain "
            f"accounting-only: {exc}"
        )
    parsed_winner = df["actual_winner"].map(parse_actual_winner)
    invalid_result_count = int(
        (
            df["actual_winner"].notna()
            & df["actual_winner"].astype(str).str.strip().ne("")
            & parsed_winner.isna()
        ).sum()
    )
    if invalid_result_count:
        print(
            f"Skipped {invalid_result_count} non-integral/invalid winner "
            "label(s) during tracker sync"
        )
    settled = df[parsed_winner.notna()].copy()
    if "record_status" in settled.columns:
        settled = settled[
            ~settled["record_status"].fillna("").astype(str).str.lower().isin(
                {"identity_conflict", "superseded_identity"}
            )
        ].copy()
    if settled.empty:
        print("No settled prediction rows found.")
        return 0

    settled_count = 0
    settled_uid = settled.get(
        "match_uid", pd.Series("", index=settled.index)
    ).fillna("").astype(str).str.strip()
    blank_uid_count = int(settled_uid.eq("").sum())
    if blank_uid_count:
        print(
            f"Skipped {blank_uid_count} settled prediction row(s) without "
            "an exact match UID"
        )
    settled = settled[settled_uid.ne("")].copy()
    settled["_exact_uid"] = settled.get(
        "match_uid", pd.Series("", index=settled.index)
    ).fillna("").astype(str).str.strip()

    for match_uid, group in settled.groupby("_exact_uid", sort=False):
        consensus = build_prediction_result_consensus(
            df,
            match_uid,
            require_settled=True,
            require_exact_support=False,
        )
        if consensus is None:
            print(
                f"Skipped conflicting or incomplete settled UID group: {match_uid}"
            )
            continue
        sort_columns = [
            column
            for column in ("prediction_uid", "settled_at")
            if column in group
        ]
        ordered_group = (
            group.sort_values(sort_columns, kind="stable")
            if sort_columns
            else group
        )
        row = ordered_group.iloc[0]
        strict_winner = parse_actual_winner(row.get("actual_winner"))
        if strict_winner is None:
            continue
        bound_result_evidence = build_prediction_result_evidence(row)
        result_evidence_kind, result_evidence_sha256 = bound_result_evidence
        settled_count += tracker.settle_pending_bets_for_match(
            match_uid=match_uid,
            alias_match_uids=(
                [
                    value.strip()
                    for value in str(
                        row.get("identity_related_match_uid") or ""
                    ).split("|")
                    if value.strip()
                ]
                if str(row.get("identity_status") or "").strip().lower()
                == "canonical_alias"
                else []
            ),
            p1=row.get("p1"),
            p2=row.get("p2"),
            actual_winner=strict_winner,
            notes=f"Backfilled from prediction log | score={row.get('score', '')}",
            exact_feature_evidence=(
                exact_feature_evidence
                if prediction_match_supports_exact_attribution(
                    verified_predictions,
                    match_uid,
                    p1=row.get("p1"),
                    p2=row.get("p2"),
                    actual_winner=strict_winner,
                )
                else {}
            ),
            result_evidence_kind=result_evidence_kind,
            result_evidence_sha256=result_evidence_sha256,
            bound_result_evidence=bound_result_evidence,
        )

    print(
        f"Backfilled {settled_count} tracked bet settlement(s); repaired "
        f"{repaired_count} exact legacy attribution row(s)"
    )
    return settled_count


if __name__ == "__main__":
    main()
