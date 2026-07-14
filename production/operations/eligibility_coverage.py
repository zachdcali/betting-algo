"""Run-pinned eligibility coverage diagnostics.

Counts are derived from one explicitly selected prediction run. Acceptance is
an operator-supplied fact from the manifest-pinned dashboard generation; this
local tool does not infer or independently certify it. The
categories are orthogonal: a snapshot may be blocked by height, identity, and
round evidence at the same time, so category counts must never be summed into
an implied recoverable-row total.  Source-profile lookup failures are counted
from skip evidence and kept outside the snapshot denominator.

This command is read-only.  It never queries or changes Supabase and never
claims an after-replay count without an actual accepted replay.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd


SNAPSHOT_CATEGORY_ORDER = (
    "height",
    "identity_conflict",
    "round",
    "structural_validation",
    "rank_volatility",
)
REGISTRY_LEVERS = frozenset({
    "height", "identity_conflict", "round", "source_profile_lookup_failure",
})
TEXT_COLUMNS = (
    "defaulted_features", "record_status", "record_note", "identity_status",
    "identity_conflict_fields", "build_status", "status", "status_detail",
    "skip_reason_code", "skip_reason_detail",
)


@dataclass(frozen=True)
class RankedLever:
    category: str
    blocked_rows: int
    evidence_source: str
    denominator: str


@dataclass(frozen=True)
class EligibilityCoverageReport:
    run_id: str
    sync_id: str
    snapshot_rows: int
    complete_rows: int
    incomplete_rows: int
    complete_rate: float | None
    orthogonal_snapshot_blockers: Mapping[str, int]
    skip_only_blockers: Mapping[str, int]
    ranked_registry_levers: tuple[RankedLever, ...]
    after_replay_rows: int | None = None
    caveat: str = (
        "Blocker counts are orthogonal and can overlap. after_replay_rows stays "
        "null until a real replay is accepted; counts are not added as a forecast."
    )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ranked_registry_levers"] = [
            asdict(item) for item in self.ranked_registry_levers
        ]
        return result


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).casefold() in {"1", "true", "yes", "y"}


def _combined_text(row: Mapping[str, Any]) -> str:
    return " | ".join(_clean(row.get(column)) for column in TEXT_COLUMNS).casefold()


def _snapshot_key(frame: pd.DataFrame) -> pd.Series:
    key = pd.Series("", index=frame.index, dtype="object")
    for column in ("feature_snapshot_id", "prediction_uid", "match_uid"):
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        key = key.mask(key.eq("") & values.ne(""), column + ":" + values)
    return key.mask(key.eq(""), "row:" + frame.index.astype(str))


def _dedupe_run_snapshots(snapshots: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if "run_id" not in snapshots.columns:
        raise ValueError("snapshot input is missing run_id")
    selected = snapshots[
        snapshots["run_id"].fillna("").astype(str).str.strip().eq(run_id)
    ].copy()
    if selected.empty:
        return selected
    selected["_snapshot_key"] = _snapshot_key(selected)
    if "logged_at" in selected.columns:
        selected["_sort_time"] = pd.to_datetime(
            selected["logged_at"], errors="coerce", utc=True,
        )
        selected = selected.sort_values("_sort_time", kind="stable")
    return selected.drop_duplicates("_snapshot_key", keep="last")


def _snapshot_categories(row: Mapping[str, Any]) -> set[str]:
    text = _combined_text(row)
    categories: set[str] = set()
    if re.search(r"(^|[^a-z])(?:p1_|p2_|player1_|player2_)?height(?:_cm)?([^a-z]|$)", text):
        categories.add("height")
    if (
        _clean(row.get("record_status")).casefold() == "identity_conflict"
        or "identity_conflict" in text
        or "match identity conflict" in text
    ):
        categories.add("identity_conflict")
    structural_round = bool(re.search(
        r"one_hot_cardinality:round:0|structural_validation[^|]*round|"
        r"feature_schema_invalid[^|]*round",
        text,
    ))
    if structural_round:
        categories.add("structural_validation")
        categories.add("round")
    if re.search(r"round_code\s*=\s*none|missing_round|round_code:none", text):
        categories.add("round")
    if re.search(r"rank[_ ]volatility", text):
        categories.add("rank_volatility")
    return categories


def _source_profile_failures(skips: pd.DataFrame, run_id: str) -> int:
    if skips.empty or "run_id" not in skips.columns:
        return 0
    selected = skips[
        skips["run_id"].fillna("").astype(str).str.strip().eq(run_id)
    ].copy()
    if selected.empty:
        return 0
    text = selected.apply(lambda row: _combined_text(row), axis=1)
    mask = text.str.contains(
        r"ta profile load failed|source[_ ]profile lookup failure|"
        r"profile lookup failed|missing[_ ]slug",
        regex=True,
        na=False,
    )
    selected = selected.loc[mask].copy()
    if selected.empty:
        return 0
    for column in ("skip_event_id", "feature_snapshot_id", "match_uid"):
        if column in selected.columns:
            keys = selected[column].fillna("").astype(str).str.strip()
            nonblank = keys.ne("")
            if nonblank.any():
                return int(keys[nonblank].nunique()) + int((~nonblank).sum())
    return int(len(selected))


def summarize_eligibility_coverage(
    snapshots: pd.DataFrame,
    *,
    run_id: str,
    sync_id: str = "",
    skips: pd.DataFrame | None = None,
) -> EligibilityCoverageReport:
    run = str(run_id or "").strip()
    if not run:
        raise ValueError("run_id is required; do not infer an accepted run")
    selected = _dedupe_run_snapshots(snapshots, run)
    if selected.empty:
        raise ValueError(
            f"run_id {run!r} is absent from snapshot input; refusing a false zero report"
        )
    total = int(len(selected))
    if total and "features_complete" not in selected.columns:
        raise ValueError("snapshot input is missing features_complete")
    complete = int(selected["features_complete"].map(_bool).sum()) if total else 0
    incomplete = total - complete
    counts = {category: 0 for category in SNAPSHOT_CATEGORY_ORDER}
    if total:
        for _, row in selected.loc[
            ~selected["features_complete"].map(_bool)
        ].iterrows():
            for category in _snapshot_categories(row):
                counts[category] += 1

    source_profile = _source_profile_failures(
        skips if skips is not None else pd.DataFrame(), run,
    )
    skip_counts = {"source_profile_lookup_failure": source_profile}
    levers: list[RankedLever] = []
    for category, count in counts.items():
        if category in REGISTRY_LEVERS and count:
            levers.append(RankedLever(
                category, int(count), "prediction_snapshots", "snapshot_rows",
            ))
    if source_profile:
        levers.append(RankedLever(
            "source_profile_lookup_failure", source_profile,
            "skipped_live_matches", "skip_events",
        ))
    stable_order = {name: index for index, name in enumerate((
        "height", "identity_conflict", "round",
        "source_profile_lookup_failure",
    ))}
    levers.sort(key=lambda item: (
        -item.blocked_rows, stable_order.get(item.category, 999), item.category,
    ))
    return EligibilityCoverageReport(
        run_id=run,
        sync_id=str(sync_id or "").strip(),
        snapshot_rows=total,
        complete_rows=complete,
        incomplete_rows=incomplete,
        complete_rate=(complete / total if total else None),
        orthogonal_snapshot_blockers=counts,
        skip_only_blockers=skip_counts,
        ranked_registry_levers=tuple(levers),
    )


def _read_csv(path: Path, *, optional: bool = False) -> pd.DataFrame:
    if optional and not path.exists():
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report orthogonal eligibility blockers for one accepted run",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sync-id", default="")
    parser.add_argument(
        "--snapshots", type=Path,
        default=Path("production/prediction_snapshots.csv"),
    )
    parser.add_argument(
        "--skips", type=Path,
        default=Path("production/logs/audit/skipped_live_matches.csv"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = summarize_eligibility_coverage(
            _read_csv(args.snapshots),
            run_id=args.run_id,
            sync_id=args.sync_id,
            skips=_read_csv(args.skips, optional=True),
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
