from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from evaluation import cohorts as eval_cohorts  # noqa: E402
from evaluation import metrics as eval_metrics  # noqa: E402

AUDIT_DIR = PRODUCTION_DIR / "logs" / "audit"
LOGS_DIR = PRODUCTION_DIR / "logs"
FEATURE_LOG_GLOB = "features_*.csv"

DATA_PATHS = {
    "prediction_log": PRODUCTION_DIR / "prediction_log.csv",
    "prediction_snapshots": PRODUCTION_DIR / "prediction_snapshots.csv",
    "odds_history": PRODUCTION_DIR / "odds_history.csv",
    "run_history": AUDIT_DIR / "run_history.csv",
    "skipped_live_matches": AUDIT_DIR / "skipped_live_matches.csv",
    "settlement_audit": AUDIT_DIR / "settlement_audit.csv",
    "all_bets": LOGS_DIR / "all_bets.csv",
    "betting_sessions": LOGS_DIR / "betting_sessions.csv",
}

NUMERIC_COLUMNS = [
    "model_p1_prob",
    "model_p2_prob",
    "xgb_p1_prob",
    "xgb_p2_prob",
    "rf_p1_prob",
    "rf_p2_prob",
    "market_p1_prob",
    "market_p2_prob",
    "edge_p1",
    "model_correct",
    "market_correct",
    "xgb_correct",
    "rf_correct",
    "p1_rank",
    "p2_rank",
    "stake",
    "actual_profit",
    "bankroll_before",
    "bankroll_after",
]


def file_mtimes() -> tuple[tuple[str, float | None], ...]:
    """Return a stable cache key based on the tracked dashboard source files."""
    items: list[tuple[str, float | None]] = []
    for name, path in DATA_PATHS.items():
        items.append((name, path.stat().st_mtime if path.exists() else None))
    for path in sorted(LOGS_DIR.glob(FEATURE_LOG_GLOB)):
        items.append((f"feature::{path.name}", path.stat().st_mtime))
    return tuple(items)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    """Parse mixed legacy/ISO timestamps into one UTC-naive representation."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_localize(None)


def _parse_datetimes(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = _parse_datetime_series(df[col])
    return df


def _coalesce_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")

    result = None
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            series = series.replace("", pd.NA)
        result = series.copy() if result is None else result.combine_first(series)
    return result if result is not None else pd.Series(pd.NA, index=df.index, dtype="object")


def _apply_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _boolean_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    """Parse booleans without Python's ``bool('False') == True`` trap."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "1.0", "t", "yes", "y"})


def _decision_grade_series(df: pd.DataFrame) -> pd.Series:
    """Apply the same fail-closed GOLD lineage gate as the evaluation ledger."""
    return (
        df.get("logging_quality", pd.Series("", index=df.index)).fillna("").eq("snapshot_v2")
        & df.get("rescore_quality", pd.Series("", index=df.index)).fillna("").eq("exact_feature_snapshot")
        & _boolean_series(df, "features_complete", default=False)
        & _boolean_series(df, "feature_snapshot_verified", default=False)
    )


def _decorate_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "p1" not in df.columns or "p2" not in df.columns:
        return df
    df["match_label"] = df["p1"].fillna("").astype(str) + " vs " + df["p2"].fillna("").astype(str)
    df["event_label"] = (
        df.get("tournament", pd.Series("", index=df.index)).fillna("").astype(str)
        + " | "
        + df.get("round", pd.Series("", index=df.index)).fillna("").astype(str)
        + " | "
        + df["match_label"]
    )
    return df


def normalize_prediction_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df = _apply_numeric(df)
    df = _parse_datetimes(
        df,
        [
            "logged_at",
            "latest_logged_at",
            "settled_at",
            "odds_scraped_at",
            "latest_odds_scraped_at",
            "match_date",
            "latest_match_date",
        ],
    )

    df["effective_logged_at"] = _parse_datetime_series(
        _coalesce_columns(df, ["latest_logged_at", "logged_at"])
    )
    df["effective_odds_scraped_at"] = _parse_datetime_series(
        _coalesce_columns(df, ["latest_odds_scraped_at", "odds_scraped_at"])
    )
    df["effective_match_start_time"] = _coalesce_columns(df, ["latest_match_start_time", "match_start_time"])
    df["effective_match_date"] = _parse_datetime_series(
        _coalesce_columns(df, ["latest_match_date", "match_date"])
    )
    df["effective_model_version"] = _coalesce_columns(df, ["latest_model_version_seen", "model_version"])
    df["effective_nn_model_version"] = _coalesce_columns(
        df,
        ["latest_nn_model_version_seen", "nn_model_version", "model_version"],
    )
    df["effective_xgb_model_version"] = _coalesce_columns(
        df,
        ["latest_xgb_model_version_seen", "xgb_model_version"],
    )
    df["effective_rf_model_version"] = _coalesce_columns(
        df,
        ["latest_rf_model_version_seen", "rf_model_version"],
    )
    df["effective_nn_probability_source"] = _coalesce_columns(
        df,
        ["latest_nn_probability_source_seen", "nn_probability_source"],
    )
    df["features_complete"] = _boolean_series(df, "features_complete", default=False)
    df["feature_snapshot_verified"] = _boolean_series(
        df, "feature_snapshot_verified", default=False
    )
    df["decision_grade"] = _decision_grade_series(df)
    winner = pd.to_numeric(df.get("actual_winner", pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["is_settled"] = winner.isin([1, 2])
    # An even-money 0.5 line is still valid market evidence. Missing is not.
    df["market_has_pick"] = df.get("market_p1_prob", pd.Series(index=df.index, dtype=float)).notna()
    df["record_status"] = df.get("record_status", pd.Series("", index=df.index)).fillna("unknown")
    df = _decorate_match_labels(df)
    return df


def normalize_prediction_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _apply_numeric(df)
    df = _parse_datetimes(df, ["logged_at", "odds_scraped_at", "match_date"])
    df["features_complete"] = _boolean_series(df, "features_complete", default=False)
    df["feature_snapshot_verified"] = _boolean_series(
        df, "feature_snapshot_verified", default=False
    )
    df["decision_grade"] = _decision_grade_series(df)
    df = _decorate_match_labels(df)
    return df


def normalize_odds_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _apply_numeric(df)
    df = _parse_datetimes(df, ["logged_at", "odds_scraped_at", "match_date"])
    df = _decorate_match_labels(df)
    return df


def normalize_run_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _parse_datetimes(df, ["started_at", "completed_at"])
    return df


def normalize_skipped_live_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _parse_datetimes(df, ["logged_at", "run_started_at", "match_date", "odds_scraped_at"])
    df = _decorate_match_labels(df)
    return df


def normalize_settlement_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _parse_datetimes(df, ["logged_at", "match_date", "ta_match_date_found"])
    df = _decorate_match_labels(df)
    return df


def normalize_all_bets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _apply_numeric(df)
    df = _parse_datetimes(df, ["timestamp", "settled_timestamp", "match_date"])
    df["status"] = df.get("status", pd.Series("", index=df.index)).fillna("unknown")
    df["outcome"] = df.get("outcome", pd.Series("", index=df.index)).fillna("")
    return df


def normalize_betting_sessions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _parse_datetimes(df, ["start_time", "end_time"])
    return df


def load_dashboard_data() -> dict[str, pd.DataFrame]:
    try:
        verified_prediction_log = eval_cohorts.load_prediction_log(str(PRODUCTION_DIR))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        verified_prediction_log = pd.DataFrame()
    return {
        # Reuse the ledger's referential feature-snapshot verifier. Verification
        # failures remain loud instead of quietly promoting ID-shaped strings
        # to decision-grade evidence.
        "prediction_log": normalize_prediction_log(verified_prediction_log),
        "prediction_snapshots": normalize_prediction_snapshots(read_csv_optional(DATA_PATHS["prediction_snapshots"])),
        "odds_history": normalize_odds_history(read_csv_optional(DATA_PATHS["odds_history"])),
        "run_history": normalize_run_history(read_csv_optional(DATA_PATHS["run_history"])),
        "skipped_live_matches": normalize_skipped_live_matches(read_csv_optional(DATA_PATHS["skipped_live_matches"])),
        "settlement_audit": normalize_settlement_audit(read_csv_optional(DATA_PATHS["settlement_audit"])),
        "all_bets": normalize_all_bets(read_csv_optional(DATA_PATHS["all_bets"])),
        "betting_sessions": normalize_betting_sessions(read_csv_optional(DATA_PATHS["betting_sessions"])),
    }


def feature_log_signature() -> tuple[tuple[str, float], ...]:
    return tuple((path.name, path.stat().st_mtime) for path in sorted(LOGS_DIR.glob(FEATURE_LOG_GLOB)))


def find_feature_snapshot_row(feature_snapshot_id: str) -> dict | None:
    feature_snapshot_id = str(feature_snapshot_id or "").strip()
    if not feature_snapshot_id:
        return None

    for path in sorted(LOGS_DIR.glob(FEATURE_LOG_GLOB), reverse=True):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if "feature_snapshot_id" not in header.columns:
            continue
        try:
            for chunk in pd.read_csv(path, chunksize=1000):
                match = chunk[chunk["feature_snapshot_id"].astype(str) == feature_snapshot_id]
                if not match.empty:
                    row = match.iloc[0].to_dict()
                    row["_source_file"] = str(path)
                    return row
        except Exception:
            continue
    return None


def build_family_results(prediction_log: pd.DataFrame) -> pd.DataFrame:
    if prediction_log.empty:
        return pd.DataFrame()

    settled = prediction_log[prediction_log["is_settled"] & prediction_log["features_complete"]].copy()
    if settled.empty:
        return pd.DataFrame()

    specs = [
        ("NN", "model_p1_prob", "effective_nn_model_version", "effective_nn_probability_source"),
        ("XGB", "xgb_p1_prob", "effective_xgb_model_version", None),
        ("RF", "rf_p1_prob", "effective_rf_model_version", None),
        ("Market", "market_p1_prob", None, None),
    ]

    frames: list[pd.DataFrame] = []
    for family, prob_col, version_col, source_col in specs:
        if prob_col not in settled.columns:
            continue

        family_df = settled[
            [
                "prediction_uid",
                "match_uid",
                "match_label",
                "event_label",
                "surface",
                "level",
                "round",
                "decision_grade",
                "market_has_pick",
                "effective_logged_at",
                "settled_at",
                "effective_match_date",
                "effective_odds_scraped_at",
                "effective_model_version",
                "actual_winner",
                prob_col,
            ]
        ].copy()
        family_df = family_df.rename(columns={prob_col: "p1_prob"})
        family_df["family"] = family
        family_df["version"] = settled[version_col].values if version_col else family
        family_df["nn_probability_source"] = settled[source_col].values if source_col else ""
        family_df["p1_prob"] = pd.to_numeric(family_df["p1_prob"], errors="coerce")
        family_df["actual_winner"] = pd.to_numeric(family_df["actual_winner"], errors="coerce")
        family_df = family_df[
            family_df["actual_winner"].isin([1, 2]) & family_df["p1_prob"].notna()
        ]
        family_df["correct"] = (
            ((family_df["p1_prob"] >= 0.5) & family_df["actual_winner"].eq(1))
            | ((family_df["p1_prob"] < 0.5) & family_df["actual_winner"].eq(2))
        ).astype(float)
        family_df["confidence"] = np.maximum(family_df["p1_prob"], 1 - family_df["p1_prob"])
        frames.append(family_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_family_accuracy_summary(prediction_log: pd.DataFrame) -> pd.DataFrame:
    apples = build_apples_to_apples_rows(prediction_log)
    metrics = build_metrics_summary(apples)
    if metrics.empty:
        return pd.DataFrame()
    market_rows = metrics[metrics["family"] == "Market"]
    market_accuracy = float(market_rows.iloc[0]["accuracy"]) if not market_rows.empty else np.nan
    result = metrics[["family", "matches", "accuracy", "versions"]].copy()
    result["market_accuracy"] = market_accuracy
    result["edge_vs_market"] = result["accuracy"] - market_accuracy
    return result.sort_values("family", ignore_index=True)


def build_version_summary(family_results: pd.DataFrame) -> pd.DataFrame:
    if family_results.empty:
        return pd.DataFrame()
    usable = family_results[family_results["family"] != "Market"].copy()
    if usable.empty:
        return pd.DataFrame()
    summary = (
        usable.groupby(["family", "version"], dropna=False)
        .agg(matches=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
        .sort_values(["family", "matches"], ascending=[True, False], ignore_index=True)
    )
    return summary


def build_calibration_summary(family_results: pd.DataFrame) -> pd.DataFrame:
    if family_results.empty:
        return pd.DataFrame()

    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.01]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-90%", "90-100%"]

    df = family_results.copy()
    df["confidence_bin"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    grouped = (
        df.groupby(["family", "confidence_bin"], observed=True)
        .agg(matches=("correct", "size"), win_rate=("correct", "mean"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )
    return grouped


def build_apples_to_apples_rows(prediction_log: pd.DataFrame) -> pd.DataFrame:
    if prediction_log.empty:
        return pd.DataFrame()

    required = [
        "model_p1_prob",
        "xgb_p1_prob",
        "rf_p1_prob",
        "market_p1_prob",
        "actual_winner",
    ]
    # Headline comparison uses the GOLD common cohort only.
    settled = prediction_log[
        prediction_log["is_settled"]
        & prediction_log["decision_grade"]
        & prediction_log["market_has_pick"]
    ].copy()
    for col in required:
        if col not in settled.columns:
            return pd.DataFrame()
        settled = settled[settled[col].notna()]
    if settled.empty:
        return pd.DataFrame()
    settled["p1_won"] = (settled["actual_winner"] == 1).astype(int)
    return settled


def build_metrics_summary(apples_df: pd.DataFrame) -> pd.DataFrame:
    if apples_df.empty:
        return pd.DataFrame()

    y_true = apples_df["p1_won"].astype(int)
    specs = [
        ("NN", "model_p1_prob", "effective_nn_model_version"),
        ("XGB", "xgb_p1_prob", "effective_xgb_model_version"),
        ("RF", "rf_p1_prob", "effective_rf_model_version"),
        ("Market", "market_p1_prob", None),
    ]
    rows = []
    for family, prob_col, version_col in specs:
        if prob_col not in apples_df.columns:
            continue
        probs = apples_df[prob_col].astype(float).clip(1e-6, 1 - 1e-6)
        # Shared scoring implementation (see production/evaluation/metrics.py)
        scores = eval_metrics.compute_all(y_true.to_numpy(), probs.to_numpy(), n_bins=10)
        versions = (
            ", ".join(sorted(apples_df[version_col].dropna().astype(str).unique()))
            if version_col and version_col in apples_df.columns
            else family
        )
        rows.append(
            {
                "family": family,
                "matches": int(len(apples_df)),
                "accuracy": scores["accuracy"],
                "auc": scores["auc"],
                "brier": scores["brier"],
                "log_loss": scores["log_loss"],
                "ece": scores["ece"],
                "avg_confidence": float(np.maximum(probs, 1 - probs).mean()),
                "versions": versions,
            }
        )
    return pd.DataFrame(rows)


def build_roc_curve_data(apples_df: pd.DataFrame) -> pd.DataFrame:
    if apples_df.empty:
        return pd.DataFrame()

    y_true = apples_df["p1_won"].astype(int)
    specs = [
        ("NN", "model_p1_prob"),
        ("XGB", "xgb_p1_prob"),
        ("RF", "rf_p1_prob"),
        ("Market", "market_p1_prob"),
    ]
    rows = []
    for family, prob_col in specs:
        if prob_col not in apples_df.columns:
            continue
        probs = apples_df[prob_col].astype(float)
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        for idx in range(len(fpr)):
            rows.append(
                {
                    "family": family,
                    "fpr": float(fpr[idx]),
                    "tpr": float(tpr[idx]),
                    "threshold": float(thresholds[idx]) if idx < len(thresholds) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_reliability_curve_data(apples_df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    if apples_df.empty:
        return pd.DataFrame()

    y_true = apples_df["p1_won"].astype(int)
    specs = [
        ("NN", "model_p1_prob"),
        ("XGB", "xgb_p1_prob"),
        ("RF", "rf_p1_prob"),
        ("Market", "market_p1_prob"),
    ]
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for family, prob_col in specs:
        if prob_col not in apples_df.columns:
            continue
        probs = apples_df[prob_col].astype(float).to_numpy()
        truth = y_true.to_numpy()
        for idx in range(bins):
            left = bin_edges[idx]
            right = bin_edges[idx + 1]
            if idx == bins - 1:
                mask = (probs >= left) & (probs <= right)
            else:
                mask = (probs >= left) & (probs < right)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "family": family,
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_mid": float((left + right) / 2),
                    "matches": int(mask.sum()),
                    "avg_predicted": float(probs[mask].mean()),
                    "actual_rate": float(truth[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def latest_pipeline_run_id(run_history: pd.DataFrame,
                           prediction_snapshots: pd.DataFrame) -> str:
    """Choose the latest accepted prediction-bearing run before UI filters."""
    snapshot_ids: set[str] = set()
    if not prediction_snapshots.empty and "run_id" in prediction_snapshots.columns:
        snapshot_ids = set(
            value
            for value in prediction_snapshots["run_id"].fillna("").astype(str)
            if value
        )
    if not run_history.empty and "run_id" in run_history.columns:
        candidates = run_history.copy()
        candidates = candidates[
            candidates["run_id"].fillna("").astype(str).str.startswith("run_")
        ]
        candidates = candidates[
            candidates["run_id"].fillna("").astype(str).isin(snapshot_ids)
        ]
        if "run_kind" in candidates.columns:
            kinds = candidates["run_kind"].fillna("").astype(str)
            candidates = candidates[kinds.isin(["", "prediction_pipeline"])]
        if "status" in candidates.columns:
            statuses = candidates["status"].fillna("").astype(str).str.lower()
            candidates = candidates[statuses.isin(["", "success", "partial"])]
        if not candidates.empty:
            if "started_at" in candidates.columns:
                candidates = candidates.assign(
                    _started=pd.to_datetime(
                        candidates["started_at"], errors="coerce", utc=True,
                        format="mixed",
                    )
                ).sort_values(["_started", "run_id"], kind="stable", na_position="first")
            else:
                candidates = candidates.sort_values("run_id", kind="stable")
            return str(
                candidates.iloc[-1]["run_id"]
            )
    if not prediction_snapshots.empty and "run_id" in prediction_snapshots.columns:
        candidates = prediction_snapshots[
            prediction_snapshots["run_id"].fillna("").astype(str).ne("")
        ]
        if not candidates.empty:
            return str(
                candidates.sort_values(["logged_at", "run_id"], kind="stable")
                .iloc[-1]["run_id"]
            )
    return ""


def build_live_latest_snapshots(prediction_snapshots: pd.DataFrame,
                                run_id: str = "") -> pd.DataFrame:
    if prediction_snapshots.empty:
        return pd.DataFrame()

    source = prediction_snapshots.copy()
    if run_id and "run_id" in source.columns:
        source = source[source["run_id"].fillna("").astype(str) == str(run_id)].copy()
        if source.empty:
            return pd.DataFrame()
    elif "run_id" in source.columns and source["run_id"].fillna("").ne("").any():
        run_order = (
            source.groupby("run_id", dropna=False)["logged_at"].max().sort_values()
        )
        source = source[source["run_id"] == run_order.index[-1]].copy()
    latest = (
        source.sort_values(["logged_at", "prediction_uid"])
        .groupby("match_uid", as_index=False)
        .tail(1)
    )
    sort_columns = [column for column in ["match_date", "logged_at", "match_label"] if column in latest]
    ascending = {"match_date": True, "logged_at": False, "match_label": True}
    if sort_columns:
        latest = latest.sort_values(sort_columns, ascending=[ascending[c] for c in sort_columns])
    latest = latest.reset_index(drop=True)
    p1_model_cols = [col for col in ["model_p1_prob", "xgb_p1_prob", "rf_p1_prob"] if col in latest.columns]
    p2_model_cols = [col for col in ["model_p2_prob", "xgb_p2_prob", "rf_p2_prob"] if col in latest.columns]
    p1_range_cols = [col for col in p1_model_cols + ["market_p1_prob"] if col in latest.columns]
    latest["probability_range"] = latest[p1_range_cols].max(axis=1) - latest[p1_range_cols].min(axis=1)
    latest["consensus_p1_prob"] = latest[p1_model_cols].mean(axis=1) if p1_model_cols else np.nan
    latest["consensus_p2_prob"] = latest[p2_model_cols].mean(axis=1) if p2_model_cols else 1 - latest["consensus_p1_prob"]

    edge_specs = [
        ("nn", "model"),
        ("xgb", "xgb"),
        ("rf", "rf"),
        ("consensus", "consensus"),
    ]
    edge_cols = []
    for label, prefix in edge_specs:
        p1_col = f"{prefix}_p1_prob" if prefix != "model" else "model_p1_prob"
        p2_col = f"{prefix}_p2_prob" if prefix != "model" else "model_p2_prob"
        p1_edge_col = f"{label}_p1_market_edge"
        p2_edge_col = f"{label}_p2_market_edge"
        p1_lift_col = f"{label}_p1_market_lift"
        p2_lift_col = f"{label}_p2_market_lift"
        if p1_col in latest.columns and "market_p1_prob" in latest.columns:
            latest[p1_edge_col] = latest[p1_col] - latest["market_p1_prob"]
            latest[p1_lift_col] = latest[p1_edge_col] / latest["market_p1_prob"].replace(0, np.nan)
            edge_cols.append(p1_edge_col)
        if p2_col in latest.columns and "market_p2_prob" in latest.columns:
            latest[p2_edge_col] = latest[p2_col] - latest["market_p2_prob"]
            latest[p2_lift_col] = latest[p2_edge_col] / latest["market_p2_prob"].replace(0, np.nan)
            edge_cols.append(p2_edge_col)

    latest["nn_vs_market_edge"] = latest.get("nn_p1_market_edge", pd.Series(np.nan, index=latest.index))
    latest["xgb_vs_market_edge"] = latest.get("xgb_p1_market_edge", pd.Series(np.nan, index=latest.index))
    latest["rf_vs_market_edge"] = latest.get("rf_p1_market_edge", pd.Series(np.nan, index=latest.index))
    latest["largest_player_market_edge"] = latest[edge_cols].abs().max(axis=1) if edge_cols else np.nan
    if {"consensus_p1_market_edge", "consensus_p2_market_edge"}.issubset(latest.columns):
        latest["consensus_edge_player"] = np.where(
            latest["consensus_p1_market_edge"] >= latest["consensus_p2_market_edge"],
            latest["p1"],
            latest["p2"],
        )
        latest["consensus_edge"] = latest[["consensus_p1_market_edge", "consensus_p2_market_edge"]].max(axis=1)
    else:
        latest["consensus_edge_player"] = ""
        latest["consensus_edge"] = np.nan
    return latest


def build_match_catalog(prediction_log: pd.DataFrame, prediction_snapshots: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not prediction_log.empty:
        frames.append(
            prediction_log[
                [
                    "match_uid",
                    "match_label",
                    "event_label",
                    "effective_logged_at",
                    "decision_grade",
                    "record_status",
                ]
            ].rename(columns={"effective_logged_at": "sort_ts"})
        )
    if not prediction_snapshots.empty:
        frames.append(
            prediction_snapshots[
                ["match_uid", "match_label", "event_label", "logged_at", "decision_grade", "record_status"]
            ].rename(columns={"logged_at": "sort_ts"})
        )
    if not frames:
        return pd.DataFrame(columns=["match_uid", "match_label", "event_label", "sort_ts", "decision_grade", "record_status"])

    catalog = pd.concat(frames, ignore_index=True).dropna(subset=["match_uid"]).copy()
    catalog = (
        catalog.sort_values("sort_ts", ascending=False)
        .drop_duplicates(subset=["match_uid"], keep="first")
        .reset_index(drop=True)
    )
    return catalog


def build_derived_run_history(prediction_snapshots: pd.DataFrame, odds_history: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not prediction_snapshots.empty and "run_id" in prediction_snapshots.columns:
        frames.append(
            prediction_snapshots.groupby("run_id", dropna=False)
            .agg(
                started_at=("logged_at", "min"),
                completed_at=("logged_at", "max"),
                prediction_rows_success=("prediction_uid", "nunique"),
            )
            .reset_index()
        )
    if odds_history is not None and not odds_history.empty and "run_id" in odds_history.columns:
        odds_group = (
            odds_history.groupby("run_id", dropna=False)
            .agg(
                odds_started_at=("logged_at", "min"),
                odds_completed_at=("logged_at", "max"),
                odds_rows_fetched=("odds_snapshot_uid", "nunique"),
            )
            .reset_index()
        )
        frames.append(odds_group)

    if not frames:
        return pd.DataFrame()

    merged = None
    for frame in frames:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="run_id", how="outer")

    started = merged.get("started_at", pd.Series(pd.NaT, index=merged.index))
    completed = merged.get("completed_at", pd.Series(pd.NaT, index=merged.index))
    merged["started_at"] = started.combine_first(merged.get("odds_started_at", started))
    merged["completed_at"] = completed.combine_first(merged.get("odds_completed_at", completed))
    merged["status"] = "derived_from_snapshots"
    merged["run_source"] = "derived"
    return merged.sort_values("started_at", ascending=False, ignore_index=True)
