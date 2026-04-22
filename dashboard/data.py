from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
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


def _parse_datetimes(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
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

    df["effective_logged_at"] = pd.to_datetime(
        _coalesce_columns(df, ["latest_logged_at", "logged_at"]),
        errors="coerce",
    )
    df["effective_odds_scraped_at"] = pd.to_datetime(
        _coalesce_columns(df, ["latest_odds_scraped_at", "odds_scraped_at"]),
        errors="coerce",
    )
    df["effective_match_start_time"] = _coalesce_columns(df, ["latest_match_start_time", "match_start_time"])
    df["effective_match_date"] = pd.to_datetime(
        _coalesce_columns(df, ["latest_match_date", "match_date"]),
        errors="coerce",
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
    df["decision_grade"] = (
        df.get("logging_quality", pd.Series("", index=df.index)).fillna("").eq("snapshot_v2")
        & df.get("rescore_quality", pd.Series("", index=df.index)).fillna("").eq("exact_feature_snapshot")
    )
    df["is_settled"] = df.get("actual_winner", pd.Series(index=df.index, dtype=float)).notna()
    df["market_has_pick"] = df.get("market_p1_prob", pd.Series(index=df.index, dtype=float)).round(4).ne(0.5)
    df["features_complete"] = df.get("features_complete", pd.Series(True, index=df.index)).fillna(True).astype(bool)
    df["record_status"] = df.get("record_status", pd.Series("", index=df.index)).fillna("unknown")
    df = _decorate_match_labels(df)
    return df


def normalize_prediction_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = _apply_numeric(df)
    df = _parse_datetimes(df, ["logged_at", "odds_scraped_at", "match_date"])
    df["decision_grade"] = (
        df.get("logging_quality", pd.Series("", index=df.index)).fillna("").eq("snapshot_v2")
        & df.get("rescore_quality", pd.Series("", index=df.index)).fillna("").eq("exact_feature_snapshot")
    )
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
    return {
        "prediction_log": normalize_prediction_log(read_csv_optional(DATA_PATHS["prediction_log"])),
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
        ("NN", "model_correct", "model_p1_prob", "effective_nn_model_version", "effective_nn_probability_source"),
        ("XGB", "xgb_correct", "xgb_p1_prob", "effective_xgb_model_version", None),
        ("RF", "rf_correct", "rf_p1_prob", "effective_rf_model_version", None),
        ("Market", "market_correct", "market_p1_prob", None, None),
    ]

    frames: list[pd.DataFrame] = []
    for family, correct_col, prob_col, version_col, source_col in specs:
        if correct_col not in settled.columns or prob_col not in settled.columns:
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
                correct_col,
                prob_col,
            ]
        ].copy()
        family_df = family_df.rename(columns={correct_col: "correct", prob_col: "p1_prob"})
        family_df["family"] = family
        family_df["version"] = settled[version_col].values if version_col else family
        family_df["nn_probability_source"] = settled[source_col].values if source_col else ""
        family_df["correct"] = pd.to_numeric(family_df["correct"], errors="coerce")
        family_df["p1_prob"] = pd.to_numeric(family_df["p1_prob"], errors="coerce")
        family_df = family_df[family_df["correct"].notna() & family_df["p1_prob"].notna()]
        family_df["confidence"] = np.maximum(family_df["p1_prob"], 1 - family_df["p1_prob"])
        frames.append(family_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_family_accuracy_summary(prediction_log: pd.DataFrame) -> pd.DataFrame:
    if prediction_log.empty:
        return pd.DataFrame()

    settled = prediction_log[prediction_log["is_settled"] & prediction_log["features_complete"]].copy()
    if settled.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    families = [
        ("NN", "model_correct", "effective_nn_model_version"),
        ("XGB", "xgb_correct", "effective_xgb_model_version"),
        ("RF", "rf_correct", "effective_rf_model_version"),
    ]
    for family, correct_col, version_col in families:
        if correct_col not in settled.columns:
            continue
        mask = settled[correct_col].notna() & settled["market_has_pick"]
        subset = settled[mask].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "family": family,
                "matches": int(len(subset)),
                "accuracy": float(subset[correct_col].mean()),
                "market_accuracy": float(subset["market_correct"].mean()),
                "edge_vs_market": float(subset[correct_col].mean() - subset["market_correct"].mean()),
                "versions": ", ".join(sorted(subset[version_col].dropna().astype(str).unique())) if version_col in subset.columns else "",
            }
        )

    market_mask = settled["market_correct"].notna() & settled["market_has_pick"]
    market_subset = settled[market_mask].copy()
    if not market_subset.empty:
        rows.append(
            {
                "family": "Market",
                "matches": int(len(market_subset)),
                "accuracy": float(market_subset["market_correct"].mean()),
                "market_accuracy": float(market_subset["market_correct"].mean()),
                "edge_vs_market": 0.0,
                "versions": "consensus line",
            }
        )

    return pd.DataFrame(rows).sort_values(["family"], ignore_index=True) if rows else pd.DataFrame()


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
        "model_correct",
        "xgb_correct",
        "rf_correct",
        "market_correct",
        "model_p1_prob",
        "xgb_p1_prob",
        "rf_p1_prob",
        "market_p1_prob",
        "actual_winner",
    ]
    settled = prediction_log[prediction_log["is_settled"] & prediction_log["features_complete"] & prediction_log["market_has_pick"]].copy()
    for col in required:
        if col not in settled.columns:
            return pd.DataFrame()
        settled = settled[settled[col].notna()]
    if settled.empty:
        return pd.DataFrame()
    settled["p1_won"] = (settled["actual_winner"] == 1).astype(int)
    return settled


def expected_calibration_error(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    probs = np.asarray(y_prob, dtype=float)
    truth = np.asarray(y_true, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(probs)
    for idx in range(bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = truth[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / total)
    return float(ece)


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
        predicted = (probs >= 0.5).astype(int)
        versions = (
            ", ".join(sorted(apples_df[version_col].dropna().astype(str).unique()))
            if version_col and version_col in apples_df.columns
            else family
        )
        rows.append(
            {
                "family": family,
                "matches": int(len(apples_df)),
                "accuracy": float((predicted == y_true).mean()),
                "auc": float(roc_auc_score(y_true, probs)),
                "brier": float(brier_score_loss(y_true, probs)),
                "log_loss": float(log_loss(y_true, probs)),
                "ece": expected_calibration_error(y_true, probs, bins=10),
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


def build_live_latest_snapshots(prediction_snapshots: pd.DataFrame) -> pd.DataFrame:
    if prediction_snapshots.empty:
        return pd.DataFrame()

    latest = (
        prediction_snapshots.sort_values(["logged_at", "prediction_uid"])
        .groupby("match_uid", as_index=False)
        .tail(1)
        .sort_values(["match_date", "logged_at", "match_label"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    latest["probability_range"] = latest[["model_p1_prob", "xgb_p1_prob", "rf_p1_prob", "market_p1_prob"]].max(axis=1) - latest[
        ["model_p1_prob", "xgb_p1_prob", "rf_p1_prob", "market_p1_prob"]
    ].min(axis=1)
    latest["nn_vs_market_edge"] = latest["model_p1_prob"] - latest["market_p1_prob"]
    latest["xgb_vs_market_edge"] = latest["xgb_p1_prob"] - latest["market_p1_prob"]
    latest["rf_vs_market_edge"] = latest["rf_p1_prob"] - latest["market_p1_prob"]
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

    merged["started_at"] = merged["started_at"].combine_first(merged.get("odds_started_at"))
    merged["completed_at"] = merged["completed_at"].combine_first(merged.get("odds_completed_at"))
    merged["status"] = "derived_from_snapshots"
    merged["run_source"] = "derived"
    return merged.sort_values("started_at", ascending=False, ignore_index=True)
