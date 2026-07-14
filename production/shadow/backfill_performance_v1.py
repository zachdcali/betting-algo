#!/usr/bin/env python3
"""Backfill performance_v1 shadow predictions for exact live feature snapshots.

This is intentionally separate from forward shadow logging. It reuses the saved
141-feature vector, scrapes current TA histories only to derive the new
performance_v1 score/stat features as of the logged match date, and writes an
experiment-only CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

PRODUCTION_DIR = Path(__file__).resolve().parents[1]
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from features.performance_v1 import PERFORMANCE_FEATURES, build_match_performance_features  # noqa: E402
from features.ta_feature_calculator import TAFeatureCalculator  # noqa: E402
from feature_lineage import read_feature_csv  # noqa: E402
from logging_utils import normalize_name  # noqa: E402
from models.inference import EXACT_141_FEATURES  # noqa: E402
from prediction_logger import upgrade_prediction_log  # noqa: E402
from shadow.performance_v1_shadow import (  # noqa: E402
    DEFAULT_MODEL_VERSION,
    PerformanceV1ShadowPredictor,
    build_shadow_uid,
    log_shadow_predictions,
    shadow_row_from_prediction,
)

DEFAULT_OUTPUT = PRODUCTION_DIR / "logs" / "performance_v1_shadow_backfill.csv"
DEFAULT_LOG = PRODUCTION_DIR / "prediction_log.csv"
DEFAULT_FEATURE_DIR = PRODUCTION_DIR / "logs"

BACKFILL_SOURCE = "exact_feature_snapshot_plus_ta_history_rescrape"
BACKFILL_QUALITY = "snapshot_v2_performance_v1_backfill"


def load_feature_snapshots(features_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(features_dir.glob("features_*.csv")):
        try:
            df = read_feature_csv(path)
        except Exception as exc:
            print(f"skip feature log {path}: {exc}")
            continue
        if "feature_snapshot_id" not in df.columns:
            continue
        df["_source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_features = pd.concat(frames, ignore_index=True)
    all_features["feature_snapshot_id"] = all_features["feature_snapshot_id"].fillna("").astype(str)
    all_features = all_features[all_features["feature_snapshot_id"].str.len().gt(0)].copy()
    return all_features.drop_duplicates(subset=["feature_snapshot_id"], keep="last")


def existing_shadow_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["shadow_prediction_uid"])
    except Exception:
        return set()
    return set(df["shadow_prediction_uid"].dropna().astype(str))


def select_candidates(
    prediction_log: pd.DataFrame,
    existing_uids: set[str] | None = None,
    limit: int | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> pd.DataFrame:
    existing_uids = existing_uids or set()
    df = prediction_log.copy()
    snapshot_id = df.get("latest_feature_snapshot_id", pd.Series("", index=df.index)).fillna("").astype(str)
    opening_snapshot = df.get("feature_snapshot_id", pd.Series("", index=df.index)).fillna("").astype(str)
    df["_backfill_feature_snapshot_id"] = snapshot_id.where(snapshot_id.str.len().gt(0), opening_snapshot)

    mask = (
        df.get("logging_quality", "").eq("snapshot_v2")
        & df.get("rescore_quality", "").eq("exact_feature_snapshot")
        & df["_backfill_feature_snapshot_id"].str.len().gt(0)
        & df.get("actual_winner", pd.Series(pd.NA, index=df.index)).notna()
        & df.get("features_complete", pd.Series(True, index=df.index)).fillna(True).astype(bool)
    )
    candidates = df[mask].copy()
    if candidates.empty:
        return candidates

    candidates["_shadow_prediction_uid"] = candidates.apply(
        lambda row: build_shadow_uid(
            row.get("match_uid", ""),
            model_version,
            row.get("_backfill_feature_snapshot_id", ""),
        ),
        axis=1,
    )
    if existing_uids:
        candidates = candidates[~candidates["_shadow_prediction_uid"].isin(existing_uids)].copy()

    sort_cols = [col for col in ["match_date", "logged_at", "p1", "p2"] if col in candidates.columns]
    if sort_cols:
        candidates = candidates.sort_values(sort_cols)
    if limit:
        candidates = candidates.head(limit)
    return candidates


def shadow_pick(p1: str, p2: str, p1_prob) -> str:
    try:
        return p1 if float(p1_prob) >= 0.5 else p2
    except (TypeError, ValueError):
        return ""


def is_correct_pick(pick: str, actual_winner, p1: str = "", p2: str = "") -> bool | None:
    if not pick or actual_winner is None or pd.isna(actual_winner):
        return None
    actual_text = str(actual_winner).strip()
    if actual_text in {"1", "1.0"}:
        return normalize_name(pick) == normalize_name(p1)
    if actual_text in {"2", "2.0"}:
        return normalize_name(pick) == normalize_name(p2)
    try:
        float(actual_text)
        return None
    except ValueError:
        pass
    return normalize_name(pick) == normalize_name(actual_winner)


def _float_or_none(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _series_for_shadow(pred_row: pd.Series, feature_row: pd.Series, perf_features: Dict, status: str, error: str) -> pd.Series:
    values = feature_row.to_dict()
    values.update(perf_features)
    values.update(
        {
            "run_id": pred_row.get("latest_run_id") or pred_row.get("run_id", ""),
            "match_uid": pred_row.get("match_uid", ""),
            "feature_snapshot_id": pred_row.get("_backfill_feature_snapshot_id", ""),
            "player1_raw": pred_row.get("p1", ""),
            "player2_raw": pred_row.get("p2", ""),
            "event": pred_row.get("tournament", ""),
            "match_time": pred_row.get("match_start_time", ""),
            "meta_match_date": feature_row.get("meta_match_date") or pred_row.get("match_date", ""),
            "meta_surface_input": feature_row.get("meta_surface_input") or pred_row.get("surface", ""),
            "meta_level_input": feature_row.get("meta_level_input") or pred_row.get("level", ""),
            "meta_round_input": feature_row.get("meta_round_input") or pred_row.get("round", ""),
            "performance_v1_features_available": status == "success",
            "performance_v1_status": status,
            "performance_v1_error": error,
        }
    )
    return pd.Series(values)


def _odds_row_from_prediction(pred_row: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "player1_implied_prob": pred_row.get("market_p1_prob"),
            "player2_implied_prob": pred_row.get("market_p2_prob"),
            "player1_odds_decimal": pred_row.get("p1_odds_decimal"),
            "player2_odds_decimal": pred_row.get("p2_odds_decimal"),
            "event": pred_row.get("tournament", ""),
            "match_time": pred_row.get("match_start_time", ""),
            "scrape_time_utc": pred_row.get("odds_scraped_at", ""),
        }
    )


def add_backfill_metadata(row: Dict, pred_row: pd.Series, feature_row: pd.Series, result: Dict) -> Dict:
    p1 = pred_row.get("p1", "")
    p2 = pred_row.get("p2", "")
    pick = shadow_pick(p1, p2, result.get("shadow_p1_prob"))
    correct = is_correct_pick(pick, pred_row.get("actual_winner"), p1=p1, p2=p2)
    market_p1 = _float_or_none(pred_row.get("market_p1_prob"))
    shadow_p1 = result.get("shadow_p1_prob")

    row.update(
        {
            "backfill_source": BACKFILL_SOURCE,
            "backfill_quality": BACKFILL_QUALITY,
            "source_feature_file": feature_row.get("_source_file", ""),
            "prediction_logged_at": pred_row.get("logged_at", ""),
            "settled_at": pred_row.get("settled_at", ""),
            "actual_winner": pred_row.get("actual_winner", ""),
            "score": pred_row.get("score", ""),
            "shadow_pick": pick,
            "shadow_correct": correct,
            "shadow_edge_p1": (
                round(float(shadow_p1) - market_p1, 6)
                if shadow_p1 is not None and market_p1 is not None
                else None
            ),
            "nn_p1_prob": pred_row.get("model_p1_prob"),
            "xgb_p1_prob": pred_row.get("xgb_p1_prob"),
            "rf_p1_prob": pred_row.get("rf_p1_prob"),
            "nn_correct": pred_row.get("model_correct"),
            "xgb_correct": pred_row.get("xgb_correct"),
            "rf_correct": pred_row.get("rf_correct"),
            "market_correct": pred_row.get("market_correct"),
        }
    )
    return row


def run_backfill(args: argparse.Namespace) -> dict:
    prediction_log = upgrade_prediction_log(Path(args.prediction_log), write=False)
    features = load_feature_snapshots(Path(args.features_dir))
    if features.empty:
        raise RuntimeError(f"No feature snapshots found under {args.features_dir}")
    features_by_id = features.set_index("feature_snapshot_id", drop=False)

    output = Path(args.output)
    candidates = select_candidates(
        prediction_log,
        existing_uids=existing_shadow_uids(output) if not args.dry_run else set(),
        limit=args.limit,
    )
    print(f"candidate rows: {len(candidates)}")
    if args.dry_run or candidates.empty:
        return {"candidates": len(candidates), "attempted": 0, "success": 0, "error": 0, "logged": 0}

    predictor = PerformanceV1ShadowPredictor()
    if not predictor.load_model():
        raise RuntimeError("performance_v1 shadow model/medians are not available")

    feature_engine = TAFeatureCalculator()
    session_cache: dict = {}
    rows = []
    stats = {"candidates": len(candidates), "attempted": 0, "success": 0, "error": 0, "logged": 0}

    total = len(candidates)
    for row_number, (_, pred_row) in enumerate(candidates.iterrows(), start=1):
        stats["attempted"] += 1
        if args.progress_every and (row_number == 1 or row_number % args.progress_every == 0 or row_number == total):
            print(
                f"processing {row_number}/{total}: "
                f"{pred_row.get('p1', '')} vs {pred_row.get('p2', '')}"
            )
        snapshot_id = pred_row.get("_backfill_feature_snapshot_id", "")
        feature_row = features_by_id.loc[snapshot_id]
        if isinstance(feature_row, pd.DataFrame):
            feature_row = feature_row.iloc[-1]

        perf_features = {name: pd.NA for name in PERFORMANCE_FEATURES}
        status = "success"
        error = ""
        result: Dict = {}
        try:
            missing = [name for name in EXACT_141_FEATURES if name not in feature_row.index]
            if missing:
                raise RuntimeError(f"snapshot_missing_base_features:{','.join(missing[:5])}")

            p1 = pred_row.get("p1", "")
            p2 = pred_row.get("p2", "")
            slug1 = feature_engine.find_slug(p1)
            slug2 = feature_engine.find_slug(p2)
            if not slug1 or not slug2:
                raise RuntimeError(f"missing_slug:{p1 if not slug1 else p2}")

            match_date = feature_row.get("meta_match_date") or pred_row.get("match_date", "")
            matches1 = feature_engine.scraper.get_player_matches(
                slug1,
                years=[],
                force_refresh=False,
                persist=False,
                session_cache=session_cache,
            )
            matches2 = feature_engine.scraper.get_player_matches(
                slug2,
                years=[],
                force_refresh=False,
                persist=False,
                session_cache=session_cache,
            )
            perf_features = build_match_performance_features(matches1, matches2, match_date)
            shadow_features = _series_for_shadow(pred_row, feature_row, perf_features, status, error)
            result = predictor.predict_match_probability(shadow_features.to_dict())
            if "error" in result:
                raise RuntimeError(result["error"])
            stats["success"] += 1
        except Exception as exc:
            status = "error"
            error = str(exc)
            result = {"error": error}
            shadow_features = _series_for_shadow(pred_row, feature_row, perf_features, status, error)
            stats["error"] += 1

        shadow_row = shadow_row_from_prediction(
            shadow_features,
            _odds_row_from_prediction(pred_row),
            result,
            model_version=predictor.model_version,
        )
        rows.append(add_backfill_metadata(shadow_row, pred_row, feature_row, result))

    stats["logged"] = log_shadow_predictions(output, rows)
    print(f"logged rows: {stats['logged']} -> {output}")
    print_summary(pd.DataFrame(rows))
    return stats


def print_summary(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    success = rows[rows["shadow_status"] == "success"].copy()
    print(f"success rows: {len(success)} / {len(rows)}")
    if success.empty or "shadow_correct" not in success:
        return
    valid = success[success["shadow_correct"].notna()].copy()
    if valid.empty:
        return
    correct = valid["shadow_correct"].astype(bool)
    print(f"shadow accuracy: {correct.sum()}/{len(valid)} = {correct.mean():.3%}")
    for label, col in [("NN", "nn_correct"), ("XGB", "xgb_correct"), ("RF", "rf_correct"), ("Market", "market_correct")]:
        vals = pd.to_numeric(valid.get(col), errors="coerce").dropna()
        if not vals.empty:
            print(f"{label} accuracy on same rows: {int(vals.sum())}/{len(vals)} = {vals.mean():.3%}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-log", default=str(DEFAULT_LOG))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Count candidates without scraping or writing.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    stats = run_backfill(args)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
