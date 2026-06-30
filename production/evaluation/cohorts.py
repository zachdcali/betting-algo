"""Load logs, derive authoritative ground truth, assemble a long scored frame.

This is the only module that knows the on-disk storage format. A future SQLite
migration swaps the ``load_*`` helpers and leaves the rest of the package intact.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Model name -> the prediction_log column holding its P(player1 wins).
MODEL_PROB_COLS = {
    "nn": "model_p1_prob",
    "xgb": "xgb_p1_prob",
    "rf": "rf_p1_prob",
    "market": "market_p1_prob",
}
SHADOW_FAMILIES = ["xgboost", "catboost", "lightgbm", "nn"]


def load_prediction_log(prod_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(prod_dir, "prediction_log.csv"), low_memory=False)


def load_shadow_log(prod_dir: str) -> pd.DataFrame | None:
    path = os.path.join(prod_dir, "logs", "performance_v1_shadow_predictions.csv")
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None


def build_ground_truth(pred_log: pd.DataFrame) -> pd.Series:
    """Authoritative match_uid -> y1 (1 if player1 won). Conflict-free, deduped."""
    s = pred_log[pred_log["actual_winner"].notna()].copy()
    s["y1"] = (s["actual_winner"].astype(float) == 1).astype(int)
    nunique = s.groupby("match_uid")["y1"].nunique()
    consistent = nunique[nunique == 1].index
    # keep="last" = the most recently logged row, matching the operational
    # "latest prediction wins" semantics and the DB's INSERT OR REPLACE.
    s = s[s["match_uid"].isin(consistent)].drop_duplicates("match_uid", keep="last")
    return s.set_index("match_uid")["y1"]


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Robustly coerce a truthy column to bool regardless of source representation.

    CSVs give Python bools; a SQLite round-trip can yield "1"/"0" or "True"/"False"
    strings or 1/0 ints. Normalize them all so cohort tiers behave identically
    whether the data came from a CSV or the DB.
    """
    truthy = {"true", "1", "1.0", "t", "yes"}
    return series.map(lambda v: str(v).strip().lower() in truthy).astype(bool)


def _tier_flags(pred_log: pd.DataFrame) -> pd.DataFrame:
    f = pred_log.copy()
    if "features_complete" in f.columns:
        f["is_complete"] = _coerce_bool(f["features_complete"])
    else:
        f["is_complete"] = False
    lq = f["logging_quality"] if "logging_quality" in f.columns else pd.Series(index=f.index, dtype=object)
    rq = f["rescore_quality"] if "rescore_quality" in f.columns else pd.Series(index=f.index, dtype=object)
    f["is_gold"] = f["is_complete"] & (lq == "snapshot_v2") & (rq == "exact_feature_snapshot")
    return f[["match_uid", "is_complete", "is_gold"]].drop_duplicates("match_uid")


def build_scored_frame(pred_log: pd.DataFrame, shadow_log: pd.DataFrame | None) -> pd.DataFrame:
    """Long format: one row per (match_uid, model), settled rows with a non-null prob."""
    gt = build_ground_truth(pred_log)
    tiers = _tier_flags(pred_log).set_index("match_uid")
    settled = (
        pred_log[pred_log["match_uid"].isin(gt.index)]
        .drop_duplicates("match_uid", keep="last")
        .set_index("match_uid")
    )
    rows = []
    for model, col in MODEL_PROB_COLS.items():
        if col not in settled.columns:
            continue
        sub = settled[settled[col].notna()]
        for uid, r in sub.iterrows():
            rows.append({
                "match_uid": uid, "model": model, "family": model,
                "p1_prob": float(r[col]),
                "p1_odds_decimal": r.get("p1_odds_decimal"),
                "p2_odds_decimal": r.get("p2_odds_decimal"),
                "y1": int(gt.loc[uid]),
                "is_gold": bool(tiers.loc[uid, "is_gold"]),
                "is_complete": bool(tiers.loc[uid, "is_complete"]),
            })

    if shadow_log is not None and "model_family" in shadow_log.columns:
        sh = shadow_log[shadow_log["match_uid"].isin(gt.index)]
        for _, r in sh.iterrows():
            if pd.isna(r.get("shadow_p1_prob")):
                continue
            uid = r["match_uid"]
            in_tiers = uid in tiers.index
            rows.append({
                "match_uid": uid,
                "model": f"shadow_{r['model_family']}",
                "family": r["model_family"],
                "p1_prob": float(r["shadow_p1_prob"]),
                "p1_odds_decimal": r.get("p1_odds_decimal"),
                "p2_odds_decimal": r.get("p2_odds_decimal"),
                "y1": int(gt.loc[uid]),
                "is_gold": bool(tiers.loc[uid, "is_gold"]) if in_tiers else False,
                "is_complete": bool(tiers.loc[uid, "is_complete"]) if in_tiers else False,
            })

    return pd.DataFrame(rows)


def intersection_uids(scored: pd.DataFrame, models: list[str], tier_col: str) -> set:
    """match_uids that appear for *every* listed model within the given tier."""
    sub = scored[scored[tier_col] & scored["model"].isin(models)]
    counts = sub.groupby("match_uid")["model"].nunique()
    needed = len(set(models))
    return set(counts[counts == needed].index)
