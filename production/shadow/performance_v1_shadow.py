from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

try:
    from features.performance_v1 import PERFORMANCE_FEATURES, PERFORMANCE_FEATURE_SET
    from logging_utils import ensure_csv_columns, stable_hash, utc_now_iso
    from models.inference import EXACT_141_FEATURES
except ModuleNotFoundError:  # pragma: no cover - package import path
    from ..features.performance_v1 import PERFORMANCE_FEATURES, PERFORMANCE_FEATURE_SET
    from ..logging_utils import ensure_csv_columns, stable_hash, utc_now_iso
    from ..models.inference import EXACT_141_FEATURES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    REPO_ROOT
    / "results"
    / "professional_tennis"
    / "experiments"
    / "2026-04-25"
    / "xgboost"
    / "performance_v1__xgb_depth5_recency_hl_12y"
)
DEFAULT_MODEL_VERSION = "performance_v1_xgb_depth5_recency_hl_12y__2026-04-25"

SHADOW_COLUMNS = [
    "shadow_prediction_uid",
    "logged_at",
    "run_id",
    "match_uid",
    "feature_snapshot_id",
    "match_date",
    "match_start_time",
    "odds_scraped_at",
    "p1",
    "p2",
    "tournament",
    "surface",
    "level",
    "round",
    "model_family",
    "model_version",
    "feature_set",
    "n_features",
    "shadow_p1_prob",
    "shadow_p2_prob",
    "market_p1_prob",
    "market_p2_prob",
    "p1_odds_decimal",
    "p2_odds_decimal",
    "p1_score_matches_last10",
    "p2_score_matches_last10",
    "p1_stat_matches_last10",
    "p2_stat_matches_last10",
    "performance_features_available",
    "shadow_status",
    "shadow_error",
]


class PerformanceV1ShadowPredictor:
    def __init__(self, model_dir: Path | str | None = None, model_version: str = DEFAULT_MODEL_VERSION):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.model_version = model_version
        self.model = None
        self.feature_names = []
        self.medians: Dict[str, float] = {}
        self.is_loaded = False

    def load_model(self) -> bool:
        try:
            import xgboost as xgb

            model_path = self.model_dir / "model.json"
            medians_path = self.model_dir / "feature_medians.json"
            if not model_path.exists():
                print(f"  ⚠️ performance_v1 shadow model missing: {model_path}")
                return False
            if not medians_path.exists():
                print(f"  ⚠️ performance_v1 shadow medians missing: {medians_path}")
                return False

            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path))
            self.feature_names = list(self.model.feature_names_in_)
            self.medians = json.loads(medians_path.read_text())
            self.is_loaded = True
            print(f"✅ performance_v1 shadow model loaded ({len(self.feature_names)} features)")
            return True
        except Exception as exc:
            print(f"  ⚠️ performance_v1 shadow model load failed: {exc}")
            return False

    def predict_match_probability(self, features: Dict) -> Dict:
        if not self.is_loaded and not self.load_model():
            return {"error": "performance_v1 shadow model not loaded"}
        try:
            values = {}
            for feature in self.feature_names:
                raw = features.get(feature)
                if pd.isna(raw):
                    raw = self.medians.get(feature, 0.0)
                values[feature] = float(raw)
            X = pd.DataFrame([values])[self.feature_names]
            prob = float(self.model.predict_proba(X)[0, 1])
            return {"shadow_p1_prob": prob, "shadow_p2_prob": 1.0 - prob}
        except Exception as exc:
            return {"error": f"performance_v1 shadow prediction failed: {exc}"}


def build_shadow_uid(match_uid: str, model_version: str, feature_snapshot_id: str) -> str:
    return "shadow_" + stable_hash(match_uid, model_version, feature_snapshot_id, length=20)


def log_shadow_predictions(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = ensure_csv_columns(path, SHADOW_COLUMNS)
    existing = set(df["shadow_prediction_uid"].dropna().astype(str)) if "shadow_prediction_uid" in df else set()
    new_rows = []
    for row in rows:
        uid = str(row.get("shadow_prediction_uid", ""))
        if uid and uid not in existing:
            new_rows.append(row)
            existing.add(uid)
    if not new_rows:
        return 0
    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(path, index=False)
    return len(new_rows)


def _float_or_none(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def shadow_row_from_prediction(
    pred_row: pd.Series,
    odds_row: pd.Series | None,
    result: Dict,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> Dict:
    p1 = pred_row.get("player1_normalized") or pred_row.get("player1_raw", "")
    p2 = pred_row.get("player2_normalized") or pred_row.get("player2_raw", "")
    logged_at = utc_now_iso()
    match_uid = pred_row.get("match_uid", "")
    feature_snapshot_id = pred_row.get("feature_snapshot_id", "")
    uid = build_shadow_uid(match_uid, model_version, feature_snapshot_id)

    if odds_row is not None:
        market_p1 = _float_or_none(odds_row.get("player1_implied_prob"))
        market_p2 = _float_or_none(odds_row.get("player2_implied_prob"))
        total = market_p1 + market_p2 if market_p1 is not None and market_p2 is not None else None
        if total and total > 0:
            market_p1 = market_p1 / total
            market_p2 = market_p2 / total
        p1_odds = odds_row.get("player1_odds_decimal")
        p2_odds = odds_row.get("player2_odds_decimal")
        odds_scraped_at = odds_row.get("scrape_time_utc", "") or odds_row.get("timestamp", "")
        tournament = odds_row.get("tourney_name", "") or odds_row.get("event", "")
        match_start_time = odds_row.get("match_time", "")
    else:
        market_p1 = market_p2 = p1_odds = p2_odds = None
        odds_scraped_at = ""
        tournament = pred_row.get("event", "")
        match_start_time = pred_row.get("match_time", "")

    status = "success" if "error" not in result else "error"
    return {
        "shadow_prediction_uid": uid,
        "logged_at": logged_at,
        "run_id": pred_row.get("run_id", ""),
        "match_uid": match_uid,
        "feature_snapshot_id": feature_snapshot_id,
        "match_date": pred_row.get("meta_match_date", ""),
        "match_start_time": match_start_time,
        "odds_scraped_at": odds_scraped_at,
        "p1": p1,
        "p2": p2,
        "tournament": tournament,
        "surface": pred_row.get("meta_surface_input", ""),
        "level": pred_row.get("meta_level_input", ""),
        "round": pred_row.get("meta_round_input", ""),
        "model_family": "xgboost",
        "model_version": model_version,
        "feature_set": PERFORMANCE_FEATURE_SET,
        "n_features": len(EXACT_141_FEATURES) + len(PERFORMANCE_FEATURES),
        "shadow_p1_prob": round(result.get("shadow_p1_prob"), 6) if "shadow_p1_prob" in result else None,
        "shadow_p2_prob": round(result.get("shadow_p2_prob"), 6) if "shadow_p2_prob" in result else None,
        "market_p1_prob": round(market_p1, 6) if market_p1 is not None and pd.notna(market_p1) else None,
        "market_p2_prob": round(market_p2, 6) if market_p2 is not None and pd.notna(market_p2) else None,
        "p1_odds_decimal": p1_odds,
        "p2_odds_decimal": p2_odds,
        "p1_score_matches_last10": pred_row.get("P1_Score_Matches_Last10"),
        "p2_score_matches_last10": pred_row.get("P2_Score_Matches_Last10"),
        "p1_stat_matches_last10": pred_row.get("P1_Stat_Matches_Last10"),
        "p2_stat_matches_last10": pred_row.get("P2_Stat_Matches_Last10"),
        "performance_features_available": bool(pred_row.get("performance_v1_features_available", False)),
        "shadow_status": status,
        "shadow_error": result.get("error", ""),
    }
