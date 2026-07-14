from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

try:
    from features.performance_v1 import PERFORMANCE_FEATURES, PERFORMANCE_FEATURE_SET
    from logging_utils import ensure_csv_columns, normalize_name, stable_hash, utc_now_iso
    from models.inference import EXACT_141_FEATURES
except ModuleNotFoundError:  # pragma: no cover - package import path
    from ..features.performance_v1 import PERFORMANCE_FEATURES, PERFORMANCE_FEATURE_SET
    from ..logging_utils import ensure_csv_columns, normalize_name, stable_hash, utc_now_iso
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
DEFAULT_MEDIANS_PATH = DEFAULT_MODEL_DIR / "feature_medians.json"


@dataclass(frozen=True)
class ShadowModelSpec:
    family: str
    model_dir: Path
    model_version: str
    artifact_name: str
    feature_set: str = PERFORMANCE_FEATURE_SET
    feature_mode: str = "one_hot"
    medians_path: Path | None = DEFAULT_MEDIANS_PATH

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.artifact_name


def _perf_v1_spec(family: str, dirname: str, artifact: str) -> "ShadowModelSpec":
    """Build a performance_v1 one_hot shadow spec from its experiment dir name.

    All performance_v1 one_hot variants share the layout + the default medians,
    so they only differ by family/dir/artifact.
    """
    slug = dirname.replace("performance_v1__", "")
    return ShadowModelSpec(
        family=family,
        model_dir=REPO_ROOT / "results" / "professional_tennis" / "experiments"
        / "2026-04-25" / family / dirname,
        model_version=f"performance_v1_{slug}__2026-04-25",
        artifact_name=artifact,
    )


DEFAULT_SHADOW_MODEL_SPECS = [
    ShadowModelSpec(
        family="xgboost",
        model_dir=DEFAULT_MODEL_DIR,
        model_version=DEFAULT_MODEL_VERSION,
        artifact_name="model.json",
    ),
    ShadowModelSpec(
        family="catboost",
        model_dir=REPO_ROOT
        / "results"
        / "professional_tennis"
        / "experiments"
        / "2026-04-25"
        / "catboost"
        / "performance_v1__cat_depth6_screening__one_hot",
        model_version="performance_v1_cat_depth6_screening_one_hot__2026-04-25",
        artifact_name="model.cbm",
    ),
    ShadowModelSpec(
        family="lightgbm",
        model_dir=REPO_ROOT
        / "results"
        / "professional_tennis"
        / "experiments"
        / "2026-04-25"
        / "lightgbm"
        / "performance_v1__lgbm_leaves31_regularized__one_hot",
        model_version="performance_v1_lgbm_leaves31_regularized_one_hot__2026-04-25",
        artifact_name="model.txt",
    ),
    ShadowModelSpec(
        family="nn",
        model_dir=REPO_ROOT
        / "results"
        / "professional_tennis"
        / "experiments"
        / "2026-04-25"
        / "nn"
        / "performance_v1__nn_logits_128_64_32_lowdrop",
        model_version="performance_v1_nn_logits_128_64_32_lowdrop__2026-04-25",
        artifact_name="model.pth",
    ),
    # Additional performance_v1 one_hot variants, tracked live so the ledger
    # accumulates settled performance for every granular/recency tree + NN tweak
    # we have artifacts for — not just one of each family. native_cat variants
    # are intentionally excluded (native-cat live inference is not wired yet).
    _perf_v1_spec("xgboost", "performance_v1__xgb_depth4_slow_regularized", "model.json"),
    _perf_v1_spec("xgboost", "performance_v1__xgb_depth5_balanced_regularized", "model.json"),
    _perf_v1_spec("xgboost", "performance_v1__xgb_depth5_recency_hl_8y", "model.json"),
    _perf_v1_spec("xgboost", "performance_v1__xgb_depth6_medium", "model.json"),
    _perf_v1_spec("nn", "performance_v1__nn_logits_128_64_robust", "model.pth"),
    _perf_v1_spec("nn", "performance_v1__nn_logits_96_48_lowdrop", "model.pth"),
]

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
    "actual_winner",
    "score",
    "settled_at",
    "shadow_pick",
    "shadow_correct",
    "market_correct",
]


class _TennisLogitsNet:
    """Small lazy wrapper so torch is only imported when an NN side model loads."""

    @staticmethod
    def build(input_size: int, hidden_dims: list[int], dropouts: list[float]):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev = input_size
                for hidden, drop in zip(hidden_dims, dropouts):
                    layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(drop)])
                    prev = hidden
                layers.append(nn.Linear(prev, 1))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x).squeeze(1)

        return Net()


class SideModelShadowPredictor:
    def __init__(self, spec: ShadowModelSpec):
        self.spec = spec
        self.family = spec.family
        self.model_dir = Path(spec.model_dir)
        self.model_version = spec.model_version
        self.feature_set = spec.feature_set
        self.feature_mode = spec.feature_mode
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.medians: Dict[str, float] = {}
        self.is_loaded = False

    def _load_medians(self) -> Dict[str, float]:
        candidates = [
            self.spec.medians_path,
            self.model_dir / "feature_medians.json",
            DEFAULT_MEDIANS_PATH,
        ]
        for path in candidates:
            if path and Path(path).exists():
                return json.loads(Path(path).read_text())
        return {}

    def _load_summary(self) -> Dict:
        summary_path = self.model_dir / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        return {}

    def _load_xgboost(self):
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(self.spec.model_path))
        self.model = model
        self.feature_names = [str(name) for name in model.feature_names_in_]

    def _load_catboost(self):
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(self.spec.model_path))
        self.model = model
        self.feature_names = [str(name) for name in getattr(model, "feature_names_", [])]

    def _load_lightgbm(self):
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(self.spec.model_path))
        self.model = model
        self.feature_names = [str(name) for name in model.feature_name()]

    def _load_nn(self):
        import torch

        summary = self._load_summary()
        config = summary.get("config", {})
        scaler_path = self.model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"missing NN scaler: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.feature_names = list(EXACT_141_FEATURES) + list(PERFORMANCE_FEATURES)
        input_size = int(getattr(self.scaler, "n_features_in_", len(self.feature_names)))
        model = _TennisLogitsNet.build(
            input_size=input_size,
            hidden_dims=list(config.get("hidden_dims", [128, 64, 32])),
            dropouts=list(config.get("dropouts", [0.1, 0.1, 0.05])),
        )
        state = torch.load(
            str(self.spec.model_path), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def load_model(self) -> bool:
        try:
            if self.feature_mode != "one_hot":
                print(f"  ⚠️ skipping {self.model_version}: native-cat shadow inference is not wired for live rows")
                return False
            if not self.spec.model_path.exists():
                print(f"  ⚠️ performance_v1 shadow model missing: {self.spec.model_path}")
                return False

            loaders = {
                "xgboost": self._load_xgboost,
                "catboost": self._load_catboost,
                "lightgbm": self._load_lightgbm,
                "nn": self._load_nn,
            }
            loader = loaders.get(self.family)
            if loader is None:
                print(f"  ⚠️ unsupported performance_v1 shadow family: {self.family}")
                return False
            loader()
            if not self.feature_names:
                raise RuntimeError("model did not expose feature names")
            self.medians = self._load_medians()
            self.is_loaded = True
            print(f"✅ performance_v1 shadow loaded: {self.model_version} ({len(self.feature_names)} features)")
            return True
        except Exception as exc:
            print(f"  ⚠️ performance_v1 shadow model load failed for {self.model_version}: {exc}")
            return False

    def _feature_frame(self, features: Dict) -> pd.DataFrame:
        values = {}
        for feature in self.feature_names:
            raw = features.get(feature)
            if raw is None or pd.isna(raw):
                raw = self.medians.get(feature, 0.0)
            values[feature] = float(raw)
        return pd.DataFrame([values])[self.feature_names]

    def predict_match_probability(self, features: Dict) -> Dict:
        if not self.is_loaded and not self.load_model():
            return {"error": f"{self.model_version} shadow model not loaded"}
        try:
            X = self._feature_frame(features)
            if self.family == "lightgbm":
                prob = float(self.model.predict(X)[0])
            elif self.family == "nn":
                import torch

                X_scaled = self.scaler.transform(X.values)
                with torch.no_grad():
                    logits = self.model(torch.FloatTensor(X_scaled)).squeeze()
                    prob = float(torch.sigmoid(logits).item())
            else:
                prob = float(self.model.predict_proba(X)[0, 1])
            return {"shadow_p1_prob": prob, "shadow_p2_prob": 1.0 - prob}
        except Exception as exc:
            return {"error": f"{self.model_version} shadow prediction failed: {exc}"}


class PerformanceV1ShadowPredictor(SideModelShadowPredictor):
    def __init__(self, model_dir: Path | str | None = None, model_version: str = DEFAULT_MODEL_VERSION):
        spec = ShadowModelSpec(
            family="xgboost",
            model_dir=Path(model_dir) if model_dir else DEFAULT_MODEL_DIR,
            model_version=model_version,
            artifact_name="model.json",
        )
        super().__init__(spec)


class PerformanceV1ShadowEnsemble:
    def __init__(self, specs: Iterable[ShadowModelSpec] | None = None):
        self.predictors = [SideModelShadowPredictor(spec) for spec in (specs or DEFAULT_SHADOW_MODEL_SPECS)]
        self.loaded_predictors: list[SideModelShadowPredictor] = []
        self.is_loaded = False

    def load_model(self) -> bool:
        self.loaded_predictors = []
        for predictor in self.predictors:
            if predictor.load_model():
                self.loaded_predictors.append(predictor)
        self.is_loaded = bool(self.loaded_predictors)
        return self.is_loaded

    def predict_match_probabilities(self, features: Dict):
        if not self.is_loaded and not self.load_model():
            return []
        return [
            (predictor, predictor.predict_match_probability(features))
            for predictor in self.loaded_predictors
        ]


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
    model_family: str = "xgboost",
    feature_set: str = PERFORMANCE_FEATURE_SET,
    n_features: int | None = None,
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
        "model_family": model_family,
        "model_version": model_version,
        "feature_set": feature_set,
        "n_features": n_features or len(EXACT_141_FEATURES) + len(PERFORMANCE_FEATURES),
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
        "actual_winner": "",
        "score": "",
        "settled_at": "",
        "shadow_pick": "",
        "shadow_correct": "",
        "market_correct": "",
    }


def _pick_from_prob(p1: str, p2: str, p1_prob) -> str:
    prob = _float_or_none(p1_prob)
    if prob is None:
        return ""
    return p1 if prob >= 0.5 else p2


def _correct_from_actual(pick: str, actual_winner, p1: str, p2: str):
    if not pick or actual_winner is None or pd.isna(actual_winner):
        return ""
    actual_text = str(actual_winner).strip()
    if actual_text in {"1", "1.0"}:
        return int(normalize_name(pick) == normalize_name(p1))
    if actual_text in {"2", "2.0"}:
        return int(normalize_name(pick) == normalize_name(p2))
    try:
        float(actual_text)
        return ""
    except ValueError:
        pass
    return int(normalize_name(pick) == normalize_name(actual_text))


def _prediction_snapshot_key(row: pd.Series) -> str:
    snapshot_id = _clean_existing(row.get("latest_feature_snapshot_id")) or _clean_existing(row.get("feature_snapshot_id"))
    return f"{row.get('match_uid', '')}::{snapshot_id}"


def sync_shadow_settlements(shadow_path: Path, prediction_log_path: Path) -> int:
    """
    Copy operational settlement outcomes onto side-model shadow rows.

    This does not recompute inference. It only scores already-logged shadow
    probabilities when the corresponding operational prediction has settled.
    """
    shadow_path = Path(shadow_path)
    prediction_log_path = Path(prediction_log_path)
    if not shadow_path.exists() or not prediction_log_path.exists():
        return 0

    shadow_df = ensure_csv_columns(shadow_path, SHADOW_COLUMNS)
    if shadow_df.empty:
        return 0

    try:
        pred_df = pd.read_csv(prediction_log_path)
    except Exception:
        return 0
    if pred_df.empty or "actual_winner" not in pred_df.columns or "match_uid" not in pred_df.columns:
        return 0

    settled = pred_df[pred_df["actual_winner"].notna()].copy()
    if "record_status" in settled.columns:
        settled = settled[
            ~settled["record_status"].fillna("").astype(str).str.lower().isin(
                {"identity_conflict", "superseded_identity"}
            )
        ].copy()
    if settled.empty:
        return 0

    settled["_snapshot_key"] = settled.apply(_prediction_snapshot_key, axis=1)
    settled_by_snapshot = settled.drop_duplicates("_snapshot_key", keep="last").set_index("_snapshot_key")
    settled_by_match = settled.drop_duplicates("match_uid", keep="last").set_index("match_uid")
    # UID drift is accepted only through the explicit canonical alias written
    # by prediction_logger. Plain players/date matching can cross rounds or
    # events and would score the wrong immutable shadow observation.
    alias_to_canonical: dict[str, str] = {}
    for _, prediction_row in pred_df.iterrows():
        status = _clean_existing(prediction_row.get("identity_status")).lower()
        uid = _clean_existing(prediction_row.get("match_uid"))
        related = [
            value.strip()
            for value in _clean_existing(
                prediction_row.get("identity_related_match_uid")
            ).split("|")
            if value.strip()
        ]
        if status == "canonical_alias" and uid:
            for old_uid in related:
                alias_to_canonical[old_uid] = uid
        elif status == "superseded_alias" and uid and len(related) == 1:
            alias_to_canonical[uid] = related[0]

    updated = 0
    for idx, shadow_row in shadow_df.iterrows():
        if _clean_existing(shadow_row.get("actual_winner")):
            continue
        key = f"{shadow_row.get('match_uid', '')}::{shadow_row.get('feature_snapshot_id', '')}"
        if key in settled_by_snapshot.index:
            pred_row = settled_by_snapshot.loc[key]
        elif shadow_row.get("match_uid", "") in settled_by_match.index:
            pred_row = settled_by_match.loc[shadow_row.get("match_uid", "")]
        elif (
            alias_to_canonical.get(_clean_existing(shadow_row.get("match_uid")), "")
            in settled_by_match.index
        ):
            pred_row = settled_by_match.loc[
                alias_to_canonical[_clean_existing(shadow_row.get("match_uid"))]
            ]
        else:
            continue

        p1 = str(shadow_row.get("p1", "") or pred_row.get("p1", ""))
        p2 = str(shadow_row.get("p2", "") or pred_row.get("p2", ""))
        actual = pred_row.get("actual_winner")
        shadow_pick = _pick_from_prob(p1, p2, shadow_row.get("shadow_p1_prob"))
        market_pick = _pick_from_prob(p1, p2, shadow_row.get("market_p1_prob"))
        shadow_df.at[idx, "actual_winner"] = actual
        shadow_df.at[idx, "score"] = pred_row.get("score", "")
        shadow_df.at[idx, "settled_at"] = pred_row.get("settled_at", "")
        shadow_df.at[idx, "shadow_pick"] = shadow_pick
        shadow_df.at[idx, "shadow_correct"] = _correct_from_actual(shadow_pick, actual, p1, p2)
        shadow_df.at[idx, "market_correct"] = _correct_from_actual(market_pick, actual, p1, p2)
        updated += 1

    if updated:
        shadow_df.to_csv(shadow_path, index=False)
    return updated


def _clean_existing(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text
