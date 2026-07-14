"""Replay promoted models on already-captured, same-schema feature vectors.

This module is intentionally evaluation-only.  It does not train, calibrate,
tune, settle, reconstruct outcomes, or update any operational log.  Historical
evidence is selected by :mod:`evaluation.replay_manifest`; this module then
reloads the selected source rows, re-verifies their ordered ``base_141`` hashes,
and applies the registry's currently promoted NN/XGBoost/Random Forest
artifacts.

Without ``--out-dir`` the CLI is read-only and prints a summary.  Supplying an
output parent creates a new timestamp-versioned directory; existing evidence is
never replaced.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import pickle
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from evaluation import metrics, replay_manifest
from feature_vector_log import feature_fingerprint, feature_validation_issues
from models.inference import EXACT_141_FEATURES
from models.registry_utils import (
    REGISTRY_PATH,
    get_current_version,
    get_model_entry,
    load_registry,
    resolve_artifact_path,
)
from versioning import (
    DATASET_MANIFEST_VERSION,
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
)


MODEL_REPLAY_VERSION = DATASET_MANIFEST_VERSION
REPLAYABLE_TIERS = ("GOLD_REPLAY", "EXACT_INCOMPLETE", "LEGACY_MATCHED")
OUTPUT_REPLAY_CSV = "model_replay.csv"
OUTPUT_METRICS_CSV = "model_replay_metrics.csv"
OUTPUT_MANIFEST_JSON = "manifest.json"

REPLAY_COLUMNS = [
    "match_uid",
    "replay_tier",
    "match_date",
    "tournament",
    "p1",
    "p2",
    "feature_snapshot_id",
    "feature_schema_id",
    "feature_schema_sha256",
    "feature_vector_sha256",
    "feature_source_file",
    "feature_source_row",
    "features_complete",
    "prediction_observed_at",
    "match_start_at_utc",
    "reason_codes",
    "actual_winner",
    "outcome_status",
    "scorable",
    "y1",
    "p1_odds_decimal",
    "p2_odds_decimal",
    "model_family",
    "model_version",
    "model_id",
    "model_sha256",
    "training_feature_semantics_id",
    "live_feature_semantics_id",
    "probability_source",
    "replay_p1_prob",
    "replay_p2_prob",
]

METRIC_COLUMNS = [
    "model_family",
    "model_version",
    "model_id",
    "replay_tier",
    "n_replayed",
    "n",
    "accuracy",
    "auc",
    "log_loss",
    "brier",
    "ece",
    "cal_slope",
    "cal_intercept",
]


@dataclass
class ReplayPredictor:
    """A promoted artifact adapter, injectable in tests without model files."""

    family: str
    version: str
    model_name: str
    model_sha256: str
    feature_schema_id: str
    feature_schema_sha256: str
    training_feature_semantics_id: str
    live_feature_semantics_id: str
    probability_source: str
    predict_fn: Callable[[pd.DataFrame], np.ndarray]
    artifact_files: list[dict[str, str]]

    @property
    def model_id(self) -> str:
        normalized = self.version if self.version.startswith("v") else f"v{self.version}"
        return f"{self.family}@{normalized}"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.predict_fn(frame), dtype=float).reshape(-1)
        if len(probabilities) != len(frame):
            raise RuntimeError(
                f"{self.model_id} returned {len(probabilities)} probabilities "
                f"for {len(frame)} vectors"
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"{self.model_id} returned non-finite probabilities")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise RuntimeError(f"{self.model_id} returned probabilities outside [0, 1]")
        return probabilities

    def metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "version": self.version,
            "model_id": self.model_id,
            "name": self.model_name,
            "model_sha256": self.model_sha256,
            "feature_schema_id": self.feature_schema_id,
            "feature_schema_sha256": self.feature_schema_sha256,
            "training_feature_semantics_id": self.training_feature_semantics_id,
            "live_feature_semantics_id": self.live_feature_semantics_id,
            "probability_source": self.probability_source,
            "artifact_files": self.artifact_files,
        }


@dataclass
class ModelReplay:
    frame: pd.DataFrame
    metric_frame: pd.DataFrame
    evidence: replay_manifest.ReplayManifest
    predictors: list[ReplayPredictor]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truthy(value: Any) -> bool:
    return _text(value).lower() in {"true", "1", "1.0", "t", "yes"}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_file(path: Path, role: str, expected_hash: str) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(f"promoted {role} artifact is missing: {path}")
    expected = _text(expected_hash).lower()
    if not expected:
        raise RuntimeError(f"promoted {role} artifact has no pinned SHA-256: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"promoted {role} artifact checksum mismatch: "
            f"expected={expected} actual={actual} path={path}"
        )
    return {"role": role, "path": str(path.resolve()), "sha256": actual}


def _entry_contract(family: str, version: str, entry: dict[str, Any]) -> None:
    if int(entry.get("features") or 0) != len(EXACT_141_FEATURES):
        raise RuntimeError(f"{family}:{version} does not declare 141 features")
    if _text(entry.get("feature_schema_id")) != FEATURE_SCHEMA_ID:
        raise RuntimeError(
            f"{family}:{version} schema is {entry.get('feature_schema_id')!r}, "
            f"expected {FEATURE_SCHEMA_ID}"
        )
    if _text(entry.get("feature_schema_sha256")).lower() != FEATURE_SCHEMA_SHA256:
        raise RuntimeError(f"{family}:{version} ordered schema hash does not match base_141")
    for field in ("training_feature_semantics_id", "live_feature_semantics_id"):
        if not _text(entry.get(field)):
            raise RuntimeError(f"{family}:{version} is missing {field}")


def _predictor_metadata(
    family: str,
    version: str,
    entry: dict[str, Any],
    artifact_files: list[dict[str, str]],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    *,
    probability_source: str = "raw",
) -> ReplayPredictor:
    return ReplayPredictor(
        family=family,
        version=version,
        model_name=_text(entry.get("name")),
        model_sha256=_text(entry.get("model_sha256")).lower(),
        feature_schema_id=_text(entry.get("feature_schema_id")),
        feature_schema_sha256=_text(entry.get("feature_schema_sha256")).lower(),
        training_feature_semantics_id=_text(entry.get("training_feature_semantics_id")),
        live_feature_semantics_id=_text(entry.get("live_feature_semantics_id")),
        probability_source=probability_source,
        predict_fn=predict_fn,
        artifact_files=artifact_files,
    )


def _load_nn_predictor(
    version: str, entry: dict[str, Any], registry: dict[str, Any]
) -> ReplayPredictor:
    import torch

    from models.nn_runtime import TennisNet

    if _text(entry.get("probability_mode") or "raw").lower() != "raw":
        raise RuntimeError(
            f"nn:{version} probability_mode must be raw for checksum-pinned replay; "
            "calibration is a separate versioned artifact contract"
        )
    model_path = resolve_artifact_path("nn", registry=registry)
    scaler_path = resolve_artifact_path("nn", "scaler_file", registry=registry)
    if model_path is None or scaler_path is None:
        raise RuntimeError(f"nn:{version} is missing model/scaler paths")
    artifacts = [
        _artifact_file(model_path, "model", _text(entry.get("model_sha256"))),
        _artifact_file(scaler_path, "scaler", _text(entry.get("scaler_sha256"))),
    ]
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    if int(getattr(scaler, "n_features_in_", 0)) != len(EXACT_141_FEATURES):
        raise RuntimeError(f"nn:{version} scaler does not accept 141 features")
    model = TennisNet(len(EXACT_141_FEATURES))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        values = scaler.transform(frame.loc[:, EXACT_141_FEATURES].to_numpy(dtype=float))
        with torch.no_grad():
            result = model(torch.as_tensor(values, dtype=torch.float32)).cpu().numpy()
        return np.asarray(result, dtype=float).reshape(-1)

    return _predictor_metadata(
        "nn", version, entry, artifacts, predict_fn, probability_source="raw"
    )


def _load_xgb_predictor(
    version: str, entry: dict[str, Any], registry: dict[str, Any]
) -> ReplayPredictor:
    import xgboost as xgb

    model_path = resolve_artifact_path("xgboost", registry=registry)
    if model_path is None:
        raise RuntimeError(f"xgboost:{version} is missing model path")
    artifacts = [_artifact_file(model_path, "model", _text(entry.get("model_sha256")))]
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    artifact_features = [str(name) for name in model.feature_names_in_]
    if artifact_features != list(EXACT_141_FEATURES):
        raise RuntimeError(f"xgboost:{version} feature order does not match base_141")

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        ordered = frame.loc[:, EXACT_141_FEATURES].astype(float)
        return np.asarray(model.predict_proba(ordered)[:, 1], dtype=float)

    return _predictor_metadata("xgboost", version, entry, artifacts, predict_fn)


def _load_rf_predictor(
    version: str, entry: dict[str, Any], registry: dict[str, Any]
) -> ReplayPredictor:
    model_path = resolve_artifact_path("random_forest", registry=registry)
    if model_path is None:
        raise RuntimeError(f"random_forest:{version} is missing model path")
    artifacts = [_artifact_file(model_path, "model", _text(entry.get("model_sha256")))]
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    artifact_features = [str(name) for name in model.feature_names_in_]
    if artifact_features != list(EXACT_141_FEATURES):
        raise RuntimeError(f"random_forest:{version} feature order does not match base_141")

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        ordered = frame.loc[:, EXACT_141_FEATURES].astype(float)
        return np.asarray(model.predict_proba(ordered)[:, 1], dtype=float)

    return _predictor_metadata("random_forest", version, entry, artifacts, predict_fn)


def load_promoted_predictors() -> list[ReplayPredictor]:
    """Resolve and checksum-validate only current promoted registry artifacts."""
    registry = load_registry()
    loaders = {
        "nn": _load_nn_predictor,
        "xgboost": _load_xgb_predictor,
        "random_forest": _load_rf_predictor,
    }
    predictors: list[ReplayPredictor] = []
    for family in ("nn", "xgboost", "random_forest"):
        version = get_current_version(family, registry)
        entry = get_model_entry(family, version=version, registry=registry)
        if not version or version == "unknown" or not entry:
            raise RuntimeError(f"registry has no current promoted {family} entry")
        _entry_contract(family, version, entry)
        predictors.append(loaders[family](version, entry, registry))
    return predictors


def _source_hashes(evidence: replay_manifest.ReplayManifest) -> dict[str, str]:
    return {
        _text(source.get("path")): _text(source.get("sha256")).lower()
        for source in evidence.source_files
        if _text(source.get("path"))
    }


def _selected_payload(source_row: pd.Series) -> dict[str, Any]:
    if "features_json" in source_row.index:
        try:
            payload = json.loads(_text(source_row.get("features_json")))
        except json.JSONDecodeError as exc:
            raise RuntimeError("selected aggregate feature row has invalid features_json") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("selected aggregate feature row is not a feature object")
        return payload
    missing = [name for name in EXACT_141_FEATURES if name not in source_row.index]
    if missing:
        raise RuntimeError(
            f"selected tabular feature row is missing base_141 columns: {missing[:5]}"
        )
    return {name: source_row.get(name) for name in EXACT_141_FEATURES}


def load_selected_vectors(
    prod_dir: str | Path,
    evidence: replay_manifest.ReplayManifest,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload and hash-verify selected vectors admitted by the evidence tiers."""
    production = Path(prod_dir).resolve()
    frame = evidence.frame.copy()
    eligible = frame[
        frame["replay_tier"].isin(REPLAYABLE_TIERS)
        & frame["artifact_schema_id"].eq(FEATURE_SCHEMA_ID)
        & frame["artifact_schema_sha256"].eq(FEATURE_SCHEMA_SHA256)
        & frame["artifact_schema_compatible"].map(_truthy)
        & frame["feature_vector_sha256"].map(_text).ne("")
    ].copy()
    # Exact rows require immutable referential verification. Legacy rows are
    # intentionally allowed only under their separate, unambiguous hash tier.
    exact = eligible["replay_tier"].isin(("GOLD_REPLAY", "EXACT_INCOMPLETE"))
    eligible = eligible[~exact | eligible["snapshot_verified"].map(_truthy)].copy()
    eligible = eligible.sort_values("match_uid", kind="stable").reset_index(drop=True)
    if eligible["match_uid"].duplicated().any():
        raise RuntimeError("replay evidence contains duplicate match_uid rows")

    expected_hashes = _source_hashes(evidence)
    source_cache: dict[str, pd.DataFrame] = {}
    verified_source_hashes: dict[str, str] = {}
    vector_rows: list[dict[str, float]] = []
    for _, row in eligible.iterrows():
        relative = _text(row.get("feature_source_file"))
        expected_source_hash = expected_hashes.get(relative, "")
        if not relative or not expected_source_hash:
            raise RuntimeError(
                f"{row['match_uid']} selected vector lacks hashed source provenance"
            )
        source_path = (production / relative).resolve()
        try:
            source_path.relative_to(production)
        except ValueError as exc:
            raise RuntimeError(f"selected feature source escapes production: {relative}") from exc
        actual_source_hash = verified_source_hashes.get(relative)
        if actual_source_hash is None:
            actual_source_hash = _sha256_file(source_path)
            verified_source_hashes[relative] = actual_source_hash
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                f"selected feature source changed after manifest build: {relative}"
            )
        if relative not in source_cache:
            source_cache[relative] = pd.read_csv(
                source_path, low_memory=False, keep_default_na=False
            )
        source_frame = source_cache[relative]
        try:
            source_line = int(float(row["feature_source_row"]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"{row['match_uid']} has invalid feature source row") from exc
        position = source_line - 2
        if position < 0 or position >= len(source_frame):
            raise RuntimeError(
                f"{row['match_uid']} feature source row {source_line} is out of range"
            )
        source_row = source_frame.iloc[position]
        source_snapshot_id = _text(source_row.get("feature_snapshot_id"))
        manifest_snapshot_id = _text(row.get("feature_snapshot_id"))
        if row["replay_tier"] != "LEGACY_MATCHED" and source_snapshot_id != manifest_snapshot_id:
            raise RuntimeError(f"{row['match_uid']} feature snapshot foreign key changed")
        payload = _selected_payload(source_row)
        issues = feature_validation_issues(payload)
        schema_hash, vector_hash, feature_count = feature_fingerprint(payload)
        if issues or not vector_hash:
            raise RuntimeError(
                f"{row['match_uid']} selected vector failed structural validation: {issues}"
            )
        if (
            schema_hash != FEATURE_SCHEMA_SHA256
            or schema_hash != _text(row["feature_schema_sha256"])
            or vector_hash != _text(row["feature_vector_sha256"])
            or feature_count != len(EXACT_141_FEATURES)
        ):
            raise RuntimeError(f"{row['match_uid']} selected vector lineage hash mismatch")
        vector_rows.append({name: float(payload[name]) for name in EXACT_141_FEATURES})

    vectors = pd.DataFrame(vector_rows, columns=EXACT_141_FEATURES)
    if len(vectors) != len(eligible):
        raise RuntimeError("selected vector cardinality changed while loading evidence")
    return eligible, vectors


def _scorable(row: pd.Series) -> bool:
    winner = pd.to_numeric(pd.Series([row.get("actual_winner")]), errors="coerce").iloc[0]
    return (
        _text(row.get("outcome_status")) == "authoritative_conflict_free"
        and not pd.isna(winner)
        and int(winner) in (1, 2)
    )


def build_metric_frame(replayed: pd.DataFrame) -> pd.DataFrame:
    """Score each model/tier independently with the ledger's metric formulas."""
    rows: list[dict[str, Any]] = []
    identities = (
        replayed[["model_family", "model_version", "model_id"]]
        .drop_duplicates()
        .sort_values(["model_family", "model_version"], kind="stable")
    )
    for _, identity in identities.iterrows():
        model_rows = replayed[
            (replayed["model_family"] == identity["model_family"])
            & (replayed["model_version"] == identity["model_version"])
        ]
        for tier in REPLAYABLE_TIERS:
            tier_rows = model_rows[model_rows["replay_tier"] == tier]
            scored = tier_rows[tier_rows["scorable"].map(_truthy)]
            base = {
                "model_family": identity["model_family"],
                "model_version": identity["model_version"],
                "model_id": identity["model_id"],
                "replay_tier": tier,
                "n_replayed": int(len(tier_rows)),
            }
            if scored.empty:
                values = {
                    "n": 0,
                    "accuracy": np.nan,
                    "auc": np.nan,
                    "log_loss": np.nan,
                    "brier": np.nan,
                    "ece": np.nan,
                    "cal_slope": np.nan,
                    "cal_intercept": np.nan,
                }
            else:
                values = metrics.compute_all(
                    scored["y1"].astype(int).to_numpy(),
                    scored["replay_p1_prob"].astype(float).to_numpy(),
                )
            rows.append({**base, **values})
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_model_replay(
    prod_dir: str | Path,
    *,
    evidence: replay_manifest.ReplayManifest | None = None,
    predictors: Sequence[ReplayPredictor] | None = None,
) -> ModelReplay:
    """Apply promoted artifacts to verified vectors without writing any files."""
    evidence = evidence or replay_manifest.build_replay_manifest(prod_dir)
    predictor_list = list(predictors) if predictors is not None else load_promoted_predictors()
    if not predictor_list:
        raise RuntimeError("model replay requires at least one predictor")
    model_ids = [predictor.model_id for predictor in predictor_list]
    if len(model_ids) != len(set(model_ids)):
        raise RuntimeError("model replay predictor identities must be unique")
    for predictor in predictor_list:
        if (
            predictor.feature_schema_id != FEATURE_SCHEMA_ID
            or predictor.feature_schema_sha256 != FEATURE_SCHEMA_SHA256
        ):
            raise RuntimeError(f"{predictor.model_id} is not compatible with base_141@1.0.0")

    selected, vectors = load_selected_vectors(prod_dir, evidence)
    rows: list[dict[str, Any]] = []
    for predictor in predictor_list:
        probabilities = predictor.predict(vectors)
        for position, (_, source) in enumerate(selected.iterrows()):
            is_scorable = _scorable(source)
            winner = pd.to_numeric(
                pd.Series([source.get("actual_winner")]), errors="coerce"
            ).iloc[0]
            y1: int | str = int(winner == 1) if is_scorable else ""
            p1_probability = float(probabilities[position])
            rows.append(
                {
                    "match_uid": source["match_uid"],
                    "replay_tier": source["replay_tier"],
                    "match_date": source["match_date"],
                    "tournament": source["tournament"],
                    "p1": source["p1"],
                    "p2": source["p2"],
                    "feature_snapshot_id": source["feature_snapshot_id"],
                    "feature_schema_id": FEATURE_SCHEMA_ID,
                    "feature_schema_sha256": source["feature_schema_sha256"],
                    "feature_vector_sha256": source["feature_vector_sha256"],
                    "feature_source_file": source["feature_source_file"],
                    "feature_source_row": source["feature_source_row"],
                    "features_complete": source["features_complete"],
                    "prediction_observed_at": source["prediction_observed_at"],
                    "match_start_at_utc": source["match_start_at_utc"],
                    "reason_codes": source["reason_codes"],
                    "actual_winner": source["actual_winner"],
                    "outcome_status": source["outcome_status"],
                    "scorable": is_scorable,
                    "y1": y1,
                    "p1_odds_decimal": source["p1_odds_decimal"],
                    "p2_odds_decimal": source["p2_odds_decimal"],
                    "model_family": predictor.family,
                    "model_version": predictor.version,
                    "model_id": predictor.model_id,
                    "model_sha256": predictor.model_sha256,
                    "training_feature_semantics_id": predictor.training_feature_semantics_id,
                    "live_feature_semantics_id": predictor.live_feature_semantics_id,
                    "probability_source": predictor.probability_source,
                    "replay_p1_prob": p1_probability,
                    "replay_p2_prob": 1.0 - p1_probability,
                }
            )
    replayed = pd.DataFrame(rows, columns=REPLAY_COLUMNS).sort_values(
        ["model_family", "model_version", "match_uid"], kind="stable"
    ).reset_index(drop=True)
    metric_frame = build_metric_frame(replayed)
    return ModelReplay(
        frame=replayed,
        metric_frame=metric_frame,
        evidence=evidence,
        predictors=predictor_list,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def replay_summary(result: ModelReplay) -> dict[str, Any]:
    row_counts = (
        result.frame.drop_duplicates(["match_uid", "replay_tier"])["replay_tier"]
        .value_counts()
        .to_dict()
    )
    return {
        "manifest_type": "same_schema_model_replay",
        "schema_version": MODEL_REPLAY_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "models": [predictor.metadata() for predictor in result.predictors],
        "replayable_match_counts": {
            tier: int(row_counts.get(tier, 0)) for tier in REPLAYABLE_TIERS
        },
        "prediction_rows": int(len(result.frame)),
        "metrics": _json_safe(result.metric_frame.to_dict(orient="records")),
    }


def write_model_replay(
    result: ModelReplay,
    out_dir: str | Path,
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Write replay rows, tiered metrics, and provenance into a new directory."""
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    destination = Path(out_dir).resolve() / timestamp.strftime(
        "model_replay_%Y%m%dT%H%M%SZ"
    )
    destination.mkdir(parents=True, exist_ok=False)
    replay_path = destination / OUTPUT_REPLAY_CSV
    metrics_path = destination / OUTPUT_METRICS_CSV
    result.frame.to_csv(replay_path, index=False)
    result.metric_frame.to_csv(metrics_path, index=False)

    summary = replay_summary(result)
    summary.update(
        {
            "generated_at": timestamp.isoformat().replace("+00:00", "Z"),
            "execution_contract": (
                "read-only same-schema inference; no training, tuning, settlement, "
                "outcome reconstruction, or operational log mutation"
            ),
            "historical_evidence": replay_manifest.manifest_summary(result.evidence),
            "source_files": result.evidence.source_files,
            "registry": {
                "path": str(REGISTRY_PATH.resolve()),
                "sha256": _sha256_file(REGISTRY_PATH),
            },
            "artifacts": {
                OUTPUT_REPLAY_CSV: {
                    "sha256": _sha256_file(replay_path),
                    "rows": int(len(result.frame)),
                },
                OUTPUT_METRICS_CSV: {
                    "sha256": _sha256_file(metrics_path),
                    "rows": int(len(result.metric_frame)),
                },
            },
        }
    )
    (destination / OUTPUT_MANIFEST_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prod-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Production directory containing prediction and feature lineage",
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Optional output parent. When omitted, nothing is written. When supplied, "
            "a new timestamp-versioned child directory is created."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_model_replay(args.prod_dir)
    summary = replay_summary(result)
    if args.out_dir:
        summary["written_to"] = str(write_model_replay(result, args.out_dir))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
