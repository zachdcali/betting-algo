"""Pure builders for future direct operational-Postgres writes.

Nothing in this module opens a connection or changes the current CSV pipeline.
It translates the values already produced by ``main.py``, ``fetch_bovada.py``,
``feature_vector_log.py`` and ``prediction_logger.py`` into deterministic typed
``RecordBatch`` objects.  Integration should initially dual-write these batches
inside one caller-owned transaction and retain CSV parity checks.

Legacy match metadata is written to append-only
``ops.match_metadata_observations``. Candidate eligibility round evidence uses
the separate ``ops.eligibility_match_round_observations`` table so the 1.2
scaffolding cannot alter ``api.current_match_metadata`` before cutover.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

try:
    from tennis_enums import ACTIVE_ROUND_CODES
except ImportError:  # pragma: no cover - package-style execution
    from production.tennis_enums import ACTIVE_ROUND_CODES  # type: ignore

try:
    from logging_utils import (
        build_feature_snapshot_id, build_match_uid, build_odds_snapshot_uid,
        normalize_name, stable_hash,
    )
    from versioning import (
        FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256, LIVE_SEMANTICS_ID,
        LOGGING_SCHEMA_VERSION, validate_semver,
    )
except ImportError:  # pragma: no cover - package-style execution
    from production.logging_utils import (  # type: ignore
        build_feature_snapshot_id, build_match_uid, build_odds_snapshot_uid,
        normalize_name, stable_hash,
    )
    from production.versioning import (  # type: ignore
        FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256, LIVE_SEMANTICS_ID,
        LOGGING_SCHEMA_VERSION, validate_semver,
    )

try:
    from feature_contract import normalize_feature_vector, vector_sha256
except ImportError:  # pragma: no cover - package-style execution
    from production.feature_contract import (  # type: ignore
        normalize_feature_vector, vector_sha256,
    )

from .eligibility import normalize_uri_scheme
from .records import RecordBatch, canonical_json, deterministic_key


MATCH_METADATA_TABLE_REQUIRED = "ops.match_metadata_observations"
ELIGIBILITY_MATCH_ROUND_TABLE = "ops.eligibility_match_round_observations"
MATCH_METADATA_FIELDS = (
    "match_date", "match_start_at_utc", "tournament", "event_title",
    "round_code", "surface", "level",
)

# A verified source can replace a weaker value; a failed or null observation
# cannot. Equal-quality evidence uses the freshest observation.
PROVENANCE_QUALITY = {
    "default": 0,
    "inferred": 100,
    "bovada": 200,
    "tournament_registry": 300,
    "canonical_store": 350,
    "official": 400,
}


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit timezone")
    return value.astimezone(timezone.utc)


def _optional_utc(value: datetime | None, field: str) -> datetime | None:
    return None if value is None else _utc(value, field)


def _decimal(value: Any, field: str, *, minimum: Decimal | None = None) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


def _uuid(kind: str, idempotency_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"betting-algo:{kind}:{idempotency_key}"))


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _probability_pair(player1: Any, player2: Any, *, label: str) -> tuple[Decimal, Decimal]:
    first = _decimal(player1, f"{label}.player1")
    second = _decimal(player2, f"{label}.player2")
    if first < 0 or first > 1 or second < 0 or second > 1:
        raise ValueError(f"{label} probabilities must be within [0, 1]")
    if abs((first + second) - Decimal("1")) > Decimal("0.0001"):
        raise ValueError(f"{label} probabilities must sum to 1")
    return first, second


def _record_batch(table: str, record: Mapping[str, Any]) -> RecordBatch:
    return RecordBatch.from_records(table, [record])


@lru_cache(maxsize=1)
def _ordered_feature_names() -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[1] / "features" / "schema_141.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ordered = sorted(payload["features"], key=lambda item: int(item["index"]))
    names = tuple(str(item["name"]) for item in ordered)
    digest = sha256("\x1f".join(names).encode("utf-8")).hexdigest()
    if digest != FEATURE_SCHEMA_SHA256:
        raise RuntimeError("live feature schema file does not match version contract")
    return names


@dataclass(frozen=True)
class MatchIdentity:
    """Existing match UID plus a less metadata-sensitive operational anchor."""

    match_uid: str
    match_anchor_key: str
    player1: str
    player2: str
    match_date: date


def match_identity(
    *,
    player1: str,
    player2: str,
    match_date: date,
    tournament: str = "",
    round_code: str = "",
    surface: str = "",
    source_event_key: str = "",
) -> MatchIdentity:
    """Use today's UID verbatim and surface its mutable-metadata limitation.

    ``match_uid`` remains compatible with every current CSV. ``match_anchor_key``
    excludes round/surface so later verified metadata does not create a new
    logical match. A bookmaker event ID should replace the event-title fallback
    once the collector captures one.
    """
    if not _clean(player1) or not _clean(player2):
        raise ValueError("both players are required")
    if not isinstance(match_date, date):
        raise TypeError("match_date must be date")
    uid = build_match_uid(
        player1, player2, match_date.isoformat(), tournament, round_code, surface,
    )
    players = sorted((normalize_name(player1), normalize_name(player2)))
    event_anchor = source_event_key or tournament
    anchor = "match_anchor_" + stable_hash(
        players[0], players[1], match_date.isoformat(), event_anchor,
    )
    return MatchIdentity(uid, anchor, player1.strip(), player2.strip(), match_date)


@dataclass(frozen=True)
class LiveRecordBuilder:
    external_run_id: str
    run_started_at: datetime
    run_kind: str = "prediction_pipeline"

    def __post_init__(self) -> None:
        if not _clean(self.external_run_id):
            raise ValueError("external_run_id is required")
        object.__setattr__(self, "run_started_at", _utc(self.run_started_at, "run_started_at"))

    @property
    def run_id(self) -> str:
        return _uuid("pipeline_run", self.run_key)

    @property
    def run_key(self) -> str:
        return f"pipeline_run:{self.external_run_id}"

    def pipeline_run(
        self,
        *,
        status: str = "running",
        completed_at: datetime | None = None,
        metrics: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> RecordBatch:
        normalized_status = (_clean(status) or "running").lower()
        if normalized_status == "failed" and not _clean(error_message):
            raise ValueError("failed pipeline runs require error_message")
        return _record_batch("ops.pipeline_runs", {
            "run_id": self.run_id,
            "idempotency_key": self.run_key,
            "external_run_id": self.external_run_id,
            "run_kind": self.run_kind,
            "status": normalized_status,
            "started_at": self.run_started_at,
            "completed_at": _optional_utc(completed_at, "completed_at"),
            "metrics": canonical_json(metrics or {}),
            "error_message": _clean(error_message),
        })

    def pipeline_stage(
        self,
        *,
        stage_name: str,
        attempt: int = 1,
        status: str = "running",
        started_at: datetime,
        completed_at: datetime | None = None,
        metrics: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> RecordBatch:
        name = _clean(stage_name)
        if not name:
            raise ValueError("stage_name is required")
        if attempt < 1:
            raise ValueError("stage attempt must be >= 1")
        normalized_status = (_clean(status) or "running").lower()
        if normalized_status == "failed" and not _clean(error_message):
            raise ValueError("failed stages require error_message")
        key = f"pipeline_stage:{self.external_run_id}:{name}:{attempt}"
        return _record_batch("ops.pipeline_run_stages", {
            "stage_id": _uuid("pipeline_stage", key),
            "idempotency_key": key,
            "run_id": self.run_id,
            "stage_name": name,
            "attempt": attempt,
            "status": normalized_status,
            "started_at": _utc(started_at, "started_at"),
            "completed_at": _optional_utc(completed_at, "completed_at"),
            "metrics": canonical_json(metrics or {}),
            "error_message": _clean(error_message),
        })

    def source_fetch(
        self,
        *,
        source_name: str,
        fetch_kind: str,
        attempt: int,
        started_at: datetime,
        status: str,
        completed_at: datetime | None = None,
        rows_observed: int = 0,
        http_status: int | None = None,
        request: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> RecordBatch:
        source = _clean(source_name)
        kind = _clean(fetch_kind)
        if not source or not kind:
            raise ValueError("source_name and fetch_kind are required")
        if attempt < 1 or rows_observed < 0:
            raise ValueError("attempt must be >= 1 and rows_observed must be >= 0")
        normalized_status = (_clean(status) or "failed").lower()
        if normalized_status in {"failed", "blocked"} and not _clean(error_message):
            raise ValueError("failed/blocked fetches require error_message")
        started = _utc(started_at, "started_at")
        key = deterministic_key(
            "source_fetch", self.external_run_id, source, kind, attempt, started,
        )
        return _record_batch("raw.source_fetches", {
            "source_fetch_id": _uuid("source_fetch", key),
            "idempotency_key": key,
            "run_id": self.run_id,
            "source_name": source,
            "fetch_kind": kind,
            "attempt": attempt,
            "status": normalized_status,
            "started_at": started,
            "completed_at": _optional_utc(completed_at, "completed_at"),
            "rows_observed": rows_observed,
            "http_status": http_status,
            "request": canonical_json(request or {}),
            "error_message": _clean(error_message),
        })

    def source_artifact(
        self,
        *,
        source_fetch_id: str,
        artifact_kind: str,
        storage_uri: str,
        content_sha256: str,
        captured_at: datetime,
        content_type: str | None = None,
        byte_size: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RecordBatch:
        """Describe a preserved private source body; never store it inline."""
        fetch_id = _clean(source_fetch_id)
        kind = _clean(artifact_kind)
        uri = _clean(storage_uri)
        digest = (_clean(content_sha256) or "").lower()
        if not fetch_id or not kind or not uri:
            raise ValueError("source_fetch_id, artifact_kind, and storage_uri are required")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("content_sha256 must be a lowercase SHA-256")
        if byte_size is not None and int(byte_size) < 0:
            raise ValueError("byte_size must be non-negative")
        artifact_metadata = dict(metadata or {})
        try:
            original_uri = normalize_uri_scheme(
                artifact_metadata.get("source_uri"),
                "source artifact metadata source_uri",
            )
        except ValueError as exc:
            raise ValueError("source artifact metadata requires source_uri") from exc
        artifact_metadata["source_uri"] = original_uri
        captured = _utc(captured_at, "captured_at")
        key = deterministic_key(
            "source_artifact", fetch_id, kind, uri, digest, captured,
        )
        return _record_batch("raw.source_artifacts", {
            "source_artifact_id": _uuid("source_artifact", key),
            "idempotency_key": key,
            "source_fetch_id": fetch_id,
            "artifact_kind": kind,
            "storage_uri": uri,
            "content_sha256": digest,
            "content_type": _clean(content_type),
            "byte_size": None if byte_size is None else int(byte_size),
            "captured_at": captured,
            "metadata": canonical_json(artifact_metadata),
        })

    def odds_observation(
        self,
        *,
        identity: MatchIdentity,
        source_fetch_id: str,
        observed_at: datetime,
        match_start_at_utc: datetime,
        player1_decimal_odds: Any,
        player2_decimal_odds: Any,
        player1_market_probability: Any,
        player2_market_probability: Any,
        tournament: str | None = None,
        event_title: str | None = None,
        surface: str | None = None,
        level: str | None = None,
        round_code: str | None = None,
        player1_american_odds: int | None = None,
        player2_american_odds: int | None = None,
        bookmaker: str = "bovada",
        external_observation_id: str | None = None,
        market_payload: Mapping[str, Any] | None = None,
    ) -> RecordBatch:
        first_odds = _decimal(player1_decimal_odds, "player1_decimal_odds", minimum=Decimal("1"))
        second_odds = _decimal(player2_decimal_odds, "player2_decimal_odds", minimum=Decimal("1"))
        if first_odds <= 1 or second_odds <= 1:
            raise ValueError("decimal odds must be greater than 1")
        first_prob, second_prob = _probability_pair(
            player1_market_probability, player2_market_probability, label="market",
        )
        observed = _utc(observed_at, "observed_at")
        starts = _utc(match_start_at_utc, "match_start_at_utc")
        if observed >= starts:
            raise ValueError("inference odds must be observed before match start")
        external_id = external_observation_id or build_odds_snapshot_uid(
            identity.match_uid, observed.isoformat(), starts.isoformat(), first_odds, second_odds,
        )
        key = f"odds_observation:{external_id}"
        return _record_batch("ops.odds_observations", {
            "odds_observation_id": _uuid("odds_observation", key),
            "idempotency_key": key,
            "source_fetch_id": source_fetch_id,
            "run_id": self.run_id,
            "external_observation_id": external_id,
            "match_uid": identity.match_uid,
            "match_anchor_key": identity.match_anchor_key,
            "observed_at": observed,
            "match_date": identity.match_date,
            "match_start_at_utc": starts,
            "tournament": _clean(tournament),
            "event_title": _clean(event_title),
            "surface": _clean(surface),
            "level": _clean(level),
            "round_code": _clean(round_code),
            "player1": identity.player1,
            "player2": identity.player2,
            "bookmaker": bookmaker,
            "market_type": "moneyline",
            "player1_decimal_odds": first_odds,
            "player2_decimal_odds": second_odds,
            "player1_american_odds": player1_american_odds,
            "player2_american_odds": player2_american_odds,
            "player1_market_probability": first_prob,
            "player2_market_probability": second_prob,
            "validation_status": "valid_two_sided_prestart",
            "inference_eligible": True,
            "market_payload": canonical_json(market_payload or {}),
        })

    def feature_snapshot(
        self,
        *,
        identity: MatchIdentity,
        player1: str,
        player2: str,
        captured_at: datetime,
        build_status: str,
        features_complete: bool,
        feature_vector: Mapping[str, Any],
        feature_vector_sha256: str | None,
        feature_count: int,
        external_feature_snapshot_id: str | None = None,
        defaulted_features: Sequence[str] = (),
    ) -> RecordBatch:
        external_id = external_feature_snapshot_id or build_feature_snapshot_id(
            identity.match_uid, self.external_run_id, player1, player2,
        )
        status = (_clean(build_status) or "error").lower()
        supplied_hash = _clean(feature_vector_sha256)
        names = _ordered_feature_names()
        normalized_vector, validation_issues = normalize_feature_vector(
            feature_vector, names
        )
        structural_hash = (
            vector_sha256(normalized_vector, names) if not validation_issues else None
        )
        issues = list(validation_issues)
        if feature_count != len(names):
            issues.append("feature_count_mismatch")
        if supplied_hash and structural_hash and supplied_hash != structural_hash:
            issues.append("feature_vector_hash_mismatch")
        defaults = [str(item).strip() for item in defaulted_features if str(item).strip()]
        defaults.extend(
            f"structural:{issue}" for issue in issues
            if f"structural:{issue}" not in defaults
        )
        verified_hash = (
            structural_hash
            if supplied_hash and structural_hash == supplied_hash and not issues
            else None
        )
        complete = bool(
            features_complete and status == "ok" and verified_hash and not defaults
        )
        if feature_count < 0:
            raise ValueError("feature_count must be >= 0")
        key = f"feature_snapshot:{external_id}"
        return _record_batch("ml.feature_snapshots", {
            "feature_snapshot_id": _uuid("feature_snapshot", key),
            "idempotency_key": key,
            "run_id": self.run_id,
            "external_feature_snapshot_id": external_id,
            "match_uid": identity.match_uid,
            "feature_schema_identifier": FEATURE_SCHEMA_ID,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_semantics_identifier": LIVE_SEMANTICS_ID,
            "captured_at": _utc(captured_at, "captured_at"),
            "build_status": status,
            "features_complete": complete,
            "lineage_quality": "exact_feature_snapshot_id",
            "feature_count": len(normalized_vector),
            "feature_vector_sha256": verified_hash,
            "feature_vector": canonical_json(normalized_vector),
            "defaulted_features": canonical_json(defaults),
        })

    def prediction_observations(
        self,
        *,
        identity: MatchIdentity,
        feature_snapshot_id: str,
        predicted_at: datetime,
        predictions: Sequence[Mapping[str, Any]],
        external_prediction_id: str,
        logging_quality: str = "snapshot_v2",
    ) -> RecordBatch:
        records: list[dict[str, Any]] = []
        for prediction in predictions:
            family = _clean(prediction.get("model_family"))
            version = _clean(prediction.get("model_version"))
            if not family or not version:
                raise ValueError("each prediction requires model_family and model_version")
            family = family.lower()
            version = validate_semver(version)
            decision_eligible = bool(prediction.get("decision_eligible", False))
            model_release_key = _clean(prediction.get("model_release_key"))
            if decision_eligible and not model_release_key:
                raise ValueError(
                    "decision-eligible prediction requires model_release_key"
                )
            if decision_eligible and logging_quality != "snapshot_v2":
                raise ValueError(
                    "decision-eligible prediction requires snapshot_v2 logging"
                )
            first, second = _probability_pair(
                prediction.get("player1_probability"),
                prediction.get("player2_probability"),
                label=f"prediction.{family}",
            )
            key = f"prediction:{external_prediction_id}:{family}:{version}"
            records.append({
                "prediction_observation_id": _uuid("prediction_observation", key),
                "idempotency_key": key,
                "run_id": self.run_id,
                "external_prediction_id": external_prediction_id,
                "match_uid": identity.match_uid,
                "feature_snapshot_id": feature_snapshot_id,
                "predicted_at": _utc(predicted_at, "predicted_at"),
                "model_family": family,
                "model_version": version,
                "model_role": _clean(prediction.get("model_role")) or "companion",
                "model_release_key": model_release_key,
                "decision_eligible": decision_eligible,
                "player1_probability": first,
                "player2_probability": second,
                "logging_schema_version": LOGGING_SCHEMA_VERSION,
                "logging_quality": logging_quality,
            })
        return RecordBatch.from_records("ml.prediction_observations", records)


def match_metadata_observation(
    *,
    identity: MatchIdentity,
    run_id: str,
    observed_at: datetime,
    observation_status: str = "success",
    source_name: str,
    source_fetch_id: str | None = None,
    field_provenance: Mapping[str, str],
    tournament: str | None = None,
    event_title: str | None = None,
    round_code: str | None = None,
    surface: str | None = None,
    level: str | None = None,
    match_start_at_utc: datetime | None = None,
) -> RecordBatch:
    """Return the proposed append-only metadata observation record.

    Failed observations are rejected here: the failure belongs in
    ``raw.source_fetches`` and must not manufacture a null metadata row.
    """
    if observation_status != "success":
        raise ValueError("failed refreshes belong in raw.source_fetches, not match metadata")
    values = {
        "match_date": identity.match_date,
        "match_start_at_utc": _optional_utc(match_start_at_utc, "match_start_at_utc"),
        "tournament": _clean(tournament),
        "event_title": _clean(event_title),
        "round_code": _clean(round_code),
        "surface": _clean(surface),
        "level": _clean(level),
    }
    present = {field: value for field, value in values.items() if value is not None}
    if not present:
        raise ValueError("a successful metadata observation must contain a value")
    unknown = set(field_provenance) - set(MATCH_METADATA_FIELDS)
    if unknown:
        raise ValueError(f"unknown metadata provenance fields: {sorted(unknown)}")
    for field in present:
        provenance = field_provenance.get(field)
        if provenance not in PROVENANCE_QUALITY:
            raise ValueError(f"{field} requires known field provenance")
    observed = _utc(observed_at, "observed_at")
    key = deterministic_key(
        "match_metadata", identity.match_anchor_key, source_name, observed,
        present, field_provenance,
    )
    return _record_batch(MATCH_METADATA_TABLE_REQUIRED, {
        "match_metadata_observation_id": _uuid("match_metadata_observation", key),
        "idempotency_key": key,
        "run_id": run_id,
        "source_fetch_id": source_fetch_id,
        "match_uid": identity.match_uid,
        "match_anchor_key": identity.match_anchor_key,
        "observed_at": observed,
        "source_name": source_name,
        **values,
        "field_provenance": canonical_json(dict(field_provenance)),
    })


def eligibility_match_round_observation(
    *,
    identity: MatchIdentity,
    run_id: str,
    observed_at: datetime,
    source_name: str,
    round_code: str,
    eligibility_generation_sha256: str,
    source_artifact_id: str,
    source_uri: str,
    source_content_sha256: str,
    confidence: Any,
    initial_review_state: str,
    expires_at: datetime,
    source_fetch_id: str,
    round_provenance: str = "official",
) -> RecordBatch:
    """Build isolated, review-gated candidate round evidence."""
    generation = str(eligibility_generation_sha256 or "").strip().lower()
    artifact_id = _clean(source_artifact_id)
    fetch_id = _clean(source_fetch_id)
    try:
        uri = normalize_uri_scheme(source_uri)
    except ValueError as exc:
        raise ValueError("eligibility round evidence requires source_uri") from exc
    digest = str(source_content_sha256 or "").strip().lower()
    source = _clean(source_name)
    provenance = str(round_provenance or "").strip().lower()
    review_state = str(initial_review_state or "").strip().lower()
    observed = _utc(observed_at, "observed_at")
    expiry = _utc(expires_at, "expires_at")
    if not re.fullmatch(r"[0-9a-f]{64}", generation):
        raise ValueError("eligibility generation must be a lowercase SHA-256")
    if not source:
        raise ValueError("source_name is required")
    if artifact_id is None:
        raise ValueError("eligibility round evidence requires source_artifact_id")
    if fetch_id is None:
        raise ValueError("eligibility round evidence requires source_fetch_id")
    normalized_round = str(round_code or "").strip().upper()
    if normalized_round not in ACTIVE_ROUND_CODES:
        raise ValueError(f"unsupported round_code: {normalized_round or '<blank>'}")
    if provenance not in PROVENANCE_QUALITY:
        raise ValueError("round_provenance must be a known provenance code")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("eligibility round evidence requires source_content_sha256")
    try:
        confidence_value = _decimal(confidence, "confidence")
    except (TypeError, ValueError) as exc:
        raise ValueError("eligibility round evidence requires confidence") from exc
    if confidence_value < 0 or confidence_value > 1:
        raise ValueError("confidence must be within [0, 1]")
    if review_state not in {"unreviewed", "quarantined"}:
        raise ValueError(
            "eligibility round evidence must start unreviewed or quarantined"
        )
    if expiry <= observed:
        raise ValueError("eligibility round evidence requires a future expires_at")
    key = deterministic_key(
        "eligibility_match_round", generation, identity.match_anchor_key,
        source, normalized_round, observed, digest,
    )
    return _record_batch(ELIGIBILITY_MATCH_ROUND_TABLE, {
        "eligibility_match_round_observation_id": _uuid(
            "eligibility_match_round_observation", key,
        ),
        "idempotency_key": key,
        "eligibility_generation_sha256": generation,
        "run_id": run_id,
        "source_fetch_id": fetch_id,
        "source_artifact_id": artifact_id,
        "match_uid": identity.match_uid,
        "match_anchor_key": identity.match_anchor_key,
        "observed_at": observed,
        "source_name": source,
        "round_code": normalized_round,
        "round_provenance": provenance,
        "source_uri": uri,
        "source_content_sha256": digest,
        "confidence": confidence_value,
        "initial_review_state": review_state,
        "expires_at": expiry,
    })


def resolve_match_metadata(observations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Project append-only metadata without null or low-quality regression."""
    chosen: dict[str, tuple[int, datetime, Any, str]] = {}
    for observation in observations:
        if observation.get("observation_status", "success") != "success":
            continue
        observed_at = observation.get("observed_at")
        if not isinstance(observed_at, datetime):
            raise TypeError("metadata observed_at must be datetime")
        observed = _utc(observed_at, "observed_at")
        raw_provenance = observation.get("field_provenance", {})
        if isinstance(raw_provenance, str):
            raw_provenance = json.loads(raw_provenance)
        for field in MATCH_METADATA_FIELDS:
            value = observation.get(field)
            if value is None or str(value).strip() == "":
                continue
            provenance = raw_provenance.get(field)
            quality = PROVENANCE_QUALITY.get(provenance, -1)
            candidate = (quality, observed, value, provenance)
            existing = chosen.get(field)
            if existing is None or candidate[:2] > existing[:2]:
                chosen[field] = candidate
    result = {field: (chosen[field][2] if field in chosen else None) for field in MATCH_METADATA_FIELDS}
    result["_field_provenance"] = {
        field: {
            "source": selected[3],
            "quality": selected[0],
            "observed_at": selected[1],
        }
        for field, selected in chosen.items()
    }
    return result
