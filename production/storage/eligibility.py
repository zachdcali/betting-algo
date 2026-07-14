"""Eligibility evidence builders, compatibility projection, and safe reads.

The normalized operational database is the eventual authority for player
identity, aliases, profile fields, and prospective round evidence.  The
existing ``public.players`` and ``public.player_aliases`` tables remain the
compatibility projection until an explicitly accepted generation passes
staging parity and cutover.

This module never opens a database connection, applies DDL, commits, or writes
local caches.  Callers supply a connection and own its lifecycle.  Read methods
verify both the exact operational schema and the expected accepted generation
before returning a value; a missing, stale, ambiguous, or conflicting value is
an explicit unresolved result, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
import os
import re
import unicodedata
from typing import Any, Iterable, Mapping, Protocol
from uuid import NAMESPACE_URL, uuid5

from .records import RecordBatch, canonical_json, content_sha256, deterministic_key

try:
    from tennis_enums import ACTIVE_ROUND_CODES, AUTHORITATIVE_HAND_CODES
except ImportError:  # pragma: no cover - package-style execution
    from production.tennis_enums import (  # type: ignore
        ACTIVE_ROUND_CODES, AUTHORITATIVE_HAND_CODES,
    )

try:
    from versioning import OPERATIONAL_SCHEMA_VERSION
except ImportError:  # pragma: no cover - package-style execution
    from production.versioning import OPERATIONAL_SCHEMA_VERSION  # type: ignore


ELIGIBILITY_CONTRACT_VERSION = "1.0.0"
ELIGIBILITY_GENERATION_ENV = "ELIGIBILITY_PROVENANCE_GENERATION_SHA256"
ELIGIBILITY_PROJECTION_SEAL_ENV = "ELIGIBILITY_PROJECTION_SEAL_SHA256"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
URI_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
ROUND_CODES = ACTIVE_ROUND_CODES
PROFILE_FIELDS = frozenset({
    "height_cm", "hand", "country", "birthdate", "ta_slug", "atp_url",
})
REVIEW_STATES = frozenset({
    "unreviewed", "accepted", "rejected", "quarantined",
})


class EligibilityContractError(RuntimeError):
    """The reader cannot prove the exact accepted database contract."""


class EligibilityMode(str, Enum):
    LEGACY = "legacy"
    REQUIRED = "required"

    @classmethod
    def parse(cls, value: Any) -> "EligibilityMode":
        normalized = str(value or "").strip().lower()
        if not normalized:
            return cls.LEGACY
        try:
            return cls(normalized)
        except ValueError as exc:
            raise EligibilityContractError(
                "ELIGIBILITY_PROVENANCE_MODE must be exactly legacy or required"
            ) from exc


def eligibility_mode(environ: Mapping[str, str] | None = None) -> EligibilityMode:
    environment = os.environ if environ is None else environ
    return EligibilityMode.parse(environment.get("ELIGIBILITY_PROVENANCE_MODE", ""))


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    MISSING = "missing"
    CONFLICT = "conflict"
    STALE = "stale"
    UNREVIEWED = "unreviewed"
    CONTRACT_ERROR = "contract_error"


@dataclass(frozen=True)
class Resolution:
    status: ResolutionStatus
    value: Any = None
    detail: str = ""
    generation_sha256: str = ""
    projection_seal_sha256: str = ""
    observed_at: datetime | None = None
    expires_at: datetime | None = None
    confidence: Decimal | None = None

    @property
    def resolved(self) -> bool:
        return self.status is ResolutionStatus.RESOLVED


@dataclass(frozen=True)
class EvidenceSource:
    source_name: str
    source_uri: str
    source_content_sha256: str
    observed_at: datetime
    confidence: Decimal
    initial_review_state: str = "unreviewed"
    source_artifact_id: str | None = None
    expires_at: datetime | None = None
    compatibility_import: bool = False

    @classmethod
    def validated(
        cls,
        *,
        source_name: str,
        source_uri: str,
        source_content_sha256: str,
        observed_at: datetime,
        confidence: Any,
        initial_review_state: str = "unreviewed",
        source_artifact_id: str | None = None,
        expires_at: datetime | None = None,
        compatibility_import: bool = False,
    ) -> "EvidenceSource":
        name = str(source_name or "").strip()
        uri = normalize_uri_scheme(source_uri)
        digest = str(source_content_sha256 or "").strip().lower()
        state = str(initial_review_state or "").strip().lower()
        if not name:
            raise ValueError("source_name is required")
        if not SHA256_RE.fullmatch(digest):
            raise ValueError("source_content_sha256 must be a lowercase SHA-256")
        observed = _utc(observed_at, "observed_at")
        expiry = None if expires_at is None else _utc(expires_at, "expires_at")
        if expiry is not None and expiry <= observed:
            raise ValueError("expires_at must be after observed_at")
        try:
            conf = Decimal(str(confidence))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if not conf.is_finite() or conf < 0 or conf > 1:
            raise ValueError("confidence must be within [0, 1]")
        if state not in {"unreviewed", "quarantined"}:
            raise ValueError(f"unknown initial_review_state: {state}")
        artifact_id = str(source_artifact_id or "").strip() or None
        compatibility = bool(compatibility_import)
        if compatibility:
            if name != "public_compatibility_projection":
                raise ValueError(
                    "compatibility_import requires public_compatibility_projection"
                )
            if not uri.startswith("compatibility://public.players-player_aliases/"):
                raise ValueError(
                    "compatibility_import requires the typed public projection URI"
                )
            if artifact_id is not None:
                raise ValueError("compatibility_import must not claim a raw artifact")
        elif artifact_id is None:
            raise ValueError("authoritative evidence requires source_artifact_id")
        return cls(
            source_name=name,
            source_uri=uri,
            source_content_sha256=digest,
            observed_at=observed,
            confidence=conf,
            initial_review_state=state,
            source_artifact_id=artifact_id,
            expires_at=expiry,
            compatibility_import=compatibility,
        )


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit timezone")
    return value.astimezone(timezone.utc)


def _uuid(kind: str, key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"betting-algo:{kind}:{key}"))


def normalize_uri_scheme(value: Any, field: str = "source_uri") -> str:
    """Trim a URI and canonicalize only its scheme to lowercase."""
    text = str(value or "").strip()
    match = URI_RE.match(text)
    if match is None:
        raise ValueError(f"{field} must be an explicit URI")
    return match.group(0).lower() + text[match.end():]


def normalize_identity_name(value: str) -> str:
    """Match the compatibility projection's accent/punctuation-free key."""
    decomposed = unicodedata.normalize("NFKD", str(value or "").casefold())
    result = "".join(
        char for char in decomposed
        if not unicodedata.combining(char) and char.isascii() and char.isalnum()
    )
    if not result:
        raise ValueError("identity name normalizes to an empty key")
    return result


def normalize_profile_value(field_name: str, value: Any) -> str:
    field = str(field_name or "").strip()
    if field not in PROFILE_FIELDS:
        raise ValueError(f"unsupported profile field: {field}")
    if field == "height_cm":
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("height_cm must be numeric") from exc
        if not number.is_finite() or number < 150 or number > 230:
            raise ValueError("height_cm must be within [150, 230]")
        return format(number.normalize(), "f")
    if field == "hand":
        hand = str(value or "").strip().upper()
        if hand not in AUTHORITATIVE_HAND_CODES:
            raise ValueError("hand must be L, R, or A; U/unknown is unresolved")
        return hand
    if field == "birthdate":
        if isinstance(value, datetime):
            value = value.date()
        if isinstance(value, date):
            return value.isoformat()
        text = str(value or "").strip()
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise ValueError("birthdate must be YYYY-MM-DD") from exc
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} cannot be blank")
    if field == "atp_url":
        return normalize_uri_scheme(text, "atp_url")
    return text


def normalize_round_code(value: Any) -> str:
    code = str(value or "").strip().upper()
    if code not in ROUND_CODES:
        raise ValueError(f"unsupported round_code: {code or '<blank>'}")
    return code


def _generation_sha(value: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError("eligibility generation must be a lowercase SHA-256")
    return digest


def _common_evidence_columns(source: EvidenceSource) -> dict[str, Any]:
    return {
        "source_name": source.source_name,
        "source_uri": source.source_uri,
        "source_content_sha256": source.source_content_sha256,
        "source_artifact_id": source.source_artifact_id,
        "observed_at": source.observed_at,
        "confidence": source.confidence,
        "initial_review_state": source.initial_review_state,
        "expires_at": source.expires_at,
        "compatibility_import": source.compatibility_import,
    }


ELIGIBILITY_PROJECTION_TABLES = frozenset({
    "ops.player_entities",
    "ops.player_identity_observations",
    "ops.player_alias_observations",
    "ops.player_profile_observations",
    "ops.eligibility_match_round_observations",
    "ops.eligibility_review_events",
})


@dataclass(frozen=True)
class ProjectionSeal:
    projection_seal_sha256: str
    projection_row_count: int


def projection_seal_from_batches(
    batches: Iterable[RecordBatch], *, generation_sha256: str,
) -> ProjectionSeal:
    """Mirror the database seal over exact generation-bound immutable facts."""
    generation = _generation_sha(generation_sha256)
    rows: dict[tuple[str, str], str] = {}
    for batch in batches:
        if batch.table not in ELIGIBILITY_PROJECTION_TABLES:
            continue
        for record in batch.unique_records:
            if str(record.get("eligibility_generation_sha256") or "") != generation:
                continue
            key = str(record.get("idempotency_key") or "")
            digest = str(record.get("record_sha256") or "")
            if not key or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"unsealable eligibility record in {batch.table}")
            identity = (batch.table, key)
            previous = rows.setdefault(identity, digest)
            if previous != digest:
                raise ValueError(f"contradictory seal row for {batch.table}:{key}")
    ordered = sorted(
        (table, key, digest) for (table, key), digest in rows.items()
    )
    def pack(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return str(len(encoded)).encode("ascii") + b":" + encoded

    packed_rows = b"".join(pack(value) for row in ordered for value in row)
    payload = (
        b"eligibility-projection-seal-v1\n"
        + str(len(ordered)).encode("ascii")
        + b"\n"
        + packed_rows
    )
    return ProjectionSeal(sha256(payload).hexdigest(), len(ordered))


def eligibility_generation_sha256(source_manifest: Mapping[str, Any]) -> str:
    return content_sha256({
        "contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "source_manifest": dict(source_manifest),
    })


def eligibility_generation(
    *,
    generation_sequence: int,
    effective_at: datetime,
    source_manifest: Mapping[str, Any],
    expected_projection_seal_sha256: str,
    expected_projection_row_count: int,
) -> RecordBatch:
    if int(generation_sequence) <= 0:
        raise ValueError("generation_sequence must be positive")
    manifest = dict(source_manifest)
    generation_sha256 = eligibility_generation_sha256(manifest)
    expected_seal = _generation_sha(expected_projection_seal_sha256)
    expected_count = int(expected_projection_row_count)
    if expected_count <= 0:
        raise ValueError("eligibility generation projection must be nonempty")
    key = f"eligibility_generation:{generation_sha256}"
    return RecordBatch.from_records("ops.eligibility_generations", [{
        "eligibility_generation_id": _uuid("eligibility_generation", key),
        "idempotency_key": key,
        "generation_sequence": int(generation_sequence),
        "generation_sha256": generation_sha256,
        "contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "effective_at": _utc(effective_at, "effective_at"),
        "source_manifest": canonical_json(manifest),
        "expected_projection_seal_sha256": expected_seal,
        "expected_projection_row_count": expected_count,
    }])


def eligibility_generation_status_event(
    *,
    generation_sha256: str,
    generation_sequence: int,
    status: str,
    effective_at: datetime,
    reviewed_by: str,
    reason: str,
    projection_seal_sha256: str | None = None,
    projection_row_count: int | None = None,
) -> RecordBatch:
    generation = _generation_sha(generation_sha256)
    sequence = int(generation_sequence)
    if sequence <= 0:
        raise ValueError("generation_sequence must be positive")
    state = str(status or "").strip().lower()
    if state not in {"candidate", "accepted", "rejected", "retired"}:
        raise ValueError(f"unknown generation status: {state}")
    reviewer = str(reviewed_by or "").strip()
    explanation = str(reason or "").strip()
    if not reviewer or not explanation:
        raise ValueError("reviewed_by and reason are required")
    seal = str(projection_seal_sha256 or "").strip().lower() or None
    count = None if projection_row_count is None else int(projection_row_count)
    if state in {"candidate", "accepted"}:
        if seal is None or not SHA256_RE.fullmatch(seal):
            raise ValueError(f"{state} status requires projection_seal_sha256")
        if count is None or count <= 0:
            raise ValueError(f"{state} status requires positive projection_row_count")
    elif seal is not None or count is not None:
        raise ValueError("only candidate/accepted status may carry a projection seal")
    effective = _utc(effective_at, "effective_at")
    key = deterministic_key(
        "eligibility_generation_status", generation, sequence, state,
        effective, reviewer, explanation, seal, count,
    )
    return RecordBatch.from_records(
        "ops.eligibility_generation_status_events", [{
            "eligibility_generation_status_event_id": _uuid(
                "eligibility_generation_status_event", key,
            ),
            "idempotency_key": key,
            "eligibility_generation_sha256": generation,
            "generation_sequence": sequence,
            "status": state,
            "effective_at": effective,
            "reviewed_by": reviewer,
            "reason": explanation,
            "projection_seal_sha256": seal,
            "projection_row_count": count,
        }],
    )


def player_entity(
    *,
    generation_sha256: str,
    legacy_player_id: int,
    canonical_name: str,
) -> RecordBatch:
    generation = _generation_sha(generation_sha256)
    player_id = int(legacy_player_id)
    if player_id <= 0:
        raise ValueError("legacy_player_id must be positive")
    name = str(canonical_name or "").strip()
    norm = normalize_identity_name(name)
    key = deterministic_key("player_entity", generation, player_id, norm)
    return RecordBatch.from_records("ops.player_entities", [{
        "player_entity_id": _uuid("player_entity", key),
        "idempotency_key": key,
        "eligibility_generation_sha256": generation,
        "legacy_player_id": player_id,
        "canonical_name": name,
        "canonical_name_norm": norm,
    }])


def player_identity_observation(
    *,
    generation_sha256: str,
    canonical_player_id: int,
    observed_name: str,
    source_player_key: str,
    source: EvidenceSource,
) -> RecordBatch:
    player_id = int(canonical_player_id)
    if player_id <= 0:
        raise ValueError("canonical_player_id must be positive")
    name = str(observed_name or "").strip()
    norm = normalize_identity_name(name)
    external_key = str(source_player_key or "").strip()
    if not external_key:
        raise ValueError("source_player_key is required")
    generation = _generation_sha(generation_sha256)
    key = deterministic_key(
        "player_identity", generation, source.source_name, external_key,
        player_id, norm, source.observed_at, source.source_content_sha256,
    )
    return RecordBatch.from_records("ops.player_identity_observations", [{
        "player_identity_observation_id": _uuid("player_identity_observation", key),
        "idempotency_key": key,
        "eligibility_generation_sha256": generation,
        "canonical_player_id": player_id,
        "observed_name": name,
        "observed_name_norm": norm,
        "source_player_key": external_key,
        **_common_evidence_columns(source),
    }])


def player_alias_observation(
    *,
    generation_sha256: str,
    canonical_player_id: int,
    alias: str,
    source: EvidenceSource,
) -> RecordBatch:
    player_id = int(canonical_player_id)
    if player_id <= 0:
        raise ValueError("canonical_player_id must be positive")
    alias_text = str(alias or "").strip()
    alias_norm = normalize_identity_name(alias_text)
    generation = _generation_sha(generation_sha256)
    key = deterministic_key(
        "player_alias", generation, source.source_name, player_id, alias_norm,
        source.observed_at, source.source_content_sha256,
    )
    return RecordBatch.from_records("ops.player_alias_observations", [{
        "player_alias_observation_id": _uuid("player_alias_observation", key),
        "idempotency_key": key,
        "eligibility_generation_sha256": generation,
        "canonical_player_id": player_id,
        "alias": alias_text,
        "alias_norm": alias_norm,
        **_common_evidence_columns(source),
    }])


def player_profile_observation(
    *,
    generation_sha256: str,
    canonical_player_id: int,
    field_name: str,
    field_value: Any,
    source: EvidenceSource,
) -> RecordBatch:
    player_id = int(canonical_player_id)
    if player_id <= 0:
        raise ValueError("canonical_player_id must be positive")
    field = str(field_name or "").strip()
    value = normalize_profile_value(field, field_value)
    generation = _generation_sha(generation_sha256)
    key = deterministic_key(
        "player_profile", generation, source.source_name, player_id, field,
        value, source.observed_at, source.source_content_sha256,
    )
    typed_values: dict[str, Any] = {
        "height_cm": None,
        "hand": None,
        "country": None,
        "birthdate": None,
        "ta_slug": None,
        "atp_url": None,
    }
    if field == "height_cm":
        typed_values[field] = Decimal(value)
    elif field == "birthdate":
        typed_values[field] = date.fromisoformat(value)
    else:
        typed_values[field] = value
    return RecordBatch.from_records("ops.player_profile_observations", [{
        "player_profile_observation_id": _uuid("player_profile_observation", key),
        "idempotency_key": key,
        "eligibility_generation_sha256": generation,
        "canonical_player_id": player_id,
        "field_name": field,
        **typed_values,
        **_common_evidence_columns(source),
    }])


def eligibility_review_event(
    *,
    generation_sha256: str,
    target_table: str,
    target_idempotency_key: str,
    review_state: str,
    reviewed_at: datetime,
    reviewed_by: str,
    reason: str,
) -> RecordBatch:
    allowed_tables = {
        "ops.player_identity_observations",
        "ops.player_alias_observations",
        "ops.player_profile_observations",
        "ops.eligibility_match_round_observations",
    }
    if target_table not in allowed_tables:
        raise ValueError(f"unsupported review target: {target_table}")
    state = str(review_state or "").strip().lower()
    if state not in {"accepted", "rejected", "quarantined", "superseded"}:
        raise ValueError(f"unsupported review_state: {state}")
    target_key = str(target_idempotency_key or "").strip()
    reviewer = str(reviewed_by or "").strip()
    explanation = str(reason or "").strip()
    if not target_key or not reviewer or not explanation:
        raise ValueError("review target, reviewer, and reason are required")
    generation = _generation_sha(generation_sha256)
    reviewed = _utc(reviewed_at, "reviewed_at")
    key = deterministic_key(
        "eligibility_review", generation, target_table, target_key, state,
        reviewed, reviewer, explanation,
    )
    return RecordBatch.from_records("ops.eligibility_review_events", [{
        "eligibility_review_event_id": _uuid("eligibility_review_event", key),
        "idempotency_key": key,
        "eligibility_generation_sha256": generation,
        "target_table": target_table,
        "target_idempotency_key": target_key,
        "review_state": state,
        "reviewed_at": reviewed,
        "reviewed_by": reviewer,
        "reason": explanation,
    }])


@dataclass(frozen=True)
class CompatibilityProjection:
    """Read-only projection plan from the current public compatibility rows."""

    batches: tuple[RecordBatch, ...]

    @property
    def row_counts(self) -> dict[str, int]:
        return {
            batch.table: len(batch.unique_records)
            for batch in self.batches
        }


def project_compatibility_rows(
    *,
    generation_sha256: str,
    players: Iterable[Mapping[str, Any]],
    aliases: Iterable[Mapping[str, Any]],
    source: EvidenceSource,
) -> CompatibilityProjection:
    """Convert exact public projection rows into unmodified evidence batches.

    The adapter never invents a height, alias, external id, review decision, or
    source location.  Invalid values raise and quarantine the whole plan for
    operator review instead of being silently defaulted.
    """
    identity_rows: list[Mapping[str, Any]] = []
    entity_rows: list[Mapping[str, Any]] = []
    alias_rows: list[Mapping[str, Any]] = []
    profile_rows: list[Mapping[str, Any]] = []
    for row in players:
        player_id = int(row["player_id"])
        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError(f"player {player_id} has no canonical name")
        entity_rows.extend(player_entity(
            generation_sha256=generation_sha256,
            legacy_player_id=player_id,
            canonical_name=name,
        ).records)
        identity_rows.extend(player_identity_observation(
            generation_sha256=generation_sha256,
            canonical_player_id=player_id,
            observed_name=name,
            source_player_key=str(player_id),
            source=source,
        ).records)
        for field in PROFILE_FIELDS:
            value = row.get(field)
            if value is None or str(value).strip() in {"", "U"}:
                continue
            profile_rows.extend(player_profile_observation(
                generation_sha256=generation_sha256,
                canonical_player_id=player_id,
                field_name=field,
                field_value=value,
                source=source,
            ).records)
    for row in aliases:
        alias_text = row.get("alias") or row.get("alias_norm")
        alias_rows.extend(player_alias_observation(
            generation_sha256=generation_sha256,
            canonical_player_id=int(row["player_id"]),
            alias=str(alias_text or ""),
            source=source,
        ).records)
    return CompatibilityProjection((
        RecordBatch.from_records("ops.player_entities", entity_rows),
        RecordBatch.from_records("ops.player_identity_observations", identity_rows),
        RecordBatch.from_records("ops.player_alias_observations", alias_rows),
        RecordBatch.from_records("ops.player_profile_observations", profile_rows),
    ))


class Cursor(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any: ...
    def fetchone(self) -> tuple[Any, ...] | None: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...
    def close(self) -> Any: ...


class Connection(Protocol):
    def cursor(self) -> Cursor: ...


class EligibilityProjectionReader:
    """Generation-pinned, read-only resolver for accepted API projections."""

    def __init__(
        self,
        connection: Connection,
        *,
        expected_generation_sha256: str,
        expected_projection_seal_sha256: str,
    ):
        self.connection = connection
        self.expected_generation_sha256 = _generation_sha(expected_generation_sha256)
        self.expected_projection_seal_sha256 = _generation_sha(
            expected_projection_seal_sha256,
        )
        self._projection_row_count: int | None = None
        self._verified = False

    @property
    def projection_row_count(self) -> int:
        self._ensure_verified()
        assert self._projection_row_count is not None
        return self._projection_row_count

    def verify_contract(self) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """SELECT version FROM ops.schema_versions
                   ORDER BY string_to_array(version, '.')::integer[] DESC
                   LIMIT 1"""
            )
            version_row = cursor.fetchone()
            actual_version = None if version_row is None else str(version_row[0])
            if actual_version != OPERATIONAL_SCHEMA_VERSION:
                raise EligibilityContractError(
                    "eligibility reads require operational schema "
                    f"{OPERATIONAL_SCHEMA_VERSION}; found {actual_version or '<missing>'}"
                )
            cursor.execute(
                """SELECT generation_sha256, contract_version,
                          projection_seal_sha256, projection_row_count
                     FROM api.current_eligibility_generation"""
            )
            generations = cursor.fetchall()
        finally:
            cursor.close()
        if len(generations) != 1:
            raise EligibilityContractError(
                f"expected exactly one accepted eligibility generation; found {len(generations)}"
            )
        generation, contract, seal, row_count = generations[0]
        generation, contract, seal = str(generation), str(contract), str(seal)
        if generation != self.expected_generation_sha256:
            raise EligibilityContractError(
                "accepted eligibility generation does not match configured generation"
            )
        if contract != ELIGIBILITY_CONTRACT_VERSION:
            raise EligibilityContractError(
                f"eligibility contract mismatch: required {ELIGIBILITY_CONTRACT_VERSION}, found {contract}"
            )
        if seal != self.expected_projection_seal_sha256:
            raise EligibilityContractError(
                "accepted eligibility projection seal does not match configured seal"
            )
        if row_count is None or int(row_count) <= 0:
            raise EligibilityContractError("accepted eligibility projection row count is invalid")
        self._projection_row_count = int(row_count)
        self._verified = True

    def _ensure_verified(self) -> None:
        if not self._verified:
            self.verify_contract()

    def _has_conflict(self, evidence_kind: str, subject_key: str) -> bool:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """SELECT 1 FROM api.eligibility_conflicts
                    WHERE eligibility_generation_sha256 = %s
                      AND evidence_kind = %s AND subject_key = %s
                    LIMIT 1""",
                (self.expected_generation_sha256, evidence_kind, subject_key),
            )
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def _has_any_conflict(
        self, evidence_kinds: Iterable[str], subject_key: str,
    ) -> bool:
        return any(
            self._has_conflict(kind, subject_key) for kind in evidence_kinds
        )

    def resolve_player_id(self, name: str) -> Resolution:
        self._ensure_verified()
        norm = normalize_identity_name(name)
        if self._has_any_conflict(
            (
                "player_identity", "player_alias", "player_source_identity",
                "player_name_binding",
            ),
            norm,
        ):
            return Resolution(
                ResolutionStatus.CONFLICT,
                detail="accepted unified identity evidence disagrees",
                generation_sha256=self.expected_generation_sha256,
                projection_seal_sha256=self.expected_projection_seal_sha256,
            )
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """SELECT canonical_player_id, valid_until
                     FROM api.current_player_name_bindings
                    WHERE eligibility_generation_sha256 = %s
                      AND player_name_norm = %s""",
                (self.expected_generation_sha256, norm),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        if len(rows) == 1:
            player_id, valid_until = rows[0]
            return Resolution(
                ResolutionStatus.RESOLVED, int(player_id),
                generation_sha256=self.expected_generation_sha256,
                projection_seal_sha256=self.expected_projection_seal_sha256,
                expires_at=valid_until,
            )
        if len(rows) > 1:
            return Resolution(
                ResolutionStatus.CONFLICT,
                detail="accepted identity evidence disagrees",
                generation_sha256=self.expected_generation_sha256,
                projection_seal_sha256=self.expected_projection_seal_sha256,
            )
        return Resolution(
            ResolutionStatus.MISSING,
            detail="no accepted identity or alias evidence",
            generation_sha256=self.expected_generation_sha256,
            projection_seal_sha256=self.expected_projection_seal_sha256,
        )

    def resolve_profile_field(self, canonical_player_id: int, field_name: str) -> Resolution:
        self._ensure_verified()
        player_id = int(canonical_player_id)
        field = str(field_name or "").strip()
        if field not in PROFILE_FIELDS:
            raise ValueError(f"unsupported profile field: {field}")
        subject = f"{player_id}:{field}"
        if self._has_conflict("player_profile", subject):
            return Resolution(
                ResolutionStatus.CONFLICT,
                detail="accepted profile evidence disagrees",
                generation_sha256=self.expected_generation_sha256,
                projection_seal_sha256=self.expected_projection_seal_sha256,
            )
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """SELECT field_value, last_observed_at, valid_until
                     FROM api.current_player_profiles
                    WHERE eligibility_generation_sha256 = %s
                      AND canonical_player_id = %s AND field_name = %s""",
                (self.expected_generation_sha256, player_id, field),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        if len(rows) != 1:
            status = ResolutionStatus.CONFLICT if len(rows) > 1 else ResolutionStatus.MISSING
            return Resolution(
                status,
                detail="profile projection is not exactly one value",
                generation_sha256=self.expected_generation_sha256,
                projection_seal_sha256=self.expected_projection_seal_sha256,
            )
        raw, observed_at, valid_until = rows[0]
        value: Any = normalize_profile_value(field, raw)
        if field == "height_cm":
            value = float(Decimal(value))
        return Resolution(
            ResolutionStatus.RESOLVED,
            value,
            generation_sha256=self.expected_generation_sha256,
            projection_seal_sha256=self.expected_projection_seal_sha256,
            observed_at=observed_at,
            expires_at=valid_until,
        )

    def resolve_round(self, match_anchor_key: str) -> Resolution:
        self._ensure_verified()
        anchor = str(match_anchor_key or "").strip()
        if not anchor:
            raise ValueError("match_anchor_key is required")
        if self._has_conflict("match_round", anchor):
            return Resolution(
                ResolutionStatus.CONFLICT,
                detail="accepted round evidence disagrees",
                generation_sha256=self.expected_generation_sha256,
                projection_seal_sha256=self.expected_projection_seal_sha256,
            )
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """SELECT round_code, last_observed_at, expires_at, confidence
                     FROM api.current_match_rounds
                    WHERE eligibility_generation_sha256 = %s
                      AND match_anchor_key = %s""",
                (self.expected_generation_sha256, anchor),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        if len(rows) != 1:
            status = ResolutionStatus.CONFLICT if len(rows) > 1 else ResolutionStatus.MISSING
            return Resolution(
                status,
                detail="round projection is not exactly one active value",
                generation_sha256=self.expected_generation_sha256,
                projection_seal_sha256=self.expected_projection_seal_sha256,
            )
        round_code, observed_at, expires_at, confidence = rows[0]
        return Resolution(
            ResolutionStatus.RESOLVED,
            normalize_round_code(round_code),
            generation_sha256=self.expected_generation_sha256,
            projection_seal_sha256=self.expected_projection_seal_sha256,
            observed_at=observed_at,
            expires_at=expires_at,
            confidence=Decimal(str(confidence)),
        )
