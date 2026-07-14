"""Durable, short-lived read bundle for accepted eligibility profiles.

The bundle is a derived read model, never authority. Its local hash checks
assume a trusted filesystem; a short validity window bounds how long a bundle
can remain readable after its database generation is retired.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping

try:
    from storage.eligibility import (
        ELIGIBILITY_CONTRACT_VERSION, normalize_identity_name,
        normalize_profile_value,
    )
    from versioning import OPERATIONAL_SCHEMA_VERSION
except ImportError:  # pragma: no cover - package-style execution
    from production.storage.eligibility import (  # type: ignore
        ELIGIBILITY_CONTRACT_VERSION, normalize_identity_name,
        normalize_profile_value,
    )
    from production.versioning import OPERATIONAL_SCHEMA_VERSION  # type: ignore


BUNDLE_NAME = "eligibility_profiles_bundle.json"
MANIFEST_NAME = "eligibility_cache_manifest.json"
LOCK_NAME = ".eligibility_cache.lock"
DEFAULT_MAX_AGE = timedelta(minutes=15)
MAX_MAX_AGE = timedelta(minutes=15)
AUTHORITY = "ops_accepted_eligibility_projection"


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_time(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return _utc(value, field)
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    return _utc(parsed, field)


def _sha(value: Any, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return digest


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class ProfileBundleData:
    entries: Mapping[str, Mapping[str, Any]]
    evidence_valid_until: datetime | None


@dataclass(frozen=True)
class VerifiedEligibilityBundle:
    generation_sha256: str
    projection_seal_sha256: str
    projection_row_count: int
    valid_until: datetime
    entries: Mapping[str, Mapping[str, Any]]

    def profile_for(self, player_name: str) -> Mapping[str, Any] | None:
        try:
            key = normalize_identity_name(player_name)
        except ValueError:
            return None
        return self.entries.get(key)


def build_profile_bundle(
    rows: Iterable[Mapping[str, Any]],
) -> ProfileBundleData:
    """Build unique normalized-name -> canonical-ID profile entries."""
    entries: dict[str, dict[str, Any]] = {}
    expiries: list[datetime] = []
    for row in rows:
        key = normalize_identity_name(str(row.get("player_name_norm") or ""))
        player_id = int(row["canonical_player_id"])
        if player_id <= 0:
            raise ValueError("canonical_player_id must be positive")
        field = str(row.get("field_name") or "").strip()
        if field not in {"height_cm", "hand"}:
            continue
        value = normalize_profile_value(field, row.get("field_value"))
        if field == "height_cm":
            number = float(value)
            value = int(number) if number.is_integer() else number

        entry = entries.setdefault(key, {"canonical_player_id": player_id})
        if int(entry["canonical_player_id"]) != player_id:
            raise ValueError(f"ambiguous canonical identity for bundle key: {key}")
        if field in entry and entry[field] != value:
            raise ValueError(f"conflicting {field} for bundle key: {key}")
        entry[field] = value

        for expiry_field in ("binding_valid_until", "profile_valid_until"):
            expiry = row.get(expiry_field)
            if expiry is not None and str(expiry).strip():
                expiries.append(_parse_time(expiry, expiry_field))

    return ProfileBundleData(
        entries={key: entries[key] for key in sorted(entries)},
        evidence_valid_until=min(expiries) if expiries else None,
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _exclusive_export_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_NAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def write_profile_bundle_export(
    *,
    output_dir: Path,
    generation_sha256: str,
    projection_seal_sha256: str,
    projection_row_count: int,
    data: ProfileBundleData,
    exported_at: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> dict[str, Any]:
    generation = _sha(generation_sha256, "generation_sha256")
    seal = _sha(projection_seal_sha256, "projection_seal_sha256")
    row_count = int(projection_row_count)
    if row_count <= 0:
        raise ValueError("projection_row_count must be positive")
    exported = _utc(exported_at or datetime.now(timezone.utc), "exported_at")
    if max_age <= timedelta(0):
        raise ValueError("max_age must be positive")
    if max_age > MAX_MAX_AGE:
        raise ValueError("max_age exceeds the 15-minute retirement bound")
    valid_until = exported + max_age
    if data.evidence_valid_until is not None:
        valid_until = min(valid_until, data.evidence_valid_until)
    if valid_until <= exported:
        raise ValueError("accepted evidence expires before bundle publication")

    entries = {key: dict(value) for key, value in data.entries.items()}
    bundle = {
        "authority": AUTHORITY,
        "operational_schema_version": OPERATIONAL_SCHEMA_VERSION,
        "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "generation_sha256": generation,
        "projection_seal_sha256": seal,
        "projection_row_count": row_count,
        "exported_at": exported.isoformat().replace("+00:00", "Z"),
        "valid_until": valid_until.isoformat().replace("+00:00", "Z"),
        "entries": entries,
    }
    bundle_bytes = _json_bytes(bundle)
    counts = {
        "entries": len(entries),
        "height_cm": sum("height_cm" in entry for entry in entries.values()),
        "hand": sum("hand" in entry for entry in entries.values()),
    }
    manifest = {
        key: bundle[key]
        for key in (
            "authority", "operational_schema_version",
            "eligibility_contract_version", "generation_sha256",
            "projection_seal_sha256", "projection_row_count",
            "exported_at", "valid_until",
        )
    }
    manifest.update({
        "bundle": {
            "name": BUNDLE_NAME,
            "sha256": sha256(bundle_bytes).hexdigest(),
            "counts": counts,
        },
    })
    with _exclusive_export_lock(output_dir):
        # The manifest is the acceptance marker and is durably replaced last.
        _durable_atomic_write(output_dir / BUNDLE_NAME, bundle_bytes)
        _durable_atomic_write(output_dir / MANIFEST_NAME, _json_bytes(manifest))
    return manifest


def load_verified_profile_bundle(
    *,
    output_dir: Path,
    expected_generation_sha256: str,
    expected_projection_seal_sha256: str,
    now: datetime | None = None,
) -> VerifiedEligibilityBundle | None:
    """Return a whole verified bundle; any mismatch invalidates every field."""
    try:
        generation = _sha(expected_generation_sha256, "generation_sha256")
        seal = _sha(expected_projection_seal_sha256, "projection_seal_sha256")
        manifest_path = output_dir / MANIFEST_NAME
        bundle_path = output_dir / BUNDLE_NAME
        if not manifest_path.exists() or not bundle_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_bytes = bundle_path.read_bytes()
        bundle_meta = manifest.get("bundle") or {}
        if bundle_meta.get("name") != BUNDLE_NAME:
            return None
        if sha256(bundle_bytes).hexdigest() != bundle_meta.get("sha256"):
            return None
        bundle = json.loads(bundle_bytes.decode("utf-8"))
        for payload in (manifest, bundle):
            if payload.get("authority") != AUTHORITY:
                return None
            if payload.get("operational_schema_version") != OPERATIONAL_SCHEMA_VERSION:
                return None
            if payload.get("eligibility_contract_version") != ELIGIBILITY_CONTRACT_VERSION:
                return None
            if payload.get("generation_sha256") != generation:
                return None
            if payload.get("projection_seal_sha256") != seal:
                return None
        if any(
            manifest.get(field) != bundle.get(field)
            for field in (
                "projection_row_count", "exported_at", "valid_until",
            )
        ):
            return None
        raw_row_count = bundle.get("projection_row_count")
        if isinstance(raw_row_count, bool):
            return None
        row_count = int(raw_row_count)
        if row_count <= 0:
            return None
        current = _utc(now or datetime.now(timezone.utc), "now")
        exported_at = _parse_time(bundle.get("exported_at"), "exported_at")
        valid_until = _parse_time(bundle.get("valid_until"), "valid_until")
        if not exported_at <= current < valid_until:
            return None
        if valid_until - exported_at > MAX_MAX_AGE:
            return None
        entries = bundle.get("entries")
        if not isinstance(entries, dict):
            return None
        validated: dict[str, dict[str, Any]] = {}
        for key, raw_entry in entries.items():
            if normalize_identity_name(key) != key or not isinstance(raw_entry, dict):
                return None
            player_id = int(raw_entry.get("canonical_player_id"))
            if player_id <= 0:
                return None
            entry: dict[str, Any] = {"canonical_player_id": player_id}
            for field in ("height_cm", "hand"):
                if field not in raw_entry:
                    continue
                value = normalize_profile_value(field, raw_entry[field])
                if field == "height_cm":
                    number = float(value)
                    value = int(number) if number.is_integer() else number
                entry[field] = value
            if set(raw_entry) - {"canonical_player_id", "height_cm", "hand"}:
                return None
            validated[key] = entry
        counts = bundle_meta.get("counts") or {}
        if counts != {
            "entries": len(validated),
            "height_cm": sum("height_cm" in entry for entry in validated.values()),
            "hand": sum("hand" in entry for entry in validated.values()),
        }:
            return None
        return VerifiedEligibilityBundle(
            generation, seal, row_count, valid_until, validated,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
