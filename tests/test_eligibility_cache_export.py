from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production.eligibility_cache import (  # noqa: E402
    BUNDLE_NAME, MANIFEST_NAME, build_profile_bundle,
    load_verified_profile_bundle, write_profile_bundle_export,
)
from production.operations.export_eligibility_cache import _read_projection  # noqa: E402
from production.storage.eligibility import (  # noqa: E402
    ELIGIBILITY_CONTRACT_VERSION, EligibilityContractError,
)
from production.versioning import OPERATIONAL_SCHEMA_VERSION  # noqa: E402


GENERATION = "b" * 64
SEAL = "c" * 64
EXPORTED_AT = datetime(2026, 7, 14, 21, tzinfo=timezone.utc)


def _rows(**overrides):
    base = {
        "player_name_norm": "reneeexample",
        "canonical_player_id": 123,
        "field_name": "height_cm",
        "field_value": "183.00",
        "binding_valid_until": EXPORTED_AT + timedelta(hours=2),
        "profile_valid_until": EXPORTED_AT + timedelta(hours=1),
    }
    base.update(overrides)
    return [base]


def _write(tmp_path, rows=None, **overrides):
    values = {
        "output_dir": tmp_path,
        "generation_sha256": GENERATION,
        "projection_seal_sha256": SEAL,
        "projection_row_count": 7,
        "data": build_profile_bundle(rows or _rows()),
        "exported_at": EXPORTED_AT,
    }
    values.update(overrides)
    return write_profile_bundle_export(**values)


def test_cache_export_is_one_id_bearing_seal_pinned_bundle(tmp_path):
    rows = _rows() + _rows(field_name="hand", field_value="A")
    manifest = _write(tmp_path, rows)
    bundle = json.loads((tmp_path / BUNDLE_NAME).read_text())
    stored_manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())

    assert manifest == stored_manifest
    assert bundle["entries"] == {
        "reneeexample": {
            "canonical_player_id": 123,
            "hand": "A",
            "height_cm": 183,
        },
    }
    assert manifest["operational_schema_version"] == OPERATIONAL_SCHEMA_VERSION
    assert manifest["generation_sha256"] == GENERATION
    assert manifest["projection_seal_sha256"] == SEAL
    assert manifest["projection_row_count"] == 7
    assert manifest["bundle"]["counts"] == {
        "entries": 1, "hand": 1, "height_cm": 1,
    }
    assert not list(tmp_path.glob("*.tmp"))

    verified = load_verified_profile_bundle(
        output_dir=tmp_path,
        expected_generation_sha256=GENERATION,
        expected_projection_seal_sha256=SEAL,
        now=EXPORTED_AT + timedelta(minutes=1),
    )
    assert verified is not None
    assert verified.profile_for("Renée Example")["canonical_player_id"] == 123
    assert verified.profile_for("Renée Example")["height_cm"] == 183


def test_bundle_aborts_on_normalized_name_to_multiple_ids():
    with pytest.raises(ValueError, match="ambiguous canonical identity"):
        build_profile_bundle(
            _rows() + _rows(canonical_player_id=456, field_name="hand", field_value="L")
        )


@pytest.mark.parametrize("field,value", [
    ("height_cm", 149),
    ("height_cm", 231),
    ("hand", "U"),
])
def test_bundle_revalidates_typed_projection_values(field, value):
    with pytest.raises(ValueError):
        build_profile_bundle(_rows(field_name=field, field_value=value))


def test_bundle_is_all_or_nothing_for_seal_hash_and_expiry(tmp_path):
    _write(tmp_path)
    wrong_seal = load_verified_profile_bundle(
        output_dir=tmp_path,
        expected_generation_sha256=GENERATION,
        expected_projection_seal_sha256="d" * 64,
        now=EXPORTED_AT + timedelta(minutes=1),
    )
    assert wrong_seal is None

    expired = load_verified_profile_bundle(
        output_dir=tmp_path,
        expected_generation_sha256=GENERATION,
        expected_projection_seal_sha256=SEAL,
        now=EXPORTED_AT + timedelta(minutes=15),
    )
    assert expired is None

    bundle_path = tmp_path / BUNDLE_NAME
    bundle_path.write_bytes(bundle_path.read_bytes() + b" ")
    tampered = load_verified_profile_bundle(
        output_dir=tmp_path,
        expected_generation_sha256=GENERATION,
        expected_projection_seal_sha256=SEAL,
        now=EXPORTED_AT + timedelta(minutes=1),
    )
    assert tampered is None


def test_bundle_validity_is_bounded_by_earliest_evidence_and_hard_ttl(tmp_path):
    evidence_expiry = EXPORTED_AT + timedelta(minutes=3)
    manifest = _write(
        tmp_path,
        _rows(profile_valid_until=evidence_expiry),
    )
    assert manifest["valid_until"] == evidence_expiry.isoformat().replace("+00:00", "Z")
    with pytest.raises(ValueError, match="15-minute"):
        _write(tmp_path, max_age=timedelta(minutes=16))


class _Cursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def execute(self, query, params=()):
        self.connection.queries.append((query, params))
        if "ops.schema_versions" in query:
            self.rows = [(OPERATIONAL_SCHEMA_VERSION,)]
        elif "current_eligibility_generation" in query:
            self.rows = [(GENERATION, ELIGIBILITY_CONTRACT_VERSION, SEAL, 7)]
        elif "api.eligibility_conflicts" in query:
            self.rows = [("player_name_binding", "ambiguousname")]
        elif "current_player_name_bindings" in query:
            raise AssertionError("profile rows must not be read after a name conflict")

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        return None


class _Connection:
    def __init__(self):
        self.queries = []

    def cursor(self):
        return _Cursor(self)


def test_export_aborts_on_name_conflicts_before_reading_profile_values():
    with pytest.raises(EligibilityContractError, match="ambiguous"):
        _read_projection(
            _Connection(),
            generation_sha256=GENERATION,
            projection_seal_sha256=SEAL,
        )
