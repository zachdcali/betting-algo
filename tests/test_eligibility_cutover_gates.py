from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))
sys.path.insert(0, str(PRODUCTION / "scraping"))

from features import ta_feature_calculator as ta_feature_module  # noqa: E402
from features.ta_feature_calculator import (  # noqa: E402
    TAFeatureCalculator, UnsafeToInferError,
)
from scraping import atp_height_scraper as scraper  # noqa: E402
from storage.eligibility import (  # noqa: E402
    EligibilityContractError,
)
from eligibility_cache import (  # noqa: E402
    VerifiedEligibilityBundle, build_profile_bundle, write_profile_bundle_export,
)


GENERATION = "b" * 64
SEAL = "c" * 64
NOW = datetime(2026, 7, 14, 21, tzinfo=timezone.utc)


def _point_cache_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(scraper, "CACHE_PATH", tmp_path / "atp_heights.json")
    monkeypatch.setattr(scraper, "HANDS_CACHE_PATH", tmp_path / "atp_hands.json")
    monkeypatch.setattr(
        scraper, "CACHE_MANIFEST_PATH", tmp_path / "eligibility_cache_manifest.json",
    )


def test_default_legacy_cache_read_and_write_behavior_is_unchanged(monkeypatch, tmp_path):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_GENERATION_SHA256", raising=False)
    legacy = {"legacy player": 183, "missing player": None}
    expected_bytes = json.dumps(legacy, indent=2).encode("utf-8")
    scraper.CACHE_PATH.write_bytes(expected_bytes)

    assert scraper._load_cache() == legacy
    scraper._save_cache(legacy)
    assert scraper.CACHE_PATH.read_bytes() == expected_bytes


def test_required_mode_fails_closed_without_manifest_and_never_scrapes_or_writes(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "required")
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_GENERATION_SHA256", GENERATION)
    monkeypatch.setenv("ELIGIBILITY_PROJECTION_SEAL_SHA256", SEAL)
    legacy_bytes = json.dumps({"legacy player": 183}, indent=2).encode("utf-8")
    scraper.CACHE_PATH.write_bytes(legacy_bytes)
    scraper.HANDS_CACHE_PATH.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        scraper, "_load_url_map",
        lambda: (_ for _ in ()).throw(AssertionError("required mode must not scrape")),
    )

    assert scraper._load_cache() == {}
    assert scraper.get_height_cm("Unreviewed Player") is None
    assert scraper.batch_get_profiles(["Unreviewed Player"], verbose=False) == {
        "Unreviewed Player": {
            "canonical_player_id": None, "height_cm": None, "hand": None,
        },
    }
    scraper._save_cache({"new value": 190})
    assert scraper.CACHE_PATH.read_bytes() == legacy_bytes
    with pytest.raises(EligibilityContractError, match="rejects caller-supplied"):
        scraper.get_height_cm("Unreviewed Player", cache={"unreviewed player": 190})
    forged = VerifiedEligibilityBundle(
        GENERATION, SEAL, 1, NOW + timedelta(days=1),
        {"unreviewedplayer": {"canonical_player_id": 999, "height_cm": 190}},
    )
    with pytest.raises(EligibilityContractError, match="rejects caller-supplied"):
        scraper.get_height_cm("Unreviewed Player", cache=forged)


def _write_bundle(tmp_path, *, exported_at=NOW):
    data = build_profile_bundle([
        {
            "player_name_norm": "acceptedplayer",
            "canonical_player_id": 42,
            "field_name": "height_cm",
            "field_value": 180,
            "binding_valid_until": NOW + timedelta(hours=1),
            "profile_valid_until": NOW + timedelta(hours=1),
        },
        {
            "player_name_norm": "acceptedplayer",
            "canonical_player_id": 42,
            "field_name": "hand",
            "field_value": "L",
            "binding_valid_until": NOW + timedelta(hours=1),
            "profile_valid_until": NOW + timedelta(hours=1),
        },
    ])
    write_profile_bundle_export(
        output_dir=tmp_path,
        generation_sha256=GENERATION,
        projection_seal_sha256=SEAL,
        projection_row_count=9,
        data=data,
        exported_at=exported_at,
    )


def test_required_cache_accepts_only_whole_exact_generation_and_seal_bundle(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "required")
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_GENERATION_SHA256", GENERATION)
    monkeypatch.setenv("ELIGIBILITY_PROJECTION_SEAL_SHA256", SEAL)
    monkeypatch.setattr(
        scraper, "load_verified_profile_bundle",
        lambda **kwargs: __import__("eligibility_cache").load_verified_profile_bundle(
            **kwargs, now=NOW + timedelta(minutes=1),
        ),
    )
    _write_bundle(tmp_path)

    assert scraper._load_cache() == {}
    assert scraper.get_height_cm("Accepted Player") == 180
    assert scraper.batch_get_profiles(["Accepted Player"], verbose=False) == {
        "Accepted Player": {
            "canonical_player_id": 42, "height_cm": 180, "hand": "L",
        },
    }


def test_required_cache_rejects_wrong_seal_and_expired_bundle(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "required")
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_GENERATION_SHA256", GENERATION)
    _write_bundle(tmp_path)
    monkeypatch.setenv("ELIGIBILITY_PROJECTION_SEAL_SHA256", "d" * 64)
    assert scraper.get_height_cm("Accepted Player") is None

    monkeypatch.setenv("ELIGIBILITY_PROJECTION_SEAL_SHA256", SEAL)
    monkeypatch.setattr(
        scraper, "load_verified_profile_bundle",
        lambda **kwargs: __import__("eligibility_cache").load_verified_profile_bundle(
            **kwargs, now=NOW + timedelta(minutes=16),
        ),
    )
    assert scraper.batch_get_profiles(["Accepted Player"], verbose=False) == {
        "Accepted Player": {
            "canonical_player_id": None, "height_cm": None, "hand": None,
        },
    }


def test_required_feature_resolution_replaces_or_clears_populated_legacy_values(
    monkeypatch,
):
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "required")
    requested = []

    def accepted_profiles(names, verbose=False):
        requested.append((tuple(names), verbose))
        return {
            "Accepted One": {
                "canonical_player_id": 42, "height_cm": 180, "hand": "L",
            },
            "Accepted Two": {
                "canonical_player_id": 43, "height_cm": None, "hand": None,
            },
        }

    monkeypatch.setattr(
        ta_feature_module, "batch_get_profiles", accepted_profiles,
    )
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    result = calc._resolve_height_hand(
        "Accepted One", "Accepted Two",
        {"player_id": 42, "height_cm": 199, "hand": "R"},
        {"player_id": 43, "height_cm": 201, "hand": "R"},
    )
    assert requested == [(("Accepted One", "Accepted Two"), False)]
    assert result == (180, "L", None, "U")


def test_required_feature_resolution_rejects_bundle_store_identity_mismatch(
    monkeypatch,
):
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "required")
    monkeypatch.setattr(
        ta_feature_module,
        "batch_get_profiles",
        lambda *_args, **_kwargs: {
            "Wrong Binding": {
                "canonical_player_id": 999, "height_cm": 180, "hand": "L",
            },
            "Exact Binding": {
                "canonical_player_id": 43, "height_cm": 181, "hand": "R",
            },
        },
    )
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    with pytest.raises(UnsafeToInferError, match="identity mismatch"):
        calc._resolve_height_hand(
            "Wrong Binding", "Exact Binding",
            {"player_id": 42, "height_cm": 199, "hand": "R"},
            {"player_id": 43, "height_cm": 201, "hand": "L"},
        )


def test_supported_module_entrypoint_has_help_without_scraping():
    completed = subprocess.run(
        [sys.executable, "-m", "production.scraping.atp_height_scraper", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ATP player name" in completed.stdout


def test_package_entrypoint_resolves_package_browser_session_import():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys,types; "
                "m=types.ModuleType('production.scraping.browser_session'); "
                "m.new_page=lambda:'package-page'; "
                "sys.modules['production.scraping.browser_session']=m; "
                "from production.scraping import atp_height_scraper as s; "
                "assert s._new_browser_page()=='package-page'"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


class _Cursor:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.calls.append((query, params))


class _Connection:
    def __init__(self):
        self.calls = []

    @contextmanager
    def transaction(self):
        yield

    def cursor(self):
        return _Cursor(self.calls)


def test_profile_write_through_is_legacy_default_and_disabled_only_after_cutover(
    monkeypatch,
):
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    connection = _Connection()
    calc._store = lambda: connection

    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    calc._persist_player_field({"player_id": 42}, "height_cm", 183)
    assert len(connection.calls) == 1
    assert connection.calls[0][1] == (183, 42)

    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "required")
    calc._persist_player_field({"player_id": 42}, "height_cm", 184)
    assert len(connection.calls) == 1


def test_unknown_cutover_mode_fails_closed_before_cache_or_store_mutation(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "requried")
    with pytest.raises(EligibilityContractError, match="exactly legacy or required"):
        scraper._load_cache()

    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    connection = _Connection()
    calc._store = lambda: connection
    with pytest.raises(EligibilityContractError, match="exactly legacy or required"):
        calc._persist_player_field({"player_id": 42}, "height_cm", 184)
    assert connection.calls == []
