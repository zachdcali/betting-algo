from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
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
        scraper, "PROFILE_LOOKUP_META_PATH", tmp_path / "atp_profile_lookup_meta.json",
    )
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


class _ProfilePage:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_profile_batch_rotates_clean_pages_before_second_official_fetch(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setenv("ATP_PROFILE_NEGATIVE_REFRESH_LIMIT", "2")
    monkeypatch.setenv("ATP_PROFILE_PAGE_BATCH_SIZE", "1")
    names = ["First Player", "Second Player"]
    scraper.CACHE_PATH.write_text(
        json.dumps({name.lower(): None for name in names}), encoding="utf-8"
    )
    scraper.HANDS_CACHE_PATH.write_text(
        json.dumps({name.lower(): None for name in names}), encoding="utf-8"
    )
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {
            "first player": "/en/players/first/f001/overview",
            "second player": "/en/players/second/s001/overview",
        },
    )
    pages = []

    class _CleanPage(_ProfilePage):
        def __init__(self):
            super().__init__()
            self.number = len(pages) + 1
            pages.append(self)

    monkeypatch.setattr(scraper, "_new_browser_page", _CleanPage)

    def _fetch(page, url):
        if "first" in url:
            assert page.number == 1
            return "First Player Height 185cm Plays: Right-Handed"
        assert page.number == 2
        return "Second Player Height 190cm Plays: Left-Handed"

    monkeypatch.setattr(scraper, "_fetch_profile_text", _fetch)

    assert scraper.batch_get_profiles(names, verbose=False) == {
        "First Player": {"height_cm": 185, "hand": "R"},
        "Second Player": {"height_cm": 190, "hand": "L"},
    }
    assert len(pages) == 2
    assert all(page.closed for page in pages)


def test_fresh_negative_profile_cache_skips_same_source(monkeypatch, tmp_path):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    source_uri = "https://www.atptour.com/en/players/retry-player/r001/overview"
    scraper.CACHE_PATH.write_text('{"retry player": null}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"retry player": null}', encoding="utf-8")
    scraper.PROFILE_LOOKUP_META_PATH.write_text(
        json.dumps({
            "retry player": {
                "source_uri": source_uri,
                "observed_at": NOW.isoformat(),
                "status": "not_found",
                "missing_fields": ["height_cm", "hand"],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"retry player": "/en/players/retry-player/r001/overview"},
    )
    monkeypatch.setattr(
        scraper,
        "_new_browser_page",
        lambda: (_ for _ in ()).throw(AssertionError("fresh negative must not scrape")),
    )

    assert scraper.batch_get_profiles(["Retry Player"], verbose=False) == {
        "Retry Player": {"height_cm": None, "hand": None},
    }


def test_expired_negative_profile_cache_retries_and_records_source_evidence(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    source_uri = "https://www.atptour.com/en/players/retry-player/r001/overview"
    scraper.CACHE_PATH.write_text('{"retry player": null}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"retry player": null}', encoding="utf-8")
    scraper.PROFILE_LOOKUP_META_PATH.write_text(
        json.dumps({
            "retry player": {
                "source_uri": source_uri,
                "observed_at": (NOW - timedelta(days=8)).isoformat(),
                "status": "not_found",
                "missing_fields": ["height_cm", "hand"],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"retry player": "/en/players/retry-player/r001/overview"},
    )
    page = _ProfilePage()
    monkeypatch.setattr(scraper, "_new_browser_page", lambda: page)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda pg, url: "Retry Player Height 188cm Plays: Left-Handed",
    )

    assert scraper.batch_get_profiles(["Retry Player"], verbose=False) == {
        "Retry Player": {"height_cm": 188, "hand": "L"},
    }
    assert page.closed is True
    metadata = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())
    assert metadata["retry player"] == {
        "source_uri": source_uri,
        "observed_at": NOW.isoformat(timespec="seconds"),
        "status": "resolved",
        "missing_fields": [],
        "source_content_sha256": sha256(
            b"Retry Player Height 188cm Plays: Left-Handed"
        ).hexdigest(),
        "identity_binding": scraper.OFFICIAL_PAGE_IDENTITY_BINDING,
        "observed_values": {"height_cm": 188, "hand": "L"},
    }


def test_changed_official_profile_source_invalidates_fresh_negative(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    scraper.CACHE_PATH.write_text('{"moved player": null}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"moved player": null}', encoding="utf-8")
    scraper.PROFILE_LOOKUP_META_PATH.write_text(
        json.dumps({
            "moved player": {
                "source_uri": "https://www.atptour.com/en/players/old/o001/overview",
                "observed_at": NOW.isoformat(),
                "status": "not_found",
                "missing_fields": ["height_cm", "hand"],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"moved player": "/en/players/new/n001/overview"},
    )
    fetched = []
    monkeypatch.setattr(scraper, "_new_browser_page", _ProfilePage)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda _pg, url: (
            fetched.append(url) or "Moved Player Height 190cm Plays: Right-Handed"
        ),
    )

    assert scraper.batch_get_profiles(["Moved Player"], verbose=False) == {
        "Moved Player": {"height_cm": 190, "hand": "R"},
    }
    assert fetched == ["/en/players/new/n001/overview"]


def test_legacy_positive_cache_remains_compatible_by_default(monkeypatch, tmp_path):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.delenv("ATP_PROFILE_REVALIDATE_LEGACY_POSITIVES", raising=False)
    scraper.CACHE_PATH.write_text('{"legacy positive": 184}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"legacy positive": "R"}', encoding="utf-8")
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: (_ for _ in ()).throw(AssertionError("positive cache must stay compatible")),
    )

    assert scraper.batch_get_profiles(["Legacy Positive"], verbose=False) == {
        "Legacy Positive": {"height_cm": 184, "hand": "R"},
    }


def test_opt_in_revalidation_withholds_unproven_positive_on_browser_failure(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setenv("ATP_PROFILE_REVALIDATE_LEGACY_POSITIVES", "1")
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    scraper.CACHE_PATH.write_text('{"legacy positive": 184}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"legacy positive": "R"}', encoding="utf-8")
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"legacy positive": "/en/players/legacy-positive/l001/overview"},
    )
    monkeypatch.setattr(
        scraper,
        "_new_browser_page",
        lambda: (_ for _ in ()).throw(RuntimeError("browser unavailable")),
    )

    assert scraper.batch_get_profiles(["Legacy Positive"], verbose=False) == {
        "Legacy Positive": {"height_cm": None, "hand": None},
    }
    # Candidate values remain available for audited migration, but the opt-in
    # authority switch refuses to expose them to feature completeness.
    assert json.loads(scraper.CACHE_PATH.read_text()) == {"legacy positive": 184}
    assert json.loads(scraper.HANDS_CACHE_PATH.read_text()) == {"legacy positive": "R"}
    metadata = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())
    assert metadata["legacy positive"]["status"] == "fetch_error"
    assert metadata["legacy positive"]["missing_fields"] == ["height_cm", "hand"]
    assert metadata["legacy positive"]["observed_values"] == {}


def test_opt_in_revalidation_accepts_only_exact_official_page_evidence(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setenv("ATP_PROFILE_REVALIDATE_LEGACY_POSITIVES", "1")
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    scraper.CACHE_PATH.write_text('{"legacy positive": 180}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"legacy positive": "R"}', encoding="utf-8")
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"legacy positive": "/en/players/legacy-positive/l001/overview"},
    )
    fetched = []
    monkeypatch.setattr(scraper, "_new_browser_page", _ProfilePage)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda *_args: fetched.append(True) or (
            "Legacy Positive Height 191cm Plays: Left-Handed"
        ),
    )

    assert scraper.batch_get_profiles(["Legacy Positive"], verbose=False) == {
        "Legacy Positive": {"height_cm": 191, "hand": "L"},
    }
    assert json.loads(scraper.CACHE_PATH.read_text()) == {"legacy positive": 191}
    assert json.loads(scraper.HANDS_CACHE_PATH.read_text()) == {"legacy positive": "L"}
    metadata = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())
    assert metadata["legacy positive"]["identity_binding"] == (
        scraper.OFFICIAL_PAGE_IDENTITY_BINDING
    )

    # The same exact source-bound evidence is reusable; revalidation does not
    # create an hourly refetch loop after it succeeds.
    monkeypatch.setattr(
        scraper,
        "_new_browser_page",
        lambda: (_ for _ in ()).throw(AssertionError("valid evidence must be reused")),
    )
    assert scraper.batch_get_profiles(["Legacy Positive"], verbose=False) == {
        "Legacy Positive": {"height_cm": 191, "hand": "L"},
    }
    assert fetched == [True]

    # Source identity alone is insufficient: cache bytes must still equal the
    # values extracted from the hashed official page body.
    scraper.CACHE_PATH.write_text('{"legacy positive": 190}', encoding="utf-8")
    assert scraper.batch_get_profiles(["Legacy Positive"], verbose=False) == {
        "Legacy Positive": {"height_cm": None, "hand": "L"},
    }


def test_sidecar_never_attributes_unobserved_cached_field_to_page(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    scraper.CACHE_PATH.write_text('{"partial legacy": 184}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"partial legacy": null}', encoding="utf-8")
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"partial legacy": "/en/players/partial-legacy/p001/overview"},
    )
    monkeypatch.setattr(scraper, "_new_browser_page", _ProfilePage)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda *_args: "Partial Legacy Plays: Right-Handed",
    )

    assert scraper.batch_get_profiles(["Partial Legacy"], verbose=False) == {
        "Partial Legacy": {"height_cm": 184, "hand": "R"},
    }
    metadata = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())
    assert metadata["partial legacy"]["status"] == "partial"
    assert metadata["partial legacy"]["missing_fields"] == ["height_cm"]


def test_negative_refresh_budget_bounds_one_pipeline_batch(monkeypatch, tmp_path):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setenv("ATP_PROFILE_NEGATIVE_REFRESH_LIMIT", "1")
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    names = ["First Player", "Second Player", "Third Player"]
    scraper.CACHE_PATH.write_text(
        json.dumps({name.lower(): None for name in names}), encoding="utf-8"
    )
    scraper.HANDS_CACHE_PATH.write_text(
        json.dumps({name.lower(): None for name in names}), encoding="utf-8"
    )
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {
            name.lower(): f"/en/players/{name.split()[0].lower()}/x001/overview"
            for name in names
        },
    )
    fetched = []
    monkeypatch.setattr(scraper, "_new_browser_page", _ProfilePage)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda _pg, url: (
            fetched.append(url) or "First Player Height 185cm Plays: Right-Handed"
        ),
    )

    results = scraper.batch_get_profiles(names, verbose=False)

    assert fetched == ["/en/players/first/x001/overview"]
    assert results["First Player"] == {"height_cm": 185, "hand": "R"}
    assert results["Second Player"] == {"height_cm": None, "hand": None}
    assert results["Third Player"] == {"height_cm": None, "hand": None}
    assert set(json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())) == {
        "first player"
    }


def test_negative_refresh_budget_is_shared_across_match_calls(monkeypatch, tmp_path):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setenv("ATP_PROFILE_NEGATIVE_REFRESH_LIMIT", "1")
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    names = ["First Player", "Second Player"]
    scraper.CACHE_PATH.write_text(
        json.dumps({name.lower(): None for name in names}), encoding="utf-8"
    )
    scraper.HANDS_CACHE_PATH.write_text(
        json.dumps({name.lower(): None for name in names}), encoding="utf-8"
    )
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {
            name.lower(): f"/en/players/{name.split()[0].lower()}/x001/overview"
            for name in names
        },
    )
    fetched = []
    monkeypatch.setattr(scraper, "_new_browser_page", _ProfilePage)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda _pg, url: (
            fetched.append(url) or "First Player Height 185cm Plays: Right-Handed"
        ),
    )
    refresh_state = {}

    first = scraper.batch_get_profiles(
        ["First Player"], verbose=False, refresh_state=refresh_state,
    )
    second = scraper.batch_get_profiles(
        ["Second Player"], verbose=False, refresh_state=refresh_state,
    )

    assert first["First Player"] == {"height_cm": 185, "hand": "R"}
    assert second["Second Player"] == {"height_cm": None, "hand": None}
    assert fetched == ["/en/players/first/x001/overview"]
    assert refresh_state["remaining"] == 0


def test_fetch_error_uses_short_retry_ttl_not_weeklong_negative(monkeypatch):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.delenv("ATP_PROFILE_TRANSIENT_TTL_MINUTES", raising=False)
    source_uri = "https://www.atptour.com/en/players/retry/r001/overview"
    entry = {
        "source_uri": source_uri,
        "observed_at": NOW.isoformat(),
        "status": "fetch_error",
        "missing_fields": ["height_cm", "hand"],
    }

    assert scraper._negative_cache_is_fresh(
        entry,
        source_uri=source_uri,
        missing_fields={"height_cm"},
        now=NOW + timedelta(minutes=59),
    )
    assert not scraper._negative_cache_is_fresh(
        entry,
        source_uri=source_uri,
        missing_fields={"height_cm"},
        now=NOW + timedelta(minutes=61),
    )


def test_unproven_legacy_identity_mismatch_uses_transient_retry_ttl(monkeypatch):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.delenv("ATP_PROFILE_TRANSIENT_TTL_MINUTES", raising=False)
    source_uri = "https://www.atptour.com/en/players/retry/r001/overview"
    entry = {
        "source_uri": source_uri,
        "observed_at": NOW.isoformat(),
        "status": "identity_mismatch",
        "missing_fields": ["height_cm", "hand"],
        "identity_binding": "",
    }

    assert scraper._negative_cache_is_fresh(
        entry,
        source_uri=source_uri,
        missing_fields={"height_cm"},
        now=NOW + timedelta(minutes=59),
    )
    assert not scraper._negative_cache_is_fresh(
        entry,
        source_uri=source_uri,
        missing_fields={"height_cm"},
        now=NOW + timedelta(minutes=61),
    )


def test_future_dated_lookup_evidence_cannot_suppress_retry(monkeypatch):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    source_uri = "https://www.atptour.com/en/players/retry/r001/overview"
    entry = {
        "source_uri": source_uri,
        "observed_at": (NOW + timedelta(days=1)).isoformat(),
        "status": "not_found",
        "missing_fields": ["height_cm"],
    }

    assert not scraper._negative_cache_is_fresh(
        entry,
        source_uri=source_uri,
        missing_fields={"height_cm"},
        now=NOW,
    )


def test_profile_url_requires_exact_normalized_name_binding():
    url_map = {
        "novakdjokovic": "/en/players/novak-djokovic/d643/overview",
        "johnsmith": "/en/players/john-smith/s001/overview",
    }

    assert scraper._find_profile_url("Novak Djoković", url_map) == (
        "/en/players/novak-djokovic/d643/overview"
    )
    assert scraper._find_profile_url("Jane Smith", url_map) is None


def test_abbreviated_rankings_name_uses_official_full_name_url_slug(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    rankings_path = tmp_path / "atp_rankings.csv"
    pd.DataFrame([{
        "player_name": "J. Sinner",
        "player_url": "/en/players/jannik-sinner/s0ag/overview",
    }]).to_csv(rankings_path, index=False)
    monkeypatch.setattr(scraper, "RANKINGS_PATH", rankings_path)
    page = _ProfilePage()
    monkeypatch.setattr(scraper, "_new_browser_page", lambda: page)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda *_args: "Jannik Sinner Height 191cm Plays: Right-Handed",
    )

    assert scraper.batch_get_profiles(["Jannik Sinner"], verbose=False) == {
        "Jannik Sinner": {"height_cm": 191, "hand": "R"},
    }
    assert page.closed is True


def test_profile_body_identity_mismatch_never_populates_cache(monkeypatch, tmp_path):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    scraper.CACHE_PATH.write_text('{"right player": null}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"right player": null}', encoding="utf-8")
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"right player": "/en/players/right-player/r001/overview"},
    )
    monkeypatch.setattr(scraper, "_new_browser_page", _ProfilePage)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda *_args: "Different Person Height 190cm Plays: Right-Handed",
    )

    assert scraper.batch_get_profiles(["Right Player"], verbose=False) == {
        "Right Player": {"height_cm": None, "hand": None},
    }
    assert json.loads(scraper.CACHE_PATH.read_text()) == {"right player": None}
    metadata = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())
    assert metadata["right player"]["status"] == "identity_mismatch"
    assert metadata["right player"]["identity_binding"] == (
        scraper.OFFICIAL_PAGE_CONFLICT_BINDING
    )


def test_non_profile_interstitial_is_a_transient_fetch_error(monkeypatch, tmp_path):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    scraper.CACHE_PATH.write_text('{"retry player": null}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"retry player": null}', encoding="utf-8")
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"retry player": "/en/players/retry-player/r001/overview"},
    )
    monkeypatch.setattr(scraper, "_new_browser_page", _ProfilePage)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda *_args: "The resource is temporarily unavailable. Please retry.",
    )

    assert scraper.batch_get_profiles(["Retry Player"], verbose=False) == {
        "Retry Player": {"height_cm": None, "hand": None},
    }
    metadata = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())
    assert metadata["retry player"]["status"] == "fetch_error"
    assert metadata["retry player"]["identity_binding"] == ""


def test_browser_launch_failure_preserves_partial_positive_and_stays_incomplete(
    monkeypatch, tmp_path,
):
    _point_cache_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "_utc_now", lambda: NOW)
    scraper.CACHE_PATH.write_text('{"partial player": 185}', encoding="utf-8")
    scraper.HANDS_CACHE_PATH.write_text('{"partial player": null}', encoding="utf-8")
    monkeypatch.setattr(
        scraper,
        "_load_url_map",
        lambda: {"partial player": "/en/players/partial-player/p001/overview"},
    )
    monkeypatch.setattr(
        scraper,
        "_new_browser_page",
        lambda: (_ for _ in ()).throw(RuntimeError("browser unavailable")),
    )

    assert scraper.batch_get_profiles(["Partial Player"], verbose=False) == {
        "Partial Player": {"height_cm": 185, "hand": None},
    }
    assert json.loads(scraper.CACHE_PATH.read_text()) == {"partial player": 185}
    metadata = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())
    assert metadata["partial player"]["status"] == "fetch_error"
    assert metadata["partial player"]["missing_fields"] == ["hand"]


def test_hourly_checkpoint_conditionally_persists_profile_lookup_evidence():
    workflow = (ROOT / ".github" / "workflows" / "hourly-pipeline.yml").read_text()
    bulk_start = workflow.index("git add -A -f production/prediction_log.csv")
    bulk_end = workflow.index("|| true", bulk_start)
    bulk_command = workflow[bulk_start:bulk_end]

    assert "production/prediction_log.csv" in bulk_command
    assert "data/atp_profile_lookup_meta.json" not in bulk_command
    assert "PROFILE_LOOKUP_META=data/atp_profile_lookup_meta.json" in workflow
    assert 'if [ -e "$PROFILE_LOOKUP_META" ] ||' in workflow
    assert 'git ls-files --error-unmatch "$PROFILE_LOOKUP_META"' in workflow
    assert 'git add -A -f -- "$PROFILE_LOOKUP_META"' in workflow


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


def test_legacy_feature_resolution_rejects_invalid_store_and_fallback_heights(
    monkeypatch,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    requested = []

    def fallback_profiles(names, verbose=False):
        requested.append((tuple(names), verbose))
        return {
            "Bad Store Good ATP": {"height_cm": 182, "hand": "R"},
            "Bad Everywhere": {"height_cm": 231, "hand": "L"},
        }

    monkeypatch.setattr(ta_feature_module, "batch_get_profiles", fallback_profiles)
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    persisted = []
    calc._persist_player_field = (
        lambda profile, field, value: persisted.append((profile["player_id"], field, value))
    )

    result = calc._resolve_height_hand(
        "Bad Store Good ATP", "Bad Everywhere",
        {"player_id": 42, "height_cm": 71, "hand": "R"},
        {"player_id": 43, "height_cm": 145, "hand": "L"},
    )

    assert requested == [(("Bad Store Good ATP", "Bad Everywhere"), False)]
    assert result == (182.0, "R", None, "L")
    assert persisted == [(42, "height_cm", 182.0)]


def test_feature_resolution_reuses_run_scoped_atp_refresh_budget(monkeypatch):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    observed_states = []

    def fallback_profiles(names, verbose=False, refresh_state=None):
        observed_states.append(refresh_state)
        return {name: {"height_cm": None, "hand": "R"} for name in names}

    monkeypatch.setattr(ta_feature_module, "batch_get_profiles", fallback_profiles)
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    session_cache = {}
    profiles = (
        {"player_id": 42, "height_cm": None, "hand": "R"},
        {"player_id": 43, "height_cm": None, "hand": "L"},
    )

    calc._resolve_height_hand(
        "First Player", "Second Player", *profiles, session_cache=session_cache,
    )
    calc._resolve_height_hand(
        "First Player", "Second Player", *profiles, session_cache=session_cache,
    )

    assert len(observed_states) == 2
    assert observed_states[0] is observed_states[1]
    assert observed_states[0] is session_cache["atp_profile_refresh"]


def test_required_feature_resolution_defensively_rejects_invalid_bundle_height(
    monkeypatch,
):
    monkeypatch.setenv("ELIGIBILITY_PROVENANCE_MODE", "required")
    monkeypatch.setattr(
        ta_feature_module,
        "batch_get_profiles",
        lambda *_args, **_kwargs: {
            "Invalid Accepted": {
                "canonical_player_id": 42, "height_cm": 132, "hand": "R",
            },
            "Valid Accepted": {
                "canonical_player_id": 43, "height_cm": 230, "hand": "L",
            },
        },
    )
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)

    assert calc._resolve_height_hand(
        "Invalid Accepted", "Valid Accepted",
        {"player_id": 42, "height_cm": 180, "hand": "R"},
        {"player_id": 43, "height_cm": 181, "hand": "L"},
    ) == (None, "R", 230.0, "L")


def test_legacy_write_through_never_persists_invalid_height(monkeypatch):
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    connection = _Connection()
    calc._store = lambda: connection
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)

    for value in (149, 231, float("nan"), float("inf"), "not-a-height"):
        calc._persist_player_field({"player_id": 42}, "height_cm", value)
    calc._persist_player_field({"player_id": 42}, "height_cm", 150)
    calc._persist_player_field({"player_id": 42}, "height_cm", 230)

    assert [params for _query, params in connection.calls] == [(150.0, 42), (230.0, 42)]


def test_invalid_store_height_remains_default_marked_end_to_end(monkeypatch):
    class NoNetworkScraper:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected network method: {name}")

    profiles = {
        "invalid-player": {
            "player_id": 42,
            "name": "Invalid Player",
            "slug": "invalid-player",
            "height_cm": 71,
            "hand": "R",
            "country": "USA",
            "birthdate": "2000-01-01",
            "age": 26,
            "current_rank": 100,
        },
        "valid-player": {
            "player_id": 43,
            "name": "Valid Player",
            "slug": "valid-player",
            "height_cm": 185,
            "hand": "L",
            "country": "GBR",
            "birthdate": "2000-01-01",
            "age": 26,
            "current_rank": 101,
        },
    }
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(ta_feature_module, "needs_stitching", lambda *_args: False)
    monkeypatch.setattr(
        ta_feature_module, "batch_get_profiles", lambda *_args, **_kwargs: {}
    )
    calc = TAFeatureCalculator(NoNetworkScraper())
    calc.use_store = True
    calc._atp_rankings = None
    calc._store_profile = lambda slug: dict(profiles[slug])
    calc._store_history_frame = lambda *_args, **_kwargs: pd.DataFrame()

    features = calc.build_141_features_from_slugs(
        slug1="invalid-player",
        slug2="valid-player",
        match_date=datetime(2026, 7, 20, 12),
        surface="Hard",
        tournament_level="A",
        draw_size=32,
        round_code="R32",
        force_refresh=False,
        persist=False,
        session_cache={},
        match_date_is_explicit=True,
    )

    assert features["Player1_Height"] == 180.0
    assert features["Player2_Height"] == 185.0
    assert any(
        marker.startswith("Player1_Height")
        for marker in features["_defaulted_features"].split(",")
    )


@pytest.mark.parametrize(
    ("display_name", "ranking_name", "player_url"),
    [
        (
            "Chak Lam Coleman Wong",
            "C. Wong",
            "/en/players/coleman-wong/w0bh/overview",
        ),
        (
            "Alexander Shevchenko",
            "A. Shevchenko",
            "/en/players/aleksandr-shevchenko/s0h2/overview",
        ),
        (
            "Stevan Popovic",
            "S. Popovic",
            "/en/players/stefan-popovic/p0g5/overview",
        ),
    ],
)
def test_rank_alias_identity_misses_are_default_marked_end_to_end(
    monkeypatch,
    display_name,
    ranking_name,
    player_url,
):
    class OfflineScraper:
        @staticmethod
        def get_player_matches(*_args, **_kwargs):
            return pd.DataFrame()

    profiles = {
        "unresolved": {
            "player_id": 101,
            "name": display_name,
            "slug": "unresolved",
            "height_cm": 183,
            "hand": "R",
            "country": "",
            "birthdate": "2000-01-01",
            "age": 26,
            "current_rank": None,
            "atp_url": "",
        },
        "valid": {
            "player_id": None,
            "name": "Valid Exact Player",
            "slug": "valid",
            "height_cm": 185,
            "hand": "L",
            "country": "",
            "birthdate": "2000-01-01",
            "age": 26,
            "current_rank": 101,
            "atp_url": "/en/players/valid-exact-player/v001/overview",
        },
    }
    rankings = pd.DataFrame([
        {
            "rank": 111,
            "player_name": ranking_name,
            "points": 520,
            "player_url": player_url,
        },
        {
            "rank": 101,
            "player_name": "Valid Exact Player",
            "points": 600,
            "player_url": "/en/players/valid-exact-player/v001/overview",
        },
    ])
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(ta_feature_module, "needs_stitching", lambda *_args: False)
    monkeypatch.setattr(
        ta_feature_module, "batch_get_profiles", lambda *_args, **_kwargs: {},
    )
    calc = TAFeatureCalculator(OfflineScraper())
    calc.use_store = True
    calc._atp_rankings = rankings
    calc._store_profile = lambda slug: dict(profiles[slug])
    calc._store_history_frame = lambda *_args, **_kwargs: pd.DataFrame()

    import canonical_store
    import store_history

    @contextmanager
    def store_connection():
        yield object()

    monkeypatch.setattr(canonical_store, "connect", store_connection)
    monkeypatch.setattr(
        store_history,
        "latest_recorded_rank",
        lambda _connection, player_id: 88 if player_id == 101 else None,
    )

    features = calc.build_141_features_from_slugs(
        slug1="unresolved",
        slug2="valid",
        match_date=datetime(2026, 7, 20, 12),
        surface="Hard",
        tournament_level="A",
        draw_size=32,
        round_code="R32",
        force_refresh=False,
        persist=False,
        session_cache={},
        match_date_is_explicit=True,
    )

    defaults = set(features["_defaulted_features"].split(","))
    assert (
        "P1_Rank=rank_lookup_unresolved(identity_unresolved)" in defaults
    )
    assert (
        "P1_Rank_Points=rank_lookup_unresolved(identity_unresolved)" in defaults
    )
    assert (
        "P1_Rank=rank_lookup_unresolved(store_history_fallback)" in defaults
    )
    assert (
        "P1_Rank_Points=rank_lookup_unresolved(store_history_fallback)"
        in defaults
    )
    assert not any(marker.startswith("P2_Rank=") for marker in defaults)
    assert features["Player1_Rank"] == 88.0


def test_malformed_exact_rank_is_default_marked_end_to_end(monkeypatch):
    class OfflineScraper:
        @staticmethod
        def get_player_matches(*_args, **_kwargs):
            return pd.DataFrame()

    profiles = {
        "invalid": {
            "player_id": None,
            "name": "Exact Invalid Rank",
            "slug": "invalid",
            "height_cm": 183,
            "hand": "R",
            "country": "",
            "birthdate": "2000-01-01",
            "age": 26,
            "current_rank": None,
            "atp_url": "/en/players/exact-invalid-rank/e001/overview",
        },
        "valid": {
            "player_id": None,
            "name": "Valid Exact Player",
            "slug": "valid",
            "height_cm": 185,
            "hand": "L",
            "country": "",
            "birthdate": "2000-01-01",
            "age": 26,
            "current_rank": 101,
            "atp_url": "/en/players/valid-exact-player/v001/overview",
        },
    }
    rankings = pd.DataFrame([
        {
            "rank": "bad",
            "player_name": "Exact Invalid Rank",
            "points": 520,
            "player_url": "/en/players/exact-invalid-rank/e001/overview",
        },
        {
            "rank": 101,
            "player_name": "Valid Exact Player",
            "points": 600,
            "player_url": "/en/players/valid-exact-player/v001/overview",
        },
    ])
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(ta_feature_module, "needs_stitching", lambda *_args: False)
    monkeypatch.setattr(
        ta_feature_module, "batch_get_profiles", lambda *_args, **_kwargs: {},
    )
    calc = TAFeatureCalculator(OfflineScraper())
    calc.use_store = True
    calc._atp_rankings = rankings
    calc._store_profile = lambda slug: dict(profiles[slug])
    calc._store_history_frame = lambda *_args, **_kwargs: pd.DataFrame()

    features = calc.build_141_features_from_slugs(
        slug1="invalid",
        slug2="valid",
        match_date=datetime(2026, 7, 20, 12),
        surface="Hard",
        tournament_level="A",
        draw_size=32,
        round_code="R32",
        force_refresh=False,
        persist=False,
        session_cache={},
        match_date_is_explicit=True,
    )

    defaults = set(features["_defaulted_features"].split(","))
    assert "P1_Rank=rank_lookup_unresolved(rank_invalid)" in defaults
    assert "P1_Rank_Points=rank_lookup_unresolved(rank_invalid)" in defaults
    assert features["Player1_Rank"] == 999.0


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
