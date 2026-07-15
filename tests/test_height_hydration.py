from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))
sys.path.insert(0, str(PRODUCTION / "scraping"))

from features.height_hydration import (  # noqa: E402
    HeightHydrationCandidate,
    plan_height_hydration,
)
from features import ta_feature_calculator as ta_feature_module  # noqa: E402
from features.ta_feature_calculator import TAFeatureCalculator  # noqa: E402
from scraping import atp_height_scraper as scraper  # noqa: E402
import store_history  # noqa: E402


NOW = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)


def _candidate(
    player_id,
    name,
    event,
    *,
    completes=False,
    evidence="unobserved",
):
    return HeightHydrationCandidate(
        canonical_player_id=player_id,
        player_name=name,
        event=event,
        opponent_has_height=completes,
        evidence_state=evidence,
    )


def test_plan_dedupes_canonical_ids_and_prioritizes_impact_deterministically():
    candidates = [
        _candidate(10, "Alias Player", "ITF Men Test", evidence="unobserved"),
        _candidate(
            10,
            "Canonical Player",
            "Challenger - Test",
            completes=True,
            evidence="expired_negative",
        ),
        _candidate(
            20,
            "ITF Completer",
            "ITF Men Test",
            completes=True,
            evidence="unobserved",
        ),
        _candidate(
            30,
            "Fresh Challenger",
            "Challenger - Test",
            evidence="fresh_negative",
        ),
        _candidate(40, "Cached ATP", "ATP - Test", evidence="resolved"),
    ]

    planned = plan_height_hydration(list(reversed(candidates)))

    assert [candidate.canonical_player_id for candidate in planned] == [40, 10, 20, 30]
    assert planned[1].player_name == "Canonical Player"


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(0, "Bad ID", "ATP - Test"),
        _candidate(1, "", "ATP - Test"),
        _candidate(1, "Bad State", "ATP - Test", evidence="guessed"),
    ],
)
def test_plan_rejects_noncanonical_or_unknown_inputs(candidate):
    with pytest.raises(ValueError):
        plan_height_hydration([candidate])


def test_evidence_state_snapshot_separates_resolved_fresh_expired_and_unobserved(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "CACHE_PATH", tmp_path / "heights.json")
    monkeypatch.setattr(scraper, "HANDS_CACHE_PATH", tmp_path / "hands.json")
    monkeypatch.setattr(scraper, "PROFILE_LOOKUP_META_PATH", tmp_path / "meta.json")
    scraper.CACHE_PATH.write_text(json.dumps({
        "resolved player": 185,
        "fresh player": None,
        "expired player": None,
    }))
    source = lambda slug: f"https://www.atptour.com/en/players/{slug}/x001/overview"
    scraper.PROFILE_LOOKUP_META_PATH.write_text(json.dumps({
        "fresh player": {
            "source_uri": source("fresh-player"),
            "observed_at": NOW.isoformat(),
            "status": "not_found",
            "missing_fields": ["height_cm"],
        },
        "expired player": {
            "source_uri": source("expired-player"),
            "observed_at": (NOW - timedelta(days=8)).isoformat(),
            "status": "not_found",
            "missing_fields": ["height_cm"],
        },
    }))
    monkeypatch.setattr(scraper, "_load_url_map", lambda: {
        "resolved player": "/en/players/resolved-player/x001/overview",
        "fresh player": "/en/players/fresh-player/x001/overview",
        "expired player": "/en/players/expired-player/x001/overview",
        "unobserved player": "/en/players/unobserved-player/x001/overview",
    })

    states = scraper.profile_lookup_evidence_states(
        ["Resolved Player", "Fresh Player", "Expired Player", "Unobserved Player"],
        now=NOW,
    )

    assert {name: row["state"] for name, row in states.items()} == {
        "Resolved Player": "resolved",
        "Fresh Player": "fresh_negative",
        "Expired Player": "expired_negative",
        "Unobserved Player": "unobserved",
    }
    assert all(row["has_profile_url"] for row in states.values())

    strict_states = scraper.profile_lookup_evidence_states(
        ["Resolved Player"],
        now=NOW,
        require_evidenced_positives=True,
    )
    assert strict_states["Resolved Player"]["state"] == "unobserved"


def test_batch_refresh_state_rejects_names_outside_canonical_plan(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "CACHE_PATH", tmp_path / "heights.json")
    monkeypatch.setattr(scraper, "HANDS_CACHE_PATH", tmp_path / "hands.json")
    monkeypatch.setattr(scraper, "PROFILE_LOOKUP_META_PATH", tmp_path / "meta.json")
    scraper.CACHE_PATH.write_text("{}")
    scraper.HANDS_CACHE_PATH.write_text("{}")
    monkeypatch.setattr(scraper, "_load_url_map", lambda: {
        "canonical player": "/en/players/canonical-player/x001/overview",
        "unplanned player": "/en/players/unplanned-player/x002/overview",
    })
    monkeypatch.setattr(
        scraper,
        "_new_browser_page",
        lambda: (_ for _ in ()).throw(AssertionError("unplanned identity must not fetch")),
    )
    refresh_state = {"remaining": 32, "allowed_keys": {"canonical player"}}

    result = scraper.batch_get_profiles(
        ["Unplanned Player"],
        verbose=False,
        refresh_state=refresh_state,
    )

    assert result == {"Unplanned Player": {"height_cm": None, "hand": None}}
    assert refresh_state["remaining"] == 32
    assert refresh_state["attempted_keys"] == set()


def test_canonical_batch_persists_player_id_with_exact_page_evidence(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "CACHE_PATH", tmp_path / "heights.json")
    monkeypatch.setattr(scraper, "HANDS_CACHE_PATH", tmp_path / "hands.json")
    monkeypatch.setattr(scraper, "PROFILE_LOOKUP_META_PATH", tmp_path / "meta.json")
    # A name-keyed legacy positive is compatibility data, not sufficient
    # evidence for the canonical run-level lane until the official page binds.
    scraper.CACHE_PATH.write_text('{"canonical player": 188}')
    scraper.HANDS_CACHE_PATH.write_text("{}")
    monkeypatch.setattr(scraper, "_load_url_map", lambda: {
        "canonical player": "/en/players/canonical-player/x001/overview",
    })

    class _Page:
        def close(self):
            pass

    monkeypatch.setattr(scraper, "_new_browser_page", _Page)
    monkeypatch.setattr(
        scraper,
        "_fetch_profile_text",
        lambda _page, _url: "Canonical Player Height 188cm Plays: Left-Handed",
    )
    refresh_state = {
        "remaining": 32,
        "allowed_keys": {"canonical player"},
        "canonical_player_ids": {"canonical player": 42},
        "require_evidenced_positives": True,
    }

    result = scraper.batch_get_profiles(
        ["Canonical Player"], verbose=False, refresh_state=refresh_state,
    )
    evidence = json.loads(scraper.PROFILE_LOOKUP_META_PATH.read_text())[
        "canonical player"
    ]

    assert result == {"Canonical Player": {"height_cm": 188, "hand": "L"}}
    assert evidence["canonical_player_id"] == 42
    assert evidence["status"] == "resolved"
    assert evidence["identity_binding"] == scraper.OFFICIAL_PAGE_IDENTITY_BINDING
    assert len(evidence["source_content_sha256"]) == 64
    assert evidence["observed_values"] == {"height_cm": 188, "hand": "L"}
    assert scraper.profile_lookup_evidence_states(
        ["Canonical Player"],
        require_evidenced_positives=True,
        canonical_player_ids={"canonical player": 42},
    )["Canonical Player"]["state"] == "resolved"
    assert scraper.profile_lookup_evidence_states(
        ["Canonical Player"],
        require_evidenced_positives=True,
        canonical_player_ids={"canonical player": 43},
    )["Canonical Player"]["state"] == "unobserved"


def test_slate_prehydration_uses_one_canonical_batch_and_shared_32_lookup_budget(
    monkeypatch,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.delenv("ATP_PROFILE_RUN_HYDRATION_LIMIT", raising=False)
    profiles = {
        "alpha alias": {
            "player_id": 10,
            "name": "Alpha Canonical",
            "height_cm": None,
            "hand": "U",
        },
        "known player": {
            "player_id": 11,
            "name": "Known Player",
            "height_cm": 184,
            "hand": "R",
        },
        "charlie player": {
            "player_id": 20,
            "name": "Charlie Player",
            "height_cm": None,
            "hand": "R",
        },
        "delta player": {
            "player_id": 21,
            "name": "Delta Player",
            "height_cm": None,
            "hand": "L",
        },
    }
    monkeypatch.setattr(
        store_history,
        "get_profile",
        lambda _conn, name: (
            None
            if str(name).strip().casefold() not in profiles
            else dict(profiles[str(name).strip().casefold()])
        ),
    )
    monkeypatch.setattr(
        ta_feature_module,
        "profile_lookup_evidence_states",
        lambda names, **_kwargs: {
            name: {
                "state": "fresh_negative" if name == "Delta Player" else "unobserved"
            }
            for name in names
        },
    )
    calls = []

    def _batch(names, verbose=False, refresh_state=None):
        calls.append((list(names), verbose, refresh_state))
        assert refresh_state["remaining"] == 32
        refresh_state.setdefault("attempted_keys", set()).update({
            "alpha canonical", "charlie player",
        })
        refresh_state["remaining"] = 30
        return {
            "Alpha Canonical": {"height_cm": 188, "hand": "L"},
            "Charlie Player": {"height_cm": None, "hand": "R"},
            "Delta Player": {"height_cm": None, "hand": "L"},
        }

    monkeypatch.setattr(ta_feature_module, "batch_get_profiles", _batch)
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    calc._store = lambda: object()
    persisted = []
    calc._persist_player_field = lambda profile, field, value: persisted.append(
        (profile["player_id"], field, value)
    )
    slate = pd.DataFrame([
        {
            "player1_normalized": "Alpha Alias",
            "player2_normalized": "Known Player",
            "event": "ITF Men Test",
        },
        {
            "player1_normalized": "Charlie Player",
            "player2_normalized": "Delta Player",
            "event": "Challenger - Test",
        },
        {
            "player1_normalized": "Alpha Alias",
            "player2_normalized": "Known Player",
            "event": "Challenger - Better Priority",
        },
    ])
    session_cache = {}

    summary = calc.prehydrate_slate_profiles(slate, session_cache=session_cache)

    assert len(calls) == 1
    assert calls[0][0] == ["Alpha Canonical", "Charlie Player", "Delta Player"]
    assert calls[0][2]["allowed_keys"] == {
        "alpha canonical", "known player", "charlie player", "delta player",
    }
    assert calls[0][2]["canonical_player_ids"] == {
        "alpha canonical": 10,
        "known player": 11,
        "charlie player": 20,
        "delta player": 21,
    }
    assert calls[0][2]["require_evidenced_positives"] is True
    assert summary == {
        "status": "complete",
        "candidate_players": 3,
        "planned_players": 3,
        "browser_attempts": 2,
        "resolved_heights": 1,
        "remaining_budget": 30,
        "evidence_states": {"unobserved": 2, "fresh_negative": 1},
    }
    assert (10, "height_cm", 188.0) in persisted
    assert (10, "hand", "L") in persisted
    assert session_cache["height_hydration"] == summary
