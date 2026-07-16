from contextlib import contextmanager
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
import itf_results_scraper as itf_scraper  # noqa: E402
import canonical_store  # noqa: E402
import main as production_main  # noqa: E402
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


def test_slate_prehydration_uses_itf_player_id_profile_for_unknown_hand(
    monkeypatch,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    profiles = {
        "itf player": {
            "player_id": 42,
            "name": "ITF Player",
            "height_cm": 181,
            "hand": "U",
        },
        "known player": {
            "player_id": 43,
            "name": "Known Player",
            "height_cm": 185,
            "hand": "R",
        },
    }
    monkeypatch.setattr(
        store_history,
        "get_profile",
        lambda _conn, name: dict(profiles[str(name).strip().casefold()]),
    )

    class _ItfClient:
        def close(self):
            pass

    monkeypatch.setattr(itf_scraper, "ItfClient", _ItfClient)
    monkeypatch.setattr(
        itf_scraper,
        "get_player_profiles",
        lambda _client, refs: {
            "ITF Player": {
                "status": "resolved",
                "hand": "L",
                "itf_player_id": refs["ITF Player"]["itf_player_id"],
                "source_uri": "https://www.itftennis.com/en/players/itf-player/800000042/usa/mt/s/",
            },
        },
    )
    monkeypatch.setattr(
        ta_feature_module,
        "batch_get_profiles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no ATP batch is needed when both heights exist")
        ),
    )

    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    calc._store = lambda: object()
    persisted = []
    calc._persist_player_field = lambda profile, field, value: persisted.append(
        (profile["player_id"], field, value)
    )
    slate = pd.DataFrame([{
        "player1_normalized": "ITF Player",
        "player2_normalized": "Known Player",
        "event": "ITF Men Test",
    }])
    event_matches = pd.DataFrame([{
        "p1": "ITF Player",
        "p2": "Known Player",
        "p1_id": 800000042,
        "p2_id": 800000043,
        "p1_profile_url": "/en/players/itf-player/800000042/usa/mt/s/",
        "p2_profile_url": "/en/players/known-player/800000043/usa/mt/s/",
        "p1_nationality": "USA",
        "p2_nationality": "USA",
    }])
    session_cache = {"itf_event_matches": {"m-itf-test-2026-1": event_matches}}

    summary = calc.prehydrate_slate_profiles(slate, session_cache=session_cache)

    assert summary["status"] == "no_height_candidates"
    assert session_cache["itf_profile_hydration"] == {
        "status": "complete",
        "candidate_players": 1,
        "official_page_attempts": 1,
        "resolved_hands": 1,
        "failed_profiles": 0,
    }
    assert session_cache["itf_hands_by_player_id"] == {42: "L"}
    assert persisted == [(42, "hand", "L")]


class _OwnedCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.connection.calls.append((str(query), params))
        if str(query).lstrip().upper().startswith("SELECT"):
            if not self.connection.autocommit:
                self.connection.outer_transaction = True


class _OwnedConnection:
    def __init__(self):
        self._autocommit = False
        self.outer_transaction = False
        self.in_transaction = False
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.calls = []

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        if self.outer_transaction:
            raise RuntimeError("cannot change autocommit inside a transaction")
        self._autocommit = bool(value)

    def cursor(self):
        return _OwnedCursor(self)

    @contextmanager
    def transaction(self):
        assert self.autocommit is True
        assert self.outer_transaction is False
        assert self.in_transaction is False
        self.in_transaction = True
        try:
            yield
        except Exception:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1
        finally:
            self.in_transaction = False

    def close(self):
        self.closed = True


def test_feature_store_reads_autocommit_writes_root_transaction_and_closes(
    monkeypatch,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    connection = _OwnedConnection()
    monkeypatch.setattr(canonical_store, "connect", lambda: connection)
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc._store_conn = None
    calc.use_store = True

    owned = calc._store()
    with owned.cursor() as cursor:
        cursor.execute("SELECT player_id FROM players", None)

    assert owned.autocommit is True
    assert owned.outer_transaction is False
    calc._persist_player_field({"player_id": 42}, "height_cm", 188)
    assert owned.commits == 1
    assert owned.rollbacks == 0
    assert any("UPDATE players" in query for query, _params in owned.calls)

    calc.close_store()
    assert connection.closed is True
    assert calc._store_conn is None
    calc.close_store()  # idempotent


def test_hand_only_canonical_display_collision_fails_without_persisting_shared_hand(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    monkeypatch.setattr(scraper, "CACHE_PATH", tmp_path / "heights.json")
    monkeypatch.setattr(scraper, "HANDS_CACHE_PATH", tmp_path / "hands.json")
    monkeypatch.setattr(scraper, "PROFILE_LOOKUP_META_PATH", tmp_path / "meta.json")
    scraper.CACHE_PATH.write_text('{"shared canonical name": 189}')
    scraper.HANDS_CACHE_PATH.write_text('{"shared canonical name": "L"}')
    profiles = {
        "alpha alias": {
            "player_id": 10,
            "name": "Shared Canonical Name",
            "height_cm": 188,
            "hand": "U",
        },
        "beta alias": {
            "player_id": 20,
            "name": "Shared Canonical Name",
            "height_cm": 190,
            "hand": "U",
        },
    }
    monkeypatch.setattr(
        store_history,
        "get_profile",
        lambda _conn, name: dict(profiles[str(name).strip().casefold()]),
    )
    monkeypatch.setattr(
        ta_feature_module,
        "profile_lookup_evidence_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous identity must fail before cache inspection")
        ),
    )
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    calc._store = lambda: object()
    persisted = []
    calc._persist_player_field = lambda profile, field, value: persisted.append(
        (profile["player_id"], field, value)
    )
    session_cache = {}

    with pytest.raises(ta_feature_module.UnsafeToInferError, match="multiple canonical"):
        calc.prehydrate_slate_profiles(
            pd.DataFrame([{
                "player1_normalized": "Alpha Alias",
                "player2_normalized": "Beta Alias",
                "event": "Challenger - Test",
            }]),
            session_cache=session_cache,
        )

    assert "atp_profile_refresh" not in session_cache
    assert persisted == []
    assert json.loads(scraper.HANDS_CACHE_PATH.read_text()) == {
        "shared canonical name": "L"
    }


def test_hand_only_slate_installs_canonical_strict_refresh_state(monkeypatch):
    monkeypatch.delenv("ELIGIBILITY_PROVENANCE_MODE", raising=False)
    profiles = {
        "alpha player": {
            "player_id": 10,
            "name": "Alpha Player",
            "height_cm": 188,
            "hand": "U",
        },
        "beta player": {
            "player_id": 20,
            "name": "Beta Player",
            "height_cm": 190,
            "hand": "U",
        },
    }
    monkeypatch.setattr(
        store_history,
        "get_profile",
        lambda _conn, name: dict(profiles[str(name).strip().casefold()]),
    )
    monkeypatch.setattr(
        ta_feature_module,
        "batch_get_profiles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("height batch must not run without height candidates")
        ),
    )
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.use_store = True
    calc._store = lambda: object()
    session_cache = {}

    summary = calc.prehydrate_slate_profiles(
        pd.DataFrame([{
            "player1_normalized": "Alpha Player",
            "player2_normalized": "Beta Player",
            "event": "Challenger - Test",
        }]),
        session_cache=session_cache,
    )

    assert summary["status"] == "no_height_candidates"
    refresh = session_cache["atp_profile_refresh"]
    assert refresh["require_evidenced_positives"] is True
    assert refresh["remaining"] == 32
    assert refresh["allowed_keys"] == {"alpha player", "beta player"}
    assert refresh["canonical_player_ids"] == {
        "alpha player": 10,
        "beta player": 20,
    }


def test_extract_features_prefilters_ineligible_hydration_and_propagates_conflict():
    captured = []

    class _ConflictingEngine:
        def prehydrate_slate_profiles(self, odds_df, *, session_cache=None):
            captured.append(odds_df.copy())
            raise ta_feature_module.UnsafeToInferError("canonical key collision")

        def build_141_features(self, **_kwargs):
            raise AssertionError("per-match fallback must not run after ambiguity")

    orchestrator = production_main.LiveBettingOrchestrator.__new__(
        production_main.LiveBettingOrchestrator
    )
    orchestrator.run_id = "run_height_guard"
    orchestrator.run_metrics = {}
    orchestrator._session_cache = {}
    orchestrator.feature_engine = _ConflictingEngine()
    reasons = {
        "future": "",
        "started": "scheduled_start_passed",
        "": "match_start_time_missing",
    }
    orchestrator.get_inference_guard_reason = lambda match_time: (
        None, reasons[match_time]
    )
    odds = pd.DataFrame([
        {
            "player1_raw": "Future One",
            "player2_raw": "Future Two",
            "event": "Challenger - Future",
            "match_time": "future",
        },
        {
            "player1_raw": "Started One",
            "player2_raw": "Started Two",
            "event": "Challenger - Started",
            "match_time": "started",
        },
        {
            "player1_raw": "No Clock One",
            "player2_raw": "No Clock Two",
            "event": "ITF Men No Clock",
            "match_time": "",
        },
    ])

    with pytest.raises(ta_feature_module.UnsafeToInferError, match="collision"):
        orchestrator.extract_features(odds)

    assert len(captured) == 1
    assert captured[0]["match_time"].tolist() == ["future"]


def test_pipeline_finally_closes_feature_store_after_failure(capsys):
    class _Engine:
        def __init__(self):
            self.closed = 0

        def close_store(self):
            self.closed += 1

    orchestrator = production_main.LiveBettingOrchestrator.__new__(
        production_main.LiveBettingOrchestrator
    )
    orchestrator.feature_engine = _Engine()
    orchestrator._start_run_context = lambda: (_ for _ in ()).throw(
        RuntimeError("early failure")
    )
    orchestrator._persist_run_state = lambda **_kwargs: None

    assert orchestrator.run_full_pipeline() is False
    assert orchestrator.feature_engine.closed == 1
    capsys.readouterr()
