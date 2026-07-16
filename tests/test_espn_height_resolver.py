from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from scraping import espn_height_resolver as resolver  # noqa: E402


def _search(*items):
    return {"results": [{"contents": list(items)}]}


def _item(name, athlete_id):
    return {
        "sport": "tennis",
        "type": "player",
        "displayName": name,
        "uid": f"s:850~l:851~a:{athlete_id}",
    }


def _athlete(name, athlete_id, *, height=73):
    return {
        "id": str(athlete_id),
        "fullName": name,
        "displayName": name,
        "height": height,
    }


def _cache():
    return {"schema_version": resolver.CACHE_SCHEMA_VERSION, "entries": {}}


def test_exact_unique_search_and_athlete_identity_resolve_with_evidence(monkeypatch):
    def get_json(url):
        if "apis/search" in url:
            return _search(_item("Nikita Bilozertsev", 16911))
        return _athlete("Nikita Bilozertsev", 16911)

    monkeypatch.setattr(resolver, "_get_json", get_json)
    observation = resolver.resolve_height_observation(
        canonical_player_id=42,
        player_name="Nikita Bilozertsev",
        cache=_cache(),
    )

    assert observation["status"] == "resolved"
    assert observation["height_cm"] == 185.4
    assert observation["external_player_id"] == "16911"
    assert observation["identity_binding"] == resolver.IDENTITY_BINDING
    assert len(observation["source_content_sha256"]) == 64
    assert resolver.validated_resolved_height(
        observation,
        canonical_player_id=42,
        player_name="Nikita Bilozertsev",
    ) == 185.4


def test_duplicate_exact_espn_athletes_fail_closed(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_get_json",
        lambda _url: _search(
            _item("Lorenzo Angelini", 9986),
            _item("Lorenzo Angelini", 9572),
        ),
    )

    observation = resolver.resolve_height_observation(
        canonical_player_id=43,
        player_name="Lorenzo Angelini",
        cache=_cache(),
    )

    assert observation["status"] == "identity_conflict"
    assert observation["height_cm"] is None


def test_abbreviated_search_hit_does_not_bind_to_full_canonical_name(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_get_json",
        lambda _url: _search(_item("M Rifqi Fitriadi", 14297)),
    )

    observation = resolver.resolve_height_observation(
        canonical_player_id=44,
        player_name="Muhammad Rifqi Fitriadi",
        cache=_cache(),
    )

    assert observation["status"] == "not_found"
    assert observation["height_cm"] is None


def test_conflicting_athlete_body_fails_write_boundary(monkeypatch):
    def get_json(url):
        if "apis/search" in url:
            return _search(_item("Flynn Thomas", 16770))
        return _athlete("Different Thomas", 16770, height=67)

    monkeypatch.setattr(resolver, "_get_json", get_json)
    observation = resolver.resolve_height_observation(
        canonical_player_id=45,
        player_name="Flynn Thomas",
        cache=_cache(),
    )

    assert observation["status"] == "identity_mismatch"
    assert resolver.validated_resolved_height(
        observation,
        canonical_player_id=45,
        player_name="Flynn Thomas",
    ) is None


def test_cached_positive_is_bound_to_canonical_id_and_body_hash(monkeypatch):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "status": "resolved",
        "canonical_player_id": 46,
        "player_name": "Exact Player",
        "observed_at": now,
        "height_cm": 188.0,
        "source_uri": (
            "https://sports.core.api.espn.com/v2/sports/tennis/"
            "leagues/atp/athletes/999"
        ),
        "source_content_sha256": "a" * 64,
        "external_player_id": "999",
        "identity_binding": resolver.IDENTITY_BINDING,
    }
    cache = _cache()
    cache["entries"][resolver._candidate_key(46, "Exact Player")] = entry
    monkeypatch.setattr(
        resolver,
        "_get_json",
        lambda _url: (_ for _ in ()).throw(AssertionError("cache should satisfy")),
    )

    hit = resolver.resolve_height_observation(
        canonical_player_id=46,
        player_name="Exact Player",
        cache=cache,
    )

    assert hit["cache_hit"] is True
    assert hit["attempt_count"] == 0
    assert resolver.validated_resolved_height(
        hit,
        canonical_player_id=47,
        player_name="Exact Player",
    ) is None


def test_batch_limit_defers_uncached_rows_without_querying(monkeypatch):
    monkeypatch.setattr(resolver, "_load_cache", _cache)
    monkeypatch.setattr(resolver, "_save_cache", lambda _cache_value: None)
    monkeypatch.setattr(
        resolver,
        "_get_json",
        lambda _url: (_ for _ in ()).throw(AssertionError("zero budget must not query")),
    )

    results = resolver.batch_height_observations(
        [{"canonical_player_id": 48, "player_name": "Deferred Player"}],
        limit=0,
    )

    assert results["Deferred Player"]["status"] == "deferred"
    assert results["Deferred Player"]["attempt_count"] == 0


def test_transient_failure_expires_before_authoritative_negative(monkeypatch):
    observed = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(
        timespec="seconds"
    )
    cache = _cache()
    cache["entries"][resolver._candidate_key(49, "Retry Player")] = {
        "status": "fetch_error",
        "canonical_player_id": 49,
        "player_name": "Retry Player",
        "observed_at": observed,
        "height_cm": None,
        "source_uri": "",
        "source_content_sha256": "",
        "external_player_id": "",
        "identity_binding": "",
    }
    monkeypatch.setattr(
        resolver,
        "_get_json",
        lambda _url: _search(_item("Retry Player", 17000)),
    )

    observation = resolver.resolve_height_observation(
        canonical_player_id=49,
        player_name="Retry Player",
        cache=cache,
    )

    # The expired transient was retried. The synthetic search body cannot act
    # as an athlete document, so this proves retry without fabricating height.
    assert observation["cache_hit"] is False
    assert observation["status"] == "identity_mismatch"
