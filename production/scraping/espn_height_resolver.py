"""Bounded ESPN height fallback with exact identity and source evidence.

ESPN's public tennis JSON surfaces can cover players whose official ATP profile
is unavailable from a cloud runner.  This module deliberately treats ESPN as a
secondary compatibility source: a value is usable only when one unique ESPN
athlete search result and the athlete document both match the canonical full
name exactly.  Ambiguous names, abbreviations, missing bodies, and implausible
values fail closed.

The returned observation carries the canonical player ID, ESPN athlete ID,
source URI, observation time, body hash, and identity-binding method.  That is
the minimum evidence needed before the legacy write-through may improve live
feature completeness.  Required eligibility-provenance mode bypasses this
compatibility hydrator in the caller.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Optional


MIN_HEIGHT_CM = 150.0
MAX_HEIGHT_CM = 230.0
CACHE_SCHEMA_VERSION = "espn_height_cache@1.0.0"
IDENTITY_BINDING = "espn_exact_search_name_plus_exact_athlete_name"

_SEARCH_URL = "https://site.web.api.espn.com/apis/search/v2?query={query}"
_ATHLETE_URL = (
    "https://sports.core.api.espn.com/v2/sports/tennis/leagues/atp/athletes/{aid}"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT_S = 15
_NEGATIVE_TTL = timedelta(days=7)
_TRANSIENT_TTL = timedelta(hours=1)
_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "espn_heights.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _identity_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return "".join(char for char in ascii_text.casefold() if char.isalnum())


def _candidate_key(canonical_player_id: int, player_name: str) -> str:
    return f"{int(canonical_player_id)}:{_identity_key(player_name)}"


def _validated_cm_from_inches(value: object) -> Optional[float]:
    try:
        inches = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(inches):
        return None
    cm = round(inches * 2.54, 1)
    return cm if MIN_HEIGHT_CM <= cm <= MAX_HEIGHT_CM else None


def _validated_stored_cm(value: object) -> Optional[float]:
    try:
        cm = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(cm) or not MIN_HEIGHT_CM <= cm <= MAX_HEIGHT_CM:
        return None
    return cm


def _canonical_json_sha256(payload: dict) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return sha256(body).hexdigest()


def _get_json(url: str) -> Optional[dict]:
    try:
        request = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_cache() -> dict:
    try:
        payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CACHE_SCHEMA_VERSION
        or not isinstance(payload.get("entries"), dict)
    ):
        return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
    return payload


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary.replace(_CACHE_PATH)


def _cached_observation(
    cache: dict,
    *,
    canonical_player_id: int,
    player_name: str,
) -> Optional[dict]:
    entry = (cache.get("entries") or {}).get(
        _candidate_key(canonical_player_id, player_name)
    )
    if not isinstance(entry, dict):
        return None
    try:
        if int(entry.get("canonical_player_id")) != int(canonical_player_id):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if _identity_key(entry.get("player_name")) != _identity_key(player_name):
        return None
    try:
        observed_at = datetime.fromisoformat(str(entry.get("observed_at", "")))
    except (TypeError, ValueError):
        return None
    if observed_at.tzinfo is None or observed_at > _now() + timedelta(minutes=5):
        return None

    status = str(entry.get("status") or "")
    if status == "resolved":
        if entry.get("identity_binding") != IDENTITY_BINDING:
            return None
        if re.fullmatch(r"[0-9a-f]{64}", str(entry.get("source_content_sha256", ""))) is None:
            return None
        if not str(entry.get("source_uri") or "").startswith(
            "https://sports.core.api.espn.com/"
        ):
            return None
        if _validated_stored_cm(entry.get("height_cm")) is None:
            return None
        return {**entry, "cache_hit": True, "attempt_count": 0}
    ttl = _TRANSIENT_TTL if status == "fetch_error" else _NEGATIVE_TTL
    if _now() <= observed_at.astimezone(timezone.utc) + ttl:
        return {**entry, "cache_hit": True, "attempt_count": 0}
    return None


def validated_resolved_height(
    observation: object,
    *,
    canonical_player_id: int,
    player_name: str,
) -> Optional[float]:
    """Revalidate a resolved observation at the compatibility write boundary."""
    if not isinstance(observation, dict) or observation.get("status") != "resolved":
        return None
    try:
        if int(observation.get("canonical_player_id")) != int(canonical_player_id):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if _identity_key(observation.get("player_name")) != _identity_key(player_name):
        return None
    if observation.get("identity_binding") != IDENTITY_BINDING:
        return None
    if not str(observation.get("external_player_id") or "").isdigit():
        return None
    if not str(observation.get("source_uri") or "").startswith(
        "https://sports.core.api.espn.com/"
    ):
        return None
    if re.fullmatch(
        r"[0-9a-f]{64}", str(observation.get("source_content_sha256") or ""),
    ) is None:
        return None
    return _validated_stored_cm(observation.get("height_cm"))


def _name_variants(player_name: str) -> list[str]:
    parts = str(player_name or "").strip().split()
    variants = [" ".join(parts)]
    if len(parts) >= 3:
        variants.append(f"{parts[0]} {parts[-1]}")
        variants.append(" ".join(parts[1:]))
    seen: set[str] = set()
    return [
        value for value in variants
        if (key := _identity_key(value)) and not (key in seen or seen.add(key))
    ]


def resolve_height_observation(
    *,
    canonical_player_id: int,
    player_name: str,
    cache: Optional[dict] = None,
) -> dict:
    """Return one exact, evidence-bearing ESPN height observation."""
    player_id = int(canonical_player_id)
    name = " ".join(str(player_name or "").strip().split())
    if player_id <= 0 or not _identity_key(name):
        raise ValueError("canonical player ID and player name are required")

    owned_cache = cache is None
    working_cache = _load_cache() if owned_cache else cache
    cached = _cached_observation(
        working_cache, canonical_player_id=player_id, player_name=name,
    )
    if cached is not None:
        return cached

    observed_at = _now().isoformat(timespec="seconds")
    exact_candidates: dict[str, dict] = {}
    attempts = 0
    search_failed = False
    for variant in _name_variants(name):
        search_url = _SEARCH_URL.format(query=urllib.parse.quote(variant))
        document = _get_json(search_url)
        attempts += 1
        if document is None:
            search_failed = True
            continue
        for result in document.get("results", []):
            if not isinstance(result, dict):
                continue
            for item in result.get("contents", []):
                if not isinstance(item, dict):
                    continue
                if str(item.get("sport") or "").casefold() != "tennis":
                    continue
                if str(item.get("type") or "").casefold() != "player":
                    continue
                if _identity_key(item.get("displayName")) != _identity_key(name):
                    continue
                uid = str(item.get("uid") or "")
                match = re.search(r"(?:^|~)a:(\d+)(?:~|$)", uid)
                if match:
                    exact_candidates[match.group(1)] = item

    base = {
        "canonical_player_id": player_id,
        "player_name": name,
        "observed_at": observed_at,
        "height_cm": None,
        "source_uri": "",
        "source_content_sha256": "",
        "external_player_id": "",
        "identity_binding": "",
        "cache_hit": False,
        "attempt_count": attempts,
    }
    if len(exact_candidates) > 1:
        observation = {**base, "status": "identity_conflict"}
    elif not exact_candidates:
        observation = {
            **base,
            "status": "fetch_error" if search_failed else "not_found",
        }
    else:
        external_id = next(iter(exact_candidates))
        source_uri = _ATHLETE_URL.format(aid=external_id)
        athlete = _get_json(source_uri)
        attempts += 1
        if athlete is None:
            observation = {
                **base,
                "status": "fetch_error",
                "source_uri": source_uri,
                "external_player_id": external_id,
                "attempt_count": attempts,
            }
        elif (
            str(athlete.get("id") or "") != external_id
            or _identity_key(athlete.get("fullName") or athlete.get("displayName"))
            != _identity_key(name)
        ):
            observation = {
                **base,
                "status": "identity_mismatch",
                "source_uri": source_uri,
                "source_content_sha256": _canonical_json_sha256(athlete),
                "external_player_id": external_id,
                "attempt_count": attempts,
            }
        else:
            height = _validated_cm_from_inches(athlete.get("height"))
            observation = {
                **base,
                "status": "resolved" if height is not None else "not_found",
                "height_cm": height,
                "source_uri": source_uri,
                "source_content_sha256": _canonical_json_sha256(athlete),
                "external_player_id": external_id,
                "identity_binding": IDENTITY_BINDING,
                "attempt_count": attempts,
            }

    entries = working_cache.setdefault("entries", {})
    entries[_candidate_key(player_id, name)] = {
        key: value for key, value in observation.items()
        if key not in {"cache_hit", "attempt_count"}
    }
    if owned_cache:
        _save_cache(working_cache)
    return observation


def batch_height_observations(
    candidates: list[dict],
    *,
    limit: Optional[int] = None,
    throttle_seconds: float = 0.35,
    verbose: bool = False,
) -> dict[str, dict]:
    """Resolve a bounded canonical-player batch using one durable cache load."""
    configured_limit = (
        int(os.environ.get("ESPN_PROFILE_RUN_HYDRATION_LIMIT", "64"))
        if limit is None else int(limit)
    )
    budget = max(0, configured_limit)
    cache = _load_cache()
    output: dict[str, dict] = {}
    attempted_live = False
    seen_ids: set[int] = set()
    for candidate in candidates:
        player_id = int(candidate["canonical_player_id"])
        name = " ".join(str(candidate["player_name"]).strip().split())
        if player_id in seen_ids:
            continue
        seen_ids.add(player_id)
        cached = _cached_observation(
            cache, canonical_player_id=player_id, player_name=name,
        )
        if cached is not None:
            output[name] = cached
            continue
        if budget <= 0:
            output[name] = {
                "status": "deferred",
                "canonical_player_id": player_id,
                "player_name": name,
                "height_cm": None,
                "cache_hit": False,
                "attempt_count": 0,
            }
            continue
        if attempted_live and throttle_seconds > 0:
            time.sleep(throttle_seconds)
        attempted_live = True
        budget -= 1
        observation = resolve_height_observation(
            canonical_player_id=player_id,
            player_name=name,
            cache=cache,
        )
        output[name] = observation
        if verbose:
            print(
                f"  ESPN height {name}: {observation.get('status')} "
                f"{observation.get('height_cm') or ''}"
            )
    _save_cache(cache)
    return output
