#!/usr/bin/env python3
"""
ATP Height Scraper

Fallback height lookup from atptour.com for players where Tennis Abstract
returns height_cm=None.

Strategy:
  1. Load player profile URLs from the ATP rankings CSV (populated by atp_rankings_scraper.py).
  2. Bind the player name exactly (accent/punctuation normalized) to that URL.
  3. Navigate to their /bio page with Playwright (JS-rendered).
  4. Extract height in cm via regex.
  5. In default legacy mode, retain positive JSON cache values while expiring
     source-bound negative observations after a bounded TTL.
     Set ATP_PROFILE_REVALIDATE_LEGACY_POSITIVES=1 to withhold name-keyed
     legacy positives until the official page identity, body hash, and exact
     extracted value have been revalidated. This remains compatibility
     evidence, not canonical-player-ID provenance.
     After explicit eligibility cutover (ELIGIBILITY_PROVENANCE_MODE=required),
     read only a schema/generation-pinned ops export and never write locally.

Usage (standalone test):
    python -m production.scraping.atp_height_scraper "Jacob Fearnley"
"""

import json
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import pandas as pd

try:
    from storage.eligibility import (
        ELIGIBILITY_GENERATION_ENV, ELIGIBILITY_PROJECTION_SEAL_ENV,
        EligibilityContractError, EligibilityMode, eligibility_mode,
    )
    from eligibility_cache import (
        VerifiedEligibilityBundle, load_verified_profile_bundle,
    )
except ImportError:  # pragma: no cover - package-style execution
    from production.storage.eligibility import (  # type: ignore
        ELIGIBILITY_GENERATION_ENV, ELIGIBILITY_PROJECTION_SEAL_ENV,
        EligibilityContractError, EligibilityMode, eligibility_mode,
    )
    from production.eligibility_cache import (  # type: ignore
        VerifiedEligibilityBundle, load_verified_profile_bundle,
    )

CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "atp_heights.json"
HANDS_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "atp_hands.json"
PROFILE_LOOKUP_META_PATH = (
    Path(__file__).parent.parent.parent / "data" / "atp_profile_lookup_meta.json"
)
CACHE_MANIFEST_PATH = (
    Path(__file__).parent.parent.parent / "data" / "eligibility_cache_manifest.json"
)
RANKINGS_PATH = Path(__file__).parent.parent.parent / "data" / "atp_rankings.csv"
ATP_BASE = "https://www.atptour.com"
DEFAULT_NEGATIVE_TTL_HOURS = 24 * 7
DEFAULT_TRANSIENT_TTL_MINUTES = 60
DEFAULT_NEGATIVE_REFRESH_LIMIT = 8
MAX_METADATA_FUTURE_SKEW = timedelta(minutes=5)
OFFICIAL_PAGE_IDENTITY_BINDING = (
    "official_rankings_or_slug_name_plus_rendered_full_name"
)
OFFICIAL_PAGE_CONFLICT_BINDING = (
    "official_rendered_profile_fields_plus_conflicting_full_name"
)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_required_bundle() -> Optional[VerifiedEligibilityBundle]:
    """Load the all-or-nothing accepted bundle configured for this process."""
    generation = os.environ.get(ELIGIBILITY_GENERATION_ENV, "").strip().lower()
    seal = os.environ.get(ELIGIBILITY_PROJECTION_SEAL_ENV, "").strip().lower()
    if not generation or not seal:
        return None
    return load_verified_profile_bundle(
        output_dir=CACHE_MANIFEST_PATH.parent,
        expected_generation_sha256=generation,
        expected_projection_seal_sha256=seal,
    )


def _load_cache() -> dict:
    if _provenance_required():
        # Required mode must never expose a plain cache dictionary as accepted
        # evidence. Public lookups consume the verified ID-bearing bundle.
        return {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    if _provenance_required():
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(name: str) -> str:
    return name.strip().lower()


def _load_hands_cache() -> dict:
    if _provenance_required():
        return {}
    if HANDS_CACHE_PATH.exists():
        with open(HANDS_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_hands_cache(cache: dict):
    if _provenance_required():
        return
    HANDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HANDS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _load_profile_lookup_meta() -> dict:
    """Load legacy lookup evidence used only to expire negative cache rows."""
    if _provenance_required() or not PROFILE_LOOKUP_META_PATH.exists():
        return {}
    try:
        with open(PROFILE_LOOKUP_META_PATH) as f:
            raw = json.load(f)
    except (OSError, ValueError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_profile_lookup_meta(metadata: dict) -> None:
    if _provenance_required():
        return
    PROFILE_LOOKUP_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_LOOKUP_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _negative_ttl() -> timedelta:
    raw = os.environ.get(
        "ATP_PROFILE_NEGATIVE_TTL_HOURS", str(DEFAULT_NEGATIVE_TTL_HOURS)
    )
    try:
        hours = max(0.0, float(raw))
    except (TypeError, ValueError, OverflowError):
        hours = float(DEFAULT_NEGATIVE_TTL_HOURS)
    return timedelta(hours=hours)


def _negative_refresh_limit() -> int:
    raw = os.environ.get(
        "ATP_PROFILE_NEGATIVE_REFRESH_LIMIT", str(DEFAULT_NEGATIVE_REFRESH_LIMIT)
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_NEGATIVE_REFRESH_LIMIT


def _revalidate_legacy_positives() -> bool:
    return os.environ.get(
        "ATP_PROFILE_REVALIDATE_LEGACY_POSITIVES", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _transient_ttl() -> timedelta:
    raw = os.environ.get(
        "ATP_PROFILE_TRANSIENT_TTL_MINUTES", str(DEFAULT_TRANSIENT_TTL_MINUTES)
    )
    try:
        minutes = max(0.0, float(raw))
    except (TypeError, ValueError, OverflowError):
        minutes = float(DEFAULT_TRANSIENT_TTL_MINUTES)
    return timedelta(minutes=minutes)


def _absolute_profile_url(profile_url: Optional[str], key: str) -> str:
    if profile_url:
        return ATP_BASE + profile_url if profile_url.startswith("/") else profile_url
    # This source changes automatically if a future rankings refresh discovers
    # an official ATP URL, invalidating the prior no-URL negative observation.
    return f"atp-rankings://unmapped/{key.replace(' ', '%20')}"


def _valid_cached_height(value) -> Optional[int]:
    try:
        height = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not height.is_integer() or not 150 <= height <= 230:
        return None
    return int(height)


def _valid_cached_hand(value) -> Optional[str]:
    hand = str(value or "").strip().upper()
    return hand if hand in {"R", "L"} else None


def _negative_cache_is_fresh(
    entry: object,
    *,
    source_uri: str,
    missing_fields: set[str],
    now: Optional[datetime] = None,
) -> bool:
    """Return whether the same source recently lacked all requested fields."""
    if not isinstance(entry, dict) or entry.get("source_uri") != source_uri:
        return False
    recorded_missing = entry.get("missing_fields")
    if not isinstance(recorded_missing, list):
        return False
    if not missing_fields.issubset({str(field) for field in recorded_missing}):
        return False
    try:
        observed = datetime.fromisoformat(str(entry.get("observed_at", "")))
    except (TypeError, ValueError):
        return False
    if observed.tzinfo is None:
        return False
    current = now or _utc_now()
    observed_utc = observed.astimezone(timezone.utc)
    if observed_utc > current.astimezone(timezone.utc) + MAX_METADATA_FUTURE_SKEW:
        return False
    # Historical rows labeled every non-empty interstitial/error page as an
    # identity mismatch. Only a mismatch with explicit conflicting-profile
    # markers deserves the long negative TTL; old/unproven mismatches retry on
    # the short transient cadence.
    transient = entry.get("status") == "fetch_error" or (
        entry.get("status") == "identity_mismatch"
        and entry.get("identity_binding") != OFFICIAL_PAGE_CONFLICT_BINDING
    )
    ttl = _transient_ttl() if transient else _negative_ttl()
    return current <= observed_utc + ttl


def _positive_cache_field_is_evidenced(
    entry: object,
    *,
    field: str,
    value,
    source_uri: str,
    now: Optional[datetime] = None,
    canonical_player_id: Optional[int] = None,
) -> bool:
    """Validate opt-in evidence for one legacy positive cache field."""
    if field not in {"height_cm", "hand"} or not isinstance(entry, dict):
        return False
    if entry.get("source_uri") != source_uri:
        return False
    parsed = urlparse(source_uri)
    if parsed.scheme != "https" or parsed.hostname not in {
        "atptour.com", "www.atptour.com",
    }:
        return False
    if entry.get("identity_binding") != OFFICIAL_PAGE_IDENTITY_BINDING:
        return False
    if canonical_player_id is not None:
        try:
            if int(entry.get("canonical_player_id")) != int(canonical_player_id):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    if entry.get("status") not in {"resolved", "partial"}:
        return False
    content_hash = str(entry.get("source_content_sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
        return False
    missing_fields = entry.get("missing_fields")
    if not isinstance(missing_fields, list) or field in missing_fields:
        return False
    observed_values = entry.get("observed_values")
    if not isinstance(observed_values, dict) or field not in observed_values:
        return False
    if field == "height_cm":
        if _valid_cached_height(observed_values[field]) != _valid_cached_height(value):
            return False
    elif _valid_cached_hand(observed_values[field]) != _valid_cached_hand(value):
        return False
    try:
        observed = datetime.fromisoformat(str(entry.get("observed_at", "")))
    except (TypeError, ValueError):
        return False
    if observed.tzinfo is None:
        return False
    current = (now or _utc_now()).astimezone(timezone.utc)
    return observed.astimezone(timezone.utc) <= current + MAX_METADATA_FUTURE_SKEW


def _record_lookup(
    metadata: dict,
    *,
    key: str,
    source_uri: str,
    height_cm: Optional[int],
    hand: Optional[str],
    status: str,
    source_content_sha256: str = "",
    identity_binding: str = "",
    canonical_player_id: Optional[int] = None,
) -> None:
    missing = []
    if height_cm is None:
        missing.append("height_cm")
    if hand is None:
        missing.append("hand")
    record = {
        "source_uri": source_uri,
        "observed_at": _utc_now().isoformat(timespec="seconds"),
        "status": status,
        "missing_fields": missing,
        "source_content_sha256": source_content_sha256,
        "identity_binding": identity_binding,
        "observed_values": (
            {
                field: value for field, value in (
                    ("height_cm", height_cm), ("hand", hand),
                ) if value is not None
            }
            if identity_binding == OFFICIAL_PAGE_IDENTITY_BINDING
            and status in {"resolved", "partial"}
            else {}
        ),
    }
    if canonical_player_id is not None:
        player_id = int(canonical_player_id)
        if player_id <= 0:
            raise ValueError("canonical_player_id must be positive")
        record["canonical_player_id"] = player_id
    metadata[key] = record


def _provenance_required() -> bool:
    return eligibility_mode() is EligibilityMode.REQUIRED


def _new_browser_page():
    """Resolve BrowserSession in both package and legacy script execution."""
    try:
        from .browser_session import new_page
    except ImportError:  # pragma: no cover - legacy production/ on sys.path
        from browser_session import new_page
    return new_page()


# ---------------------------------------------------------------------------
# URL lookup from rankings CSV
# ---------------------------------------------------------------------------

def _load_url_map() -> dict:
    """
    Build {normalized_name: bio_url} from atp_rankings.csv.
    Requires the CSV to have a player_url column (populated when rankings
    scraper is re-run after the atp_rankings_scraper.py update).
    """
    if not RANKINGS_PATH.exists():
        return {}
    df = pd.read_csv(RANKINGS_PATH)
    if "player_url" not in df.columns:
        return {}

    url_map = {}
    ambiguous = set()
    for _, row in df.iterrows():
        url = row.get("player_url")
        name = str(row.get("player_name", "")).strip()
        if pd.isna(url) or not url or not name:
            continue
        candidate = str(url)
        keys = {
            key for key in (
                _identity_key(name),
                _official_profile_slug_identity_key(candidate),
            )
            if key
        }
        for key in keys:
            if key in ambiguous:
                continue
            if key in url_map and url_map[key] != candidate:
                url_map.pop(key, None)
                ambiguous.add(key)
            else:
                url_map[key] = candidate
    for key in ambiguous:
        url_map.pop(key, None)
    return url_map


def _identity_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return "".join(char for char in ascii_text.casefold() if char.isalnum())


def _official_profile_slug_identity_key(profile_url: str) -> str:
    """Derive a full-name key from ATP's official `/players/<slug>/` path."""
    try:
        path = urlparse(str(profile_url or "")).path
    except (TypeError, ValueError):
        return ""
    match = re.search(r"/players/([^/]+)/", path, flags=re.IGNORECASE)
    if not match:
        return ""
    slug = unquote(match.group(1)).strip(" -")
    parts = [part for part in slug.split("-") if part]
    # A single surname/initial is not a safe identity bind. ATP's current
    # official profile URLs expose full-name slugs (for example
    # `jannik-sinner`), which the rendered-page check independently confirms.
    if len(parts) < 2 or any(not _identity_key(part) for part in parts):
        return ""
    return _identity_key(" ".join(parts))


def _find_profile_url(player_name: str, url_map: dict) -> Optional[str]:
    """
    Find an ATP bio URL through an exact normalized name/full-name-slug bind.

    The old unique-surname/first-initial fallback could bind two different
    people. It remains unsuitable for an automatic feature that can make a row
    bet-eligible, so ambiguous/fuzzy candidates now stay unresolved.
    """
    key = _identity_key(player_name)
    # Accept both the normalized maps produced above and test/legacy callers
    # that still provide lower-case display keys.
    return url_map.get(key) or url_map.get(str(player_name or "").strip().lower())


def _profile_text_matches_name(text: str, player_name: str) -> bool:
    """Require the requested full name to appear on the rendered ATP page."""
    requested = _identity_key(player_name)
    rendered = _identity_key(text)
    return bool(requested and requested in rendered)


def _looks_like_rendered_profile(text: str) -> bool:
    """Separate a real wrong-player profile from a block/interstitial page."""
    value = str(text or "").casefold()
    markers = (
        "personal details",
        "career high rank",
        "height",
        "plays",
        "country",
        "birthplace",
    )
    return sum(marker in value for marker in markers) >= 2




# ---------------------------------------------------------------------------
# Height extraction
# ---------------------------------------------------------------------------

def _extract_height_cm(text: str) -> Optional[int]:
    """Extract height in cm from rendered bio page text."""
    # ATP shows: 6'0" (183cm) or 183 cm or 183cm
    m = re.search(r'(\d{2,3})\s*cm', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 150 <= val <= 230:
            return val
    return None


def _extract_hand(text: str) -> Optional[str]:
    """'Plays: Right-Handed' / 'Left-Handed' on the bio page -> 'R'/'L'."""
    m = re.search(r"(right|left)\s*-?\s*handed", text, re.IGNORECASE)
    if m:
        return "R" if m.group(1).lower() == "right" else "L"
    return None


def _fetch_profile_text(page, profile_url: str) -> str:
    full_url = ATP_BASE + profile_url if profile_url.startswith("/") else profile_url
    try:
        page.goto(full_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(5)
        return page.inner_text("body")
    except Exception as e:
        print(f"  ATP profile page error ({full_url}): {e}")
        return ""


def _scrape_profile(page, profile_url: str) -> Optional[int]:
    """Navigate to an ATP overview page and extract height."""
    return _extract_height_cm(_fetch_profile_text(page, profile_url))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_lookup_evidence_states(
    player_names: list,
    *,
    now: Optional[datetime] = None,
    require_evidenced_positives: bool = False,
    canonical_player_ids: Optional[dict] = None,
) -> dict:
    """Describe current height evidence without fetching or mutating anything.

    The run-level planner uses these exact source-bound states to order work.
    Fresh negative evidence remains authoritative until its TTL expires; an
    unobserved or expired source can enter the bounded browser batch.
    """
    if _provenance_required():
        bundle = _load_required_bundle()
        states = {}
        for name in player_names:
            profile = None if bundle is None else bundle.profile_for(name)
            states[name] = {
                "state": (
                    "resolved"
                    if profile is not None
                    and _valid_cached_height(profile.get("height_cm")) is not None
                    else "fresh_negative"
                ),
                "source_uri": "accepted-eligibility-bundle",
                "has_profile_url": False,
            }
        return states

    h_cache = _load_cache()
    metadata = _load_profile_lookup_meta()
    url_map = _load_url_map()
    revalidate_positive = (
        _revalidate_legacy_positives() or require_evidenced_positives
    )
    states = {}
    for name in player_names:
        key = _cache_key(name)
        expected_player_id = (canonical_player_ids or {}).get(key)
        bio_url = _find_profile_url(name, url_map)
        source_uri = _absolute_profile_url(bio_url, key)
        entry = metadata.get(key)
        cached_height = _valid_cached_height(h_cache.get(key))
        positive_is_accepted = cached_height is not None and (
            not revalidate_positive
            or _positive_cache_field_is_evidenced(
                entry,
                field="height_cm",
                value=cached_height,
                source_uri=source_uri,
                now=now,
                canonical_player_id=expected_player_id,
            )
        )
        if positive_is_accepted:
            state = "resolved"
        elif _negative_cache_is_fresh(
            entry,
            source_uri=source_uri,
            missing_fields={"height_cm"},
            now=now,
        ):
            state = "fresh_negative"
        elif (
            isinstance(entry, dict)
            and entry.get("source_uri") == source_uri
            and "height_cm" in {
                str(field) for field in (entry.get("missing_fields") or [])
            }
        ):
            state = "expired_negative"
        else:
            state = "unobserved"
        states[name] = {
            "state": state,
            "source_uri": source_uri,
            "has_profile_url": bool(bio_url),
        }
    return states

def get_height_cm(
    player_name: str,
    cache: Optional[dict | VerifiedEligibilityBundle] = None,
) -> Optional[int]:
    """
    Return height in cm for player_name from ATP website.
    Checks cache first; only launches Playwright when needed.
    """
    if _provenance_required():
        if cache is not None:
            raise EligibilityContractError(
                "required eligibility mode rejects caller-supplied cache objects"
            )
        bundle = _load_required_bundle()
        profile = None if bundle is None else bundle.profile_for(player_name)
        value = None if profile is None else profile.get("height_cm")
        return None if value is None else int(float(value))

    own_cache = cache is None
    if cache is None:
        cache = _load_cache()
    assert isinstance(cache, dict)

    key = _cache_key(player_name)
    cached_height = _valid_cached_height(cache.get(key))
    metadata = _load_profile_lookup_meta()
    revalidate_positive = _revalidate_legacy_positives()
    url_map = None
    bio_url = None
    source_uri = ""
    if cached_height is not None and revalidate_positive and own_cache:
        url_map = _load_url_map()
        bio_url = _find_profile_url(player_name, url_map)
        source_uri = _absolute_profile_url(bio_url, key)
        if _positive_cache_field_is_evidenced(
            metadata.get(key),
            field="height_cm",
            value=cached_height,
            source_uri=source_uri,
        ):
            return cached_height
    elif cached_height is not None:
        return cached_height
    # Caller-owned dictionaries have no durable lookup-evidence sidecar. Keep
    # their historical first-wins behavior unless opt-in positive revalidation
    # explicitly withholds values that cannot carry durable evidence.
    if not own_cache and key in cache:
        return None

    if url_map is None:
        url_map = _load_url_map()
        bio_url = _find_profile_url(player_name, url_map)
        source_uri = _absolute_profile_url(bio_url, key)
    if key in cache and _negative_cache_is_fresh(
        metadata.get(key),
        source_uri=source_uri,
        missing_fields={"height_cm"},
    ):
        return None

    if not bio_url:
        print(f"  ATP: no URL found for '{player_name}' (re-run atp_rankings_scraper.py to refresh)")
        cache[key] = None
        _record_lookup(
            metadata,
            key=key,
            source_uri=source_uri,
            height_cm=None,
            hand=None,
            status="no_url",
        )
        if own_cache:
            _save_cache(cache)
            _save_profile_lookup_meta(metadata)
        return None

    print(f"  ATP height lookup: {player_name}")
    text = ""
    pg = None
    try:
        pg = _new_browser_page()
        text = _fetch_profile_text(pg, bio_url)
    except Exception as exc:
        print(f"  ATP profile browser error ({source_uri}): {exc}")
    finally:
        if pg is not None:
            try:
                pg.close()
            except Exception:
                pass

    identity_matches = _profile_text_matches_name(text, player_name)
    height = _extract_height_cm(text) if identity_matches else None

    if height is not None or key not in cache:
        cache[key] = height
    if own_cache:
        _save_cache(cache)
        _record_lookup(
            metadata,
            key=key,
            source_uri=source_uri,
            height_cm=height,
            hand=None,
            status=(
                "resolved" if height is not None
                else "identity_mismatch" if str(text or "").strip() and not identity_matches
                else "not_found" if str(text or "").strip()
                else "fetch_error"
            ),
            source_content_sha256=(
                sha256(text.encode("utf-8")).hexdigest()
                if str(text or "").strip() else ""
            ),
            identity_binding=(
                OFFICIAL_PAGE_IDENTITY_BINDING if identity_matches else ""
            ),
        )
        _save_profile_lookup_meta(metadata)

    if height is not None:
        print(f"  ATP height found: {player_name} → {height}cm")
    else:
        print(f"  ATP height not found: {player_name}")
    return height


def batch_get_profiles(
    player_names: list,
    verbose: bool = True,
    refresh_state: Optional[dict] = None,
) -> dict:
    """
    Fetch height AND handedness for multiple players, one page fetch each,
    shared browser. Returns {name: {"height_cm": int|None, "hand": 'R'/'L'/None}}.
    Default legacy mode retains positive persistent caches. Missing fields are
    retried only after their source-bound evidence TTL, with a run-scoped
    refresh budget supplied by the caller. Required cutover mode reads only a
    generation-pinned export and returns the accepted canonical player ID with
    each profile so callers can reject identity mismatches. Fresh values remain
    compatibility evidence pending normalized ingestion/review.
    """
    if _provenance_required():
        bundle = _load_required_bundle()
        results: dict = {}
        for name in player_names:
            profile = None if bundle is None else bundle.profile_for(name)
            results[name] = {
                "canonical_player_id": (
                    None if profile is None else profile.get("canonical_player_id")
                ),
                "height_cm": None if profile is None else profile.get("height_cm"),
                "hand": None if profile is None else profile.get("hand"),
            }
        return results

    state = refresh_state if refresh_state is not None else {}
    canonical_player_ids = state.get("canonical_player_ids") or {}
    if not isinstance(canonical_player_ids, dict):
        canonical_player_ids = {}

    def _canonical_player_id(key: str) -> Optional[int]:
        value = canonical_player_ids.get(key)
        try:
            player_id = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return player_id if player_id > 0 else None

    h_cache = _load_cache()
    hd_cache = _load_hands_cache()
    metadata = _load_profile_lookup_meta()
    revalidate_positive = (
        _revalidate_legacy_positives()
        or bool(state.get("require_evidenced_positives"))
    )
    url_map = _load_url_map() if revalidate_positive else None
    results: dict = {}
    incomplete = []

    for name in player_names:
        key = _cache_key(name)
        height = _valid_cached_height(h_cache.get(key))
        hand = _valid_cached_hand(hd_cache.get(key))
        if revalidate_positive:
            assert url_map is not None
            bio_url = _find_profile_url(name, url_map)
            source_uri = _absolute_profile_url(bio_url, key)
            if height is not None and not _positive_cache_field_is_evidenced(
                metadata.get(key),
                field="height_cm",
                value=height,
                source_uri=source_uri,
                canonical_player_id=_canonical_player_id(key),
            ):
                height = None
            if hand is not None and not _positive_cache_field_is_evidenced(
                metadata.get(key),
                field="hand",
                value=hand,
                source_uri=source_uri,
                canonical_player_id=_canonical_player_id(key),
            ):
                hand = None
        if height is not None and hand is not None:
            results[name] = {"height_cm": height, "hand": hand}
        else:
            incomplete.append((name, key, height, hand))

    if not incomplete:
        return results

    if url_map is None:
        url_map = _load_url_map()
    needs_scraping = []
    no_url = []
    if "remaining" not in state:
        state["remaining"] = _negative_refresh_limit()
    try:
        refresh_slots = max(0, int(state["remaining"]))
    except (TypeError, ValueError, OverflowError):
        refresh_slots = 0
    attempted_keys = state.setdefault("attempted_keys", set())
    if not isinstance(attempted_keys, set):
        attempted_keys = set(attempted_keys or ())
        state["attempted_keys"] = attempted_keys
    allowed_keys = state.get("allowed_keys")
    if allowed_keys is not None and not isinstance(allowed_keys, set):
        allowed_keys = set(allowed_keys or ())
        state["allowed_keys"] = allowed_keys
    for name, key, height, hand in incomplete:
        bio_url = _find_profile_url(name, url_map)
        source_uri = _absolute_profile_url(bio_url, key)
        missing_fields = {
            field for field, value in (("height_cm", height), ("hand", hand))
            if value is None
        }
        if _negative_cache_is_fresh(
            metadata.get(key),
            source_uri=source_uri,
            missing_fields=missing_fields,
        ):
            results[name] = {"height_cm": height, "hand": hand}
        elif not bio_url:
            no_url.append((name, key, height, hand, source_uri))
        elif (
            refresh_slots > 0
            and key not in attempted_keys
            and (allowed_keys is None or key in allowed_keys)
        ):
            needs_scraping.append((name, key, height, hand, bio_url, source_uri))
            attempted_keys.add(key)
            refresh_slots -= 1
        else:
            # Bound stale-negative refresh work so an old cache cannot turn one
            # hourly pipeline run into an unbounded ATP crawl. Deferred rows
            # remain incomplete; fresh failures rotate to later rows next run.
            results[name] = {"height_cm": height, "hand": hand}
    state["remaining"] = refresh_slots

    for name, key, height, hand, source_uri in no_url:
        results[name] = {"height_cm": height, "hand": hand}
        if key not in h_cache:
            h_cache[key] = height
        if key not in hd_cache:
            hd_cache[key] = hand
        _record_lookup(
            metadata,
            key=key,
            source_uri=source_uri,
            height_cm=height,
            hand=hand,
            status="no_url",
            canonical_player_id=_canonical_player_id(key),
        )
        if verbose:
            print(f"  ATP: no URL for '{name}' — skipping profile lookup")

    if needs_scraping:
        if verbose:
            print(f"  ATP profile scraper: fetching {len(needs_scraping)} players...")
        pg = None
        try:
            pg = _new_browser_page()
            for name, key, cached_height, cached_hand, bio_url, source_uri in needs_scraping:
                try:
                    text = _fetch_profile_text(pg, bio_url)
                except Exception:
                    text = ""
                if not str(text or "").strip():
                    results[name] = {
                        "height_cm": cached_height,
                        "hand": cached_hand,
                    }
                    _record_lookup(
                        metadata,
                        key=key,
                        source_uri=source_uri,
                        height_cm=cached_height,
                        hand=cached_hand,
                        status="fetch_error",
                        canonical_player_id=_canonical_player_id(key),
                    )
                    continue
                if not _profile_text_matches_name(text, name):
                    results[name] = {
                        "height_cm": cached_height,
                        "hand": cached_hand,
                    }
                    real_profile_conflict = _looks_like_rendered_profile(text)
                    _record_lookup(
                        metadata,
                        key=key,
                        source_uri=source_uri,
                        height_cm=cached_height,
                        hand=cached_hand,
                        status=(
                            "identity_mismatch"
                            if real_profile_conflict
                            else "fetch_error"
                        ),
                        source_content_sha256=sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        identity_binding=(
                            OFFICIAL_PAGE_CONFLICT_BINDING
                            if real_profile_conflict
                            else ""
                        ),
                        canonical_player_id=_canonical_player_id(key),
                    )
                    continue
                observed_height = _extract_height_cm(text)
                observed_hand = _extract_hand(text)
                h = cached_height or observed_height
                hd = cached_hand or observed_hand
                results[name] = {"height_cm": h, "hand": hd}
                if observed_height is not None or key not in h_cache:
                    h_cache[key] = observed_height
                if observed_hand is not None or key not in hd_cache:
                    hd_cache[key] = observed_hand
                missing_count = int(observed_height is None) + int(observed_hand is None)
                status = (
                    "resolved" if missing_count == 0
                    else "partial" if missing_count == 1
                    else "not_found"
                )
                _record_lookup(
                    metadata,
                    key=key,
                    source_uri=source_uri,
                    height_cm=observed_height,
                    hand=observed_hand,
                    status=status,
                    source_content_sha256=sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    identity_binding=OFFICIAL_PAGE_IDENTITY_BINDING,
                    canonical_player_id=_canonical_player_id(key),
                )
                if verbose:
                    print(f"    {name}: {h or '?'}cm hand={hd or '?'}")
        except Exception:
            if pg is not None:
                raise
            # Browser launch failure is operationally transient. Preserve any
            # positive cached fields, keep missing fields incomplete, and let a
            # future run retry after the short transient TTL.
            for name, key, height, hand, _bio_url, source_uri in needs_scraping:
                if name in results:
                    continue
                results[name] = {"height_cm": height, "hand": hand}
                _record_lookup(
                    metadata,
                    key=key,
                    source_uri=source_uri,
                    height_cm=height,
                    hand=hand,
                    status="fetch_error",
                    canonical_player_id=_canonical_player_id(key),
                )
        finally:
            if pg is not None:
                try:
                    pg.close()
                except Exception:
                    pass

    _save_cache(h_cache)
    _save_hands_cache(hd_cache)
    _save_profile_lookup_meta(metadata)
    return results


def batch_get_heights(player_names: list, verbose: bool = True) -> dict:
    """Back-compat wrapper: heights only."""
    profs = batch_get_profiles(player_names, verbose=verbose)
    return {n: (v or {}).get("height_cm") for n, v in profs.items()}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Resolve one ATP player height")
    parser.add_argument("player_name", nargs="+", help="ATP player name")
    args = parser.parse_args(argv)
    name = " ".join(args.player_name)
    h = get_height_cm(name)
    print(f"\nResult: {name} → {h}cm" if h else f"\nResult: {name} → not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
