#!/usr/bin/env python3
"""
Tennis Abstract Feature Calculator
Mirrors LiveFeatureEngine logic but uses Tennis Abstract as data source.
Returns the exact 141 features expected by NN-141 model.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import os
import re
import sys

# Import TA scraper
sys.path.insert(0, str(Path(__file__).parent.parent / "scraping"))
from ta_scraper import TennisAbstractScraper
from atp_rankings_scraper import (
    get_player_lookup_status,
    get_player_points,
    get_player_rank,
    load_rankings,
)
from atp_height_scraper import batch_get_profiles, profile_lookup_evidence_states

# Import shared round offset function (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from round_offsets import get_round_day_offset, infer_draw_size
from history_stitch import (
    gather_atp_rows,
    gather_itf_rows,
    itf_round_for,
    infer_next_round_any,
    needs_stitching,
    round_from_draws,
    stitch_history,
)
from base_141_shared import (
    SEMANTICS_ID as SHARED_SEMANTICS_ID,
    as_of_day,
    feature_default as shared_feature_default,
    h2h_features_from_frame as shared_h2h_features_from_frame,
    normalize_player_snapshot,
    observations_from_records,
    player_temporal_features,
    surface_transition_flag as shared_surface_transition_flag,
)
from height_hydration import HeightHydrationCandidate, plan_height_hydration

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_contract import normalize_feature_vector
from versioning import LIVE_SEMANTICS_ID

# Import schema contract
import json
SCHEMA_PATH = Path(__file__).parent / "schema_141.json"
with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)
    EXACT_141_FEATURES = [feat["name"] for feat in SCHEMA["features"]]


class UnsafeToInferError(RuntimeError):
    """Raised when source evidence cannot support an unambiguous inference."""


MIN_HEIGHT_CM = 150.0
MAX_HEIGHT_CM = 230.0
DEFAULT_RUN_HEIGHT_HYDRATION_LIMIT = 32
DEFAULT_RUN_ITF_PROFILE_LIMIT = 48


def _validated_height_cm(value) -> Optional[float]:
    """Return a finite, physically plausible height or ``None``.

    The normalized eligibility contract already enforces this domain, but the
    still-active legacy compatibility path can contain old non-null garbage
    values.  Treat those values exactly like missing evidence so the existing
    default marker keeps the feature snapshot ineligible for betting/GOLD.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        height = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(height) or not MIN_HEIGHT_CM <= height <= MAX_HEIGHT_CM:
        return None
    return height


def _run_height_hydration_limit() -> int:
    raw = os.environ.get(
        "ATP_PROFILE_RUN_HYDRATION_LIMIT",
        str(DEFAULT_RUN_HEIGHT_HYDRATION_LIMIT),
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_RUN_HEIGHT_HYDRATION_LIMIT


def _run_itf_profile_limit() -> int:
    raw = os.environ.get(
        "ITF_PROFILE_RUN_HYDRATION_LIMIT",
        str(DEFAULT_RUN_ITF_PROFILE_LIMIT),
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_RUN_ITF_PROFILE_LIMIT


def reconcile_upcoming_surface(surface: str, ta_surface: str, metadata_source: str) -> Tuple[str, bool]:
    """Prefer explicit TA upcoming surface over heuristic/default tournament metadata."""
    if not ta_surface or ta_surface.lower() == str(surface or "").lower():
        return surface, False
    if metadata_source in {"fallback_heuristic", "default", "level_hint"}:
        return ta_surface, True
    return surface, False


def apply_round_offsets_to_history(df: pd.DataFrame, ref: datetime = None) -> pd.DataFrame:
    """Apply training's round offsets and quarantine non-historical rows.

    A history row whose inferred timestamp is at or after the upcoming match is
    not valid evidence for that prediction.  Older code moved such rows to one
    second before ``ref``; that converted wrong-week/future dates into maximally
    recent form.  Drop them instead so malformed store rows cannot leak into
    streak, activity, or form features.
    """
    if df.empty or "round" not in df.columns or "date" not in df.columns:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    def _infer_date(row):
        level = str(row.get("level", "")) if pd.notna(row.get("level")) else ""
        event = str(row.get("event", "")) if pd.notna(row.get("event")) else ""
        draw = infer_draw_size(event, level)
        round_code = str(row.get("round", ""))
        offset = get_round_day_offset(level, draw, round_code, tourney_date=row["date"])
        return row["date"] + pd.Timedelta(days=offset) if pd.notna(row["date"]) else pd.NaT

    out["date"] = out.apply(_infer_date, axis=1)
    quarantined = out["date"].isna()
    if ref is not None:
        quarantined = quarantined | (out["date"] >= pd.Timestamp(ref))
    count = int(quarantined.sum())
    if count:
        out = out.loc[~quarantined].copy()
    out.attrs["future_history_rows_quarantined"] = count
    return out


class TAFeatureCalculator:
    """
    Calculate the exact 143 features from Tennis Abstract data.
    Mirrors LiveFeatureEngine but uses TA as primary data source.
    """

    def __init__(
        self,
        ta_scraper: Optional[TennisAbstractScraper] = None,
        *,
        feature_semantics_id: str = LIVE_SEMANTICS_ID,
    ):
        if feature_semantics_id not in {LIVE_SEMANTICS_ID, SHARED_SEMANTICS_ID}:
            raise ValueError(f"unsupported live feature semantics: {feature_semantics_id}")
        self.scraper = ta_scraper or TennisAbstractScraper()
        # The shared contract is opt-in until a separately versioned model is
        # trained and promoted.  Existing callers therefore remain on the
        # registry-declared live legacy semantics.
        self.feature_semantics_id = feature_semantics_id
        self.player_slug_map = self._load_player_mapping()
        self._atp_rankings = load_rankings()  # None if not yet scraped
        if self._atp_rankings is None:
            print("WARNING: data/atp_rankings.csv not found — Rank_Points will default to 500. "
                  "Run: python scraping/atp_rankings_scraper.py")
        # Store-backed histories/profiles (no TA at predict time -> cloud-deployable).
        # DEFAULT since 2026-07-08 (parity 138/145, residuals = training-lineage-faithful
        # rank differences). Opt out with USE_STORE_HISTORY=0. Store failures fall
        # back to TA for the run — loudly, never silently.
        self.use_store = os.environ.get("USE_STORE_HISTORY", "1") == "1"
        self._store_conn = None

    def _store(self):
        if self._store_conn is None:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            import canonical_store
            conn = canonical_store.connect()
            try:
                # This is a long-lived run-scoped connection.  Reads must not
                # leave an implicit outer transaction open; otherwise the
                # explicit write-through transaction below is only a savepoint
                # and its update can roll back when the runner exits.
                conn.autocommit = True
            except Exception:
                conn.close()
                raise
            self._store_conn = conn
        return self._store_conn

    def close_store(self) -> None:
        """Close the run-owned compatibility-store connection exactly once."""
        conn = getattr(self, "_store_conn", None)
        self._store_conn = None
        if conn is None:
            return
        try:
            conn.close()
        except Exception as exc:
            print(f"      ⚠️ canonical feature-store close failed: {exc}")

    def _persist_player_field(self, profile: dict, field: str, value) -> None:
        """Preserve the legacy write-through until explicit ops cutover.

        In required eligibility-provenance mode, accepted normalized
        observations are authoritative and feature calculation never mutates
        the compatibility projection.
        """
        try:
            from storage.eligibility import EligibilityMode, eligibility_mode
        except ImportError:  # pragma: no cover - package-style execution
            from production.storage.eligibility import (  # type: ignore
                EligibilityMode, eligibility_mode,
            )
        if eligibility_mode() is EligibilityMode.REQUIRED:
            return
        if field == 'height_cm':
            value = _validated_height_cm(value)
        pid = profile.get('player_id')
        if pid is None or value in (None, '', 'U') or not self.use_store:
            return
        try:
            conn = self._store()
            # ``_store`` owns an autocommit connection, so this is always a
            # root transaction whose successful exit durably commits.
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE players SET {field} = %s, updated_at = now() "
                        f"WHERE player_id = %s AND ({field} IS NULL OR {field} = 'U')"
                        if field == 'hand' else
                        f"UPDATE players SET {field} = %s, updated_at = now() "
                        f"WHERE player_id = %s AND {field} IS NULL",
                        (value, pid),
                    )
        except Exception as exc:
            print(f"      ⚠️ player {field} write-through failed (non-fatal): {exc}")

    def _resolve_height_hand(
        self,
        p1_display: str,
        p2_display: str,
        profile1: dict,
        profile2: dict,
        *,
        session_cache: Optional[Dict] = None,
    ) -> Tuple[object, str, object, str]:
        """Resolve height/hand under the active provenance authority.

        Required mode is an authority switch, not a fallback preference. Both
        players are therefore looked up in the sealed accepted bundle even if
        the compatibility store already has populated values. Missing accepted
        fields are cleared, and a bundle/store identity disagreement blocks
        inference instead of silently mixing two players' evidence.
        """
        try:
            from storage.eligibility import EligibilityMode, eligibility_mode
        except ImportError:  # pragma: no cover - package-style execution
            from production.storage.eligibility import (  # type: ignore
                EligibilityMode, eligibility_mode,
            )

        # The compatibility table predates normalized field validation and has
        # contained values such as 3, 71, 132, and 145.  A non-null value is not
        # evidence unless it satisfies the same domain as the normalized path.
        h1 = _validated_height_cm(profile1.get('height_cm'))
        h2 = _validated_height_cm(profile2.get('height_cm'))
        hand1 = profile1.get('hand') or 'U'
        hand2 = profile2.get('hand') or 'U'
        if eligibility_mode() is EligibilityMode.REQUIRED:
            names = [
                name for name in (p1_display, p2_display)
                if isinstance(name, str) and name.strip()
            ]
            accepted_profiles = batch_get_profiles(names, verbose=False)

            def _accepted(display: str, legacy_profile: dict) -> Tuple[object, str]:
                accepted = accepted_profiles.get(display) or {}
                accepted_id = accepted.get('canonical_player_id')
                if accepted_id is None:
                    return None, 'U'
                legacy_id = legacy_profile.get('player_id')
                if legacy_id is not None:
                    try:
                        identity_matches = int(legacy_id) == int(accepted_id)
                    except (TypeError, ValueError):
                        identity_matches = False
                    if not identity_matches:
                        raise UnsafeToInferError(
                            "required eligibility identity mismatch for "
                            f"{display}: store={legacy_id!r}, bundle={accepted_id!r}"
                        )
                return (
                    _validated_height_cm(accepted.get('height_cm')),
                    accepted.get('hand') or 'U',
                )

            h1, hand1 = _accepted(p1_display, profile1)
            h2, hand2 = _accepted(p2_display, profile2)
            return h1, hand1, h2, hand2

        # The ITF order-of-play carries a stable numeric player ID and exact
        # official profile URL. Pre-hydration validates that profile before
        # installing this canonical-ID-keyed run cache. It is a handedness
        # source only: ITF profiles do not publish height.
        if session_cache is not None:
            itf_hands = session_cache.get("itf_hands_by_player_id") or {}
            if hand1 == "U":
                hand1 = str(itf_hands.get(profile1.get("player_id")) or "U")
            if hand2 == "U":
                hand2 = str(itf_hands.get(profile2.get("player_id")) or "U")

        if h1 is None or h2 is None or hand1 == 'U' or hand2 == 'U':
            missing = []
            if (h1 is None or hand1 == 'U') and p1_display:
                missing.append(p1_display)
            if (h2 is None or hand2 == 'U') and p2_display:
                missing.append(p2_display)
            missing = [m for m in missing if isinstance(m, str) and m.strip()]
            refresh_state = (
                session_cache.setdefault("atp_profile_refresh", {})
                if session_cache is not None
                else None
            )
            if not missing:
                atp_profiles = {}
            elif refresh_state is None:
                atp_profiles = batch_get_profiles(missing, verbose=False)
            else:
                atp_profiles = batch_get_profiles(
                    missing,
                    verbose=False,
                    refresh_state=refresh_state,
                )

            def _fill(display: str, height, hand: str, profile: dict):
                fallback = atp_profiles.get(display) or {}
                fallback_height = _validated_height_cm(fallback.get('height_cm'))
                if height is None and fallback_height is not None:
                    height = fallback_height
                    print(f"  ATP height fallback: {display} → {height}cm")
                    self._persist_player_field(profile, 'height_cm', height)
                if hand == 'U' and fallback.get('hand'):
                    hand = fallback['hand']
                    print(f"  ATP hand fallback: {display} → {hand}")
                    self._persist_player_field(profile, 'hand', hand)
                return height, hand

            h1, hand1 = _fill(p1_display, h1, hand1, profile1)
            h2, hand2 = _fill(p2_display, h2, hand2, profile2)
        return h1, hand1, h2, hand2

    def _prehydrate_itf_hands(
        self, profiles_by_id: dict[int, dict], cache: dict,
    ) -> dict:
        """Hydrate unknown hands from exact-ID-bound official ITF profiles."""
        summary = {
            "status": "not_started",
            "candidate_players": 0,
            "official_page_attempts": 0,
            "resolved_hands": 0,
            "failed_profiles": 0,
        }
        unknown_names = sorted({
            str(profile.get("name") or "").strip()
            for profile in profiles_by_id.values()
            if str(profile.get("name") or "").strip()
            and str(profile.get("hand") or "U").strip().upper() not in {"R", "L"}
        })
        if not unknown_names:
            summary["status"] = "no_hand_candidates"
            cache["itf_profile_hydration"] = summary
            return summary

        event_frames = cache.get("itf_event_matches") or {}
        try:
            from itf_results_scraper import (
                ItfClient, get_player_profiles, profile_refs_for_names,
            )
        except ImportError:  # pragma: no cover - package-style execution
            from scraping.itf_results_scraper import (  # type: ignore
                ItfClient, get_player_profiles, profile_refs_for_names,
            )
        refs = profile_refs_for_names(event_frames, unknown_names)
        limit = _run_itf_profile_limit()
        refs = dict(sorted(refs.items(), key=lambda item: item[0].casefold())[:limit])
        summary["candidate_players"] = len(refs)
        if not refs:
            summary["status"] = "no_exact_itf_profile_refs"
            cache["itf_profile_hydration"] = summary
            return summary

        run_profiles = cache.setdefault("itf_player_profiles", {})
        cached_by_name: dict[str, dict] = {}
        pending: dict[str, dict] = {}
        for name, ref in refs.items():
            player_id = str(ref.get("itf_player_id") or "")
            cached = run_profiles.get(player_id)
            if isinstance(cached, dict):
                cached_by_name[name] = cached
            else:
                pending[name] = ref

        fetched: dict[str, dict] = {}
        if pending:
            client = ItfClient()
            try:
                fetched = get_player_profiles(client, pending)
            finally:
                client.close()
            summary["official_page_attempts"] = len(pending)
            for name, result in fetched.items():
                player_id = str((pending.get(name) or {}).get("itf_player_id") or "")
                if player_id:
                    run_profiles[player_id] = result

        results = {**cached_by_name, **fetched}
        hands_by_player_id = cache.setdefault("itf_hands_by_player_id", {})
        profiles_by_name = {
            str(profile.get("name") or "").strip(): profile
            for profile in profiles_by_id.values()
        }
        for name, result in results.items():
            profile = profiles_by_name.get(name)
            hand = str(result.get("hand") or "").strip().upper()
            if profile is None or result.get("status") != "resolved" or hand not in {"R", "L"}:
                summary["failed_profiles"] += 1
                continue
            player_id = int(profile["player_id"])
            hands_by_player_id[player_id] = hand
            profile["hand"] = hand
            self._persist_player_field(profile, "hand", hand)
            summary["resolved_hands"] += 1
        summary["status"] = "complete"
        cache["itf_profile_hydration"] = summary
        return summary

    def prehydrate_slate_profiles(
        self,
        odds_df: pd.DataFrame,
        *,
        session_cache: Optional[Dict] = None,
    ) -> dict:
        """Hydrate current-slate missing heights once, before feature builds.

        Identity resolution stays in the canonical store.  The method dedupes
        by that player ID, orders the bounded work by decision impact, then
        delegates all fetching and source evidence recording to the existing
        strict ATP batch path.  Required provenance mode remains read-only and
        therefore intentionally bypasses this legacy compatibility hydrator.
        """
        summary = {
            "status": "not_started",
            "candidate_players": 0,
            "planned_players": 0,
            "browser_attempts": 0,
            "resolved_heights": 0,
            "remaining_budget": 0,
            "evidence_states": {},
        }
        if odds_df is None or odds_df.empty:
            summary["status"] = "empty_slate"
            return summary

        try:
            from storage.eligibility import EligibilityMode, eligibility_mode
        except ImportError:  # pragma: no cover - package-style execution
            from production.storage.eligibility import (  # type: ignore
                EligibilityMode, eligibility_mode,
            )
        if eligibility_mode() is EligibilityMode.REQUIRED:
            summary["status"] = "required_mode_read_only"
            return summary
        if not self.use_store:
            summary["status"] = "canonical_store_disabled"
            return summary

        import store_history

        cache = session_cache if session_cache is not None else {}
        conn = self._store()
        profiles_by_input: dict[str, Optional[dict]] = {}
        profiles_by_id: dict[int, dict] = {}
        candidate_rows: list[tuple[dict, Optional[dict], str]] = []

        def _profile(display_name: str) -> Optional[dict]:
            key = str(display_name or "").strip().casefold()
            if not key:
                return None
            if key not in profiles_by_input:
                profiles_by_input[key] = store_history.get_profile(conn, display_name)
            profile = profiles_by_input[key]
            if profile is not None and profile.get("player_id") is not None:
                profiles_by_id[int(profile["player_id"])] = profile
            return profile

        for _, row in odds_df.iterrows():
            names = (
                row.get("player1_normalized") or row.get("player1_raw", ""),
                row.get("player2_normalized") or row.get("player2_raw", ""),
            )
            profiles = (_profile(names[0]), _profile(names[1]))
            event = str(row.get("event", "") or "")
            for index, profile in enumerate(profiles):
                if profile is None or profile.get("player_id") is None:
                    continue
                if _validated_height_cm(profile.get("height_cm")) is not None:
                    continue
                opponent = profiles[1 - index]
                candidate_rows.append((profile, opponent, event))

        # Every slate name is bound to one canonical store ID for the whole
        # run, including players whose height is already valid but whose hand
        # is unknown. A display-key collision is ambiguous and fails closed
        # before any name-keyed cache evidence can influence completeness.
        slate_ids_by_key: dict[str, int] = {}
        for player_id, profile in profiles_by_id.items():
            key = str(profile.get("name") or "").strip().lower()
            if not key:
                continue
            existing_id = slate_ids_by_key.get(key)
            if existing_id is not None and existing_id != player_id:
                raise UnsafeToInferError(
                    "slate profile cache key maps to multiple canonical players: "
                    f"{profile.get('name')}"
                )
            slate_ids_by_key[key] = player_id

        # Install the canonical allowlist and strict-positive policy even when
        # there are no missing heights. Per-match hand fallback shares this
        # state; without it a hand-only row could still consume a legacy
        # name-keyed cache value across two canonical IDs.
        refresh_state = cache.setdefault("atp_profile_refresh", {})
        existing_player_ids = refresh_state.get("canonical_player_ids") or {}
        if not isinstance(existing_player_ids, dict):
            existing_player_ids = {}
        for key, player_id in slate_ids_by_key.items():
            existing_id = existing_player_ids.get(key)
            if existing_id is not None and int(existing_id) != player_id:
                raise UnsafeToInferError(
                    "height hydration cache key changed canonical player within run: "
                    f"{key}"
                )
        refresh_state["require_evidenced_positives"] = True
        if "remaining" not in refresh_state:
            refresh_state["remaining"] = _run_height_hydration_limit()
        allowed_keys = refresh_state.setdefault("allowed_keys", set())
        if not isinstance(allowed_keys, set):
            allowed_keys = set(allowed_keys or ())
            refresh_state["allowed_keys"] = allowed_keys
        allowed_keys.update(slate_ids_by_key)
        canonical_player_ids = refresh_state.setdefault("canonical_player_ids", {})
        if not isinstance(canonical_player_ids, dict):
            canonical_player_ids = {}
            refresh_state["canonical_player_ids"] = canonical_player_ids
        canonical_player_ids.update(slate_ids_by_key)

        # Prefetch has already warmed the current ITF order-of-play frames.
        # Use their stable player IDs to fill handedness before match feature
        # builds; this lane is independent of the ATP-only height planner.
        try:
            self._prehydrate_itf_hands(profiles_by_id, cache)
        except Exception as itf_profile_exc:
            cache["itf_profile_hydration"] = {
                "status": "error",
                "candidate_players": 0,
                "official_page_attempts": 0,
                "resolved_hands": 0,
                "failed_profiles": 0,
                "error": str(itf_profile_exc),
            }
            print(f"   ⚠️ ITF profile hydration skipped (non-fatal): {itf_profile_exc}")

        if not candidate_rows:
            summary["status"] = "no_height_candidates"
            summary["remaining_budget"] = max(
                0, int(refresh_state.get("remaining", 0) or 0)
            )
            cache["height_hydration"] = summary
            return summary

        canonical_names = sorted({
            str(profile.get("name") or "").strip()
            for profile, _opponent, _event in candidate_rows
            if str(profile.get("name") or "").strip()
        })
        evidence = profile_lookup_evidence_states(
            canonical_names,
            require_evidenced_positives=True,
            canonical_player_ids=slate_ids_by_key,
        )
        evidenced_candidates = []
        for profile, opponent, event in candidate_rows:
            player_name = str(profile.get("name") or "").strip()
            opponent_name = (
                str(opponent.get("name") or "").strip()
                if opponent is not None else ""
            )
            opponent_has_height = (
                opponent is not None
                and (
                    _validated_height_cm(opponent.get("height_cm")) is not None
                    or str(
                        (evidence.get(opponent_name) or {}).get("state", "")
                    ) == "resolved"
                )
            )
            evidenced_candidates.append(HeightHydrationCandidate(
                canonical_player_id=int(profile["player_id"]),
                player_name=player_name,
                event=event,
                opponent_has_height=opponent_has_height,
                evidence_state=str(
                    (evidence.get(player_name) or {}).get(
                        "state", "unobserved"
                    )
                ),
            ))
        plan = plan_height_hydration(evidenced_candidates)
        planned_names = [candidate.player_name for candidate in plan]
        summary["candidate_players"] = len({
            int(profile["player_id"])
            for profile, _opponent, _event in candidate_rows
        })
        summary["planned_players"] = len(plan)
        for candidate in plan:
            states = summary["evidence_states"]
            states[candidate.evidence_state] = states.get(candidate.evidence_state, 0) + 1

        attempted_before = set(refresh_state.get("attempted_keys") or ())

        resolved = batch_get_profiles(
            planned_names,
            verbose=False,
            refresh_state=refresh_state,
        )
        attempted_after = set(refresh_state.get("attempted_keys") or ())
        summary["browser_attempts"] = len(attempted_after - attempted_before)

        for candidate in plan:
            values = resolved.get(candidate.player_name) or {}
            height = _validated_height_cm(values.get("height_cm"))
            profile = profiles_by_id[candidate.canonical_player_id]
            if height is not None:
                summary["resolved_heights"] += 1
                self._persist_player_field(profile, "height_cm", height)
            hand = str(values.get("hand") or "").strip().upper()
            if hand in {"R", "L"}:
                self._persist_player_field(profile, "hand", hand)

        summary["remaining_budget"] = max(
            0, int(refresh_state.get("remaining", 0) or 0)
        )
        summary["status"] = "complete"
        cache["height_hydration"] = summary
        return summary

    @staticmethod
    def _slug_to_name(slug: str) -> str:
        """TA slug -> display name ('JanLennardStruff' -> 'Jan Lennard Struff')."""
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(slug)).strip()

    def _store_profile(self, slug: str) -> Optional[dict]:
        """Profile from the players table + current rank from the rankings CSV."""
        import store_history
        name = self._slug_to_name(slug)
        prof = store_history.get_profile(self._store(), name)
        if prof is None:
            return None
        prof["slug"] = slug
        prof["current_rank"] = get_player_rank(
            prof["name"],
            player_url=prof.get("atp_url", ""),
        )
        return prof

    def _store_history_frame(self, profile: dict, session_cache: Optional[Dict]) -> pd.DataFrame:
        import store_history
        cache = session_cache if session_cache is not None else {}
        frames = cache.setdefault("store_history", {})
        pid = profile["player_id"]
        if pid not in frames:
            frames[pid] = store_history.get_history_frame(self._store(), pid)
        return frames[pid].copy()

    def _load_player_mapping(self) -> Dict[str, str]:
        """Load player name -> TA slug mapping from CSV"""
        mapping_file = Path(__file__).parent.parent / "ta_player_mapping.csv"
        if not mapping_file.exists():
            return {}

        df = pd.read_csv(mapping_file)
        name_to_slug = {}

        for _, row in df.iterrows():
            slug = str(row.get('ta_slug', '')).strip()
            primary = str(row.get('primary_name', '')).strip()
            bovada = str(row.get('bovada_name', '')).strip()
            variants = str(row.get('name_variants', ''))

            if slug:
                if primary:
                    name_to_slug[self._norm(primary)] = slug
                if bovada:
                    name_to_slug[self._norm(bovada)] = slug
                if variants:
                    for v in variants.split('|'):
                        v = v.strip()
                        if v:
                            name_to_slug[self._norm(v)] = slug

        return name_to_slug

    @staticmethod
    def _norm(s: str) -> str:
        """Normalize name for fuzzy matching"""
        s = (s or "").strip().lower()
        try:
            import unicodedata
            s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
        except Exception:
            pass
        s = re.sub(r'[^a-z\s\-]', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    # Suffixes TA omits from player names
    _NAME_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv'}

    @staticmethod
    def _strip_name_suffixes(name: str) -> str:
        """Remove trailing generational suffixes TA doesn't include in slugs."""
        parts = name.strip().split()
        clean = [p for p in parts if p.lower().rstrip('.') not in TAFeatureCalculator._NAME_SUFFIXES]
        return ' '.join(clean) if len(clean) >= 2 else name

    def find_slug(self, player_name: str) -> Optional[str]:
        """
        Find TA slug for player name.
        Priority: 1) explicit mapping CSV  2) CamelCase derivation (suffix-stripped)
                  3) TA HTTP search
        """
        normalized = self._norm(player_name)
        if normalized in self.player_slug_map:
            return self.player_slug_map[normalized]

        # Strip Jr/Sr/II/III before deriving slug — TA omits these
        clean_name = self._strip_name_suffixes(player_name)
        derived = self.scraper.name_to_slug(clean_name)
        if derived:
            self.player_slug_map[normalized] = derived
            return derived

        # Last resort: HTTP search
        slug = self.scraper.search_player(player_name)
        if slug:
            self.player_slug_map[normalized] = slug
        return slug

    @staticmethod
    def _name_tokens(text: str) -> List[str]:
        norm = TAFeatureCalculator._norm(text)
        return [tok for tok in norm.split() if tok]

    @staticmethod
    def _event_tokens(text: str) -> set[str]:
        norm = TAFeatureCalculator._norm(text)
        tokens = re.findall(r"[a-z0-9]+", norm)
        stop = {
            'atp', 'itf', 'men', 'mens', 'challenger', 'masters', 'master',
            'grand', 'slam', 'round', 'qualifying', 'qualifier', 'final',
            'semifinal', 'semifinals', 'quarterfinal', 'quarterfinals',
            'open', 'cup', 'tour', 'event', 'singles', 'of', 'the', 'and',
            'q1', 'q2', 'q3', 'q4', 'r16', 'r32', 'r64', 'r128', 'sf', 'qf',
            'm15', 'm25'
        }
        return {tok for tok in tokens if len(tok) >= 3 and tok not in stop}

    def _opponent_name_matches(self, candidate_name: str, target_name: str) -> bool:
        cand_norm = self._norm(candidate_name)
        target_norm = self._norm(target_name)
        if not cand_norm or not target_norm:
            return False
        if cand_norm == target_norm:
            return True
        cand_tokens = self._name_tokens(candidate_name)
        target_tokens = self._name_tokens(target_name)
        if cand_tokens and target_tokens and cand_tokens[-1] == target_tokens[-1]:
            return True
        return False

    def _find_completed_match_candidate(
        self,
        df: pd.DataFrame,
        opponent_name: str,
        ref_date: datetime,
        round_code: Optional[str] = None,
        expected_event_title: str = "",
    ) -> Optional[Dict[str, object]]:
        """
        Look for a likely "this exact match already finished" row in TA history.

        TA stores tournament start dates, so the search window must be wide enough to
        capture same-event rows even when `ref_date` is an inferred round date.
        """
        if df.empty or 'opp_name' not in df.columns or 'date' not in df.columns:
            return None

        work = df.copy()
        work['date'] = pd.to_datetime(work['date'], errors='coerce')
        window_start = pd.Timestamp(ref_date) - timedelta(days=21)
        window_end = pd.Timestamp(ref_date) + timedelta(days=14)
        work = work[(work['date'] >= window_start) & (work['date'] <= window_end)].copy()
        if work.empty:
            return None

        work = work[
            work['opp_name'].astype(str).apply(lambda name: self._opponent_name_matches(name, opponent_name))
        ].copy()
        if work.empty:
            return None

        expected_tokens = self._event_tokens(expected_event_title)
        best = None

        for _, row in work.iterrows():
            candidate_round = str(row.get('round', '') or '').upper()
            round_match = bool(round_code) and candidate_round == str(round_code).upper()
            row_tokens = self._event_tokens(str(row.get('event', '') or ''))
            shared_tokens = expected_tokens & row_tokens
            event_match = bool(shared_tokens)
            date_diff = abs((pd.Timestamp(ref_date) - row['date']).days) if pd.notna(row['date']) else 999
            close_match = date_diff <= 2

            # Require at least two strong signals before treating the row as the
            # current matchup; this avoids excluding prior H2Hs too aggressively.
            signal_count = int(round_match) + int(event_match) + int(close_match)
            if signal_count < 2:
                continue

            score = (3 if round_match else 0) + (min(3, len(shared_tokens)) if event_match else 0)
            if close_match:
                score += 2
            elif date_diff <= 7:
                score += 1

            candidate = {
                'date': row['date'],
                'event': str(row.get('event', '') or ''),
                'round': candidate_round,
                'score': int(score),
                'date_diff_days': int(date_diff),
                'shared_event_tokens': sorted(shared_tokens),
            }
            if best is None or candidate['score'] > best['score']:
                best = candidate

        return best

    # ========== Laplace-smoothed win rate helper ==========

    @staticmethod
    def _laplace(wins: int, total: int, alpha: float = 3.0) -> float:
        """
        Bayesian (Laplace) smoothed win rate.
        Formula: (wins + alpha/2) / (total + alpha)
        Prior is 0.5 (neutral); alpha controls regularization strength.
        alpha=3 → need ~3 observations to trust data 50% over prior.

        Replaces both min_n threshold AND hard 0.5 default with a
        continuous, well-behaved estimate. Requires retraining to take
        full effect in the model — applied consistently with preprocess.py.

        Examples (alpha=3):
          0/0  → 0.500   1/1  → 0.625   2/2  → 0.700
          3/3  → 0.750   0/3  → 0.250  10/15 → 0.639
        """
        return (wins + alpha / 2.0) / (total + alpha)

    # ========== Temporal Features from Match History ==========

    @staticmethod
    def _count_period(df: pd.DataFrame, ref: datetime, days: int) -> int:
        """Count matches in time window [ref-days, ref). Matches training logic."""
        if df.empty or 'date' not in df.columns:
            return 0
        cut = pd.Timestamp(ref) - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return int(((df['date'] >= cut) & (df['date'] < pd.Timestamp(ref))).sum())

    @staticmethod
    def _surface_mask(df: pd.DataFrame, surface: str) -> pd.Series:
        """Boolean mask for surface matches"""
        if 'surface' not in df.columns:
            return pd.Series(False, index=df.index)
        return df['surface'].astype(str).str.lower() == (surface or '').lower()

    def _count_surface(self, df: pd.DataFrame, ref: datetime, surface: str, days: int) -> int:
        """Count surface-specific matches in time window [ref-days, ref)."""
        if df.empty:
            return 0
        cut = pd.Timestamp(ref) - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        m = (df['date'] >= cut) & (df['date'] < pd.Timestamp(ref)) & self._surface_mask(df, surface)
        return int(m.sum())

    def _surface_winrate(self, df: pd.DataFrame, ref: datetime, surface: str, days: int) -> float:
        """Surface-specific win rate — Laplace smoothed (alpha=3)."""
        if df.empty:
            return 0.5
        cut = pd.Timestamp(ref) - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        m = df[(df['date'] >= cut) & (df['date'] < pd.Timestamp(ref)) & self._surface_mask(df, surface)]
        wins = (m['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(m))

    def _winrate_lastN_within(self, df: pd.DataFrame, ref: datetime, N: int, days: int) -> float:
        """Last N matches within window win rate — Laplace smoothed (alpha=3).
        Matches training: cutoff <= date < ref, sorted newest-first, capped at N."""
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        dates = pd.to_datetime(df['date'], errors='coerce')
        mask = (dates >= cut) & (dates < ref)
        m = df[mask].copy()
        m['_date'] = dates[mask]
        m = m.sort_values('_date', ascending=False).head(N)
        wins = (m['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(m))

    def _form_trend_ewm(self, df: pd.DataFrame, ref: datetime, days: int = 30) -> float:
        """
        Exponentially weighted moving average form trend — mirrors training.
        Half-life 15 days. Requires >=3 matches, else returns 0.5.
        """
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        m = df[(df['date'] >= cut) & (df['date'] < ref)]
        if len(m) < 3:
            return 0.5
        wins_w, total_w = 0.0, 0.0
        for _, r in m.iterrows():
            days_ago = (ref - r['date']).days
            w = float(np.exp(-days_ago / 15.0))
            total_w += w
            if str(r.get('result', '')).upper() == 'W':
                wins_w += w
        return wins_w / total_w if total_w > 0 else 0.5

    @staticmethod
    def _streak(df: pd.DataFrame) -> int:
        """
        Current win/loss streak — mirrors training preprocess.py.
        Positive = win streak (+N), negative = loss streak (-N), 0 = no history.
        df must be sorted most-recent first.
        """
        if df.empty:
            return 0
        results = df['result'].astype(str).str.upper().tolist()
        first = results[0]
        if first not in ('W', 'L'):
            return 0
        s = 0
        for r in results:
            if r == first:
                s += 1
            else:
                break
        return s if first == 'W' else -s

    @staticmethod
    def _monday_of(dt: pd.Timestamp) -> pd.Timestamp:
        """Monday of dt's week, at midnight. Without normalize(), a ref carrying
        a clock time (Bovada datetimes) makes current_week 'Mon 12:20' while row
        Mondays are 'Mon 00:00' — strictly less, so the CURRENT week's rows count
        as a previous tournament and days-since collapses to ref-minus-own-Monday
        (the impossible '5 days' class). Training refs were midnight-based."""
        return (dt - pd.Timedelta(days=int(dt.weekday()))).normalize()

    def _days_since_last_tournament(self, df: pd.DataFrame, ref: datetime) -> Optional[int]:
        """Days since last tournament (week-based).

        Matches training: training used inferred_match_dt (tourney_start + round_offset)
        as the reference and computed days_since = ref - last_tournament_monday.
        We do the same here so the value is consistent with the training distribution.
        """
        if df.empty or 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        d = df[df['date'] < ref].copy()
        if d.empty:
            return None
        d['week_monday'] = d['date'].apply(lambda x: self._monday_of(pd.Timestamp(x)))
        current_week = self._monday_of(pd.Timestamp(ref))
        prev_weeks = d[d['week_monday'] < current_week]['week_monday']
        if prev_weeks.empty:
            return None
        last_week = prev_weeks.max()
        # Use (ref - last_week) not (current_week - last_week) to match training formula:
        # training computed days_since = inferred_match_dt - last_tournament_monday
        return int((pd.Timestamp(ref) - last_week).days)

    @staticmethod
    def _count_sets_from_score(score: str) -> int:
        """Count number of sets from score string (e.g., '6-3 6-4' → 2 sets)"""
        if not score or pd.isna(score):
            return 0
        # Score format: "6-3 6-4" or "6-4 6-7(3) 6-1"
        # Count space-separated set scores
        sets = [s.strip() for s in str(score).split() if s.strip()]
        return len(sets)

    def _sets_14d(self, df: pd.DataFrame, ref: datetime) -> int:
        """Count actual sets played in last 14 days from scores"""
        if df.empty or 'date' not in df.columns:
            return 0
        cut = ref - timedelta(days=14)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        recent = df[df['date'] >= cut]

        if recent.empty:
            return 0

        # Count actual sets from scores if available
        if 'score' in recent.columns:
            total_sets = recent['score'].apply(self._count_sets_from_score).sum()
            return int(total_sets)

        # Fallback to estimate if no scores
        return int(len(recent) * 2.5)

    # ========== Level & Round Stats from TA Match History ==========

    @staticmethod
    def _level_code(level: str) -> Optional[str]:
        """Normalize level to Sackmann code"""
        s = (level or "").strip().upper()
        if s in {"A", "ATP", "ATP 250", "ATP 500"}: return "A"
        if s in {"M", "MASTERS", "MASTERS 1000"}: return "M"
        if s in {"G", "GRAND SLAM", "SLAM"}: return "G"
        if s in {"C", "CHALLENGER"}: return "C"
        if s in {"25", "ITF M25", "M25", "25K"}: return "25"
        if s in {"15", "ITF M15", "M15", "15K"}: return "15"
        if s in {"F", "ATP FINALS"}: return "F"
        if s in {"S", "ITF", "FUTURES"}: return "S"
        return None

    def _level_stats(self, df: pd.DataFrame, level_code: Optional[str]) -> Tuple[float, int]:
        """Win rate and match count at specific level — Laplace smoothed (alpha=3)."""
        if df.empty or not level_code or 'level' not in df.columns:
            return (0.5, 0)
        sub = df[df['level'].astype(str).str.upper() == level_code]
        total = len(sub)
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return (self._laplace(int(wins), total), total)

    def _round_winrate(self, df: pd.DataFrame, round_code: Optional[str]) -> float:
        """Win rate at specific round — Laplace smoothed (alpha=3)."""
        if not round_code or df.empty or 'round' not in df.columns:
            return 0.5
        sub = df[df['round'].astype(str).str.upper() == str(round_code).upper()]
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(sub))

    def _semis_finals_winrates(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Win rates in SF and F — Laplace smoothed (alpha=3)."""
        if df.empty or 'round' not in df.columns:
            return (0.5, 0.5)

        def wr(code):
            sub = df[df['round'].astype(str).str.upper() == code]
            wins = (sub['result'].astype(str).str.upper() == 'W').sum()
            return self._laplace(int(wins), len(sub))

        return (wr('SF'), wr('F'))

    def _big_match_wr(self, df: pd.DataFrame) -> float:
        """Big match win rate (Grand Slams + Masters pooled) — Laplace smoothed (alpha=3)."""
        if df.empty or 'level' not in df.columns:
            return 0.5
        sub = df[df['level'].astype(str).str.upper().isin(['G', 'M'])]
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(sub))

    # ========== Rank Features from TA Match History ==========

    def _rank_change(self, df: pd.DataFrame, ref_date: datetime, days: int) -> float:
        """Rank change over time window (positive = improved)"""
        if df.empty or 'rank' not in df.columns or 'date' not in df.columns:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

        # rank change = between the last two KNOWN ranks; rows without a rank
        # (e.g. ATP-stitched rows, which never carry it) must not blank this out
        past = df[(df['date'] < ref_date) & df['rank'].notna()].sort_values('date')
        if past.empty:
            return 0.0

        t_now = past.iloc[-1]
        t_then_cut = ref_date - timedelta(days=days)
        past_then = past[past['date'] < t_then_cut]
        if past_then.empty:
            return 0.0
        t_then = past_then.iloc[-1]

        r_now = t_now.get('rank')
        r_then = t_then.get('rank')
        if pd.notna(r_now) and pd.notna(r_then):
            return float(r_then) - float(r_now)
        return 0.0

    def _rank_volatility(self, df: pd.DataFrame, ref_date: datetime, days: int) -> float:
        """Standard deviation of rank over time window"""
        if df.empty or 'rank' not in df.columns or 'date' not in df.columns:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

        cut = ref_date - timedelta(days=days)
        window = df[(df['date'] < ref_date) & (df['date'] >= cut)]
        if window.empty:
            return 0.0

        ranks = window['rank'].dropna()
        if ranks.empty:
            return 0.0
        return float(ranks.std())

    # ========== Opponent-Specific Features ==========

    def _winrate_vs_hand(self, df: pd.DataFrame, opp_hand: str) -> float:
        """Win rate vs specific opponent handedness — Laplace smoothed (alpha=3)."""
        if df.empty or 'opp_hand' not in df.columns:
            return 0.5
        sub = df[df['opp_hand'].astype(str).str.upper() == opp_hand.upper()]
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(sub))

    # ========== H2H from Tennis Abstract H2H Page ==========

    def _get_h2h_stats(self, slug1: str, slug2: str, session_cache: Optional[Dict] = None,
                       _frame: Optional[pd.DataFrame] = None, _name2: Optional[str] = None) -> Dict[str, float]:
        """
        Get H2H stats from Tennis Abstract H2H page.
        For now, calculate from match histories until H2H page parser is added.
        ``_frame``/``_name2`` (store mode): use a pre-fetched P1 history frame and
        P2 display name — identical filter semantics, zero TA calls.
        """
        if _frame is not None and _name2:
            h2h = pd.DataFrame()
            if not _frame.empty and 'opp_name' in _frame.columns:
                h2h = _frame[_frame['opp_name'].str.contains(re.escape(_name2), case=False, na=False)].copy()
            return self._h2h_from_matches(h2h)
        # Load both players' FULL CAREER matches for H2H (years=[] means all years)
        matches1 = self.scraper.get_player_matches(slug1, years=[], force_refresh=False, persist=False, session_cache=session_cache)
        matches2 = self.scraper.get_player_matches(slug2, years=[], force_refresh=False, persist=False, session_cache=session_cache)

        if matches1.empty and matches2.empty:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }

        # Filter for matches against each other (by opponent name matching)
        # This is approximate - ideally we'd parse the dedicated H2H page
        profile1 = self.scraper.get_player_profile(slug1, force_refresh=False, persist=False, session_cache=session_cache)
        profile2 = self.scraper.get_player_profile(slug2, force_refresh=False, persist=False, session_cache=session_cache)

        if not profile1 or not profile2:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }

        name1 = profile1.get('name', '')
        name2 = profile2.get('name', '')

        # Get H2H matches from player1's perspective only
        # (Don't concatenate both perspectives - that counts each match twice!)
        if not matches1.empty and 'opp_name' in matches1.columns:
            h2h = matches1[matches1['opp_name'].str.contains(name2, case=False, na=False)].copy()
        else:
            h2h = pd.DataFrame()

        return self._h2h_from_matches(h2h)

    def _h2h_from_matches(self, h2h: pd.DataFrame) -> Dict[str, float]:
        """H2H stat dict from P1-perspective meeting rows (source-agnostic)."""
        if h2h is None or h2h.empty:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }
        if 'date' in h2h.columns:
            h2h['date'] = pd.to_datetime(h2h['date'], errors='coerce')
            h2h = h2h.sort_values('date', ascending=False)

        total = len(h2h)
        p1_wins = (h2h['result'].astype(str).str.upper() == 'W').sum()
        p2_wins = total - p1_wins

        # Recent advantage (last 3 meetings) — Laplace smoothed, centered at 0
        recent = h2h.head(3)
        recent_wins = (recent['result'].astype(str).str.upper() == 'W').sum()
        adv = float(self._laplace(int(recent_wins), len(recent)) - 0.5) if len(recent) >= 2 else 0.0

        # Career H2H win rate — Laplace smoothed
        wr = self._laplace(int(p1_wins), total)

        return {
            'H2H_Total_Matches': int(total),
            'H2H_P1_Wins': int(p1_wins),
            'H2H_P2_Wins': int(p2_wins),
            'H2H_P1_WinRate': float(wr),
            'H2H_Recent_P1_Advantage': float(adv)
        }

    def _shared_h2h_stats(
        self,
        p1_history: pd.DataFrame,
        p2_display_name: str,
        as_of: datetime,
        p2_player_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """Candidate P1-oriented H2H adapter over the already as-of history.

        The formula is shared and source-neutral.  Store history selects by
        stable opponent ID.  TA fallback accepts only an exact normalized full
        name; surname-only matching is deliberately forbidden.
        """

        def _identity_key(value) -> Optional[str]:
            if value is None or pd.isna(value):
                return None
            text = str(value).strip()
            if not text:
                return None
            try:
                numeric = float(text)
            except (TypeError, ValueError, OverflowError):
                return text
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
            return text

        if p1_history is None or p1_history.empty:
            meetings = pd.DataFrame()
        elif (
            p2_player_id is not None
            and 'opp_id' in p1_history.columns
            and p1_history['opp_id'].notna().any()
        ):
            target_id = _identity_key(p2_player_id)
            meetings = p1_history.loc[
                p1_history['opp_id'].map(_identity_key).eq(target_id)
            ].copy()
        elif 'opp_name' not in p1_history.columns:
            meetings = pd.DataFrame()
        else:
            target_name = self._norm(p2_display_name)
            mask = p1_history['opp_name'].astype(str).map(self._norm).eq(target_name)
            meetings = p1_history.loc[mask].copy()
            if 'opp_id' in meetings.columns:
                opponent_ids = {
                    identity
                    for value in meetings['opp_id'].dropna().tolist()
                    if (identity := _identity_key(value)) is not None
                }
                if len(opponent_ids) > 1:
                    raise UnsafeToInferError(
                        "shared_h2h_ambiguous_opponent_identity:"
                        f"{p2_display_name} maps to {sorted(opponent_ids)}"
                    )
        return dict(shared_h2h_features_from_frame(meetings, as_of))

    # ========== One-Hot Encodings ==========

    @staticmethod
    def _season_flags(surface: str, when: datetime) -> Dict[str, int]:
        """Seasonal flags based on month"""
        m = when.month
        return {
            'Clay_Season': 1 if 4 <= m <= 6 else 0,
            'Grass_Season': 1 if m in (6, 7) else 0,
            'Indoor_Season': 1 if (m >= 10 or m <= 2) else 0
        }

    @staticmethod
    def _levels_onehot(level: str) -> Dict[str, int]:
        """One-hot encoding for tournament level"""
        s = (level or "").strip().upper()

        # Normalize text variants to codes
        if s in {"GRAND SLAM", "SLAM"}: s = "G"
        elif s in {"MASTERS 1000", "MASTERS"}: s = "M"
        elif s in {"ATP", "ATP 250", "ATP 500"}: s = "A"
        elif s in {"CHALLENGER"}: s = "C"
        elif s in {"ITF", "FUTURES"}: s = "S"
        elif s in {"ITF M25", "M25", "25K", "25"}: s = "25"
        elif s in {"ITF M15", "M15", "15K", "15"}: s = "15"
        elif s in {"ATP FINALS", "NITTO ATP FINALS"}: s = "F"

        return {
            "Level_G": 1 if s == "G" else 0,
            "Level_M": 1 if s == "M" else 0,
            "Level_A": 1 if s == "A" else 0,
            "Level_C": 1 if s == "C" else 0,
            "Level_S": 1 if s == "S" else 0,
            "Level_F": 1 if s == "F" else 0,
            "Level_25": 1 if s == "25" else 0,
            "Level_15": 1 if s == "15" else 0,
            "Level_O": 1 if s == "O" else 0,
            "Level_D": 1 if s == "D" else 0,
        }

    @staticmethod
    def _rounds_onehot(round_code: Optional[str]) -> Dict[str, int]:
        """One-hot encoding for round"""
        rc = (round_code or '').upper()
        keys = ['R128', 'R64', 'R32', 'R16', 'Q1', 'Q2', 'Q3', 'Q4', 'QF', 'SF', 'F', 'RR', 'ER', 'BR']
        return {f'Round_{k}': (1 if rc == k else 0) for k in keys}

    @staticmethod
    def _hand_onehot(hand: str, prefix: str) -> Dict[str, int]:
        """One-hot encoding for handedness"""
        h = (hand or 'R').upper()
        return {
            f'{prefix}_Hand_R': 1 if h == 'R' else 0,
            f'{prefix}_Hand_L': 1 if h == 'L' else 0,
            f'{prefix}_Hand_U': 1 if h == 'U' else 0,
            f'{prefix}_Hand_A': 1 if h == 'A' else 0,
        }

    @staticmethod
    def _country_onehot(country: str, prefix: str) -> Dict[str, int]:
        """One-hot encoding for country"""
        c = (country or 'Other').upper()
        keys = ['USA', 'GBR', 'FRA', 'ITA', 'AUS', 'SRB', 'CZE', 'ESP', 'SUI', 'GER', 'ARG', 'RUS']
        vals = {f'{prefix}_Country_{cc}': int(c == cc) for cc in keys}
        vals[f'{prefix}_Country_Other'] = 0 if any(vals.values()) else 1
        return vals

    @staticmethod
    def _handedness_matchup(p1_hand: str, p2_hand: str) -> Dict[str, int]:
        """One-hot encoding for handedness matchup"""
        a = (p1_hand or 'R').upper()
        b = (p2_hand or 'R').upper()
        combos = ['RR', 'RL', 'LR', 'LL']
        out = {f'Handedness_Matchup_{cmb}': 0 for cmb in combos}
        key = f'{a}{b}'
        if key in combos:
            out[f'Handedness_Matchup_{key}'] = 1
        return out

    # ========== Main Feature Builder ==========

    def build_141_features(
        self,
        player1_name: Optional[str] = None,
        player2_name: Optional[str] = None,
        slug1: Optional[str] = None,
        slug2: Optional[str] = None,
        match_date: Optional[datetime] = None,
        surface: str = "Hard",
        tournament_level: str = "A",
        draw_size: int = 32,
        round_code: Optional[str] = None,
        expected_event_title: str = "",
        force_refresh: bool = True,
        persist: bool = False,
        session_cache: Optional[Dict] = None,
        match_date_is_explicit: bool = False,
        metadata_source: str = "resolved",
        canonical_match_date: Optional[datetime | date | str] = None,
    ) -> Dict[str, float]:
        """
        Build exactly 143 features from Tennis Abstract data.

        Args:
            player1_name: Player 1 name (if slug1 not provided)
            player2_name: Player 2 name (if slug2 not provided)
            slug1: Player 1 TA slug (if provided, skips name resolution)
            slug2: Player 2 TA slug (if provided, skips name resolution)
            match_date: Match date (default: now)
            surface: Surface (Hard/Clay/Grass/Carpet)
            tournament_level: Level code (G/M/A/C/25/15)
            draw_size: Tournament draw size
            round_code: Round code (R32/QF/SF/F/etc)
            match_date_is_explicit: If True, match_date came from a reliable source (e.g. Bovada
                absolute date) and should not be overridden by the TA round-offset heuristic.
            metadata_source: Source label for tournament metadata. Fallback/default sources
                may be overridden by TA's explicit upcoming-match surface.
            canonical_match_date: Candidate-only timezone-naive event-local
                date. Required when ``match_date`` is timezone-aware.
            force_refresh: If True, always fetch fresh data (default: True)
            persist: If True, read/write disk cache (default: False)
            session_cache: Dict for in-run memoization (default: None)

        Returns:
            Dict with exactly 143 features in correct order
        """
        # Resolve slugs if names provided
        if not slug1:
            if not player1_name:
                raise ValueError("Must provide either player1_name or slug1")

            # Check session cache for slug resolution
            if session_cache is not None:
                slug_map = session_cache.setdefault('slug_resolutions', {})
                if player1_name in slug_map:
                    slug1 = slug_map[player1_name]
                else:
                    slug1 = self.find_slug(player1_name)
                    if slug1:
                        slug_map[player1_name] = slug1
            else:
                slug1 = self.find_slug(player1_name)

            if not slug1:
                raise RuntimeError(f"Could not resolve TA slug for: {player1_name}")

        if not slug2:
            if not player2_name:
                raise ValueError("Must provide either player2_name or slug2")

            # Check session cache for slug resolution
            if session_cache is not None:
                slug_map = session_cache.setdefault('slug_resolutions', {})
                if player2_name in slug_map:
                    slug2 = slug_map[player2_name]
                else:
                    slug2 = self.find_slug(player2_name)
                    if slug2:
                        slug_map[player2_name] = slug2
            else:
                slug2 = self.find_slug(player2_name)

            if not slug2:
                raise RuntimeError(f"Could not resolve TA slug for: {player2_name}")

        # Build features from slugs, passing cache params through
        return self.build_141_features_from_slugs(
            slug1=slug1,
            slug2=slug2,
            match_date=match_date,
            surface=surface,
            tournament_level=tournament_level,
            draw_size=draw_size,
            round_code=round_code,
            expected_event_title=expected_event_title,
            force_refresh=force_refresh,
            persist=persist,
            session_cache=session_cache,
            match_date_is_explicit=match_date_is_explicit,
            metadata_source=metadata_source,
            canonical_match_date=canonical_match_date,
        )

    def build_141_features_from_slugs(
        self,
        slug1: str,
        slug2: str,
        match_date: Optional[datetime] = None,
        surface: str = "Hard",
        tournament_level: str = "A",
        draw_size: int = 32,
        round_code: Optional[str] = None,
        expected_event_title: str = "",
        force_refresh: bool = True,
        persist: bool = False,
        session_cache: Optional[Dict] = None,
        match_date_is_explicit: bool = False,
        metadata_source: str = "resolved",
        canonical_match_date: Optional[datetime | date | str] = None,
    ) -> Dict[str, float]:
        """
        Build exactly 143 features from Tennis Abstract slugs.
        Bypasses name→slug search for tests/warmers.
        Returns dict with features in exact order expected by model.

        Args:
            slug1: Player 1 TA slug
            slug2: Player 2 TA slug
            match_date: Match date (default: now)
            surface: Surface (Hard/Clay/Grass/Carpet)
            tournament_level: Level code (G/M/A/C/25/15)
            draw_size: Tournament draw size
            round_code: Round code (R32/QF/SF/F/etc)
            canonical_match_date: Candidate-only timezone-naive event-local
                date. Required when ``match_date`` is timezone-aware.
            force_refresh: If True, always fetch fresh data (default: True)
            persist: If True, read/write disk cache (default: False)
            session_cache: Dict for in-run memoization (default: None)

        Returns:
            Dict with exactly 143 features in correct order
        """
        use_shared_semantics = self.feature_semantics_id == SHARED_SEMANTICS_ID
        if use_shared_semantics:
            candidate_date = (
                canonical_match_date
                if canonical_match_date is not None
                else match_date
            )
            if candidate_date is None:
                raise ValueError(
                    "base_141_shared requires a provenance-backed, timezone-naive "
                    "canonical_match_date (or canonical timezone-naive match_date)"
                )
            # A clock-bearing aware ``match_date`` may be retained by callers as
            # match_start_at_utc lineage, but never determines the event day.
            when = as_of_day(candidate_date).to_pydatetime()
        else:
            when = match_date or datetime.utcnow()
        surface = (surface or "Hard").strip().title()

        # Get profiles: store-backed by default (no TA at predict time), else
        # Tennis Abstract. Store outage -> loud TA fallback for the whole run.
        if self.use_store:
            try:
                profile1 = self._store_profile(slug1)
                profile2 = self._store_profile(slug2)
                # per-player fallback: store couldn't resolve this name -> try TA
                # for that player only (loud; cloud's TA circuit breaker makes it
                # a clean honest skip when TA is blocked)
                for _i, (_slug, _prof) in enumerate(((slug1, profile1), (slug2, profile2))):
                    if _prof is None:
                        print(f"      ↩️  Store has no profile for {_slug} — trying Tennis Abstract")
                        _ta_prof = self.scraper.get_player_profile(
                            _slug, force_refresh=force_refresh, persist=persist,
                            session_cache=session_cache)
                        if _i == 0:
                            profile1 = _ta_prof
                        else:
                            profile2 = _ta_prof
            except Exception as _store_exc:
                print(f"      🚨 Canonical store unavailable ({_store_exc}) — falling back to "
                      f"Tennis Abstract for this run (provenance: ta)")
                self.use_store = False
        if not self.use_store:
            profile1 = self.scraper.get_player_profile(
                slug1,
                force_refresh=force_refresh,
                persist=persist,
                session_cache=session_cache
            )
            profile2 = self.scraper.get_player_profile(
                slug2,
                force_refresh=force_refresh,
                persist=persist,
                session_cache=session_cache
            )

        if not profile1 or not profile2:
            missing_slug = slug1 if not profile1 else slug2
            raise RuntimeError(f"TA profile load failed for slug: {missing_slug}")

        # Get match histories (years=[] means ALL years for career stats)
        if self.use_store:
            def _history(profile, slug):
                if profile.get("player_id") is not None:
                    return self._store_history_frame(profile, session_cache)
                # per-player TA-profile fallback -> TA history for that player too
                return self.scraper.get_player_matches(
                    slug, years=[], force_refresh=force_refresh,
                    persist=persist, session_cache=session_cache)
            matches1 = _history(profile1, slug1)
            matches2 = _history(profile2, slug2)
        else:
            matches1 = self.scraper.get_player_matches(
                slug1,
                years=[],  # All years for career stats
                force_refresh=force_refresh,
                persist=persist,
                session_cache=session_cache
            )
            matches2 = self.scraper.get_player_matches(
                slug2,
                years=[],  # All years for career stats
                force_refresh=force_refresh,
                persist=persist,
                session_cache=session_cache
            )
        # keep pre-stitch frames for H2H (same semantics both modes)
        _m1_prestitch = matches1

        # For profile-only testing, allow empty match histories
        # (temporal features will use defaults)
        if matches1.empty and matches2.empty:
            print(f"⚠️ No match history available (will use profile-only features)")
            # Don't return defaults - proceed with profile-only features

        def _fractional_age(profile: dict, ref: datetime) -> float:
            """Fractional age in years at match date — matches Sackmann's (match_date - dob) / 365.25"""
            bd = profile.get('birthdate')  # 'YYYY-MM-DD'
            if bd:
                try:
                    dob = datetime.strptime(str(bd), '%Y-%m-%d')
                    return (ref - dob).days / 365.25
                except Exception:
                    pass
            return float(profile.get('age') or 25.0)

        _semantic_defaults: list = []  # Meaningful defaults bypassing _default_for()

        # Unranked players: training's exact convention is rank=999 (preprocess.py:151)
        # with 0 rank points — 'unranked' is REAL information the models learned on
        # (futures training data is full of it), not a placeholder. A strict
        # identity rejection is different: the numeric fallback may keep the row
        # observable, but explicit markers make it ineligible for betting/GOLD.
        for label, profile in [('P1', profile1), ('P2', profile2)]:
            lookup_status = 'resolved' if profile.get('current_rank') is not None else None

            def _mark_rank_default(reason: str) -> None:
                profile['_rank_lookup_unresolved'] = reason
                markers = (
                    f'{label}_Rank=rank_lookup_unresolved({reason})',
                    f'{label}_Rank_Points=rank_lookup_unresolved({reason})',
                )
                for marker in markers:
                    if marker not in _semantic_defaults:
                        _semantic_defaults.append(marker)

            if profile.get('current_rank') is None and profile.get('name'):
                profile['current_rank'] = get_player_rank(
                    profile['name'],
                    self._atp_rankings,
                    player_url=profile.get('atp_url', ''),
                )
                if profile.get('current_rank') is None:
                    lookup_status = get_player_lookup_status(
                        profile['name'],
                        self._atp_rankings,
                        player_url=profile.get('atp_url', ''),
                    )
                    if lookup_status in {
                        'identity_unresolved', 'rank_invalid',
                        'rankings_unavailable',
                    }:
                        _mark_rank_default(lookup_status)
                else:
                    lookup_status = 'resolved'
            if profile.get('current_rank') is None and profile.get('player_id') is not None:
                # deeper than the rankings file reaches (~1500): use the latest
                # rank recorded on their matches — the exact lineage training saw
                try:
                    from canonical_store import connect as _cs_connect
                    from store_history import latest_recorded_rank
                    with _cs_connect() as _conn:
                        _lr = latest_recorded_rank(_conn, profile['player_id'])
                    if _lr:
                        profile['current_rank'] = _lr
                        # A historical observation is useful for observability,
                        # but it is not proof of current ranking identity.  Even
                        # a true rankings-file ``not_ranked`` miss must remain
                        # ineligible when this stale numeric fallback is used.
                        if lookup_status != 'resolved':
                            _mark_rank_default('store_history_fallback')
                        print(f"      {label} rank from store history: {_lr} (deeper than rankings file)")
                except Exception as _lr_exc:
                    print(f"      ⚠️ store-rank fallback failed ({_lr_exc})")
            if profile.get('current_rank') is None:
                profile['current_rank'] = 999
                profile['_unranked'] = True
                print(f"      {label} unranked -> rank 999 / 0 pts (training convention): {profile.get('name', profile.get('slug','?'))}")

        def _identity_key(value) -> Optional[str]:
            if value is None or pd.isna(value):
                return None
            text = str(value).strip()
            if not text:
                return None
            try:
                numeric = float(text)
            except (TypeError, ValueError, OverflowError):
                return text
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
            return text

        def _exact_observed_points(
            display_name: str,
            player_id=None,
        ) -> Optional[float]:
            rankings = self._atp_rankings
            if (
                rankings is None
                or rankings.empty
                or 'player_name' not in rankings.columns
                or 'points' not in rankings.columns
            ):
                return None
            stable_id = _identity_key(player_id)
            if stable_id is not None and 'player_id' in rankings.columns:
                identity_mask = rankings['player_id'].map(_identity_key).eq(stable_id)
            else:
                target = self._norm(display_name)
                identity_mask = rankings['player_name'].astype(str).map(self._norm).eq(target)
            exact = rankings.loc[identity_mask, 'points']
            # Identity uniqueness is decided before inspecting the points cell.
            # Otherwise two same-name/ID rows could be accepted merely because
            # one duplicate happened to have null or malformed points.
            if int(identity_mask.sum()) != 1:
                return None
            values = pd.to_numeric(exact, errors='coerce').dropna()
            values = values[np.isfinite(values) & (values >= 0)]
            # Duplicate full-name rows are identity-ambiguous even when the
            # point values happen to match. Stable-ID evidence follows the same
            # one-current-observation rule.
            return float(values.iloc[0]) if len(values) == 1 else None

        def _atp_points(
            display_name: str,
            ta_rank: float,
            player_label: str,
            player_id=None,
            player_url: str = '',
        ) -> float:
            """Look up ATP points from rankings cache with rank cross-validation."""
            if use_shared_semantics:
                observed = _exact_observed_points(display_name, player_id)
                if observed is not None:
                    return observed
                fallback = 0.0 if ta_rank >= 999 else 500.0
                _semantic_defaults.append(
                    f'{player_label}_Rank_Points={int(fallback)}(shared_missing_exact)'
                )
                return fallback

            # Frozen live-legacy behavior: permissive name lookup followed by
            # curve interpolation when a ranked player has no matched points.
            pts = get_player_points(
                display_name,
                self._atp_rankings,
                player_url=player_url,
            )
            if pts is not None:
                atp_rank = get_player_rank(
                    display_name,
                    self._atp_rankings,
                    player_url=player_url,
                )
                if atp_rank is not None and abs(atp_rank - ta_rank) > 20:
                    print(f"  RANK MISMATCH for {display_name}: TA={int(ta_rank)}, ATP={atp_rank} (using TA rank, ATP points={pts})")
                return float(pts)
            if ta_rank >= 999:  # unranked: 0 points is the factual value, not a default
                return 0.0
            # ranked player missing from the CSV (beyond scrape depth / name form):
            # interpolate points from the official rank->points curve — a real,
            # deterministic estimate; a flat 500 is a top-110 player's points
            if self._atp_rankings is not None and len(self._atp_rankings) > 50:
                curve = self._atp_rankings[["rank", "points"]].dropna().sort_values("rank")
                import numpy as _np
                est = float(_np.interp(float(ta_rank), curve["rank"], curve["points"]))
                print(f"  ATP points for '{display_name}' interpolated from rank curve: rank {int(ta_rank)} → {est:.0f} pts")
                return est
            print(f"  ATP points not found for '{display_name}' — defaulting to 500")
            _semantic_defaults.append(f'{player_label}_Rank_Points=500(not_found)')
            return 500.0

        # TA profiles for obscure players can carry name=None — fall back to the
        # slug-derived display form so downstream name matching keeps working
        p1_display = profile1.get('name') or self._slug_to_name(slug1)
        p2_display = profile2.get('name') or self._slug_to_name(slug2)

        # Stitch missing recent matches from ATP when TA is stale (real results
        # with source provenance — never silent defaults; see history_stitch.py).
        # Rank/hand-dependent features still compute only on TA rows (NaN-safe).
        for _label, _display, _matches in (("P1", p1_display, matches1), ("P2", p2_display, matches2)):
            if not needs_stitching(_matches, when):
                continue
            try:
                if str(tournament_level or "") in ("15", "25", "S"):
                    # ITF tier: atptour has nothing; itftennis.com order-of-play
                    # provides the current event's completed rows (score-only)
                    _atp_rows = gather_itf_rows(_display, expected_event_title, when, session_cache)
                else:
                    _atp_rows = gather_atp_rows(_display, when, self._atp_rankings, session_cache,
                                                level_hint=str(tournament_level or ""))
                if _atp_rows.empty:
                    continue
                _stitched = stitch_history(_matches, _atp_rows)
                _added = len(_stitched) - len(_matches)
                if _added > 0:
                    _ta_last = pd.to_datetime(_matches['date'], errors='coerce').max() if not _matches.empty else None
                    print(f"      🧵 {_display}: stitched {_added} ATP row(s) onto TA history (TA last: {_ta_last})")
                    if _label == "P1":
                        matches1 = _stitched
                    else:
                        matches2 = _stitched
            except Exception as _exc:
                print(f"      ⚠️ ATP history stitch failed for {_display} (non-fatal): {_exc}")

        # Legacy mode retains ATP fallback behavior. Required mode instead
        # replaces both fields from the generation/seal-pinned accepted bundle.
        h1, hand1, h2, hand2 = self._resolve_height_hand(
            p1_display, p2_display, profile1, profile2,
            session_cache=session_cache,
        )

        s1 = {
            'height': h1,
            'age': _fractional_age(profile1, when),
            'hand': hand1,
            'country': profile1.get('country'),
            'rank': float(profile1['current_rank']),
            'rank_points': _atp_points(
                p1_display,
                float(profile1['current_rank']),
                'P1',
                profile1.get('player_id'),
                profile1.get('atp_url', ''),
            ),
        }
        s2 = {
            'height': h2,
            'age': _fractional_age(profile2, when),
            'hand': hand2,
            'country': profile2.get('country'),
            'rank': float(profile2['current_rank']),
            'rank_points': _atp_points(
                p2_display,
                float(profile2['current_rank']),
                'P2',
                profile2.get('player_id'),
                profile2.get('atp_url', ''),
            ),
        }
        if use_shared_semantics:
            s1 = normalize_player_snapshot(**s1)
            s2 = normalize_player_snapshot(**s2)

        def _coerce_numeric(value, default_feature: str, note: str | None = None) -> float:
            """Coerce optional numeric stats to floats before derived arithmetic."""
            if pd.notna(value):
                return float(value)
            fallback = float(self._default_for(default_feature, p1=s1, p2=s2, surface=surface))
            _semantic_defaults.append(note or f'{default_feature}=default')
            return fallback

        # Get TA tournament start date + round BEFORE temporal features.
        # TA (like Sackmann CSV) stores tournament START DATE for all rounds.
        # Store mode: no TA upcoming lookup — rounds come from inference/draws,
        # dates from Bovada; the completed-match guard runs on the store frame.
        _upcoming = None if self.use_store else self.scraper.get_upcoming_match(
            slug1, p2_display, session_cache=session_cache)
        completed_candidate = self._find_completed_match_candidate(
            matches1,
            p2_display,
            when,
            round_code=round_code,
            expected_event_title=expected_event_title,
        )
        if completed_candidate and not _upcoming:
            event_label = completed_candidate['event'] or 'unknown event'
            round_label = completed_candidate['round'] or '?'
            date_label = completed_candidate['date'].date().isoformat() if pd.notna(completed_candidate['date']) else '?'
            raise UnsafeToInferError(
                "ta_completed_match_candidate:"
                f"{p1_display} vs {p2_display} already appears in TA history "
                f"({event_label}, {round_label}, {date_label})"
            )
        _tourney_start = None
        if _upcoming:
            if not round_code and _upcoming.get('round'):
                round_code = _upcoming['round']
                print(f"      📋 Round from TA: {round_code} (upcoming match listing)")
            _ta_date_str = str(_upcoming.get('date', ''))
            if len(_ta_date_str) == 8 and _ta_date_str.isdigit():
                try:
                    _tourney_start = datetime.strptime(_ta_date_str, '%Y%m%d')
                    _day_offset = get_round_day_offset(tournament_level, draw_size, round_code or '', tourney_date=_tourney_start)
                    _ta_inferred = _tourney_start + timedelta(days=_day_offset)
                    if match_date_is_explicit or use_shared_semantics:
                        # A shared-candidate date is already the required
                        # provenance-backed canonical event-local day.  TA's
                        # tournament-start heuristic may inform diagnostics,
                        # but it can never replace that date (including when
                        # canonical_match_date was supplied separately from a
                        # timezone-aware kickoff instant).
                        date_source = (
                            "shared canonical event date"
                            if use_shared_semantics else "explicit Bovada date"
                        )
                        print(
                            f"      📅 TA inferred {_ta_inferred.date()} but using "
                            f"{date_source} {when.date()}"
                        )
                    else:
                        when = _ta_inferred
                        print(f"      📅 inferred_match_dt: {_tourney_start.date()} + {_day_offset}d ({round_code}) = {when.date()}")
                except Exception:
                    pass
            _ta_surface = _upcoming.get('surface', '')
            if _ta_surface and _ta_surface.lower() != surface.lower():
                reconciled_surface, used_ta_surface = reconcile_upcoming_surface(
                    surface,
                    _ta_surface,
                    metadata_source,
                )
                if used_ta_surface:
                    print(
                        f"      📋 Surface from TA: {_ta_surface} "
                        f"(overriding {metadata_source}={surface})"
                    )
                    surface = reconciled_surface
                else:
                    print(f"      ⚠️  Surface mismatch: resolver={surface}, TA upcoming={_ta_surface} — using resolver")

        # If TA's upcoming listing couldn't provide the round, infer it from both
        # players' ATP-stitched completed rows (real data: both just finished
        # round R at the same event -> this match is the next round).
        if not round_code:
            _inferred_rc = infer_next_round_any(matches1, matches2, when)
            if _inferred_rc:
                round_code = _inferred_rc
                print(f"      📋 Round inferred from stitched event history: {round_code}")
        if not round_code and str(tournament_level or "") in ("15", "25", "S"):
            _itf_rc = itf_round_for(p1_display, p2_display, expected_event_title, when, session_cache)
            if _itf_rc:
                round_code = _itf_rc
                print(f"      📋 Round from ITF order of play: {round_code}")
        if not round_code:
            # first-round matches can't infer from results — read the bracket
            _draw_rc = round_from_draws(p1_display, p2_display, when, session_cache)
            if _draw_rc:
                round_code = _draw_rc
                print(f"      📋 Round from event draw page: {round_code}")

        # Apply round-day offsets to historical match data to match training methodology.
        # Training (preprocess.py) used inferred_match_dt = tourney_date + ROUND_DAY_OFFSET[round]
        # for every match before computing temporal windows. TA stores tournament START DATE for
        # all rounds (same as Sackmann), so without this step live features diverge from training.
        matches1 = apply_round_offsets_to_history(matches1, ref=when)
        matches2 = apply_round_offsets_to_history(matches2, ref=when)

        # Calculate temporal features
        if use_shared_semantics:
            t1 = player_temporal_features(
                observations_from_records(matches1),
                when,
                surface,
                rank_as_of=s1['rank'],
            )
            t2 = player_temporal_features(
                observations_from_records(matches2),
                when,
                surface,
                rank_as_of=s2['rank'],
            )
            p1_rc30 = t1['rank_change_30d']
            p1_rc90 = t1['rank_change_90d']
            p2_rc30 = t2['rank_change_30d']
            p2_rc90 = t2['rank_change_90d']
            p1_vol90 = t1['rank_volatility_90d']
            p2_vol90 = t2['rank_volatility_90d']
        else:
            def temporal(df):
                return {
                    'matches_14d': self._count_period(df, when, 14),
                    'matches_30d': self._count_period(df, when, 30),
                    'matches_90d': self._count_period(df, when, 90),
                    'surface_matches_30d': self._count_surface(df, when, surface, 30),
                    'surface_matches_90d': self._count_surface(df, when, surface, 90),
                    'surface_experience': self._count_surface(df, when, surface, 9999),
                    'surface_winrate_90d': self._surface_winrate(df, when, surface, 90),
                    'winrate_last10_120d': self._winrate_lastN_within(df, when, 10, 120),
                    'streak': self._streak(df),
                    'form_trend_30d': self._form_trend_ewm(df, when, 30),
                    'days_since_last': (self._days_since_last_tournament(df, when) or 60),
                    'sets_14d': self._sets_14d(df, when),
                    'last_surface': str(df.iloc[0]['surface']).title() if not df.empty and pd.notna(df.iloc[0].get('surface')) else None
                }

            t1 = temporal(matches1)
            t2 = temporal(matches2)

            # Rank changes and volatility
            p1_rc30 = self._rank_change(matches1, when, 30)
            p1_rc90 = self._rank_change(matches1, when, 90)
            p2_rc30 = self._rank_change(matches2, when, 30)
            p2_rc90 = self._rank_change(matches2, when, 90)
            p1_vol90 = self._rank_volatility(matches1, when, 90)
            p2_vol90 = self._rank_volatility(matches2, when, 90)
        p1_rc30 = _coerce_numeric(p1_rc30, 'P1_Rank_Change_30d')
        p1_rc90 = _coerce_numeric(p1_rc90, 'P1_Rank_Change_90d')
        p2_rc30 = _coerce_numeric(p2_rc30, 'P2_Rank_Change_30d')
        p2_rc90 = _coerce_numeric(p2_rc90, 'P2_Rank_Change_90d')
        p1_vol90 = _coerce_numeric(p1_vol90, 'P1_Rank_Volatility_90d')
        p2_vol90 = _coerce_numeric(p2_vol90, 'P2_Rank_Volatility_90d')

        # Level and round stats
        level_code = self._level_code(tournament_level)
        p1_level_wr, p1_level_matches = self._level_stats(matches1, level_code)
        p2_level_wr, p2_level_matches = self._level_stats(matches2, level_code)
        p1_round_wr = self._round_winrate(matches1, round_code)
        p2_round_wr = self._round_winrate(matches2, round_code)
        p1_sf_wr, p1_f_wr = self._semis_finals_winrates(matches1)
        p2_sf_wr, p2_f_wr = self._semis_finals_winrates(matches2)

        # Vs-lefty win rates
        p1_vs_lefty = self._winrate_vs_hand(matches1, 'L')
        p2_vs_lefty = self._winrate_vs_hand(matches2, 'L')

        # H2H stats (pass session_cache to avoid duplicate requests)
        if use_shared_semantics:
            h2h = self._shared_h2h_stats(
                matches1,
                p2_display,
                when,
                p2_player_id=profile2.get('player_id'),
            )
        elif self.use_store:
            h2h = self._get_h2h_stats(slug1, slug2, session_cache=session_cache,
                                      _frame=_m1_prestitch, _name2=p2_display)
        else:
            h2h = self._get_h2h_stats(slug1, slug2, session_cache=session_cache)

        # One-hot encodings
        seasons = self._season_flags(surface, when)
        levels = self._levels_onehot(tournament_level)
        rounds = self._rounds_onehot(round_code)

        p1_hand = s1['hand']
        p2_hand = s2['hand']
        hand1 = self._hand_onehot(p1_hand, 'P1')
        hand2 = self._hand_onehot(p2_hand, 'P2')
        matchup = self._handedness_matchup(p1_hand, p2_hand)
        c1 = self._country_onehot(s1['country'], 'P1')
        c2 = self._country_onehot(s2['country'], 'P2')

        # Surface transition flag
        if use_shared_semantics:
            st_flag = shared_surface_transition_flag(
                t1['last_surface'], t2['last_surface'], surface
            )
        else:
            st_flag = 1 if ((t1['last_surface'] and t1['last_surface'].lower() != surface.lower()) or
                            (t2['last_surface'] and t2['last_surface'].lower() != surface.lower())) else 0

        p1_height = _coerce_numeric(s1['height'], 'Player1_Height')
        p2_height = _coerce_numeric(s2['height'], 'Player2_Height')
        p1_age = _coerce_numeric(s1['age'], 'Player1_Age')
        p2_age = _coerce_numeric(s2['age'], 'Player2_Age')
        p1_rank = _coerce_numeric(s1['rank'], 'Player1_Rank')
        p2_rank = _coerce_numeric(s2['rank'], 'Player2_Rank')
        p1_rank_points = _coerce_numeric(s1['rank_points'], 'Player1_Rank_Points')
        p2_rank_points = _coerce_numeric(s2['rank_points'], 'Player2_Rank_Points')

        # Assemble all features
        features: Dict[str, float] = {}

        # Direct player attributes
        features.update({
            'Player1_Height': p1_height,
            'Player2_Height': p2_height,
            'Player1_Age': p1_age,
            'Player2_Age': p2_age,
            'Player1_Rank': p1_rank,
            'Player2_Rank': p2_rank,
            'Player1_Rank_Points': p1_rank_points,
            'Player2_Rank_Points': p2_rank_points,

            'P1_Matches_14d': t1['matches_14d'],
            'P1_Matches_30d': t1['matches_30d'],
            'P1_Surface_Matches_30d': t1['surface_matches_30d'],
            'P1_Surface_Matches_90d': t1['surface_matches_90d'],
            'P1_Surface_Experience': t1['surface_experience'],
            'P1_Surface_WinRate_90d': t1['surface_winrate_90d'],
            'P1_WinRate_Last10_120d': t1['winrate_last10_120d'],
            'P1_WinStreak_Current': t1['streak'],
            'P1_Form_Trend_30d': t1['form_trend_30d'],
            'P1_Days_Since_Last': t1['days_since_last'],
            'P1_Sets_14d': t1['sets_14d'],
            'P1_Rust_Flag': (
                t1['rust_flag'] if use_shared_semantics
                else (1 if t1['days_since_last'] > 21 else 0)
            ),
            'P1_Rank_Change_30d': p1_rc30,
            'P1_Rank_Change_90d': p1_rc90,
            'P1_Rank_Volatility_90d': p1_vol90,

            'P2_Matches_14d': t2['matches_14d'],
            'P2_Matches_30d': t2['matches_30d'],
            'P2_Surface_Matches_30d': t2['surface_matches_30d'],
            'P2_Surface_Matches_90d': t2['surface_matches_90d'],
            'P2_Surface_Experience': t2['surface_experience'],
            'P2_Surface_WinRate_90d': t2['surface_winrate_90d'],
            'P2_WinRate_Last10_120d': t2['winrate_last10_120d'],
            'P2_WinStreak_Current': t2['streak'],
            'P2_Form_Trend_30d': t2['form_trend_30d'],
            'P2_Days_Since_Last': t2['days_since_last'],
            'P2_Sets_14d': t2['sets_14d'],
            'P2_Rust_Flag': (
                t2['rust_flag'] if use_shared_semantics
                else (1 if t2['days_since_last'] > 21 else 0)
            ),
            'P2_Rank_Change_30d': p2_rc30,
            'P2_Rank_Change_90d': p2_rc90,
            'P2_Rank_Volatility_90d': p2_vol90,
        })

        # Derived features
        features.update({
            'Height_Diff': p1_height - p2_height,
            'Age_Diff': p1_age - p2_age,
            'Avg_Height': (p1_height + p2_height) / 2,
            'Avg_Age': (p1_age + p2_age) / 2,
            'Rank_Diff': p1_rank - p2_rank,
            'Rank_Points_Diff': p1_rank_points - p2_rank_points,
            'Avg_Rank': (p1_rank + p2_rank) / 2,
            'Avg_Rank_Points': (p1_rank_points + p2_rank_points) / 2,
            'draw_size': int(draw_size),
            'Rank_Ratio': (
                max(p1_rank, p2_rank) / min(p1_rank, p2_rank)
                if min(p1_rank, p2_rank) > 0 else 1.0
            ),
            'Surface_Transition_Flag': st_flag,
        })

        # Peak age flags
        features['P1_Peak_Age'] = 1 if 24 <= p1_age <= 28 else 0
        features['P2_Peak_Age'] = 1 if 24 <= p2_age <= 28 else 0

        # Surfaces
        features.update({
            'Surface_Hard': 1 if surface == 'Hard' else 0,
            'Surface_Clay': 1 if surface == 'Clay' else 0,
            'Surface_Grass': 1 if surface == 'Grass' else 0,
            'Surface_Carpet': 1 if surface == 'Carpet' else 0
        })

        # Seasons, levels, rounds
        features.update(seasons)
        features.update(levels)
        features.update(rounds)

        # Level & round win rates
        features.update({
            'P1_Level_WinRate_Career': p1_level_wr,
            'P1_Level_Matches_Career': p1_level_matches,
            'P2_Level_WinRate_Career': p2_level_wr,
            'P2_Level_Matches_Career': p2_level_matches,
            'P1_Round_WinRate_Career': p1_round_wr,
            'P2_Round_WinRate_Career': p2_round_wr,
            'P1_Semifinals_WinRate': p1_sf_wr,
            'P1_Finals_WinRate': p1_f_wr,
            'P2_Semifinals_WinRate': p2_sf_wr,
            'P2_Finals_WinRate': p2_f_wr,
            'P1_BigMatch_WinRate': self._big_match_wr(matches1),
            'P2_BigMatch_WinRate': self._big_match_wr(matches2),
        })

        # Lefty & handedness & country
        features.update({
            'P1_vs_Lefty_WinRate': p1_vs_lefty,
            'P2_vs_Lefty_WinRate': p2_vs_lefty
        })
        features.update(hand1)
        features.update(hand2)
        features.update(matchup)
        features.update(c1)
        features.update(c2)

        # H2H
        features.update(h2h)

        # Momentum diffs
        features['Rank_Momentum_Diff_30d'] = p1_rc30 - p2_rc30
        features['Rank_Momentum_Diff_90d'] = p1_rc90 - p2_rc90

        # Ensure all 143 features exist in correct order
        final = {}
        defaulted = []
        for k in EXACT_141_FEATURES:
            if k in features and pd.notna(features[k]):
                final[k] = float(features[k]) if isinstance(features[k], (int, float, np.floating)) else features[k]
            else:
                default_val = (
                    shared_feature_default(k)
                    if use_shared_semantics
                    else self._default_for(k, p1=s1, p2=s2, surface=surface)
                )
                final[k] = default_val
                defaulted.append((k, default_val))

        # Guard against NaNs
        for k, v in list(final.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                default_val = (
                    shared_feature_default(k)
                    if use_shared_semantics
                    else self._default_for(k, p1=s1, p2=s2, surface=surface)
                )
                final[k] = default_val
                defaulted.append((k, default_val))

        # Skip one-hot / surface / round / season / country / hand features — those defaulting to 0 is expected
        _SILENT_PREFIXES = (
            'Surface_', 'Season_', 'Level_', 'Round_', 'Country_', 'Hand_',
            'Handedness_', 'P1_Country_', 'P2_Country_', 'P1_Hand_', 'P2_Hand_',
        )
        noisy_defaults = [
            (k, v) for k, v in defaulted
            if not any(k.startswith(pfx) for pfx in _SILENT_PREFIXES)
        ]
        if noisy_defaults:
            p1_slug = profile1.get('slug', slug1)
            p2_slug = profile2.get('slug', slug2)
            print(f"  ⚠️  DEFAULTED FEATURES for {p1_slug} vs {p2_slug}:")
            for k, v in noisy_defaults:
                print(f"       {k} -> {v}")

        # Track round missing as a semantic default
        if not round_code:
            _semantic_defaults.append('round_code=None')

        # Combine all defaults: structural (_default_for) + semantic (ATP points, round)
        all_defaults = noisy_defaults + [(k, None) for k in _semantic_defaults]
        self._last_noisy_defaults = all_defaults  # used by main.py for features_complete

        # Expose defaulted feature names as comma-separated string for logging
        all_default_names = [k for k, _ in noisy_defaults] + _semantic_defaults
        final['_defaulted_features'] = ','.join(all_default_names) if all_default_names else ''

        # Expose resolved round_code and inferred match date so caller (main.py) can log correctly
        final['_resolved_round_code'] = round_code or ''
        final['_build_ref'] = str(when)
        try:
            final['_hist_tail_p1'] = ','.join(str(d)[:10] for d in pd.to_datetime(matches1['date'], errors='coerce').sort_values(ascending=False).head(3)) if not matches1.empty else ''
            final['_hist_tail_p2'] = ','.join(str(d)[:10] for d in pd.to_datetime(matches2['date'], errors='coerce').sort_values(ascending=False).head(3)) if not matches2.empty else ''
        except Exception:
            pass
        final['_resolved_match_date'] = when.strftime('%Y-%m-%d')
        final['_resolved_surface'] = surface or ''
        if use_shared_semantics:
            _, structural_issues = normalize_feature_vector(final, EXACT_141_FEATURES)
            if structural_issues:
                raise ValueError(
                    "base_141_shared live candidate produced an invalid vector: "
                    + ",".join(structural_issues)
                )
            final['_feature_semantics_id'] = self.feature_semantics_id

        return final

    # ========== Defaults ==========

    def _default_for(self, feature_name: str, p1=None, p2=None, surface='Hard') -> float:
        """Default value for missing feature"""
        if 'Rank' in feature_name and 'Points' not in feature_name:
            return 100.0
        if 'Points' in feature_name:
            return 500.0
        if 'Age' in feature_name:
            return 25.0
        if 'Height' in feature_name:
            return 180.0
        if 'WinRate' in feature_name:
            return 0.5
        if any(x in feature_name for x in [
            'Country_', 'Hand_', 'Round_', 'Level_', 'Surface_', 'Handedness_',
            'Season'
        ]):
            return 0.0
        if feature_name in ('draw_size', 'H2H_Total_Matches',
                            'P1_Matches_14d', 'P1_Matches_30d', 'P1_Sets_14d',
                            'P2_Matches_14d', 'P2_Matches_30d', 'P2_Sets_14d',
                            'P1_Days_Since_Last', 'P2_Days_Since_Last',
                            'P1_WinStreak_Current', 'P2_WinStreak_Current',
                            'P1_Surface_Matches_30d', 'P2_Surface_Matches_30d',
                            'P1_Surface_Matches_90d', 'P2_Surface_Matches_90d',
                            'P1_Surface_Experience', 'P2_Surface_Experience'):
            return 0.0
        if feature_name.endswith('_Flag'):
            return 0.0
        return 0.0

    def _defaults_143(self) -> Dict[str, float]:
        """Return dict of all 143 features with default values"""
        return {k: self._default_for(k) for k in EXACT_141_FEATURES}


def main():
    """Test the TA feature calculator"""
    calc = TAFeatureCalculator()

    # Test with two players
    features = calc.build_141_features(
        player1_name="Arthur Cazaux",
        player2_name="Mackenzie McDonald",
        match_date=datetime(2025, 10, 13),
        surface="Hard",
        tournament_level="C",
        draw_size=32,
        round_code="F"
    )

    print(f"✅ Built {len(features)} features from Tennis Abstract")
    print("\nFirst 10 features:")
    for i, k in enumerate(EXACT_141_FEATURES[:10]):
        print(f"  {k}: {features[k]}")

    print(f"\n✅ All 143 features present: {len(features) == 143}")
    print(f"✅ Feature names match schema: {list(features.keys()) == EXACT_141_FEATURES}")


if __name__ == "__main__":
    main()
