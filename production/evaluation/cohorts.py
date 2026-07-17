"""Load logs, derive authoritative ground truth, assemble a long scored frame.

This is the only module that knows the on-disk storage format. A future SQLite
migration swaps the ``load_*`` helpers and leaves the rest of the package intact.
"""
from __future__ import annotations

import os
from glob import glob
import json

import numpy as np
import pandas as pd

# Model name -> the prediction_log column holding its P(player1 wins).
MODEL_PROB_COLS = {
    "nn": "model_p1_prob",
    "xgb": "xgb_p1_prob",
    "rf": "rf_p1_prob",
    "market": "market_p1_prob",
}
IDENTITY_TERMINAL_STATUSES = {"identity_conflict", "superseded_identity"}
SHADOW_FAMILIES = ["xgboost", "catboost", "lightgbm", "nn"]
SCORED_COLUMNS = [
    "match_uid", "run_id", "model", "family", "p1_prob",
    "p1_odds_decimal", "p2_odds_decimal", "y1",
    "is_gold", "is_complete", "prediction_time",
    "kalshi_p1_ask", "kalshi_p2_ask", "kalshi_observation_at",
]


def _identity_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def feature_identity_contract(
    row: pd.Series | dict,
    *,
    match_uid: str | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    """Extract the immutable match/run/oriented-player snapshot contract."""
    return {
        "match_uid": _identity_text(
            row.get("match_uid") if match_uid is None else match_uid
        ),
        "run_id": _identity_text(
            row.get("run_id") if run_id is None else run_id
        ),
        "p1": _identity_text(row.get("player1_raw"))
              or _identity_text(row.get("p1")),
        "p2": _identity_text(row.get("player2_raw"))
              or _identity_text(row.get("p2")),
    }


def prediction_feature_identity_matches(
    frame: pd.DataFrame,
    authority_contracts: dict[str, dict[str, str]],
) -> pd.Series:
    """Verify match UID, run, orientation, and deterministic snapshot ID."""
    from logging_utils import build_feature_snapshot_id, normalize_name

    def matches(row: pd.Series) -> bool:
        snapshot_id = _identity_text(row.get("feature_snapshot_id"))
        authority = authority_contracts.get(snapshot_id)
        if not snapshot_id or not authority:
            return False
        prediction = feature_identity_contract(row)
        values = [*prediction.values(), *authority.values()]
        if not all(values):
            return False
        return bool(
            prediction["match_uid"] == authority["match_uid"]
            and prediction["run_id"] == authority["run_id"]
            and normalize_name(prediction["p1"]) == normalize_name(authority["p1"])
            and normalize_name(prediction["p2"]) == normalize_name(authority["p2"])
            and build_feature_snapshot_id(
                prediction["match_uid"], prediction["run_id"],
                prediction["p1"], prediction["p2"],
            ) == snapshot_id
            and build_feature_snapshot_id(
                authority["match_uid"], authority["run_id"],
                authority["p1"], authority["p2"],
            ) == snapshot_id
        )

    return frame.apply(matches, axis=1).astype(bool)


def _coerce_probability(value) -> float | None:
    """Return a finite probability in [0, 1], otherwise no evidence."""
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    return probability


def _coerce_decimal_odds(value) -> float:
    """Return valid finite decimal odds, otherwise NaN so ROI skips the row."""
    try:
        odds = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return odds if np.isfinite(odds) and odds > 1.0 else float("nan")


def verify_feature_frame(frame: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    """Validate exact-vector evidence and return id->content hash plus bad IDs."""
    from feature_vector_log import feature_fingerprint
    from models.inference import EXACT_141_FEATURES

    hashes: dict[str, set[str]] = {}
    invalid: set[str] = set()
    if frame.empty or "feature_snapshot_id" not in frame.columns:
        return {}, invalid

    aggregate_format = "features_json" in frame.columns
    for _, row in frame.iterrows():
        snapshot_id = str(row.get("feature_snapshot_id", "") or "").strip()
        if not snapshot_id:
            continue
        try:
            if aggregate_format:
                status = str(row.get("build_status", "") or "").strip().lower()
                complete = str(row.get("features_complete", "")).strip().lower()
                if status != "ok" or complete not in {"true", "1", "1.0", "t", "yes"}:
                    invalid.add(snapshot_id)
                    continue
                payload = json.loads(str(row.get("features_json", "") or ""))
                if not isinstance(payload, dict):
                    raise ValueError("features_json is not an object")
            else:
                status = str(row.get("status", "") or "").strip().lower()
                if status != "ok":
                    invalid.add(snapshot_id)
                    continue
                if any(name not in frame.columns for name in EXACT_141_FEATURES):
                    invalid.add(snapshot_id)
                    continue
                payload = {name: row.get(name) for name in EXACT_141_FEATURES}

            schema_hash, vector_hash, feature_count = feature_fingerprint(payload)
            if not vector_hash or feature_count != len(EXACT_141_FEATURES):
                invalid.add(snapshot_id)
                continue
            stored_schema = str(row.get("feature_schema_sha256", "") or "").strip()
            stored_vector = str(row.get("feature_vector_sha256", "") or "").strip()
            if ((stored_schema and stored_schema != schema_hash)
                    or (stored_vector and stored_vector != vector_hash)):
                invalid.add(snapshot_id)
                continue
            hashes.setdefault(snapshot_id, set()).add(vector_hash)
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid.add(snapshot_id)

    for snapshot_id, values in hashes.items():
        if len(values) != 1:
            invalid.add(snapshot_id)
    return {
        snapshot_id: next(iter(values))
        for snapshot_id, values in hashes.items()
        if snapshot_id not in invalid and len(values) == 1
    }, invalid


def load_prediction_log(prod_dir: str) -> pd.DataFrame:
    from feature_lineage import (
        expected_feature_hash_matches,
        load_production_feature_lineage,
        read_feature_csv,
    )
    from models.inference import EXACT_141_FEATURES

    frame = pd.read_csv(os.path.join(prod_dir, "prediction_log.csv"), low_memory=False)
    scanned_paths: list[str] = []
    legacy_paths: list[str] = []
    paths = [
        *glob(os.path.join(prod_dir, "logs", "features_*.csv")),
        os.path.join(prod_dir, "logs", "feature_vectors.csv"),
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            header = read_feature_csv(path, nrows=0)
        except Exception as exc:
            raise RuntimeError(f"feature snapshot verification could not read {path}: {exc}") from exc
        if "feature_snapshot_id" not in header.columns:
            legacy_paths.append(path)
            continue
        scanned_paths.append(path)
    try:
        lineage = load_production_feature_lineage(prod_dir, EXACT_141_FEATURES)
    except Exception as exc:
        raise RuntimeError(
            f"feature snapshot authority reconciliation failed: {exc}"
        ) from exc
    invalid_ids = set(lineage.invalid_ids)
    evidence = {
        snapshot_id: occurrence.verified_vector_sha256
        for snapshot_id, occurrence in lineage.canonical_by_id.items()
        if snapshot_id not in invalid_ids and occurrence.verified_vector_sha256
    }
    referential_hashes = {
        snapshot_id: lineage.referential_vector_hashes(snapshot_id)
        for snapshot_id in evidence
    }
    authority_contracts = {
        snapshot_id: feature_identity_contract(
            occurrence.row,
            match_uid=occurrence.match_uid,
            run_id=occurrence.run_id,
        )
        for snapshot_id, occurrence in lineage.canonical_by_id.items()
        if snapshot_id not in invalid_ids
    }
    ids = frame.get("feature_snapshot_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    # Referential verification is fail-closed. If no compatible lineage file
    # exists, no row can claim GOLD merely because it carries an ID-shaped
    # string. Publication also fails loudly on partially unreadable inputs.
    expected_hash = frame.get(
        "feature_vector_sha256", pd.Series("", index=frame.index)
    ).fillna("").astype(str).str.strip()
    matched_hash = pd.Series(
        [evidence.get(snapshot_id, "") for snapshot_id in ids], index=frame.index
    )
    expected_matches = pd.Series([
        expected_feature_hash_matches(
            expected, referential_hashes.get(snapshot_id, frozenset())
        )
        for snapshot_id, expected in zip(ids, expected_hash)
    ], index=frame.index)
    identity_matches = prediction_feature_identity_matches(
        frame, authority_contracts
    )
    frame["feature_snapshot_verified"] = (
        ids.ne("") & matched_hash.ne("")
        & expected_matches & identity_matches
    )
    frame.attrs["feature_snapshot_verification"] = {
        "scanned_paths": scanned_paths,
        "legacy_paths_without_ids": legacy_paths,
        "verified_id_count": len(evidence),
        "invalid_id_count": len(invalid_ids),
        "match_uid_mismatch_count": int(
            (ids.ne("") & matched_hash.ne("") & ~identity_matches).sum()
        ),
    }
    return frame


def load_shadow_log(prod_dir: str) -> pd.DataFrame | None:
    path = os.path.join(prod_dir, "logs", "performance_v1_shadow_predictions.csv")
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None


def load_odds_history(prod_dir: str) -> pd.DataFrame | None:
    """Load immutable market observations when the lineage file exists."""
    path = os.path.join(prod_dir, "odds_history.csv")
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None


def load_kalshi_history(prod_dir: str) -> pd.DataFrame | None:
    """Load forward-only raw Kalshi observations when the lineage file exists."""
    path = os.path.join(prod_dir, "kalshi_odds_history.csv")
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None


def _kalshi_price_frame(history: pd.DataFrame | None) -> pd.DataFrame:
    """Collapse two raw yes markets into one exact run/match ask pair."""
    columns = [
        "match_uid", "run_id", "kalshi_p1_ask", "kalshi_p2_ask",
        "kalshi_observation_at",
    ]
    required = {
        "match_uid", "run_id", "polled_at", "event_ticker", "market_ticker",
        "board_side", "yes_ask_dollars", "match_status",
    }
    if history is None or history.empty or not required.issubset(history.columns):
        return pd.DataFrame(columns=columns)
    frame = history.copy()
    frame = frame[
        frame["match_status"].fillna("").astype(str).eq("matched")
    ].copy()
    frame["match_uid"] = frame["match_uid"].fillna("").astype(str).str.strip()
    frame["run_id"] = frame["run_id"].fillna("").astype(str).str.strip()
    frame["board_side"] = frame["board_side"].fillna("").astype(str).str.lower()
    frame["_ask"] = pd.to_numeric(frame["yes_ask_dollars"], errors="coerce")
    frame["_polled_at"] = pd.to_datetime(
        frame["polled_at"], errors="coerce", utc=True, format="mixed",
    )
    frame = frame[
        frame["match_uid"].ne("")
        & frame["run_id"].ne("")
        & frame["board_side"].isin(["p1", "p2"])
        & frame["_ask"].between(0.0, 1.0, inclusive="neither")
        & frame["_polled_at"].notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    grouping = ["match_uid", "run_id", "_polled_at", "event_ticker"]
    for keys, group in frame.groupby(grouping, sort=False):
        if len(group) != 2 or group["market_ticker"].nunique() != 2:
            continue
        side_counts = group["board_side"].value_counts().to_dict()
        if side_counts != {"p1": 1, "p2": 1}:
            continue
        asks = group.set_index("board_side")["_ask"]
        rows.append({
            "match_uid": keys[0],
            "run_id": keys[1],
            "kalshi_p1_ask": float(asks.loc["p1"]),
            "kalshi_p2_ask": float(asks.loc["p2"]),
            "kalshi_observation_at": keys[2],
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    # A normal pipeline run polls once. If a manual run polls more than once,
    # use its earliest complete pair so later price movement cannot improve an
    # already-issued prediction retrospectively.
    return (
        pd.DataFrame(rows)
        .sort_values(["match_uid", "run_id", "kalshi_observation_at"], kind="stable")
        .drop_duplicates(["match_uid", "run_id"], keep="first")
        .reset_index(drop=True)
    )


def _oriented_player_keys(row: pd.Series | dict) -> tuple[str, str] | None:
    """Return normalized (player1, player2), or no identity when incomplete."""
    from logging_utils import normalize_name

    p1 = normalize_name(
        _identity_text(row.get("p1"))
        or _identity_text(row.get("player1_raw"))
    )
    p2 = normalize_name(
        _identity_text(row.get("p2"))
        or _identity_text(row.get("player2_raw"))
    )
    if not p1 or not p2 or p1 == p2:
        return None
    return p1, p2


def _semantic_match_key(row: pd.Series | dict) -> tuple[str, str, str, str] | None:
    """Build the conservative evaluation identity, excluding volatile metadata.

    Round and surface are intentionally absent: they are useful match metadata,
    but changes to either have historically produced multiple operational UIDs
    for one match.  Missing player, date, or event evidence returns ``None`` so
    legacy/incomplete rows are never fuzzy-collapsed.
    """
    from logging_utils import canonicalize_live_event_key

    players = _oriented_player_keys(row)
    if players is None:
        return None

    date_key = ""
    # The latest field is the operational correction/enrichment, while the
    # original field remains the fallback for historical rows.
    for field in ("latest_match_date", "match_date"):
        value = _identity_text(row.get(field))
        if not value:
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            date_key = parsed.date().isoformat()
            break

    event_value = (
        _identity_text(row.get("identity_event_key"))
        or _identity_text(row.get("tournament"))
    )
    event_key = canonicalize_live_event_key(event_value)
    if not date_key or not event_key:
        return None

    left, right = sorted(players)
    return left, right, date_key, event_key


def _semantic_representative_rank(row: pd.Series | dict) -> tuple[int, ...]:
    """Rank equivalent UIDs by decision-grade evidence, never model results.

    The fields are ordered from strongest lineage/cohort guarantees through
    usable model coverage and canonical identity metadata.  Probability values
    affect only whether evidence exists, not whether it was correct. UID text is
    intentionally absent and is used only as the final deterministic tie-break.
    """
    truthy = {"true", "1", "1.0", "t", "yes"}

    def as_bool(value) -> bool:
        return str(value).strip().lower() in truthy

    verified = as_bool(row.get("feature_snapshot_verified"))
    complete = as_bool(row.get("features_complete"))
    snapshot_v2 = (
        _identity_text(row.get("logging_quality")).lower() == "snapshot_v2"
    )
    exact_rescore = (
        _identity_text(row.get("rescore_quality")).lower()
        == "exact_feature_snapshot"
    )
    decision_grade = verified and complete and snapshot_v2 and exact_rescore
    model_coverage = sum(
        _coerce_probability(row.get(column)) is not None
        for column in MODEL_PROB_COLS.values()
    )
    identity_status = _identity_text(row.get("identity_status")).lower()
    canonical_rank = {
        "canonical_alias": 2,
        "canonical": 1,
    }.get(identity_status, 0)
    return (
        int(decision_grade),
        int(verified),
        int(complete),
        int(snapshot_v2),
        int(exact_rescore),
        model_coverage,
        canonical_rank,
    )


def build_ground_truth(pred_log: pd.DataFrame) -> pd.Series:
    """Authoritative representative match_uid -> oriented binary outcome.

    A complete semantic identity is normalized unordered players + effective
    match date + canonical event key. Equivalent operational UIDs contribute
    exactly one deterministic representative, selected by decision-grade
    lineage/coverage quality with UID text only as the final tie-break. A
    contradictory winner for that same semantic match excludes the whole group.
    Rows without a complete semantic identity retain the legacy per-UID,
    fail-closed behavior and are never merged across UIDs.
    """
    winner = pd.to_numeric(pred_log["actual_winner"], errors="coerce")
    # Only explicit player 1/2 outcomes are ground truth. Historical void
    # sentinels such as -1 must never become an implicit player-two win.
    status = pred_log.get(
        "record_status", pd.Series("", index=pred_log.index)
    ).fillna("").astype(str).str.strip().str.lower()
    terminal_uids = set(
        pred_log.loc[
            status.isin(IDENTITY_TERMINAL_STATUSES), "match_uid"
        ].dropna()
    )
    identity_eligible = ~pred_log["match_uid"].isin(terminal_uids)
    settled = pred_log[winner.isin([1, 2]) & identity_eligible].copy()
    settled = settled[
        settled["match_uid"].map(_identity_text).ne("")
    ]

    # build_scored_frame uses the last operational row for each retained UID.
    # Derive y1 against that exact row orientation, not whichever settlement
    # observation happened to be encountered first.
    latest_by_uid = pred_log.drop_duplicates("match_uid", keep="last").set_index(
        "match_uid", drop=False
    )
    semantic_candidates: list[dict] = []
    conflicted_semantic_keys: set[tuple[str, str, str, str]] = set()
    legacy_results: dict[str, int] = {}

    for uid, uid_rows in settled.groupby("match_uid", sort=True):
        semantic_keys = [_semantic_match_key(row) for _, row in uid_rows.iterrows()]
        complete_semantic = [key for key in semantic_keys if key is not None]

        if complete_semantic:
            # Partially described or semantically reused UIDs are not safe
            # evidence. They must not fall back to positional winner integers.
            if (
                len(complete_semantic) != len(uid_rows)
                or len(set(complete_semantic)) != 1
            ):
                conflicted_semantic_keys.update(complete_semantic)
                continue
            winner_players = set()
            valid_outcomes = True
            for (_, row), semantic_key in zip(uid_rows.iterrows(), semantic_keys):
                players = _oriented_player_keys(row)
                outcome = int(pd.to_numeric(row.get("actual_winner"), errors="coerce"))
                if players is None or semantic_key is None:
                    valid_outcomes = False
                    break
                winner_players.add(players[0] if outcome == 1 else players[1])
            if not valid_outcomes or len(winner_players) != 1:
                conflicted_semantic_keys.update(complete_semantic)
                continue

            reference = latest_by_uid.loc[uid]
            reference_players = _oriented_player_keys(reference)
            semantic_key = complete_semantic[0]
            winner_player = next(iter(winner_players))
            if (
                reference_players is None
                or _semantic_match_key(reference) != semantic_key
                or winner_player not in reference_players
            ):
                conflicted_semantic_keys.add(semantic_key)
                continue
            semantic_candidates.append({
                "match_uid": uid,
                "semantic_key": semantic_key,
                "winner_player": winner_player,
                "y1": int(winner_player == reference_players[0]),
                "representative_rank": _semantic_representative_rank(reference),
            })
            continue

        # Legacy rows without enough semantic evidence keep the historical
        # per-UID rule. Opposite positional outcomes remain a conflict.
        positional = (
            pd.to_numeric(uid_rows["actual_winner"], errors="coerce") == 1
        ).astype(int)
        if positional.nunique() == 1:
            legacy_results[str(uid)] = int(positional.iloc[-1])

    semantic_results: dict[str, int] = {}
    if semantic_candidates:
        candidates = pd.DataFrame(semantic_candidates)
        for semantic_key, group in candidates.groupby("semantic_key", sort=True):
            if semantic_key in conflicted_semantic_keys:
                continue
            # Resolve outcomes in canonical player space before choosing a UID;
            # this is what makes reversed P1/P2 ordering safe.
            if group["winner_player"].nunique() != 1:
                continue
            best_rank = max(group["representative_rank"])
            safest = group[group["representative_rank"] == best_rank]
            representative_uid = min(safest["match_uid"].astype(str))
            representative = group[group["match_uid"] == representative_uid].iloc[0]
            semantic_results[representative_uid] = int(representative["y1"])

    results = {**legacy_results, **semantic_results}
    ground_truth = pd.Series(results, dtype="int64", name="y1")
    ground_truth.index.name = "match_uid"
    return ground_truth.sort_index()


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Robustly coerce a truthy column to bool regardless of source representation.

    CSVs give Python bools; a SQLite round-trip can yield "1"/"0" or "True"/"False"
    strings or 1/0 ints. Normalize them all so cohort tiers behave identically
    whether the data came from a CSV or the DB.
    """
    truthy = {"true", "1", "1.0", "t", "yes"}
    return series.map(lambda v: str(v).strip().lower() in truthy).astype(bool)


def _tier_flags(pred_log: pd.DataFrame) -> pd.DataFrame:
    f = pred_log.copy()
    status = f.get(
        "record_status", pd.Series("", index=f.index)
    ).fillna("").astype(str).str.strip().str.lower()
    terminal_uids = set(
        f.loc[status.isin(IDENTITY_TERMINAL_STATUSES), "match_uid"].dropna()
    )
    identity_eligible = ~f["match_uid"].isin(terminal_uids)
    if "features_complete" in f.columns:
        f["is_complete"] = _coerce_bool(f["features_complete"]) & identity_eligible
    else:
        f["is_complete"] = False
    lq = f["logging_quality"] if "logging_quality" in f.columns else pd.Series(index=f.index, dtype=object)
    rq = f["rescore_quality"] if "rescore_quality" in f.columns else pd.Series(index=f.index, dtype=object)
    verified = (
        _coerce_bool(f["feature_snapshot_verified"])
        if "feature_snapshot_verified" in f.columns
        else pd.Series(False, index=f.index, dtype=bool)
    )
    f["is_gold"] = (
        f["is_complete"] & verified
        & (lq == "snapshot_v2") & (rq == "exact_feature_snapshot")
    )
    flags = (
        f[["match_uid", "is_complete", "is_gold"]]
        .groupby("match_uid", as_index=False, dropna=False)
        .any()
    )
    terminal_mask = flags["match_uid"].isin(terminal_uids)
    flags.loc[terminal_mask, ["is_complete", "is_gold"]] = False
    return flags


def _market_start_timestamp(row: pd.Series) -> pd.Timestamp | None:
    """Return a UTC start, using the documented Bovada ET fallback for legacy rows."""
    exact = str(row.get("match_start_at_utc", "") or "").strip()
    if exact:
        parsed = pd.to_datetime(exact, errors="coerce", utc=True, format="mixed")
        if not pd.isna(parsed):
            return parsed
    legacy = str(row.get("match_start_time", "") or "").strip()
    if not legacy:
        return None
    try:
        parsed = pd.Timestamp(legacy)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(
                "America/New_York", ambiguous="NaT", nonexistent="NaT"
            )
        if pd.isna(parsed):
            return None
        return parsed.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return None


def _market_timing_rows(
    odds_history: pd.DataFrame | None,
    gt: pd.Series,
    tiers: pd.DataFrame,
) -> list[dict]:
    """Build first-observed and last-pre-start market evidence.

    A closing comparison is only meaningful when at least two distinct capture
    times exist for a match.  Legacy display times are interpreted in
    America/New_York, the explicit Bovada contract. The observation clock uses
    the explicitly UTC ``odds_scraped_at`` first; legacy ``logged_at`` is only a
    fallback because older writers emitted it without a timezone offset.
    """
    if odds_history is None or odds_history.empty or "match_uid" not in odds_history:
        return []
    frame = odds_history[odds_history["match_uid"].isin(gt.index)].copy()
    if frame.empty:
        return []
    logged = pd.to_datetime(
        frame.get("logged_at", pd.Series("", index=frame.index)),
        errors="coerce", utc=True, format="mixed",
    )
    scraped = pd.to_datetime(
        frame.get("odds_scraped_at", pd.Series("", index=frame.index)),
        errors="coerce", utc=True, format="mixed",
    )
    frame["_observation_at"] = scraped.fillna(logged)
    frame["_match_start_at"] = frame.apply(_market_start_timestamp, axis=1)
    frame["_market_probability"] = pd.to_numeric(
        frame.get("market_p1_prob", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    frame = frame[
        frame["_observation_at"].notna()
        & frame["_match_start_at"].notna()
        & frame["_market_probability"].between(0.0, 1.0)
        & (frame["_observation_at"] < frame["_match_start_at"])
    ].copy()
    if frame.empty:
        return []
    frame = frame.sort_values(
        ["match_uid", "_observation_at"], kind="stable"
    ).drop_duplicates(["match_uid", "_observation_at"], keep="last")
    observation_counts = frame.groupby("match_uid")["_observation_at"].nunique()
    comparable = set(observation_counts[observation_counts >= 2].index)
    frame = frame[frame["match_uid"].isin(comparable)]
    if frame.empty:
        return []

    selected = {
        "market_open": frame.drop_duplicates("match_uid", keep="first"),
        "market_close": frame.drop_duplicates("match_uid", keep="last"),
    }
    rows: list[dict] = []
    for model, observations in selected.items():
        for _, observation in observations.iterrows():
            uid = observation["match_uid"]
            if uid not in tiers.index:
                continue
            rows.append({
                "match_uid": uid,
                "run_id": _identity_text(observation.get("run_id")),
                "model": model,
                "family": "market",
                "p1_prob": float(observation["_market_probability"]),
                "p1_odds_decimal": _coerce_decimal_odds(
                    observation.get("p1_odds_decimal")
                ),
                "p2_odds_decimal": _coerce_decimal_odds(
                    observation.get("p2_odds_decimal")
                ),
                "y1": int(gt.loc[uid]),
                "is_gold": bool(tiers.loc[uid, "is_gold"]),
                "is_complete": bool(tiers.loc[uid, "is_complete"]),
                "prediction_time": observation["_observation_at"],
            })
    return rows


def build_scored_frame(
    pred_log: pd.DataFrame,
    shadow_log: pd.DataFrame | None,
    odds_history: pd.DataFrame | None = None,
    kalshi_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long format: one row per (match_uid, model), settled rows with a non-null prob."""
    kalshi_logging_start = ""
    if (
        kalshi_history is not None
        and not kalshi_history.empty
        and "polled_at" in kalshi_history
    ):
        observed_at = pd.to_datetime(
            kalshi_history["polled_at"], errors="coerce", utc=True, format="mixed",
        ).dropna()
        if not observed_at.empty:
            kalshi_logging_start = observed_at.min().isoformat()
    gt = build_ground_truth(pred_log)
    tiers = _tier_flags(pred_log).set_index("match_uid")
    settled = (
        pred_log[pred_log["match_uid"].isin(gt.index)]
        .drop_duplicates("match_uid", keep="last")
        .set_index("match_uid")
    )
    operational_feature_ids = (
        settled.get("feature_snapshot_id", pd.Series("", index=settled.index))
        .fillna("").astype(str)
    )
    rows = []
    for model, col in MODEL_PROB_COLS.items():
        if col not in settled.columns:
            continue
        for uid, r in settled.iterrows():
            probability = _coerce_probability(r[col])
            if probability is None:
                continue
            rows.append({
                "match_uid": uid, "run_id": _identity_text(r.get("run_id")),
                "model": model, "family": model,
                "p1_prob": probability,
                "p1_odds_decimal": _coerce_decimal_odds(r.get("p1_odds_decimal")),
                "p2_odds_decimal": _coerce_decimal_odds(r.get("p2_odds_decimal")),
                "y1": int(gt.loc[uid]),
                "is_gold": bool(tiers.loc[uid, "is_gold"]),
                "is_complete": bool(tiers.loc[uid, "is_complete"]),
                "prediction_time": r.get("logged_at", r.get("odds_scraped_at")),
            })

    rows.extend(_market_timing_rows(odds_history, gt, tiers))

    if shadow_log is not None and "model_family" in shadow_log.columns:
        sh = shadow_log[shadow_log["match_uid"].isin(gt.index)].copy()
        version = sh.get("model_version", sh["model_family"]).fillna("").astype(str).str.strip()
        sh["_model_label"] = version.where(version.ne(""), sh["model_family"].astype(str))
        if "feature_snapshot_id" in sh.columns:
            opening = operational_feature_ids.rename("_operational_feature_snapshot_id")
            sh = sh.merge(opening, left_on="match_uid", right_index=True, how="left")
            opening_id = sh["_operational_feature_snapshot_id"].fillna("").astype(str)
            shadow_id = sh["feature_snapshot_id"].fillna("").astype(str)
            sh = sh[opening_id.eq("") | shadow_id.eq(opening_id)].copy()
        if "logged_at" in sh.columns:
            sh["_logged_sort"] = pd.to_datetime(
                sh["logged_at"], errors="coerce", utc=True, format="mixed"
            )
            sh = sh.sort_values(["_logged_sort"], kind="stable", na_position="last")
        # One deterministic opening observation per match/model variant. Hourly
        # repeats are correlated evidence, not independent samples.
        sh = sh.drop_duplicates(["match_uid", "_model_label"], keep="first")
        for _, r in sh.iterrows():
            probability = _coerce_probability(r.get("shadow_p1_prob"))
            if probability is None:
                continue
            uid = r["match_uid"]
            in_tiers = uid in tiers.index
            # Key by model_version so multiple variants of the same family (e.g.
            # several XGB recency/depth configs) stay distinct in the ledger.
            # Fall back to family when version is absent (older rows / fixtures).
            label = str(r["_model_label"])
            rows.append({
                "match_uid": uid,
                "run_id": _identity_text(r.get("run_id")),
                "model": f"shadow_{label}",
                "family": r["model_family"],
                "p1_prob": probability,
                "p1_odds_decimal": _coerce_decimal_odds(r.get("p1_odds_decimal")),
                "p2_odds_decimal": _coerce_decimal_odds(r.get("p2_odds_decimal")),
                "y1": int(gt.loc[uid]),
                "is_gold": bool(tiers.loc[uid, "is_gold"]) if in_tiers else False,
                "is_complete": bool(tiers.loc[uid, "is_complete"]) if in_tiers else False,
                "prediction_time": settled.loc[uid].get(
                    "logged_at", settled.loc[uid].get("odds_scraped_at")
                ),
            })

    frame = pd.DataFrame(rows, columns=SCORED_COLUMNS)
    prices = _kalshi_price_frame(kalshi_history)
    if not prices.empty and not frame.empty:
        frame = frame.drop(columns=[
            "kalshi_p1_ask", "kalshi_p2_ask", "kalshi_observation_at",
        ]).merge(
            prices,
            on=["match_uid", "run_id"],
            how="left",
            validate="many_to_one",
        )
    # Preserve a typed empty result so a fully invalid hydrated cohort produces
    # an empty ledger instead of failing during boolean tier selection.
    frame["is_gold"] = frame["is_gold"].astype(bool)
    frame["is_complete"] = frame["is_complete"].astype(bool)
    # Source-level start is deliberately independent of settlement. This lets
    # the dashboard say "0 bets since <first poll>" while the first matched
    # cohort is still pending instead of hiding the tiny forward-only sample.
    frame.attrs["kalshi_logging_start"] = kalshi_logging_start
    return frame


def intersection_uids(scored: pd.DataFrame, models: list[str], tier_col: str) -> set:
    """match_uids that appear for *every* listed model within the given tier."""
    sub = scored[scored[tier_col] & scored["model"].isin(models)]
    counts = sub.groupby("match_uid")["model"].nunique()
    needed = len(set(models))
    return set(counts[counts == needed].index)
