"""Fail-closed evidence contract for paper-bet model attribution.

Settlement and attribution are deliberately separate.  A known result may
close paper exposure, but a settled bet is decision-grade model evidence only
when its own feature snapshot resolves to the exact persisted vector authority
for the same match, run, and oriented players.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from logging_utils import build_feature_snapshot_id, normalize_name


SETTLEMENT_QUALITY_EXACT_UID = "authoritative_result_exact_match_uid"
SETTLEMENT_QUALITY_CANONICAL_ALIAS = "authoritative_result_canonical_match_uid_alias"
SETTLEMENT_QUALITY_COMPATIBILITY_PAIR = "result_compatibility_player_pair_only"
SETTLEMENT_QUALITY_UNATTRIBUTED = "result_recorded_without_attribution_proof"

ATTRIBUTION_QUALITY_EXACT_UID = "exact_match_uid"
ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED = (
    "exact_match_uid_unverified_feature_snapshot"
)
ATTRIBUTION_QUALITY_ROTATED_UID = "unattributed_rotated_match_uid"
ATTRIBUTION_QUALITY_UID_UNLINKED = "uid_unlinked"
ATTRIBUTION_QUALITY_UNVERIFIED = "unverified"

AUTO_RESULT_EVIDENCE_SCHEMA_VERSION = "auto_settle_result_evidence@1.0.0"
PREDICTION_RESULT_EVIDENCE_SCHEMA_VERSION = (
    "prediction_log_result_evidence@1.0.0"
)
REPAIR_RESULT_EVIDENCE_SCHEMA_VERSION = (
    "settled_bet_attribution_repair_evidence@1.1.0"
)
REPAIR_RESULT_EVIDENCE_KIND = (
    "prediction_log_exact_match_uid_feature_snapshot_bound"
)
IDENTITY_TERMINAL_STATUSES = {"identity_conflict", "superseded_identity"}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_evidence_sha256(value: Any) -> str:
    """Return one canonical SHA-256 or blank so malformed claims fail closed."""
    text = _text(value).lower()
    return text if _HASH_RE.fullmatch(text) else ""


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text in {"true", "1", "1.0", "t", "yes", "y"}:
        return True
    if text in {"false", "0", "0.0", "f", "no", "n"}:
        return False
    return None


def parse_actual_winner(value: Any) -> int | None:
    """Return only exact ground-truth labels 1/2; never truncate values."""
    if isinstance(value, bool):
        return None
    text = _text(value)
    if not text:
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    winner = int(number)
    return winner if winner in {1, 2} else None


def _canonical_json(value: Any) -> str:
    """Serialize evidence deterministically without accepting NaN/Infinity."""

    def scalar(item: Any) -> Any:
        if item is None or isinstance(item, (str, int, bool)):
            return item
        if isinstance(item, float):
            if not pd.notna(item) or item in {float("inf"), float("-inf")}:
                raise ValueError("non-finite evidence value")
            return item
        if isinstance(item, Mapping):
            return {str(key): scalar(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [scalar(val) for val in item]
        if pd.isna(item):
            return None
        if hasattr(item, "item"):
            return scalar(item.item())
        return str(item)

    return json.dumps(
        scalar(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureAttributionEvidence:
    """One complete, structurally verified persisted feature-vector binding."""

    feature_snapshot_id: str
    run_id: str
    match_uid: str
    p1: str
    p2: str
    feature_schema_sha256: str
    feature_vector_sha256: str


@dataclass(frozen=True)
class BoundResultEvidence:
    """One in-memory result payload whose hash and identity can be rechecked.

    Storing only a caller-supplied kind/hash pair is not enough to authorize a
    metric claim: the pair could describe a different match or winner.  The
    canonical payload travels to the bet write boundary so that the digest and
    exact UID/player/winner binding can be verified again immediately before
    persistence.  ``__iter__``/``__getitem__`` preserve the historical
    two-value builder API for accounting-only callers and tests.
    """

    kind: str = ""
    sha256: str = ""
    match_uid: str = ""
    p1: str = ""
    p2: str = ""
    actual_winner: int | None = None
    canonical_payload_json: str = ""

    def __iter__(self):
        yield self.kind
        yield self.sha256

    def __getitem__(self, index: int) -> str:
        return (self.kind, self.sha256)[index]


@dataclass(frozen=True)
class PredictionResultConsensus:
    """One compatible UID group with a single canonical winner identity."""

    match_uid: str
    p1: str
    p2: str
    actual_winner: int
    winner_name: str
    loser_name: str
    observations: tuple[dict[str, Any], ...]


def result_evidence_matches_settlement(
    evidence: BoundResultEvidence | None,
    *,
    match_uid: Any,
    p1: Any,
    p2: Any,
    actual_winner: Any,
) -> bool:
    """Revalidate a result payload against the exact settlement identity."""
    if not isinstance(evidence, BoundResultEvidence):
        return False
    kind = _text(evidence.kind)
    digest = normalize_evidence_sha256(evidence.sha256)
    if not kind or not digest or not evidence.canonical_payload_json:
        return False
    if sha256(evidence.canonical_payload_json.encode("utf-8")).hexdigest() != digest:
        return False
    try:
        payload = json.loads(evidence.canonical_payload_json)
    except json.JSONDecodeError:
        return False
    winner = parse_actual_winner(actual_winner)
    evidence_winner = parse_actual_winner(evidence.actual_winner)
    payload_winner = parse_actual_winner(payload.get("actual_winner"))
    expected_uid = _text(match_uid)
    expected_p1, expected_p2 = _text(p1), _text(p2)
    schema_version = _text(payload.get("schema_version"))
    if schema_version == AUTO_RESULT_EVIDENCE_SCHEMA_VERSION:
        source_slug = re.sub(
            r"[^a-z0-9]+", "_", _text(payload.get("source")).lower()
        ).strip("_")
        expected_kind = f"auto_settle_{source_slug}" if source_slug else ""
    elif schema_version == PREDICTION_RESULT_EVIDENCE_SCHEMA_VERSION:
        expected_kind = "prediction_log_exact_match_uid"
    else:
        return False
    if (
        winner is None
        or evidence_winner is None
        or payload_winner is None
        or kind != expected_kind
        or evidence_winner != winner
        or evidence.match_uid != expected_uid
        or normalize_name(evidence.p1) != normalize_name(expected_p1)
        or normalize_name(evidence.p2) != normalize_name(expected_p2)
    ):
        return False
    return bool(
        _text(payload.get("match_uid")) == expected_uid
        and normalize_name(_text(payload.get("p1"))) == normalize_name(expected_p1)
        and normalize_name(_text(payload.get("p2"))) == normalize_name(expected_p2)
        and payload_winner == winner
    )


def _feature_players(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _text(row.get("player1_raw")) or _text(row.get("p1")),
        _text(row.get("player2_raw")) or _text(row.get("p2")),
    )


def _feature_is_complete(row: Mapping[str, Any]) -> bool:
    if "features_complete" in row:
        return _bool(row.get("features_complete")) is True
    if (
        "_has_defaulted_features" not in row
        and "meta_defaulted_features" not in row
        and "defaulted_features" not in row
    ):
        return False
    default_flag = _bool(row.get("_has_defaulted_features"))
    if default_flag is True:
        return False
    if _text(
        row.get("meta_defaulted_features")
        or row.get("defaulted_features")
    ):
        return False
    # Immutable run rows carry both fields. An unreadable boolean is not proof
    # of completeness even when the default list happens to be blank.
    return default_flag is False


def load_feature_attribution_evidence(
    production_dir: str | Path,
) -> dict[str, FeatureAttributionEvidence]:
    """Load exact feature bindings through the shared lineage resolver.

    The resolver accepts the immutable per-run source when present and the
    durable recovered projection under its existing strict compatibility
    contract.  Any global lineage conflict raises; callers should then disable
    metric attribution while allowing accounting settlement to continue.
    """
    from feature_lineage import load_production_feature_lineage
    from models.inference import EXACT_141_FEATURES

    lineage = load_production_feature_lineage(
        production_dir, EXACT_141_FEATURES
    )
    evidence: dict[str, FeatureAttributionEvidence] = {}
    for snapshot_id, authority in lineage.canonical_by_id.items():
        if (
            snapshot_id in lineage.invalid_ids
            or not authority.structurally_verified
            or not authority.run_id
            or not authority.match_uid
            or not authority.schema_sha256
            or not authority.verified_vector_sha256
            or not _feature_is_complete(authority.row)
        ):
            continue
        p1, p2 = _feature_players(authority.row)
        if (
            not normalize_name(p1)
            or not normalize_name(p2)
            or normalize_name(p1) == normalize_name(p2)
            or build_feature_snapshot_id(
                authority.match_uid, authority.run_id, p1, p2
            ) != snapshot_id
        ):
            continue
        evidence[snapshot_id] = FeatureAttributionEvidence(
            feature_snapshot_id=snapshot_id,
            run_id=authority.run_id,
            match_uid=authority.match_uid,
            p1=p1,
            p2=p2,
            feature_schema_sha256=authority.schema_sha256,
            feature_vector_sha256=authority.verified_vector_sha256,
        )
    return evidence


def load_verified_prediction_log(production_dir: str | Path) -> pd.DataFrame:
    """Load predictions with the evaluation ledger's shared lineage verdict."""
    from evaluation.cohorts import load_prediction_log

    return load_prediction_log(str(Path(production_dir)))


def prediction_supports_exact_attribution(row: Mapping[str, Any]) -> bool:
    """Return whether the exact result row carries the snapshot-v2 contract.

    The bet's own feature snapshot is verified separately against persisted
    vector authority. Historical prediction rows may not repeat that bet-time
    vector/hash after later observation enrichment, so they are result identity
    evidence here, not the feature-vector authority.
    """
    return bool(
        _text(row.get("match_uid"))
        and _text(row.get("record_status")).lower()
        not in IDENTITY_TERMINAL_STATUSES
        and _text(row.get("logging_quality")).lower() == "snapshot_v2"
        and _text(row.get("rescore_quality")).lower()
        == "exact_feature_snapshot"
        and _bool(row.get("features_complete")) is True
    )


def _prediction_pair(row: Mapping[str, Any]) -> tuple[str, str] | None:
    p1, p2 = _text(row.get("p1")), _text(row.get("p2"))
    normalized = (normalize_name(p1), normalize_name(p2))
    if not all(normalized) or normalized[0] == normalized[1]:
        return None
    return normalized


def build_prediction_result_consensus(
    predictions: pd.DataFrame,
    match_uid: Any,
    *,
    expected_p1: Any = "",
    expected_p2: Any = "",
    expected_actual_winner: Any = None,
    require_settled: bool = True,
    require_exact_support: bool = True,
) -> PredictionResultConsensus | None:
    """Resolve compatible duplicate UID rows without first-row-wins behavior.

    Every row must describe the same participant pair and remain free of an
    identity tombstone. Settled observations must carry strict 1/2 labels and
    agree on winner identity even when their P1/P2 orientations differ. For a
    historical repair at least one settled observation must itself carry the
    snapshot-v2 exact-attribution contract.
    """
    uid = _text(match_uid)
    if not uid or predictions.empty or "match_uid" not in predictions.columns:
        return None
    candidates = predictions[
        predictions["match_uid"].fillna("").astype(str).str.strip().eq(uid)
    ]
    if candidates.empty:
        return None

    expected_pair = (
        normalize_name(_text(expected_p1)),
        normalize_name(_text(expected_p2)),
    )
    if any(expected_pair) and (
        not all(expected_pair) or expected_pair[0] == expected_pair[1]
    ):
        return None
    expected_pair_set = set(expected_pair) if all(expected_pair) else None
    expected_winner = parse_actual_winner(expected_actual_winner)
    if expected_actual_winner is not None and expected_winner is None:
        return None
    expected_winner_name = ""
    if expected_winner is not None:
        if expected_pair_set is None:
            return None
        expected_winner_name = expected_pair[expected_winner - 1]

    pair_set: set[str] | None = expected_pair_set
    exact_any = False
    exact_settled = False
    winner_names: set[str] = set()
    observations: list[dict[str, Any]] = []
    identity_bad = {
        "conflict", "identity_conflict", "superseded", "superseded_identity",
    }
    for _, row in candidates.iterrows():
        if (
            _text(row.get("record_status")).lower() in IDENTITY_TERMINAL_STATUSES
            or _text(row.get("identity_status")).lower() in identity_bad
        ):
            return None
        pair = _prediction_pair(row)
        if pair is None:
            return None
        if pair_set is None:
            pair_set = set(pair)
        elif set(pair) != pair_set:
            return None

        exact_row = prediction_supports_exact_attribution(row)
        exact_any = exact_any or exact_row
        winner_text = _text(row.get("actual_winner"))
        if not winner_text:
            continue
        winner = parse_actual_winner(winner_text)
        if winner is None or not _text(row.get("settled_at")):
            return None
        winner_name = pair[winner - 1]
        winner_names.add(winner_name)
        exact_settled = exact_settled or exact_row
        observations.append({
            "prediction_uid": _text(row.get("prediction_uid")),
            "p1": _text(row.get("p1")),
            "p2": _text(row.get("p2")),
            "actual_winner": winner,
            "score": _text(row.get("score")),
            "settled_at": _text(row.get("settled_at")),
        })

    if require_exact_support and not exact_any:
        return None
    if require_settled and (
        not observations or (require_exact_support and not exact_settled)
    ):
        return None
    if expected_winner_name:
        if any(name != expected_winner_name for name in winner_names):
            return None
    elif len(winner_names) != 1:
        return None

    observations.sort(key=lambda item: _canonical_json(item))
    if observations:
        representative = observations[0]
        p1, p2 = representative["p1"], representative["p2"]
        winner = int(representative["actual_winner"])
    else:
        p1, p2 = _text(expected_p1), _text(expected_p2)
        winner = expected_winner
    if winner is None:
        return None
    winner_name = p1 if winner == 1 else p2
    loser_name = p2 if winner == 1 else p1
    return PredictionResultConsensus(
        match_uid=uid,
        p1=p1,
        p2=p2,
        actual_winner=winner,
        winner_name=winner_name,
        loser_name=loser_name,
        observations=tuple(observations),
    )


def prediction_match_supports_exact_attribution(
    predictions: pd.DataFrame,
    match_uid: Any,
    *,
    p1: Any = "",
    p2: Any = "",
    actual_winner: Any = None,
) -> bool:
    """Require compatible UID consensus plus at least one exact observation."""
    return bool(
        build_prediction_result_consensus(
            predictions,
            match_uid,
            expected_p1=p1,
            expected_p2=p2,
            expected_actual_winner=actual_winner,
            require_settled=False,
        )
    )


def _match_players(match_label: Any) -> tuple[str, str] | None:
    parts = re.split(r"\s+vs\.?\s+", _text(match_label), flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    p1, p2 = (normalize_name(part) for part in parts)
    if not p1 or not p2 or p1 == p2:
        return None
    return p1, p2


def feature_evidence_matches_bet(
    bet: Mapping[str, Any],
    direct_match_uid: Any,
    feature_evidence: Mapping[str, FeatureAttributionEvidence],
) -> bool:
    """Bind a bet to its exact feature run, UID, orientation, and side."""
    direct_uid = _text(direct_match_uid)
    if not direct_uid or _text(bet.get("match_uid")) != direct_uid:
        return False
    snapshot_id = _text(bet.get("feature_snapshot_id"))
    evidence = feature_evidence.get(snapshot_id)
    if evidence is None:
        return False
    if (
        evidence.match_uid != direct_uid
        or _text(bet.get("run_id")) != evidence.run_id
        or evidence.feature_snapshot_id != snapshot_id
    ):
        return False
    match_players = _match_players(bet.get("match"))
    evidence_players = (
        normalize_name(evidence.p1), normalize_name(evidence.p2)
    )
    if match_players != evidence_players:
        return False
    bet_on = normalize_name(_text(bet.get("bet_on")))
    side = _bool(bet.get("bet_on_player1"))
    if bet_on not in evidence_players or side is None:
        return False
    return bool((bet_on == evidence_players[0]) == side)


def bet_players_match_result(
    bet: Mapping[str, Any], p1: Any, p2: Any
) -> bool:
    """Require the complete exact player pair, allowing orientation reversal."""
    bet_players = _match_players(bet.get("match"))
    result_players = (normalize_name(_text(p1)), normalize_name(_text(p2)))
    return bool(
        bet_players
        and all(result_players)
        and result_players[0] != result_players[1]
        and set(bet_players) == set(result_players)
    )


def bet_outcome_matches_winner(
    bet: Mapping[str, Any], winner_name: Any, loser_name: Any
) -> bool:
    bet_on = normalize_name(_text(bet.get("bet_on")))
    winner = normalize_name(_text(winner_name))
    loser = normalize_name(_text(loser_name))
    if not bet_on or not winner or not loser or winner == loser:
        return False
    expected = "win" if bet_on == winner else "loss" if bet_on == loser else ""
    return bool(expected and _text(bet.get("outcome")).lower() == expected)


def bet_accounting_matches_outcome(bet: Mapping[str, Any]) -> bool:
    """Require finite, internally consistent win/loss settlement arithmetic."""
    outcome = _text(bet.get("outcome")).lower()
    if outcome not in {"win", "loss"}:
        return False
    try:
        stake = float(bet.get("stake"))
        odds = float(bet.get("odds_decimal"))
        actual_profit = float(bet.get("actual_profit"))
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not all(math.isfinite(value) for value in (stake, odds, actual_profit))
        or stake <= 0
        or odds <= 1
    ):
        return False
    expected = stake * (odds - 1.0) if outcome == "win" else -stake
    return math.isclose(actual_profit, expected, rel_tol=1e-9, abs_tol=1e-8)


def build_auto_result_evidence(
    *,
    source_evidence: Any,
    match_uid: Any,
    p1: Any,
    p2: Any,
    actual_winner: Any,
    score: Any,
) -> BoundResultEvidence:
    """Hash the exact result-source payload already written to settlement audit."""
    if not isinstance(source_evidence, Mapping) or not source_evidence:
        return BoundResultEvidence()
    winner = parse_actual_winner(actual_winner)
    if winner is None or not _text(match_uid):
        return BoundResultEvidence()
    source = _text(source_evidence.get("source")) or "tennis_abstract"
    source_slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    if not source_slug:
        return BoundResultEvidence()
    payload = {
        "schema_version": AUTO_RESULT_EVIDENCE_SCHEMA_VERSION,
        "source": source_slug,
        "match_uid": _text(match_uid),
        "p1": _text(p1),
        "p2": _text(p2),
        "actual_winner": winner,
        "score": _text(score),
        "source_evidence": dict(source_evidence),
    }
    try:
        canonical_payload = _canonical_json(payload)
        return BoundResultEvidence(
            kind=f"auto_settle_{source_slug}",
            sha256=sha256(canonical_payload.encode("utf-8")).hexdigest(),
            match_uid=_text(match_uid),
            p1=_text(p1),
            p2=_text(p2),
            actual_winner=winner,
            canonical_payload_json=canonical_payload,
        )
    except (TypeError, ValueError, OverflowError):
        return BoundResultEvidence()


def build_prediction_result_evidence(
    prediction: Mapping[str, Any],
) -> BoundResultEvidence:
    """Hash an exact settled prediction used by the tracker sync path."""
    match_uid = _text(prediction.get("match_uid"))
    settled_at = _text(prediction.get("settled_at"))
    actual_winner = parse_actual_winner(prediction.get("actual_winner"))
    if not match_uid or not settled_at or actual_winner is None:
        return BoundResultEvidence()
    payload = {
        "schema_version": PREDICTION_RESULT_EVIDENCE_SCHEMA_VERSION,
        "source_role": "prediction_log",
        "result_binding": "exact_match_uid",
        "match_uid": match_uid,
        "prediction_uid": _text(prediction.get("prediction_uid")),
        "p1": _text(prediction.get("p1")),
        "p2": _text(prediction.get("p2")),
        "actual_winner": actual_winner,
        "score": _text(prediction.get("score")),
        "settled_at": settled_at,
    }
    canonical_payload = _canonical_json(payload)
    return BoundResultEvidence(
        kind="prediction_log_exact_match_uid",
        sha256=sha256(canonical_payload.encode("utf-8")).hexdigest(),
        match_uid=match_uid,
        p1=_text(prediction.get("p1")),
        p2=_text(prediction.get("p2")),
        actual_winner=actual_winner,
        canonical_payload_json=canonical_payload,
    )


def repair_settled_bet_attribution_frame(
    bets: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_evidence: Mapping[str, FeatureAttributionEvidence],
) -> tuple[pd.DataFrame, int]:
    """Upgrade only wholly blank legacy rows supported by exact evidence.

    Explicit ``metric_eligible=false`` is immutable here. A coherent direct-UID
    unknown may upgrade once exact proof becomes available; aliases, pair-only
    settlements, partial bundles, and conflicting classifications may not.
    """
    output = bets.copy()
    attribution_columns = [
        "settlement_quality",
        "attribution_quality",
        "metric_eligible",
        "result_evidence_kind",
        "result_evidence_sha256",
    ]
    for column in attribution_columns:
        if column not in output.columns:
            output[column] = ""
        output[column] = output[column].astype(object)
    if output.empty or predictions.empty or "match_uid" not in predictions.columns:
        return output, 0

    def repairable_state(bet: Mapping[str, Any]) -> bool:
        values = {column: _text(bet.get(column)) for column in attribution_columns}
        if not any(values.values()):
            return True
        # Forward direct-UID settlement can remain explicitly unknown while a
        # transient lineage source is unavailable. This is the only classified
        # state eligible for later enrichment.
        if values["metric_eligible"]:
            return False
        if (
            values["settlement_quality"] != SETTLEMENT_QUALITY_EXACT_UID
            or values["attribution_quality"]
            != ATTRIBUTION_QUALITY_EXACT_UID_UNVERIFIED
        ):
            return False
        kind = values["result_evidence_kind"]
        digest = normalize_evidence_sha256(values["result_evidence_sha256"])
        return bool(kind) == bool(digest)

    uid_values = predictions["match_uid"].fillna("").astype(str).str.strip()
    consensus_by_uid = {
        uid: build_prediction_result_consensus(
            predictions, uid, require_settled=True
        )
        for uid in uid_values.unique()
        if uid
    }
    bet_ids = output.get(
        "bet_id", pd.Series("", index=output.index)
    ).fillna("").astype(str).str.strip()
    unique_bet_ids = set(
        bet_ids.value_counts().loc[lambda counts: counts.eq(1)].index
    ) - {""}
    repaired = 0
    for index, bet in output.iterrows():
        if (
            _text(bet.get("status")).lower() != "settled"
            or _text(bet.get("bet_id")) not in unique_bet_ids
            or not repairable_state(bet)
        ):
            continue
        uid = _text(bet.get("match_uid"))
        consensus = consensus_by_uid.get(uid)
        if consensus is None:
            continue
        if (
            not feature_evidence_matches_bet(bet, uid, feature_evidence)
            or not bet_players_match_result(bet, consensus.p1, consensus.p2)
            or not bet_outcome_matches_winner(
                bet, consensus.winner_name, consensus.loser_name
            )
            or not bet_accounting_matches_outcome(bet)
        ):
            continue
        feature = feature_evidence[_text(bet.get("feature_snapshot_id"))]
        evidence_payload = {
            "schema_version": REPAIR_RESULT_EVIDENCE_SCHEMA_VERSION,
            "source_role": "prediction_log",
            "result_binding": "exact_match_uid",
            "bet_id": _text(bet.get("bet_id")),
            "match_uid": uid,
            "result_observations": list(consensus.observations),
            "p1": consensus.p1,
            "p2": consensus.p2,
            "actual_winner": consensus.actual_winner,
            "outcome": _text(bet.get("outcome")).lower(),
            "stake": _text(bet.get("stake")),
            "odds_decimal": _text(bet.get("odds_decimal")),
            "actual_profit": _text(bet.get("actual_profit")),
            "bankroll_after": _text(bet.get("bankroll_after")),
            "feature_snapshot_id": feature.feature_snapshot_id,
            "run_id": feature.run_id,
            "feature_schema_sha256": feature.feature_schema_sha256,
            "feature_vector_sha256": feature.feature_vector_sha256,
        }
        output.at[index, "settlement_quality"] = SETTLEMENT_QUALITY_EXACT_UID
        output.at[index, "attribution_quality"] = ATTRIBUTION_QUALITY_EXACT_UID
        output.at[index, "metric_eligible"] = "true"
        existing_kind = _text(bet.get("result_evidence_kind"))
        existing_hash = normalize_evidence_sha256(
            bet.get("result_evidence_sha256")
        )
        if existing_kind and existing_hash:
            # Result provenance is immutable. The enrichment supplies feature
            # proof only and must not replace an already-bound result digest.
            output.at[index, "result_evidence_kind"] = existing_kind
            output.at[index, "result_evidence_sha256"] = existing_hash
        else:
            output.at[index, "result_evidence_kind"] = REPAIR_RESULT_EVIDENCE_KIND
            output.at[index, "result_evidence_sha256"] = _payload_sha256(
                evidence_payload
            )
        repaired += 1
    return output, repaired
