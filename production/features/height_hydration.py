"""Deterministic prioritization for run-level player-height hydration.

This module is deliberately pure.  It chooses *which already-canonical player*
should be offered to the existing ATP profile evidence path first; it does not
resolve identities, fetch pages, or make missing values usable for inference.
"""

from __future__ import annotations

from dataclasses import dataclass


EVIDENCE_STATES = {
    "resolved",
    "unobserved",
    "expired_negative",
    "fresh_negative",
}


@dataclass(frozen=True)
class HeightHydrationCandidate:
    """One canonical player whose compatibility-store height is missing."""

    canonical_player_id: int
    player_name: str
    event: str
    opponent_has_height: bool
    evidence_state: str = "unobserved"


def _tour_priority(event: str) -> int:
    """ATP first, then Challenger, then ITF, then an unclassified event."""
    text = str(event or "").casefold()
    if "challenger" in text:
        return 1
    if "itf" in text:
        return 2
    if "atp" in text:
        return 0
    return 3


def _evidence_priority(state: str) -> tuple[int, int]:
    """Keep zero-cost positives first and fresh negatives out of the work lane."""
    if state == "resolved":
        return (0, 0)
    if state == "unobserved":
        return (1, 0)
    if state == "expired_negative":
        return (1, 1)
    # A source-bound fresh negative is still included so the batch can return
    # its cached truth, but it must never displace actionable evidence.
    return (2, 0)


def _priority(candidate: HeightHydrationCandidate) -> tuple:
    evidence_bucket, evidence_detail = _evidence_priority(candidate.evidence_state)
    return (
        evidence_bucket,
        0 if candidate.opponent_has_height else 1,
        _tour_priority(candidate.event),
        evidence_detail,
        candidate.player_name.casefold(),
        candidate.canonical_player_id,
    )


def plan_height_hydration(
    candidates: list[HeightHydrationCandidate],
) -> list[HeightHydrationCandidate]:
    """Dedupe by canonical ID and return a stable, impact-first work order.

    A player can appear more than once on a noisy current board.  The best row
    for that canonical ID wins: a lookup that completes a matchup is preferred,
    Challenger precedes ITF, and actionable evidence precedes a fresh negative.
    Canonical ID and normalized name provide a deterministic final tie-break.
    """
    best_by_player: dict[int, HeightHydrationCandidate] = {}
    for candidate in candidates:
        try:
            player_id = int(candidate.canonical_player_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("canonical_player_id must be a positive integer") from exc
        if player_id <= 0:
            raise ValueError("canonical_player_id must be a positive integer")
        if not str(candidate.player_name or "").strip():
            raise ValueError("player_name must be non-empty")
        if candidate.evidence_state not in EVIDENCE_STATES:
            raise ValueError(
                f"unsupported height evidence state: {candidate.evidence_state!r}"
            )

        existing = best_by_player.get(player_id)
        if existing is None or _priority(candidate) < _priority(existing):
            best_by_player[player_id] = candidate

    return sorted(best_by_player.values(), key=_priority)
