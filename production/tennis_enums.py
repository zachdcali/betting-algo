"""Closed-world tennis values shared by scraping, lineage, and eligibility.

These codes mirror the active ``base_141@1.0.0`` one-hot schema. ``U`` is a
missing-value marker for handedness, not authoritative profile evidence.
"""

from __future__ import annotations


AUTHORITATIVE_HAND_CODES = frozenset({"L", "R", "A"})
UNRESOLVED_HAND_CODE = "U"

ACTIVE_ROUND_CODES = frozenset({
    "Q1", "Q2", "Q3", "Q4",
    "R128", "R64", "R32", "R16",
    "RR", "QF", "SF", "F", "ER", "BR",
})
