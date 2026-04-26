import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from shadow.backfill_performance_v1 import (  # noqa: E402
    DEFAULT_MODEL_VERSION,
    is_correct_pick,
    select_candidates,
    shadow_pick,
)
from shadow.performance_v1_shadow import build_shadow_uid  # noqa: E402


def test_select_candidates_keeps_only_exact_settled_complete_rows():
    rows = pd.DataFrame(
        [
            _row("match_1", "feat_1", "snapshot_v2", "exact_feature_snapshot", "Player A", True),
            _row("match_2", "feat_2", "legacy_backfilled", "legacy_fallback_match", "Player B", True),
            _row("match_3", "feat_3", "snapshot_v2", "exact_feature_snapshot", pd.NA, True),
            _row("match_4", "feat_4", "snapshot_v2", "exact_feature_snapshot", "Player D", False),
        ]
    )

    selected = select_candidates(rows)

    assert selected["match_uid"].tolist() == ["match_1"]
    assert selected["_backfill_feature_snapshot_id"].tolist() == ["feat_1"]


def test_select_candidates_skips_existing_shadow_uid():
    rows = pd.DataFrame(
        [_row("match_1", "feat_1", "snapshot_v2", "exact_feature_snapshot", "Player A", True)]
    )
    existing = {build_shadow_uid("match_1", DEFAULT_MODEL_VERSION, "feat_1")}

    selected = select_candidates(rows, existing_uids=existing)

    assert selected.empty


def test_shadow_pick_and_correct_are_name_normalized():
    assert shadow_pick("Rafael Nadal", "Roger Federer", 0.51) == "Rafael Nadal"
    assert shadow_pick("Rafael Nadal", "Roger Federer", 0.49) == "Roger Federer"
    assert is_correct_pick("Joao Fonseca", "João Fonseca", p1="", p2="") is True
    assert is_correct_pick("Player A", "Player B", p1="", p2="") is False
    assert is_correct_pick("Player A", 1.0, p1="Player A", p2="Player B") is True
    assert is_correct_pick("Player B", 2.0, p1="Player A", p2="Player B") is True
    assert is_correct_pick("Player A", 2.0, p1="Player A", p2="Player B") is False


def _row(match_uid, snapshot_id, logging_quality, rescore_quality, actual_winner, features_complete):
    return {
        "match_uid": match_uid,
        "latest_feature_snapshot_id": snapshot_id,
        "feature_snapshot_id": "",
        "logging_quality": logging_quality,
        "rescore_quality": rescore_quality,
        "actual_winner": actual_winner,
        "features_complete": features_complete,
        "match_date": "2026-04-26",
        "logged_at": "2026-04-26T00:00:00Z",
        "p1": "Player A",
        "p2": "Player B",
    }
