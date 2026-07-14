from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from operations.eligibility_coverage import (  # noqa: E402
    summarize_eligibility_coverage,
)


RUN = "run_20260714T210027Z"
SYNC = "sync_20260714T212732Z_33131875"


def _accepted_run_fixture():
    rows = []
    for index in range(83):
        incomplete_index = index - 19
        defaults = []
        status = "pending"
        note = ""
        if incomplete_index >= 0:
            if incomplete_index < 36:
                defaults.append("Player1_Height")
            if incomplete_index < 30:
                status = "identity_conflict"
                note = "match_identity_conflict:event_key"
            if incomplete_index < 21:
                defaults.append("round_code=None")
                note += " feature_schema_invalid:one_hot_cardinality:round:0"
            if incomplete_index < 2:
                defaults.append("P1_Rank_Volatility_90d")
        rows.append({
            "run_id": RUN,
            "feature_snapshot_id": f"feat_{index}",
            "logged_at": f"2026-07-14T21:{index % 60:02d}:00Z",
            "features_complete": incomplete_index < 0,
            "defaulted_features": ",".join(defaults),
            "record_status": status,
            "record_note": note,
        })
    skips = pd.DataFrame([
        {
            "run_id": RUN,
            "skip_event_id": f"skip_{index}",
            "skip_reason_code": "feature_error",
            "skip_reason_detail": f"TA profile load failed for slug: Missing{index}",
        }
        for index in range(3)
    ])
    return pd.DataFrame(rows), skips


def test_latest_accepted_run_counts_and_priority_are_derived_not_summed():
    snapshots, skips = _accepted_run_fixture()
    report = summarize_eligibility_coverage(
        snapshots, run_id=RUN, sync_id=SYNC, skips=skips,
    )

    assert (report.snapshot_rows, report.complete_rows, report.incomplete_rows) == (
        83, 19, 64,
    )
    assert report.orthogonal_snapshot_blockers == {
        "height": 36,
        "identity_conflict": 30,
        "round": 21,
        "structural_validation": 21,
        "rank_volatility": 2,
    }
    assert report.skip_only_blockers == {"source_profile_lookup_failure": 3}
    assert [item.category for item in report.ranked_registry_levers] == [
        "height", "identity_conflict", "round", "source_profile_lookup_failure",
    ]
    assert report.after_replay_rows is None
    assert "orthogonal" in report.caveat


def test_priority_changes_with_run_evidence_instead_of_hard_coding_height_or_round():
    snapshots = pd.DataFrame([
        {
            "run_id": "run_dynamic", "feature_snapshot_id": f"feat_{index}",
            "features_complete": False,
            "defaulted_features": (
                "round_code=None" if index < 4 else "Player1_Height"
            ),
            "record_status": "pending",
        }
        for index in range(5)
    ])
    report = summarize_eligibility_coverage(
        snapshots, run_id="run_dynamic",
    )
    assert [item.category for item in report.ranked_registry_levers[:2]] == [
        "round", "height",
    ]


def test_duplicate_snapshots_and_unrelated_runs_do_not_inflate_counts():
    snapshots = pd.DataFrame([
        {
            "run_id": RUN, "feature_snapshot_id": "feat_same",
            "logged_at": "2026-07-14T21:00:00Z", "features_complete": False,
            "defaulted_features": "Player1_Height",
        },
        {
            "run_id": RUN, "feature_snapshot_id": "feat_same",
            "logged_at": "2026-07-14T21:01:00Z", "features_complete": True,
            "defaulted_features": "",
        },
        {
            "run_id": "run_other", "feature_snapshot_id": "feat_other",
            "features_complete": False, "defaulted_features": "round_code=None",
        },
    ])
    report = summarize_eligibility_coverage(snapshots, run_id=RUN)
    assert report.snapshot_rows == 1
    assert report.complete_rows == 1
    assert sum(report.orthogonal_snapshot_blockers.values()) == 0


def test_absent_run_fails_closed_instead_of_reporting_false_zeroes():
    snapshots = pd.DataFrame([{
        "run_id": "run_other",
        "feature_snapshot_id": "feat_other",
        "features_complete": True,
    }])
    with pytest.raises(ValueError, match="refusing a false zero report"):
        summarize_eligibility_coverage(snapshots, run_id=RUN)
