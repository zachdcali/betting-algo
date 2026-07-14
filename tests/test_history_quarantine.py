import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
FEATURES_DIR = PRODUCTION_DIR / "features"
SCRAPING_DIR = PRODUCTION_DIR / "scraping"
for path in (PRODUCTION_DIR, FEATURES_DIR, SCRAPING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from store_history import quarantine_shifted_duplicates  # noqa: E402
from features.ta_feature_calculator import apply_round_offsets_to_history  # noqa: E402


def test_shifted_exact_result_keeps_earliest_copy_only():
    df = pd.DataFrame([
        dict(match_id=1, date="2026-06-29", opp_id=22, result="W", round="Q1",
             score="6-4 6-3", event="Wimbledon", match_source="atp_results", surface="Grass"),
        # Same winner/loser orientation, round, and score shifted +14 days.
        dict(match_id=2, date="2026-07-13", opp_id=22, result="W", round="Q1",
             score="6-4 6-3", event="Wimbledon, Great Britain The Championships",
             match_source="atp_results", surface=""),
        # Same result at a different event is a possible real rematch, not a copy.
        dict(match_id=3, date="2026-07-06", opp_id=22, result="W", round="Q1",
             score="6-4 6-3", event="Queens", match_source="atp_results", surface="Grass"),
        # A different score is a distinct result and must remain.
        dict(match_id=4, date="2026-07-13", opp_id=22, result="W", round="Q1",
             score="7-6 6-3", event="Wimbledon", match_source="atp_results", surface=""),
        # A different provenance source is not silently collapsed.
        dict(match_id=5, date="2026-07-06", opp_id=22, result="W", round="Q1",
             score="6-4 6-3", event="Wimbledon", match_source="sackmann", surface="Grass"),
        # Same exact result outside the quarantine window may be a real rematch.
        dict(match_id=6, date="2026-07-20", opp_id=22, result="W", round="Q1",
             score="6-4 6-3", event="Wimbledon", match_source="atp_results", surface="Hard"),
    ])

    out = quarantine_shifted_duplicates(df)

    assert out["match_id"].tolist() == [1, 3, 4, 5, 6]
    assert out.attrs["shifted_duplicate_rows_quarantined"] == 1
    assert len(df) == 6  # caller-owned history is not mutated


def test_round_offset_quarantine_drops_future_exact_ref_and_bad_dates():
    ref = datetime(2026, 7, 14, 12, 0)
    df = pd.DataFrame([
        dict(row="past", date="2026-07-06", event="Past", level="A", round="R32"),
        # A final inferred from this tournament start lands after prediction time.
        dict(row="future", date="2026-07-14", event="Future", level="A", round="F"),
        # At prediction time is not historical evidence either.
        dict(row="at_ref", date="2026-07-14 12:00", event="Cup", level="D", round="RR"),
        dict(row="bad_date", date="not-a-date", event="Broken", level="A", round="R32"),
    ])

    out = apply_round_offsets_to_history(df, ref=ref)

    assert out["row"].tolist() == ["past"]
    assert (out["date"] < pd.Timestamp(ref)).all()
    assert out.attrs["future_history_rows_quarantined"] == 3
    assert df.loc[0, "date"] == "2026-07-06"  # caller-owned history is not mutated
