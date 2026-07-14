from hashlib import sha256
from pathlib import Path
import json
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from operations.pending_reconciliation import (  # noqa: E402
    DUPLICATE_IDENTITY,
    EXACT_WINNER,
    ORPHAN_MATCH_UID,
    UNRESOLVED,
    build_pending_review,
    build_summary,
    compact_summary,
    reconcile_paths,
    write_review_exports,
)


def _bet(**updates):
    row = {
        "bet_id": "b1",
        "status": "pending",
        "match": "Player One vs Player Two",
        "match_uid": "m_exact",
        "bet_on": "Player One",
        "bet_on_player1": True,
        "match_date": "2026-07-10",
        "stake": 10.0,
        "odds_decimal": 2.0,
        "timestamp": "2026-07-10 01:00:00",
        "event": "Test",
        "session_id": "s1",
        "feature_snapshot_id": "f1",
        "run_id": "r1",
    }
    row.update(updates)
    return row


def _prediction(**updates):
    row = {
        "match_uid": "m_exact",
        "actual_winner": 1,
        "p1": "Player One",
        "p2": "Player Two",
        "match_date": "2026-07-10",
        "record_status": "settled",
    }
    row.update(updates)
    return row


def _fixtures():
    bets = pd.DataFrame(
        [
            _bet(),
            # Same match/side/date under a changing scrape UID: duplicate and orphan.
            _bet(bet_id="b2", match_uid="m_missing", timestamp="2026-07-10 02:00:00"),
            _bet(
                bet_id="b3", match="Three vs Four", bet_on="Four",
                match_uid="m_orphan", stake=5.0,
            ),
            _bet(
                bet_id="b4", match="Five vs Six", bet_on="Five",
                match_uid="m_pending", stake=7.0,
            ),
            _bet(
                bet_id="b5", match="Seven vs Eight", bet_on="Seven",
                match_uid="m_void", stake=8.0,
            ),
            _bet(
                bet_id="b6", match="Nine vs Ten", bet_on="Nine",
                match_uid="m_conflict", stake=9.0,
            ),
            _bet(bet_id="not_pending", status="settled", stake=99.0),
        ]
    )
    predictions = pd.DataFrame(
        [
            _prediction(),
            _prediction(match_uid="m_pending", actual_winner="", p1="Five", p2="Six"),
            _prediction(match_uid="m_void", actual_winner=-1, p1="Seven", p2="Eight"),
            _prediction(match_uid="m_conflict", actual_winner=1, p1="Nine", p2="Ten"),
            _prediction(match_uid="m_conflict", actual_winner=2, p1="Nine", p2="Ten"),
        ]
    )
    return bets, predictions


def test_classifies_every_pending_row_and_keeps_duplicate_label_orthogonal():
    bets, predictions = _fixtures()
    review = build_pending_review(bets, predictions)
    by_id = review.set_index("bet_id")

    assert len(review) == 6
    assert by_id.loc["b1", "outcome_classification"] == EXACT_WINNER
    assert by_id.loc["b1", "bet_result_if_applied"] == "win"
    assert by_id.loc["b1", "profit_if_applied"] == 10.0
    assert by_id.loc["b2", "outcome_classification"] == ORPHAN_MATCH_UID
    assert by_id.loc["b3", "outcome_classification"] == ORPHAN_MATCH_UID
    assert by_id.loc["b4", "outcome_classification"] == UNRESOLVED
    assert by_id.loc["b5", "outcome_classification"] == UNRESOLVED
    assert by_id.loc["b5", "authoritative_outcome_status"] == "void_or_cancelled"
    assert pd.isna(by_id.loc["b5", "authoritative_actual_winner"])
    assert by_id.loc["b6", "authoritative_outcome_status"] == "ambiguous_conflicting_winners"

    assert bool(by_id.loc["b1", "is_duplicate_pending_identity"])
    assert bool(by_id.loc["b2", "is_duplicate_pending_identity"])
    assert by_id.loc["b1", "pending_identity_key"] == by_id.loc["b2", "pending_identity_key"]
    assert DUPLICATE_IDENTITY in by_id.loc["b1", "review_classifications"]
    assert DUPLICATE_IDENTITY in by_id.loc["b2", "review_classifications"]

    summary = build_summary(review)
    assert summary["pending"] == {
        "rows": 6, "stake_total": 49.0, "invalid_stake_rows": 0,
    }
    assert summary["outcome_classifications"][EXACT_WINNER]["rows"] == 1
    assert summary["outcome_classifications"][ORPHAN_MATCH_UID]["rows"] == 2
    assert summary["outcome_classifications"][UNRESOLVED]["rows"] == 3
    assert summary[DUPLICATE_IDENTITY]["rows"] == 2
    assert summary[DUPLICATE_IDENTITY]["identities"] == 1
    assert summary[DUPLICATE_IDENTITY]["by_outcome_classification"] == {
        EXACT_WINNER: {"rows": 1, "stake_total": 10.0},
        ORPHAN_MATCH_UID: {"rows": 1, "stake_total": 10.0},
        UNRESOLVED: {"rows": 0, "stake_total": 0.0},
    }
    assert summary["integrity"]["every_pending_row_outcome_classified"] is True


def test_match_identity_is_order_insensitive_and_normalized():
    bets = pd.DataFrame(
        [
            _bet(bet_id="a", match="José One vs Anne-Marie Two", bet_on="José One"),
            _bet(
                bet_id="b", match="Anne Marie Two VS Jose One", bet_on="Jose One",
                match_uid="other",
            ),
        ]
    )
    review = build_pending_review(bets, pd.DataFrame([_prediction()]))
    assert review["pending_identity_key"].nunique() == 1
    assert review["duplicate_group_size"].tolist() == [2, 2]


def test_void_or_cancelled_is_never_coerced_to_a_player_two_win():
    bets = pd.DataFrame([_bet(match_uid="void")])
    predictions = pd.DataFrame(
        [_prediction(match_uid="void", actual_winner=-1, record_status="cancelled")]
    )
    row = build_pending_review(bets, predictions).iloc[0]
    assert row["outcome_classification"] == UNRESOLVED
    assert row["authoritative_outcome_status"] == "void_or_cancelled"
    assert pd.isna(row["authoritative_actual_winner"])
    assert row["bet_result_if_applied"] == ""
    assert pd.isna(row["profit_if_applied"])


def test_exact_winner_identity_survives_reversed_prediction_orientation():
    bets = pd.DataFrame([_bet(match_uid="flipped")])
    predictions = pd.DataFrame(
        [
            _prediction(match_uid="flipped", actual_winner=1),
            _prediction(
                match_uid="flipped", actual_winner=2,
                p1="Player Two", p2="Player One",
            ),
        ]
    )
    row = build_pending_review(bets, predictions).iloc[0]
    assert row["outcome_classification"] == EXACT_WINNER
    assert row["authoritative_winner_name"] == "player one"
    assert pd.isna(row["authoritative_actual_winner"])
    assert row["bet_result_if_applied"] == "win"


@pytest.mark.parametrize(
    "second_prediction",
    [
        {"p2": "Player Three"},
        {"match_date": "2026-07-11"},
    ],
)
def test_reused_uid_with_multiple_match_identities_fails_closed(second_prediction):
    bets = pd.DataFrame([_bet(match_uid="reused")])
    predictions = pd.DataFrame(
        [
            _prediction(match_uid="reused"),
            _prediction(match_uid="reused", **second_prediction),
        ]
    )
    row = build_pending_review(bets, predictions).iloc[0]
    assert row["outcome_classification"] == UNRESOLVED
    assert row["authoritative_outcome_status"] == "ambiguous_match_identity"
    assert row["bet_result_if_applied"] == ""
    assert pd.isna(row["profit_if_applied"])


def test_invalid_stakes_and_exact_outcome_odds_are_surfaced():
    bets = pd.DataFrame(
        [
            _bet(bet_id="valid", stake=10.0, odds_decimal=2.0),
            _bet(bet_id="zero", stake=0.0, odds_decimal=2.0),
            _bet(bet_id="negative", stake=-5.0, odds_decimal=2.0),
            _bet(bet_id="infinite_stake", stake=float("inf"), odds_decimal=2.0),
            _bet(bet_id="missing_stake", stake=None, odds_decimal=2.0),
            _bet(bet_id="short_odds", stake=3.0, odds_decimal=1.0),
            _bet(bet_id="infinite_odds", stake=4.0, odds_decimal=float("inf")),
            _bet(bet_id="missing_odds", stake=5.0, odds_decimal=None),
        ]
    )
    review = build_pending_review(bets, pd.DataFrame([_prediction()]))
    summary = build_summary(review)

    assert summary["pending"] == {
        "rows": 8,
        "stake_total": 22.0,
        "invalid_stake_rows": 4,
    }
    assert summary["invalid_exact_outcome_odds"] == {
        "rows": 3,
        "bet_ids": ["infinite_odds", "missing_odds", "short_odds"],
    }
    assert pd.isna(review.set_index("bet_id").loc["infinite_stake", "stake"])


def test_compact_summary_omits_long_ids_and_input_hashes(tmp_path):
    bets, predictions = _fixtures()
    bets_path = tmp_path / "all_bets.csv"
    predictions_path = tmp_path / "prediction_log.csv"
    bets.to_csv(bets_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    _, summary = reconcile_paths(bets_path, predictions_path)

    compact = compact_summary(summary)
    serialized = json.dumps(compact, allow_nan=False)
    assert "bet_ids" not in serialized
    assert "identity_keys" not in serialized
    assert "sha256" not in serialized
    assert compact["inputs"]["bets"] == {"path": str(bets_path.resolve())}
    assert compact["pending"] == summary["pending"]


def test_inputs_are_not_mutated_and_exports_require_explicit_paths(tmp_path):
    bets, predictions = _fixtures()
    bets_path = tmp_path / "all_bets.csv"
    predictions_path = tmp_path / "prediction_log.csv"
    bets.to_csv(bets_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    before = {
        path: sha256(path.read_bytes()).hexdigest()
        for path in (bets_path, predictions_path)
    }

    review, summary = reconcile_paths(bets_path, predictions_path)
    write_review_exports(review, summary)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "all_bets.csv", "prediction_log.csv"
    ]
    assert before == {
        path: sha256(path.read_bytes()).hexdigest()
        for path in (bets_path, predictions_path)
    }

    csv_path = tmp_path / "review" / "pending.csv"
    json_path = tmp_path / "review" / "pending.json"
    write_review_exports(
        review, summary, output_csv=csv_path, output_json=json_path
    )
    assert csv_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["read_only"] is True
    assert len(payload["rows"]) == 6
    assert "NaN" not in json_path.read_text()
    assert "Infinity" not in json_path.read_text()

    first = json_path.read_bytes()
    write_review_exports(review, summary, output_json=json_path)
    assert json_path.read_bytes() == first
    assert before == {
        path: sha256(path.read_bytes()).hexdigest()
        for path in (bets_path, predictions_path)
    }


def test_missing_required_columns_fail_closed():
    bets, predictions = _fixtures()
    with pytest.raises(ValueError, match="bet log missing required columns"):
        build_pending_review(bets.drop(columns=["match_uid"]), predictions)
    with pytest.raises(ValueError, match="prediction log missing required columns"):
        build_pending_review(bets, predictions.drop(columns=["actual_winner"]))
