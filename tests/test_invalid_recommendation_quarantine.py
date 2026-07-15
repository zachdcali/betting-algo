from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))


def _prediction_row(**overrides):
    from prediction_logger import COLUMNS

    row = {column: "" for column in COLUMNS}
    row.update({
        "logged_at": "2026-07-15T10:05:35+00:00",
        "match_date": "2026-07-15",
        "tournament": "ITF Gubbio",
        "surface": "Clay",
        "level": "25",
        "round": "R32",
        "p1": "Vito Antonio Darderi",
        "p2": "Giacomo Crisostomo",
        "p1_rank": "18.0",
        "p2_rank": "1122.0",
        "model_p1_prob": "0.7708",
        "model_p2_prob": "0.2292",
        "record_status": "pending",
        "identity_status": "canonical",
        "features_complete": "True",
        "run_id": "run_collision",
        "latest_run_id": "run_collision",
        "match_uid": "match_collision",
        "feature_snapshot_id": "feat_collision",
        "latest_feature_snapshot_id": "feat_collision",
        "prediction_uid": "pred_collision",
        "latest_prediction_uid": "pred_collision",
        "match_start_time": "7/15/26 12:00 PM",
        "match_start_at_utc": "2026-07-15T19:00:00+00:00",
        "odds_scraped_at": "2026-07-15T10:05:35+00:00",
    })
    row.update(overrides)
    return row


def _write_prediction(production_dir: Path, **overrides) -> None:
    from prediction_logger import COLUMNS

    production_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [_prediction_row(**overrides)], columns=COLUMNS,
    ).to_csv(production_dir / "prediction_log.csv", index=False)


def _logged_bet(production_dir: Path) -> str:
    from utils.bet_tracker import BetTracker

    tracker = BetTracker(str(production_dir / "logs"), initial_bankroll=1000.0)
    session_id = tracker.start_session(1000.0, 0.18, "quarantine integration")
    slip = pd.DataFrame([{
        "event": "ITF Gubbio",
        "match": "Vito Antonio Darderi vs Giacomo Crisostomo",
        "match_uid": "match_collision",
        "feature_snapshot_id": "feat_collision",
        "run_id": "run_collision",
        "bet_on": "Vito Antonio Darderi",
        "bet_on_player1": True,
        "odds_decimal": 3.75,
        "stake": 10.0,
        "stake_fraction": 0.01,
        "model_prob": 0.7708,
        "market_prob": 0.27,
        "edge": 0.5008,
        "kelly_fraction": 0.10,
        "potential_profit": 27.5,
        "potential_loss": 10.0,
        "bankroll": 1000.0,
        "model_version": "v-test",
        "match_date": "2026-07-15",
        "match_start_time": "7/15/26 12:00 PM",
    }])
    assert tracker.log_bets(slip, session_id, 1000.0) == 1
    return str(pd.read_csv(tracker.bets_file).iloc[0]["bet_id"])


def _remediation_kwargs(bet_id: str) -> dict:
    from utils.bet_tracker import (
        INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
    )

    return {
        "pipeline_paused": True,
        "bet_id": bet_id,
        "reason_code": INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
        "expected_match_uid": "match_collision",
        "expected_feature_snapshot_id": "feat_collision",
        "expected_run_id": "run_collision",
        "expected_p1": "Vito Antonio Darderi",
        "expected_p2": "Giacomo Crisostomo",
        "detail": "Vito Antonio Darderi joined to L. Darderi ranking row",
    }


def test_exact_remediation_quarantines_before_refund_and_repairs_replays(tmp_path):
    from auto_settle import _settlement_uid_gate
    from operations.invalid_recommendation import remediate_invalid_recommendation

    production_dir = tmp_path / "production"
    _write_prediction(production_dir)
    snapshots = production_dir / "prediction_snapshots.csv"
    immutable_bytes = b"immutable,snapshot\nkept,verbatim\n"
    snapshots.write_bytes(immutable_bytes)
    bet_id = _logged_bet(production_dir)

    result = remediate_invalid_recommendation(
        production_dir,
        **_remediation_kwargs(bet_id),
    )
    assert result.prediction_changed is True
    assert result.audit_appended is True
    assert result.bet_refunded is True
    assert snapshots.read_bytes() == immutable_bytes

    prediction = pd.read_csv(
        production_dir / "prediction_log.csv",
        dtype=str,
        keep_default_na=False,
    ).iloc[0]
    assert prediction["record_status"] == "identity_conflict"
    assert prediction["identity_status"] == "conflict"
    assert prediction["features_complete"] == "False"
    assert prediction["identity_conflict_fields"] == (
        "p1_rank,p1_rank_points,ranking_player_identity"
    )
    assert "underlying match result not asserted" in prediction["record_note"]
    assert "immutable_snapshots_retained=true" in prediction["record_note"]
    assert "Player1_Rank=rank_identity_collision" in (
        prediction["defaulted_features"]
    )
    assert "Player1_Rank_Points=rank_identity_collision" in (
        prediction["defaulted_features"]
    )
    # The bad opening evidence remains visible; quarantine classifies it rather
    # than rewriting history to a fabricated replacement rank.
    assert prediction["p1_rank"] == "18.0"
    assert prediction["actual_winner"] == ""

    audits_path = production_dir / "logs" / "audit" / "skipped_live_matches.csv"
    audits = pd.read_csv(audits_path, dtype=str, keep_default_na=False)
    assert len(audits) == 1
    audit = audits.iloc[0]
    assert audit["run_id"] == "run_collision"
    assert audit["match_uid"] == "match_collision"
    assert audit["feature_snapshot_id"] == "feat_collision"
    assert audit["prediction_uid"] == "pred_collision"
    assert audit["stage"] == "administrative_quarantine"
    assert audit["skip_reason_code"] == "match_identity_conflict"
    assert "immutable snapshots retained" in audit["skip_reason_detail"]

    # Auto-settlement sees the operational tombstone and excludes the entire
    # UID before any result-source matching is attempted.
    predictions = pd.read_csv(production_dir / "prediction_log.csv")
    pending, counts = _settlement_uid_gate(predictions, predictions.copy())
    assert pending.empty
    assert counts["identity_terminal_uid"] == 1

    bets = pd.read_csv(production_dir / "logs" / "all_bets.csv")
    bet = bets.loc[bets["bet_id"] == bet_id].iloc[0]
    assert bet["status"] == "void"
    assert bet["outcome"] == "void"
    assert bet["actual_profit"] == pytest.approx(0.0)
    assert str(bet["metric_eligible"]).lower() == "false"

    # Exact replay is a no-op across all terminal writes.
    replay = remediate_invalid_recommendation(
        production_dir,
        **_remediation_kwargs(bet_id),
    )
    assert replay.prediction_changed is False
    assert replay.audit_appended is False
    assert replay.bet_refunded is False
    assert len(pd.read_csv(audits_path)) == 1

    # Repair the narrow crash window where the prediction tombstone reached
    # disk but the matching skipped-live audit did not.
    audits_path.unlink()
    repaired = remediate_invalid_recommendation(
        production_dir,
        **_remediation_kwargs(bet_id),
    )
    assert repaired.prediction_changed is False
    assert repaired.audit_appended is True
    assert repaired.bet_refunded is False
    assert len(pd.read_csv(audits_path)) == 1


def test_remediation_repairs_crash_after_atomic_bet_before_bankroll(
    monkeypatch,
    tmp_path,
):
    from operations.invalid_recommendation import remediate_invalid_recommendation
    from utils.bet_tracker import BetTracker

    production_dir = tmp_path / "production"
    _write_prediction(production_dir)
    bet_id = _logged_bet(production_dir)
    original_log_bankroll_change = BetTracker.log_bankroll_change

    def crash_before_bankroll(*_args, **_kwargs):
        raise RuntimeError("injected crash before bankroll audit")

    monkeypatch.setattr(
        BetTracker,
        "log_bankroll_change",
        crash_before_bankroll,
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        remediate_invalid_recommendation(
            production_dir,
            **_remediation_kwargs(bet_id),
        )

    # The terminal bet replace is complete and parseable; the missing audit is
    # the supported between-file crash window repaired by an exact replay.
    interrupted_bet = pd.read_csv(
        production_dir / "logs" / "all_bets.csv",
    ).iloc[0]
    assert interrupted_bet["status"] == "void"
    assert interrupted_bet["outcome"] == "void"
    assert interrupted_bet["actual_profit"] == pytest.approx(0.0)

    monkeypatch.setattr(
        BetTracker,
        "log_bankroll_change",
        original_log_bankroll_change,
    )
    replay = remediate_invalid_recommendation(
        production_dir,
        **_remediation_kwargs(bet_id),
    )
    assert replay.prediction_changed is False
    assert replay.audit_appended is False
    assert replay.bet_refunded is False

    bankroll = pd.read_csv(
        production_dir / "logs" / "bankroll_history.csv",
    )
    audit = bankroll[
        bankroll["change_reason"].str.contains(
            "Administrative invalid-recommendation refund",
            na=False,
        )
    ]
    assert len(audit) == 1


def test_atomic_csv_replace_preserves_original_on_replace_failure(
    monkeypatch,
    tmp_path,
):
    import logging_utils

    target = tmp_path / "ledger.csv"
    original = b"value\nold\n"
    target.write_bytes(original)

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(logging_utils.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        logging_utils.atomic_write_csv(
            pd.DataFrame([{"value": "new"}]),
            target,
        )
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".ledger.csv.*.tmp"))


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"record_status": "settled", "actual_winner": "1"}, "competing"),
        ({
            "record_status": "identity_conflict",
            "identity_status": "conflict",
            "record_note": "different reviewed conflict",
            "identity_conflict_fields": "round",
            "features_complete": "False",
        }, "competing identity"),
    ],
)
def test_quarantine_rejects_competing_terminal_state(tmp_path, overrides, error):
    from operations.invalid_recommendation import quarantine_invalid_recommendation
    from utils.bet_tracker import (
        INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
    )

    production_dir = tmp_path / "production"
    _write_prediction(production_dir, **overrides)
    before = (production_dir / "prediction_log.csv").read_bytes()
    with pytest.raises(RuntimeError, match=error):
        quarantine_invalid_recommendation(
            production_dir,
            pipeline_paused=True,
            reason_code=INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
            expected_match_uid="match_collision",
            expected_feature_snapshot_id="feat_collision",
            expected_run_id="run_collision",
            expected_p1="Vito Antonio Darderi",
            expected_p2="Giacomo Crisostomo",
            detail="reviewed collision",
        )
    assert (production_dir / "prediction_log.csv").read_bytes() == before
    assert not (
        production_dir / "logs" / "audit" / "skipped_live_matches.csv"
    ).exists()


def test_quarantine_requires_exact_oriented_player_pair(tmp_path):
    from operations.invalid_recommendation import quarantine_invalid_recommendation
    from utils.bet_tracker import (
        INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
    )

    production_dir = tmp_path / "production"
    _write_prediction(production_dir)
    before = (production_dir / "prediction_log.csv").read_bytes()
    with pytest.raises(RuntimeError, match="exactly one row"):
        quarantine_invalid_recommendation(
            production_dir,
            pipeline_paused=True,
            reason_code=INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
            expected_match_uid="match_collision",
            expected_feature_snapshot_id="feat_collision",
            expected_run_id="run_collision",
            expected_p1="Giacomo Crisostomo",
            expected_p2="Vito Antonio Darderi",
            detail="reviewed collision",
        )
    assert (production_dir / "prediction_log.csv").read_bytes() == before


def test_remediation_refuses_to_run_without_pipeline_pause(tmp_path):
    from operations.invalid_recommendation import quarantine_invalid_recommendation
    from utils.bet_tracker import (
        INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
    )

    production_dir = tmp_path / "production"
    _write_prediction(production_dir)
    before = (production_dir / "prediction_log.csv").read_bytes()
    with pytest.raises(RuntimeError, match="pipeline and auto-settlement"):
        quarantine_invalid_recommendation(
            production_dir,
            pipeline_paused=False,
            reason_code=INVALID_RECOMMENDATION_REASON_RANK_IDENTITY_COLLISION,
            expected_match_uid="match_collision",
            expected_feature_snapshot_id="feat_collision",
            expected_run_id="run_collision",
            expected_p1="Vito Antonio Darderi",
            expected_p2="Giacomo Crisostomo",
            detail="reviewed collision",
        )
    assert (production_dir / "prediction_log.csv").read_bytes() == before
