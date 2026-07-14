from hashlib import sha256
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from operations.pending_reconciliation import (  # noqa: E402
    APPLY_AUDIT_COLUMNS,
    AtomicApplyError,
    ReconciliationConflict,
    SettlementPaths,
    _SimulatedProcessCrash,
    apply_settlement_plan,
    build_settlement_plan,
    write_settlement_plan,
)
from operations import pending_reconciliation as reconciliation_module  # noqa: E402
from operations.operational_lock import operational_csv_lock  # noqa: E402
from utils.bet_tracker import (  # noqa: E402
    BANKROLL_COLUMNS,
    BETS_COLUMNS,
    SESSION_COLUMNS,
)


def _bet(**updates):
    row = {column: "" for column in BETS_COLUMNS}
    row.update(
        {
            "bet_id": "b1",
            "session_id": "s1",
            "timestamp": "2026-07-10T01:00:00Z",
            "event": "Test Event",
            "match": "Player One vs Player Two",
            "match_uid": "m1",
            "feature_snapshot_id": "f1",
            "run_id": "r1",
            "bet_on": "Player One",
            "bet_on_player1": "True",
            "odds_decimal": "2.0",
            "stake": "10.0",
            "stake_fraction": "0.01",
            "model_prob": "0.60",
            "market_prob": "0.50",
            "edge": "0.10",
            "kelly_fraction": "0.02",
            "potential_profit": "10.0",
            "potential_loss": "10.0",
            "bankroll_before": "1000.0",
            "model_version": "nn@1.2.0",
            "status": "pending",
            "match_date": "2026-07-10",
            "match_start_time": "2026-07-10T12:00:00Z",
        }
    )
    row.update(updates)
    return row


def _prediction(**updates):
    row = {
        "match_uid": "m1",
        "actual_winner": "1",
        "p1": "Player One",
        "p2": "Player Two",
        "match_date": "2026-07-10",
        "record_status": "settled",
        "settled_at": "2026-07-10T18:00:00Z",
    }
    row.update(updates)
    return row


def _session(**updates):
    row = {column: "" for column in SESSION_COLUMNS}
    row.update(
        {
            "session_id": "s1",
            "start_time": "2026-07-10T00:30:00Z",
            "initial_bankroll": "1000",
            "total_bets_placed": "1",
            "total_staked": "10",
            "total_profit_loss": "0",
            "kelly_multiplier_used": "0.18",
        }
    )
    row.update(updates)
    return row


def _bankroll(**updates):
    row = {column: "" for column in BANKROLL_COLUMNS}
    row.update(
        {
            "timestamp": "2026-07-10T00:30:00Z",
            "session_id": "s1",
            "bankroll": "1000",
            "change_amount": "0",
            "change_reason": "Session started",
            "account_equity": "1000",
            "pending_exposure": "10",
            "available_bankroll": "990",
            "total_staked": "10",
            "num_pending_bets": "1",
            "num_settled_bets": "0",
        }
    )
    row.update(updates)
    return row


def _write_fixture(
    tmp_path,
    *,
    bets=None,
    predictions=None,
    bankroll=None,
    sessions=None,
):
    logs = tmp_path / "logs"
    logs.mkdir()
    bets_path = logs / "all_bets.csv"
    predictions_path = tmp_path / "prediction_log.csv"
    bankroll_path = logs / "bankroll_history.csv"
    sessions_path = logs / "betting_sessions.csv"
    pd.DataFrame(bets or [_bet()], columns=BETS_COLUMNS).to_csv(bets_path, index=False)
    pd.DataFrame(predictions or [_prediction()]).to_csv(predictions_path, index=False)
    pd.DataFrame(
        [_bankroll()] if bankroll is None else bankroll,
        columns=BANKROLL_COLUMNS,
    ).to_csv(bankroll_path, index=False)
    pd.DataFrame(
        [_session()] if sessions is None else sessions,
        columns=SESSION_COLUMNS,
    ).to_csv(sessions_path, index=False)
    return SettlementPaths(
        bets=bets_path,
        predictions=predictions_path,
        bankroll=bankroll_path,
        sessions=sessions_path,
        apply_audit=(
            tmp_path / ".private" / "pending_reconciliation_apply_audit.csv"
        ),
        lock=logs / ".operational_csv.lock",
        transaction_dir=logs / ".pending_reconciliation_transaction",
    )


def _plan_file(tmp_path, paths):
    plan = build_settlement_plan(paths)
    plan_path = tmp_path / "review" / "plan.json"
    write_settlement_plan(plan, plan_path)
    return plan, plan_path


def _hash(path):
    return sha256(path.read_bytes()).hexdigest() if path.exists() else None


def test_plan_is_deterministic_and_accepts_reversed_numeric_orientation(tmp_path):
    paths = _write_fixture(
        tmp_path,
        predictions=[
            _prediction(),
            _prediction(actual_winner="2", p1="Player Two", p2="Player One"),
        ],
    )
    first = build_settlement_plan(paths)
    second = build_settlement_plan(paths)

    assert first == second
    assert first["plan_schema_version"] == "1.0.0"
    assert first["summary"]["eligible_rows"] == 1
    assert first["candidates"][0]["authoritative_winner_name"] == "player one"
    assert first["candidates"][0]["winner_source_values"] == [1, 2]
    assert first["post_state"]["bet_rows"][0]["row"]["status"] == "settled"
    assert first["post_state"]["bankroll_appends"][0]["row"]["account_equity"] == "1010"
    assert first["post_state"]["session_rows"][0]["row"]["total_profit_loss"] == "10"
    assert not paths.apply_audit.exists()
    assert not paths.lock.exists()


def test_apply_updates_all_files_once_and_verified_replay_is_noop(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    result = apply_settlement_plan(
        plan_path,
        expected_digest=plan["plan_digest"],
        paths=paths,
    )
    assert result["status"] == "applied"
    assert result["applied_rows"] == 1

    bets = pd.read_csv(paths.bets, dtype=str, keep_default_na=False)
    assert bets.loc[0, "status"] == "settled"
    assert bets.loc[0, "outcome"] == "win"
    assert bets.loc[0, "actual_profit"] == "10"
    assert bets.loc[0, "bankroll_after"] == "1010"
    assert plan["operation_id"] in bets.loc[0, "notes"]

    bankroll = pd.read_csv(paths.bankroll, dtype=str, keep_default_na=False)
    assert len(bankroll) == 2
    assert bankroll.iloc[-1]["account_equity"] == "1010"
    assert bankroll.iloc[-1]["pending_exposure"] == "0"
    assert bankroll.iloc[-1]["available_bankroll"] == "1010"
    assert bankroll.iloc[-1]["num_pending_bets"] == "0"
    assert bankroll.iloc[-1]["num_settled_bets"] == "1"

    sessions = pd.read_csv(paths.sessions, dtype=str, keep_default_na=False)
    assert sessions.loc[0, "total_profit_loss"] == "10"
    assert sessions.loc[0, "win_rate"] == "1"
    assert sessions.loc[0, "final_bankroll"] == "1010"
    assert sessions.loc[0, "end_time"] == "2026-07-10T18:00:00+00:00"

    audit = pd.read_csv(paths.apply_audit, dtype=str, keep_default_na=False)
    assert tuple(audit.columns) == APPLY_AUDIT_COLUMNS
    assert len(audit) == 1
    assert paths.apply_audit.parent.stat().st_mode & 0o077 == 0
    assert audit.loc[0, "event_id"].startswith("pending_apply_")
    assert audit.loc[0, "plan_digest"] == plan["plan_digest"]
    assert audit.loc[0, "applied_at_utc"] == result["applied_at_utc"]

    hashes_after_apply = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    replay = apply_settlement_plan(
        plan_path,
        expected_digest=plan["plan_digest"],
        paths=paths,
    )
    assert replay == {
        "status": "replay_noop",
        "plan_digest": plan["plan_digest"],
        "applied_rows": 0,
        "previously_applied_rows": 1,
    }
    assert hashes_after_apply == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda bets, predictions, bankroll, sessions: bets.append(
                _bet(
                    bet_id="b1",
                    status="settled",
                    match="Other One vs Other Two",
                    match_uid="other",
                    bet_on="Other One",
                    actual_profit="5",
                )
            ),
            "nonunique_bet_id",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bets.append(
                _bet(bet_id="old", status="settled", actual_profit="5")
            ),
            "duplicate_bet_identity_any_status",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bets[0].update(
                {"match_uid": ""}
            ),
            "blank_match_uid",
        ),
        (
            lambda bets, predictions, bankroll, sessions: predictions[0].update(
                {"actual_winner": "-1", "record_status": "void"}
            ),
            "void_or_cancelled_source_conflict",
        ),
        (
            lambda bets, predictions, bankroll, sessions: predictions[0].update(
                {"actual_winner": "3"}
            ),
            "invalid_nonblank_winner_value",
        ),
        (
            lambda bets, predictions, bankroll, sessions: predictions.append(
                _prediction(actual_winner="2")
            ),
            "conflicting_or_missing_winner_identity",
        ),
        (
            lambda bets, predictions, bankroll, sessions: predictions.append(
                _prediction(p2="Different Player")
            ),
            "reused_or_ambiguous_match_uid",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bets[0].update(
                {"match": "Player One vs Different Player"}
            ),
            "prediction_pair_mismatch",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bets[0].update(
                {"match_date": "2026-07-11"}
            ),
            "prediction_date_mismatch",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bets[0].update(
                {"bet_on_player1": "False"}
            ),
            "bet_side_orientation_mismatch",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bets[0].update(
                {"stake": "0"}
            ),
            "invalid_positive_stake",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bets[0].update(
                {"odds_decimal": "1"}
            ),
            "invalid_decimal_odds",
        ),
        (
            lambda bets, predictions, bankroll, sessions: sessions.clear(),
            "missing_or_nonunique_session_lineage",
        ),
        (
            lambda bets, predictions, bankroll, sessions: sessions.append(_session()),
            "missing_or_nonunique_session_lineage",
        ),
        (
            lambda bets, predictions, bankroll, sessions: sessions[0].update(
                {"start_time": "2026-07-10T02:00:00Z"}
            ),
            "bet_precedes_session_start",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bankroll.clear(),
            "session_missing_bankroll_lineage",
        ),
        (
            lambda bets, predictions, bankroll, sessions: bankroll[0].update(
                {"timestamp": "2030-01-01T00:00:00Z"}
            ),
            "session_start_event_lineage_mismatch",
        ),
    ],
)
def test_high_risk_rows_are_rejected(tmp_path, mutator, reason):
    bets = [_bet()]
    predictions = [_prediction()]
    bankroll = [_bankroll()]
    sessions = [_session()]
    mutator(bets, predictions, bankroll, sessions)
    paths = _write_fixture(
        tmp_path,
        bets=bets,
        predictions=predictions,
        bankroll=bankroll,
        sessions=sessions,
    )
    plan = build_settlement_plan(paths)
    assert plan["summary"]["eligible_rows"] == 0
    assert reason in plan["rejected"][0]["reasons"]


def test_no_name_or_pair_fallback_when_uid_is_wrong(tmp_path):
    paths = _write_fixture(tmp_path, bets=[_bet(match_uid="wrong-exact-uid")])
    plan = build_settlement_plan(paths)
    assert plan["summary"]["eligible_rows"] == 0
    assert plan["rejected"][0]["reasons"] == [
        "match_uid_absent_from_prediction_log"
    ]


def test_legacy_bankroll_header_is_padded_only_by_reviewed_apply(tmp_path):
    paths = _write_fixture(tmp_path)
    legacy = pd.read_csv(paths.bankroll, dtype=str, keep_default_na=False).drop(
        columns=["account_equity", "pending_exposure", "available_bankroll"]
    )
    legacy.to_csv(paths.bankroll, index=False)
    original_hash = _hash(paths.bankroll)

    plan, plan_path = _plan_file(tmp_path, paths)
    assert plan["summary"]["eligible_rows"] == 1
    assert _hash(paths.bankroll) == original_hash

    apply_settlement_plan(
        plan_path,
        expected_digest=plan["plan_digest"],
        paths=paths,
    )
    applied = pd.read_csv(paths.bankroll, dtype=str, keep_default_na=False)
    assert all(
        column in applied.columns
        for column in ("account_equity", "pending_exposure", "available_bankroll")
    )
    assert applied.iloc[-1]["account_equity"] == "1010"


def test_changed_input_hash_refuses_apply_before_any_operational_write(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    original = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    paths.predictions.write_bytes(paths.predictions.read_bytes() + b"\n")

    with pytest.raises(ReconciliationConflict, match="input hash changed"):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
        )
    assert original == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }


def test_plan_tamper_and_wrong_expected_digest_fail_closed(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    with pytest.raises(ReconciliationConflict, match="expected digest"):
        apply_settlement_plan(
            plan_path,
            expected_digest="0" * 64,
            paths=paths,
        )

    tampered = plan_path.read_text().replace('"actual_profit": "10"', '"actual_profit": "99"')
    plan_path.write_text(tampered)
    with pytest.raises(ReconciliationConflict, match="embedded digest"):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
        )


def test_mid_commit_failure_restores_every_original_file(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    originals = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }

    with pytest.raises(AtomicApplyError, match="durable backups restored"):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _fail_after_replace=2,
        )
    assert originals == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }


def test_durable_journal_recovers_simulated_process_death_before_reading(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    originals = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    with pytest.raises(_SimulatedProcessCrash):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _simulate_crash_after_replace=2,
        )
    assert paths.transaction_dir.is_dir()
    assert _hash(paths.bets) != originals[paths.bets]
    assert _hash(paths.bankroll) != originals[paths.bankroll]
    assert _hash(paths.sessions) == originals[paths.sessions]
    assert not paths.apply_audit.exists()

    recovered = apply_settlement_plan(
        plan_path,
        expected_digest=plan["plan_digest"],
        paths=paths,
    )
    assert recovered["status"] == "applied"
    assert not paths.transaction_dir.exists()
    assert pd.read_csv(paths.bets).loc[0, "status"] == "settled"
    assert paths.apply_audit.is_file()


@pytest.mark.parametrize("replace_count", [1, 2, 3, 4])
def test_every_precommit_replacement_boundary_restores_the_complete_old_set(
    tmp_path,
    replace_count,
):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    originals = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    with pytest.raises(_SimulatedProcessCrash):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _simulate_crash_after_replace=replace_count,
        )

    with operational_csv_lock(paths.bets.parent):
        observed = {
            path: _hash(path)
            for path in (
                paths.bets,
                paths.bankroll,
                paths.sessions,
                paths.apply_audit,
            )
        }

    assert observed == originals
    assert not paths.transaction_dir.exists()


def test_next_ordinary_lock_holder_recovers_before_entering_critical_section(
    tmp_path,
):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    originals = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }

    with pytest.raises(_SimulatedProcessCrash):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _simulate_crash_after_replace=2,
        )
    assert paths.transaction_dir.is_dir()

    observed = {}
    with operational_csv_lock(paths.bets.parent):
        observed.update(
            {
                path: _hash(path)
                for path in (
                    paths.bets,
                    paths.bankroll,
                    paths.sessions,
                    paths.apply_audit,
                )
            }
        )

    assert observed == originals
    assert not paths.transaction_dir.exists()


def test_existing_bet_tracker_public_read_recovers_before_loading_csv(tmp_path):
    paths = _write_fixture(tmp_path)
    from utils.bet_tracker import BetTracker

    tracker = BetTracker(str(paths.bets.parent), initial_bankroll=1000)
    plan, plan_path = _plan_file(tmp_path, paths)
    with pytest.raises(_SimulatedProcessCrash):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _simulate_crash_after_replace=2,
        )
    assert paths.transaction_dir.is_dir()

    pending = tracker.get_pending_bets()

    assert pending["bet_id"].tolist() == ["b1"]
    assert not paths.transaction_dir.exists()


def test_waiting_lock_holder_recovers_before_observing_crashed_apply(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    originals = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    started = threading.Event()
    entered = threading.Event()
    observed = {}

    def ordinary_holder():
        started.set()
        with operational_csv_lock(paths.bets.parent):
            observed.update(
                {
                    path: _hash(path)
                    for path in (
                        paths.bets,
                        paths.bankroll,
                        paths.sessions,
                        paths.apply_audit,
                    )
                }
            )
            entered.set()

    thread = threading.Thread(target=ordinary_holder, daemon=True)
    with pytest.raises(_SimulatedProcessCrash):
        with operational_csv_lock(paths.bets.parent):
            thread.start()
            assert started.wait(1)
            assert not entered.wait(0.1)
            apply_settlement_plan(
                plan_path,
                expected_digest=plan["plan_digest"],
                paths=paths,
                _simulate_crash_after_replace=2,
            )

    thread.join(timeout=2)
    assert entered.is_set()
    assert observed == originals
    assert not paths.transaction_dir.exists()


def test_waiting_process_recovers_under_flock_before_observing_csvs(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    originals = [
        _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    ]
    started_path = tmp_path / "observer-started"
    output_path = tmp_path / "observer-output.json"
    script = r"""
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from operations.operational_lock import operational_csv_lock

logs = Path(sys.argv[2])
started = Path(sys.argv[3])
output = Path(sys.argv[4])
targets = [Path(value) for value in sys.argv[5:]]

def file_hash(path):
    return sha256(path.read_bytes()).hexdigest() if path.exists() else None

started.write_text("started", encoding="utf-8")
with operational_csv_lock(logs):
    output.write_text(
        json.dumps([file_hash(path) for path in targets]),
        encoding="utf-8",
    )
"""
    process = None
    try:
        with pytest.raises(_SimulatedProcessCrash):
            with operational_csv_lock(paths.bets.parent):
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(PRODUCTION_DIR),
                        str(paths.bets.parent),
                        str(started_path),
                        str(output_path),
                        str(paths.bets),
                        str(paths.bankroll),
                        str(paths.sessions),
                        str(paths.apply_audit),
                    ],
                    cwd=REPO_ROOT,
                )
                deadline = time.monotonic() + 3
                while (
                    not started_path.exists()
                    and process.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                assert started_path.exists()
                assert not output_path.exists()
                apply_settlement_plan(
                    plan_path,
                    expected_digest=plan["plan_digest"],
                    paths=paths,
                    _simulate_crash_after_replace=2,
                )

        assert process is not None
        assert process.wait(timeout=5) == 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=2)

    assert json.loads(output_path.read_text(encoding="utf-8")) == originals
    assert not paths.transaction_dir.exists()


def test_ordinary_lock_holder_fails_closed_on_unreadable_recovery_journal(
    tmp_path,
):
    paths = _write_fixture(tmp_path)
    paths.transaction_dir.mkdir(mode=0o700)
    manifest_path = paths.transaction_dir / "manifest.json"
    manifest_path.write_text("not-json")
    manifest_path.chmod(0o600)
    entered = False

    with pytest.raises(AtomicApplyError, match="manifest is unreadable"):
        with operational_csv_lock(paths.bets.parent):
            entered = True

    assert entered is False
    assert paths.transaction_dir.is_dir()


def test_ordinary_recovery_rejects_symlinked_transaction_directory(tmp_path):
    paths = _write_fixture(tmp_path)
    external = tmp_path / "external-journal"
    external.mkdir()
    marker = external / "must-remain"
    marker.write_text("outside lock scope", encoding="utf-8")
    paths.transaction_dir.symlink_to(external, target_is_directory=True)

    with pytest.raises(AtomicApplyError, match="must not be a symbolic link"):
        with operational_csv_lock(paths.bets.parent):
            pytest.fail("symlinked recovery journal must not yield")

    assert marker.read_text(encoding="utf-8") == "outside lock scope"
    assert paths.transaction_dir.is_symlink()


def test_ordinary_recovery_rejects_dangling_transaction_symlink(tmp_path):
    paths = _write_fixture(tmp_path)
    paths.transaction_dir.symlink_to(
        tmp_path / "missing-external-journal",
        target_is_directory=True,
    )

    with pytest.raises(AtomicApplyError, match="must not be a symbolic link"):
        with operational_csv_lock(paths.bets.parent):
            pytest.fail("dangling recovery symlink must not yield")

    assert paths.transaction_dir.is_symlink()


@pytest.mark.parametrize("entry_index", [0, 1, 2, 3])
def test_recovery_refuses_every_tampered_target_without_mutating_any_file(
    tmp_path,
    entry_index,
):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    with pytest.raises(_SimulatedProcessCrash):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _simulate_crash_after_replace=2,
        )
    partial = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    manifest_path = paths.transaction_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][entry_index]["target"] = str(
        tmp_path / f"out-of-scope-{entry_index}.csv"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AtomicApplyError, match="canonical allowlist"):
        with operational_csv_lock(paths.bets.parent):
            pytest.fail("tampered journal must not yield")

    assert partial == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }


def test_recovery_refuses_tampered_role_and_backup_traversal_before_mutation(
    tmp_path,
):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    with pytest.raises(_SimulatedProcessCrash):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _simulate_crash_after_replace=2,
        )
    manifest_path = paths.transaction_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partial = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }

    manifest["entries"][3]["role"] = "external_audit"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AtomicApplyError, match="target roles"):
        with operational_csv_lock(paths.bets.parent):
            pytest.fail("wrong-role journal must not yield")
    assert partial == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }

    manifest["entries"][3]["role"] = "apply_audit"
    manifest["entries"][1]["backup_file"] = "../outside-backup"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AtomicApplyError, match="escapes the durable journal"):
        with operational_csv_lock(paths.bets.parent):
            pytest.fail("escaping backup journal must not yield")
    assert partial == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }


def test_corrupt_later_backup_refuses_recovery_before_restoring_earlier_file(
    tmp_path,
):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    with pytest.raises(_SimulatedProcessCrash):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
            _simulate_crash_after_replace=2,
        )
    partial = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    manifest = json.loads(
        (paths.transaction_dir / "manifest.json").read_text(encoding="utf-8")
    )
    later_backup = paths.transaction_dir / manifest["entries"][1]["backup_file"]
    later_backup.unlink()

    with pytest.raises(AtomicApplyError, match="backup is missing"):
        with operational_csv_lock(paths.bets.parent):
            pytest.fail("corrupt-backup journal must not yield")

    assert partial == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }


def test_ordinary_lock_holder_verifies_committed_state_before_cleanup(
    tmp_path,
    monkeypatch,
):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    remove_transaction_dir = reconciliation_module._remove_transaction_dir

    def leave_committed_journal(_transaction_dir):
        raise OSError("injected cleanup interruption")

    monkeypatch.setattr(
        reconciliation_module,
        "_remove_transaction_dir",
        leave_committed_journal,
    )
    result = apply_settlement_plan(
        plan_path,
        expected_digest=plan["plan_digest"],
        paths=paths,
    )
    assert result["status"] == "applied"
    assert paths.transaction_dir.is_dir()
    committed = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }

    monkeypatch.setattr(
        reconciliation_module,
        "_remove_transaction_dir",
        remove_transaction_dir,
    )
    observed = {}
    with operational_csv_lock(paths.bets.parent):
        observed.update(
            {
                path: _hash(path)
                for path in (
                    paths.bets,
                    paths.bankroll,
                    paths.sessions,
                    paths.apply_audit,
                )
            }
        )

    assert observed == committed
    assert not paths.transaction_dir.exists()


def test_partial_or_conflicting_replay_fails_instead_of_reapplying(tmp_path):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    apply_settlement_plan(
        plan_path,
        expected_digest=plan["plan_digest"],
        paths=paths,
    )
    audit = pd.read_csv(paths.apply_audit, dtype=str, keep_default_na=False)
    audit.loc[0, "actual_profit"] = "999"
    audit.to_csv(paths.apply_audit, index=False)

    with pytest.raises(ReconciliationConflict, match="replay conflict"):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
        )


@pytest.mark.parametrize(
    ("target", "column", "value"),
    [
        ("bets", "match_uid", "corrupt"),
        ("bets", "bankroll_after", "999"),
        ("bankroll", "account_equity", "999"),
        ("bankroll", "pending_exposure", "999"),
        ("bankroll", "available_bankroll", "999"),
        ("sessions", "final_bankroll", "999"),
        ("sessions", "total_profit_loss", "999"),
    ],
)
def test_replay_rejects_any_corrupt_post_state(tmp_path, target, column, value):
    paths = _write_fixture(tmp_path)
    plan, plan_path = _plan_file(tmp_path, paths)
    apply_settlement_plan(
        plan_path,
        expected_digest=plan["plan_digest"],
        paths=paths,
    )
    path = getattr(paths, target)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[len(frame) - 1 if target == "bankroll" else 0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ReconciliationConflict):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
        )


@pytest.mark.parametrize(
    ("extra", "expected_prefix"),
    [
        (
            _bet(
                bet_id="mystery",
                status="mystery",
                match="Other One vs Other Two",
                bet_on="Other One",
                match_uid="other",
            ),
            "unsupported_status_at_source_row_",
        ),
        (
            _bet(
                bet_id="bad-settled",
                status="settled",
                outcome="loss",
                actual_profit="999",
                bankroll_after="1999",
                settled_timestamp="2026-07-10T19:00:00Z",
                match="Other One vs Other Two",
                bet_on="Other One",
                match_uid="other",
            ),
            "settled_profit_semantics_mismatch_at_source_row_",
        ),
    ],
)
def test_existing_ledger_integrity_blocks_every_apply_candidate(
    tmp_path, extra, expected_prefix
):
    paths = _write_fixture(tmp_path, bets=[_bet(), extra])
    plan = build_settlement_plan(paths)
    assert plan["summary"]["eligible_rows"] == 0
    assert any(
        reason.startswith(expected_prefix)
        for reason in plan["account"]["global_blockers"]
    )


def test_pending_row_with_terminal_fields_is_never_overwritten(tmp_path):
    paths = _write_fixture(
        tmp_path,
        bets=[
            _bet(
                outcome="win",
                actual_profit="123",
                bankroll_after="1123",
                settled_timestamp="2026-07-10T19:00:00Z",
                notes="already marked",
            )
        ],
    )
    plan = build_settlement_plan(paths)
    assert plan["summary"]["eligible_rows"] == 0
    assert "pending_settlement_fields_not_blank" in plan["rejected"][0]["reasons"]


def test_canonical_lock_blocks_normal_bet_tracker_writer(tmp_path):
    paths = _write_fixture(tmp_path)
    from utils.bet_tracker import BetTracker

    tracker = BetTracker(str(paths.bets.parent), initial_bankroll=1000)
    started = threading.Event()
    finished = threading.Event()

    def writer():
        started.set()
        tracker.start_session(1000, 0.18, "lock test")
        finished.set()

    with operational_csv_lock(paths.bets.parent):
        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        assert started.wait(1)
        assert not finished.wait(0.1)
    thread.join(timeout=2)
    assert finished.is_set()


def test_operator_cannot_choose_a_different_lock_or_recovery_directory(tmp_path):
    paths = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="canonical operational lock"):
        build_settlement_plan(replace(paths, lock=tmp_path / "other.lock"))
    with pytest.raises(ValueError, match="canonical recovery path"):
        build_settlement_plan(
            replace(paths, transaction_dir=tmp_path / "other-transaction")
        )
    with pytest.raises(ValueError, match="canonical recovery allowlist"):
        build_settlement_plan(
            replace(
                paths,
                apply_audit=paths.transaction_dir / "apply-audit.csv",
            )
        )
    with pytest.raises(ValueError, match="must not be inside"):
        build_settlement_plan(
            replace(
                paths,
                predictions=paths.transaction_dir / "prediction-log.csv",
            )
        )


def test_insecure_private_audit_directory_blocks_before_journaling(tmp_path):
    paths = _write_fixture(tmp_path)
    paths.apply_audit.parent.mkdir(mode=0o755)
    paths.apply_audit.parent.chmod(0o755)
    plan, plan_path = _plan_file(tmp_path, paths)
    originals = {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }

    with pytest.raises(AtomicApplyError, match="group/other access"):
        apply_settlement_plan(
            plan_path,
            expected_digest=plan["plan_digest"],
            paths=paths,
        )

    assert originals == {
        path: _hash(path)
        for path in (paths.bets, paths.bankroll, paths.sessions, paths.apply_audit)
    }
    assert not paths.transaction_dir.exists()
