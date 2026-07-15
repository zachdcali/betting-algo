from pathlib import Path
import sys

import pandas as pd
import pytest


PRODUCTION = Path(__file__).resolve().parents[1] / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

import prediction_logger as PL  # noqa: E402
import auto_settle as AS  # noqa: E402
from main import exclude_identity_conflicts  # noqa: E402
from logging_utils import (  # noqa: E402
    build_feature_snapshot_id,
    build_match_uid,
    canonicalize_live_event_key,
    normalize_name,
    stable_hash,
)


def _configure_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(PL, "LOG_PATH", tmp_path / "prediction_log.csv")
    monkeypatch.setattr(PL, "SNAPSHOT_LOG_PATH", tmp_path / "prediction_snapshots.csv")
    monkeypatch.setattr(PL, "ODDS_HISTORY_LOG_PATH", tmp_path / "odds_history.csv")


def _legacy_raw_event_uid(p1, p2, match_date, event, round_code, surface):
    players = sorted((normalize_name(p1), normalize_name(p2)))
    return "match_" + stable_hash(
        players[0], players[1], match_date, event, round_code, surface,
    )


def _log(
    *,
    p1,
    p2,
    match_date="2026-07-14",
    tournament="Lincoln",
    event_title="Challenger - Lincoln (8)",
    round_code="R32",
    surface="Hard",
    run_id,
    match_uid=None,
    feature_snapshot_id=None,
    features_complete=True,
    p1_hand="",
    p2_hand="",
):
    uid = match_uid or build_match_uid(
        p1, p2, match_date, event_title, round_code, surface,
    )
    snapshot = feature_snapshot_id or build_feature_snapshot_id(
        uid, run_id, p1, p2,
    )
    action = PL.log_prediction(
        p1=p1,
        p2=p2,
        tournament=tournament,
        surface=surface,
        level="C",
        round_code=round_code,
        match_date=match_date,
        run_id=run_id,
        match_uid=uid,
        feature_snapshot_id=snapshot,
        identity_event_key=canonicalize_live_event_key(event_title),
        model_p1_prob=0.61,
        model_p2_prob=0.39,
        market_p1_prob=None,
        market_p2_prob=None,
        model_version="1.2.3",
        nn_model_version="1.2.3",
        features_complete=features_complete,
        p1_hand=p1_hand,
        p2_hand=p2_hand,
    )
    return action, uid, snapshot


def test_prediction_snapshot_preserves_hands_used_by_live_inference(
    tmp_path, monkeypatch,
):
    _configure_paths(monkeypatch, tmp_path)

    _log(
        p1="Player One", p2="Player Two", run_id="run_hands",
        p1_hand="R", p2_hand="U",
    )

    operational = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    snapshots = pd.read_csv(
        PL.SNAPSHOT_LOG_PATH, dtype=str, keep_default_na=False,
    )
    assert operational.loc[0, ["p1_hand", "p2_hand"]].tolist() == ["R", "U"]
    assert snapshots.loc[0, ["p1_hand", "p2_hand"]].tolist() == ["R", "U"]


def test_bovada_bucket_counts_are_not_live_match_identity():
    expected = build_match_uid(
        "Player One", "Player Two", "2026-07-14",
        "Challenger - Lincoln (1)", "R32", "Hard",
    )

    assert canonicalize_live_event_key("Challenger - Lincoln (17)") == (
        "challenger - lincoln"
    )
    for count in range(1, 18):
        assert build_match_uid(
            "Player Two", "Player One", "2026-07-14",
            f"Challenger - Lincoln ({count})", "R32", "Hard",
        ) == expected

    assert build_match_uid(
        "Player One", "Player Two", "2026-07-15",
        "Challenger - Lincoln (17)", "R32", "Hard",
    ) != expected
    assert build_match_uid(
        "Player One", "Player Two", "2026-07-14",
        "Challenger - Lincoln (17)", "R16", "Hard",
    ) != expected


def test_feature_snapshot_for_another_match_fails_before_any_write(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2, run_id = "Player One", "Player Two", "run_bad_pointer"
    correct_uid = build_match_uid(
        p1, p2, "2026-07-14", "Challenger - Lincoln (8)", "R32", "Hard",
    )
    other_uid = build_match_uid(
        p1, p2, "2026-07-14", "Challenger - Lincoln (8)", "R16", "Hard",
    )
    wrong_snapshot = build_feature_snapshot_id(other_uid, run_id, p1, p2)

    with pytest.raises(PL.LiveMatchIdentityError, match="does not belong"):
        _log(
            p1=p1,
            p2=p2,
            run_id=run_id,
            match_uid=correct_uid,
            feature_snapshot_id=wrong_snapshot,
        )

    assert not Path(PL.LOG_PATH).exists()
    assert not Path(PL.SNAPSHOT_LOG_PATH).exists()
    assert not Path(PL.ODDS_HISTORY_LOG_PATH).exists()


def test_identity_conflicts_are_removed_from_shadow_and_staking_inputs():
    predictions = pd.DataFrame([
        {"match_uid": "match_ok", "player1_win_prob": 0.55},
        {"match_uid": "match_conflict", "player1_win_prob": 0.75},
    ])
    bets = pd.DataFrame([
        {"match_uid": "match_conflict", "stake": 50.0},
        {"match_uid": "match_ok", "stake": 10.0},
    ])

    shadow_input, stake_input = exclude_identity_conflicts(
        predictions, bets, {"match_conflict"},
    )

    assert shadow_input["match_uid"].tolist() == ["match_ok"]
    assert stake_input["match_uid"].tolist() == ["match_ok"]


def test_settlement_candidate_scan_excludes_identity_terminal_states(
    tmp_path, monkeypatch,
):
    log_path = tmp_path / "prediction_log.csv"
    rows = []
    for uid, status in (
        ("match_conflict", "identity_conflict"),
        ("match_superseded", "superseded_identity"),
    ):
        row = {column: "" for column in PL.COLUMNS}
        row.update({
            "logged_at": "2026-07-14T10:00:00",
            "match_date": "2026-07-14",
            "p1": f"{uid} A",
            "p2": f"{uid} B",
            "model_p1_prob": 0.6,
            "model_p2_prob": 0.4,
            "model_version": "1.2.3",
            "match_uid": uid,
            "record_status": status,
        })
        rows.append(row)
    pd.DataFrame(rows, columns=PL.COLUMNS).to_csv(log_path, index=False)
    monkeypatch.setattr(AS, "LOG_PATH", log_path)
    monkeypatch.setattr(
        AS,
        "_recently_attempted_identity_keys",
        lambda **_kwargs: (set(), set(), set()),
    )

    summary = AS.run(
        dry_run=True,
        record_run_history=False,
        min_age_hours=0,
        max_candidates=10,
    )

    assert summary["settlement_candidates"] == 0
    preserved = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    assert set(preserved["record_status"]) == {
        "identity_conflict", "superseded_identity",
    }


def test_settlement_uid_gate_blocks_whole_tombstoned_group():
    rows = pd.DataFrame([
        {
            "match_uid": "match_shared", "actual_winner": None,
            "record_status": "pending", "p1": "Player A", "p2": "Player B",
            "match_date": "2026-07-14", "tournament": "Lincoln",
            "round": "R32", "surface": "Hard",
        },
        {
            "match_uid": "match_shared", "actual_winner": None,
            "record_status": "identity_conflict", "p1": "Player A",
            "p2": "Player B", "match_date": "2026-07-14",
            "tournament": "Lincoln", "round": "R32", "surface": "Hard",
        },
    ])

    gated, counts = AS._settlement_uid_gate(rows, rows.copy())

    assert gated.empty
    assert counts["identity_terminal_uid"] == 1


def test_settlement_attempts_compatible_duplicate_uid_once():
    pending = pd.DataFrame([
        {"match_uid": "match_shared", "p1": "Player A", "p2": "Player B"},
        {"match_uid": "match_shared", "p1": "Player A", "p2": "Player B"},
        {"match_uid": "", "p1": "Legacy A", "p2": "Legacy B"},
        {"match_uid": "", "p1": "Legacy C", "p2": "Legacy D"},
    ])

    deduped = AS._dedupe_pending_match_uids(pending)

    assert len(deduped) == 3
    assert (deduped["match_uid"] == "match_shared").sum() == 1
    assert (deduped["match_uid"] == "").sum() == 2


def test_settlement_gate_counts_only_uids_in_pending_backlog():
    rows = pd.DataFrame([
        {
            "match_uid": "already_done", "actual_winner": 1,
            "record_status": "settled",
        },
        {
            "match_uid": "still_pending", "actual_winner": None,
            "record_status": "pending",
        },
    ])
    pending = rows.loc[[1]].copy()

    gated, counts = AS._settlement_uid_gate(rows, pending)

    assert gated["match_uid"].tolist() == ["still_pending"]
    assert counts["already_settled_uid"] == 0


def test_compatible_pending_group_propagates_one_result_to_all_row_indices():
    rows = pd.DataFrame([
        {
            "match_uid": "match_shared", "actual_winner": None,
            "record_status": "pending", "p1": "Player A", "p2": "Player B",
            "match_date": "2026-07-14", "identity_event_key": "lincoln",
            "round": "R32", "surface": "Hard",
        },
        {
            "match_uid": "match_shared", "actual_winner": None,
            "record_status": "pending", "p1": "Player A", "p2": "Player B",
            "match_date": "2026-07-14", "identity_event_key": "lincoln",
            "round": "R32", "surface": "Hard",
        },
    ])

    assert AS._compatible_pending_group_indices(rows, 0) == [0, 1]


def test_seventeen_volatile_suffix_transitions_never_cross_attach_snapshots(
    tmp_path, monkeypatch,
):
    """Reproduce the production mechanism across the observed 17-row cohort.

    Before this contract, the second call fuzzy-updated the first operational
    row and attached a snapshot built for another match UID. The transition now
    retires the pre-contract UID and creates one explicit canonical alias row.
    """
    _configure_paths(monkeypatch, tmp_path)
    snapshot_owner = {}
    legacy_rows = []

    for index in range(17):
        p1 = f"Player A {index}"
        p2 = f"Player B {index}"
        old_event = f"Challenger - Lincoln ({index + 1})"
        old_uid = _legacy_raw_event_uid(
            p1, p2, "2026-07-14", old_event, "R32", "Hard",
        )
        old_run = f"run_old_{index}"
        old_snapshot = build_feature_snapshot_id(old_uid, old_run, p1, p2)
        legacy_row = {column: "" for column in PL.COLUMNS}
        legacy_row.update({
            "logged_at": f"2026-07-14T09:{index:02d}:00",
            "match_date": "2026-07-14",
            "tournament": "Lincoln",
            "surface": "Hard",
            "level": "C",
            "round": "R32",
            "p1": p1,
            "p2": p2,
            "model_p1_prob": 0.61,
            "model_p2_prob": 0.39,
            "model_version": "1.2.3",
            "nn_model_version": "1.2.3",
            "features_complete": True,
            "record_status": "pending",
            "run_id": old_run,
            "latest_run_id": old_run,
            "match_uid": old_uid,
            "feature_snapshot_id": old_snapshot,
            "latest_feature_snapshot_id": old_snapshot,
        })
        legacy_rows.append(legacy_row)
        snapshot_owner[old_snapshot] = old_uid

    pd.DataFrame(legacy_rows, columns=PL.COLUMNS).to_csv(PL.LOG_PATH, index=False)

    for index in range(17):
        p1 = f"Player A {index}"
        p2 = f"Player B {index}"
        new_event = f"Challenger - Lincoln ({99 - index})"
        new_uid = build_match_uid(
            p1, p2, "2026-07-14", new_event, "R32", "Hard",
        )
        new_run = f"run_new_{index}"
        new_snapshot = build_feature_snapshot_id(new_uid, new_run, p1, p2)
        action, _, _ = _log(
            p1=p1,
            p2=p2,
            event_title=new_event,
            run_id=new_run,
            match_uid=new_uid,
            feature_snapshot_id=new_snapshot,
        )
        assert action == "created_alias"
        snapshot_owner[new_snapshot] = new_uid

    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    assert len(rows) == 34
    assert (rows["record_status"] == "superseded_identity").sum() == 17
    assert (rows["identity_status"] == "canonical_alias").sum() == 17
    assert all(
        snapshot_owner[row.latest_feature_snapshot_id] == row.match_uid
        for row in rows.itertuples()
    )

    upgraded = PL.upgrade_prediction_log(Path(PL.LOG_PATH), write=False)
    assert (upgraded["record_status"] == "superseded_identity").sum() == 17


@pytest.mark.parametrize(
    ("new_date", "new_round", "expected_field"),
    [
        ("2026-07-15", "R32", "match_date"),
        ("2026-07-14", "R16", "round"),
    ],
)
def test_date_and_round_shifts_are_explicit_nondecision_conflicts(
    tmp_path, monkeypatch, new_date, new_round, expected_field,
):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    first_action, first_uid, first_snapshot = _log(
        p1=p1, p2=p2, run_id="run_opening",
    )
    assert first_action == "created"

    action, shifted_uid, shifted_snapshot = _log(
        p1=p1,
        p2=p2,
        match_date=new_date,
        round_code=new_round,
        run_id="run_shifted",
    )
    assert action == "identity_conflict"
    assert shifted_uid != first_uid

    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    conflict = rows.loc[rows["match_uid"] == shifted_uid].iloc[0]
    assert conflict["record_status"] == "identity_conflict"
    assert conflict["features_complete"].lower() == "false"
    assert expected_field in conflict["identity_conflict_fields"].split(",")
    assert conflict["identity_related_match_uid"] == first_uid
    assert conflict["latest_feature_snapshot_id"] == shifted_snapshot
    related = rows.loc[rows["match_uid"] == first_uid].iloc[0]
    assert related["latest_feature_snapshot_id"] == first_snapshot
    assert related["record_status"] == "identity_conflict"
    assert related["identity_status"] == "conflict_related"
    assert related["features_complete"].lower() == "false"

    upgraded = PL.upgrade_prediction_log(Path(PL.LOG_PATH), write=False)
    preserved = upgraded.loc[upgraded["match_uid"] == shifted_uid].iloc[0]
    assert preserved["record_status"] == "identity_conflict"
    assert bool(preserved["features_complete"]) is False


def test_non_suffix_event_change_is_not_auto_aliased(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    _, first_uid, _ = _log(
        p1=p1,
        p2=p2,
        event_title="Challenger - Lincoln (8)",
        run_id="run_lincoln",
    )
    action, changed_uid, _ = _log(
        p1=p1,
        p2=p2,
        event_title="Challenger - Lexington (8)",
        # Keep display metadata constant to prove the stored identity event key
        # itself prevents an unsafe fuzzy alias.
        tournament="Lincoln",
        run_id="run_lexington",
    )

    assert action == "identity_conflict"
    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    conflict = rows.loc[rows["match_uid"] == changed_uid].iloc[0]
    assert conflict["identity_related_match_uid"] == first_uid
    assert "event_key" in conflict["identity_conflict_fields"].split(",")
    related = rows.loc[rows["match_uid"] == first_uid].iloc[0]
    assert related["record_status"] == "identity_conflict"


def test_blank_round_incomplete_row_can_enrich_to_explicit_complete_round(
    tmp_path, monkeypatch,
):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    first_action, incomplete_uid, _ = _log(
        p1=p1,
        p2=p2,
        round_code="",
        run_id="run_round_unknown",
        features_complete=False,
    )
    assert first_action == "created"

    action, complete_uid, _ = _log(
        p1=p1,
        p2=p2,
        round_code="R32",
        run_id="run_round_known",
        features_complete=True,
    )

    assert action == "created_alias"
    assert complete_uid != incomplete_uid
    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    old = rows.loc[rows["match_uid"] == incomplete_uid].iloc[0]
    new = rows.loc[rows["match_uid"] == complete_uid].iloc[0]
    assert old["record_status"] == "superseded_identity"
    assert old["identity_status"] == "superseded_alias"
    assert new["record_status"] == "pending"
    assert new["identity_status"] == "canonical_alias"
    assert new["record_note"].startswith("canonical_round_enrichment_from:")
    assert new["features_complete"].lower() == "true"


def test_blank_round_identity_tombstone_cannot_be_cleared_by_enrichment(
    tmp_path, monkeypatch,
):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    _, conflicted_uid, _ = _log(
        p1=p1,
        p2=p2,
        round_code="",
        run_id="run_conflicted_unknown_round",
        features_complete=False,
    )
    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    rows.loc[0, "record_status"] = "identity_conflict"
    rows.loc[0, "identity_status"] = "conflict"
    rows.loc[0, "identity_conflict_fields"] = "round"
    rows.to_csv(PL.LOG_PATH, index=False)

    action, new_uid, _ = _log(
        p1=p1,
        p2=p2,
        round_code="R32",
        run_id="run_conflicted_known_round",
        features_complete=True,
    )

    assert action == "identity_conflict"
    preserved = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    assert set(
        preserved.loc[
            preserved["match_uid"].isin({conflicted_uid, new_uid}),
            "record_status",
        ]
    ) == {"identity_conflict"}


def test_display_tournament_enrichment_does_not_change_event_identity(
    tmp_path, monkeypatch,
):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    _, uid, _ = _log(
        p1=p1,
        p2=p2,
        tournament="",
        event_title="Challenger - Lincoln (8)",
        run_id="run_display_blank",
    )

    action, same_uid, _ = _log(
        p1=p1,
        p2=p2,
        tournament="Lincoln",
        event_title="Challenger - Lincoln (7)",
        run_id="run_display_known",
    )

    assert same_uid == uid
    assert action == "updated"
    row = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False).iloc[0]
    assert row["record_status"] == "pending"
    assert row["identity_event_key"] == "challenger - lincoln"


def test_stale_uid_with_changed_round_fails_before_lineage_write(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    stale_uid = build_match_uid(
        p1, p2, "2026-07-14", "Challenger - Lincoln", "R32", "Hard",
    )
    stale_snapshot = build_feature_snapshot_id(
        stale_uid, "run_stale_round", p1, p2,
    )

    with pytest.raises(PL.LiveMatchIdentityError, match="does not match canonical"):
        _log(
            p1=p1,
            p2=p2,
            round_code="R16",
            run_id="run_stale_round",
            match_uid=stale_uid,
            feature_snapshot_id=stale_snapshot,
        )

    assert not Path(PL.LOG_PATH).exists()
    assert not Path(PL.SNAPSHOT_LOG_PATH).exists()


def test_compatible_duplicate_operational_rows_are_reconciled(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    _, uid, opening_snapshot = _log(
        p1=p1, p2=p2, run_id="run_opening",
    )
    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    duplicate = rows.iloc[0].copy()
    duplicate_run = "run_duplicate"
    duplicate_snapshot = build_feature_snapshot_id(uid, duplicate_run, p1, p2)
    duplicate["logged_at"] = (
        pd.to_datetime(rows.iloc[0]["logged_at"]) + pd.Timedelta(minutes=1)
    ).isoformat()
    duplicate["run_id"] = duplicate_run
    duplicate["latest_run_id"] = duplicate_run
    duplicate["feature_snapshot_id"] = duplicate_snapshot
    duplicate["latest_feature_snapshot_id"] = duplicate_snapshot
    duplicate["prediction_uid"] = ""
    duplicate["latest_prediction_uid"] = ""
    pd.concat([rows, pd.DataFrame([duplicate])], ignore_index=True).to_csv(
        PL.LOG_PATH, index=False,
    )

    action, _, latest_snapshot = _log(
        p1=p1, p2=p2, run_id="run_reconcile",
    )

    assert action == "updated"
    reconciled = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    assert len(reconciled) == 1
    assert reconciled.iloc[0]["feature_snapshot_id"] == opening_snapshot
    assert reconciled.iloc[0]["latest_feature_snapshot_id"] == latest_snapshot
    assert reconciled.iloc[0]["record_note"] == (
        "reconciled_1_duplicate_operational_rows"
    )


def test_duplicate_reconciliation_preserves_identity_tombstone(
    tmp_path, monkeypatch,
):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    _, uid, _ = _log(p1=p1, p2=p2, run_id="run_opening")
    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    conflict = rows.iloc[0].copy()
    conflict["features_complete"] = "False"
    conflict["record_status"] = "identity_conflict"
    conflict["identity_status"] = "conflict"
    conflict["identity_related_match_uid"] = "match_related"
    conflict["identity_conflict_fields"] = "round"
    conflict["record_note"] = "match_identity_conflict:round"
    conflict["logged_at"] = "2026-07-14T11:00:00Z"
    pd.concat([rows, pd.DataFrame([conflict])], ignore_index=True).to_csv(
        PL.LOG_PATH, index=False,
    )
    conflict_uids: set[str] = set()

    action = PL.log_prediction(
        p1=p1,
        p2=p2,
        tournament="Lincoln",
        surface="Hard",
        level="C",
        round_code="R32",
        match_date="2026-07-14",
        run_id="run_refresh",
        match_uid=uid,
        feature_snapshot_id=build_feature_snapshot_id(
            uid, "run_refresh", p1, p2
        ),
        identity_event_key=canonicalize_live_event_key(
            "Challenger - Lincoln (12)"
        ),
        model_p1_prob=0.62,
        model_p2_prob=0.38,
        market_p1_prob=None,
        market_p2_prob=None,
        model_version="1.2.3",
        nn_model_version="1.2.3",
        features_complete=True,
        identity_conflict_uids_out=conflict_uids,
    )

    assert action == "identity_conflict"
    assert conflict_uids == {uid, "match_related"}
    repaired = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    assert len(repaired) == 1
    assert repaired.iloc[0]["record_status"] == "identity_conflict"
    assert repaired.iloc[0]["identity_status"] == "conflict"
    assert repaired.iloc[0]["features_complete"].lower() == "false"


def test_settled_match_uid_cannot_be_reopened(tmp_path, monkeypatch):
    _configure_paths(monkeypatch, tmp_path)
    p1, p2 = "Player One", "Player Two"
    _, uid, _ = _log(p1=p1, p2=p2, run_id="run_settled")
    rows = pd.read_csv(PL.LOG_PATH, dtype=str, keep_default_na=False)
    rows.loc[0, "actual_winner"] = "1"
    rows.loc[0, "record_status"] = "settled"
    rows.to_csv(PL.LOG_PATH, index=False)
    snapshot_count = len(pd.read_csv(PL.SNAPSHOT_LOG_PATH))

    with pytest.raises(PL.LiveMatchIdentityError, match="already settled"):
        _log(
            p1=p1,
            p2=p2,
            run_id="run_reopen",
            match_uid=uid,
        )

    assert len(pd.read_csv(PL.LOG_PATH)) == 1
    assert len(pd.read_csv(PL.SNAPSHOT_LOG_PATH)) == snapshot_count
