from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from operations.pending_identity_remediation import (  # noqa: E402
    DECISION_SCHEMA_VERSION,
    DUPLICATE_OF,
    REMAP_MATCH_UID,
    RETAIN_DISTINCT,
    RemediationConflict,
    RemediationPaths,
    apply_plan,
    build_apply_plan,
    build_decision_template,
    build_review_report,
    load_registry,
    validate_decisions,
    write_review_report,
)


def _bet(**updates):
    row = {
        "bet_id": "b_orphan",
        "status": "pending",
        "match": "Player One vs Player Two",
        "match_uid": "match_old",
        "bet_on": "Player One",
        "match_date": "2026-07-10",
        "stake": "10",
        "feature_snapshot_id": "feat_old",
        "run_id": "run_old",
        "timestamp": "2026-07-09T01:00:00+00:00",
    }
    row.update(updates)
    return row


def _prediction(**updates):
    row = {
        "match_uid": "match_canonical",
        "p1": "Player One",
        "p2": "Player Two",
        "match_date": "2026-07-10",
        "actual_winner": "1",
        "feature_snapshot_id": "feat_old",
        "run_id": "run_old",
    }
    row.update(updates)
    return row


def _lineage(**updates):
    row = {
        "match_uid": "match_old",
        "p1": "Player One",
        "p2": "Player Two",
        "match_date": "2026-07-10",
        "feature_snapshot_id": "feat_old",
        "run_id": "run_old",
    }
    row.update(updates)
    return row


def _write_sources(root: Path) -> RemediationPaths:
    prod = root / "production"
    logs = prod / "logs"
    logs.mkdir(parents=True)
    pd.DataFrame(
        [
            _bet(),
            _bet(
                bet_id="b_canonical",
                match_uid="match_canonical",
                feature_snapshot_id="feat_new",
                run_id="run_new",
                timestamp="2026-07-09T02:00:00+00:00",
                stake="12",
            ),
        ]
    ).to_csv(logs / "all_bets.csv", index=False)
    pd.DataFrame([_prediction()]).to_csv(prod / "prediction_log.csv", index=False)
    pd.DataFrame([_lineage()]).to_csv(prod / "prediction_snapshots.csv", index=False)
    pd.DataFrame(
        [{key: value for key, value in _lineage().items() if key != "feature_snapshot_id"}]
    ).to_csv(prod / "odds_history.csv", index=False)
    return RemediationPaths.from_prod_dir(prod)


def _report_and_cases(tmp_path: Path):
    paths = _write_sources(tmp_path)
    report = build_review_report(paths)
    remap = next(case for case in report["cases"] if case["case_type"] == "match_uid_remap")
    duplicate = next(case for case in report["cases"] if case["case_type"] == "duplicate_intent")
    return paths, report, remap, duplicate


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _payload_hash(value: dict) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _redigest(payload: dict, digest_field: str) -> dict:
    payload = dict(payload)
    payload.pop(digest_field, None)
    return {**payload, digest_field: _payload_hash(payload)}


def _write_decision(
    tmp_path: Path,
    report: dict,
    case: dict,
    *,
    decision: str = REMAP_MATCH_UID,
    target_match_uid: str = "match_canonical",
    evidence_hash: str | None = None,
    envelope_hash: str | None = None,
    evidence_kind: str | None = None,
    canonical_bet_id: str = "b_canonical",
    duplicate_bet_ids: list[str] | None = None,
    supersedes_decision_id: str = "",
    binding_override: dict | None = None,
    artifact_override: dict | str | None = None,
    extra: dict | None = None,
) -> Path:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    artifact = evidence_dir / "book-record.json"
    evidence_kind = evidence_kind or (
        "bookmaker_event_record"
        if case["case_type"] == "match_uid_remap"
        else "operator_intent_record"
    )
    if decision == REMAP_MATCH_UID:
        resolution = {
            "original_match_uid": case["identity"]["original_match_uid"],
            "target_match_uid": target_match_uid,
            "bet_id": case["identity"]["bet_id"],
            "bet_row_sha256": case["identity"]["bet_row_sha256"],
        }
        assertion = "same_external_match_identity"
        binding = {
            "case_type": case["case_type"],
            "subject_key": case["subject_key"],
            "case_id": case["case_id"],
            "bet_id": case["identity"]["bet_id"],
            "bet_row_sha256": case["identity"]["bet_row_sha256"],
            "original_match_uid": case["identity"]["original_match_uid"],
            "target_match_uid": target_match_uid,
            "match_pair": case["identity"]["match_pair"],
            "match_date": case["identity"]["match_date"],
            "supersedes_decision_id": supersedes_decision_id,
        }
    else:
        members = [member["bet_id"] for member in case["identity"]["members"]]
        if decision == DUPLICATE_OF:
            if duplicate_bet_ids is None:
                duplicate_bet_ids = sorted(set(members) - {canonical_bet_id})
            resolution = {
                "canonical_bet_id": canonical_bet_id,
                "duplicate_bet_ids": sorted(duplicate_bet_ids),
                "pending_identity_key": case["identity"]["pending_identity_key"],
            }
            assertion = "same_operator_decision_identity"
        else:
            resolution = {
                "distinct_bet_ids": sorted(members),
                "pending_identity_key": case["identity"]["pending_identity_key"],
            }
            assertion = "distinct_operator_decision_identities"
        binding = {
            "case_type": case["case_type"],
            "subject_key": case["subject_key"],
            "case_id": case["case_id"],
            "pending_identity_key": case["identity"]["pending_identity_key"],
            "members": [
                {
                    "bet_id": member["bet_id"],
                    "bet_row_sha256": member["bet_row_sha256"],
                }
                for member in case["identity"]["members"]
            ],
            "decision": decision,
            "resolution": resolution,
            "supersedes_decision_id": supersedes_decision_id,
        }
    source_uri = "https://example.test/events/book-123"
    source_system = "bovada" if case["case_type"] == "match_uid_remap" else "pipeline-audit"
    external_record_id = "book-123" if source_system == "bovada" else "intent-run-123"
    observed_at = "2026-07-09T00:55:00+00:00"
    claim = "Stable external record identity binds this exact reviewed subject."
    if artifact_override is not None:
        if isinstance(artifact_override, dict):
            artifact_body = json.dumps(
                artifact_override, indent=2, sort_keys=True
            ) + "\n"
        else:
            artifact_body = artifact_override
    elif case["case_type"] == "match_uid_remap" and evidence_kind in {
        "bookmaker_event_record",
        "identity_capture_record",
    }:
        original_source = case["uid_binding_sources"]["original"][0]
        target_source = case["uid_binding_sources"]["targets"].get(
            target_match_uid,
            [
                {
                    "operational_source": "predictions",
                    "source_row": 2,
                    "source_row_sha256": "0" * 64,
                }
            ],
        )[0]
        raw_artifact = {
            "identity_capture_schema_version": "1.0.0",
            "record_type": "bookmaker_event_identity_capture",
            "source_system": source_system,
            "external_record_id": external_record_id,
            "source_uri": source_uri,
            "observed_at_utc": observed_at,
            "match_pair": case["identity"]["match_pair"],
            "match_date": case["identity"]["match_date"],
            "uid_bindings": [
                {
                    "role": "original",
                    "match_uid": case["identity"]["original_match_uid"],
                    **original_source,
                },
                {
                    "role": "target",
                    "match_uid": target_match_uid,
                    **target_source,
                },
            ],
        }
        artifact_body = json.dumps(raw_artifact, indent=2, sort_keys=True) + "\n"
    elif (
        case["case_type"] == "duplicate_intent"
        and evidence_kind == "operator_intent_record"
    ):
        member_rows = case["identity"]["members"]
        shared_intent = "intent-shared-123"
        raw_artifact = {
            "intent_record_schema_version": "1.0.0",
            "record_type": "operator_intent_record",
            "source_system": source_system,
            "external_record_id": external_record_id,
            "source_uri": source_uri,
            "observed_at_utc": observed_at,
            "subject_key": case["subject_key"],
            "pending_identity_key": case["identity"]["pending_identity_key"],
            "disposition": (
                "same_intent" if decision == DUPLICATE_OF else "distinct_intents"
            ),
            "members": [
                {
                    "bet_id": member["bet_id"],
                    "bet_row_sha256": member["bet_row_sha256"],
                    "intent_id": (
                        shared_intent
                        if decision == DUPLICATE_OF
                        else f"intent-{member['bet_id']}"
                    ),
                }
                for member in member_rows
            ],
            "resolution": resolution,
        }
        artifact_body = json.dumps(raw_artifact, indent=2, sort_keys=True) + "\n"
    else:
        artifact_body = '{"supplemental_record":"not_sufficient_alone"}\n'
    artifact.write_text(artifact_body, encoding="utf-8")
    declared_artifact_hash = evidence_hash or _sha(artifact)
    envelope = {
        "evidence_envelope_schema_version": "1.0.0",
        "evidence_id": "book-event-123",
        "evidence_kind": evidence_kind,
        "source": {
            "source_uri": source_uri,
            "source_system": source_system,
            "external_record_id": external_record_id,
            "observed_at_utc": observed_at,
            "artifact_sha256": declared_artifact_hash,
        },
        "claim": claim,
        "assertion": assertion,
        "binding": binding_override or binding,
    }
    envelope_path = evidence_dir / "book-record.envelope.json"
    envelope_path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload = {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "report_id": report["report_id"],
        "report_digest": report["report_digest"],
        "reviewer": "reviewer@example.com",
        "reviewed_at_utc": "2026-07-14T20:00:00+00:00",
        "evidence": [
            {
                "evidence_id": "book-event-123",
                "evidence_kind": evidence_kind,
                "source_uri": source_uri,
                "source_system": source_system,
                "external_record_id": external_record_id,
                "observed_at_utc": observed_at,
                "artifact_path": "evidence/book-record.json",
                "artifact_sha256": declared_artifact_hash,
                "envelope_path": "evidence/book-record.envelope.json",
                "envelope_sha256": envelope_hash or _sha(envelope_path),
                "claim": claim,
            }
        ],
        "decisions": [
            {
                "case_id": case["case_id"],
                "decision": decision,
                "reason": "The retained raw bookmaker record proves one event identity.",
                "evidence_ids": ["book-event-123"],
                "supersedes_decision_id": supersedes_decision_id,
            }
        ],
    }
    if decision == REMAP_MATCH_UID:
        payload["decisions"][0]["target_match_uid"] = target_match_uid
    elif decision == DUPLICATE_OF:
        payload["decisions"][0]["canonical_bet_id"] = canonical_bet_id
        payload["decisions"][0]["duplicate_bet_ids"] = sorted(duplicate_bet_ids or [])
    if extra:
        payload.update(extra)
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_plan(path: Path, plan: dict) -> None:
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_structured_artifact(decision_path: Path, mutate) -> None:
    payload = json.loads(decision_path.read_text())
    evidence = payload["evidence"][0]
    artifact_path = decision_path.parent / evidence["artifact_path"]
    artifact = json.loads(artifact_path.read_text())
    mutate(artifact)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence["artifact_sha256"] = _sha(artifact_path)
    envelope_path = decision_path.parent / evidence["envelope_path"]
    envelope = json.loads(envelope_path.read_text())
    envelope["source"]["artifact_sha256"] = evidence["artifact_sha256"]
    envelope_path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence["envelope_sha256"] = _sha(envelope_path)
    decision_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_report_is_deterministic_and_candidates_are_never_authority(tmp_path):
    _, first, remap, duplicate = _report_and_cases(tmp_path)
    second = build_review_report(RemediationPaths.from_prod_dir(tmp_path / "production"))

    assert first == second
    assert first["automatic_approvals"] == 0
    assert first["summary"] == {
        "pending_rows": 2,
        "orphan_uid_rows": 1,
        "orphan_uid_stake": 10.0,
        "ambiguous_identity_uid_rows": 0,
        "duplicate_labeled_rows": 2,
        "duplicate_identity_groups": 1,
        "duplicate_labeled_stake": 22.0,
        "outcome_pending_not_identity_cases": 0,
        "cases": 2,
        "automatically_proven_cases": 0,
    }
    assert remap["candidate_target_match_uids"] == ["match_canonical"]
    assert remap["lineage_signals"]["same_run_feature_snapshot_target_match_uids"] == [
        "match_canonical"
    ]
    assert remap["uid_binding_sources"]["original"][0] == {
        "operational_source": "bets",
        "source_row": 2,
        "source_row_sha256": remap["identity"]["bet_row_sha256"],
    }
    target_rows = remap["uid_binding_sources"]["targets"]["match_canonical"]
    assert len(target_rows) == 1
    assert target_rows[0]["operational_source"] == "predictions"
    assert len(target_rows[0]["source_row_sha256"]) == 64
    assert remap["candidate_is_authority"] is False
    assert duplicate["candidate_is_authority"] is False
    assert "not evidence" in duplicate["candidate_basis"]


def test_semantically_reused_uid_is_an_identity_case_not_an_outcome_only_row(tmp_path):
    paths = _write_sources(tmp_path)
    bets = pd.read_csv(paths.bets, dtype=str, keep_default_na=False)
    bets.loc[bets["bet_id"].eq("b_orphan"), "match_uid"] = "match_reused"
    bets.to_csv(paths.bets, index=False)
    predictions = pd.read_csv(paths.predictions, dtype=str, keep_default_na=False)
    predictions = pd.concat(
        [
            predictions,
            pd.DataFrame(
                [
                    _prediction(match_uid="match_reused"),
                    _prediction(
                        match_uid="match_reused",
                        p1="Different One",
                        p2="Different Two",
                        match_date="2026-07-11",
                    ),
                ]
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(paths.predictions, index=False)

    report = build_review_report(paths)
    remap = next(
        case
        for case in report["cases"]
        if case["case_type"] == "match_uid_remap"
    )
    assert report["summary"]["orphan_uid_rows"] == 0
    assert report["summary"]["ambiguous_identity_uid_rows"] == 1
    assert report["summary"]["outcome_pending_not_identity_cases"] == 0
    assert remap["candidate_target_match_uids"] == ["match_canonical"]
    assert "reused_uid_requires_authoritative_replacement" in remap["blocked_reasons"]


def test_semantically_reused_target_uid_is_rejected_before_decision_review(tmp_path):
    paths = _write_sources(tmp_path)
    predictions = pd.read_csv(paths.predictions, dtype=str, keep_default_na=False)
    predictions = pd.concat(
        [
            predictions,
            pd.DataFrame(
                [
                    _prediction(
                        p1="Different One",
                        p2="Different Two",
                        match_date="2026-07-11",
                        feature_snapshot_id="feat_other",
                        run_id="run_other",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(paths.predictions, index=False)
    report = build_review_report(paths)
    remap = next(
        case
        for case in report["cases"]
        if case["case_type"] == "match_uid_remap"
        and case["identity"]["bet_id"] == "b_orphan"
    )
    assert remap["candidate_target_match_uids"] == []
    rejected = remap["rejected_semantically_ambiguous_target_match_uids"]
    assert rejected["match_canonical"]["semantically_unique"] is False

    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    with pytest.raises(RemediationConflict, match="not an exact report candidate"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


@pytest.mark.parametrize("invalid_kind", ["blank", "duplicate"])
def test_blank_or_duplicate_bet_ids_fail_before_case_construction(tmp_path, invalid_kind):
    paths = _write_sources(tmp_path)
    bets = pd.read_csv(paths.bets, dtype=str, keep_default_na=False)
    if invalid_kind == "blank":
        bets.loc[0, "bet_id"] = ""
        expected = "blank bet_id"
    else:
        bets.loc[1, "bet_id"] = bets.loc[0, "bet_id"]
        expected = "duplicate bet_id"
    bets.to_csv(paths.bets, index=False)
    with pytest.raises(ValueError, match=expected):
        build_review_report(paths)


def test_forged_self_digested_report_is_rebuilt_and_rejected_at_plan(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    forged = json.loads(json.dumps(report))
    forged_case = next(
        case for case in forged["cases"] if case["case_id"] == remap["case_id"]
    )
    forged_case["candidate_target_match_uids"] = ["match_forged"]
    forged_case["candidate_target_identity_contracts"] = {
        "match_forged": {
            "prediction_rows": 1,
            "all_rows_identity_complete": True,
            "identity_count": 1,
            "identities": [
                {
                    "match_pair": remap["identity"]["match_pair"],
                    "match_date": remap["identity"]["match_date"],
                }
            ],
            "semantically_unique": True,
        }
    }
    forged = _redigest(forged, "report_digest")
    report_path = tmp_path / "forged-review.json"
    write_review_report(forged, report_path)
    forged_remap = next(
        case for case in forged["cases"] if case["case_id"] == remap["case_id"]
    )
    decision_path = _write_decision(
        tmp_path, forged, forged_remap, target_match_uid="match_forged"
    )
    with pytest.raises(
        RemediationConflict, match="differs from deterministic source regeneration"
    ):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_apply_independently_rebuilds_report_even_if_plan_is_redigested(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    registry_path = tmp_path / "registry.json"
    plan = build_apply_plan(
        report_path=report_path,
        decision_path=decision_path,
        registry_path=registry_path,
    )

    forged = json.loads(report_path.read_text())
    forged["cases"][0]["candidate_basis"] = "attacker-controlled candidate authority"
    forged = _redigest(forged, "report_digest")
    report_path.write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan["inputs"]["report"] = {
        **plan["inputs"]["report"],
        "sha256": _sha(report_path),
        "byte_size": report_path.stat().st_size,
    }
    plan = _redigest(plan, "plan_digest")
    plan_path = tmp_path / "forged-plan.json"
    _write_plan(plan_path, plan)
    with pytest.raises(
        RemediationConflict, match="differs from deterministic source regeneration"
    ):
        apply_plan(plan_path=plan_path, expected_plan_digest=plan["plan_digest"])


def test_report_requires_exactly_four_named_distinct_sources(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    for mode in ("missing", "duplicate"):
        forged = json.loads(json.dumps(report))
        if mode == "missing":
            forged["inputs"].pop("odds_history")
            expected = "exactly four named source descriptors"
        else:
            forged["inputs"]["odds_history"] = dict(forged["inputs"]["bets"])
            expected = "source paths must be distinct"
        forged = _redigest(forged, "report_digest")
        report_path = tmp_path / f"{mode}-review.json"
        write_review_report(forged, report_path)
        decision_path = _write_decision(tmp_path, forged, remap)
        with pytest.raises(RemediationConflict, match=expected):
            build_apply_plan(
                report_path=report_path,
                decision_path=decision_path,
                registry_path=tmp_path / "registry.json",
            )


def test_report_and_template_refuse_different_overwrite(tmp_path):
    _, report, _, _ = _report_and_cases(tmp_path)
    output = tmp_path / "review.json"
    write_review_report(report, output)
    original = output.read_bytes()
    write_review_report(report, output)
    assert output.read_bytes() == original

    changed = dict(report)
    changed["report_id"] = "changed"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_review_report(changed, output)

    template = build_decision_template(report)
    assert template["reviewer"] == ""
    assert {row["decision"] for row in template["decisions"]} == {"defer"}


def test_valid_remap_plan_applies_only_to_immutable_registry(tmp_path):
    paths, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    registry_path = tmp_path / "pending_identity_registry.json"
    before_sources = {path: _sha(path) for path in paths.as_dict().values()}

    plan = build_apply_plan(
        report_path=report_path,
        decision_path=decision_path,
        registry_path=registry_path,
    )
    assert plan["summary"]["new_registry_entries"] == 1
    assert plan["canonical_bet_log_mutation"] is False
    assert plan["settlement_mutation"] is False
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)

    result = apply_plan(plan_path=plan_path, expected_plan_digest=plan["plan_digest"])
    assert result["status"] == "applied"
    assert result["registry_generation"] == 1
    registry = load_registry(registry_path)
    assert registry["entries"][0]["decision"] == REMAP_MATCH_UID
    assert registry["entries"][0]["resolution"]["target_match_uid"] == "match_canonical"
    parsed = registry["entries"][0]["evidence"][0]["parsed_artifact"]
    assert parsed["record_type"] == "bookmaker_event_identity_capture"
    assert {binding["role"] for binding in parsed["uid_bindings"]} == {
        "original",
        "target",
    }
    assert before_sources == {path: _sha(path) for path in paths.as_dict().values()}


def test_rebuilt_plan_turns_exact_registry_replay_into_verified_noop(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    registry_path = tmp_path / "registry.json"
    first = build_apply_plan(
        report_path=report_path, decision_path=decision_path, registry_path=registry_path
    )
    first_path = tmp_path / "first-plan.json"
    _write_plan(first_path, first)
    apply_plan(plan_path=first_path, expected_plan_digest=first["plan_digest"])

    replay = build_apply_plan(
        report_path=report_path, decision_path=decision_path, registry_path=registry_path
    )
    assert replay["summary"]["new_registry_entries"] == 0
    assert replay["summary"]["verified_replays"] == 1
    replay_path = tmp_path / "replay-plan.json"
    _write_plan(replay_path, replay)
    result = apply_plan(
        plan_path=replay_path, expected_plan_digest=replay["plan_digest"]
    )
    assert result == {
        "status": "verified_noop",
        "plan_digest": replay["plan_digest"],
        "registry_generation": 1,
    }


def test_stale_operational_input_fails_before_plan_or_apply(tmp_path):
    paths, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    registry_path = tmp_path / "registry.json"
    plan = build_apply_plan(
        report_path=report_path, decision_path=decision_path, registry_path=registry_path
    )
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)

    with paths.bets.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RemediationConflict, match="stale report input hash: bets"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=registry_path,
        )
    with pytest.raises(RemediationConflict, match="stale report input hash: bets"):
        apply_plan(plan_path=plan_path, expected_plan_digest=plan["plan_digest"])


def test_stale_decision_or_registry_hash_fails_apply(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    registry_path = tmp_path / "registry.json"
    plan = build_apply_plan(
        report_path=report_path, decision_path=decision_path, registry_path=registry_path
    )
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)

    original_decisions = decision_path.read_text()
    decision_path.write_text(original_decisions + "\n", encoding="utf-8")
    with pytest.raises(RemediationConflict, match="stale apply input: decisions"):
        apply_plan(plan_path=plan_path, expected_plan_digest=plan["plan_digest"])

    decision_path.write_text(original_decisions, encoding="utf-8")
    registry_path.write_text(
        json.dumps(
            {
                "registry_schema_version": "1.0.0",
                "registry_generation": 0,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RemediationConflict, match="stale apply input: registry"):
        apply_plan(plan_path=plan_path, expected_plan_digest=plan["plan_digest"])


def test_evidence_checksum_and_required_review_fields_fail_closed(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap, evidence_hash="0" * 64)
    with pytest.raises(RemediationConflict, match="evidence artifact hash mismatch"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )

    decision_path = _write_decision(tmp_path, report, remap)
    payload = json.loads(decision_path.read_text())
    payload["reviewer"] = ""
    decision_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer is required"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_typed_evidence_envelope_must_bind_exact_remap_subject(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    wrong_binding = {
        "case_type": "match_uid_remap",
        "subject_key": remap["subject_key"],
        "case_id": remap["case_id"],
        "bet_id": remap["identity"]["bet_id"],
        "bet_row_sha256": remap["identity"]["bet_row_sha256"],
        "original_match_uid": remap["identity"]["original_match_uid"],
        "target_match_uid": "match_someone_else",
        "match_pair": remap["identity"]["match_pair"],
        "match_date": remap["identity"]["match_date"],
        "supersedes_decision_id": "",
    }
    decision_path = _write_decision(
        tmp_path, report, remap, binding_override=wrong_binding
    )
    with pytest.raises(
        RemediationConflict, match="not bound to decision subject"
    ):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_evidence_declaration_must_match_typed_external_record_identity(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    payload = json.loads(decision_path.read_text())
    payload["evidence"][0]["external_record_id"] = "forged-external-record"
    decision_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        RemediationConflict, match="declaration/envelope identity mismatch"
    ):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_hashed_arbitrary_bytes_cannot_substitute_for_typed_envelope(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    payload = json.loads(decision_path.read_text())
    envelope_path = tmp_path / payload["evidence"][0]["envelope_path"]
    envelope_path.write_text('{"claim":"trust me"}\n', encoding="utf-8")
    payload["evidence"][0]["envelope_sha256"] = _sha(envelope_path)
    decision_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence envelope .* invalid fields"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_unrelated_raw_bytes_cannot_authorize_remap(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(
        tmp_path,
        report,
        remap,
        artifact_override="unrelated retained bytes with a valid checksum\n",
    )
    with pytest.raises(ValueError, match="structured evidence artifact .* not valid JSON"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_well_formed_but_unbound_identity_capture_fails_source_row_crosscheck(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)

    def break_target_binding(artifact):
        target = next(
            binding
            for binding in artifact["uid_bindings"]
            if binding["role"] == "target"
        )
        target["source_row_sha256"] = "f" * 64

    _rewrite_structured_artifact(decision_path, break_target_binding)
    with pytest.raises(RemediationConflict, match="target source-row hash is not report-bound"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


@pytest.mark.parametrize("supplemental_kind", ["official_match_record", "raw_source_artifact"])
def test_supplemental_opaque_evidence_is_not_sufficient_alone(
    tmp_path, supplemental_kind
):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(
        tmp_path,
        report,
        remap,
        evidence_kind=supplemental_kind,
    )
    with pytest.raises(ValueError, match="requires a structured identity capture"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_name_date_candidate_cannot_be_approved_without_evidence(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    decisions = {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "report_id": report["report_id"],
        "report_digest": report["report_digest"],
        "reviewer": "reviewer@example.com",
        "reviewed_at_utc": "2026-07-14T20:00:00+00:00",
        "evidence": [],
        "decisions": [
            {
                "case_id": remap["case_id"],
                "decision": REMAP_MATCH_UID,
                "target_match_uid": "match_canonical",
                "reason": "Names and date happen to be an exact normalized candidate.",
                "evidence_ids": [],
                "supersedes_decision_id": "",
            }
        ],
    }
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(json.dumps(decisions), encoding="utf-8")
    with pytest.raises(ValueError, match="approved decision requires evidence"):
        validate_decisions(report, decisions, decision_path=decision_path)


def test_conflicting_decisions_for_one_case_fail_closed(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    decision_path = _write_decision(tmp_path, report, remap)
    decisions = json.loads(decision_path.read_text())
    decisions["decisions"].append(dict(decisions["decisions"][0]))
    decision_path.write_text(json.dumps(decisions), encoding="utf-8")
    with pytest.raises(RemediationConflict, match="conflicting duplicate decision"):
        validate_decisions(report, decisions, decision_path=decision_path)


def test_duplicate_disposition_must_partition_case_and_have_evidence(tmp_path):
    _, report, _, duplicate = _report_and_cases(tmp_path)
    decision_path = _write_decision(
        tmp_path,
        report,
        duplicate,
        decision=DUPLICATE_OF,
        evidence_kind="operator_intent_record",
        extra=None,
    )
    decisions = json.loads(decision_path.read_text())
    row = decisions["decisions"][0]
    row.pop("target_match_uid", None)
    row["canonical_bet_id"] = "b_canonical"
    row["duplicate_bet_ids"] = []
    decision_path.write_text(json.dumps(decisions), encoding="utf-8")
    with pytest.raises(RemediationConflict, match="must partition every case member"):
        validate_decisions(report, decisions, decision_path=decision_path)

    row["duplicate_bet_ids"] = ["b_orphan"]
    decision_path.write_text(json.dumps(decisions), encoding="utf-8")
    approved, deferred = validate_decisions(
        report, decisions, decision_path=decision_path
    )
    assert deferred == []
    assert approved[0]["resolution"] == {
        "canonical_bet_id": "b_canonical",
        "duplicate_bet_ids": ["b_orphan"],
        "pending_identity_key": duplicate["identity"]["pending_identity_key"],
    }


def test_operator_intent_artifact_must_bind_exact_duplicate_member_rows(tmp_path):
    _, report, _, duplicate = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(
        tmp_path, report, duplicate, decision=RETAIN_DISTINCT
    )

    def break_member_binding(artifact):
        artifact["members"][0]["bet_row_sha256"] = "e" * 64

    _rewrite_structured_artifact(decision_path, break_member_binding)
    with pytest.raises(RemediationConflict, match="members do not match report rows"):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=tmp_path / "registry.json",
        )


def test_existing_registry_requires_explicit_supersession_for_same_subject(tmp_path):
    _, report, _, duplicate = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(
        tmp_path,
        report,
        duplicate,
        decision=RETAIN_DISTINCT,
        evidence_kind="operator_intent_record",
    )
    registry_path = tmp_path / "registry.json"
    first = build_apply_plan(
        report_path=report_path, decision_path=decision_path, registry_path=registry_path
    )
    first_path = tmp_path / "first.json"
    _write_plan(first_path, first)
    apply_plan(plan_path=first_path, expected_plan_digest=first["plan_digest"])

    _write_decision(
        tmp_path,
        report,
        duplicate,
        decision=DUPLICATE_OF,
        evidence_kind="operator_intent_record",
    )
    with pytest.raises(
        RemediationConflict, match="requires explicit active supersession"
    ):
        build_apply_plan(
            report_path=report_path,
            decision_path=decision_path,
            registry_path=registry_path,
        )


def test_membership_change_keeps_subject_and_requires_hash_bound_supersession(tmp_path):
    paths, report, _, duplicate = _report_and_cases(tmp_path)
    first_report_path = tmp_path / "review-first.json"
    write_review_report(report, first_report_path)
    first_decision_path = _write_decision(
        tmp_path, report, duplicate, decision=RETAIN_DISTINCT
    )
    registry_path = tmp_path / "registry.json"
    first = build_apply_plan(
        report_path=first_report_path,
        decision_path=first_decision_path,
        registry_path=registry_path,
    )
    first_plan_path = tmp_path / "plan-first.json"
    _write_plan(first_plan_path, first)
    apply_plan(
        plan_path=first_plan_path, expected_plan_digest=first["plan_digest"]
    )
    first_entry = load_registry(registry_path)["entries"][0]

    bets = pd.read_csv(paths.bets, dtype=str, keep_default_na=False)
    bets = pd.concat(
        [
            bets,
            pd.DataFrame(
                [
                    _bet(
                        bet_id="b_third",
                        match_uid="match_third_orphan",
                        feature_snapshot_id="feat_third",
                        run_id="run_third",
                        timestamp="2026-07-09T03:00:00+00:00",
                        stake="8",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    bets.to_csv(paths.bets, index=False)
    changed_report = build_review_report(paths)
    changed_duplicate = next(
        case
        for case in changed_report["cases"]
        if case["case_type"] == "duplicate_intent"
    )
    assert changed_duplicate["subject_key"] == duplicate["subject_key"]
    assert changed_duplicate["case_id"] != duplicate["case_id"]
    assert len(changed_duplicate["identity"]["members"]) == 3
    changed_report_path = tmp_path / "review-changed.json"
    write_review_report(changed_report, changed_report_path)

    _write_decision(
        tmp_path, changed_report, changed_duplicate, decision=RETAIN_DISTINCT
    )
    with pytest.raises(
        RemediationConflict, match="requires explicit active supersession"
    ):
        build_apply_plan(
            report_path=changed_report_path,
            decision_path=tmp_path / "decisions.json",
            registry_path=registry_path,
        )

    _write_decision(
        tmp_path,
        changed_report,
        changed_duplicate,
        decision=RETAIN_DISTINCT,
        supersedes_decision_id=first_entry["decision_id"],
    )
    second = build_apply_plan(
        report_path=changed_report_path,
        decision_path=tmp_path / "decisions.json",
        registry_path=registry_path,
    )
    assert second["summary"]["new_registry_entries"] == 1
    second_plan_path = tmp_path / "plan-second.json"
    _write_plan(second_plan_path, second)
    apply_plan(
        plan_path=second_plan_path, expected_plan_digest=second["plan_digest"]
    )
    registry = load_registry(registry_path)
    assert registry["registry_generation"] == 2
    assert registry["entries"][1]["subject_key"] == first_entry["subject_key"]
    assert (
        registry["entries"][1]["supersedes_decision_id"]
        == first_entry["decision_id"]
    )


def test_registry_rejects_a_forked_subject_supersession_chain(tmp_path):
    _, report, _, duplicate = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(
        tmp_path, report, duplicate, decision=RETAIN_DISTINCT
    )
    registry_path = tmp_path / "registry.json"
    plan = build_apply_plan(
        report_path=report_path,
        decision_path=decision_path,
        registry_path=registry_path,
    )
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    apply_plan(plan_path=plan_path, expected_plan_digest=plan["plan_digest"])

    registry = json.loads(registry_path.read_text())
    fork_without_hashes = {
        key: value
        for key, value in registry["entries"][0].items()
        if key not in {"decision_id", "record_sha256"}
    }
    fork_without_hashes["reason"] = "A forged sibling decision tries to fork the active subject."
    fork_without_hashes["supersedes_decision_id"] = ""
    fork_hash = _payload_hash(fork_without_hashes)
    registry["entries"].append(
        {
            **fork_without_hashes,
            "decision_id": "pending_identity_decision_" + fork_hash[:24],
            "record_sha256": fork_hash,
        }
    )
    registry["registry_generation"] = 2
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RemediationConflict, match="supersession chain is invalid"):
        load_registry(registry_path)


def test_expected_and_embedded_plan_digests_are_both_required(tmp_path):
    _, report, remap, _ = _report_and_cases(tmp_path)
    report_path = tmp_path / "review.json"
    write_review_report(report, report_path)
    decision_path = _write_decision(tmp_path, report, remap)
    plan = build_apply_plan(
        report_path=report_path,
        decision_path=decision_path,
        registry_path=tmp_path / "registry.json",
    )
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    with pytest.raises(RemediationConflict, match="expected plan digest mismatch"):
        apply_plan(plan_path=plan_path, expected_plan_digest="0" * 64)

    tampered = dict(plan)
    tampered["summary"] = dict(plan["summary"], deferred_cases=999)
    _write_plan(plan_path, tampered)
    with pytest.raises(RemediationConflict, match="apply plan digest mismatch"):
        apply_plan(plan_path=plan_path, expected_plan_digest=plan["plan_digest"])
