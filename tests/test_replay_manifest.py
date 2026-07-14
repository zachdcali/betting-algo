import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

from evaluation import replay_manifest
from feature_vector_log import feature_fingerprint
from models.inference import EXACT_141_FEATURES


def _valid_features(*, rank: float = 100.0) -> dict:
    row = {name: 0.0 for name in EXACT_141_FEATURES}
    row.update(
        {
            "Player1_Rank": rank,
            "Surface_Hard": 1.0,
            "Level_A": 1.0,
            "Round_R32": 1.0,
            "P1_Hand_U": 1.0,
            "P2_Hand_U": 1.0,
            "P1_Country_Other": 1.0,
            "P2_Country_Other": 1.0,
        }
    )
    return row


def _prediction(uid: str, p1: str, p2: str, **overrides) -> dict:
    row = {
        "match_uid": uid,
        "match_date": "2026-07-10",
        "tournament": "Test Open",
        "p1": p1,
        "p2": p2,
        "logged_at": "2026-07-10T10:05:00Z",
        "odds_scraped_at": "2026-07-10T10:00:00Z",
        "match_start_time": "7/10/26 9:00 AM",
        "feature_snapshot_id": "",
        "latest_feature_snapshot_id": "",
        "feature_vector_sha256": "",
        "features_complete": True,
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "p1_odds_decimal": 1.8,
        "p2_odds_decimal": 2.1,
        "market_p1_prob": 0.54,
        "market_p2_prob": 0.46,
        "actual_winner": 1,
    }
    row.update(overrides)
    return row


def _feature(
    snapshot_id: str,
    uid: str,
    p1: str,
    p2: str,
    *,
    rank: float = 100.0,
    complete: bool = True,
) -> dict:
    row = _valid_features(rank=rank)
    schema_hash, vector_hash, _ = feature_fingerprint(row)
    row.update(
        {
            "feature_snapshot_id": snapshot_id,
            "match_uid": uid,
            "player1_raw": p1,
            "player2_raw": p2,
            "meta_match_date": "2026-07-10",
            "timestamp": "2026-07-10T10:01:00Z",
            "status": "ok",
            "_has_defaulted_features": not complete,
            "feature_schema_sha256": schema_hash,
            "feature_vector_sha256": vector_hash,
        }
    )
    return row


def _write_inputs(tmp_path: Path, predictions: list[dict], features: list[dict]) -> Path:
    production = tmp_path / "production"
    logs = production / "logs"
    logs.mkdir(parents=True)
    pd.DataFrame(predictions).to_csv(production / "prediction_log.csv", index=False)
    pd.DataFrame(features).to_csv(logs / "features_20260710_100100.csv", index=False)
    return production


def test_classifies_gold_exact_incomplete_legacy_and_no_vector(tmp_path):
    gold_features = _feature("feat_gold", "gold", "Gold P1", "Gold P2")
    gold_hash = gold_features["feature_vector_sha256"]
    incomplete_features = _feature(
        "feat_incomplete", "incomplete", "Incomplete P1", "Incomplete P2"
    )
    legacy_features = _feature("", "", "Legacy P1", "Legacy P2", rank=111.0)
    predictions = [
        _prediction(
            "gold",
            "Gold P1",
            "Gold P2",
            feature_snapshot_id="feat_gold",
            feature_vector_sha256=gold_hash,
        ),
        _prediction(
            "incomplete",
            "Incomplete P1",
            "Incomplete P2",
            feature_snapshot_id="feat_incomplete",
            feature_vector_sha256=incomplete_features["feature_vector_sha256"],
            features_complete=False,
        ),
        _prediction("legacy", "Legacy P1", "Legacy P2"),
        _prediction("none", "No P1", "No P2"),
    ]
    production = _write_inputs(
        tmp_path,
        predictions,
        [gold_features, incomplete_features, legacy_features],
    )

    result = replay_manifest.build_replay_manifest(production).frame.set_index("match_uid")

    assert result["replay_tier"].to_dict() == {
        "gold": "GOLD_REPLAY",
        "incomplete": "EXACT_INCOMPLETE",
        "legacy": "LEGACY_MATCHED",
        "none": "NO_VECTOR",
    }
    assert result.loc["gold", "feature_vector_sha256"] == gold_hash
    assert result.loc["gold", "artifact_schema_id"] == "base_141@1.0.0"
    assert result.loc["gold", "artifact_schema_compatible"]
    assert result.loc["gold", "is_prestart_snapshot"]
    assert result.loc["legacy", "reason_codes"].startswith("LEGACY_NO_IMMUTABLE_LINEAGE")
    assert "NO_REPLAYABLE_VECTOR" in result.loc["none", "reason_codes"]


def test_alternative_snapshot_is_preserved_but_never_cherry_picked(tmp_path):
    bad = _feature("feat_opening", "match_1", "Player A", "Player B")
    bad["Round_R32"] = 0.0  # structural one-hot failure
    good = _feature("feat_later", "match_1", "Player A", "Player B", rank=101.0)
    prediction = _prediction(
        "match_1",
        "Player A",
        "Player B",
        feature_snapshot_id="feat_opening",
        latest_feature_snapshot_id="feat_later",
    )
    production = _write_inputs(tmp_path, [prediction], [bad, good])

    row = replay_manifest.build_replay_manifest(production).frame.iloc[0]

    assert row["replay_tier"] == "EXACT_INCOMPLETE"
    assert row["feature_snapshot_id"] == "feat_opening"
    assert not row["snapshot_verified"]
    assert "EXACT_LINEAGE_UNVERIFIED" in row["reason_codes"]
    assert json.loads(row["alternative_snapshot_ids"]) == ["feat_later"]
    alternatives = json.loads(row["alternative_snapshot_provenance"])
    assert any(item.get("feature_snapshot_id") == "feat_later" for item in alternatives)


def test_rejects_fabricated_half_market_without_real_decimal_odds(tmp_path):
    fake = _feature("feat_fake", "fake", "Fake P1", "Fake P2")
    genuine = _feature("feat_even", "even", "Even P1", "Even P2")
    predictions = [
        _prediction(
            "fake",
            "Fake P1",
            "Fake P2",
            feature_snapshot_id="feat_fake",
            feature_vector_sha256=fake["feature_vector_sha256"],
            p1_odds_decimal="",
            p2_odds_decimal="",
            market_p1_prob=0.5,
            market_p2_prob=0.5,
        ),
        _prediction(
            "even",
            "Even P1",
            "Even P2",
            feature_snapshot_id="feat_even",
            feature_vector_sha256=genuine["feature_vector_sha256"],
            p1_odds_decimal=2.0,
            p2_odds_decimal=2.0,
            market_p1_prob=0.5,
            market_p2_prob=0.5,
        ),
    ]
    production = _write_inputs(tmp_path, predictions, [fake, genuine])

    result = replay_manifest.build_replay_manifest(production).frame.set_index("match_uid")

    assert result.loc["fake", "replay_tier"] == "EXACT_INCOMPLETE"
    assert result.loc["fake", "odds_evidence_status"] == "rejected_0_5_without_decimal_odds"
    assert "FABRICATED_MARKET_0_5_WITHOUT_DECIMAL_ODDS" in result.loc["fake", "reason_codes"]
    assert result.loc["even", "replay_tier"] == "GOLD_REPLAY"


def test_conflicting_outcomes_are_not_authoritative_and_uid_is_deduped(tmp_path):
    feature = _feature("feat_conflict", "conflict", "Player A", "Player B")
    base = _prediction(
        "conflict",
        "Player A",
        "Player B",
        feature_snapshot_id="feat_conflict",
        feature_vector_sha256=feature["feature_vector_sha256"],
        actual_winner=1,
    )
    conflicting = dict(base)
    conflicting.update(logged_at="2026-07-10T10:06:00Z", actual_winner=2)
    production = _write_inputs(tmp_path, [base, conflicting], [feature])

    frame = replay_manifest.build_replay_manifest(production).frame

    assert len(frame) == 1
    assert frame.iloc[0]["outcome_status"] == "conflicting_terminal_outcomes"
    assert frame.iloc[0]["replay_tier"] == "EXACT_INCOMPLETE"
    assert "CONFLICTING_AUTHORITATIVE_OUTCOME" in frame.iloc[0]["reason_codes"]


def test_gold_requires_feature_evidence_and_operational_completeness(tmp_path):
    feature = _feature(
        "feat_defaulted", "defaulted", "Player A", "Player B", complete=False
    )
    production = _write_inputs(
        tmp_path,
        [
            _prediction(
                "defaulted",
                "Player A",
                "Player B",
                feature_snapshot_id="feat_defaulted",
                feature_vector_sha256=feature["feature_vector_sha256"],
                features_complete=True,
            )
        ],
        [feature],
    )

    row = replay_manifest.build_replay_manifest(production).frame.iloc[0]

    assert row["snapshot_verified"]
    assert not row["features_complete"]
    assert row["replay_tier"] == "EXACT_INCOMPLETE"
    assert "FEATURES_INCOMPLETE" in row["reason_codes"]


def test_gold_requires_feature_vector_itself_to_be_prestart(tmp_path):
    feature = _feature("feat_late", "late", "Player A", "Player B")
    # The prediction/odds evidence remains pre-start, but the selected vector was
    # captured exactly at the 13:00Z match start. At-start is not pre-start.
    feature["timestamp"] = "2026-07-10T13:00:00Z"
    production = _write_inputs(
        tmp_path,
        [
            _prediction(
                "late",
                "Player A",
                "Player B",
                feature_snapshot_id="feat_late",
                feature_vector_sha256=feature["feature_vector_sha256"],
            )
        ],
        [feature],
    )

    row = replay_manifest.build_replay_manifest(production).frame.iloc[0]

    assert row["replay_tier"] == "EXACT_INCOMPLETE"
    assert not row["is_prestart_snapshot"]
    assert "FEATURE_VECTOR_NOT_PRESTART" in row["reason_codes"]


def test_gold_requires_snapshot_v2_contract_and_oriented_feature_identity(tmp_path):
    wrong_identity = _feature("feat_wrong", "wrong", "Other A", "Player B")
    production = _write_inputs(
        tmp_path,
        [
            _prediction(
                "wrong",
                "Player A",
                "Player B",
                feature_snapshot_id="feat_wrong",
                feature_vector_sha256=wrong_identity["feature_vector_sha256"],
                logging_quality="legacy_backfilled",
                rescore_quality="legacy_fallback_match",
            )
        ],
        [wrong_identity],
    )

    row = replay_manifest.build_replay_manifest(production).frame.iloc[0]

    assert row["replay_tier"] == "EXACT_INCOMPLETE"
    assert not row["snapshot_verified"]
    assert "FEATURE_IDENTITY_UNVERIFIED" in row["reason_codes"]
    assert "LOGGING_NOT_SNAPSHOT_V2" in row["reason_codes"]
    assert "RESCORE_NOT_EXACT_FEATURE_SNAPSHOT" in row["reason_codes"]


def test_legacy_multiple_distinct_vectors_is_no_vector_not_nearest_pick(tmp_path):
    first = _feature("", "", "Legacy P1", "Legacy P2", rank=100.0)
    second = _feature("", "", "Legacy P1", "Legacy P2", rank=101.0)
    production = _write_inputs(
        tmp_path,
        [_prediction("legacy", "Legacy P1", "Legacy P2")],
        [first, second],
    )

    row = replay_manifest.build_replay_manifest(production).frame.iloc[0]

    assert row["replay_tier"] == "NO_VECTOR"
    assert "AMBIGUOUS_LEGACY_VECTOR" in row["reason_codes"]
    assert row["feature_vector_sha256"] == ""


def test_default_cli_is_read_only_and_explicit_write_is_versioned(tmp_path, capsys):
    feature = _feature("feat_1", "match_1", "Player A", "Player B")
    production = _write_inputs(
        tmp_path,
        [
            _prediction(
                "match_1",
                "Player A",
                "Player B",
                feature_snapshot_id="feat_1",
                feature_vector_sha256=feature["feature_vector_sha256"],
            )
        ],
        [feature],
    )
    before = sorted(path.relative_to(production) for path in production.rglob("*"))

    assert replay_manifest.main(["--prod-dir", str(production)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["tier_counts"]["GOLD_REPLAY"] == 1
    assert sorted(path.relative_to(production) for path in production.rglob("*")) == before

    manifest = replay_manifest.build_replay_manifest(production)
    destination = replay_manifest.write_replay_manifest(
        manifest,
        tmp_path / "outputs",
        generated_at=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
    )
    assert destination.name == "historical_replay_20260713T120000Z"
    csv_path = destination / replay_manifest.OUTPUT_CSV
    json_path = destination / replay_manifest.OUTPUT_JSON
    metadata = json.loads(json_path.read_text())
    assert csv_path.exists()
    assert metadata["schema_version"] == "1.0.0"
    assert metadata["artifact_schema"]["id"] == "base_141@1.0.0"
    assert metadata["artifacts"][replay_manifest.OUTPUT_CSV]["rows"] == 1
    assert len(metadata["artifacts"][replay_manifest.OUTPUT_CSV]["sha256"]) == 64
    assert metadata["source_files"][0]["sha256"]
