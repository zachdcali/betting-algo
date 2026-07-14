from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import sys

import pytest

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

from storage.live_records import (  # noqa: E402
    MATCH_METADATA_TABLE_REQUIRED, LiveRecordBuilder, match_identity,
    match_metadata_observation, resolve_match_metadata,
)
from feature_contract import vector_sha256  # noqa: E402


NOW = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)


def _builder():
    return LiveRecordBuilder("run_20260713T180000Z", NOW)


def _identity(round_code="R32", surface="Hard"):
    return match_identity(
        player1="Player One", player2="Player Two",
        match_date=date(2026, 7, 14), tournament="Test Open",
        round_code=round_code, surface=surface,
        source_event_key="bovada-event-123",
    )


def _record(batch):
    assert len(batch.unique_records) == 1
    return batch.unique_records[0]


def test_match_anchor_survives_metadata_correction_while_legacy_uid_does_not():
    provisional = _identity(round_code="", surface="Hard")
    corrected = _identity(round_code="R32", surface="Clay")

    assert provisional.match_anchor_key == corrected.match_anchor_key
    assert provisional.match_uid != corrected.match_uid
    assert MATCH_METADATA_TABLE_REQUIRED == "ops.match_metadata_observations"


def test_run_and_stage_records_are_typed_deterministic_and_timezone_strict():
    builder = _builder()
    first = _record(builder.pipeline_run(metrics={"odds_rows_fetched": 12}))
    second = _record(builder.pipeline_run(metrics={"odds_rows_fetched": 12}))
    stage = _record(builder.pipeline_stage(
        stage_name="odds_fetch", status="success", started_at=NOW,
        completed_at=NOW + timedelta(minutes=1), metrics={"rows": 12},
    ))

    assert first["run_id"] == second["run_id"]
    assert first["idempotency_key"] == "pipeline_run:run_20260713T180000Z"
    assert json.loads(first["metrics"])["odds_rows_fetched"] == 12
    assert stage["run_id"] == first["run_id"]
    assert stage["status"] == "success"
    with pytest.raises(ValueError, match="explicit timezone"):
        LiveRecordBuilder("bad", datetime(2026, 7, 13, 18, 0))


def test_fetch_failure_is_a_first_class_observation_with_no_fake_data():
    builder = _builder()
    batch = builder.source_fetch(
        source_name="bovada", fetch_kind="odds", attempt=1,
        started_at=NOW, completed_at=NOW + timedelta(seconds=30),
        status="failed", error_message="coverage 4/20", rows_observed=4,
    )
    record = _record(batch)

    assert batch.table == "raw.source_fetches"
    assert record["status"] == "failed"
    assert record["rows_observed"] == 4
    assert record["error_message"] == "coverage 4/20"
    assert "match_uid" not in record
    with pytest.raises(ValueError, match="require error_message"):
        builder.source_fetch(
            source_name="bovada", fetch_kind="odds", attempt=1,
            started_at=NOW, status="failed",
        )


def test_odds_observation_requires_complete_finite_two_sided_market():
    builder = _builder()
    fetch = _record(builder.source_fetch(
        source_name="bovada", fetch_kind="odds", attempt=1,
        started_at=NOW, completed_at=NOW + timedelta(seconds=10),
        status="success", rows_observed=1,
    ))
    batch = builder.odds_observation(
        identity=_identity(), source_fetch_id=fetch["source_fetch_id"],
        observed_at=NOW + timedelta(seconds=10),
        match_start_at_utc=NOW + timedelta(hours=5),
        player1_decimal_odds="1.80", player2_decimal_odds="2.10",
        player1_market_probability="0.54", player2_market_probability="0.46",
        tournament="Test Open", event_title="ATP Test Open",
        surface="Hard", level="A", round_code="R32",
    )
    record = _record(batch)

    assert batch.table == "ops.odds_observations"
    assert record["source_fetch_id"] == fetch["source_fetch_id"]
    assert record["match_start_at_utc"] == NOW + timedelta(hours=5)
    assert str(record["player1_decimal_odds"]) == "1.80"
    assert record["validation_status"] == "valid_two_sided_prestart"
    assert record["inference_eligible"] is True
    with pytest.raises(ValueError, match="must be numeric"):
        builder.odds_observation(
            identity=_identity(), source_fetch_id=fetch["source_fetch_id"],
            observed_at=NOW, match_start_at_utc=NOW + timedelta(hours=5),
            player1_decimal_odds=None, player2_decimal_odds="2.10",
            player1_market_probability="0.54", player2_market_probability="0.46",
        )
    with pytest.raises(ValueError, match="greater than 1"):
        builder.odds_observation(
            identity=_identity(), source_fetch_id=fetch["source_fetch_id"],
            observed_at=NOW, match_start_at_utc=NOW + timedelta(hours=5),
            player1_decimal_odds="1.0", player2_decimal_odds="2.10",
            player1_market_probability="0.54", player2_market_probability="0.46",
        )
    with pytest.raises(ValueError, match="sum to 1"):
        builder.odds_observation(
            identity=_identity(), source_fetch_id=fetch["source_fetch_id"],
            observed_at=NOW, match_start_at_utc=NOW + timedelta(hours=5),
            player1_decimal_odds="1.80", player2_decimal_odds="2.10",
            player1_market_probability="0.54", player2_market_probability="0.40",
        )
    with pytest.raises(ValueError, match="before match start"):
        builder.odds_observation(
            identity=_identity(), source_fetch_id=fetch["source_fetch_id"],
            observed_at=NOW + timedelta(hours=5),
            match_start_at_utc=NOW + timedelta(hours=5),
            player1_decimal_odds="1.80", player2_decimal_odds="2.10",
            player1_market_probability="0.54", player2_market_probability="0.46",
        )


def test_feature_snapshot_fails_closed_and_predictions_are_family_observations():
    builder = _builder()
    identity = _identity()
    snapshot = _record(builder.feature_snapshot(
        identity=identity, player1=identity.player1, player2=identity.player2,
        captured_at=NOW, build_status="skip", features_complete=True,
        feature_vector={}, feature_vector_sha256=None, feature_count=141,
        defaulted_features=["match_start_time"],
    ))
    assert snapshot["features_complete"] is False
    assert snapshot["feature_vector_sha256"] is None

    predictions = builder.prediction_observations(
        identity=identity, feature_snapshot_id=snapshot["feature_snapshot_id"],
        predicted_at=NOW, external_prediction_id="pred_1",
        predictions=[
            {"model_family": "nn", "model_version": "1.2.1",
             "model_role": "promoted", "player1_probability": 0.6,
             "player2_probability": 0.4},
            {"model_family": "xgboost", "model_version": "1.0.0",
             "player1_probability": 0.55, "player2_probability": 0.45},
        ],
    )
    assert predictions.table == "ml.prediction_observations"
    assert {record["model_family"] for record in predictions.unique_records} == {
        "nn", "xgboost"
    }
    assert len({record["idempotency_key"] for record in predictions.unique_records}) == 2


def test_complete_feature_and_decision_eligible_prediction_bind_exact_contract():
    schema = json.loads((PRODUCTION / "features/schema_141.json").read_text())
    names = tuple(
        item["name"] for item in sorted(schema["features"], key=lambda item: item["index"])
    )
    vector = {name: 0.0 for name in names}
    vector.update({
        "Surface_Hard": 1.0,
        "Level_A": 1.0,
        "Round_R32": 1.0,
        "P1_Hand_U": 1.0,
        "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0,
        "P2_Country_Other": 1.0,
    })
    builder = _builder()
    identity = _identity()
    snapshot = _record(builder.feature_snapshot(
        identity=identity, player1=identity.player1, player2=identity.player2,
        captured_at=NOW, build_status="ok", features_complete=True,
        feature_vector=vector,
        feature_vector_sha256=vector_sha256(vector, names),
        feature_count=len(names),
    ))
    prediction = _record(builder.prediction_observations(
        identity=identity, feature_snapshot_id=snapshot["feature_snapshot_id"],
        predicted_at=NOW, external_prediction_id="pred_decision_grade",
        predictions=[{
            "model_family": "nn", "model_version": "v1.2.3",
            "model_role": "promoted", "player1_probability": 0.6,
            "player2_probability": 0.4,
            "model_release_key": "model_release:nn:1.2.3",
            "decision_eligible": True,
        }],
    ))

    assert snapshot["features_complete"] is True
    assert snapshot["feature_count"] == 141
    assert prediction["model_version"] == "1.2.3"
    assert prediction["decision_eligible"] is True
    assert prediction["model_release_key"] == "model_release:nn:1.2.3"

    with pytest.raises(ValueError, match="requires model_release_key"):
        builder.prediction_observations(
            identity=identity, feature_snapshot_id=snapshot["feature_snapshot_id"],
            predicted_at=NOW, external_prediction_id="missing_release",
            predictions=[{
                "model_family": "nn", "model_version": "1.2.3",
                "model_role": "promoted", "player1_probability": 0.6,
                "player2_probability": 0.4, "decision_eligible": True,
            }],
        )


def test_metadata_projection_never_erases_or_downgrades_known_values():
    observations = [
        {
            "observed_at": NOW,
            "surface": "Clay", "round_code": "R32",
            "match_start_at_utc": NOW + timedelta(hours=5),
            "field_provenance": {
                "surface": "official", "round_code": "official",
                "match_start_at_utc": "bovada",
            },
        },
        {
            "observed_at": NOW + timedelta(hours=1),
            "observation_status": "failed",
            "surface": None, "round_code": None, "match_start_at_utc": None,
            "field_provenance": {},
        },
        {
            "observed_at": NOW + timedelta(hours=2),
            "surface": "Hard", "round_code": None,
            "match_start_at_utc": None,
            "field_provenance": {"surface": "default"},
        },
    ]

    resolved = resolve_match_metadata(observations)

    assert resolved["surface"] == "Clay"
    assert resolved["round_code"] == "R32"
    assert resolved["match_start_at_utc"] == NOW + timedelta(hours=5)
    assert resolved["_field_provenance"]["surface"]["source"] == "official"


def test_proposed_metadata_row_requires_success_and_field_level_provenance():
    identity = _identity()
    record = _record(match_metadata_observation(
        identity=identity, run_id=_builder().run_id, observed_at=NOW,
        source_name="tournament_resolver",
        field_provenance={
            "match_date": "bovada", "surface": "tournament_registry",
            "round_code": "official",
        },
        surface="Clay", round_code="R32",
    ))
    assert record["match_anchor_key"] == identity.match_anchor_key
    assert json.loads(record["field_provenance"])["surface"] == "tournament_registry"

    with pytest.raises(ValueError, match="failed refreshes"):
        match_metadata_observation(
            identity=identity, run_id=_builder().run_id, observed_at=NOW,
            observation_status="failed", source_name="bovada",
            field_provenance={},
        )
    with pytest.raises(ValueError, match="requires known field provenance"):
        match_metadata_observation(
            identity=identity, run_id=_builder().run_id, observed_at=NOW,
            source_name="resolver", field_provenance={}, surface="Clay",
        )
