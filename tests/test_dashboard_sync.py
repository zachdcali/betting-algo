import json
from pathlib import Path
import sys

import pandas as pd
import pytest

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

import dashboard_sync  # noqa: E402
from evaluation import cohorts  # noqa: E402
from feature_contract import feature_fingerprint  # noqa: E402
from feature_lineage import FeatureLineageConflict, resolve_feature_lineage  # noqa: E402
from feature_vector_log import COLS as FEATURE_VECTOR_COLS  # noqa: E402
from models.inference import EXACT_141_FEATURES  # noqa: E402


def _aggregate_feature(snapshot_id: str) -> dict:
    payload = {name: 0.0 for name in EXACT_141_FEATURES}
    payload.update({
        "Surface_Hard": 1.0, "Level_A": 1.0, "Round_R32": 1.0,
        "P1_Hand_U": 1.0, "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0, "P2_Country_Other": 1.0,
    })
    return {
        "feature_snapshot_id": snapshot_id,
        "build_status": "ok",
        "features_complete": True,
        "features_json": json.dumps(payload),
    }


def _lineage_vector(rank: float) -> dict[str, float]:
    payload = {name: 0.0 for name in EXACT_141_FEATURES}
    payload.update({
        "Player1_Rank": rank,
        "Surface_Hard": 1.0,
        "Level_A": 1.0,
        "Round_R32": 1.0,
        "P1_Hand_U": 1.0,
        "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0,
        "P2_Country_Other": 1.0,
    })
    return payload


def _derived_lineage_row(vector: dict[str, float]) -> dict:
    schema_hash, vector_hash, count = feature_fingerprint(
        vector, EXACT_141_FEATURES
    )
    return {
        "p1": "Player One", "p2": "Player Two",
        "match_date": "2026-07-13", "logged_at": "2026-07-13T10:00:00Z",
        "run_id": "run_1", "match_uid": "match_1",
        "feature_snapshot_id": "feature_1", "build_status": "ok",
        "features_complete": True,
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": vector_hash,
        "feature_count": count,
        "features_json": json.dumps(vector, separators=(",", ":")),
    }


def test_materialized_metrics_use_remote_plus_local_merged_cohort(monkeypatch):
    base = {
        "features_complete": True,
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "model_p1_prob": 0.60,
        "p1_odds_decimal": 1.8,
        "p2_odds_decimal": 2.1,
    }
    local = pd.DataFrame([{
        **base, "match_uid": "local", "actual_winner": 1,
        "feature_snapshot_id": "feat_local", "feature_snapshot_verified": True,
    }])
    merged = pd.DataFrame([
        local.iloc[0].to_dict(),
        {**base, "match_uid": "remote", "actual_winner": 2,
         "feature_snapshot_id": "feat_remote"},
    ])
    features = pd.DataFrame([
        _aggregate_feature("feat_local"),
        _aggregate_feature("feat_remote"),
    ])
    monkeypatch.setattr(cohorts, "load_prediction_log", lambda _prod: local)

    result = dashboard_sync._build_model_metrics(
        "sync_test", pred_log=merged, shadow_log=pd.DataFrame(), feature_log=features,
    )

    nn_gold = result[(result["model"] == "nn") & (result["tier"] == "gold")]
    assert int(nn_gold.iloc[0]["n"]) == 2
    assert nn_gold.iloc[0]["sync_id"] == "sync_test"
    assert nn_gold.iloc[0]["dashboard_row_key"] == "gold:nn"
    assert result["dashboard_row_key"].is_unique


def test_feature_projection_recovers_exact_rows_from_immutable_run_files(tmp_path):
    aggregate = tmp_path / "feature_vectors.csv"
    pd.DataFrame([{
        "p1": "Legacy", "p2": "Row", "match_date": "2026-07-12",
        "logged_at": "2026-07-12T10:00:00Z",
    }]).to_csv(aggregate, index=False)
    run_feature = {name: 0.0 for name in EXACT_141_FEATURES}
    run_feature.update({
        "Surface_Hard": 1.0, "Level_A": 1.0, "Round_R32": 1.0,
        "P1_Hand_U": 1.0, "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0, "P2_Country_Other": 1.0,
        "player1_raw": "Player One", "player2_raw": "Player Two",
        "meta_match_date": "2026-07-13", "timestamp": "2026-07-13T10:00:00Z",
        "run_id": "run_1", "match_uid": "match_1",
        "feature_snapshot_id": "feature_1", "status": "ok",
        "meta_defaulted_features": "", "Player1_Rank": 12,
        "P1_Hand_R": 1, "P1_Hand_L": 0, "P1_Hand_U": 0,
        "P2_Hand_R": 0, "P2_Hand_L": 1, "P2_Hand_U": 0,
    })
    pd.DataFrame([run_feature]).to_csv(tmp_path / "features_20260713_100000.csv", index=False)

    projected = dashboard_sync._load_feature_state(aggregate)
    recovered = projected[projected["feature_snapshot_id"] == "feature_1"].iloc[0]

    assert recovered["match_uid"] == "match_1"
    assert recovered["run_id"] == "run_1"
    assert recovered["features_complete"]
    assert len(recovered["feature_vector_sha256"]) == 64
    assert recovered["p1_hand"] == "R"
    assert recovered["p2_hand"] == "L"
    assert json.loads(recovered["features_json"])["Player1_Rank"] == 12.0


def test_manifest_separates_latest_attempt_from_prediction_slate():
    runs = pd.DataFrame([
        {"run_id": "run_good", "run_kind": "prediction_pipeline",
         "status": "success", "started_at": "2026-07-13T10:00:00Z"},
        {"run_id": "run_running", "run_kind": "prediction_pipeline",
         "status": "running", "started_at": "2026-07-13T11:00:00Z"},
    ])
    snapshots = pd.DataFrame([{
        "run_id": "run_good", "logged_at": "2026-07-13T10:30:00Z",
    }])

    assert dashboard_sync._latest_run_id(runs) == "run_running"
    assert dashboard_sync._accepted_prediction_run_id(runs, snapshots) == "run_good"


def test_manifest_never_accepts_snapshot_from_explicit_failed_run():
    runs = pd.DataFrame([{
        "run_id": "run_failed", "run_kind": "prediction_pipeline",
        "status": "failed", "started_at": "2026-07-13T12:00:00Z",
    }])
    snapshots = pd.DataFrame([{
        "run_id": "run_failed", "logged_at": "2026-07-13T12:01:00Z",
    }])

    assert dashboard_sync._accepted_prediction_run_id(runs, snapshots) == ""


def test_latest_run_prefers_valid_started_at_over_missing_timestamp():
    runs = pd.DataFrame([
        {"run_id": "run_valid", "run_kind": "prediction_pipeline",
         "status": "success", "started_at": "2026-07-13T12:00:00Z"},
        {"run_id": "run_missing_time", "run_kind": "prediction_pipeline",
         "status": "running", "started_at": ""},
    ])

    assert dashboard_sync._latest_run_id(runs) == "run_valid"


def test_feature_projection_normalizes_nonfinite_json_values():
    assert dashboard_sync._json_scalar(float("nan")) is None
    assert dashboard_sync._json_scalar(float("inf")) is None
    assert dashboard_sync._json_scalar("-Infinity") is None


def test_feature_publish_then_clean_clone_hydrate_preserves_immutable_repair(tmp_path):
    immutable_vector = _lineage_vector(12.0)
    stale_vector = _lineage_vector(12.000000000000002)
    stale = _derived_lineage_row(stale_vector)
    production = tmp_path / "production"
    logs = production / "logs"
    logs.mkdir(parents=True)
    aggregate = logs / "feature_vectors.csv"
    pd.DataFrame([stale]).to_csv(aggregate, index=False)

    schema_hash, immutable_hash, count = feature_fingerprint(
        immutable_vector, EXACT_141_FEATURES
    )
    immutable = {
        **immutable_vector,
        "player1_raw": "Player One", "player2_raw": "Player Two",
        "meta_match_date": "2026-07-13", "timestamp": "2026-07-13T10:00:00Z",
        "run_id": "run_1", "match_uid": "match_1",
        "feature_snapshot_id": "feature_1", "status": "ok",
        "meta_defaulted_features": "",
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": immutable_hash,
        "feature_count": count,
    }
    immutable_path = logs / "features_20260713_100000.csv"
    pd.DataFrame([immutable]).to_csv(immutable_path, index=False)

    published = dashboard_sync._load_feature_state(aggregate)
    published_row = published.iloc[0]
    assert published_row["feature_vector_sha256"] == immutable_hash
    assert json.loads(published_row["features_json"])["Player1_Rank"] == 12.0

    # A fresh clone can have the stale checked-in aggregate while the accepted
    # durable generation already contains the repair. With no local immutable
    # authority available, hydration must retain the durable exact row.
    immutable_path.unlink()
    stale_local = pd.DataFrame([stale])
    durable = published.copy()
    durable["dashboard_row_key"] = "dash_features:feature_snapshot_id:feature_1"
    durable["sync_id"] = "sync_previous"
    hydrated = dashboard_sync._merge_feature_state(durable, stale_local)
    assert list(hydrated.columns) == FEATURE_VECTOR_COLS
    hydrated.to_csv(aggregate, index=False)
    reloaded = dashboard_sync._load_feature_state(aggregate)

    assert len(reloaded) == 1
    assert reloaded.iloc[0]["feature_vector_sha256"] == immutable_hash
    assert json.loads(reloaded.iloc[0]["features_json"])["Player1_Rank"] == 12.0

    prediction = {
        "match_uid": "match_1", "feature_snapshot_id": "feature_1",
        "feature_vector_sha256": immutable_hash,
    }
    pd.DataFrame([prediction]).to_csv(production / "prediction_log.csv", index=False)
    assert bool(cohorts.load_prediction_log(str(production)).iloc[0][
        "feature_snapshot_verified"
    ])
    prediction["feature_vector_sha256"] = stale["feature_vector_sha256"]
    pd.DataFrame([prediction]).to_csv(production / "prediction_log.csv", index=False)
    assert not bool(cohorts.load_prediction_log(str(production)).iloc[0][
        "feature_snapshot_verified"
    ])


def test_invalid_durable_projection_cannot_become_clean_clone_repair_authority():
    durable = _derived_lineage_row(_lineage_vector(12.0))
    durable["feature_vector_sha256"] = "0" * 64
    materially_different_local = _derived_lineage_row(_lineage_vector(13.0))

    with pytest.raises(
        FeatureLineageConflict,
        match="invalid decision-grade projection authority",
    ):
        dashboard_sync._merge_feature_state(
            pd.DataFrame([durable]),
            pd.DataFrame([materially_different_local]),
        )


def test_remote_only_invalid_durable_projection_fails_before_hydration():
    durable = _derived_lineage_row(_lineage_vector(12.0))
    durable["feature_vector_sha256"] = "0" * 64

    with pytest.raises(
        FeatureLineageConflict,
        match="invalid decision-grade projection authority",
    ):
        dashboard_sync._merge_feature_state(
            pd.DataFrame([durable]),
            pd.DataFrame(columns=FEATURE_VECTOR_COLS),
        )


@pytest.mark.parametrize("claim", ["features_complete", "feature_vector_sha256"])
def test_remote_only_nondecision_durable_projection_cannot_claim_exactness(claim):
    durable = _derived_lineage_row(_lineage_vector(12.0))
    exact_hash = durable["feature_vector_sha256"]
    durable.update(
        build_status="skip",
        features_complete=False,
        feature_vector_sha256="",
    )
    durable[claim] = True if claim == "features_complete" else exact_hash

    with pytest.raises(
        FeatureLineageConflict,
        match="exactness claim on a non-decision-grade projection authority",
    ):
        dashboard_sync._merge_feature_state(
            pd.DataFrame([durable]),
            pd.DataFrame(columns=FEATURE_VECTOR_COLS),
        )


def test_duplicate_remote_only_durable_feature_id_fails_before_merge():
    first = _derived_lineage_row(_lineage_vector(12.0))
    second = _derived_lineage_row(_lineage_vector(13.0))

    with pytest.raises(RuntimeError, match="has 2 projection rows"):
        dashboard_sync._merge_feature_state(
            pd.DataFrame([first, second]),
            pd.DataFrame(columns=FEATURE_VECTOR_COLS),
        )


def test_lone_derived_only_invalid_decision_grade_row_cannot_publish(tmp_path):
    aggregate = tmp_path / "feature_vectors.csv"
    invalid = _derived_lineage_row(_lineage_vector(12.0))
    invalid["feature_vector_sha256"] = "0" * 64
    pd.DataFrame([invalid]).to_csv(aggregate, index=False)

    with pytest.raises(
        FeatureLineageConflict,
        match="invalid decision-grade projection authority",
    ):
        dashboard_sync._load_feature_state(aggregate)


def test_explicitly_incomplete_durable_status_ok_row_remains_non_gold_repair():
    durable = _derived_lineage_row(_lineage_vector(12.0))
    payload = json.loads(durable["features_json"])
    payload.pop("Player2_Age")
    durable.update(
        features_json=json.dumps(payload, separators=(",", ":")),
        features_complete=False,
        feature_vector_sha256="",
    )
    stale_local = dict(durable)

    merged = dashboard_sync._merge_feature_state(
        pd.DataFrame([durable]), pd.DataFrame([stale_local])
    )

    assert len(merged) == 1
    assert not bool(merged.iloc[0]["features_complete"])
    assert merged.iloc[0]["feature_vector_sha256"] == ""


def test_materialized_dashboard_gold_matches_ledger_after_ulp_reconciliation(monkeypatch):
    immutable_vector = _lineage_vector(12.0)
    derived_vector = _lineage_vector(12.000000000000002)
    schema_hash, immutable_hash, count = feature_fingerprint(
        immutable_vector, EXACT_141_FEATURES
    )
    _, derived_hash, _ = feature_fingerprint(derived_vector, EXACT_141_FEATURES)
    immutable = {
        **immutable_vector,
        "run_id": "run_1", "match_uid": "match_1",
        "feature_snapshot_id": "feature_1", "status": "ok",
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": immutable_hash,
        "feature_count": count,
    }
    derived = _derived_lineage_row(derived_vector)
    lineage = resolve_feature_lineage(
        ordered_names=EXACT_141_FEATURES,
        immutable_sources=[("features_run.csv", pd.DataFrame([immutable]))],
        derived_sources=[("feature_vectors.csv", pd.DataFrame([derived]))],
    )
    prediction = pd.DataFrame([{
        "match_uid": "match_1", "actual_winner": 1,
        "feature_snapshot_id": "feature_1",
        "feature_vector_sha256": immutable_hash,
        "features_complete": True,
        "logging_quality": "snapshot_v2",
        "rescore_quality": "exact_feature_snapshot",
        "model_p1_prob": 0.61,
        "p1_odds_decimal": 1.8, "p2_odds_decimal": 2.1,
    }])
    ledger_input = prediction.copy()
    ledger_input["feature_snapshot_verified"] = True
    expected_gold = int(
        cohorts.build_scored_frame(ledger_input, None)
        .query("model == 'nn'")["is_gold"].sum()
    )
    monkeypatch.setattr(cohorts, "load_prediction_log", lambda _prod: ledger_input)
    monkeypatch.setattr(
        dashboard_sync, "load_production_feature_lineage", lambda *_args, **_kwargs: lineage
    )

    metrics = dashboard_sync._build_model_metrics(
        "sync_ulp",
        pred_log=prediction,
        shadow_log=pd.DataFrame(),
        feature_log=None,
    )
    nn_gold = metrics[(metrics["model"] == "nn") & (metrics["tier"] == "gold")]

    assert expected_gold == 1
    assert int(nn_gold.iloc[0]["n"]) == expected_gold


def test_calibration_reuses_the_verified_scored_frame_from_materialized_metrics(monkeypatch):
    scored = pd.DataFrame([
        {
            "match_uid": "verified", "model": "nn", "family": "nn",
            "p1_prob": 0.70, "p1_odds_decimal": 1.8, "p2_odds_decimal": 2.1,
            "y1": 1, "is_gold": True, "is_complete": True,
            "prediction_time": "2026-07-13T10:00:00Z",
        },
        {
            "match_uid": "verified_2", "model": "nn", "family": "nn",
            "p1_prob": 0.30, "p1_odds_decimal": 2.2, "p2_odds_decimal": 1.7,
            "y1": 0, "is_gold": True, "is_complete": True,
            "prediction_time": "2026-07-13T10:01:00Z",
        },
    ])
    from evaluation.ledger import build_live_ledger

    metrics_frame = build_live_ledger(scored)
    metrics_frame.attrs["scored_frame"] = scored

    def fail_rebuild(*_args, **_kwargs):
        raise AssertionError("calibration rebuilt an unverified scored frame")

    monkeypatch.setattr(cohorts, "build_scored_frame", fail_rebuild)

    calibration = dashboard_sync._build_model_calibration(
        "sync_verified", pred_log=pd.DataFrame(), shadow_log=pd.DataFrame(),
        odds_history=pd.DataFrame(), metrics_frame=metrics_frame,
    )

    nn_gold = calibration[
        calibration["model"].eq("nn") & calibration["tier"].eq("gold")
    ]
    assert int(nn_gold["count"].sum()) == 2
    assert calibration["calibration_row_key"].is_unique
    assert set(calibration["sync_id"]) == {"sync_verified"}


def test_sync_publishes_calibration_in_same_generation_and_manifest_counts(monkeypatch):
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args, **_kwargs):
            return None

    class FakeConnection:
        def __init__(self):
            self.cur = FakeCursor()
            self.committed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return self.cur

        def commit(self):
            self.committed = True

    connection = FakeConnection()
    monkeypatch.setattr(dashboard_sync, "connect", lambda: connection)
    monkeypatch.setattr(dashboard_sync, "STATE_SPECS", [])

    generation = {}

    def fake_metrics(sync_id, *_args, **_kwargs):
        generation["sync_id"] = sync_id
        return pd.DataFrame([{
            "model": "nn", "tier": "gold", "sync_id": sync_id,
        }])

    def fake_calibration(sync_id, *_args, metrics_frame, **_kwargs):
        assert sync_id == generation["sync_id"]
        assert set(metrics_frame["sync_id"]) == {sync_id}
        return pd.DataFrame([
            {"model": "nn", "tier": "gold", "bin_index": 0, "sync_id": sync_id},
            {"model": "nn", "tier": "gold", "bin_index": 1, "sync_id": sync_id},
        ])

    monkeypatch.setattr(dashboard_sync, "_build_model_metrics", fake_metrics)
    monkeypatch.setattr(
        dashboard_sync, "_build_model_calibration",
        fake_calibration,
    )

    published = {}
    monkeypatch.setattr(
        dashboard_sync, "_replace_table",
        lambda _cur, table, frame: published.setdefault(table, frame.copy()),
    )
    monkeypatch.setattr(dashboard_sync, "_enable_public_read", lambda *_args: None)
    manifest = {}
    monkeypatch.setattr(
        dashboard_sync, "_write_manifest",
        lambda _cur, **kwargs: manifest.update(kwargs),
    )

    counts = dashboard_sync.sync_dashboard_tables(verbose=False)

    assert counts == {"dash_model_metrics": 1, "dash_model_calibration": 2}
    assert set(published) == {"dash_model_metrics", "dash_model_calibration"}
    assert manifest["counts"] == counts
    assert manifest["sync_id"] == generation["sync_id"]
    assert set(published["dash_model_metrics"]["sync_id"]) == {manifest["sync_id"]}
    assert set(published["dash_model_calibration"]["sync_id"]) == {manifest["sync_id"]}
    assert connection.committed
