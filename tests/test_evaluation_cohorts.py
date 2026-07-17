import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
import pytest
from evaluation import cohorts
from evaluation.ledger import build_live_ledger
from feature_vector_log import feature_fingerprint
from logging_utils import build_feature_snapshot_id
from models.inference import EXACT_141_FEATURES


def _valid_features() -> dict:
    row = {name: 0.0 for name in EXACT_141_FEATURES}
    row.update({
        "Surface_Hard": 1.0, "Level_A": 1.0, "Round_R32": 1.0,
        "P1_Hand_U": 1.0, "P2_Hand_U": 1.0,
        "P1_Country_Other": 1.0, "P2_Country_Other": 1.0,
    })
    return row


def _pred_log():
    return pd.DataFrame([
        # settled gold row, all models present
        dict(match_uid="m1", actual_winner=1, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             feature_snapshot_verified=True,
             model_p1_prob=0.7, xgb_p1_prob=0.65, rf_p1_prob=0.6, market_p1_prob=0.62,
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        # settled complete-but-legacy row, NN+market only (xgb/rf null)
        dict(match_uid="m2", actual_winner=2, features_complete=True,
             logging_quality="legacy_backfilled", rescore_quality="legacy_fallback_match",
             model_p1_prob=0.4, xgb_p1_prob=np.nan, rf_p1_prob=np.nan, market_p1_prob=0.45,
             p1_odds_decimal=2.3, p2_odds_decimal=1.6),
        # unsettled row -> excluded
        dict(match_uid="m3", actual_winner=np.nan, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.5, xgb_p1_prob=0.5, rf_p1_prob=0.5, market_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=1.8),
        # historical void sentinel -> excluded, never scored as a P2 win
        dict(match_uid="m4", actual_winner=-1, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.5, xgb_p1_prob=0.5, rf_p1_prob=0.5, market_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=2.0),
    ])


def _shadow_log():
    # shadow rows for m1 (settled) and m3 (unsettled -> excluded)
    return pd.DataFrame([
        dict(match_uid="m1", model_family="catboost", shadow_p1_prob=0.68,
             model_version="cat_v1", feature_snapshot_id="feat_m1",
             logged_at="2026-07-01T10:00:00Z",
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        # Hourly repeat of the same match/version must not inflate n.
        dict(match_uid="m1", model_family="catboost", shadow_p1_prob=0.99,
             model_version="cat_v1", feature_snapshot_id="feat_m1",
             logged_at="2026-07-01T11:00:00Z",
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        dict(match_uid="m1", model_family="lightgbm", shadow_p1_prob=0.66,
             model_version="lgbm_v1", feature_snapshot_id="feat_m1",
             logged_at="2026-07-01T10:00:00Z",
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        dict(match_uid="m3", model_family="catboost", shadow_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=1.8),
    ])


def test_ground_truth_orientation_and_dedup():
    gt = cohorts.build_ground_truth(_pred_log())
    assert gt.loc["m1"] == 1   # player1 won
    assert gt.loc["m2"] == 0   # player2 won
    assert "m3" not in gt.index  # unsettled excluded
    assert "m4" not in gt.index  # void sentinel excluded


def _semantic_prediction(
    match_uid: str,
    *,
    p1: str,
    p2: str,
    actual_winner: int,
    event: str,
    model_p1_prob: float,
) -> dict:
    return dict(
        match_uid=match_uid,
        p1=p1,
        p2=p2,
        match_date="2026-07-14",
        tournament=event,
        actual_winner=actual_winner,
        record_status="settled",
        features_complete=True,
        logging_quality="snapshot_v2",
        rescore_quality="exact_feature_snapshot",
        feature_snapshot_verified=True,
        model_p1_prob=model_p1_prob,
        xgb_p1_prob=model_p1_prob,
        rf_p1_prob=model_p1_prob,
        market_p1_prob=model_p1_prob,
        p1_odds_decimal=2.0,
        p2_odds_decimal=2.0,
    )


def test_semantic_duplicate_uids_keep_one_deterministic_oriented_result():
    predictions = pd.DataFrame([
        _semantic_prediction(
            "uid_z", p1="Alice Smith", p2="Bob-Jones", actual_winner=1,
            event="Challenger - Lincoln (9)", model_p1_prob=0.70,
        ),
        # Same winner and match, but the source flipped player orientation and
        # changed the volatile Bovada event-count suffix.
        _semantic_prediction(
            "uid_a", p1="Bob Jones", p2="Alice Smith", actual_winner=2,
            event="Challenger - Lincoln (4)", model_p1_prob=0.30,
        ),
    ])

    forward = cohorts.build_ground_truth(predictions)
    reversed_input = cohorts.build_ground_truth(predictions.iloc[::-1])
    assert forward.to_dict() == {"uid_a": 0}
    assert reversed_input.to_dict() == {"uid_a": 0}

    scored = cohorts.build_scored_frame(predictions, None)
    assert set(scored["match_uid"]) == {"uid_a"}
    assert len(scored) == 4
    assert scored["y1"].eq(0).all()
    assert scored.loc[scored.model == "nn", "p1_prob"].iloc[0] == 0.30


def test_semantic_representative_prefers_decision_grade_over_lexical_uid():
    lower_quality = _semantic_prediction(
        "uid_a", p1="Alice Smith", p2="Bob Jones", actual_winner=1,
        event="Lincoln", model_p1_prob=0.70,
    )
    lower_quality.update({
        "features_complete": False,
        "feature_snapshot_verified": False,
        "logging_quality": "legacy_backfilled",
        "rescore_quality": "legacy_fallback_match",
        "identity_status": "legacy_unclassified",
        "xgb_p1_prob": np.nan,
        "rf_p1_prob": np.nan,
        "market_p1_prob": np.nan,
    })
    decision_grade = _semantic_prediction(
        "uid_z", p1="Bob Jones", p2="Alice Smith", actual_winner=2,
        event="Lincoln", model_p1_prob=0.30,
    )
    decision_grade["identity_status"] = "canonical"
    predictions = pd.DataFrame([lower_quality, decision_grade])

    assert cohorts.build_ground_truth(predictions).to_dict() == {"uid_z": 0}
    scored = cohorts.build_scored_frame(predictions, None)
    assert set(scored["match_uid"]) == {"uid_z"}
    assert len(scored) == 4
    assert scored["is_gold"].all()
    assert scored["y1"].eq(0).all()


def test_semantic_duplicate_uids_with_contradictory_winners_fail_closed():
    predictions = pd.DataFrame([
        _semantic_prediction(
            "uid_a", p1="Alice Smith", p2="Bob Jones", actual_winner=1,
            event="Lincoln", model_p1_prob=0.70,
        ),
        # With reversed orientation, actual_winner=1 now asserts that Bob won.
        _semantic_prediction(
            "uid_b", p1="Bob Jones", p2="Alice Smith", actual_winner=1,
            event="Lincoln", model_p1_prob=0.70,
        ),
    ])

    assert cohorts.build_ground_truth(predictions).empty
    assert cohorts.build_scored_frame(predictions, None).empty


def test_internal_uid_outcome_conflict_poisoning_semantic_group_fails_closed():
    valid = _semantic_prediction(
        "uid_a", p1="Alice Smith", p2="Bob Jones", actual_winner=1,
        event="Lincoln", model_p1_prob=0.70,
    )
    conflicting_alice = _semantic_prediction(
        "uid_b", p1="Alice Smith", p2="Bob Jones", actual_winner=1,
        event="Lincoln", model_p1_prob=0.70,
    )
    conflicting_bob = {
        **conflicting_alice,
        "actual_winner": 2,
        "model_p1_prob": 0.65,
    }
    predictions = pd.DataFrame([valid, conflicting_alice, conflicting_bob])

    assert cohorts.build_ground_truth(predictions).empty
    assert cohorts.build_scored_frame(predictions, None).empty


def test_any_identity_tombstone_blocks_uid_ground_truth_and_tiers():
    rows = pd.DataFrame([
        {
            "match_uid": "m1", "actual_winner": 1,
            "record_status": "settled", "features_complete": True,
            "feature_snapshot_verified": True,
            "logging_quality": "snapshot_v2",
            "rescore_quality": "exact_feature_snapshot",
        },
        {
            "match_uid": "m1", "actual_winner": None,
            "record_status": "identity_conflict", "features_complete": False,
            "feature_snapshot_verified": False,
            "logging_quality": "snapshot_v2",
            "rescore_quality": "exact_feature_snapshot",
        },
    ])

    assert "m1" not in cohorts.build_ground_truth(rows).index
    tier = cohorts._tier_flags(rows).iloc[0]
    assert not bool(tier["is_complete"])
    assert not bool(tier["is_gold"])


def test_scored_frame_models_and_tiers():
    scored = cohorts.build_scored_frame(_pred_log(), None)
    m1 = scored[scored.match_uid == "m1"]
    assert set(m1.model) == {"nn", "xgb", "rf", "market"}
    assert bool(m1.is_gold.iloc[0]) is True
    m2 = scored[scored.match_uid == "m2"]
    assert set(m2.model) == {"nn", "market"}
    assert bool(m2.is_gold.iloc[0]) is False
    assert bool(m2.is_complete.iloc[0]) is True
    assert scored[(scored.match_uid == "m1") & (scored.model == "nn")].y1.iloc[0] == 1


def test_tour_segment_classification_uses_explicit_level_contract_and_itf_override():
    cases = [
        ("A", "Miami", "atp"),
        ("M", "Monte Carlo", "atp"),
        ("G", "French Open Men's Singles", "atp"),
        ("C", "Iasi", "challenger"),
        ("CH", "Alicante", "challenger"),
        ("15", "Monastir", "itf"),
        (25.0, "Mumbai", "itf"),
        # Legacy source defect: explicit ITF identity outranks level A.
        ("A", "ITF Men Wodonga", "itf"),
        ("", "Unknown event", "unclassified"),
    ]
    for level, tournament, expected in cases:
        assert cohorts.classify_tour_segment(level, tournament) == expected


def test_hydrated_string_probabilities_and_odds_are_sanitized_before_metrics():
    hydrated = pd.DataFrame([
        dict(
            match_uid="full", actual_winner="1", features_complete="true",
            logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
            feature_snapshot_verified="true",
            model_p1_prob="0.0", xgb_p1_prob="1.0", rf_p1_prob="0.25",
            market_p1_prob="0.55", p1_odds_decimal="2.0", p2_odds_decimal="2.1",
        ),
        dict(
            match_uid="partial", actual_winner="2", features_complete="1",
            logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
            feature_snapshot_verified="1",
            model_p1_prob="1.0", xgb_p1_prob="", rf_p1_prob="not-a-number",
            market_p1_prob="0.0", p1_odds_decimal="", p2_odds_decimal="Infinity",
        ),
        dict(
            match_uid="bad_odds", actual_winner="1", features_complete="true",
            logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
            feature_snapshot_verified="true",
            model_p1_prob="0.5", xgb_p1_prob="", rf_p1_prob="",
            market_p1_prob="", p1_odds_decimal="not-odds", p2_odds_decimal="-Infinity",
        ),
        dict(
            match_uid="invalid", actual_winner="1", features_complete="true",
            logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
            feature_snapshot_verified="true",
            model_p1_prob="NaN", xgb_p1_prob="inf", rf_p1_prob="-0.01",
            market_p1_prob="1.01", p1_odds_decimal="garbage", p2_odds_decimal="1.0",
        ),
    ])

    scored = cohorts.build_scored_frame(hydrated, None)

    assert set(scored.loc[scored.match_uid == "full", "model"]) == {
        "nn", "xgb", "rf", "market",
    }
    assert set(scored.loc[scored.match_uid == "partial", "model"]) == {"nn", "market"}
    assert "invalid" not in set(scored["match_uid"])
    assert set(scored["p1_prob"]) >= {0.0, 1.0}
    partial = scored[scored.match_uid == "partial"]
    assert partial["p1_odds_decimal"].isna().all()
    assert partial["p2_odds_decimal"].isna().all()
    bad_odds = scored[scored.match_uid == "bad_odds"]
    assert bad_odds["p1_odds_decimal"].isna().all()
    assert bad_odds["p2_odds_decimal"].isna().all()
    assert cohorts.intersection_uids(
        scored, ["nn", "xgb", "rf", "market"], "is_complete",
    ) == {"full"}

    # This exercises metrics and both ROI modes. The blank hydrated odds must
    # be skipped while the valid-odds row remains a candidate.
    ledger = build_live_ledger(scored)
    nn_complete = ledger[(ledger.model == "nn") & (ledger.tier == "complete")].iloc[0]
    assert int(nn_complete["n"]) == 3
    assert int(nn_complete["n_bets_flat"]) == 1
    assert int(nn_complete["n_bets_kelly"]) == 1


def test_shadow_probabilities_use_the_same_fail_closed_numeric_boundary():
    pred_log = _pred_log().iloc[:1].copy()
    shadow = pd.DataFrame([
        dict(match_uid="m1", model_family="xgboost", model_version="zero",
             shadow_p1_prob="0", p1_odds_decimal="", p2_odds_decimal="2.2"),
        dict(match_uid="m1", model_family="xgboost", model_version="one",
             shadow_p1_prob="1.0", p1_odds_decimal="2.1", p2_odds_decimal="NaN"),
        *[
            dict(match_uid="m1", model_family="xgboost", model_version=f"bad_{i}",
                 shadow_p1_prob=value, p1_odds_decimal="2.1", p2_odds_decimal="2.2")
            for i, value in enumerate((
                "", "NaN", "inf", "-inf", "-0.01", "1.01", "bad", 10**10000,
            ))
        ],
    ])

    scored = cohorts.build_scored_frame(pred_log, shadow)
    shadows = scored[scored.model.str.startswith("shadow_")]

    assert set(shadows["model"]) == {"shadow_zero", "shadow_one"}
    assert set(shadows["p1_prob"]) == {0.0, 1.0}
    assert pd.isna(shadows.loc[shadows.model == "shadow_zero", "p1_odds_decimal"].iloc[0])
    assert pd.isna(shadows.loc[shadows.model == "shadow_one", "p2_odds_decimal"].iloc[0])


def test_all_invalid_probabilities_produce_an_empty_ledger_not_an_exception():
    hydrated = pd.DataFrame([
        dict(
            match_uid="invalid", actual_winner="1", features_complete="true",
            model_p1_prob="", xgb_p1_prob="NaN", rf_p1_prob="inf",
            market_p1_prob="-1",
        ),
    ])

    scored = cohorts.build_scored_frame(hydrated, None)

    assert scored.empty
    assert scored["is_gold"].dtype == bool
    assert scored["is_complete"].dtype == bool
    assert build_live_ledger(scored).empty


def test_shadow_models_join_to_ground_truth():
    scored = cohorts.build_scored_frame(_pred_log(), _shadow_log())
    sh = scored[scored.model.str.startswith("shadow_")]
    # only m1 shadows survive (m3 unsettled)
    assert set(sh.match_uid) == {"m1"}
    assert set(sh.model) == {"shadow_cat_v1", "shadow_lgbm_v1"}
    assert sh.y1.unique().tolist() == [1]
    assert len(sh) == 2
    assert sh.loc[sh.model == "shadow_cat_v1", "p1_prob"].iloc[0] == 0.68


def test_load_prediction_log_verifies_feature_snapshot_foreign_key(tmp_path):
    log = _pred_log().iloc[:2].copy()
    log.loc[log.index[0], ["run_id", "p1", "p2"]] = [
        "run_1", "Player A", "Player B",
    ]
    present_id = build_feature_snapshot_id(
        "m1", "run_1", "Player A", "Player B"
    )
    log["feature_snapshot_id"] = [present_id, "feat_missing"]
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    feature_row = _valid_features()
    feature_row.update({
        "feature_snapshot_id": present_id,
        "match_uid": "m1",
        "run_id": "run_1",
        "player1_raw": "Player A",
        "player2_raw": "Player B",
        "status": "ok",
    })
    pd.DataFrame([feature_row]).to_csv(
        logs / "features_20260701.csv", index=False,
    )

    loaded = cohorts.load_prediction_log(str(tmp_path)).set_index("match_uid")
    assert bool(loaded.loc["m1", "feature_snapshot_verified"])
    assert not bool(loaded.loc["m2", "feature_snapshot_verified"])
    scored = cohorts.build_scored_frame(loaded.reset_index(), None)
    assert bool(scored.loc[scored.match_uid == "m1", "is_gold"].iloc[0])
    assert not bool(scored.loc[scored.match_uid == "m2", "is_gold"].iloc[0])


def test_feature_snapshot_verification_rejects_cross_match_pointer(tmp_path):
    log = _pred_log().iloc[:1].copy()
    log.loc[log.index[0], ["run_id", "p1", "p2"]] = [
        "run_prediction", "Player A", "Player B",
    ]
    cross_id = build_feature_snapshot_id(
        "different_match_uid", "run_authority", "Player A", "Player B"
    )
    log["feature_snapshot_id"] = [cross_id]
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    feature_row = _valid_features()
    feature_row.update({
        "feature_snapshot_id": cross_id,
        "match_uid": "different_match_uid",
        "run_id": "run_authority",
        "player1_raw": "Player A",
        "player2_raw": "Player B",
        "status": "ok",
    })
    pd.DataFrame([feature_row]).to_csv(logs / "features_cross.csv", index=False)

    loaded = cohorts.load_prediction_log(str(tmp_path))

    assert not bool(loaded.iloc[0]["feature_snapshot_verified"])
    assert loaded.attrs["feature_snapshot_verification"][
        "match_uid_mismatch_count"
    ] == 1


@pytest.mark.parametrize(
    ("prediction_run", "prediction_p1", "prediction_p2"),
    [
        ("wrong_run", "Player A", "Player B"),
        ("run_authority", "Player B", "Player A"),
    ],
)
def test_feature_snapshot_verification_rejects_wrong_run_or_orientation(
    tmp_path, prediction_run, prediction_p1, prediction_p2,
):
    match_uid = "match_identity"
    authority_run = "run_authority"
    snapshot_id = build_feature_snapshot_id(
        match_uid, authority_run, "Player A", "Player B"
    )
    log = _pred_log().iloc[:1].copy()
    log["match_uid"] = match_uid
    log["run_id"] = prediction_run
    log["p1"] = prediction_p1
    log["p2"] = prediction_p2
    log["feature_snapshot_id"] = snapshot_id
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    feature_row = _valid_features()
    feature_row.update({
        "feature_snapshot_id": snapshot_id,
        "match_uid": match_uid,
        "run_id": authority_run,
        "player1_raw": "Player A",
        "player2_raw": "Player B",
        "status": "ok",
    })
    pd.DataFrame([feature_row]).to_csv(logs / "features_identity.csv", index=False)

    loaded = cohorts.load_prediction_log(str(tmp_path))

    assert not bool(loaded.iloc[0]["feature_snapshot_verified"])


def test_ulp_derived_hash_is_not_accepted_as_prediction_referential_hash(tmp_path):
    immutable = _valid_features()
    immutable["Player1_Rank"] = 12.0
    schema_hash, immutable_hash, count = feature_fingerprint(immutable)
    derived = dict(immutable)
    derived["Player1_Rank"] = math.nextafter(12.0, math.inf)
    _, derived_hash, _ = feature_fingerprint(derived)
    pd.DataFrame([{
        "match_uid": "match_1", "feature_snapshot_id": "feature_1",
        "feature_vector_sha256": derived_hash,
    }]).to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    pd.DataFrame([{
        **immutable,
        "run_id": "run_1", "match_uid": "match_1",
        "feature_snapshot_id": "feature_1", "status": "ok",
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": immutable_hash,
        "feature_count": count,
    }]).to_csv(logs / "features_run.csv", index=False)
    pd.DataFrame([{
        "run_id": "run_1", "match_uid": "match_1",
        "feature_snapshot_id": "feature_1", "build_status": "ok",
        "features_complete": True,
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": derived_hash,
        "feature_count": count,
        "features_json": json.dumps(derived, separators=(",", ":")),
    }]).to_csv(logs / "feature_vectors.csv", index=False)

    loaded = cohorts.load_prediction_log(str(tmp_path))

    assert not bool(loaded.iloc[0]["feature_snapshot_verified"])


def test_load_prediction_log_fails_on_corrupt_feature_lineage_file(tmp_path):
    log = _pred_log().iloc[:1].copy()
    log["feature_snapshot_id"] = ["feat_present"]
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "features_corrupt.csv").write_bytes(b'feature_snapshot_id,broken\n"unterminated')

    with pytest.raises(RuntimeError, match="feature snapshot verification"):
        cohorts.load_prediction_log(str(tmp_path))


def test_feature_snapshot_verification_rejects_tampered_content_hash(tmp_path):
    log = _pred_log().iloc[:1].copy()
    log["feature_snapshot_id"] = ["feat_tampered"]
    log["feature_vector_sha256"] = ["0" * 64]
    log.to_csv(tmp_path / "prediction_log.csv", index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    feature_row = _valid_features()
    feature_row.update({"feature_snapshot_id": "feat_tampered", "status": "ok"})
    pd.DataFrame([feature_row]).to_csv(logs / "features_20260701.csv", index=False)

    loaded = cohorts.load_prediction_log(str(tmp_path))
    assert not bool(loaded.iloc[0]["feature_snapshot_verified"])


def test_intersection_excludes_partial_coverage():
    scored = cohorts.build_scored_frame(_pred_log(), None)
    inter = cohorts.intersection_uids(scored, ["nn", "xgb", "rf", "market"], "is_complete")
    assert inter == {"m1"}   # m2 lacks xgb/rf


def test_market_timing_selects_first_and_last_strictly_pre_start_observations():
    predictions = _pred_log().iloc[:1].copy()
    odds_history = pd.DataFrame([
        {
            "match_uid": "m1", "logged_at": "2026-07-01T09:00:00Z",
            "match_start_at_utc": "2026-07-01T10:30:00Z",
            "market_p1_prob": 0.40, "p1_odds_decimal": 2.40,
            "p2_odds_decimal": 1.60,
        },
        {
            "match_uid": "m1", "logged_at": "2026-07-01T10:00:00Z",
            "match_start_at_utc": "2026-07-01T10:30:00Z",
            "market_p1_prob": 0.55, "p1_odds_decimal": 1.80,
            "p2_odds_decimal": 2.10,
        },
        {
            "match_uid": "m1", "logged_at": "2026-07-01T10:30:00Z",
            "match_start_at_utc": "2026-07-01T10:30:00Z",
            "market_p1_prob": 0.70, "p1_odds_decimal": 1.40,
            "p2_odds_decimal": 3.00,
        },
        {
            "match_uid": "m1", "logged_at": "2026-07-01T11:00:00Z",
            "match_start_at_utc": "2026-07-01T10:30:00Z",
            "market_p1_prob": 0.90, "p1_odds_decimal": 1.10,
            "p2_odds_decimal": 8.00,
        },
    ])

    scored = cohorts.build_scored_frame(predictions, None, odds_history)
    timing = scored[scored["model"].isin(["market_open", "market_close"])]

    assert timing.set_index("model")["p1_prob"].to_dict() == {
        "market_open": 0.40,
        "market_close": 0.55,
    }
    assert timing.set_index("model")["prediction_time"].to_dict() == {
        "market_open": pd.Timestamp("2026-07-01T09:00:00Z"),
        "market_close": pd.Timestamp("2026-07-01T10:00:00Z"),
    }


def test_market_timing_requires_two_distinct_pre_start_observations():
    predictions = _pred_log().iloc[:1].copy()
    odds_history = pd.DataFrame([
        {
            "match_uid": "m1", "logged_at": "2026-07-01T09:00:00Z",
            "match_start_at_utc": "2026-07-01T10:00:00Z",
            "market_p1_prob": 0.45,
        },
        # A duplicate capture timestamp and a post-start row cannot manufacture
        # the two-observation opening/last-pre-start comparison.
        {
            "match_uid": "m1", "logged_at": "2026-07-01T09:00:00Z",
            "match_start_at_utc": "2026-07-01T10:00:00Z",
            "market_p1_prob": 0.46,
        },
        {
            "match_uid": "m1", "logged_at": "2026-07-01T10:01:00Z",
            "match_start_at_utc": "2026-07-01T10:00:00Z",
            "market_p1_prob": 0.60,
        },
    ])

    scored = cohorts.build_scored_frame(predictions, None, odds_history)

    assert not scored["model"].isin(["market_open", "market_close"]).any()


def test_market_timing_legacy_bovada_start_is_interpreted_in_eastern_time():
    predictions = _pred_log().iloc[:1].copy()
    odds_history = pd.DataFrame([
        {
            "match_uid": "m1", "logged_at": "2026-07-01T13:00:00Z",
            "match_start_time": "2026-07-01 10:00:00",
            "market_p1_prob": 0.42,
        },
        {
            "match_uid": "m1", "logged_at": "2026-07-01T13:59:00Z",
            "match_start_time": "2026-07-01 10:00:00",
            "market_p1_prob": 0.48,
        },
        # July is EDT, so the naive 10:00 display time is 14:00 UTC. A capture
        # exactly at that instant is not part of immutable pre-start evidence.
        {
            "match_uid": "m1", "logged_at": "2026-07-01T14:00:00Z",
            "match_start_time": "2026-07-01 10:00:00",
            "market_p1_prob": 0.51,
        },
    ])

    scored = cohorts.build_scored_frame(predictions, None, odds_history)
    timing = scored[scored["model"].isin(["market_open", "market_close"])]

    assert timing.set_index("model")["p1_prob"].to_dict() == {
        "market_open": 0.42,
        "market_close": 0.48,
    }


def test_market_timing_prefers_explicit_utc_scrape_clock_over_naive_logged_at():
    predictions = _pred_log().iloc[:1].copy()
    odds_history = pd.DataFrame([
        {
            "match_uid": "m1", "logged_at": "2026-07-01T15:00:00",
            "odds_scraped_at": "2026-07-01T13:00:00Z",
            "match_start_at_utc": "2026-07-01T14:00:00Z",
            "market_p1_prob": 0.43,
        },
        {
            "match_uid": "m1", "logged_at": "2026-07-01T15:59:00",
            "odds_scraped_at": "2026-07-01T13:59:00Z",
            "match_start_at_utc": "2026-07-01T14:00:00Z",
            "market_p1_prob": 0.49,
        },
        # A misleading naive logger clock looks pre-start when coerced to UTC,
        # but the authoritative scrape clock is exactly at start and excluded.
        {
            "match_uid": "m1", "logged_at": "2026-07-01T12:30:00",
            "odds_scraped_at": "2026-07-01T14:00:00Z",
            "match_start_at_utc": "2026-07-01T14:00:00Z",
            "market_p1_prob": 0.70,
        },
    ])

    scored = cohorts.build_scored_frame(predictions, None, odds_history)
    timing = scored[scored["model"].isin(["market_open", "market_close"])]

    assert timing.set_index("model")["p1_prob"].to_dict() == {
        "market_open": 0.43,
        "market_close": 0.49,
    }
    assert timing["prediction_time"].max() < pd.Timestamp("2026-07-01T14:00:00Z")


def test_kalshi_asks_join_only_on_complete_exact_run_and_match_pair():
    predictions = _pred_log().iloc[:1].copy()
    predictions["run_id"] = "run_exact"
    kalshi_history = pd.DataFrame([
        {
            "kalshi_observation_uid": "k1", "match_uid": "m1",
            "run_id": "run_exact", "polled_at": "2026-07-01T09:00:00Z",
            "event_ticker": "event_1", "market_ticker": "market_p1",
            "board_side": "p1", "yes_ask_dollars": "0.6100",
            "match_status": "matched",
        },
        {
            "kalshi_observation_uid": "k2", "match_uid": "m1",
            "run_id": "run_exact", "polled_at": "2026-07-01T09:00:00Z",
            "event_ticker": "event_1", "market_ticker": "market_p2",
            "board_side": "p2", "yes_ask_dollars": "0.4000",
            "match_status": "matched",
        },
        # Same match but another run must never be borrowed by run_exact.
        {
            "kalshi_observation_uid": "k3", "match_uid": "m1",
            "run_id": "run_other", "polled_at": "2026-07-01T09:01:00Z",
            "event_ticker": "event_1", "market_ticker": "other_p1",
            "board_side": "p1", "yes_ask_dollars": "0.1000",
            "match_status": "matched",
        },
        {
            "kalshi_observation_uid": "k4", "match_uid": "m1",
            "run_id": "run_other", "polled_at": "2026-07-01T09:01:00Z",
            "event_ticker": "event_1", "market_ticker": "other_p2",
            "board_side": "p2", "yes_ask_dollars": "0.9000",
            "match_status": "matched",
        },
    ])

    scored = cohorts.build_scored_frame(
        predictions, None, kalshi_history=kalshi_history,
    )

    assert scored["run_id"].eq("run_exact").all()
    assert scored["kalshi_p1_ask"].eq(0.61).all()
    assert scored["kalshi_p2_ask"].eq(0.40).all()
    assert scored["kalshi_observation_at"].eq(
        pd.Timestamp("2026-07-01T09:00:00Z")
    ).all()


def test_kalshi_price_join_rejects_one_sided_or_unmatched_observations():
    predictions = _pred_log().iloc[:1].copy()
    predictions["run_id"] = "run_exact"
    history = pd.DataFrame([
        {
            "match_uid": "m1", "run_id": "run_exact",
            "polled_at": "2026-07-01T09:00:00Z", "event_ticker": "event_1",
            "market_ticker": "market_p1", "board_side": "p1",
            "yes_ask_dollars": "0.6100", "match_status": "matched",
        },
        {
            "match_uid": "m1", "run_id": "run_exact",
            "polled_at": "2026-07-01T09:00:00Z", "event_ticker": "event_1",
            "market_ticker": "market_p2", "board_side": "p2",
            "yes_ask_dollars": "0.4000", "match_status": "ambiguous_board_match",
        },
    ])

    scored = cohorts.build_scored_frame(
        predictions, None, kalshi_history=history,
    )

    assert scored["kalshi_p1_ask"].isna().all()
    assert scored["kalshi_p2_ask"].isna().all()
