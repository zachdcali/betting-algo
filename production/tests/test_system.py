#!/usr/bin/env python3
"""
Smoke tests for the production betting system.

These tests are intentionally lightweight:
- no Bovada scraping
- no Tennis Abstract scraping
- local model loading only
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROD_ROOT = Path(__file__).resolve().parents[1]
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))


def test_schema_contract():
    print("🧪 Testing schema contract...")
    try:
        from models.inference import EXACT_141_FEATURES as model_features
        from features.ta_feature_calculator import EXACT_141_FEATURES as ta_features

        if len(model_features) != 141 or len(ta_features) != 141:
            print(f"❌ Expected 141 features, got model={len(model_features)} ta={len(ta_features)}")
            return False
        if list(model_features) != list(ta_features):
            print("❌ Feature order mismatch between inference and TA calculator")
            return False
        print("✅ Schema contract test passed")
        return True
    except Exception as e:
        print(f"❌ Schema contract test failed: {e}")
        return False


def test_round_offsets():
    print("🧪 Testing round offsets...")
    try:
        from features.round_offsets import get_round_day_offset

        off = get_round_day_offset('M', 96, 'F', pd.Timestamp('2026-03-18'))
        if off != 11:
            print(f"❌ Unexpected Miami final offset: {off}")
            return False
        print("✅ Round offset test passed")
        return True
    except Exception as e:
        print(f"❌ Round offset test failed: {e}")
        return False


def test_model_inference():
    print("🧪 Testing model inference...")
    try:
        from models.inference import TennisPredictor, EXACT_141_FEATURES

        pred = TennisPredictor()
        if not pred.load_model():
            print("❌ Model loading failed")
            return False

        rnd = {f: np.random.rand() for f in EXACT_141_FEATURES}
        out = pred.predict_match_probability(rnd)
        if "error" in out:
            print(f"❌ Inference error: {out['error']}")
            return False
        print(f"✅ Model inference test passed - P1: {out['player1_win_prob']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Model inference test failed: {e}")
        return False


def test_stake_calculation():
    print("🧪 Testing stake calculation...")
    try:
        from utils.stake_calculator import simulate_daily_betting

        slips = simulate_daily_betting(bankroll=1000.0, kelly_multiplier=0.18)
        if slips is None or slips.empty:
            print("⚠️ No profitable opportunities in synthetic simulation")
            return True
        required = {'match_uid', 'match_date', 'match_start_time', 'stake', 'edge'}
        if not required.issubset(set(slips.columns)):
            print(f"❌ Missing expected bet slip columns: {sorted(required - set(slips.columns))}")
            return False
        print(f"✅ Stake calculation test passed - {len(slips)} bets")
        return True
    except Exception as e:
        print(f"❌ Stake calculation test failed: {e}")
        return False


def test_prediction_logger():
    print("🧪 Testing prediction logger...")
    try:
        import prediction_logger

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prediction_logger.LOG_PATH = str(tmp_path / "prediction_log.csv")
            prediction_logger.SNAPSHOT_LOG_PATH = tmp_path / "prediction_snapshots.csv"
            prediction_logger.ODDS_HISTORY_LOG_PATH = tmp_path / "odds_history.csv"

            prediction_logger.log_prediction(
                p1="Player One",
                p2="Player Two",
                tournament="Test Event",
                surface="Hard",
                level="A",
                round_code="R32",
                match_date="2026-04-18",
                run_id="run_test",
                match_uid="match_test",
                feature_snapshot_id="feat_test",
                model_p1_prob=0.61,
                model_p2_prob=0.39,
                market_p1_prob=0.55,
                market_p2_prob=0.45,
                p1_odds_decimal=1.82,
                p2_odds_decimal=2.10,
                model_version="v-test",
                odds_scraped_at="2026-04-18T12:00:00+00:00",
                match_start_time="2026-04-18 09:00 AM",
            )

            log_df = pd.read_csv(prediction_logger.LOG_PATH)
            snap_df = pd.read_csv(prediction_logger.SNAPSHOT_LOG_PATH)
            odds_df = pd.read_csv(prediction_logger.ODDS_HISTORY_LOG_PATH)
            upgraded = prediction_logger.upgrade_prediction_log(Path(prediction_logger.LOG_PATH), write=False)

            if len(log_df) != 1 or len(snap_df) != 1 or len(odds_df) != 1:
                print("❌ Logger did not create exactly one row in each output")
                return False
            required = {
                'match_uid', 'prediction_uid', 'logging_quality', 'rescore_quality',
                'record_status', 'nn_model_version'
            }
            if not required.issubset(set(upgraded.columns)):
                print(f"❌ Logger missing metadata columns: {sorted(required - set(upgraded.columns))}")
                return False
            if upgraded.loc[0, 'logging_quality'] != 'snapshot_v2':
                print(f"❌ Expected snapshot_v2 logging quality, got {upgraded.loc[0, 'logging_quality']}")
                return False
            if upgraded.loc[0, 'record_status'] != 'pending':
                print(f"❌ Expected pending record status, got {upgraded.loc[0, 'record_status']}")
                return False
            if 'match_uid' not in log_df.columns or 'prediction_uid' not in log_df.columns:
                print("❌ Logger missing immutable ID columns")
                return False

        print("✅ Prediction logger test passed")
        return True
    except Exception as e:
        print(f"❌ Prediction logger test failed: {e}")
        return False


def test_audit_logger():
    print("🧪 Testing audit logger...")
    try:
        import audit_logger

        orig_audit_dir = audit_logger.AUDIT_DIR
        orig_skipped = audit_logger.SKIPPED_MATCHES_LOG_PATH
        orig_settlement = audit_logger.SETTLEMENT_AUDIT_LOG_PATH
        orig_run_history = audit_logger.RUN_HISTORY_LOG_PATH
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                audit_logger.AUDIT_DIR = tmp_path
                audit_logger.SKIPPED_MATCHES_LOG_PATH = tmp_path / "skipped_live_matches.csv"
                audit_logger.SETTLEMENT_AUDIT_LOG_PATH = tmp_path / "settlement_audit.csv"
                audit_logger.RUN_HISTORY_LOG_PATH = tmp_path / "run_history.csv"

                audit_logger.log_skipped_live_match(
                    run_id="run_test",
                    run_started_at="2026-04-22T01:00:00+00:00",
                    stage="feature_extraction",
                    skip_reason_code="scheduled_start_passed",
                    skip_reason_detail="scheduled_start_passed",
                    match_uid="match_test",
                    feature_snapshot_id="feat_test",
                    match_date="2026-04-22",
                    match_start_time="4/22/26 8:00 AM",
                    tournament="Munich",
                    event_title="ATP Munich - Semifinal",
                    surface="Clay",
                    level="A",
                    round_code="SF",
                    p1="Player One",
                    p2="Player Two",
                )
                audit_logger.log_settlement_event(
                    run_id="settle_test",
                    dry_run=False,
                    row_index=7,
                    match_uid="match_test",
                    prediction_uid="pred_test",
                    match_date="2026-04-22",
                    tournament="Munich",
                    round_code="SF",
                    p1="Player One",
                    p2="Player Two",
                    ta_player_slug="PlayerOne",
                    outcome_code="matched_and_settled",
                    outcome_detail="Player One won",
                    ta_match_date_found="2026-04-20",
                    actual_winner=1,
                    score="6-4 6-4",
                )
                audit_logger.upsert_run_history({
                    "run_id": "run_test",
                    "run_kind": "prediction_pipeline",
                    "started_at": "2026-04-22T01:00:00+00:00",
                    "completed_at": "2026-04-22T01:05:00+00:00",
                    "status": "success",
                    "feature_rows_ok": 12,
                    "feature_rows_skipped": 3,
                    "feature_skip_reason_summary": {"scheduled_start_passed": 2},
                })

                skipped_df = pd.read_csv(audit_logger.SKIPPED_MATCHES_LOG_PATH)
                settlement_df = pd.read_csv(audit_logger.SETTLEMENT_AUDIT_LOG_PATH)
                run_df = pd.read_csv(audit_logger.RUN_HISTORY_LOG_PATH)
                if len(skipped_df) != 1 or len(settlement_df) != 1 or len(run_df) != 1:
                    print("❌ Audit logger did not create exactly one row in each audit file")
                    return False
                if skipped_df.loc[0, "skip_reason_code"] != "scheduled_start_passed":
                    print(f"❌ Unexpected skip reason code: {skipped_df.loc[0, 'skip_reason_code']}")
                    return False
                if settlement_df.loc[0, "outcome_code"] != "matched_and_settled":
                    print(f"❌ Unexpected settlement outcome: {settlement_df.loc[0, 'outcome_code']}")
                    return False
                if run_df.loc[0, "status"] != "success":
                    print(f"❌ Unexpected run history status: {run_df.loc[0, 'status']}")
                    return False
        finally:
            audit_logger.AUDIT_DIR = orig_audit_dir
            audit_logger.SKIPPED_MATCHES_LOG_PATH = orig_skipped
            audit_logger.SETTLEMENT_AUDIT_LOG_PATH = orig_settlement
            audit_logger.RUN_HISTORY_LOG_PATH = orig_run_history

        print("✅ Audit logger test passed")
        return True
    except Exception as e:
        print(f"❌ Audit logger test failed: {e}")
        return False


def test_orchestrator_init():
    print("🧪 Testing orchestrator initialization...")
    try:
        from main import LiveBettingOrchestrator

        _ = LiveBettingOrchestrator()
        print("✅ Orchestrator initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Orchestrator initialization test failed: {e}")
        return False


def test_inference_guardrails():
    print("🧪 Testing inference guardrails...")
    try:
        from main import LiveBettingOrchestrator
        from features.ta_feature_calculator import TAFeatureCalculator

        orch = LiveBettingOrchestrator()
        parsed = orch.parse_match_start_datetime("4/21/26 7:30 PM", now=datetime(2026, 4, 21, 12, 0))
        if parsed != datetime(2026, 4, 21, 19, 30):
            print(f"❌ Unexpected parsed match start: {parsed}")
            return False

        _, guard_reason = orch.get_inference_guard_reason("4/21/26 7:30 PM")
        if guard_reason not in {"", "inside_pre_match_buffer_5m", "scheduled_start_passed"}:
            print(f"❌ Unexpected guard reason: {guard_reason}")
            return False

        calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
        completed_rows = pd.DataFrame([
            {
                "date": "2026-04-13",
                "event": "Munich",
                "round": "SF",
                "opp_name": "Flavio Cobolli",
                "result": "W",
            },
            {
                "date": "2026-03-10",
                "event": "Indian Wells",
                "round": "R64",
                "opp_name": "Flavio Cobolli",
                "result": "W",
            },
        ])
        candidate = calc._find_completed_match_candidate(
            completed_rows,
            opponent_name="Flavio Cobolli",
            ref_date=datetime(2026, 4, 18, 4, 30),
            round_code="SF",
            expected_event_title="ATP Munich - Semifinal",
        )
        if not candidate or candidate.get("event") != "Munich":
            print(f"❌ Failed to identify likely completed current match: {candidate}")
            return False

        safe_rows = pd.DataFrame([
            {
                "date": "2026-02-01",
                "event": "Montpellier",
                "round": "R32",
                "opp_name": "Flavio Cobolli",
                "result": "W",
            }
        ])
        safe_candidate = calc._find_completed_match_candidate(
            safe_rows,
            opponent_name="Flavio Cobolli",
            ref_date=datetime(2026, 4, 18, 4, 30),
            round_code="SF",
            expected_event_title="ATP Munich - Semifinal",
        )
        if safe_candidate is not None:
            print(f"❌ False positive completed-match candidate: {safe_candidate}")
            return False

        print("✅ Inference guardrails test passed")
        return True
    except Exception as e:
        print(f"❌ Inference guardrails test failed: {e}")
        return False


def main():
    print("🎾 Testing Tennis Betting Production System")
    print("=" * 50)
    tests = [
        ("Schema Contract", test_schema_contract),
        ("Round Offsets", test_round_offsets),
        ("Model Inference", test_model_inference),
        ("Stake Calculation", test_stake_calculation),
        ("Prediction Logger", test_prediction_logger),
        ("Audit Logger", test_audit_logger),
        ("Orchestrator Init", test_orchestrator_init),
        ("Inference Guardrails", test_inference_guardrails),
    ]

    ok = 0
    for name, fn in tests:
        print(f"\n{name}:")
        print("-" * 30)
        try:
            res = bool(fn())
            print(f"{name}: {'✅ PASS' if res else '❌ FAIL'}")
            ok += int(res)
        except Exception as e:
            print(f"💥 {name} crashed: {e}")

    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Overall: {ok}/{len(tests)} tests passed")
    return ok == len(tests)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
