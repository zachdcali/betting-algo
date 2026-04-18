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

            if len(log_df) != 1 or len(snap_df) != 1 or len(odds_df) != 1:
                print("❌ Logger did not create exactly one row in each output")
                return False
            if 'match_uid' not in log_df.columns or 'prediction_uid' not in log_df.columns:
                print("❌ Logger missing immutable ID columns")
                return False

        print("✅ Prediction logger test passed")
        return True
    except Exception as e:
        print(f"❌ Prediction logger test failed: {e}")
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


def main():
    print("🎾 Testing Tennis Betting Production System")
    print("=" * 50)
    tests = [
        ("Schema Contract", test_schema_contract),
        ("Round Offsets", test_round_offsets),
        ("Model Inference", test_model_inference),
        ("Stake Calculation", test_stake_calculation),
        ("Prediction Logger", test_prediction_logger),
        ("Orchestrator Init", test_orchestrator_init),
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
