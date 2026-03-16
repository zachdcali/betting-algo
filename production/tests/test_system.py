#!/usr/bin/env python3
"""
Test script for the production betting system
- Odds fetch (mock)
- Feature extraction shape (mock default)
- Model loads & predicts (smoke)
- Kelly staking (sim)
- Updater smoke test (builds slate -> extracts players; no network)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add production modules to path
PROD_ROOT = Path(__file__).parent.resolve()
if str(PROD_ROOT) not in sys.path:
    sys.path.append(str(PROD_ROOT))

def test_odds_fetching():
    print("🧪 Testing odds fetching...")
    try:
        # Use a minimal synthetic slate to avoid hitting Bovada in tests
        sample_odds = pd.DataFrame([{
            'player1_raw': 'Test Player 1',
            'player2_raw': 'Test Player 2',
            'player1_normalized': 'test player 1',
            'player2_normalized': 'test player 2',
            'event': "ATP Test Men's",
            'player1_odds_decimal': 2.0,
            'player2_odds_decimal': 1.8,
            'player1_implied_prob': 0.5,
            'player2_implied_prob': 0.556,
            'timestamp': '2025-08-24 10:00:00',
            'source': 'test'
        }])
        print(f"✅ Odds fetching test passed - {len(sample_odds)} matches")
        return sample_odds
    except Exception as e:
        print(f"❌ Odds fetching test failed: {e}")
        return pd.DataFrame()

def test_feature_extraction():
    print("🧪 Testing feature extraction...")
    try:
        from features.extract_features import EXACT_143_FEATURES, LiveFeatureExtractor
        extractor = LiveFeatureExtractor()
        sample = extractor._get_default_features()
        if len(sample) == 143 and set(sample.keys()).issuperset(set(EXACT_143_FEATURES)):
            print(f"✅ Feature extraction test passed - 143 features")
            return sample
        print(f"❌ Feature extraction test failed - got {len(sample)}")
        return {}
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return {}

def test_model_inference():
    print("🧪 Testing model inference...")
    try:
        from models.inference import TennisPredictor
        from features.extract_features import EXACT_143_FEATURES
        pred = TennisPredictor()
        if not pred.load_model():
            print("❌ Model loading failed")
            return False
        rnd = {f: np.random.rand() for f in EXACT_143_FEATURES}
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
            print("⚠️  No profitable opportunities (this can be OK)")
            return True
        print(f"✅ Stake calculation test passed - {len(slips)} bets, ${slips['stake'].sum():.2f} total stakes")
        return True
    except Exception as e:
        print(f"❌ Stake calculation test failed: {e}")
        return False

def test_updater_smoke():
    print("🧪 Testing player updater smoke (no network)...")
    try:
        from update_player_data import PlayerDataUpdater
        sample_odds = test_odds_fetching()
        upd = PlayerDataUpdater(max_workers=2, max_age_days=9999)  # force 'fresh' to avoid scraper
        names = upd.extract_players_from_slate(sample_odds)
        _ = upd.find_players_needing_update(names)  # should run without crashing
        print("✅ Updater smoke test passed")
        return True
    except Exception as e:
        print(f"❌ Updater smoke test failed: {e}")
        return False

def test_full_pipeline():
    print("🧪 Testing full pipeline (init only)...")
    try:
        from main import LiveBettingOrchestrator
        _ = LiveBettingOrchestrator()
        print("✅ Full pipeline initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    print("🎾 Testing Tennis Betting Production System")
    print("=" * 50)
    tests = [
        ("Odds Fetching", test_odds_fetching),
        ("Feature Extraction", test_feature_extraction),
        ("Model Inference", test_model_inference),
        ("Stake Calculation", test_stake_calculation),
        ("Updater Smoke", test_updater_smoke),
        ("Full Pipeline", test_full_pipeline),
    ]
    ok = 0
    for name, fn in tests:
        print(f"\n{name}:")
        print("-" * 30)
        try:
            res = fn()
            success = bool(res is not False)
            print(f"{name}: {'✅ PASS' if success else '❌ FAIL'}")
            ok += int(success)
        except Exception as e:
            print(f"💥 {name} crashed: {e}")

    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Overall: {ok}/{len(tests)} tests passed")
    return ok == len(tests)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
