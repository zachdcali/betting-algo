#!/usr/bin/env python3
"""
Live Feature Extraction for Tennis Betting Models
Extracts the 143 features needed for NN-143 model from UTR + static data.

This module delegates all core logic to LiveFeatureEngine so both paths
(offline batch vs. live slate) stay perfectly consistent.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import hashlib

# Keep the canonical list in ONE place (engine); we still define here to enforce order in tests.
EXACT_143_FEATURES = [
    'P2_WinStreak_Current','P1_WinStreak_Current','P2_Surface_Matches_30d','Height_Diff',
    'P1_Surface_Matches_30d','Player2_Height','P1_Matches_30d','P2_Matches_30d',
    'P2_Surface_Experience','P2_Form_Trend_30d','Player1_Height','P1_Form_Trend_30d',
    'Round_R16','Surface_Transition_Flag','P1_Surface_Matches_90d','P1_Surface_Experience',
    'Rank_Diff','Round_R32','Rank_Points_Diff','P2_Level_WinRate_Career',
    'P2_Surface_Matches_90d','P1_Level_WinRate_Career','P2_Level_Matches_Career',
    'P2_WinRate_Last10_120d','Round_QF','Level_25','P1_Round_WinRate_Career',
    'P1_Surface_WinRate_90d','Round_Q1','Player1_Rank','P1_Level_Matches_Career',
    'P2_Round_WinRate_Career','draw_size','P1_WinRate_Last10_120d','Age_Diff',
    'Level_15','Player1_Rank_Points','Handedness_Matchup_RL','Player2_Rank',
    'Avg_Age','P1_Country_RUS','Player2_Age','P2_vs_Lefty_WinRate',
    'Round_F','Surface_Clay','P2_Sets_14d','Rank_Momentum_Diff_30d',
    'H2H_P2_Wins','Player2_Rank_Points','Player1_Age','P2_Rank_Volatility_90d',
    'P1_Days_Since_Last','Grass_Season','P1_Semifinals_WinRate','Level_A',
    'Level_D','P1_Country_USA','P1_Country_GBR','P1_Country_FRA',
    'P2_Matches_14d','P2_Country_USA','P2_Country_ITA','Round_Q2',
    'P2_Surface_WinRate_90d','P1_Hand_L','P2_Hand_L','P1_Country_ITA',
    'P2_Rust_Flag','P1_Rank_Change_90d','P1_Country_AUS','P1_Hand_U',
    'P1_Hand_R','Round_RR','Avg_Height','P1_Sets_14d',
    'P2_Country_Other','Round_SF','P1_vs_Lefty_WinRate','Indoor_Season',
    'Avg_Rank','P1_Rust_Flag','Avg_Rank_Points','Level_F',
    'Round_R64','P2_Country_CZE','P2_Hand_R','Surface_Hard',
    'P1_Matches_14d','Surface_Carpet','Round_R128','P1_Country_SRB',
    'P2_Hand_U','P1_Rank_Volatility_90d','Level_M','P2_Country_ESP',
    'Handedness_Matchup_LR','P1_Country_CZE','P2_Country_SUI','Surface_Grass',
    'H2H_Total_Matches','Level_O','P1_Hand_A','P1_Finals_WinRate',
    'Rank_Momentum_Diff_90d','P2_Finals_WinRate',
    'Round_Q4','Peak_Age_P1','Level_G','Round_ER','Level_S','Round_BR','Peak_Age_P2',
    'Round_Q3','Rank_Ratio','P1_Country_SUI','Clay_Season','P1_Country_GER',
    'P2_Rank_Change_30d','P1_Country_ESP','P2_Hand_A','H2H_Recent_P1_Advantage',
    'P2_Country_AUS','P2_Country_SRB','P2_Country_GBR','P2_Country_ARG',
    'Handedness_Matchup_RR','P1_Rank_Change_30d','P2_Country_GER','Handedness_Matchup_LL',
    'P2_Country_RUS','P1_Country_ARG','Level_C','P2_Semifinals_WinRate',
    'P2_Days_Since_Last','P1_Peak_Age','P2_Peak_Age','H2H_P1_WinRate',
    'P1_Country_Other','H2H_P1_Wins','P1_BigMatch_WinRate','P2_Rank_Change_90d',
    'P2_BigMatch_WinRate','P2_Country_FRA'
]

# Compute SHA1 fingerprint of feature schema for parity tracking
FEATURE_SCHEMA_SHA = hashlib.sha1('|'.join(EXACT_143_FEATURES).encode('utf-8')).hexdigest()

# Import from sibling module (ensure production/features is on path)
import sys
from pathlib import Path
features_dir = Path(__file__).parent
if str(features_dir) not in sys.path:
    sys.path.insert(0, str(features_dir))
from live_feature_engine import LiveFeatureEngine, MissingData

def extract_features_for_slate(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        odds_df columns required:
          ['player1_raw','player2_raw','player1_normalized','player2_normalized',
           'event','timestamp'] and optional ['surface','tournament_level','draw_size','round_code']
    """
    engine = LiveFeatureEngine()
    rows = []

    for idx, r in odds_df.iterrows():
        print(f"🔧 Extracting features for {r['player1_raw']} vs {r['player2_raw']}")
        try:
            features = engine.build_match_features(
                r['player1_normalized'],
                r['player2_normalized'],
                match_date=pd.to_datetime(r.get('timestamp')),
                surface=r.get('surface', 'Hard'),
                tournament_level=r.get('tournament_level', 'ATP'),
                draw_size=int(r.get('draw_size', 32) or 32),
                round_code=r.get('round_code'),
                strict=True
            )
            status = "ok"
            missing_category = ""
            missing_players = ""
        except MissingData as e:
            print(f"⚠️ Skipping match due to missing {e.category} data for: {', '.join(e.players)}")
            features = {}  # don't default-fill in strict mode
            status = "skip"
            missing_category = e.category
            missing_players = ",".join(e.players)

        ordered = {k: features.get(k, None) for k in EXACT_143_FEATURES}
        ordered.update({
            'match_id': idx,
            'player1_raw': r['player1_raw'],
            'player2_raw': r['player2_raw'],
            'event': r['event'],
            'timestamp': r['timestamp'],
            'status': status,
            'missing_category': missing_category,
            'missing_players': missing_players,
            'feature_schema_sha': FEATURE_SCHEMA_SHA
        })
        rows.append(ordered)

    out = pd.DataFrame(rows)
    # Print SHA for verification
    print(f"\n📊 Feature Schema SHA1: {FEATURE_SCHEMA_SHA}")
    print(f"   (143 features in canonical order)")
    # Don't raise on missing columns here—downstream will filter status=="ok"
    return out


def main():
    # tiny self-test
    sample = pd.DataFrame([{
        'player1_raw': 'Novak Djokovic',
        'player2_raw': 'Rafael Nadal',
        'player1_normalized': 'novak djokovic',
        'player2_normalized': 'rafael nadal',
        'event': 'ATP Masters 1000 - Indian Wells',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'surface': 'Hard',
        'tournament_level': 'ATP',
        'draw_size': 64,
        'round_code': 'R32'
    }])
    df = extract_features_for_slate(sample)
    print("✅ Feature extraction test completed")
    print("Shape:", df.shape)
    print("First 5 of 143:", df[[EXACT_143_FEATURES[i] for i in range(5)]].head())


if __name__ == "__main__":
    main()
