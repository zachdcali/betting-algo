#!/usr/bin/env python3
"""
Model Inference for Live Tennis Betting
Loads NN-141 model and generates calibrated predictions
"""

import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
try:
    from models.nn_runtime import NNWrapper, TennisNet
    from models.registry_utils import (
        load_registry,
        get_current_version,
        get_model_entry,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .nn_runtime import NNWrapper, TennisNet
    from .registry_utils import (
        load_registry,
        get_current_version,
        get_model_entry,
    )

_REGISTRY = load_registry()
MODEL_VERSION = get_current_version("nn", _REGISTRY)
XGB_MODEL_VERSION = get_current_version("xgboost", _REGISTRY)
RF_MODEL_VERSION = get_current_version("random_forest", _REGISTRY)
warnings.filterwarnings('ignore')


def _registry_model_entry(section: str, version: str) -> Dict:
    """Return a model-registry entry or an empty dict."""
    family = "nn" if section == "nn" else section
    return get_model_entry(family, version=version, registry=_REGISTRY, include_candidates=True)

# Exact 141 features for NN-141 model (Peak_Age_P1/P2 removed, consolidated to P1_Peak_Age/P2_Peak_Age)
EXACT_141_FEATURES = [
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
    'Round_Q4','Level_G','Round_ER','Level_S','Round_BR',
    'Round_Q3','Rank_Ratio','P1_Country_SUI','Clay_Season','P1_Country_GER',
    'P2_Rank_Change_30d','P1_Country_ESP','P2_Hand_A','H2H_Recent_P1_Advantage',
    'P2_Country_AUS','P2_Country_SRB','P2_Country_GBR','P2_Country_ARG',
    'Handedness_Matchup_RR','P1_Rank_Change_30d','P2_Country_GER','Handedness_Matchup_LL',
    'P2_Country_RUS','P1_Country_ARG','Level_C','P2_Semifinals_WinRate',
    'P2_Days_Since_Last','P1_Peak_Age','P2_Peak_Age','H2H_P1_WinRate',
    'P1_Country_Other','H2H_P1_Wins','P1_BigMatch_WinRate','P2_Rank_Change_90d',
    'P2_BigMatch_WinRate','P2_Country_FRA',
]


class TennisPredictor:
    """Live tennis match predictor using NN-141 model"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Resolve relative to this file: production/models/ -> ../../results/...
            model_dir = str(Path(__file__).parent.parent.parent / "results" / "professional_tennis" / "Neural_Network")
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.calibrated_model = None
        self.runtime_model_version = MODEL_VERSION
        self.probability_source = "raw"
        self.feature_names = EXACT_141_FEATURES
        self.is_loaded = False

    def load_model(self) -> bool:
        """Load the current NN model, optionally using registry-enabled calibration."""
        try:
            entry = _registry_model_entry("nn", MODEL_VERSION)
            model_path = self.model_dir / entry.get("model_file", "neural_network_model_SURFACE_FIX.pth")
            scaler_path = self.model_dir / entry.get("scaler_file", "scaler_SURFACE_FIX.pkl")
            calibrated_path = self.model_dir / entry.get("calibrated_model_file", "neural_network_calibrated_SURFACE_FIX.pkl")
            probability_mode = entry.get("probability_mode", "raw")

            if not model_path.exists():
                print(f"❌ Model file not found: {model_path}")
                return False

            if not scaler_path.exists():
                print(f"❌ Scaler file not found: {scaler_path}")
                return False

            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            # Initialize and load model
            input_size = self.scaler.n_features_in_  # 141
            self.model = TennisNet(input_size)

            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()

            self.calibrated_model = None
            self.runtime_model_version = MODEL_VERSION
            self.probability_source = "raw"
            if probability_mode == "calibrated" and calibrated_path.exists():
                try:
                    with open(calibrated_path, 'rb') as f:
                        self.calibrated_model = pickle.load(f)
                    self.probability_source = "calibrated"
                    print(f"✅ Calibrated NN probabilities loaded from {calibrated_path.name}")
                except Exception as e:
                    print(f"⚠️  Failed to load calibrated NN artifact {calibrated_path}: {e}")
            elif probability_mode == "calibrated":
                print(f"⚠️  Registry requested calibrated NN probabilities but artifact was not found: {calibrated_path}")

            self.is_loaded = True
            print(f"✅ NN model loaded ({input_size} features)")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_match_probability(self, features_dict: Dict) -> Dict:
        """
        Predict match probability for a single match
        
        Args:
            features_dict: Dictionary with 141 features
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        try:
            # Extract the 141 features in the exact order the model was trained on
            feature_values = [float(features_dict.get(f, 0.0)) for f in EXACT_141_FEATURES]

            X = np.array(feature_values).reshape(1, -1)
            if self.calibrated_model is not None:
                raw_prob = float(self.calibrated_model.predict_proba(X)[:, 1][0])
            else:
                X_scaled = self.scaler.transform(X)
                with torch.no_grad():
                    raw_prob = float(self.model(torch.FloatTensor(X_scaled)).squeeze().numpy())

            return {
                "player1_win_prob": raw_prob,
                "player2_win_prob": 1.0 - raw_prob,
                "raw_prob": raw_prob,
                "model_version": self.runtime_model_version,
                "probability_source": self.probability_source,
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def predict_slate(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probabilities for all matches in a slate
        
        Args:
            features_df: DataFrame with features for each match
            
        Returns:
            DataFrame with predictions added
        """
        if not self.is_loaded:
            if not self.load_model():
                return features_df
        
        predictions = []
        
        for idx, row in features_df.iterrows():
            # Skip matches where feature extraction failed in strict mode
            if row.get('status') == 'skip':
                print(f"⏭️ Skipping prediction due to missing {row.get('missing_category', 'unknown')} data: {row.get('player1_raw', 'Unknown')} vs {row.get('player2_raw', 'Unknown')}")
                predictions.append({
                    **row.to_dict(),
                    "player1_win_prob": None,
                    "player2_win_prob": None,
                    "prediction_status": "skipped_missing_data"
                })
                continue
                
            print(f"🎯 Predicting: {row.get('player1_raw', 'Unknown')} vs {row.get('player2_raw', 'Unknown')}")
            
            # Extract features for this match
            features_dict = row.to_dict()
            pred_result = self.predict_match_probability(features_dict)
            
            # Add prediction results to row
            if "error" not in pred_result:
                predictions.append({
                    **row.to_dict(),
                    **pred_result,
                    "prediction_status": "success"
                })
            else:
                print(f"⚠️  Prediction error: {pred_result['error']}")
                predictions.append({
                    **row.to_dict(),
                    "player1_win_prob": None,
                    "player2_win_prob": None,
                    "error": pred_result["error"],
                    "prediction_status": "failed"
                })
        
        return pd.DataFrame(predictions)

class XGBoostPredictor:
    """Live tennis match predictor using XGBoost SURFACE_FIX model (141 features)"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(Path(__file__).parent.parent.parent / "results" / "professional_tennis" / "XGBoost")
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self) -> bool:
        try:
            import xgboost as xgb
            entry = _registry_model_entry("xgboost", XGB_MODEL_VERSION)
            model_path = self.model_dir / entry.get("model_file", "xgboost_model_SURFACE_FIX.json")
            if not model_path.exists():
                print(f"❌ XGBoost model not found: {model_path}")
                return False
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path))
            self.feature_names = list(self.model.feature_names_in_)
            self.is_loaded = True
            print(f"✅ XGBoost model loaded ({len(self.feature_names)} features)")
            return True
        except Exception as e:
            print(f"❌ Error loading XGBoost model: {e}")
            return False

    def predict_match_probability(self, features_dict: Dict) -> Dict:
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "XGBoost model not loaded"}
        try:
            feat_values = {f: float(features_dict.get(f, 0.0)) for f in self.feature_names}
            X = pd.DataFrame([feat_values])[self.feature_names].fillna(0.0).astype(float)
            prob = float(self.model.predict_proba(X)[0, 1])
            return {"xgb_p1_prob": prob, "xgb_p2_prob": 1.0 - prob}
        except Exception as e:
            return {"error": f"XGBoost prediction failed: {e}"}


class RandomForestPredictor:
    """Live tennis match predictor using Random Forest SURFACE_FIX model (141 features)"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(Path(__file__).parent.parent.parent / "results" / "professional_tennis" / "Random_Forest")
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self) -> bool:
        try:
            entry = _registry_model_entry("random_forest", RF_MODEL_VERSION)
            model_path = self.model_dir / entry.get("model_file", "random_forest_model_SURFACE_FIX.pkl")
            if not model_path.exists():
                print(f"❌ Random Forest model not found: {model_path}")
                return False
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.feature_names = list(self.model.feature_names_in_)
            self.is_loaded = True
            print(f"✅ Random Forest model loaded ({len(self.feature_names)} features)")
            return True
        except Exception as e:
            print(f"❌ Error loading Random Forest model: {e}")
            return False

    def predict_match_probability(self, features_dict: Dict) -> Dict:
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Random Forest model not loaded"}
        try:
            feat_values = {f: float(features_dict.get(f, 0.0)) for f in self.feature_names}
            X = pd.DataFrame([feat_values])[self.feature_names].fillna(0.0).astype(float)
            prob = float(self.model.predict_proba(X)[0, 1])
            return {"rf_p1_prob": prob, "rf_p2_prob": 1.0 - prob}
        except Exception as e:
            return {"error": f"Random Forest prediction failed: {e}"}


def calculate_betting_edges(predictions_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate betting edges by comparing model probabilities to market odds
    
    Args:
        predictions_df: DataFrame with model predictions
        odds_df: DataFrame with market odds
        
    Returns:
        DataFrame with edges calculated
    """
    # Merge predictions with odds
    merged_df = predictions_df.merge(
        odds_df[['player1_raw', 'player2_raw', 'player1_odds_decimal', 'player2_odds_decimal', 
                'player1_implied_prob', 'player2_implied_prob', 'event', 'match_time']],
        on=['player1_raw', 'player2_raw'],
        how='left'
    )
    
    # Calculate edges
    merged_df['edge_player1'] = merged_df['player1_win_prob'] - merged_df['player1_implied_prob']
    merged_df['edge_player2'] = merged_df['player2_win_prob'] - merged_df['player2_implied_prob']
    
    # Determine best bet for each match
    merged_df['best_edge'] = merged_df[['edge_player1', 'edge_player2']].max(axis=1)
    merged_df['bet_on_player1'] = merged_df['edge_player1'] >= merged_df['edge_player2']
    
    # Add betting details
    merged_df['bet_player'] = merged_df.apply(
        lambda row: row['player1_raw'] if row['bet_on_player1'] else row['player2_raw'], axis=1
    )
    merged_df['bet_odds'] = merged_df.apply(
        lambda row: row['player1_odds_decimal'] if row['bet_on_player1'] else row['player2_odds_decimal'], axis=1
    )
    merged_df['bet_prob'] = merged_df.apply(
        lambda row: row['player1_win_prob'] if row['bet_on_player1'] else row['player2_win_prob'], axis=1
    )
    merged_df['market_prob'] = merged_df.apply(
        lambda row: row['player1_implied_prob'] if row['bet_on_player1'] else row['player2_implied_prob'], axis=1
    )
    
    return merged_df

def main():
    """Test the model inference"""
    predictor = TennisPredictor()

    if predictor.load_model():
        print("✅ Model loading test passed")

        sample_features = {f: np.random.rand() for f in EXACT_141_FEATURES}
        result = predictor.predict_match_probability(sample_features)
        if "error" not in result:
            print("✅ Prediction test passed")
            print(f"   Player 1 win prob: {result['player1_win_prob']:.3f}")
            print(f"   Player 2 win prob: {result['player2_win_prob']:.3f}")
        else:
            print(f"❌ Prediction test failed: {result['error']}")
    else:
        print("❌ Model loading test failed")

if __name__ == "__main__":
    main()
