#!/usr/bin/env python3
"""
Generate PURE bet logs (no bankroll) from verified, leak-free base.

Inputs (from verify_identical_matches.py):
  - analysis_scripts/verified/ml_verified.csv
      * leak-free features + Player1_Name, Player2_Name, tourney_date, Player1_Wins
  - analysis_scripts/verified/tennis_verified.csv
      * Date, Winner, Loser, AvgW, AvgL

Process:
  - Load trained models (XGB, RF, NN-143, NN-98) and NN scalers
  - Compute P1 win probability per match for each model
  - Compute market probs from AvgW/AvgL and edges for both sides
  - If max(edge_p1, edge_p2) > EDGE_THRESHOLD, emit a bet row for that side
  - Save one CSV per model to analysis_scripts/pure_bet_logs/

Output schema (one row per qualifying bet):
  date, week_id, player1, player2, bet_on_player, bet_on_p1,
  prob, market_prob, edge, odds, outcome, model, avgw, avgl
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import xgboost as xgb

# === Exact 143-feature set used in training (order matters) ===
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

# "remove for 98" list you used during training (symmetric)
LOW_IMPORTANCE_REMOVE = [
    'P1_Country_FRA','P2_Country_FRA','P1_Country_RUS','P2_Country_RUS','P1_Country_ARG','P2_Country_ARG',
    'P1_Country_GER','P2_Country_GER','P1_Country_GBR','P2_Country_GBR','P1_Country_SRB','P2_Country_SRB',
    'P1_Country_AUS','P2_Country_AUS','P1_Country_ESP','P2_Country_ESP','P1_Country_SUI','P2_Country_SUI',
    'P1_Country_Other','P2_Country_Other','P1_BigMatch_WinRate','P2_BigMatch_WinRate','P1_Peak_Age','P2_Peak_Age',
    'P1_Semifinals_WinRate','P2_Semifinals_WinRate','P1_Days_Since_Last','P2_Days_Since_Last',
    'P1_Hand_A','P2_Hand_A','P1_Rank_Change_30d','P2_Rank_Change_30d','P2_Rank_Change_90d',
    'H2H_P1_Wins','H2H_P1_WinRate','H2H_Recent_P1_Advantage','Level_C','Clay_Season','Rank_Ratio','Round_Q3',
    'Handedness_Matchup_LL','Handedness_Matchup_RR','Round_Q4','Peak_Age_P1','Level_G','Round_ER','Level_S','Round_BR','Peak_Age_P2'
]

# --------------------------
# Config / Paths
# --------------------------
REPO = Path(__file__).resolve().parents[2]
VERIFIED_DIR = REPO / "analysis_scripts" / "verified"
ML_VERIFIED_PATH = VERIFIED_DIR / "ml_verified.csv"
TENNIS_VERIFIED_PATH = VERIFIED_DIR / "tennis_verified.csv"

# Model files — adjust if your repo uses different names
XGB_PATH       = REPO / "results" / "professional_tennis" / "XGBoost" / "xgboost_model.json"
RF_PATH        = REPO / "results" / "professional_tennis" / "Random_Forest" / "random_forest_model.pkl"
NN143_PATH     = REPO / "results" / "professional_tennis" / "Neural_Network" / "neural_network_143_features.pth"
SCALER143_PATH = REPO / "results" / "professional_tennis" / "Neural_Network" / "scaler_143_features.pkl"
NN98_PATH      = REPO / "results" / "professional_tennis" / "Neural_Network" / "neural_network_symmetric_features.pth"
SCALER98_PATH  = REPO / "results" / "professional_tennis" / "Neural_Network" / "scaler_symmetric_features.pkl"

OUT_DIR = REPO / "analysis_scripts" / "pure_bet_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_THRESHOLD = 0.00  # log everything with positive edge; filter later if you want
MIN_ODDS = 1.01        # sanity

# --------------------------
# NN definition (matches training)
# --------------------------
class TennisNet(nn.Module):
    def __init__(self, input_size):
        super(TennisNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# --------------------------
# Helpers
# --------------------------
EXCLUDE_COLS = {
    'tourney_date','tourney_name','tourney_id','match_num','winner_id','loser_id',
    'winner_name','loser_name','score','Player1_Name','Player2_Name','Player1_Wins',
    'best_of','round','minutes','data_source','year'
}

def get_numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype.name in ('int64','float64','bool')]

def week_id_from_date(series: pd.Series) -> pd.Series:
    d = pd.to_datetime(series)
    return d.dt.isocalendar().year.astype(str) + "-" + d.dt.isocalendar().week.astype(str).str.zfill(2)

def load_verified():
    ml = pd.read_csv(ML_VERIFIED_PATH, low_memory=False)
    t  = pd.read_csv(TENNIS_VERIFIED_PATH, low_memory=False)
    ml['tourney_date'] = pd.to_datetime(ml['tourney_date'])
    t['Date'] = pd.to_datetime(t['Date'])
    assert len(ml) == len(t), "ML and Tennis verified rows must be aligned & equal length"
    return ml.reset_index(drop=True), t.reset_index(drop=True)

# --------------------------
# Model loaders
# --------------------------
def load_xgb(X_143_df: pd.DataFrame):
    if not XGB_PATH.exists():
        print("   ⚠️  Skipping XGBoost (model file not found)")
        return None
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(XGB_PATH))
        probs = model.predict_proba(X_143_df.values)[:, 1]
        return probs
    except Exception as e:
        print(f"   ⚠️  Skipping XGBoost (load/predict failed): {e}")
        return None

def load_rf(X_143_df: pd.DataFrame):
    if not RF_PATH.exists():
        print("   ⚠️  Skipping Random Forest (model file not found)")
        return None
    try:
        with open(RF_PATH, "rb") as f:
            model = pickle.load(f)
        probs = model.predict_proba(X_143_df.values)[:, 1]
        return probs
    except Exception as e:
        print(f"   ⚠️  Skipping Random Forest (load/predict failed): {e}")
        return None

def load_nn_probs(ml: pd.DataFrame, feature_cols, model_path: Path, scaler_path: Path):
    if not model_path.exists() or not scaler_path.exists():
        print(f"   ⚠️  Skipping NN ({model_path.name}) — missing model/scaler")
        return None
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = ml[feature_cols].values
        # Trim or warn if widths differ from scaler
        if hasattr(scaler, "n_features_in_") and X.shape[1] != scaler.n_features_in_:
            if X.shape[1] > scaler.n_features_in_:
                # keep the first n that the scaler expects (preserves training order if feature_cols was built that way)
                X = X[:, :scaler.n_features_in_]
            else:
                raise ValueError(f"X has {X.shape[1]} features, but scaler expects {scaler.n_features_in_}.")
        Xs = scaler.transform(X)
        net = TennisNet(Xs.shape[1])
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval()
        with torch.no_grad():
            probs = net(torch.FloatTensor(Xs)).squeeze().numpy()
        return probs
    except Exception as e:
        print(f"   ⚠️  Skipping NN ({model_path.name}) — predict failed: {e}")
        return None

# --------------------------
# Build bets for one model
# --------------------------
def build_bets_for_model(model_name, p1_probs, ml, tennis, edge_threshold=EDGE_THRESHOLD):
    if p1_probs is None:
        return None

    p1_won = ml["Player1_Wins"].values.astype(int)
    avgw = tennis["AvgW"].values
    avgl = tennis["AvgL"].values

    p1_odds = np.where(p1_won == 1, avgw, avgl)
    p2_odds = np.where(p1_won == 1, avgl, avgw)

    # Filter out bad odds rows
    valid = (p1_odds > MIN_ODDS) & (p2_odds > MIN_ODDS) & np.isfinite(p1_probs)
    if not valid.all():
        p1_probs = p1_probs[valid]
        p1_odds = p1_odds[valid]
        p2_odds = p2_odds[valid]
        ml = ml.loc[valid].reset_index(drop=True)
        tennis = tennis.loc[valid].reset_index(drop=True)
        p1_won = p1_won[valid]
        avgw = avgw[valid]
        avgl = avgl[valid]

    # convert to probs
    mkt_p1 = 1.0 / p1_odds
    mkt_p2 = 1.0 / p2_odds

    model_p1 = p1_probs
    model_p2 = 1.0 - model_p1

    edge_p1 = model_p1 - mkt_p1
    edge_p2 = model_p2 - mkt_p2

    pick_p1 = (edge_p1 > edge_threshold) & (edge_p1 >= edge_p2)
    pick_p2 = (edge_p2 > edge_threshold) & (edge_p2 > edge_p1)

    rows = []
    dates = ml["tourney_date"]
    week_ids = week_id_from_date(dates)
    p1 = ml["Player1_Name"]
    p2 = ml["Player2_Name"]
    outcome = p1_won  # 1 if P1 actually won

    for i in range(len(ml)):
        if pick_p1[i]:
            bet_on_p1 = True
            odds = float(p1_odds[i])
            prob = float(model_p1[i])
            mprob = float(mkt_p1[i])
            edge = float(edge_p1[i])
            won = int(outcome[i] == 1)
            who = p1.iloc[i]
        elif pick_p2[i]:
            bet_on_p1 = False
            odds = float(p2_odds[i])
            prob = float(model_p2[i])
            mprob = float(mkt_p2[i])
            edge = float(edge_p2[i])
            won = int(outcome[i] == 0)
            who = p2.iloc[i]
        else:
            continue

        rows.append({
            "date": dates.iloc[i].date().isoformat(),
            "week_id": week_ids.iloc[i],
            "player1": p1.iloc[i],
            "player2": p2.iloc[i],
            "bet_on_player": who,
            "bet_on_p1": bet_on_p1,
            "prob": prob,                  # model probability for the side we bet
            "market_prob": mprob,          # implied from AvgW/AvgL for that side
            "edge": edge,
            "odds": odds,
            "outcome": won,                # 1 if the bet side actually won
            "model": model_name,
            "avgw": float(avgw[i]),
            "avgl": float(avgl[i]),
        })

    return pd.DataFrame(rows)

# --------------------------
# Main
# --------------------------
def main():
    print("\n" + "="*80)
    print("PURE BET LOGS (AvgW/AvgL only) — no bankroll simulation")
    print("="*80)

    # 1) load verified base
    ml, tennis = load_verified()
    print(f"   ✅ Verified base rows: {len(ml):,}")

    # 2) build feature DataFrames with correct columns/order
    #    - exact 143 for XGB/RF/NN143
    missing_143 = [c for c in EXACT_143_FEATURES if c not in ml.columns]
    if missing_143:
        print(f"   ⚠️  Missing {len(missing_143)} of 143 required columns:"
              f" {missing_143[:6]}{'...' if len(missing_143) > 6 else ''}")
        X_143_df = None
    else:
        X_143_df = ml[EXACT_143_FEATURES].copy()

    numeric = get_numeric_cols(ml)
    feature_98 = [c for c in numeric if c not in LOW_IMPORTANCE_REMOVE]

    # 3) predict probs per model
    print("\nPredicting probabilities...")
    probs = {}

    # XGB
    print(" - XGBoost...", end="", flush=True)
    if X_143_df is not None:
        probs["xgboost"] = load_xgb(X_143_df)
    else:
        probs["xgboost"] = None
    print(" done" if probs["xgboost"] is not None else " skipped")

    # RF
    print(" - Random Forest...", end="", flush=True)
    if X_143_df is not None:
        probs["random_forest"] = load_rf(X_143_df)
    else:
        probs["random_forest"] = None
    print(" done" if probs["random_forest"] is not None else " skipped")

    # NN-143 (143 exact + scaler)
    print(" - Neural Net 143...", end="", flush=True)
    if X_143_df is not None:
        probs["neural_network_143"] = load_nn_probs(ml, EXACT_143_FEATURES, NN143_PATH, SCALER143_PATH)
    else:
        probs["neural_network_143"] = None
    print(" done" if probs["neural_network_143"] is not None else " skipped")

    # NN-98 (symmetric removal to reach 98; trim to scaler's width if needed)
    print(" - Neural Net 98...", end="", flush=True)
    probs["neural_network_98"] = load_nn_probs(ml, feature_98, NN98_PATH, SCALER98_PATH)
    print(" done" if probs["neural_network_98"] is not None else " skipped")

    # 4) build per-model pure bet logs
    print("\nBuilding pure bet logs (edge > {:.2%})...".format(EDGE_THRESHOLD))
    saved = []
    for mname, p1p in probs.items():
        df = build_bets_for_model(mname, p1p, ml, tennis, EDGE_THRESHOLD)
        if df is None or df.empty:
            print(f"   ⚠️  {mname}: no bets (or model skipped).")
            continue
        out = OUT_DIR / f"{mname}_pure_bets.csv"
        df.to_csv(out, index=False)
        saved.append(out)
        wr = df["outcome"].mean() if len(df) else 0
        print(f"   ✅ {mname}: {len(df):,} bets, win rate {wr:.1%} -> {out}")

    if not saved:
        print("\n❌ No pure bet logs produced. Check model/scaler paths and feature lists.")
    else:
        print("\n🎉 Done. Wrote {} files:".format(len(saved)))
        for p in saved:
            print("   •", p)

if __name__ == "__main__":
    main()
