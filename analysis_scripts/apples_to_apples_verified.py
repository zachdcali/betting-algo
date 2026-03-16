#!/usr/bin/env python3
"""
Apples-to-apples comparison on the 1,525 verified matched sample.
All three measures evaluated on IDENTICAL matches:
  - ATP ranking accuracy (lower rank wins)
  - Betting odds accuracy (shorter decimal odds wins)
  - Model accuracy: NN-143, XGBoost, Random Forest
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import xgboost as xgb
import os

REPO    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(REPO, 'analysis_scripts', 'backtests')

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
    'Round_Q4','Peak_Age_P1','Level_G','Round_ER','Level_S','Round_BR','Peak_Age_P2',
    'Round_Q3','Rank_Ratio','P1_Country_SUI','Clay_Season','P1_Country_GER',
    'P2_Rank_Change_30d','P1_Country_ESP','P2_Hand_A','H2H_Recent_P1_Advantage',
    'P2_Country_AUS','P2_Country_SRB','P2_Country_GBR','P2_Country_ARG',
    'Handedness_Matchup_RR','P1_Rank_Change_30d','P2_Country_GER','Handedness_Matchup_LL',
    'P2_Country_RUS','P1_Country_ARG','Level_C','P2_Semifinals_WinRate',
    'P2_Days_Since_Last','H2H_P1_WinRate',
    'P1_Country_Other','H2H_P1_Wins','P1_BigMatch_WinRate','P2_Rank_Change_90d',
    'P2_BigMatch_WinRate','P2_Country_FRA'
]

class TennisNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),    nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# ── Load verified data ─────────────────────────────────────────────────────────
ml  = pd.read_csv(os.path.join(REPO, 'analysis_scripts/verified/ml_verified.csv'), low_memory=False)
ten = pd.read_csv(os.path.join(REPO, 'analysis_scripts/verified/tennis_verified.csv'))

assert len(ml) == len(ten), "Row mismatch between ml_verified and tennis_verified"
n_total = len(ml)
print(f"Verified matched sample: {n_total:,} matches")

y_true = ml['Player1_Wins'].values

# ── Feature matrix ─────────────────────────────────────────────────────────────
feature_cols = [c for c in EXACT_141_FEATURES if c in ml.columns]
missing = [c for c in EXACT_141_FEATURES if c not in ml.columns]
if missing:
    print(f"  Missing {len(missing)} features (will be filled with 0): {missing[:5]}")

X = ml[feature_cols].fillna(0)

# ── ATP ranking accuracy ───────────────────────────────────────────────────────
rank_mask = ml['Player1_Rank'].notna() & ml['Player2_Rank'].notna()
rank_correct = (
    ((ml.loc[rank_mask, 'Player1_Rank'] < ml.loc[rank_mask, 'Player2_Rank']) & (ml.loc[rank_mask, 'Player1_Wins'] == 1)) |
    ((ml.loc[rank_mask, 'Player1_Rank'] > ml.loc[rank_mask, 'Player2_Rank']) & (ml.loc[rank_mask, 'Player1_Wins'] == 0))
).sum()
atp_acc = rank_correct / rank_mask.sum()
print(f"\nATP Ranking accuracy:  {atp_acc*100:.2f}%  (n={rank_mask.sum():,})")

# ── Betting odds accuracy ──────────────────────────────────────────────────────
ten['AvgW'] = pd.to_numeric(ten['AvgW'], errors='coerce')
ten['AvgL'] = pd.to_numeric(ten['AvgL'], errors='coerce')
odds_mask = ten['AvgW'].notna() & ten['AvgL'].notna() & (ten['AvgW'] > 1.01) & (ten['AvgL'] > 1.01)

# AvgW = avg odds on the winner, AvgL = avg odds on the loser
# Favourite = lower decimal odds. Winner has AvgW < AvgL means winner was favourite.
# Player1_Wins=1 means Player1 won. AvgW=winner's odds, AvgL=loser's odds.
# If Player1 won (P1_Wins=1): AvgW = P1 odds, AvgL = P2 odds → odds predict P1 if AvgW < AvgL
# If Player1 lost (P1_Wins=0): AvgW = P2 odds, AvgL = P1 odds → odds predict P2 if AvgW < AvgL (which is correct)
# So: odds are correct whenever AvgW < AvgL (winner had shorter odds)
odds_correct = (ten.loc[odds_mask, 'AvgW'] < ten.loc[odds_mask, 'AvgL']).sum()
odds_acc = odds_correct / odds_mask.sum()
print(f"Betting odds accuracy: {odds_acc*100:.2f}%  (n={odds_mask.sum():,})")

# ── Model predictions ──────────────────────────────────────────────────────────
results = {}

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(REPO, 'results/professional_tennis/XGBoost/xgboost_model.json'))
xgb_proba = xgb_model.predict_proba(X.values)[:, 1]
xgb_acc   = ((xgb_proba > 0.5).astype(int) == y_true).mean()
results['XGBoost'] = {'proba': xgb_proba, 'acc': xgb_acc}
print(f"XGBoost accuracy:      {xgb_acc*100:.2f}%")

# Random Forest
with open(os.path.join(REPO, 'results/professional_tennis/Random_Forest/random_forest_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
rf_proba = rf_model.predict_proba(X.values)[:, 1]
rf_acc   = ((rf_proba > 0.5).astype(int) == y_true).mean()
results['Random Forest'] = {'proba': rf_proba, 'acc': rf_acc}
print(f"Random Forest accuracy:{rf_acc*100:.2f}%")

# NN-143
with open(os.path.join(REPO, 'results/professional_tennis/Neural_Network/scaler_UNBIASED_TEMPORAL.pkl'), 'rb') as f:
    nn_scaler = pickle.load(f)
X_nn     = X[feature_cols[:nn_scaler.n_features_in_]].values
X_scaled = nn_scaler.transform(X_nn)
nn_model = TennisNet(nn_scaler.n_features_in_)
nn_model.load_state_dict(torch.load(
    os.path.join(REPO, 'results/professional_tennis/Neural_Network/neural_network_model_UNBIASED_TEMPORAL.pth'),
    map_location='cpu'))
nn_model.eval()
with torch.no_grad():
    nn_proba = nn_model(torch.FloatTensor(X_scaled)).squeeze().numpy()
nn_acc = ((nn_proba > 0.5).astype(int) == y_true).mean()
results['NN-143'] = {'proba': nn_proba, 'acc': nn_acc}
print(f"NN-143 accuracy:       {nn_acc*100:.2f}%")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("APPLES-TO-APPLES: Same 1,525 matches, four methods")
print("=" * 60)
print(f"{'Method':<22} {'Accuracy':>9}  {'vs Odds':>9}  {'vs ATP':>9}")
print("-" * 55)
print(f"{'Betting Odds':22} {odds_acc*100:>8.2f}%  {'—':>9}  {(odds_acc-atp_acc)*100:>+8.2f}pp")
print(f"{'ATP Ranking':22} {atp_acc*100:>8.2f}%  {(atp_acc-odds_acc)*100:>+8.2f}pp  {'—':>9}")
for name, m in results.items():
    vs_odds = (m['acc'] - odds_acc) * 100
    vs_atp  = (m['acc'] - atp_acc) * 100
    print(f"{name:<22} {m['acc']*100:>8.2f}%  {vs_odds:>+8.2f}pp  {vs_atp:>+8.2f}pp")

# ── Bar chart ──────────────────────────────────────────────────────────────────
methods = ['ATP\nRanking', 'Betting\nOdds', 'XGBoost', 'Random\nForest', 'NN-143']
accs    = [atp_acc, odds_acc, results['XGBoost']['acc'], results['Random Forest']['acc'], results['NN-143']['acc']]
colors  = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50', '#FF5722']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(methods, [a * 100 for a in accs], color=colors, edgecolor='black', alpha=0.85, width=0.55)

# Label each bar
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Reference line at ATP baseline
ax.axhline(atp_acc * 100, color='#9E9E9E', linewidth=1.2, linestyle='--', alpha=0.7, label=f'ATP baseline ({atp_acc*100:.2f}%)')
ax.axhline(odds_acc * 100, color='#FF9800', linewidth=1.2, linestyle='--', alpha=0.7, label=f'Betting odds ({odds_acc*100:.2f}%)')

ax.set_ylim(60, 74)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(
    f'Apples-to-Apples Accuracy — Identical {n_total:,} Verified Matches\n'
    'ATP ranking, closing betting odds, and all three models on the exact same set',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

out = os.path.join(OUT_DIR, 'apples_to_apples.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
