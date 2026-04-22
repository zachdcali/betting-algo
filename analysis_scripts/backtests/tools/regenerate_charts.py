#!/usr/bin/env python3
"""
Regenerate calibration, ROC, EV bucket, and bankroll charts using correct models.
Loads pure_bet_logs (which are built from LEAK_FREE.csv + UNBIASED_TEMPORAL NN).
Also evaluates full test set (2023+) calibration directly from model predictions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.dirname(__file__)
os.makedirs(OUT_DIR, exist_ok=True)

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
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(conf - acc) * mask.mean()
    return ece

# ─── PART 1: Full test-set metrics (2023+) from models directly ────────────────
print("=" * 70)
print("PART 1: Full 2023+ test set calibration")
print("=" * 70)

ml_path = os.path.join(REPO, 'data', 'JeffSackmann', 'jeffsackmann_ml_ready_LEAK_FREE.csv')
print(f"Loading {ml_path}...")
ml_df = pd.read_csv(ml_path, low_memory=False)
ml_df['tourney_date'] = pd.to_datetime(ml_df['tourney_date'])
ml_df = ml_df[ml_df['tourney_date'] >= '1990-01-01'].copy()
ml_df = ml_df.dropna(subset=['Player1_Rank', 'Player2_Rank'])
test_df = ml_df[ml_df['tourney_date'] >= '2023-01-01'].copy()
print(f"Test set: {len(test_df):,} matches")

feature_cols = [c for c in EXACT_141_FEATURES if c in test_df.columns]
train_df = ml_df[ml_df['tourney_date'] < '2022-01-01'].copy()
train_medians = train_df[feature_cols].median()

X_test = test_df[feature_cols].fillna(train_medians)
y_test = test_df['Player1_Wins'].values

full_metrics = {}

# XGBoost
print("Loading XGBoost...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(REPO, 'results/professional_tennis/XGBoost/xgboost_model.json'))
xgb_proba = xgb_model.predict_proba(X_test)[:,1]
full_metrics['XGBoost'] = {
    'proba': xgb_proba,
    'auc': roc_auc_score(y_test, xgb_proba),
    'brier': brier_score_loss(y_test, xgb_proba),
    'ece': compute_ece(y_test, xgb_proba),
    'acc': ((xgb_proba > 0.5).astype(int) == y_test).mean(),
}
print(f"  XGBoost: ACC={full_metrics['XGBoost']['acc']:.4f}  AUC={full_metrics['XGBoost']['auc']:.4f}  ECE={full_metrics['XGBoost']['ece']:.4f}  Brier={full_metrics['XGBoost']['brier']:.4f}")

# Random Forest
print("Loading Random Forest...")
with open(os.path.join(REPO, 'results/professional_tennis/Random_Forest/random_forest_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
rf_proba = rf_model.predict_proba(X_test)[:,1]
full_metrics['Random Forest'] = {
    'proba': rf_proba,
    'auc': roc_auc_score(y_test, rf_proba),
    'brier': brier_score_loss(y_test, rf_proba),
    'ece': compute_ece(y_test, rf_proba),
    'acc': ((rf_proba > 0.5).astype(int) == y_test).mean(),
}
print(f"  RF: ACC={full_metrics['Random Forest']['acc']:.4f}  AUC={full_metrics['Random Forest']['auc']:.4f}  ECE={full_metrics['Random Forest']['ece']:.4f}  Brier={full_metrics['Random Forest']['brier']:.4f}")

# Neural Network 143
print("Loading NN-143...")
with open(os.path.join(REPO, 'results/professional_tennis/Neural_Network/scaler_UNBIASED_TEMPORAL.pkl'), 'rb') as f:
    nn_scaler = pickle.load(f)
X_test_nn = X_test[feature_cols[:nn_scaler.n_features_in_]].values
X_test_scaled = nn_scaler.transform(X_test_nn)
nn_model = TennisNet(nn_scaler.n_features_in_)
nn_model.load_state_dict(torch.load(
    os.path.join(REPO, 'results/professional_tennis/Neural_Network/neural_network_model_UNBIASED_TEMPORAL.pth'),
    map_location='cpu'))
nn_model.eval()
with torch.no_grad():
    nn_proba = nn_model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()
full_metrics['NN-143'] = {
    'proba': nn_proba,
    'auc': roc_auc_score(y_test, nn_proba),
    'brier': brier_score_loss(y_test, nn_proba),
    'ece': compute_ece(y_test, nn_proba),
    'acc': ((nn_proba > 0.5).astype(int) == y_test).mean(),
}
print(f"  NN-143: ACC={full_metrics['NN-143']['acc']:.4f}  AUC={full_metrics['NN-143']['auc']:.4f}  ECE={full_metrics['NN-143']['ece']:.4f}  Brier={full_metrics['NN-143']['brier']:.4f}")

# ATP ranking baseline
baseline_correct = (
    ((test_df['Player1_Rank'] < test_df['Player2_Rank']) & (test_df['Player1_Wins'] == 1)) |
    ((test_df['Player1_Rank'] > test_df['Player2_Rank']) & (test_df['Player1_Wins'] == 0))
).sum()
baseline_acc = baseline_correct / len(test_df)
print(f"\n  ATP Baseline: {baseline_acc:.4f}")
for name, m in full_metrics.items():
    print(f"  {name} vs baseline: {(m['acc'] - baseline_acc)*100:+.2f}pp")

# ─── PART 2: Calibration curves ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2: Calibration curves + ROC curves")
print("=" * 70)

colors = {'XGBoost': '#2196F3', 'Random Forest': '#4CAF50', 'NN-143': '#FF5722'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Calibration — Full 2023+ Test Set', fontsize=14, fontweight='bold')

for ax, (name, m) in zip(axes, full_metrics.items()):
    prob_true, prob_pred = calibration_curve(y_test, m['proba'], n_bins=10)
    ax.plot(prob_pred, prob_true, 'o-', color=colors[name], linewidth=2, markersize=7, label=name)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.set_title(f"{name}\nACC={m['acc']:.3f}  AUC={m['auc']:.3f}  ECE={m['ece']:.4f}")
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'calibration_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: calibration_curves.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 8))
for name, m in full_metrics.items():
    fpr, tpr, _ = roc_curve(y_test, m['proba'])
    ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=f"{name} (AUC={m['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Full 2023+ Test Set')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: roc_curves.png")

# ─── PART 3: Load pure bet logs for betting analysis ──────────────────────────
print("\n" + "=" * 70)
print("PART 3: Betting analysis from pure_bet_logs")
print("=" * 70)

logs_dir = os.path.join(REPO, 'analysis_scripts', 'pure_bet_logs')
bet_models = {
    'NN-143': 'neural_network_143',
    'XGBoost': 'xgboost',
    'Random Forest': 'random_forest',
}

bets = {}
for display, fname in bet_models.items():
    p = os.path.join(logs_dir, f'{fname}_pure_bets.csv')
    if os.path.exists(p):
        bets[display] = pd.read_csv(p)
        print(f"  {display}: {len(bets[display]):,} bets, WR={bets[display]['outcome'].mean():.3f}")

# ─── PART 4: EV bucket analysis ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 4: EV bucket analysis")
print("=" * 70)

edge_bins = [0.0, 0.02, 0.04, 0.06, 0.10, 0.20, 1.0]
bin_labels = ['0-2%', '2-4%', '4-6%', '6-10%', '10-20%', '>20%']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ROI by Edge Bucket (pure bets at flat stake)', fontsize=14, fontweight='bold')

for ax, (name, df) in zip(axes, bets.items()):
    df = df.copy()
    df['edge_bin'] = pd.cut(df['edge'], bins=edge_bins, labels=bin_labels, right=True)

    bucket_stats = []
    for label in bin_labels:
        sub = df[df['edge_bin'] == label]
        if len(sub) == 0:
            continue
        # ROI: sum(payout - 1) / n_bets  where payout = odds if won else 0
        payouts = np.where(sub['outcome'] == 1, sub['odds'] - 1, -1)
        roi = payouts.mean()
        bucket_stats.append({'bin': label, 'roi': roi, 'n': len(sub), 'wr': sub['outcome'].mean()})

    stats_df = pd.DataFrame(bucket_stats)
    bar_colors = ['#4CAF50' if r > 0 else '#F44336' for r in stats_df['roi']]
    bars = ax.bar(stats_df['bin'], stats_df['roi'] * 100, color=bar_colors, alpha=0.8, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8)

    for bar, row in zip(bars, stats_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if row.roi >= 0 else -1.5),
                f'n={row.n}', ha='center', va='bottom', fontsize=8)

    ax.set_title(f'{name}\nTotal: {len(df):,} bets, WR={df["outcome"].mean():.1%}')
    ax.set_xlabel('Edge Bucket')
    ax.set_ylabel('ROI (%)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=30)

    # Print stats
    print(f"\n  {name}:")
    for row in bucket_stats:
        print(f"    {row['bin']:>6}: ROI={row['roi']*100:+6.1f}%  WR={row['wr']:.1%}  n={row['n']}")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ev_bucket_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: ev_bucket_analysis.png")

# ─── PART 5: Kelly bankroll simulation ─────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 5: Kelly bankroll simulation")
print("=" * 70)

KELLY_FRACTION = 0.10  # 10% Kelly
MAX_BET_PCT = 0.05     # max 5% of bankroll per bet

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title(f'Bankroll Simulation — {int(KELLY_FRACTION*100)}% Kelly (max {int(MAX_BET_PCT*100)}% per bet)', fontsize=13)

for name, df in bets.items():
    df = df.sort_values('date').reset_index(drop=True)
    bankroll = 1.0
    curve = [1.0]
    for _, row in df.iterrows():
        kelly_raw = (row['edge'] * row['odds']) / (row['odds'] - 1) if row['odds'] > 1 else 0
        bet_size = min(kelly_raw * KELLY_FRACTION, MAX_BET_PCT) * bankroll
        bet_size = max(0, bet_size)
        if row['outcome'] == 1:
            bankroll += bet_size * (row['odds'] - 1)
        else:
            bankroll -= bet_size
        curve.append(bankroll)

    final_return = (bankroll - 1) * 100
    ax.plot(curve, label=f'{name} ({final_return:+.1f}%)', linewidth=1.5, color=colors[name])
    print(f"  {name}: final bankroll={bankroll:.3f} ({final_return:+.1f}%), bets={len(df):,}")

ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Break even')
ax.set_xlabel('Bet number')
ax.set_ylabel('Bankroll (starting = 1.0)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'bankroll_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: bankroll_comparison.png")

# ─── PART 6: Summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE — Full 2023+ test set")
print("=" * 70)
print(f"{'Model':<15} {'ACC':>8} {'vs Base':>8} {'AUC':>8} {'ECE':>8} {'Brier':>8}")
print("-" * 59)
for name, m in full_metrics.items():
    edge = (m['acc'] - baseline_acc) * 100
    print(f"{name:<15} {m['acc']*100:>7.2f}% {edge:>+7.2f}pp {m['auc']:>8.4f} {m['ece']:>8.4f} {m['brier']:>8.4f}")
print(f"{'ATP Baseline':<15} {baseline_acc*100:>7.2f}%")

print("\nAll charts saved to:", OUT_DIR)
