#!/usr/bin/env python3
"""
Edge threshold analysis:
  1. Kelly bankroll curves filtering to edge > threshold (per model)
  2. Flat ROI per bucket with 95% CI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

REPO    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.dirname(__file__)
LOGS    = os.path.join(REPO, 'analysis_scripts', 'pure_bet_logs')

KELLY_FRAC = 0.10
MAX_BET    = 0.05   # max 5% of bankroll per bet

EDGE_THRESHOLDS  = [0.00, 0.03, 0.06, 0.10, 0.15, 0.20]
THRESHOLD_LABELS = ['All (>0%)', '>3%', '>6%', '>10%', '>15%', '>20%']
THRESHOLD_COLORS = ['#9E9E9E', '#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

EDGE_BINS   = [0.0, 0.02, 0.04, 0.06, 0.10, 0.20, 1.0]
BIN_LABELS  = ['0-2%', '2-4%', '4-6%', '6-10%', '10-20%', '>20%']

bet_models = {
    'NN-143':        'neural_network_143',
    'XGBoost':       'xgboost',
    'Random Forest': 'random_forest',
}

bets = {}
for display, fname in bet_models.items():
    p = os.path.join(LOGS, f'{fname}_pure_bets.csv')
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    bets[display] = df

BUCKET_RANGES = [
    (0.00, 0.02, '0-2%',   '#9E9E9E'),
    (0.02, 0.04, '2-4%',   '#2196F3'),
    (0.04, 0.06, '4-6%',   '#00BCD4'),
    (0.06, 0.10, '6-10%',  '#4CAF50'),
    (0.10, 0.20, '10-20%', '#FF9800'),
    (0.20, 1.00, '>20%',   '#E91E63'),
]

def run_kelly(sub):
    bankroll = 1.0
    curve = [1.0]
    for _, row in sub.iterrows():
        k = KELLY_FRAC * row['edge'] * row['odds'] / (row['odds'] - 1)
        stake = min(k, MAX_BET) * bankroll
        stake = max(0.0, stake)
        bankroll += stake * (row['odds'] - 1) if row['outcome'] == 1 else -stake
        curve.append(bankroll)
    return bankroll, curve

# ──────────────────────────────────────────────────────────────────────────────
# CHART 1A: Cumulative threshold (floor only — bet everything above X)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    f'Kelly Bankroll — Cumulative Edge Floor ({int(KELLY_FRAC*100)}% Kelly, max {int(MAX_BET*100)}%/bet)\n'
    'Each curve bets everything with edge ≥ threshold  |  Higher lines include all bets below them too',
    fontsize=13, fontweight='bold'
)

print("=" * 70)
print("CHART 1A: CUMULATIVE THRESHOLD (floor only)")
print("=" * 70)

for ax, (name, df) in zip(axes, bets.items()):
    print(f"\n{name}:")
    print(f"  {'Threshold':<10} {'#Bets':>6}  {'Return':>9}")
    for thresh, label, color in zip(EDGE_THRESHOLDS, THRESHOLD_LABELS, THRESHOLD_COLORS):
        sub = df[df['edge'] >= thresh].reset_index(drop=True)
        if len(sub) == 0:
            continue
        final_br, curve = run_kelly(sub)
        final_ret = (final_br - 1) * 100
        ax.plot(curve, label=f'{label} | n={len(sub)} | {final_ret:+.0f}%',
                color=color, linewidth=1.8, alpha=0.9)
        print(f"  {label:<10} {len(sub):>6}  {final_ret:>+8.1f}%")

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Bet number (date order)')
    ax.set_ylabel('Bankroll (start = 1.0)')
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'kelly_cumulative_threshold.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: kelly_cumulative_threshold.png")

# ──────────────────────────────────────────────────────────────────────────────
# CHART 1B: Isolated buckets (each range independently, no overlap)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    f'Kelly Bankroll — Isolated Edge Buckets ({int(KELLY_FRAC*100)}% Kelly, max {int(MAX_BET*100)}%/bet)\n'
    'Each curve bets ONLY within that range — completely independent, no overlap',
    fontsize=13, fontweight='bold'
)

print("\n" + "=" * 70)
print("CHART 1B: ISOLATED BUCKETS (no overlap)")
print("=" * 70)

for ax, (name, df) in zip(axes, bets.items()):
    print(f"\n{name}:")
    print(f"  {'Bucket':<8} {'#Bets':>6}  {'Return':>9}")
    for lo, hi, label, color in BUCKET_RANGES:
        sub = df[(df['edge'] >= lo) & (df['edge'] < hi)].reset_index(drop=True)
        if len(sub) == 0:
            continue
        final_br, curve = run_kelly(sub)
        final_ret = (final_br - 1) * 100
        ax.plot(curve, label=f'{label} | n={len(sub)} | {final_ret:+.0f}%',
                color=color, linewidth=1.8, alpha=0.9)
        print(f"  {label:<8} {len(sub):>6}  {final_ret:>+8.1f}%")

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label='Break even')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Bet number within bucket (date order)')
    ax.set_ylabel('Bankroll (start = 1.0 per bucket)')
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'kelly_isolated_buckets.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: kelly_isolated_buckets.png")

# ──────────────────────────────────────────────────────────────────────────────
# CHART 2: Flat ROI per bucket with 95% CI (one subplot per model)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    'Flat-Stake ROI per Edge Bucket with 95% Confidence Intervals\n'
    'Error bars show how uncertain each estimate is given sample size',
    fontsize=13, fontweight='bold'
)

print("\n" + "=" * 70)
print("FLAT ROI PER BUCKET WITH 95% CI")
print("=" * 70)

for ax, (name, df) in zip(axes, bets.items()):
    df = df.copy()
    df['edge_bin'] = pd.cut(df['edge'], bins=EDGE_BINS, labels=BIN_LABELS, right=True)

    rows = []
    for label in BIN_LABELS:
        sub = df[df['edge_bin'] == label]
        if len(sub) == 0:
            continue
        payouts  = np.where(sub['outcome'] == 1, sub['odds'] - 1, -1.0)
        roi      = payouts.mean()
        se       = payouts.std() / np.sqrt(len(sub))
        ci95     = 1.96 * se
        rows.append({'bin': label, 'roi': roi, 'ci95': ci95, 'n': len(sub),
                     'wr': sub['outcome'].mean(), 'avg_odds': sub['odds'].mean()})

    stats = pd.DataFrame(rows)

    bar_colors = ['#4CAF50' if r > 0 else '#F44336' for r in stats['roi']]
    bars = ax.bar(stats['bin'], stats['roi'] * 100, color=bar_colors, alpha=0.75, edgecolor='black')
    ax.errorbar(stats['bin'], stats['roi'] * 100, yerr=stats['ci95'] * 100,
                fmt='none', color='black', capsize=5, linewidth=1.5, label='95% CI')
    ax.axhline(0, color='black', linewidth=1.0)

    for bar, r in zip(bars, stats.itertuples()):
        y_pos = r.roi * 100 + r.ci95 * 100 + 1.0
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'n={r.n}', ha='center', va='bottom', fontsize=8)

    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Edge Bucket')
    ax.set_ylabel('Flat ROI per bet (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    ax.tick_params(axis='x', rotation=25)

    print(f"\n  {name}:")
    print(f"  {'Bucket':>6}  {'ROI':>8}  {'95% CI':>9}  {'Significant?':>13}  {'WR':>6}  {'AvgOdds':>8}  {'n':>5}")
    for r in stats.itertuples():
        sig = 'YES (loss)' if (r.roi + r.ci95) < 0 else ('YES (gain)' if (r.roi - r.ci95) > 0 else 'no (noise)')
        print(f"  {r.bin:>6}  {r.roi*100:>+7.1f}%  ±{r.ci95*100:>6.1f}pp  {sig:>13}  {r.wr:.1%}  {r.avg_odds:>8.2f}  {r.n:>5}")

plt.tight_layout()
out = os.path.join(OUT_DIR, 'flat_roi_by_bucket.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: flat_roi_by_bucket.png")

print("\nDone.")
