#!/usr/bin/env python3
"""
UTR Predictive Analysis Report Generator
Recreated from HTML and JSON remnants to analyze UTR match data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class UTRAnalysisReport:
    def __init__(self, data_file_path):
        """Initialize with the path to valid_matches_for_model.csv"""
        self.data_file = Path(data_file_path)
        self.output_dir = Path("analysis_output")
        self.plots_dir = self.output_dir / "plots"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.results = {}
        self.plots_generated = []
        
    def _wilson_ci(self, k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2/(2*n)) / denom
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
        return center - margin, center + margin

    def _train_test_split(self, test_size=0.2, seed=42):
        df = self.df.copy()
        # If a date column exists, prefer a temporal split (train: before 2023)
        date_col = None
        for c in ['match_date', 'date', 'start_time', 'event_date']:
            if c in df.columns:
                date_col = c
                break
        if date_col is not None:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                train_df = df[df[date_col] < pd.Timestamp('2023-01-01')]
                test_df  = df[df[date_col] >= pd.Timestamp('2023-01-01')]
                if len(train_df) > 0 and len(test_df) > 0:
                    return train_df, test_df
            except Exception:
                pass
        # Fallback: random split
        return train_test_split(df, test_size=test_size, random_state=seed, stratify=df['player_won'])

    def _bin_summary(self, grp):
        # Returns mean, count, and Wilson CI for the player_won column
        n = grp['player_won'].size
        k = grp['player_won'].sum()
        mean = 0.0 if n == 0 else k / n
        lo, hi = self._wilson_ci(k, n)
        return pd.Series({'mean': mean, 'count': n, 'lo': lo, 'hi': hi})
        
    def load_data(self):
        """Load and prepare the match data"""
        print("Loading match data...")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
        self.df = pd.read_csv(self.data_file)
        print(f"Loaded {len(self.df)} matches")
        
        # Basic data validation and cleaning
        required_cols = ['player_utr_at_match', 'opponent_utr_at_match', 'utr_diff']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove matches with missing UTR data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['player_utr_at_match', 'opponent_utr_at_match', 'utr_diff'])
        print(f"Removed {initial_count - len(self.df)} matches with missing UTR data")
        
        # Create binary outcome (1 if player won, 0 if lost)
        if 'result' in self.df.columns:
            self.df['player_won'] = (self.df['result'] == 'W').astype(int)
        elif 'player_won' not in self.df.columns:
            # If no result column, we'll need to infer or create dummy data
            print("Warning: No result/player_won column found. Creating dummy results for demonstration.")
            # Create results based on UTR differential with some noise
            prob_win = 1 / (1 + np.exp(-1.5 * self.df['utr_diff']))
            self.df['player_won'] = np.random.binomial(1, prob_win)
        
        # Create UTR level categories
        self.df['player_utr_level'] = pd.cut(self.df['player_utr_at_match'], 
                                           bins=[0, 8, 10, 12, 13, 14, 15, 20], 
                                           labels=['<8', '8-10', '10-12', '12-13', '13-14', '14-15', '15+'])
        
        # Create average UTR for each match
        self.df['avg_utr'] = (self.df['player_utr_at_match'] + self.df['opponent_utr_at_match']) / 2
        self.df['avg_utr_level'] = pd.cut(self.df['avg_utr'],
                                        bins=[0, 12, 14, 20],
                                        labels=['UTR < 12', 'UTR 12-14', 'UTR 14+'])
        
        print("Data preparation complete")
        
    def calculate_basic_stats(self):
        """Calculate basic statistics about the dataset"""
        print("Calculating basic statistics...")

        # Correctness from the "higher UTR wins" rule (ties count 0.5)
        higher_utr_wins = ((self.df['utr_diff'] > 0) & (self.df['player_won'] == 1)).sum()
        higher_utr_losses = ((self.df['utr_diff'] > 0) & (self.df['player_won'] == 0)).sum()
        lower_utr_wins = ((self.df['utr_diff'] < 0) & (self.df['player_won'] == 0)).sum()
        lower_utr_losses = ((self.df['utr_diff'] < 0) & (self.df['player_won'] == 1)).sum()
        ties = (self.df['utr_diff'] == 0).sum()

        total = len(self.df)
        correct = higher_utr_wins + lower_utr_wins + 0.5 * ties
        overall_accuracy = correct / total if total else np.nan

        # UTR distribution stats
        utr_stats = {
            'mean': float(self.df['player_utr_at_match'].mean()),
            'median': float(self.df['player_utr_at_match'].median()),
            'min': float(self.df['player_utr_at_match'].min()),
            'max': float(self.df['player_utr_at_match'].max())
        }

        # Accuracy by player UTR level (with CIs and counts)
        accuracy_by_level = {}
        for level in self.df['player_utr_level'].cat.categories:
            level_df = self.df[self.df['player_utr_level'] == level]
            n = len(level_df)
            if n == 0:
                continue
            # "higher-UTR wins" correctness within this level
            k = (((level_df['utr_diff'] > 0) & (level_df['player_won'] == 1)) |
                 ((level_df['utr_diff'] < 0) & (level_df['player_won'] == 0))).sum()
            ties_l = (level_df['utr_diff'] == 0).sum()
            acc = (k + 0.5 * ties_l) / n
            lo, hi = self._wilson_ci(k + 0.5 * ties_l, n)  # treat tie-half as success for CI approx
            accuracy_by_level[str(level)] = {'acc': float(acc), 'n': int(n), 'lo': float(lo), 'hi': float(hi)}

        self.results = {
            'overall_accuracy': float(overall_accuracy),
            'accuracy_by_level': accuracy_by_level,
            'utr_distribution': utr_stats,
            'total_matches': int(total),
            'ties': int(ties)
        }

        print(f"Overall UTR prediction accuracy (higher-UTR rule, ties=0.5): {overall_accuracy:.2%}")
        print(f"Total matches analyzed: {total:,}")
        
    def plot_win_rate_by_utr_diff(self):
        """Plot win rate by UTR differential ranges with Wilson CIs and sample sizes"""
        print("Generating win rate by UTR differential plot...")

        bins = np.arange(-4, 4.5, 0.5)
        self.df['utr_diff_bin'] = pd.cut(self.df['utr_diff'], bins=bins, include_lowest=True)

        # From player's perspective: mean(player_won)
        binned = self.df.groupby('utr_diff_bin').apply(self._bin_summary).reset_index()
        # keep only reasonably-sized bins
        binned = binned[binned['count'] >= 50]

        plt.figure(figsize=(12, 6))
        xs = [iv.mid for iv in binned['utr_diff_bin']]
        
        # Plot shaded confidence interval
        plt.fill_between(xs, binned['lo'], binned['hi'], alpha=0.15, color='steelblue', label='95% CI')
        
        # Plot main line
        plt.plot(xs, binned['mean'], 'o-', linewidth=2, markersize=6, color='steelblue', label='Win rate')
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
        
        # Add sample sizes
        for x, y, n in zip(xs, binned['mean'], binned['count']):
            plt.text(x, y + 0.02, f"n={n:,}", ha='center', fontsize=8)
        
        plt.xlabel('UTR Differential (Player - Opponent)')
        plt.ylabel('Player win rate')
        plt.title('Win Rate vs UTR Differential (95% Wilson CIs, n≥50 bins)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = self.plots_dir / "win_rate_by_utr_diff.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("win_rate_by_utr_diff.png")
        
    def plot_win_rate_by_small_utr_diff(self):
        """Plot win rate for small UTR differentials (±2) with CIs and Ns"""
        print("Generating small UTR differential plot...")

        sdf = self.df[abs(self.df['utr_diff']) <= 2].copy()
        bins = np.arange(-2, 2.2, 0.2)
        sdf['utr_diff_bin'] = pd.cut(sdf['utr_diff'], bins=bins, include_lowest=True)
        binned = sdf.groupby('utr_diff_bin').apply(self._bin_summary).reset_index()
        binned = binned[binned['count'] >= 20]

        plt.figure(figsize=(12, 6))
        xs = [iv.mid for iv in binned['utr_diff_bin']]
        
        # Plot shaded confidence interval
        plt.fill_between(xs, binned['lo'], binned['hi'], alpha=0.15, color='steelblue', label='95% CI')
        
        # Plot main line
        plt.plot(xs, binned['mean'], 'o-', linewidth=2, markersize=6, color='steelblue', label='Win rate')
        plt.axvline(0.0, color='black', linestyle=':', alpha=0.7, label='Even UTR')
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
        
        # Add sample sizes
        for x, y, n in zip(xs, binned['mean'], binned['count']):
            plt.text(x, y + 0.02, f"n={n:,}", ha='center', fontsize=8)
            
        plt.xlabel('UTR Differential (Player - Opponent)')
        plt.ylabel('Player win rate')
        plt.title('Win Rate for Small UTR Differentials (95% Wilson CIs, n≥20 bins)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = self.plots_dir / "win_rate_by_small_utr_diff.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("win_rate_by_small_utr_diff.png")
        
    def plot_accuracy_by_utr_level(self):
        """Plot accuracy by UTR level with Wilson CIs and sample sizes"""
        print("Generating accuracy by UTR level plot...")

        levels = list(self.results['accuracy_by_level'].keys())
        accs = [self.results['accuracy_by_level'][lvl]['acc'] for lvl in levels]
        los  = [self.results['accuracy_by_level'][lvl]['lo'] for lvl in levels]
        his  = [self.results['accuracy_by_level'][lvl]['hi'] for lvl in levels]
        ns   = [self.results['accuracy_by_level'][lvl]['n']  for lvl in levels]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(levels, accs, color='steelblue', alpha=0.8, yerr=[np.array(accs)-np.array(los), np.array(his)-np.array(accs)],
                       capsize=3, error_kw={'elinewidth':1})
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
        plt.xlabel('Player UTR Level')
        plt.ylabel('Higher-UTR rule accuracy')
        plt.title('UTR Prediction Accuracy by Skill Level (95% CIs)')
        plt.xticks(rotation=45)
        for bar, acc, n, hi in zip(bars, accs, ns, his):
            # Position text above the confidence interval bar (hi) rather than the bar itself
            plt.text(bar.get_x() + bar.get_width()/2, hi + 0.01, f'{acc:.1%}\nn={n:,}',
                     ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Add extra space above bars to prevent text clipping
        plt.ylim(bottom=0.45, top=max(his) + 0.08)
        plt.legend()
        plt.tight_layout()
        plot_path = self.plots_dir / "accuracy_by_utr_level.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("accuracy_by_utr_level.png")
        
    def plot_accuracy_heatmap(self):
        """Plot accuracy heatmap by UTR level and differential; mask low-N cells"""
        print("Generating accuracy heatmap...")

        self.df['utr_diff_range'] = pd.cut(
            self.df['utr_diff'],
            bins=[-10, -2, -1, -0.5, 0.5, 1, 2, 10],
            labels=['≤-2', '(-2,-1]', '(-1,-0.5]', '(-0.5,0.5]', '(0.5,1]', '(1,2]', '≥2']
        )

        # Accuracy per cell
        def cell_acc(g):
            n = len(g)
            if n == 0:
                return pd.Series({'acc': np.nan, 'n': 0})
            k = (((g['utr_diff'] > 0) & (g['player_won'] == 1)) |
                 ((g['utr_diff'] < 0) & (g['player_won'] == 0))).sum()
            ties = (g['utr_diff'] == 0).sum()
            acc = (k + 0.5 * ties) / n
            return pd.Series({'acc': acc, 'n': n})

        gdf = self.df.groupby(['player_utr_level', 'utr_diff_range']).apply(cell_acc).reset_index()
        acc_p = gdf.pivot(index='player_utr_level', columns='utr_diff_range', values='acc')
        n_p   = gdf.pivot(index='player_utr_level', columns='utr_diff_range', values='n')

        # Mask cells with very low sample size
        mask = n_p < 50

        plt.figure(figsize=(12, 8))
        sns.heatmap(acc_p, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0.5, vmin=0, vmax=1, mask=mask)
        plt.title('UTR Prediction Accuracy by Skill Level and Differential (masked n<50)')
        plt.xlabel('UTR Differential Range (Player - Opponent)')
        plt.ylabel('Player UTR Level')
        plot_path = self.plots_dir / "accuracy_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("accuracy_heatmap.png")

        # Optional: counts heatmap (comment out if you don't want it)
        plt.figure(figsize=(12, 8))
        sns.heatmap(n_p, annot=True, fmt='.0f', cmap='Greys', cbar=False)
        plt.title('Sample Sizes by Skill Level and Differential')
        plt.xlabel('UTR Differential Range')
        plt.ylabel('Player UTR Level')
        plot_path2 = self.plots_dir / "count_heatmap.png"
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("count_heatmap.png")
        
    def logistic_regression_analysis(self):
        """Perform logistic regression analysis with a proper holdout"""
        print("Performing logistic regression analysis...")

        train_df, test_df = self._train_test_split()

        def fit_eval(df_tr, df_te, label):
            X_tr = df_tr[['utr_diff']].values
            y_tr = df_tr['player_won'].values
            X_te = df_te[['utr_diff']].values
            y_te = df_te['player_won'].values

            model = LogisticRegression()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_te, y_pred)
            brier = brier_score_loss(y_te, y_proba)
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            auc_val = auc(fpr, tpr)

            return {
                'Level': label,
                'Matches': int(len(df_te)),
                'Coefficient': float(model.coef_[0][0]),
                'Intercept': float(model.intercept_[0]),
                'Accuracy': float(acc),
                'Brier': float(brier),
                'AUC': float(auc_val)
            }, y_te, y_proba

        regression_results = []
        overall_res, y_all, p_all = fit_eval(train_df, test_df, 'All')
        regression_results.append(overall_res)

        # By UTR band
        by_levels = ['UTR < 12', 'UTR 12-14', 'UTR 14+']
        for lvl in by_levels:
            tr = train_df[train_df['avg_utr_level'] == lvl]
            te = test_df[test_df['avg_utr_level'] == lvl]
            if len(tr) >= 100 and len(te) >= 100:
                res, _, _ = fit_eval(tr, te, lvl)
                regression_results.append(res)

        # Save preds for calibration/ROC plots
        self._calib_y = y_all
        self._calib_p = p_all
        self.results['regression_results'] = regression_results
        
    def plot_logistic_regression_curve(self):
        """Plot logistic regression curve"""
        print("Generating logistic regression curve...")
        
        X = self.df[['utr_diff']].values
        y = self.df['player_won'].values
        
        model = LogisticRegression()
        model.fit(X, y)
        
        utr_diff_range = np.linspace(-4, 4, 100).reshape(-1, 1)
        probabilities = model.predict_proba(utr_diff_range)[:, 1]
        
        plt.figure(figsize=(10, 6))
        plt.plot(utr_diff_range, probabilities, 'b-', linewidth=2, label='Logistic Regression')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
        plt.xlabel('UTR Differential (Player - Opponent)')
        plt.ylabel('Probability of Player Winning')
        plt.title('Logistic Regression: Win Probability vs UTR Differential')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = self.plots_dir / "logistic_regression_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("logistic_regression_curve.png")
        
    def plot_logistic_regression_by_level(self):
        """Plot logistic regression curves by UTR level"""
        print("Generating logistic regression by level plot...")
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'green', 'red']
        utr_diff_range = np.linspace(-3, 3, 100).reshape(-1, 1)
        
        for i, level in enumerate(['UTR < 12', 'UTR 12-14', 'UTR 14+']):
            level_df = self.df[self.df['avg_utr_level'] == level]
            if len(level_df) > 100:
                X_level = level_df[['utr_diff']].values
                y_level = level_df['player_won'].values
                
                model = LogisticRegression()
                model.fit(X_level, y_level)
                probabilities = model.predict_proba(utr_diff_range)[:, 1]
                
                plt.plot(utr_diff_range, probabilities, color=colors[i], 
                        linewidth=2, label=f'{level} ({len(level_df):,} matches)')
        
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='50% baseline')
        plt.xlabel('UTR Differential (Player - Opponent)')
        plt.ylabel('Probability of Player Winning')
        plt.title('Logistic Regression Curves by UTR Level')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = self.plots_dir / "logistic_regression_by_level.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("logistic_regression_by_level.png")
        
    def plot_calibration(self, n_bins=15):
        """Reliability diagram + ECE/MCE and Brier"""
        print("Generating calibration plot...")
        y_true = getattr(self, '_calib_y', None)
        y_prob = getattr(self, '_calib_p', None)
        if y_true is None or y_prob is None:
            print("Calibration skipped (no stored predictions). Run logistic_regression_analysis first.")
            return

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

        # ECE/MCE
        bins = np.linspace(0, 1, n_bins+1)
        idx = np.digitize(y_prob, bins) - 1
        ece = 0.0
        mce = 0.0
        N = len(y_prob)
        for b in range(n_bins):
            mask = (idx == b)
            if mask.sum() == 0:
                continue
            conf = y_prob[mask].mean()
            acc  = y_true[mask].mean()
            gap  = abs(acc - conf)
            ece += (mask.sum() / N) * gap
            mce = max(mce, gap)

        brier = brier_score_loss(y_true, y_prob)

        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Observed')
        plt.xlabel('Predicted probability')
        plt.ylabel('Empirical win rate')
        plt.title(f'Calibration (Brier={brier:.3f}, ECE={ece:.3f}, MCE={mce:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = self.plots_dir / "calibration_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("calibration_curve.png")

        self.results['calibration'] = {'brier': float(brier), 'ece': float(ece), 'mce': float(mce)}
        
    def plot_roc_curve(self):
        """Plot ROC curve using the same test set as logistic_regression_analysis"""
        print("Generating ROC curve...")

        y_true = getattr(self, '_calib_y', None)
        y_prob = getattr(self, '_calib_p', None)
        if y_true is None or y_prob is None:
            # fallback: quick split
            train_df, test_df = self._train_test_split()
            model = LogisticRegression().fit(train_df[['utr_diff']], train_df['player_won'])
            y_true = test_df['player_won'].values
            y_prob = model.predict_proba(test_df[['utr_diff']])[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for UTR Differential Prediction (Test Set)')
        plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
        plot_path = self.plots_dir / "roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("roc_curve.png")
        
    def plot_utr_distribution(self):
        """Plot UTR distribution histogram"""
        print("Generating UTR distribution plot...")
        
        plt.figure(figsize=(10, 6))
        
        # Combine player and opponent UTRs for overall distribution
        all_utrs = pd.concat([self.df['player_utr_at_match'], self.df['opponent_utr_at_match']])
        
        plt.hist(all_utrs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(all_utrs.mean(), color='red', linestyle='--', 
                   label=f'Mean: {all_utrs.mean():.2f}')
        plt.axvline(all_utrs.median(), color='orange', linestyle='--', 
                   label=f'Median: {all_utrs.median():.2f}')
        
        plt.xlabel('UTR Rating')
        plt.ylabel('Frequency')
        plt.title('Distribution of UTR Ratings in Dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.plots_dir / "utr_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("utr_distribution.png")
        
    def plot_utr_distribution_by_range(self):
        """Plot UTR distribution by range"""
        print("Generating UTR distribution by range plot...")
        
        all_utrs = pd.concat([self.df['player_utr_at_match'], self.df['opponent_utr_at_match']])
        # Fix UTR range bins - max UTR is 16.5, so don't go beyond that
        bins = list(range(4, 17)) + [16.5]  # [4, 5, 6, ..., 16, 16.5]
        labels = [f'{i}-{i+1}' for i in range(4, 16)] + ['16-16.5']
        utr_ranges = pd.cut(all_utrs, bins=bins, labels=labels)
        range_counts = utr_ranges.value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(range_counts)), range_counts.values, 
                      color='steelblue', alpha=0.7)
        plt.xlabel('UTR Range')
        plt.ylabel('Number of Players')
        plt.title('Distribution of Players by UTR Range')
        plt.xticks(range(len(range_counts)), range_counts.index, rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, range_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.plots_dir / "utr_distribution_by_range.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.plots_generated.append("utr_distribution_by_range.png")
        
    def plot_accuracy_by_year(self):
        """Plot accuracy trends by year to show temporal stability"""
        print("Generating accuracy by year plot...")
        
        # Check if we have a date column
        date_col = None
        for c in ['match_date', 'date', 'start_time', 'event_date']:
            if c in self.df.columns:
                date_col = c
                break
        
        if date_col is None:
            print("No date column found, skipping year analysis")
            return
            
        try:
            # Convert to datetime and extract year
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            self.df['year'] = self.df[date_col].dt.year
            
            # Filter to reasonable years and remove NaN years
            self.df = self.df.dropna(subset=['year'])
            year_range = self.df['year'].quantile([0.05, 0.95])
            filtered_df = self.df[(self.df['year'] >= year_range.iloc[0]) & (self.df['year'] <= year_range.iloc[1])]
            
            if len(filtered_df) < 1000:
                print("Not enough data with valid dates for year analysis")
                return
                
            # Calculate accuracy by year using higher-UTR rule
            def year_accuracy(group):
                n = len(group)
                if n < 100:  # Skip years with too little data
                    return pd.Series({'acc': np.nan, 'n': n, 'lo': np.nan, 'hi': np.nan})
                
                k = (((group['utr_diff'] > 0) & (group['player_won'] == 1)) |
                     ((group['utr_diff'] < 0) & (group['player_won'] == 0))).sum()
                ties = (group['utr_diff'] == 0).sum()
                acc = (k + 0.5 * ties) / n
                lo, hi = self._wilson_ci(k + 0.5 * ties, n)
                return pd.Series({'acc': acc, 'n': n, 'lo': lo, 'hi': hi})
            
            yearly_stats = filtered_df.groupby('year').apply(year_accuracy).reset_index()
            yearly_stats = yearly_stats.dropna(subset=['acc'])
            
            if len(yearly_stats) < 2:
                print("Not enough years with sufficient data for temporal analysis")
                return
            
            plt.figure(figsize=(12, 6))
            years = yearly_stats['year']
            
            # Plot shaded confidence interval
            plt.fill_between(years, yearly_stats['lo'], yearly_stats['hi'], alpha=0.15, color='steelblue', label='95% CI')
            
            # Plot main line
            plt.plot(years, yearly_stats['acc'], 'o-', linewidth=2, markersize=6, color='steelblue', label='Yearly accuracy')
            plt.axhline(self.results['overall_accuracy'], color='red', linestyle='--', alpha=0.7, label='Overall average')
            
            # Add sample sizes
            for x, y, n in zip(years, yearly_stats['acc'], yearly_stats['n']):
                plt.text(x, y + 0.005, f"n={n:,}", ha='center', fontsize=8, rotation=45)
            
            plt.xlabel('Year')
            plt.ylabel('Higher-UTR rule accuracy')
            plt.title('UTR Prediction Accuracy by Year (Temporal Stability)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_path = self.plots_dir / "accuracy_by_year.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.plots_generated.append("accuracy_by_year.png")
            
        except Exception as e:
            print(f"Error in year analysis: {e}")
        
    def generate_html_report(self):
        """Generate HTML report"""
        print("Generating HTML report...")
        
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>UTR Predictive Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
        .stats {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UTR Predictive Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        
        <div class="stats">
            <h2>Key Statistics</h2>
            <ul>
                <li>Overall UTR prediction accuracy (higher-UTR rule, ties=0.5): {self.results['overall_accuracy']:.2%}</li>
                <li>Total matches analyzed: {self.results['total_matches']:,} (ties: {self.results.get('ties', 0):,})</li>
                <li>Mean UTR in dataset: {self.results['utr_distribution']['mean']:.2f} (range: {self.results['utr_distribution']['min']:.2f}-{self.results['utr_distribution']['max']:.2f})</li>
                {f'<li>Brier score (test): {self.results.get("calibration", {}).get("brier", 0):.3f}</li>' if self.results.get('calibration') else ''}
                {f'<li>ECE (test): {self.results.get("calibration", {}).get("ece", 0):.3f} | MCE (test): {self.results.get("calibration", {}).get("mce", 0):.3f}</li>' if self.results.get('calibration') else ''}
            </ul>
        </div>
        
        <div class="stats">
            <h2>Methodology Notes</h2>
            <p><strong>Higher-UTR Rule:</strong> Predictions are based on the simple rule that the player with higher UTR wins. 
            Ties (equal UTR) are counted as 0.5 credit to both outcomes, reflecting the inherent uncertainty when players have identical ratings.</p>
            <p><strong>Validation:</strong> All model metrics (Brier, ECE, MCE, AUC) are computed on a holdout test set to ensure unbiased evaluation.</p>
        </div>
        
        <h2>UTR Differential Analysis</h2>
        <div class="plot">
            <img src="plots/win_rate_by_utr_diff.png" alt="Win Rate by UTR Differential">
            <p>Win rate by UTR differential with 95% Wilson confidence intervals. Only bins with n≥50 matches shown.</p>
        </div>
        
        <div class="plot">
            <img src="plots/win_rate_by_small_utr_diff.png" alt="Win Rate by Small UTR Differential">
            <p>Detailed view of win rates for small UTR differentials (±2) with 95% confidence intervals. Only bins with n≥20 matches shown.</p>
        </div>
        
        <h2>UTR Level Analysis</h2>
        <div class="plot">
            <img src="plots/accuracy_by_utr_level.png" alt="Accuracy by UTR Level">
            <p>This plot shows how UTR's predictive accuracy changes across different skill levels.</p>
        </div>
        
        <div class="plot">
            <img src="plots/accuracy_heatmap.png" alt="Accuracy Heatmap">
            <p>Predictive accuracy heatmap by skill level and UTR differential. Cells with n&lt;50 matches are masked for reliability.</p>
        </div>
        
        <h2>Logistic Regression Analysis</h2>
        <div class="plot">
            <img src="plots/logistic_regression_curve.png" alt="Logistic Regression Curve">
            <p>This plot shows the modeled probability of winning based on UTR differential.</p>
        </div>
        
        <div class="plot">
            <img src="plots/logistic_regression_by_level.png" alt="Logistic Regression by Level">
            <p>This plot compares the probability curves across different UTR levels.</p>
        </div>
        
        <div class="plot">
            <img src="plots/roc_curve.png" alt="ROC Curve">
            <p>ROC curve showing the predictive power of UTR differential.</p>
        </div>
        
        <div class="plot">
            <img src="plots/calibration_curve.png" alt="Calibration Curve">
            <p>Reliability diagram on the holdout set with Brier score and calibration errors (ECE/MCE).</p>
        </div>

        <div class="plot">
            <img src="plots/count_heatmap.png" alt="Sample Size Heatmap">
            <p>Sample sizes by skill level and UTR differential (useful context for the accuracy heatmap).</p>
        </div>
        
        <h2>Temporal Stability</h2>
        <div class="plot">
            <img src="plots/accuracy_by_year.png" alt="Accuracy by Year">
            <p>UTR prediction accuracy by year showing temporal stability and consistency over time.</p>
        </div>
        
        <h2>UTR Distribution</h2>
        <div class="plot">
            <img src="plots/utr_distribution.png" alt="UTR Distribution">
            <p>This histogram shows the distribution of UTR ratings in the dataset.</p>
        </div>
        
        <div class="plot">
            <img src="plots/utr_distribution_by_range.png" alt="UTR Distribution by Range">
            <p>This bar chart shows the count of players in each UTR range.</p>
        </div>
        
        <h2>Regression Results</h2>
        <table>
            <tr>
                <th>Level</th>
                <th>Matches</th>
                <th>Coefficient</th>
                <th>Intercept</th>
                <th>Accuracy</th>
                <th>Brier</th>
                <th>AUC</th>
            </tr>'''
        
        for result in self.results.get('regression_results', []):
            html_content += f'''
            <tr>
                <td>{result['Level']}</td>
                <td>{result['Matches']:,}</td>
                <td>{result['Coefficient']:.4f}</td>
                <td>{result['Intercept']:.6f}</td>
                <td>{result['Accuracy']:.2%}</td>
                <td>{result.get('Brier', 0):.3f}</td>
                <td>{result.get('AUC', 0):.3f}</td>
            </tr>'''
        
        html_content += '''
        </table>
    </div>
</body>
</html>'''
        
        with open(self.output_dir / "utr_analysis_report.html", 'w') as f:
            f.write(html_content)
            
    def save_results_json(self):
        """Save results to JSON file"""
        print("Saving results to JSON...")
        
        results_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "results": self.results,
            "plots_generated": self.plots_generated
        }
        
        with open(self.output_dir / "analysis_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
            
    def generate_full_report(self):
        """Generate the complete analysis report"""
        print("Starting UTR Analysis Report Generation...")
        print("="*50)
        self.load_data()
        self.calculate_basic_stats()

        # Plots
        self.plot_win_rate_by_utr_diff()
        self.plot_win_rate_by_small_utr_diff()
        self.plot_accuracy_by_utr_level()
        self.plot_accuracy_heatmap()

        # Modeling on holdout + plots
        self.logistic_regression_analysis()
        self.plot_logistic_regression_curve()  # keeps the pretty curve
        self.plot_roc_curve()
        self.plot_calibration()                # NEW

        # Distributions  
        self.plot_utr_distribution()
        self.plot_utr_distribution_by_range()
        self.plot_logistic_regression_by_level()
        
        # Temporal analysis
        self.plot_accuracy_by_year()

        # Outputs
        self.generate_html_report()
        self.save_results_json()
        print("\n" + "="*50)
        print("Analysis complete!")
        print(f"Generated {len(self.plots_generated)} plots")
        print(f"HTML report: {self.output_dir / 'utr_analysis_report.html'}")
        print(f"JSON results: {self.output_dir / 'analysis_results.json'}")
        print(f"All plots saved in: {self.plots_dir}")


if __name__ == "__main__":
    # Initialize the analysis with your data file
    data_file = "data/valid_matches_for_model.csv"
    
    analyzer = UTRAnalysisReport(data_file)
    analyzer.generate_full_report()