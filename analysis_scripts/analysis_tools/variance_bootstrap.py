#!/usr/bin/env python3
import argparse, math, sys, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional charts (won't crash if matplotlib/seaborn is missing)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except Exception:
    HAS_PARQUET = False

# =========================
# Helpers
# =========================
def stable_hash(s: str) -> int:
    """Generate stable hash across Python processes"""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
def max_drawdown_pct(series: pd.Series) -> float:
    """Calculate max drawdown as percentage"""
    if series.empty: return 0.0
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    return float(-dd.min()) * 100.0

def sharpe_academic(per_bet_ret: np.ndarray, bets_per_year: float) -> float:
    """Academic Sharpe: (μ/σ)*sqrt(bets_per_year)"""
    if per_bet_ret.size == 0: return 0.0
    mu = float(np.nanmean(per_bet_ret))
    sd = float(np.nanstd(per_bet_ret, ddof=1))
    if sd == 0 or np.isnan(sd): return 0.0
    return (mu / sd) * math.sqrt(bets_per_year)

def sharpe_industry(per_bet_ret: np.ndarray, bets_per_year: float) -> float:
    """Industry Sharpe: (μ*bets_per_year)/σ"""
    if per_bet_ret.size == 0: return 0.0
    mu = float(np.nanmean(per_bet_ret))
    sd = float(np.nanstd(per_bet_ret, ddof=1))
    if sd == 0 or np.isnan(sd): return 0.0
    return (mu * bets_per_year) / sd

def log_sharpe(per_bet_ret: np.ndarray, bets_per_year: float) -> float:
    """Log Sharpe ratio"""
    if per_bet_ret.size == 0: return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log1p(per_bet_ret)
    mu = float(np.nanmean(lr))
    sd = float(np.nanstd(lr, ddof=1))
    if sd == 0 or np.isnan(sd): return 0.0
    return (mu / sd) * math.sqrt(bets_per_year)

def calendar_sharpe(bankroll_ts: pd.Series, freq: str, periods_per_year: float) -> float:
    """Calendar-based Sharpe ratio"""
    if bankroll_ts.empty: return 0.0
    s = bankroll_ts.sort_index().asfreq(freq, method="pad")
    rets = s.pct_change().dropna()
    if rets.size == 0: return 0.0
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1))
    if sd == 0 or np.isnan(sd): return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)

def implied_bets_per_year(dates: pd.Series) -> float:
    """Calculate implied betting frequency"""
    if dates.empty: return np.nan
    span_days = (dates.iloc[-1] - dates.iloc[0]).days + 1
    if span_days <= 0: return np.nan
    return len(dates) * (365.25 / span_days)

def expected_log_growth(per_bet_ret: np.ndarray) -> float:
    """Expected log growth E[log(1+r)]"""
    if per_bet_ret.size == 0: return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        log_rets = np.log1p(per_bet_ret)
    return float(np.nanmean(log_rets))

def hit_rate(outcomes: np.ndarray) -> float:
    """Calculate hit rate (win percentage)"""
    if len(outcomes) == 0: return 0.0
    return float(np.mean(outcomes))

def avg_win_loss(per_bet_ret: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    """Calculate average win and loss amounts"""
    if len(per_bet_ret) == 0: return 0.0, 0.0
    wins = per_bet_ret[outcomes > 0]
    losses = per_bet_ret[outcomes == 0]
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    return avg_win, avg_loss

def ensure_strict_index(idx: pd.Index) -> pd.Index:
    """Ensure datetime index with no duplicates"""
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    if idx.duplicated().any():
        idx = idx + pd.to_timedelta(np.cumsum(idx.duplicated()).astype(int), unit="s")
    return idx

# =========================
# Sizing (mirror deterministic backtest)
# =========================
def kelly_stake_frac(k: float, edge: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Kelly criterion: f* = edge / (odds - 1), scaled by k"""
    denom = np.maximum(odds - 1.0, 1e-9)
    f = k * (edge / denom)
    return np.clip(f, 0.0, 1.0)

def calculate_stake(sizing: str, k: float, bankroll: float, edge: float, odds: float, 
                   fixed_frac: float, fixed_amt: float, min_stake: float) -> float:
    """Calculate stake amount (mirrors deterministic backtest)"""
    if sizing == "kelly":
        if edge <= 0:
            return 0.0
        kelly_frac = k * edge / max(odds - 1.0, 1e-9)
        kelly_frac = max(0.0, min(1.0, kelly_frac))
        stake = kelly_frac * bankroll
    elif sizing == "fixed":
        if fixed_amt > 0:
            stake = fixed_amt
        else:
            stake = fixed_frac * bankroll
    else:
        raise ValueError(f"Unknown sizing mode: {sizing}")
    
    # Apply minimum stake constraint
    return max(min_stake, min(stake, bankroll)) if bankroll > min_stake else 0.0

# =========================
# Simulation (mirror deterministic backtest)
# =========================
def simulate_path(df: pd.DataFrame, sizing: str, k: float, start_bankroll: float,
                  friction_bps: float, shortfall_policy: str, min_stake: float,
                  fixed_frac: float, fixed_amt: float) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate betting path (mirrors deterministic backtest logic)"""
    dates = pd.to_datetime(df["date"].values)
    edges = df["edge"].to_numpy(float)
    odds = df["odds"].to_numpy(float)
    outcomes = df["outcome"].to_numpy(float)  # 1=win, 0=loss

    bankroll = float(start_bankroll)
    bankroll_history = [bankroll]
    per_bet_returns = []
    outcome_history = []
    stake_fracs = []
    
    friction_factor = friction_bps / 10000.0

    for i in range(len(df)):
        edge = edges[i]
        odds_val = odds[i]
        outcome = outcomes[i]
        
        # Calculate stake
        stake = calculate_stake(sizing, k, bankroll, edge, odds_val, fixed_frac, fixed_amt, min_stake)
        
        # Track stake fraction
        stake_frac = 0.0 if bankroll <= 0 else (stake / bankroll)
        
        if stake <= 0 or bankroll <= min_stake:
            if shortfall_policy == "skip":
                continue
            elif shortfall_policy == "all-in" and bankroll <= 0:
                break
        
        # Calculate return
        if outcome > 0:  # Win
            gross_return = stake * (odds_val - 1.0)
            net_return = gross_return - (stake * friction_factor)
            per_bet_ret = net_return / bankroll
        else:  # Loss
            net_return = -stake - (stake * friction_factor)
            per_bet_ret = net_return / bankroll
        
        per_bet_returns.append(per_bet_ret)
        outcome_history.append(outcome)
        stake_fracs.append(stake_frac)
        bankroll = max(0.0, bankroll + net_return)
        bankroll_history.append(bankroll)
        
        if shortfall_policy == "all-in" and bankroll <= 0:
            break

    # Create time series with proper indexing
    if len(dates):
        idx0 = pd.to_datetime(dates).min() - pd.Timedelta(seconds=1)
        date_indices = [idx0] + list(dates[:len(bankroll_history)-1])
    else:
        date_indices = [pd.Timestamp.now()] * len(bankroll_history)
    
    idx = ensure_strict_index(pd.Index(date_indices))
    bankroll_series = pd.Series(bankroll_history, index=idx)
    
    return bankroll_series, np.array(per_bet_returns, dtype=float), np.array(outcome_history, dtype=float), np.array(stake_fracs, dtype=float)

# =========================
# Bootstrap Sampling
# =========================
def sample_iid(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """IID bootstrap: sample with replacement, preserve temporal order"""
    pool = df.reset_index(drop=True)
    n = len(pool)
    idx = rng.integers(0, n, size=n)
    sampled = pool.iloc[idx].copy()
    # Sort by date to preserve temporal coherence for calendar Sharpe
    return sampled.sort_values("date").reset_index(drop=True)

def sample_blocks(df: pd.DataFrame, block_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Block bootstrap: sample blocks of consecutive days"""
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    dmin, dmax = dates.min(), dates.max()
    target_len = len(df)
    
    out = []
    while sum(len(x) for x in out) < target_len:
        span = max(1, (dmax - dmin).days - block_days + 1)
        start = dmin + pd.Timedelta(days=int(rng.integers(0, span)))
        end = start + pd.Timedelta(days=block_days-1)
        chunk = df[(dates >= start) & (dates <= end)]
        if not chunk.empty:
            out.append(chunk)
    
    res = pd.concat(out, axis=0).head(target_len)
    return res.sort_values("date").reset_index(drop=True)

# =========================
# Deterministic-Style Metrics (comprehensive)
# =========================
def calculate_cagr(start_bankroll: float, final_bankroll: float, start_date: str, end_date: str) -> float:
    """Calculate Compound Annual Growth Rate"""
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        years = (end_dt - start_dt).days / 365.25
        if years <= 0 or final_bankroll <= 0 or start_bankroll <= 0:
            return 0.0
        return ((final_bankroll / start_bankroll) ** (1.0 / years)) - 1.0
    except:
        return 0.0

def calculate_mar_ratio(cagr: float, max_dd_pct: float) -> float:
    """Calculate MAR ratio (CAGR / Max Drawdown)"""
    if max_dd_pct <= 0:
        return 0.0
    return cagr / (max_dd_pct / 100.0)

def calculate_calmar_ratio(cagr: float, max_dd_pct: float) -> float:
    """Calculate Calmar ratio (same as MAR)"""
    return calculate_mar_ratio(cagr, max_dd_pct)

def calculate_sortino_ratio(per_bet_rets: np.ndarray, bets_per_year: float, target_return: float = 0.0) -> float:
    """Calculate Sortino ratio"""
    if len(per_bet_rets) == 0:
        return 0.0
    
    excess_returns = per_bet_rets - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(downside_returns, ddof=1)
    if downside_deviation == 0:
        return 0.0
    
    return (np.mean(excess_returns) / downside_deviation) * math.sqrt(bets_per_year)

def calculate_win_rate(outcomes: np.ndarray) -> float:
    """Calculate win rate (hit rate)"""
    if len(outcomes) == 0:
        return 0.0
    return float(np.mean(outcomes))

def calculate_avg_win_loss_pct(per_bet_rets: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    """Calculate average win/loss percentages"""
    if len(per_bet_rets) == 0:
        return 0.0, 0.0
    
    wins = per_bet_rets[outcomes > 0] * 100  # Convert to percentages
    losses = per_bet_rets[outcomes == 0] * 100
    
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    return avg_win, avg_loss

def calculate_profit_factor(per_bet_rets: np.ndarray, outcomes: np.ndarray) -> float:
    """Calculate profit factor (gross wins / gross losses)"""
    if len(per_bet_rets) == 0:
        return 0.0
    
    wins = per_bet_rets[outcomes > 0]
    losses = per_bet_rets[outcomes == 0]
    
    gross_wins = np.sum(wins) if len(wins) > 0 else 0.0
    gross_losses = -np.sum(losses) if len(losses) > 0 else 0.0  # Make positive
    
    if gross_losses == 0:
        return float('inf') if gross_wins > 0 else 0.0
    
    return gross_wins / gross_losses

def calculate_geometric_mean_return(per_bet_rets: np.ndarray) -> float:
    """Calculate geometric mean return"""
    if len(per_bet_rets) == 0:
        return 0.0
    
    # Convert returns to growth factors, calculate geometric mean
    growth_factors = 1.0 + per_bet_rets
    # Handle negative growth factors (bankruptcy)
    growth_factors = np.maximum(growth_factors, 1e-10)
    
    try:
        geo_mean = np.exp(np.mean(np.log(growth_factors))) - 1.0
        return float(geo_mean)
    except:
        return 0.0

def calculate_comprehensive_metrics(bankroll_ts: pd.Series, per_bet_rets: np.ndarray, outcomes: np.ndarray, 
                                  stake_fracs: np.ndarray, start_bankroll: float, bets_per_year: float, 
                                  min_stake: float, start_date: str, end_date: str) -> dict:
    """Calculate comprehensive metrics like deterministic backtest"""
    
    # Basic values
    final_bankroll = bankroll_ts.iloc[-1] if not bankroll_ts.empty else start_bankroll
    final_return_pct = ((final_bankroll / start_bankroll) - 1.0) * 100.0
    max_dd_pct = max_drawdown_pct(bankroll_ts)
    
    # Sharpe ratios
    sh_academic = sharpe_academic(per_bet_rets, bets_per_year)
    sh_industry = sharpe_industry(per_bet_rets, bets_per_year)
    sh_log = log_sharpe(per_bet_rets, bets_per_year)
    sh_weekly = calendar_sharpe(bankroll_ts, "W", 52)
    sh_monthly = calendar_sharpe(bankroll_ts, "ME", 12)
    
    # Additional metrics
    cagr = calculate_cagr(start_bankroll, final_bankroll, start_date, end_date)
    mar_ratio = calculate_mar_ratio(cagr, max_dd_pct)
    calmar_ratio = calculate_calmar_ratio(cagr, max_dd_pct)
    sortino = calculate_sortino_ratio(per_bet_rets, bets_per_year)
    
    # Win/Loss metrics
    win_rate = calculate_win_rate(outcomes)
    avg_win, avg_loss = calculate_avg_win_loss_pct(per_bet_rets, outcomes)
    profit_factor = calculate_profit_factor(per_bet_rets, outcomes)
    
    # Return metrics
    geometric_mean_ret = calculate_geometric_mean_return(per_bet_rets)
    exp_log_growth = expected_log_growth(per_bet_rets)
    
    # Risk metrics
    ruin = 1 if (bankroll_ts <= min_stake).any() else 0
    exposure = float(np.nanmean(stake_fracs)) if len(stake_fracs) else 0.0
    
    return {
        'final_return_pct': final_return_pct,
        'final_bankroll': final_bankroll,
        'max_dd_pct': max_dd_pct,
        'cagr': cagr * 100.0,  # Convert to percentage
        'mar_ratio': mar_ratio,
        'calmar_ratio': calmar_ratio,
        'sharpe_academic': sh_academic,
        'sharpe_industry': sh_industry,
        'sharpe_log': sh_log,
        'sharpe_weekly': sh_weekly,
        'sharpe_monthly': sh_monthly,
        'sortino_ratio': sortino,
        'win_rate': win_rate * 100.0,  # Convert to percentage
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'geometric_mean_return': geometric_mean_ret * 100.0,  # Convert to percentage
        'exp_log_growth': exp_log_growth,
        'exposure': exposure * 100.0,  # Convert to percentage
        'ruin': ruin,
        'num_bets': len(per_bet_rets),
        'implied_bets_per_year': bets_per_year
    }

# =========================
# Metrics Calculation
# =========================
def calculate_sim_metrics(bankroll_ts: pd.Series, per_bet_rets: np.ndarray, outcomes: np.ndarray, stake_fracs: np.ndarray,
                         start_bankroll: float, bets_per_year: float, min_stake: float) -> dict:
    """Calculate all metrics for a single simulation"""
    # Final return percentage
    final_bankroll = bankroll_ts.iloc[-1] if not bankroll_ts.empty else start_bankroll
    final_return_pct = ((final_bankroll / start_bankroll) - 1.0) * 100.0
    
    # Max drawdown percentage
    max_dd_pct = max_drawdown_pct(bankroll_ts)
    
    # Sharpe ratios
    sh_academic = sharpe_academic(per_bet_rets, bets_per_year)
    sh_industry = sharpe_industry(per_bet_rets, bets_per_year)
    sh_log = log_sharpe(per_bet_rets, bets_per_year)
    sh_weekly = calendar_sharpe(bankroll_ts, "W", 52)
    sh_monthly = calendar_sharpe(bankroll_ts, "ME", 12)
    
    # Other metrics
    hit_rate_val = hit_rate(outcomes)
    avg_win, avg_loss = avg_win_loss(per_bet_rets, outcomes)
    exp_log_growth = expected_log_growth(per_bet_rets)
    
    # Ruin (ever hit zero or below min_stake during the path)
    ruin = 1 if (bankroll_ts <= min_stake).any() else 0
    
    # Exposure (average stake fraction)
    exposure = float(np.nanmean(stake_fracs)) if len(stake_fracs) else 0.0
    
    return {
        'final_return_pct': final_return_pct,
        'max_dd_pct': max_dd_pct,
        'sharpe_academic': sh_academic,
        'sharpe_industry': sh_industry,
        'sharpe_log': sh_log,
        'sharpe_weekly': sh_weekly,
        'sharpe_monthly': sh_monthly,
        'hit_rate': hit_rate_val,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'exp_log_growth': exp_log_growth,
        'ruin': ruin,
        'exposure': exposure
    }

# =========================
# Core per-model runner
# =========================
def run_model(df: pd.DataFrame, model_name: str, args, rng) -> pd.DataFrame:
    """Run variance analysis for a single model"""
    # Apply edge filter (same as deterministic)
    base = df[df["edge"] >= args.edge].copy()
    if base.empty:
        print(f"  ⚠️  No bets after edge filter for {model_name}")
        return pd.DataFrame()

    base = base.sort_values("date").reset_index(drop=True)
    bpy = implied_bets_per_year(pd.to_datetime(base["date"]))

    # K values for Kelly sizing, or just one iteration for fixed sizing
    if args.sizing == "kelly":
        k_vals = np.round(np.arange(args.k_min, args.k_max + 1e-12, args.k_step), 2)
    else:
        k_vals = [1.0]  # Dummy k for fixed sizing
    
    sims = int(args.num_sims)
    summary_rows = []
    
    # Optional sim-level data storage
    sim_level_data = [] if getattr(args, 'save_sim_stats', False) else None

    print(f"  📊 Running {len(k_vals)} k-values × {sims:,} sims = {len(k_vals) * sims:,} total")
    
    for k in tqdm(k_vals, desc=f"  {model_name}"):
        # Set reproducible seed for this (model, k) combination
        model_k_seed = args.seed + stable_hash(model_name) + int(k * 100)
        model_rng = np.random.default_rng(model_k_seed)
        random.seed(model_k_seed)  # For pandas sampling
        
        # Collect metrics across simulations
        sim_metrics = []
        example_paths = []
        
        for s in range(sims):
            # Bootstrap sample
            if args.bootstrap == "iid":
                boot = sample_iid(base, model_rng)
            else:
                boot = sample_blocks(base, args.block_days, model_rng)

            # Simulate path
            bk_ts, per_bet_rets, outcomes, stake_fracs = simulate_path(
                boot,
                sizing=args.sizing,
                k=k,
                start_bankroll=args.start_bankroll,
                friction_bps=args.friction_bps,
                shortfall_policy=args.shortfall_policy,
                min_stake=args.min_stake,
                fixed_frac=args.fixed_frac,
                fixed_amt=args.fixed_amt,
            )

            # Calculate comprehensive metrics for this simulation (like deterministic)
            start_date = str(boot["date"].min())
            end_date = str(boot["date"].max())
            metrics = calculate_comprehensive_metrics(
                bk_ts, per_bet_rets, outcomes, stake_fracs, 
                args.start_bankroll, bpy, args.min_stake, start_date, end_date
            )
            metrics['sim_id'] = s
            metrics['k'] = k
            metrics['model'] = model_name
            sim_metrics.append(metrics)
            
            # Save simulation data if requested
            if sim_level_data is not None:
                sim_level_data.append(metrics.copy())
            
            # Save example paths
            if len(example_paths) < args.save_example_paths:
                # Weekly resample for example paths
                weekly_path = bk_ts.sort_index().asfreq("W", method="pad")
                example_paths.append(weekly_path)

        # Aggregate metrics across simulations
        metrics_df = pd.DataFrame(sim_metrics)
        
        def agg_stats(col):
            """Calculate aggregation statistics"""
            vals = metrics_df[col].dropna()
            if len(vals) == 0:
                return {f'{col}_mean': np.nan, f'{col}_median': np.nan, f'{col}_std': np.nan,
                       f'{col}_p05': np.nan, f'{col}_p25': np.nan, f'{col}_p75': np.nan, f'{col}_p95': np.nan}
            return {
                f'{col}_mean': float(vals.mean()),
                f'{col}_median': float(vals.median()),
                f'{col}_std': float(vals.std()),
                f'{col}_p05': float(vals.quantile(0.05)),
                f'{col}_p25': float(vals.quantile(0.25)),
                f'{col}_p75': float(vals.quantile(0.75)),
                f'{col}_p95': float(vals.quantile(0.95))
            }
        
        # Aggregate key metrics
        row = {
            "model": model_name,
            "sizing": args.sizing,
            "k": k if args.sizing == "kelly" else np.nan,
            "sims": sims,
            "edge_filter": args.edge,
            "friction_bps": args.friction_bps,
            "implied_bets_per_year": bpy,
            "ruin_prob": float(metrics_df['ruin'].mean()),
        }
        
        # Add aggregated statistics for all comprehensive metrics
        metrics_to_aggregate = [
            'final_return_pct', 'final_bankroll', 'max_dd_pct', 'cagr', 'mar_ratio', 'calmar_ratio',
            'sharpe_academic', 'sharpe_industry', 'sharpe_log', 'sharpe_weekly', 'sharpe_monthly', 
            'sortino_ratio', 'win_rate', 'avg_win_pct', 'avg_loss_pct', 'profit_factor',
            'geometric_mean_return', 'exp_log_growth', 'exposure'
        ]
        
        for metric in metrics_to_aggregate:
            if metric in metrics_df.columns:
                row.update(agg_stats(metric))
        
        summary_rows.append(row)
        
        # Save outputs for this k value
        save_k_outputs(model_name, k, args, metrics_df, example_paths)
    
    # Save simulation-level data if requested
    if sim_level_data and HAS_PARQUET:
        sim_df = pd.DataFrame(sim_level_data)
        sim_outdir = Path(args.outdir) / "sims"
        sim_outdir.mkdir(parents=True, exist_ok=True)
        sim_file = sim_outdir / f"{model_name}_{args.sizing}_simlevel_stats.parquet"
        sim_df.to_parquet(sim_file, index=False)
        print(f"  💾 Saved sim-level data: {sim_file}")

    # Create and save model summary
    model_summary = pd.DataFrame(summary_rows)
    summary_outdir = Path(args.outdir) / "summaries"
    summary_outdir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_outdir / f"{model_name}_{args.sizing}_bootstrap_summary.csv"
    model_summary.to_csv(summary_file, index=False)
    print(f"  💾 Saved summary: {summary_file}")
    
    # Generate plots
    if HAS_MPL:
        create_model_plots(model_name, model_summary, args)
    
    return model_summary

def save_k_outputs(model_name: str, k: float, args, metrics_df: pd.DataFrame, example_paths: list):
    """Save outputs for a specific k value"""
    k_str = f"k_{k:0.2f}" if args.sizing == "kelly" else "fixed"
    outdir = Path(args.outdir) / "detailed" / model_name / f"{args.sizing}_{k_str}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Save detailed metrics distribution
    metrics_df.to_csv(outdir / "sim_metrics.csv", index=False)
    
    # Save example paths
    if example_paths:
        # Align all paths to common date index
        all_indices = [p.index for p in example_paths]
        if all_indices:
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.union(idx)
            common_index = common_index.sort_values()
            
            path_data = []
            for i, path in enumerate(example_paths):
                aligned = path.reindex(common_index, method='pad')
                path_data.append(aligned.rename(f'path_{i+1}'))
            
            if path_data:
                example_df = pd.concat(path_data, axis=1)
                example_df.index.name = 'date'
                example_df.to_csv(outdir / "example_paths.csv")
    
    # Add per-k distribution histograms
    if HAS_MPL and len(metrics_df) > 0:
        for metric in ["final_return_pct", "max_dd_pct", "sharpe_academic"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(metrics_df[metric].dropna().values, bins=60, alpha=0.8)
            ax.set_title(f"{model_name} | {args.sizing} {k_str} | {metric.replace('_',' ').title()}")
            ax.set_xlabel(metric.replace('_',' ').title())
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / f"hist_{metric}.png", dpi=130)
            plt.close()

# =========================
# Plotting Functions
# =========================
def create_model_plots(model_name: str, summary_df: pd.DataFrame, args):
    """Create comprehensive plots for a model"""
    if not HAS_MPL:
        return
        
    plot_dir = Path(args.outdir) / "plots" / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    if HAS_MPL:
        try:
            sns.set_palette("husl")
        except:
            pass  # Continue without seaborn styling if it fails
    
    if args.sizing == "kelly" and len(summary_df) > 1:
        create_kelly_plots(model_name, summary_df, plot_dir)
    else:
        create_fixed_plots(model_name, summary_df, plot_dir)

def create_kelly_plots(model_name: str, summary_df: pd.DataFrame, plot_dir: Path):
    """Create Kelly-specific plots"""
    k_vals = summary_df['k'].values
    
    # Metrics to plot vs k
    metrics = {
        'final_return_pct': 'Final Return %',
        'max_dd_pct': 'Max Drawdown %', 
        'sharpe_academic': 'Academic Sharpe',
        'sharpe_industry': 'Industry Sharpe',
        'sharpe_weekly': 'Weekly Sharpe',
        'sharpe_monthly': 'Monthly Sharpe',
        'sharpe_log': 'Log Sharpe'
    }
    
    for metric_col, metric_name in metrics.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        mean_col = f'{metric_col}_mean'
        p05_col = f'{metric_col}_p05'
        p95_col = f'{metric_col}_p95'
        
        if all(col in summary_df.columns for col in [mean_col, p05_col, p95_col]):
            # Line plot with confidence bands
            ax.plot(k_vals, summary_df[mean_col], 'o-', linewidth=2, markersize=4, label='Mean')
            ax.fill_between(k_vals, summary_df[p05_col], summary_df[p95_col], 
                          alpha=0.3, label='P05-P95 Range')
            
            ax.set_xlabel('Kelly Multiplier (k)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{model_name}: {metric_name} vs Kelly Multiplier')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_dir / f'kelly_metric_{metric_col}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Ruin probability plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_vals, summary_df['ruin_prob'] * 100, 'ro-', linewidth=2, markersize=4)
    ax.set_xlabel('Kelly Multiplier (k)')
    ax.set_ylabel('Ruin Probability %')
    ax.set_title(f'{model_name}: Ruin Probability vs Kelly Multiplier')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'kelly_ruin_probability.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_fixed_plots(model_name: str, summary_df: pd.DataFrame, plot_dir: Path):
    """Create fixed sizing plots (histograms)"""
    if len(summary_df) != 1:
        return
        
    row = summary_df.iloc[0]
    
    # Create distribution plots for key metrics
    metrics = ['final_return_pct', 'max_dd_pct', 'sharpe_academic']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        mean_val = row.get(f'{metric}_mean', np.nan)
        std_val = row.get(f'{metric}_std', np.nan)
        p05_val = row.get(f'{metric}_p05', np.nan)
        p95_val = row.get(f'{metric}_p95', np.nan)
        
        if not np.isnan(mean_val) and not np.isnan(std_val):
            # Generate approximate distribution for visualization
            x = np.linspace(p05_val, p95_val, 100) if not np.isnan(p05_val) else np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
            y = np.exp(-0.5 * ((x - mean_val) / std_val) ** 2) / (std_val * np.sqrt(2 * np.pi))
            
            axes[i].plot(x, y, 'b-', linewidth=2)
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name}: Fixed Sizing Distributions')
    plt.tight_layout()
    plt.savefig(plot_dir / 'fixed_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

# =========================
# IO / Loading Functions
# =========================
def load_models_from_csvs(input_csvs: list, model_col: str, edge_threshold: float) -> dict[str, pd.DataFrame]:
    """Load models from multiple CSV files (mirrors deterministic behavior)"""
    all_data = []
    
    for path_str in input_csvs:
        path = Path(path_str)
        if not path.exists():
            print(f"⚠️  Warning: File not found: {path}")
            continue
            
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            
            # Verify required columns
            required_cols = ["date", "edge", "prob", "odds", "outcome"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"⚠️  Warning: Missing columns in {path}: {missing}")
                continue
            
            # Add model column if missing (derive from filename)
            if model_col not in df.columns:
                model_name = path.stem.replace("_pure_bets", "").replace("_bets", "")
                model_name = model_name.replace("_", " ").title()
                df[model_col] = model_name
            
            all_data.append(df)
            print(f"  ✅ Loaded {len(df):,} bets from {path.name}")
            
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
            continue
    
    if not all_data:
        return {}
    
    # Combine all data
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values("date").reset_index(drop=True)
    
    # Apply edge filter (same as deterministic)
    print(f"📊 Total bets before edge filter: {len(df_all):,}")
    df_all = df_all[df_all["edge"] >= edge_threshold]
    print(f"📊 Total bets after edge ≥ {edge_threshold:.1%}: {len(df_all):,}")
    
    # Group by model
    models = {}
    for model_name, group in df_all.groupby(model_col):
        models[str(model_name)] = group.sort_values("date").reset_index(drop=True)
        print(f"  📈 {model_name}: {len(group):,} bets")
    
    return models

def load_models_legacy(input_csv: Path, model_col: str) -> dict[str, pd.DataFrame]:
    """Legacy loading function for backward compatibility"""
    if not input_csv.exists():
        return {}
        
    df = pd.read_csv(input_csv, parse_dates=["date"])
    out = {}
    for m, g in df.groupby(model_col):
        out[str(m)] = g.sort_values("date").reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Monte Carlo / Bootstrap variance analysis for betting models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options (new multi-CSV support + backward compatibility)
    ap.add_argument("--input-csvs", nargs="+", 
                    help="List of model bet CSVs (pure bet logs)")
    ap.add_argument("--input-csv", type=Path,
                    help="Legacy: single CSV with all models (kept for backward compatibility)")
    ap.add_argument("--model-col", type=str, default="model")
    ap.add_argument("--edge", type=float, default=0.02, 
                    help="Minimum edge threshold (e.g. 0.02 = 2%)")

    # Sizing options (parity with deterministic)
    ap.add_argument("--sizing", type=str, default="kelly",
                    choices=["kelly", "fixed"])
    ap.add_argument("--k-min", type=float, default=0.01, 
                    help="Minimum Kelly multiplier")
    ap.add_argument("--k-max", type=float, default=1.00,
                    help="Maximum Kelly multiplier")
    ap.add_argument("--k-step", type=float, default=0.01,
                    help="Kelly multiplier step size")
    ap.add_argument("--fixed-frac", type=float, default=0.02,
                    help="Fixed fraction of bankroll per bet (e.g., 0.02 = 2%)")
    ap.add_argument("--fixed-amt", type=float, default=0.0,
                    help="Fixed stake amount per bet (overrides fixed-frac if > 0)")

    # Bankroll and risk management
    ap.add_argument("--start-bankroll", type=float, default=300.0)
    ap.add_argument("--shortfall-policy", type=str, default="all-in", 
                    choices=["all-in", "skip"])
    ap.add_argument("--min-stake", type=float, default=0.0,
                    help="Minimum stake threshold")
    ap.add_argument("--friction-bps", type=float, default=0.0,
                    help="Transaction costs in basis points")

    # Bootstrap options
    ap.add_argument("--bootstrap", type=str, default="iid", 
                    choices=["iid", "block"])
    ap.add_argument("--block-days", type=int, default=7,
                    help="Block size in days for block bootstrap")
    ap.add_argument("--num-sims", type=int, default=10000,
                    help="Number of bootstrap simulations")
    
    # Output options
    ap.add_argument("--save-example-paths", type=int, default=5,
                    help="Number of example paths to save per k")
    ap.add_argument("--save-sim-stats", action="store_true",
                    help="Save simulation-level statistics (large files)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--print-all", action="store_true")

    args = ap.parse_args()
    
    # Validate inputs
    if not args.input_csvs and not args.input_csv:
        ap.error("Must provide either --input-csvs or --input-csv")
    
    # Set global random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("="*88)
    print("VARIANCE / MONTE CARLO BOOTSTRAP ANALYSIS")
    print("="*88)
    print(f"Edge ≥ {args.edge*100:.1f}% | Sims {args.num_sims:,} | Bootstrap {args.bootstrap}")
    if args.bootstrap == "block":
        print(f"Block size: {args.block_days} days")
    
    sizing_desc = f"{args.sizing.title()} sizing"
    if args.sizing == "kelly":
        sizing_desc += f" (k: {args.k_min:.2f}–{args.k_max:.2f} step {args.k_step:.2f})"
    elif args.sizing == "fixed":
        if args.fixed_amt > 0:
            sizing_desc += f" (${args.fixed_amt:.2f}/bet)"
        else:
            sizing_desc += f" ({args.fixed_frac*100:.1f}% of bankroll)"
    print(f"Sizing: {sizing_desc}")
    
    print(f"Bankroll: ${args.start_bankroll:.0f} | Friction: {args.friction_bps:.1f} bps | Seed: {args.seed}")
    print(f"Charts: {'ON' if HAS_MPL else 'OFF (matplotlib/seaborn not found)'}")
    print(f"Parquet export: {'ON' if HAS_PARQUET else 'OFF (pyarrow not found)'}\n")

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    if args.input_csvs:
        print("📂 Loading models from multiple CSVs...")
        models = load_models_from_csvs(args.input_csvs, args.model_col, args.edge)
    else:
        print("📂 Loading models from single CSV...")
        models = load_models_legacy(args.input_csv, args.model_col)
        # Apply edge filter for legacy mode
        filtered_models = {}
        for name, df in models.items():
            filtered_df = df[df["edge"] >= args.edge]
            if not filtered_df.empty:
                filtered_models[name] = filtered_df
                print(f"  📈 {name}: {len(filtered_df):,} bets (after edge filter)")
        models = filtered_models
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return

    print(f"\n🚀 Running variance analysis on {len(models)} model(s)...\n")
    
    # Run analysis for each model
    all_summaries = []
    for name, dfm in models.items():
        print(f"🔄 Processing model: {name}")
        print(f"  📊 {len(dfm):,} bets after edge filter")
        
        if len(dfm) == 0:
            print("  ⚠️  Skipping (no bets)\n")
            continue
            
        # Initialize RNG for this model
        model_rng = np.random.default_rng(args.seed + stable_hash(name))
        
        summary = run_model(dfm, name, args, model_rng)
        if not summary.empty:
            all_summaries.append(summary)
            
            # Print comprehensive summary if requested
            if args.print_all:
                # Full Kelly sweep (like deterministic script)
                print(f"\n  📋 Full Kelly Sweep (ALL {len(summary)} k values):")
                print("  k     ret%    geo%    bust%   DD%     sharpe  sortino")
                for _, row in summary.iterrows():
                    k = row.get('k', 0.0)
                    ret_mean = row.get('final_return_pct_mean', 0.0)
                    geo_mean = row.get('geometric_mean_return_mean', 0.0)
                    bust = row.get('ruin_prob', 0.0) * 100
                    dd_mean = row.get('max_dd_pct_mean', 0.0)
                    sharpe = row.get('sharpe_academic_mean', 0.0)
                    sortino = row.get('sortino_ratio_mean', 0.0)
                    
                    print(f"  {k:.2f}  "
                          f"{ret_mean:+6.1f}%  "
                          f"{geo_mean:+6.1f}%  "
                          f"{bust:5.1f}%  "
                          f"{dd_mean:5.1f}%  "
                          f"{sharpe:6.2f}  "
                          f"{sortino:6.2f}")
                    
                # Also show compact comparison for key k values
                comparison_ks = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
                comparison_data = summary[summary['k'].isin(comparison_ks)].copy()
                
                if not comparison_data.empty and 'geometric_mean_return_mean' in summary.columns:
                    print("\n  📊 Key K Values Summary:")
                    print("  k     arith%   geo%    P5%     P95%    bust%   sharpe")
                    for _, row in comparison_data.iterrows():
                        k = row.get('k', 0.0)
                        arith = row.get('final_return_pct_mean', 0.0)
                        geo = row.get('geometric_mean_return_mean', 0.0)
                        p5 = row.get('final_return_pct_p05', 0.0)
                        p95 = row.get('final_return_pct_p95', 0.0)
                        bust = row.get('ruin_prob', 0.0) * 100
                        sharpe = row.get('sharpe_academic_mean', 0.0)
                        
                        print(f"  {k:.2f}  "
                              f"{arith:+6.1f}%  "
                              f"{geo:+6.1f}%  "
                              f"{p5:+6.1f}%  "
                              f"{p95:+6.1f}%  "
                              f"{bust:5.1f}%  "
                              f"{sharpe:6.2f}")
        
        print(f"  ✅ Completed {name}\n")

    # Save combined summary
    if all_summaries:
        combo = pd.concat(all_summaries, ignore_index=True)
        combo_file = Path(args.outdir) / f"ALL_{args.sizing}_variance_summary.csv"
        combo.to_csv(combo_file, index=False)
        print(f"💾 Combined summary: {combo_file}")
        
        # Print final summary with better metrics
        print(f"\n📊 FINAL RESULTS ({args.sizing.upper()} SIZING):")
        print("=" * 80)
        
        for model in combo['model'].unique():
            model_data = combo[combo['model'] == model]
            
            if args.sizing == "kelly":
                # Find optimal k using different criteria
                if 'mar_ratio_mean' in model_data.columns:
                    best_mar_k = model_data.loc[model_data['mar_ratio_mean'].idxmax(), 'k']
                    best_mar = model_data['mar_ratio_mean'].max()
                else:
                    best_mar_k, best_mar = 0.0, 0.0
                
                if 'geometric_mean_return_mean' in model_data.columns:
                    best_geo_k = model_data.loc[model_data['geometric_mean_return_mean'].idxmax(), 'k'] 
                    best_geo = model_data['geometric_mean_return_mean'].max()
                else:
                    best_geo_k, best_geo = 0.0, 0.0
                
                if 'cagr_mean' in model_data.columns:
                    best_cagr_k = model_data.loc[model_data['cagr_mean'].idxmax(), 'k']
                    best_cagr = model_data['cagr_mean'].max()
                else:
                    best_cagr_k, best_cagr = 0.0, 0.0
                
                print(f"{model}:")
                print(f"  Best MAR Ratio: {best_mar:.2f} at k={best_mar_k:.2f}")
                print(f"  Best Geometric Return: {best_geo:.1f}% at k={best_geo_k:.2f}")  
                print(f"  Best CAGR: {best_cagr:.1f}% at k={best_cagr_k:.2f}")
                
            else:
                # Fixed sizing summary
                if 'cagr_mean' in model_data.columns:
                    cagr = model_data['cagr_mean'].iloc[0]
                    mar = model_data['mar_ratio_mean'].iloc[0] if 'mar_ratio_mean' in model_data.columns else 0.0
                    ruin = model_data['ruin_prob'].iloc[0]
                    print(f"{model}: CAGR {cagr:.1f}%, MAR {mar:.2f}, Ruin {ruin:.1%}")
            print()
    
    print("\n✅ Variance analysis complete!")
    print(f"📁 Results saved to: {args.outdir}")

if __name__ == "__main__":
    main()
