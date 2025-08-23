#!/usr/bin/env python3
"""
Deterministic backtest (chronological) over PURE bet logs with extended metrics.

Outputs per model:
- CSV grid with k from 1%..100% (or --klist) including:
    * return_pct, max_dd_pct
    * avg_log_growth, geo_growth_per_bet
    * max_log_dd, time_in_dd_pct
    * ulcer_index_pct, ulcer_index_decimal, pain_ratio_log
    * sharpe, sortino, avg_simple_ret, median_simple_ret, std_simple_ret
    * max_gain_ret, max_loss_ret, max_gain_usd, max_loss_usd
    * avg_stake_frac_eff, avg_stake_usd
    * bets, bets_placed, skipped_floor
- Optional per-k bet logs (for representative k values)
- Equity & DD charts for representative k values

Supports:
- Fractional Kelly sweep (compounding) or custom list via --klist
- Optional cap on stake fraction (--max_fraction)
- Optional fixed sizing (--fixed_size): stake is based on a fixed base, not current bankroll
- Min stake floor (--min_stake). Set to 0 for theoretical Kelly.
- Shortfall policy when stake > bankroll: "skip" or "all-in"

Assumptions:
- analysis_scripts/pure_bet_logs/{model}_pure_bets.csv exists
- Columns: date, prob (0..1), odds (>1.01), outcome (1 win else 0), edge, bet_on_*
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------
# Config
# -------------------
REPO = Path(__file__).resolve().parents[2]
PURE_DIR = REPO / "analysis_scripts" / "pure_bet_logs"

BACKTEST_DIR = REPO / "analysis_scripts" / "backtests" / "deterministic"
RESULTS_DIR = BACKTEST_DIR / "results"
CHARTS_DIR = BACKTEST_DIR / "charts"
for p in [BACKTEST_DIR, RESULTS_DIR, CHARTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_MODELS = ["xgboost", "random_forest", "neural_network_143", "neural_network_98"]
START_BANKROLL_DEFAULT = 100.0
EDGE_MIN_DEFAULT = 0.02
KELLY_GRID = [i/100 for i in range(1, 101)]  # 1%..100%

REP_KS = {0.01, 0.05, 0.10, 0.25, 0.50, 1.00}

# -------------------
# Helpers
# -------------------
def load_pure(model: str) -> pd.DataFrame | None:
    f = PURE_DIR / f"{model}_pure_bets.csv"
    if not f.exists():
        print(f"  ⚠️  Missing: {f}")
        return None
    df = pd.read_csv(f, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    if "week_id" not in df.columns:
        week = df["date"].dt.isocalendar()
        df["week_id"] = week.year.astype(str) + "-" + week.week.astype(str).str.zfill(2)
    # sanity
    df = df[(df["prob"] > 0) & (df["prob"] < 1) & (df["odds"] > 1.01)]
    df = df.sort_values(["date", "player1", "player2"]).reset_index(drop=True)
    return df

def kelly_star(p: float, b: float) -> float:
    """Optimal Kelly fraction before user multiplier; f* = (bp - (1-p)) / b = p - (1-p)/b."""
    if b <= 0:
        return 0.0
    f = (b * p - (1.0 - p)) / b
    return max(0.0, f)

def compute_metrics_from_path(equity: np.ndarray) -> dict:
    """Compute classic + log-based drawdowns, time in DD, ulcer, etc."""
    if len(equity) == 0:
        return {
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
            "avg_log_growth": 0.0,
            "geo_growth_per_bet": 0.0,
            "max_log_dd": 0.0,
            "time_in_dd_pct": 0.0,
            "ulcer_index_pct": 0.0,
            "ulcer_index_decimal": 0.0,
            "pain_ratio_log": 0.0,
        }

    # Returns (path-based)
    ret_pct = (equity[-1] - equity[0]) / equity[0] * 100.0

    # Classic % drawdowns
    peaks = np.maximum.accumulate(np.maximum(equity[0], equity))
    dd = (peaks - equity) / peaks  # decimal
    max_dd_pct = dd.max() * 100.0 if dd.size else 0.0

    # Log space metrics
    log_eq = np.log(np.maximum(equity, 1e-12))
    log_peaks = np.maximum.accumulate(np.maximum(log_eq[0], log_eq))
    log_dd = log_peaks - log_eq  # in nats
    max_log_dd = float(log_dd.max()) if log_dd.size else 0.0

    # Avg log growth per bet & geometric growth per bet
    log_steps = np.diff(log_eq)
    avg_log_growth = float(np.mean(log_steps)) if log_steps.size else 0.0
    geo_growth_per_bet = float(np.exp(avg_log_growth) - 1.0)

    # Time in drawdown
    time_in_dd_pct = float(np.mean(dd > 0) * 100.0) if dd.size else 0.0

    # Ulcer index
    ulcer_index_decimal = float(np.sqrt(np.mean(np.square(dd)))) if dd.size else 0.0
    ulcer_index_pct = ulcer_index_decimal * 100.0

    # Pain ratio (log)
    pain_ratio_log = float(avg_log_growth / ulcer_index_decimal) if ulcer_index_decimal > 0 else 0.0

    return {
        "return_pct": float(ret_pct),
        "max_dd_pct": float(max_dd_pct),
        "avg_log_growth": avg_log_growth,
        "geo_growth_per_bet": geo_growth_per_bet,
        "max_log_dd": max_log_dd,
        "time_in_dd_pct": time_in_dd_pct,
        "ulcer_index_pct": ulcer_index_pct,
        "ulcer_index_decimal": ulcer_index_decimal,
        "pain_ratio_log": pain_ratio_log,
    }

def run_path(rows: pd.DataFrame,
             k_mult: float,
             max_fraction: float | None,
             fixed_size: bool,
             start_bankroll: float,
             fixed_base: float,
             shortfall_policy: str,
             min_stake: float) -> tuple[pd.DataFrame, dict]:
    """
    Simulate one deterministic path. Returns (bet_log_df, metrics_dict).
    shortfall_policy: "skip" (default) or "all-in" when stake > bankroll.
    min_stake: minimum $ stake to place a bet; set 0 for theoretical Kelly.
    """
    bankroll = start_bankroll
    equity_list = [bankroll]

    logs = []
    skipped_floor = 0

    for idx, r in rows.iterrows():
        p = float(r["prob"])
        b = float(r["odds"]) - 1.0
        outcome = int(r["outcome"])  # 1 if we win

        # Check for bankruptcy FIRST (before calculating stakes)
        if bankroll < min_stake:
            equity_list.append(bankroll)
            break  # Stop completely when bankrupt

        f_star = kelly_star(p, b)
        stake_frac = k_mult * f_star
        if max_fraction is not None:
            stake_frac = min(stake_frac, max_fraction)
        stake_frac = max(0.0, min(stake_frac, 1.0))
        if stake_frac <= 0:
            equity_list.append(bankroll)
            continue

        # Base for stake sizing
        bankroll_base = fixed_base if fixed_size else bankroll
        stake = bankroll_base * stake_frac

        # Apply min stake floor to the calculated stake
        if min_stake > 0 and stake < min_stake:
            stake = min_stake

        # Shortfall handling if stake > bankroll
        if stake > bankroll:
            if shortfall_policy == "skip":
                equity_list.append(bankroll)
                continue
            elif shortfall_policy == "all-in":
                stake = bankroll
            else:
                raise ValueError("shortfall_policy must be 'skip' or 'all-in'")

        before = bankroll
        if outcome == 1:
            profit = stake * b
            bankroll = bankroll + profit
            won = True
        else:
            profit = -stake
            bankroll = bankroll + profit
            won = False

        # Per-bet simple return (relative to current bankroll at entry)
        simple_ret = profit / before
        # Effective stake fraction vs current bankroll at entry
        stake_frac_eff = stake / before if before > 0 else 0.0

        logs.append({
            "idx": idx,
            "date": r.get("date"),
            "player1": r.get("player1",""),
            "player2": r.get("player2",""),
            "bet_on_player": r.get("bet_on_player",""),
            "bet_on_p1": r.get("bet_on_p1",""),
            "prob": p,
            "market_prob": r.get("market_prob",""),
            "edge": r.get("edge",""),
            "odds": r.get("odds",""),
            "kelly_star": f_star,
            "k_multiplier": k_mult,
            "stake_frac_nominal": stake_frac,
            "stake_frac_eff": stake_frac_eff,
            "fixed_size_mode": fixed_size,
            "bankroll_before": before,
            "stake": stake,
            "won": won,
            "profit": profit,
            "simple_return": simple_ret,
            "bankroll_after": bankroll,
        })

        equity_list.append(bankroll)

    betlog = pd.DataFrame(logs)
    equity = np.array(equity_list, dtype=float)

    # Path-based summaries
    base_metrics = compute_metrics_from_path(equity)

    # Per-bet statistics (only for placed bets)
    if not betlog.empty:
        rets_simple = betlog["simple_return"].values

        # --- Log returns per bet (path-consistent) ---
        # Stop at ruin to avoid log(0) artifacts
        eps = 1e-12
        br_before = betlog["bankroll_before"].values
        br_after  = betlog["bankroll_after"].values
        alive_mask = (br_before > eps) & (br_after > eps)
        # Cut at first ruin if it occurs
        if np.any(~alive_mask):
            cutoff = np.argmax(~alive_mask)  # first False
            br_before = br_before[:cutoff]
            br_after  = br_after[:cutoff]

        if br_before.size > 0:
            rets_log = np.log(np.maximum(br_after, eps) / np.maximum(br_before, eps))
        else:
            rets_log = np.array([])

        wins = betlog["won"].values[:br_before.size] if br_before.size else np.array([])
        win_rate_pct = float(wins.mean() * 100.0) if wins.size else 0.0

        avg_simple_ret   = float(np.mean(rets_simple)) if rets_simple.size else 0.0
        median_simple_ret= float(np.median(rets_simple)) if rets_simple.size else 0.0
        std_simple_ret   = float(np.std(rets_simple)) if rets_simple.size else 0.0

        # Downside std for simple returns (kept for reference)
        downside_simple = rets_simple[rets_simple < 0]
        downside_std_simple = float(np.std(downside_simple)) if downside_simple.size else 0.0

        # --- Sharpe / Sortino on ARITHMETIC returns (leverage-invariant, per-bet) ---
        if rets_simple.size >= 2 and np.std(rets_simple) > 0:
            mean_ret = float(np.mean(rets_simple))
            std_ret = float(np.std(rets_simple))

            # Annualize by bets per year using date span
            if "date" in betlog.columns and betlog["date"].notna().any():
                d0 = pd.to_datetime(betlog["date"].iloc[0])
                d1 = pd.to_datetime(betlog["date"].iloc[-1])
                years = max((d1 - d0).days / 365.25, 1e-6)
                bets_per_year = rets_simple.size / years
            else:
                bets_per_year = 1000.0  # fallback

            # Academic Sharpe (leverage-invariant): (μ/σ) × √(bets/year)
            sharpe_per_bet_arith = float((mean_ret - 0.0) / std_ret * np.sqrt(bets_per_year))
            
            # Industry Sharpe (hedge fund marketing): (μ × bets/year) / σ
            sharpe_industry = float((mean_ret * bets_per_year) / std_ret)

            # Sortino (academic style): (μ/σ_downside) × √(bets/year)
            downside_rets = rets_simple[rets_simple < 0]
            if downside_rets.size > 0 and np.std(downside_rets) > 0:
                sortino_per_bet_arith = float((mean_ret - 0.0) / np.std(downside_rets) * np.sqrt(bets_per_year))
            else:
                sortino_per_bet_arith = 0.0
        else:
            sharpe_per_bet_arith = 0.0
            sharpe_industry = 0.0
            sortino_per_bet_arith = 0.0
            bets_per_year = 0.0

        # --- Per-bet LOG Sharpe (sensitive to over-leverage/near-ruin) ---
        # Log Sharpe: (μ_log/σ_log) × √(bets/year) - penalizes large losses more heavily
        if rets_log.size >= 2 and np.std(rets_log) > 0:
            mean_log = float(np.mean(rets_log))
            std_log  = float(np.std(rets_log))
            sharpe_per_bet_log = float(mean_log / std_log * np.sqrt(bets_per_year))
        else:
            sharpe_per_bet_log = 0.0

        # --- Calendarized (industry-style) Sharpe/Sortino ---
        # Calendar Sharpe: resampled to weekly/monthly periods, then (μ/σ) × √(periods/year)
        sharpe_weekly,  sortino_weekly,  _ = calendarized_sharpe(betlog, start_bankroll, freq="W")
        sharpe_monthly, sortino_monthly, _ = calendarized_sharpe(betlog, start_bankroll, freq="ME")

        max_gain_ret = float(np.max(rets_simple)) if rets_simple.size else 0.0
        max_loss_ret = float(np.min(rets_simple)) if rets_simple.size else 0.0
        max_gain_usd = float(np.max(betlog["profit"])) if not betlog.empty else 0.0
        max_loss_usd = float(np.min(betlog["profit"])) if not betlog.empty else 0.0

        avg_stake_frac_eff = float(np.mean(betlog["stake_frac_eff"])) if not betlog.empty else 0.0
        avg_stake_usd = float(np.mean(betlog["stake"])) if not betlog.empty else 0.0
    else:
        win_rate_pct = 0.0
        avg_simple_ret = median_simple_ret = std_simple_ret = 0.0
        downside_std_simple = 0.0
        sharpe_per_bet_arith = sortino_per_bet_arith = 0.0
        sharpe_industry = 0.0
        sharpe_per_bet_log = 0.0
        sharpe_weekly = sortino_weekly = 0.0
        sharpe_monthly = sortino_monthly = 0.0
        max_gain_ret = max_loss_ret = 0.0
        max_gain_usd = max_loss_usd = 0.0
        avg_stake_frac_eff = avg_stake_usd = 0.0

    metrics = {
        "bets": int(len(rows)),
        "bets_placed": int(len(betlog)),
        "skipped_floor": int(skipped_floor),
        "final_bankroll": float(equity[-1] if equity.size else start_bankroll),

        # Path-based
        **base_metrics,

        # Per-bet
        "win_rate_pct": float(win_rate_pct),
        "avg_simple_ret": float(avg_simple_ret),
        "median_simple_ret": float(median_simple_ret),
        "std_simple_ret": float(std_simple_ret),
        "downside_std_simple": float(downside_std_simple),
        
        # Multiple Sharpe/Sortino metrics with transparency
        "bets_per_year": float(bets_per_year) if 'bets_per_year' in locals() else 0.0,
        "sharpe_per_bet_arith": float(sharpe_per_bet_arith),     # leverage-invariant; per-bet arithmetic
        "sharpe_industry": float(sharpe_industry),               # industry-standard hedge fund style
        "sortino_per_bet_arith": float(sortino_per_bet_arith),
        
        "sharpe_per_bet_log": float(sharpe_per_bet_log),         # per-bet log-return Sharpe (penalizes big losses)
        
        "sharpe_weekly": float(sharpe_weekly),                   # calendarized, industry-style
        "sortino_weekly": float(sortino_weekly),
        "sharpe_monthly": float(sharpe_monthly),
        "sortino_monthly": float(sortino_monthly),
        "max_gain_ret": float(max_gain_ret),
        "max_loss_ret": float(max_loss_ret),
        "max_gain_usd": float(max_gain_usd),
        "max_loss_usd": float(max_loss_usd),
        "avg_stake_frac_eff": float(avg_stake_frac_eff),
        "avg_stake_usd": float(avg_stake_usd),
    }
    return betlog, metrics

def plot_equity(df: pd.DataFrame, out_png: Path, title: str, start_bankroll: float):
    if df.empty:
        return
    x = np.arange(0, len(df)+1)
    y = np.concatenate([[start_bankroll], df["bankroll_after"].values])
    plt.figure(figsize=(11,6))
    plt.plot(x, y, linewidth=2)
    plt.axhline(start_bankroll, color="black", alpha=0.4, linestyle="--", linewidth=1)
    plt.xlabel("Bet #"); plt.ylabel("Bankroll ($)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

def calendarized_sharpe(betlog: pd.DataFrame, start_bankroll: float, freq: str = "W"):
    """
    freq: 'W' for weekly, 'M' for monthly.
    Returns (sharpe, sortino, n_periods).
    """
    if betlog.empty or betlog["date"].isna().all():
        return 0.0, 0.0, 0

    df = betlog[["date", "bankroll_after"]].dropna(subset=["date"]).sort_values("date").copy()

    # prepend a start point so the first resample has a level to pct_change from
    first_date = pd.to_datetime(df["date"].iloc[0]) - pd.Timedelta(seconds=1)
    df0 = pd.DataFrame({"date": [first_date], "bankroll_after": [start_bankroll]})
    df = pd.concat([df0, df], ignore_index=True).set_index("date")

    # resample to calendar frequency and forward-fill the latest equity
    df = df.resample(freq).last().ffill()

    rets = df["bankroll_after"].pct_change().dropna().values
    if rets.size < 2 or np.std(rets) == 0:
        return 0.0, 0.0, int(rets.size)

    mean = float(np.mean(rets))
    std = float(np.std(rets))
    # Handle both old and new pandas frequency codes
    if freq.upper().startswith("W"):
        periods_per_year = 52
    elif freq.upper() in ["M", "ME"]:
        periods_per_year = 12
    else:
        periods_per_year = 52  # default to weekly

    sharpe = mean / std * np.sqrt(periods_per_year)

    downside = rets[rets < 0]
    sortino = float(mean / np.std(downside) * np.sqrt(periods_per_year)) if downside.size and np.std(downside) > 0 else 0.0

    return float(sharpe), float(sortino), int(rets.size)

def plot_drawdown(df: pd.DataFrame, out_png: Path, title: str, start_bankroll: float):
    if df.empty:
        return
    equity = np.concatenate([[start_bankroll], df["bankroll_after"].values])
    peaks = np.maximum.accumulate(np.maximum(start_bankroll, equity))
    dd = (peaks - equity) / peaks
    plt.figure(figsize=(11,3.5))
    plt.plot(np.arange(0, len(dd)), dd*100.0, linewidth=1.8)
    plt.xlabel("Bet #"); plt.ylabel("Drawdown (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Models to run")
    ap.add_argument("--edge", type=float, default=EDGE_MIN_DEFAULT, help="Minimum edge (e.g., 0.02)")
    ap.add_argument("--klist", type=str, default="", help="Comma list of Kelly multipliers (e.g. 0.05,0.10,0.25)")
    ap.add_argument("--max_fraction", type=float, default=None, help="Cap on stake fraction, e.g., 0.05")

    # Compounding (fractional Kelly) vs fixed-size
    ap.add_argument("--fixed_size", action="store_true", help="Bet based on fixed base (no compounding sizing)")
    ap.add_argument("--fixed-base", type=float, default=100.0, help="Base used in fixed-size mode")
    ap.add_argument("--fixed-start-bankroll", type=float, default=300.0, help="Starting bankroll for fixed-size runs")

    # Compounding start bankroll
    ap.add_argument("--start-bankroll", type=float, default=START_BANKROLL_DEFAULT, help="Starting bankroll for compounding runs")

    # Execution realism vs theoretical
    ap.add_argument("--min_stake", type=float, default=0.10, help="Minimum stake to place a bet; set 0 for theoretical Kelly")

    # Shortfall policy if stake > bankroll
    ap.add_argument("--shortfall-policy", type=str, default="all-in", choices=["skip", "all-in"], help="Handle stake>bankroll")

    # Printing options
    ap.add_argument("--print-all", action="store_true", help="Print full 1..100% sweep lines")
    ap.add_argument("--save-all-k", action="store_true", help="Save bet logs and charts for ALL Kelly fractions (not just representative ones)")
    args = ap.parse_args()

    # Kelly grid
    if args.klist.strip():
        kmults = [float(x) for x in args.klist.split(",")]
    else:
        kmults = KELLY_GRID

    # Header
    print("="*88)
    print("DETERMINISTIC BACKTEST (chronological)")
    print("="*88)
    print(f"Edge filter: ≥ {args.edge:.2%} | Kelly points: {len(kmults)} | Fixed sizing: {args.fixed_size}")
    if args.fixed_size:
        print(f"Fixed base: {args.fixed_base:.2f} | Fixed start bankroll: {args.fixed_start_bankroll:.2f}")
    else:
        print(f"Start bankroll: {args.start_bankroll:.2f}")
    print(f"Shortfall policy: {args.shortfall_policy}")
    print(f"Min stake: {args.min_stake:.2f}")
    print()

    combined_rows = []
    for model in args.models:
        label = model.replace("_"," ").title()
        print(f"\nMODEL: {label}")
        df = load_pure(model)
        if df is None or df.empty:
            print("  ❌ No data. Skipping."); continue

        df_use = df[df["edge"] >= args.edge].copy()
        print(f"  ✅ After edge filter: {len(df_use):,} bets")

        # Output folders per model
        model_res = RESULTS_DIR / model
        model_png = CHARTS_DIR / model
        model_res.mkdir(parents=True, exist_ok=True)
        model_png.mkdir(parents=True, exist_ok=True)

        # Save the filtered input for traceability
        df_use.to_csv(model_res / f"{model}_filtered_input.csv", index=False)

        grid = []
        for k in kmults:
            # choose bankroll & base depending on mode
            if args.fixed_size:
                start_bankroll = args.fixed_start_bankroll
                fixed_base = args.fixed_base
            else:
                start_bankroll = args.start_bankroll
                fixed_base = args.start_bankroll  # unused in compounding, but harmless

            betlog, metrics = run_path(
                rows=df_use,
                k_mult=k,
                max_fraction=args.max_fraction,
                fixed_size=args.fixed_size,
                start_bankroll=start_bankroll,
                fixed_base=fixed_base,
                shortfall_policy=args.shortfall_policy,
                min_stake=args.min_stake
            )

            row = {
                "model": model,
                "kelly_multiplier": k,
                **metrics
            }
            grid.append(row)

            # Save per-bet log + charts for representative k's OR all k's if requested
            save_this_k = k in REP_KS or args.save_all_k
            if save_this_k:
                suffix = ("_fixed" if args.fixed_size else "") + ("" if args.min_stake == 0 else "_floor")
                out_csv = model_res / f"{model}_betlog_k{k:.2f}{suffix}.csv"
                betlog.to_csv(out_csv, index=False)

                ttl = f"{label} — k={k:.2f}{' (fixed size)' if args.fixed_size else ''}{' — floor' if args.min_stake>0 else ' — theoretical'}"
                plot_equity(betlog, model_png / f"{model}_equity_k{k:.2f}{suffix}.png", ttl, start_bankroll)
                plot_drawdown(betlog, model_png / f"{model}_dd_k{k:.2f}{suffix}.png", ttl + " — Drawdown", start_bankroll)

        grid_df = pd.DataFrame(grid).sort_values("kelly_multiplier").reset_index(drop=True)
        out_grid = model_res / f"{model}_deterministic_grid{'_fixed' if args.fixed_size else ''}{'' if args.min_stake==0 else '_floor'}.csv"
        grid_df.to_csv(out_grid, index=False)

        # Print compact 1–10% band with bullet-proof headers
        band = grid_df[(grid_df["kelly_multiplier"]>=0.01)&(grid_df["kelly_multiplier"]<=0.10)].copy()
        if not band.empty:
            print("  SHARPE RATIO METRICS [¹Academic (2-3 decimals) | ²Industry/Marketing (1 decimal)]:")
            print(f"  • Academic Sharpe: (μ/σ)×√(bets/yr) - leverage invariant, penalizes volatility")
            print(f"  • Industry Sharpe: (μ×bets/yr)/σ - hedge fund marketing style, no √ scaling")
            print(f"  • Log Sharpe: log returns, penalizes large losses more heavily")
            print(f"  • Calendar Sharpe: resampled to periods, then (μ/σ)×√(periods/yr)")
            print()
            
            # Show implied bets/year for first row (should be same for all in deterministic backtest)
            if 'bets_per_year' in band.columns and not band.empty:
                bpy = band.iloc[0]['bets_per_year']
                print(f"  Implied bets/year: {bpy:.1f} (based on date span)")
            
            print("  Kelly 1–10% summary:")
            print("  k     ret%    DD%   Academic¹  Industry²  Log¹     Weekly¹   Monthly¹")
            for _, r in band.iterrows():
                print(f"  {r['kelly_multiplier']:.2f}  "
                      f"{r['return_pct']:+5.1f}%  "
                      f"{r['max_dd_pct']:4.1f}%  "
                      f"{r['sharpe_per_bet_arith']:7.2f}   "
                      f"{r['sharpe_industry']:7.1f}   "
                      f"{r['sharpe_per_bet_log']:6.2f}   "
                      f"{r['sharpe_weekly']:6.2f}   "
                      f"{r['sharpe_monthly']:7.2f}")

        if args.print_all:
            print("  Full Kelly sweep (all k values):")
            print("  k     ret%    DD%   Academic¹  Industry²  Log¹     Weekly¹   Monthly¹")
            for _, r in grid_df.iterrows():
                print(f"  {r['kelly_multiplier']:.2f}  "
                      f"{r['return_pct']:+5.1f}%  "
                      f"{r['max_dd_pct']:4.1f}%  "
                      f"{r['sharpe_per_bet_arith']:7.2f}   "
                      f"{r['sharpe_industry']:7.1f}   "
                      f"{r['sharpe_per_bet_log']:6.2f}   "
                      f"{r['sharpe_weekly']:6.2f}   "
                      f"{r['sharpe_monthly']:7.2f}")

        combined_rows.extend(grid)

    if combined_rows:
        all_df = pd.DataFrame(combined_rows)
        out_all = RESULTS_DIR / f"ALL_deterministic_grid{'_fixed' if args.fixed_size else ''}{'' if args.min_stake==0 else '_floor'}.csv"
        (all_df
         .sort_values(["model","kelly_multiplier"])
         .to_csv(out_all, index=False))
        print(f"\n💾 Combined grid written to: {out_all}")

    # Explanatory footnotes
    print("\n" + "="*88)
    print("SHARPE RATIO METHODOLOGY")
    print("="*88)
    print("¹ Academic Sharpe: (μ/σ) × √(annualization_factor)")
    print("  • Standard in academic finance literature")
    print("  • Leverage-invariant for Kelly strategies")
    print("  • Typical range: 0.5-4.0 for good strategies")
    print()
    print("² Industry Sharpe: (μ × annualization_factor) / σ")
    print("  • Common in hedge fund marketing materials")
    print("  • Produces higher, more impressive numbers")
    print("  • Typical range: 10-200+ for the same strategies")
    print()
    print("Both metrics measure risk-adjusted performance but use different scaling.")
    print("Academic Sharpe is more conservative and widely accepted in research.")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
