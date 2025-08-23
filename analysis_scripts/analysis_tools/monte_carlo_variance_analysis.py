#!/usr/bin/env python3
"""
MONTE CARLO VARIANCE (from PURE bet logs)

Modes:
  - bootstrap       : sample weeks WITH replacement (duplicates allowed). Optional intra-week shuffle.
  - permute_weeks   : permute weeks WITHOUT replacement (no duplicates). Optional intra-week shuffle.
  - permute_matches : permute ALL matches WITHOUT replacement (no duplicates). Ignores weekly structure.

Paths:
  - --common_paths reuses the SAME sampled paths across all Kelly fractions
    (supported for bootstrap & permute_weeks; permute_matches generates per-sim on the fly to avoid huge memory).

Inputs (from analysis_scripts/pure_bet_logs/*.csv):
  date, week_id, player1, player2, bet_on_player, bet_on_p1,
  prob, market_prob, edge, odds, outcome, model, avgw, avgl

Outputs:
  analysis_scripts/variance_analysis/results/{model}_kelly_grid.csv
  analysis_scripts/variance_analysis/results/all_models_kelly_grid.csv
  analysis_scripts/variance_analysis/charts/*
  (optional) results/samples_*/*.csv when exporting paths/topK

Run examples:
  # 1) bootstrap, keep intra-week order, allow duplicates, same paths across Kelly, symlog
  python .../monte_carlo_variance_analysis.py --mode bootstrap --sims 10000 --edge 0.02 --common_paths --symlog

  # 2) permute weeks w/o replacement, shuffle inside weeks
  python .../monte_carlo_variance_analysis.py --mode permute_weeks --shuffle_within_week --sims 10000

  # 3) permute all matches w/o replacement
  python .../monte_carlo_variance_analysis.py --mode permute_matches --sims 10000
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------
# Config
# -------------------
REPO = Path(__file__).resolve().parents[2]
PURE_DIR = REPO / "analysis_scripts" / "pure_bet_logs"
OUT_DIR = REPO / "analysis_scripts" / "variance_analysis"
CHARTS_DIR = OUT_DIR / "charts"
RESULTS_DIR = OUT_DIR / "results"
SAMPLES_DIR = RESULTS_DIR / "samples"
for p in [OUT_DIR, CHARTS_DIR, RESULTS_DIR, SAMPLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MODELS = ["xgboost", "random_forest", "neural_network_143", "neural_network_98"]

START_BANKROLL = 100.0
EDGE_MIN_DEFAULT = 0.02
SIMS_DEFAULT = 10_000
KELLY_GRID = [i/100 for i in range(1, 101)]  # 0.01..1.00


# -------------------
# Helpers
# -------------------
def load_pure(model: str) -> pd.DataFrame | None:
    f = PURE_DIR / f"{model}_pure_bets.csv"
    if not f.exists():
        print(f"  ⚠️  Missing pure log for {model}: {f}")
        return None
    df = pd.read_csv(f, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    if "week_id" not in df.columns:
        week = df["date"].dt.isocalendar()
        df["week_id"] = week.year.astype(str) + "-" + week.week.astype(str).str.zfill(2)
    # basic sanity filters
    df = df[(df["prob"] > 0) & (df["prob"] < 1) & (df["odds"] > 1.01)]
    return df


def prepare_weeks(df: pd.DataFrame, edge_min: float) -> list[pd.DataFrame]:
    d = df[(df["edge"] >= edge_min)].sort_values(["week_id", "date", "player1", "player2"]).reset_index(drop=True)
    weeks = []
    for _, wdf in d.groupby("week_id", sort=True):
        weeks.append(wdf.reset_index(drop=True))
    return weeks


def flatten_weeks(weeks: list[pd.DataFrame]) -> pd.DataFrame:
    if not weeks:
        return pd.DataFrame(columns=["prob","odds","outcome"])
    return pd.concat(weeks, ignore_index=True)


def kelly_fraction(p: float, b: float) -> float:
    # Kelly f* = (b*p - (1-p)) / b ; floor at 0
    if b <= 0:
        return 0.0
    f = (b * p - (1.0 - p)) / b
    return max(0.0, f)


# -------------------
# Sampling recipes
# -------------------
def sample_paths_bootstrap(weeks, sims, rng, shuffle_within_week: bool):
    n_weeks = len(weeks)
    week_idx = rng.integers(0, n_weeks, size=(sims, n_weeks))  # with replacement
    perms = None
    if shuffle_within_week:
        perms = []
        for w in weeks:
            if len(w) <= 1:
                perms.append(None)
            else:
                perms.append(np.array([rng.permutation(len(w)) for _ in range(sims)]))
    return week_idx, perms


def sample_paths_permute_weeks(weeks, sims, rng, shuffle_within_week: bool):
    n_weeks = len(weeks)
    # each sim gets a fresh permutation of weeks (no duplicates)
    week_idx = np.vstack([rng.permutation(n_weeks) for _ in range(sims)])
    perms = None
    if shuffle_within_week:
        perms = []
        for w in weeks:
            if len(w) <= 1:
                perms.append(None)
            else:
                perms.append(np.array([rng.permutation(len(w)) for _ in range(sims)]))
    return week_idx, perms


# -------------------
# Core sim
# -------------------
def run_sim_sequence(rows: pd.DataFrame, frac: float, max_fraction: float) -> tuple[float, float]:
    """Run a single simulation over an already-ordered sequence of rows."""
    bankroll = START_BANKROLL
    peak = START_BANKROLL
    max_dd = 0.0

    for _, r in rows.iterrows():
        p = float(r["prob"])
        b = float(r["odds"]) - 1.0

        f_star = kelly_fraction(p, b)
        stake_frac = frac * f_star
        if max_fraction is not None:
            stake_frac = min(stake_frac, max_fraction)
        stake_frac = max(0.0, min(stake_frac, 1.0))

        if stake_frac <= 0 or bankroll <= 0:
            continue

        stake = bankroll * stake_frac
        if int(r["outcome"]) == 1:
            bankroll += stake * b
        else:
            bankroll -= stake

        peak = max(peak, bankroll)
        if peak > 0:
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd

        if bankroll <= 1e-9:
            bankroll = 0.0
            break

    return bankroll, max_dd


def simulate(
    mode: str,
    weeks: list[pd.DataFrame],
    frac: float,
    sims: int,
    rng: np.random.Generator,
    *,
    shuffle_within_week: bool,
    common_week_idx=None,
    common_perms=None,
    max_fraction: float | None = None,
    export_samples: int = 0,
    export_topk: int = 0,
    model_name: str = "",
    kelly_fraction: float = 0.0,
):
    """Return stats dict (and optionally export sample/topK path CSVs)."""
    if not weeks:
        return None

    n_weeks = len(weeks)
    all_rows = flatten_weeks(weeks)  # used by permute_matches

    final_bankrolls = np.zeros(sims, dtype=float)
    returns = np.zeros(sims, dtype=float)
    drawdowns = np.zeros(sims, dtype=float)

    # path exports (lightweight: only bankroll trajectory summary)
    sample_records = []

    for s in range(sims):
        # choose sequence
        if mode == "bootstrap":
            if common_week_idx is None:
                idx = rng.integers(0, n_weeks, size=n_weeks)
            else:
                idx = common_week_idx[s]
            seq = []
            for wi in idx:
                w = weeks[wi]
                if shuffle_within_week:
                    if common_perms is not None and common_perms[wi] is not None:
                        w_use = w.iloc[common_perms[wi][s]]
                    elif len(w) > 1:
                        w_use = w.iloc[rng.permutation(len(w))]
                    else:
                        w_use = w
                else:
                    w_use = w
                seq.append(w_use)
            sim_rows = pd.concat(seq, ignore_index=True)

        elif mode == "permute_weeks":
            if common_week_idx is None:
                idx = rng.permutation(n_weeks)
            else:
                idx = common_week_idx[s]
            seq = []
            for wi in idx:
                w = weeks[wi]
                if shuffle_within_week:
                    if common_perms is not None and common_perms[wi] is not None:
                        w_use = w.iloc[common_perms[wi][s]]
                    elif len(w) > 1:
                        w_use = w.iloc[rng.permutation(len(w))]
                    else:
                        w_use = w
                else:
                    w_use = w
                seq.append(w_use)
            sim_rows = pd.concat(seq, ignore_index=True)

        elif mode == "permute_matches":
            # global permutation of all matches (no duplicates). We don’t pre-store all perms (memory heavy).
            perm = rng.permutation(len(all_rows))
            sim_rows = all_rows.iloc[perm]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # run one path
        bankroll, max_dd = run_sim_sequence(sim_rows, frac, max_fraction)
        final_bankrolls[s] = bankroll
        returns[s] = (bankroll - START_BANKROLL) / START_BANKROLL
        drawdowns[s] = max_dd

        # optional sampling export
        if export_samples and s < export_samples:
            sample_records.append({
                "sim": s, "final_bankroll": bankroll, "return": returns[s], "max_dd": max_dd
            })

    # summary stats
    mean_return = returns.mean()
    std_return = returns.std(ddof=0)
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0) if len(downside) else 0.0

    # geometric mean of returns (compounding-consistent)
    safe_final = np.maximum(final_bankrolls, 1e-12)
    geom_return = np.exp(np.mean(np.log(safe_final / START_BANKROLL))) - 1.0

    result = {
        "final_bankrolls": final_bankrolls,
        "returns": returns,
        "drawdowns": drawdowns,
        "mean_bankroll": final_bankrolls.mean(),
        "median_bankroll": float(np.median(final_bankrolls)),
        "p5_bankroll": float(np.percentile(final_bankrolls, 5)),
        "p95_bankroll": float(np.percentile(final_bankrolls, 95)),
        "mean_return_pct": mean_return * 100.0,
        "median_return_pct": float(np.median(returns)) * 100.0,
        "geometric_return_pct": geom_return * 100.0,
        "p5_return_pct": float(np.percentile(returns, 5)) * 100.0,
        "p95_return_pct": float(np.percentile(returns, 95)) * 100.0,
        "std_return_pct": std_return * 100.0,
        "sharpe": (mean_return / std_return) if std_return > 0 else 0.0,
        "sortino": (mean_return / downside_std) if downside_std > 0 else 0.0,
        "avg_max_drawdown_pct": drawdowns.mean() * 100.0,
        "worst_drawdown_pct": drawdowns.max() * 100.0,
        "bust_rate_pct": (np.sum(final_bankrolls <= 0) / sims) * 100.0,
        "profit_rate_pct": (np.sum(final_bankrolls > START_BANKROLL) / sims) * 100.0,
        "sims": sims,
        "weeks": len(weeks),
    }

    # Exports for auditability
    if export_samples and sample_records:
        fn = SAMPLES_DIR / f"samples_{model_name}_k{kelly_fraction:.2f}_{mode}.csv"
        pd.DataFrame(sample_records).to_csv(fn, index=False)

    if export_topk:
        top_idx = np.argsort(-final_bankrolls)[:export_topk]
        top_df = pd.DataFrame({
            "sim": top_idx,
            "final_bankroll": final_bankrolls[top_idx],
            "return": returns[top_idx],
            "max_dd": drawdowns[top_idx],
        }).reset_index(drop=True)
        fn = SAMPLES_DIR / f"topk_{model_name}_k{kelly_fraction:.2f}_{mode}.csv"
        top_df.to_csv(fn, index=False)

    return result


# -------------------
# Plotting & printing
# -------------------
def plot_curves(model: str, grid_df: pd.DataFrame, symlog: bool = False):
    mpretty = model.replace("_"," ").title()
    x = grid_df["kelly_fraction"]

    # Kelly vs Return (mean, median, band)
    fig = plt.figure(figsize=(10,6))
    plt.plot(x, grid_df["mean_return_pct"], linewidth=1.5, label="Mean")
    plt.plot(x, grid_df["median_return_pct"], linewidth=2, linestyle="--", label="Median")
    plt.fill_between(x, grid_df["p5_return_pct"], grid_df["p95_return_pct"], alpha=0.15, label="P5–P95")
    plt.axhline(0, color="black", alpha=0.3)
    plt.xlabel("Kelly Fraction"); plt.ylabel("Return (%)")
    plt.title(f"{mpretty} — Kelly vs Return")
    plt.xlim(0,1.0); plt.grid(True, alpha=0.3); plt.legend()
    if symlog: plt.yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"{model}_kelly_vs_return.png", dpi=300)
    plt.close(fig)

    # Kelly vs Bust
    fig = plt.figure(figsize=(10,6))
    plt.plot(x, grid_df["bust_rate_pct"], linewidth=2)
    plt.xlabel("Kelly Fraction"); plt.ylabel("Bust Rate (%)")
    plt.title(f"{mpretty} — Kelly vs Bust Rate")
    plt.xlim(0,1.0); plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"{model}_kelly_vs_bust.png", dpi=300)
    plt.close(fig)

    # Efficiency frontier
    fig = plt.figure(figsize=(10,6))
    plt.plot(grid_df["bust_rate_pct"], grid_df["median_return_pct"], linewidth=2, label="Median")
    plt.scatter(grid_df["bust_rate_pct"], grid_df["mean_return_pct"], s=10, alpha=0.6, label="Mean (pts)")
    plt.xlabel("Bust Rate (%)"); plt.ylabel("Return (%)")
    plt.title(f"{mpretty} — Efficiency Frontier")
    plt.grid(True, alpha=0.3); plt.legend()
    if symlog: plt.yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"{model}_efficiency_frontier.png", dpi=300)
    plt.close(fig)


def plot_overview(all_df: pd.DataFrame, symlog: bool = False):
    if all_df.empty: return

    # Kelly vs Median Return
    fig = plt.figure(figsize=(12,7))
    for m, g in all_df.groupby("model"):
        g = g.sort_values("kelly_fraction")
        plt.plot(g["kelly_fraction"], g["median_return_pct"], linewidth=2, linestyle="--",
                 label=m.replace("_"," ").title())
    plt.axhline(0, color="black", alpha=0.3)
    plt.xlabel("Kelly Fraction"); plt.ylabel("Median Return (%)")
    plt.title("Kelly vs Median Return — All Models")
    plt.xlim(0,1.0); plt.grid(True, alpha=0.3); plt.legend()
    if symlog: plt.yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ALL_kelly_vs_return.png", dpi=300)
    plt.close(fig)

    # Kelly vs Bust
    fig = plt.figure(figsize=(12,7))
    for m, g in all_df.groupby("model"):
        g = g.sort_values("kelly_fraction")
        plt.plot(g["kelly_fraction"], g["bust_rate_pct"], linewidth=2, label=m.replace("_"," ").title())
    plt.xlabel("Kelly Fraction"); plt.ylabel("Bust Rate (%)")
    plt.title("Kelly vs Bust Rate — All Models")
    plt.xlim(0,1.0); plt.grid(True, alpha=0.3); plt.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ALL_kelly_vs_bust.png", dpi=300)
    plt.close(fig)

    # Efficiency frontier (median)
    fig = plt.figure(figsize=(12,7))
    for m, g in all_df.groupby("model"):
        g = g.sort_values("kelly_fraction")
        plt.plot(g["bust_rate_pct"], g["median_return_pct"], linewidth=2, label=m.replace("_"," ").title())
    plt.xlabel("Bust Rate (%)"); plt.ylabel("Median Return (%)")
    plt.title("Efficiency Frontier — All Models (Median)")
    plt.grid(True, alpha=0.3); plt.legend()
    if symlog: plt.yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ALL_efficiency_frontier.png", dpi=300)
    plt.close(fig)


def print_model_table(model_label: str, grid_df: pd.DataFrame, print_all: bool):
    band = grid_df if print_all else grid_df[(grid_df["kelly_fraction"]>=0.01) & (grid_df["kelly_fraction"]<=0.10)]
    
    # Print header like deterministic script
    print(f"\n{model_label} — {'All Kelly (1–100%)' if print_all else '1–10% Kelly'}")
    print("  k     ret%    geo%    bust%   DD%     sharpe  sortino  P5%     P95%")
    if band.empty:
        print("  (no rows)"); return
    
    for _, r in band.iterrows():
        print(f"  {r['kelly_fraction']:.2f}  "
              f"{r['mean_return_pct']:+6.1f}%  "
              f"{r['geometric_return_pct']:+6.1f}%  "
              f"{r['bust_rate_pct']:5.1f}%  "
              f"{r['avg_max_drawdown_pct']:5.1f}%  "
              f"{r['sharpe']:6.2f}  "
              f"{r['sortino']:6.2f}  "
              f"{r['p5_return_pct']:+6.1f}%  "
              f"{r['p95_return_pct']:+6.1f}%")


# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bootstrap","permute_weeks","permute_matches"], default="bootstrap",
                    help="Resampling mode")
    ap.add_argument("--shuffle_within_week", action="store_true",
                    help="Shuffle match order inside each week block")
    ap.add_argument("--sims", type=int, default=SIMS_DEFAULT, help="Simulations per Kelly fraction")
    ap.add_argument("--edge", type=float, default=EDGE_MIN_DEFAULT, help="Minimum edge filter")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--common_paths", action="store_true",
                    help="Use identical sampled paths for all Kelly fractions (bootstrap/permute_weeks only)")
    ap.add_argument("--symlog", action="store_true", help="Symlog Y axis for return plots")
    ap.add_argument("--print-all", action="store_true", help="Print the table for ALL 100 Kelly points")
    ap.add_argument("--max_fraction", type=float, default=None,
                    help="Hard cap on fraction of bankroll staked on any bet (e.g. 0.25)")
    ap.add_argument("--export-samples", type=int, default=0,
                    help="Export the first N simulation summaries per (model, k) for auditing")
    ap.add_argument("--export-topk", type=int, default=0,
                    help="Export top-K final-bankroll sims per (model, k) for tail analysis")
    args = ap.parse_args()

    print("="*88)
    print("MONTE CARLO VARIANCE (from PURE bet logs)")
    print("="*88)
    print(f"Mode: {args.mode} | Intra-week shuffle: {args.shuffle_within_week}")
    print(f"Filters: edge ≥ {args.edge:.2%} | Sims per k: {args.sims:,} | Start bankroll: ${START_BANKROLL:.0f}")
    if args.common_paths and args.mode in ("bootstrap","permute_weeks"):
        print("Path control: SAME paths reused across Kelly fractions")
    elif args.mode == "permute_matches":
        print("Path control: per-sim global match permutations (not stored, to save memory)")
    else:
        print("Path control: independent paths per Kelly fraction")
    if args.max_fraction is not None:
        print(f"Bet cap: max {args.max_fraction:.2%} of bankroll per wager")
    print()

    rng = np.random.default_rng(args.seed)
    combined_rows = []

    for model in MODELS:
        label = model.replace('_',' ').title()
        print(f"\nMODEL: {label}")
        df = load_pure(model)
        if df is None or df.empty:
            print("  ❌ No data — skipping"); continue

        weeks = prepare_weeks(df, args.edge)
        n_bets = sum(len(w) for w in weeks)
        print(f"  ✅ Bets after edge filter: {n_bets:,} across {len(weeks)} weeks")
        if not weeks:
            print("  ❌ No usable weeks — skipping"); continue

        # Pre-sample common paths if requested & supported
        common_week_idx, common_perms = (None, None)
        if args.common_paths and args.mode in ("bootstrap","permute_weeks"):
            if args.mode == "bootstrap":
                common_week_idx, common_perms = sample_paths_bootstrap(
                    weeks, args.sims, rng, args.shuffle_within_week)
            else:
                common_week_idx, common_perms = sample_paths_permute_weeks(
                    weeks, args.sims, rng, args.shuffle_within_week)

        grid_out = []
        for frac in tqdm(KELLY_GRID, desc="  Kelly sweep", leave=False):
            res = simulate(
                mode=args.mode,
                weeks=weeks,
                frac=frac,
                sims=args.sims,
                rng=rng,
                shuffle_within_week=args.shuffle_within_week,
                common_week_idx=common_week_idx,
                common_perms=common_perms,
                max_fraction=args.max_fraction,
                export_samples=args.export_samples,
                export_topk=args.export_topk,
                model_name=model,
                kelly_fraction=frac,
            )
            if res is None: 
                continue
            row = {
                "model": model,
                "kelly_fraction": frac,
                "mean_bankroll": res["mean_bankroll"],
                "median_bankroll": res["median_bankroll"],
                "p5_bankroll": res["p5_bankroll"],
                "p95_bankroll": res["p95_bankroll"],
                "mean_return_pct": res["mean_return_pct"],
                "median_return_pct": res["median_return_pct"],
                "geometric_return_pct": res["geometric_return_pct"],
                "p5_return_pct": res["p5_return_pct"],
                "p95_return_pct": res["p95_return_pct"],
                "std_return_pct": res["std_return_pct"],
                "sharpe": res["sharpe"],
                "sortino": res["sortino"],
                "avg_max_drawdown_pct": res["avg_max_drawdown_pct"],
                "worst_drawdown_pct": res["worst_drawdown_pct"],
                "bust_rate_pct": res["bust_rate_pct"],
                "profit_rate_pct": res["profit_rate_pct"],
                "weeks": res["weeks"],
                "sims": res["sims"],
                "mode": args.mode,
                "shuffle_within_week": args.shuffle_within_week,
                "max_fraction": args.max_fraction if args.max_fraction is not None else "",
            }
            grid_out.append(row)
            combined_rows.append(row)

        grid_df = pd.DataFrame(grid_out).sort_values("kelly_fraction").reset_index(drop=True)
        # save per-model
        model_csv = RESULTS_DIR / f"{model}_kelly_grid.csv"
        grid_df.to_csv(model_csv, index=False)
        print(f"  💾 Saved per-model grid: {model_csv}")

        # console table
        print_model_table(label, grid_df, print_all=args.print_all)

        # charts per model
        plot_curves(model, grid_df, symlog=args.symlog)
        print(f"  📈 Charts written to {CHARTS_DIR}")

    # combined outputs
    all_df = pd.DataFrame(combined_rows)
    if not all_df.empty:
        all_csv = RESULTS_DIR / "all_models_kelly_grid.csv"
        all_df.to_csv(all_csv, index=False)
        print(f"\n💾 Combined grid: {all_csv}")

        plot_overview(all_df, symlog=args.symlog)
        print(f"📊 Overview charts written to {CHARTS_DIR}")

        # toplines (conservative band, bust ≤ 5%)
        print("\nTOPLINES (1%–10% Kelly, bust ≤ 5%):")
        view = all_df[(all_df["kelly_fraction"]>=0.01) & (all_df["kelly_fraction"]<=0.10)].copy()
        safe = view[view["bust_rate_pct"]<=5.0]
        for model, g in safe.groupby("model"):
            g = g.sort_values("median_return_pct", ascending=False)
            row = g.iloc[0] if not g.empty else None
            label = model.replace("_"," ").title()
            if row is not None:
                print(f"  • {label:<20} best k={row['kelly_fraction']:.2f}  "
                      f"median {row['median_return_pct']:+.1f}%  geo {row['geometric_return_pct']:+.1f}%  "
                      f"bust {row['bust_rate_pct']:.1f}%  sharpe {row['sharpe']:.2f}")
            else:
                print(f"  • {label:<20} no Kelly ≤10% with bust ≤5%")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
