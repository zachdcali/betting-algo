"""Assemble the model evaluation ledger and write CSV + markdown outputs.

Live models are scored on labeled cohort tiers (gold / complete), the core-model
intersection, the all-current-model intersection, and the first-observed versus
last-pre-start market cohort. Offline experiment metrics are ingested separately
and never mixed with live numbers.

CLI:
    cd production && ../tennis_env/bin/python -m evaluation.ledger \\
        --prod-dir . \\
        --experiments-root ../results/professional_tennis/experiments \\
        --out-dir ../results/professional_tennis/ledger/<date> \\
        --report ../docs/modeling/MODEL_LEDGER.md
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from evaluation import cohorts, metrics, offline, roi
from shadow.performance_v1_shadow import DEFAULT_SHADOW_MODEL_SPECS

CORE_MODELS = ["nn", "xgb", "rf", "market"]
ACTIVE_SHADOW_MODELS = [
    f"shadow_{spec.model_version}" for spec in DEFAULT_SHADOW_MODEL_SPECS
]

LIVE_COLUMNS = [
    "model", "tier", "n", "accuracy", "auc", "log_loss", "brier", "ece",
    "cal_slope", "cal_intercept",
    "roi_flat", "n_bets_flat", "pnl_flat", "win_rate_flat",
    "roi_kelly", "n_bets_kelly", "pnl_kelly", "max_drawdown_kelly",
]

CALIBRATION_COLUMNS = [
    "model", "tier", "bin_index", "bin_lo", "bin_hi", "mean_pred",
    "frac_pos", "count",
]


def _score_block(model: str, tier: str, g: pd.DataFrame) -> dict:
    m = metrics.compute_all(g["y1"].values, g["p1_prob"].values)
    flat = roi.simulate(g, mode="flat")
    kelly = roi.simulate(g, mode="kelly")
    return {
        "model": model, "tier": tier,
        "n": m["n"], "accuracy": m["accuracy"], "auc": m["auc"],
        "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
        "cal_slope": m["cal_slope"], "cal_intercept": m["cal_intercept"],
        "roi_flat": flat["roi"], "n_bets_flat": flat["n_bets"],
        "pnl_flat": flat["pnl"], "win_rate_flat": flat["win_rate"],
        "roi_kelly": kelly["roi"], "n_bets_kelly": kelly["n_bets"],
        "pnl_kelly": kelly["pnl"], "max_drawdown_kelly": kelly["max_drawdown"],
    }


def _unrestricted_intersection_uids(scored: pd.DataFrame, models: list[str]) -> set:
    """Return settled match IDs carrying every requested model observation."""
    sub = scored[scored["model"].isin(models)]
    counts = sub.groupby("match_uid")["model"].nunique()
    return set(counts[counts == len(set(models))].index)


def build_live_ledger(
    scored: pd.DataFrame,
    intersection_models: list[str] | None = None,
    active_shadow_models: list[str] | None = None,
) -> pd.DataFrame:
    intersection_models = intersection_models or CORE_MODELS
    active_shadow_models = (
        ACTIVE_SHADOW_MODELS
        if active_shadow_models is None
        else active_shadow_models
    )
    timing_models = ["market_open", "market_close"]
    rows = []
    for tier_name, tier_col in [("gold", "is_gold"), ("complete", "is_complete")]:
        tier_df = scored[scored[tier_col]]
        generic_df = tier_df[~tier_df["model"].isin(timing_models)]
        for model, g in generic_df.groupby("model"):
            rows.append(_score_block(model, tier_name, g))
        inter = cohorts.intersection_uids(scored, intersection_models, tier_col)
        inter_df = tier_df[tier_df["match_uid"].isin(inter) & tier_df["model"].isin(intersection_models)]
        for model, g in inter_df.groupby("model"):
            rows.append(_score_block(model, f"{tier_name}_intersection", g))
        observed_models = set(tier_df["model"].astype(str))
        shadow_models = [
            model for model in active_shadow_models if model in observed_models
        ]
        if shadow_models:
            all_models = [*intersection_models, *shadow_models]
            all_inter = cohorts.intersection_uids(scored, all_models, tier_col)
            all_inter_df = tier_df[
                tier_df["match_uid"].isin(all_inter)
                & tier_df["model"].isin(all_models)
            ]
            for model, g in all_inter_df.groupby("model"):
                rows.append(_score_block(
                    model, f"{tier_name}_all_model_intersection", g
                ))
        timing_inter = cohorts.intersection_uids(scored, timing_models, tier_col)
        timing_df = tier_df[
            tier_df["match_uid"].isin(timing_inter)
            & tier_df["model"].isin(timing_models)
        ]
        for model, g in timing_df.groupby("model"):
            rows.append(_score_block(
                model, f"{tier_name}_market_timing", g
            ))
    settled_timing_uids = _unrestricted_intersection_uids(scored, timing_models)
    settled_timing = scored[
        scored["match_uid"].isin(settled_timing_uids)
        & scored["model"].isin(timing_models)
    ]
    for model, g in settled_timing.groupby("model"):
        rows.append(_score_block(model, "settled_market_timing", g))
    return pd.DataFrame(rows, columns=LIVE_COLUMNS)


def build_calibration_ledger(scored: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Materialize reliability bins for every authoritative aggregate row."""
    rows: list[dict] = []
    for _, aggregate in live.iterrows():
        model = str(aggregate["model"])
        tier = str(aggregate["tier"])
        if tier == "gold":
            block = scored[scored["is_gold"] & scored["model"].eq(model)]
        elif tier == "complete":
            block = scored[scored["is_complete"] & scored["model"].eq(model)]
        elif tier == "settled_market_timing":
            common = _unrestricted_intersection_uids(
                scored, ["market_open", "market_close"]
            )
            block = scored[
                scored["model"].eq(model) & scored["match_uid"].isin(common)
            ]
        else:
            base_tier = "is_gold" if tier.startswith("gold_") else "is_complete"
            if tier.endswith("_all_model_intersection"):
                models = sorted(
                    value for value in live.loc[live["tier"].eq(tier), "model"]
                )
            elif tier.endswith("_market_timing"):
                models = ["market_open", "market_close"]
            else:
                models = CORE_MODELS
            common = cohorts.intersection_uids(scored, models, base_tier)
            block = scored[
                scored[base_tier]
                & scored["model"].eq(model)
                & scored["match_uid"].isin(common)
            ]
        if block.empty:
            continue
        reliability = metrics.reliability_table(
            block["y1"].values, block["p1_prob"].values
        )
        for bin_index, bin_row in reliability.iterrows():
            rows.append({
                "model": model,
                "tier": tier,
                "bin_index": int(bin_index),
                "bin_lo": bin_row["bin_lo"],
                "bin_hi": bin_row["bin_hi"],
                "mean_pred": bin_row["mean_pred"],
                "frac_pos": bin_row["frac_pos"],
                "count": int(bin_row["count"]),
            })
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _verdict(live: pd.DataFrame) -> str:
    block = live[live.tier == "gold_intersection"].copy()
    if block.empty:
        block = live[live.tier == "complete_intersection"].copy()
    if block.empty:
        return "_No intersection cohort available to render a verdict._"
    block = block.sort_values("log_loss")
    best = block.iloc[0]
    n = int(best["n"])
    lines = [
        f"On the **{best['tier']}** cohort (n={n}, all of nn/xgb/rf/market predicted), "
        f"ranked by log loss:",
        "",
    ]
    for _, r in block.iterrows():
        lines.append(
            f"- **{r['model']}** — log_loss {_fmt(r['log_loss'])}, brier {_fmt(r['brier'])}, "
            f"acc {_fmt(r['accuracy'])}, AUC {_fmt(r['auc'])}, "
            f"ROI(flat) {_fmt(100*r['roi_flat'],1)}% over {int(r['n_bets_flat'])} bets, "
            f"ROI(Kelly) {_fmt(100*r['roi_kelly'],1)}%"
        )
    nn = block[block.model == "nn"]
    mkt = block[block.model == "market"]
    lines.append("")
    if best["model"] != "nn" and not nn.empty:
        lines.append(
            f"**Best discriminator: `{best['model']}`** (log_loss {_fmt(best['log_loss'])}) vs "
            f"current betting model `nn` (log_loss {_fmt(nn.iloc[0]['log_loss'])}). "
            f"Bets are currently staked entirely off `nn`."
        )
    elif best["model"] == "nn":
        lines.append("**Best discriminator is `nn`**, the current betting model.")
    if not mkt.empty:
        beat = block[(block.model != "market") & (block["log_loss"] < float(mkt.iloc[0]["log_loss"]))]
        if not beat.empty:
            lines.append(
                f"Models beating the market on log loss: {', '.join('`'+m+'`' for m in beat['model'])}."
            )
        else:
            lines.append("No model beats the market on log loss on this cohort.")

    # Calibration flag: slope << 1 means over-confident, the usual cause of bad log loss.
    nonmkt = block[block.model != "market"].dropna(subset=["cal_slope"])
    if not nonmkt.empty:
        worst_cal = nonmkt.sort_values("cal_slope").iloc[0]
        if float(worst_cal["cal_slope"]) < 0.6:
            lines.append(
                f"**Calibration:** `{worst_cal['model']}` is severely over-confident "
                f"(slope {_fmt(worst_cal['cal_slope'],2)}; 1.0 = calibrated) — post-hoc "
                f"calibration is the obvious lever."
            )

    # Profitable models across the full gold tier (includes shadows on their own coverage).
    # market is excluded: its edge versus the de-vigged market is ~0 by construction.
    gold = live[(live.tier == "gold") & (live.model != "market")].dropna(subset=["roi_flat"])
    if not gold.empty:
        pos = gold[gold["roi_flat"] > 0].sort_values("roi_flat", ascending=False)
        if not pos.empty:
            items = "; ".join(
                f"`{r['model']}` {_fmt(100*r['roi_flat'],1)}% flat / {_fmt(100*r['roi_kelly'],1)}% Kelly "
                f"({int(r['n_bets_flat'])} bets, n={int(r['n'])})"
                for _, r in pos.iterrows()
            )
            lines.append(
                f"**ROI (GOLD, profitable models):** {items}. Beating the vig (>0%) is the bar; "
                f"treat n<250 as a lead, not proof."
            )
        else:
            lines.append("**ROI:** no model is profitable on GOLD — every flat ROI is negative.")
    return "\n".join(lines)


def _report_md(live: pd.DataFrame, offline_df: pd.DataFrame, run_date: str) -> str:
    show = ["model", "n", "accuracy", "auc", "log_loss", "brier", "ece",
            "cal_slope", "roi_flat", "roi_kelly", "n_bets_flat"]
    parts = [
        f"# Model Ledger — {run_date}",
        "",
        "_Generated by `python -m evaluation.ledger`. Source of truth for model "
        "performance. Probabilities are P(player1 wins); ground truth joined from "
        "`prediction_log.csv` by `match_uid`._",
        "",
        "## Cohort tiers",
        "",
        "- **GOLD** = settled & `snapshot_v2` & `exact_feature_snapshot` & features_complete & feature snapshot ID verified against persisted lineage (decision-grade; headline).",
        "- **COMPLETE** = settled & features_complete (~65% legacy_backfilled; context).",
        "- **\\*_intersection** = restricted to match_uids where all of nn/xgb/rf/market predicted (apples-to-apples).",
        "- **\\*_all_model_intersection** = restricted to match_uids shared by nn/xgb/rf/market and every scored variant in the current `DEFAULT_SHADOW_MODEL_SPECS`; retired historical variants do not shrink the active comparison.",
        "- **\\*_market_timing** = the same settled match_uids at first observed and last valid pre-start prices; requires at least two distinct pre-start captures and never treats first observed as the sportsbook's true opener.",
        "- **settled_market_timing** = every explicit valid winner with comparable market captures, independent of feature completeness; use this for the broad standalone market-open versus market-close diagnostic.",
        "- **Shadow variants** = one deterministic opening observation per `(match_uid, model_version)`, joined to the operational opening feature snapshot; hourly repeats do not increase n.",
        "",
        "## Verdict (live, settled)",
        "",
        _verdict(live),
        "",
        "## Live model table",
        "",
    ]
    for tier in [
        "gold_intersection", "gold_all_model_intersection", "gold_market_timing",
        "settled_market_timing", "gold",
        "complete_intersection", "complete_all_model_intersection",
        "complete_market_timing", "complete",
    ]:
        block = live[live.tier == tier].sort_values("log_loss")
        if block.empty:
            continue
        parts += [f"### {tier} (n={int(block['n'].max())})", "", _md_table(block, show), ""]

    parts += ["## Offline experiments (backtest, not live)", ""]
    if offline_df is None or offline_df.empty:
        parts.append("_No offline experiments ingested._")
    else:
        parts.append(
            "_Different `split` labels use different test eras and are NOT comparable "
            "across splits. Compare within a split only._\n"
        )
        ocols = ["family", "experiment", "split", "accuracy", "auc", "log_loss", "brier", "ece"]
        odf = offline_df.dropna(subset=["log_loss"]).sort_values(["split", "log_loss"])
        parts.append(_md_table(odf[ocols], ocols))
    parts.append("")
    return "\n".join(parts)


def write_outputs(live: pd.DataFrame, offline_df: pd.DataFrame,
                  out_dir: str, report_path: str, run_date: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    live.to_csv(os.path.join(out_dir, "model_ledger.csv"), index=False)
    if offline_df is not None:
        offline_df.to_csv(os.path.join(out_dir, "offline_experiments.csv"), index=False)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as fh:
        fh.write(_report_md(live, offline_df, run_date))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the model evaluation ledger.")
    ap.add_argument("--prod-dir", default=".")
    ap.add_argument("--experiments-root", default="../results/professional_tennis/experiments")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--run-date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--db", default=None,
                    help="Read logs from this SQLite DB (predictions/shadow_predictions tables) instead of CSVs")
    args = ap.parse_args(argv)

    if args.db:
        import db as _db
        pred_log = _db.read_table(args.db, "predictions")
        shadow_log = _db.read_table(args.db, "shadow_predictions")
    else:
        pred_log = cohorts.load_prediction_log(args.prod_dir)
        shadow_log = cohorts.load_shadow_log(args.prod_dir)
    odds_history = None if args.db else cohorts.load_odds_history(args.prod_dir)
    scored = cohorts.build_scored_frame(pred_log, shadow_log, odds_history)
    live = build_live_ledger(scored)
    offline_df = offline.discover_experiment_metrics(args.experiments_root)
    write_outputs(live, offline_df, args.out_dir, args.report, args.run_date)

    print(f"Live ledger rows: {len(live)} | offline experiments: {len(offline_df)}")
    print(f"Wrote: {os.path.join(args.out_dir, 'model_ledger.csv')}")
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
