# Sprint 1 — Model Evaluation Ledger + Pipeline Audit

- **Date:** 2026-06-21
- **Status:** Approved (design)
- **Scope:** Measurement + guardrails only. **No model changes, no promotions, no SQLite migration, no cloud.**

## 1. Problem / Motivation

- The live pipeline stakes **100% off the Neural Network** (v1.2.1). `calculate_betting_edges` builds edges from `player1_win_prob`, which is set only by the NN (`production/models/inference.py:176,356`); XGB/RF probabilities are logged but never influence a stake (`production/main.py:753`). Prior live observation suggests the NN trails XGB/RF and the market — so we may be sizing real money off the weakest model. We need a rigorous, repeatable scoreboard to confirm and decide which model should drive bets.
- Model knowledge is **scattered**: live correctness columns in `prediction_log.csv`, shadow predictions in a separate CSV, offline experiment metrics in `results/professional_tennis/experiments/` and docs. There is no single source of truth for "how every model actually performs."
- **Most settled rows are not trustworthy.** Of 1,256 settled rows, **820 are `legacy_backfilled`** and only **358 are decision-grade**. Headline numbers must not be computed on the contaminated cohort (the previously-cited "896 rows, 60.2% vs 66.4%" was ~65% legacy).
- Pipeline reliability is uncertain (Tennis Abstract flakiness reported, worked fine before); there is **no end-to-end smoke test**.

## 2. Goals (success criteria)

1. **One command** produces a **versioned ledger** (CSV + markdown) scoring every available model on clearly-labeled cohorts with: accuracy, log loss, Brier, ECE, AUC, calibration slope/intercept, reliability bins, and counterfactual ROI (flat + Kelly).
2. The ledger cleanly separates **(a) LIVE/settled** performance from **(b) OFFLINE backtest** metrics from experiments.
3. A clear **verdict**: which model is best by calibration / log-loss / Brier / ROI on the intersection cohort, and whether NN-primary staking is leaving money on the table.
4. An **offline end-to-end smoke test** guards the pipeline; **one live run** validates scrapers + settlement and unlocks the 3 currently-unsettled shadow models.
5. Repo stays clean, organized, documented; `AGENTS.md` + relevant docs updated.

## 3. Cohorts (the honesty core)

- `match_uid → actual_winner` is the **authoritative ground truth** (verified: 1,254 unique settled match_uids, **0 conflicting winners**, 2 benign dups). Correctness is computed by joining each model's probability to this map — never by trusting pre-computed per-log `*_correct` flags (which are stale/missing for several models).
- **Tier GOLD (358):** `settled & logging_quality=snapshot_v2 & rescore_quality=exact_feature_snapshot & features_complete`. **Headline.**
- **Tier COMPLETE (1,162):** `settled & features_complete`. Secondary; flagged ~65% legacy.
- **Tier ALL-SETTLED (1,256):** context only.
- **Per-model** metrics on each model's own settled coverage, but the **cross-model winner verdict is computed only on the INTERSECTION** (rows where all compared models have a prediction) to remove coverage bias (NN 1,003 settled vs XGB/RF 845).
- Every reported number carries its **cohort label and n**. No silent mixing.

## 4. Models in scope

**LIVE/settled — scoreable now:**
- NN v1.2.1 (primary), XGB v1.2.0, RF v1.2.0, shadow-XGB (recency 12y), **Market** baseline.

**LIVE/settled — after one settlement catch-up:**
- shadow CatBoost, shadow LightGBM, shadow NN (logits). Currently 115 *unsettled* rows each (only ever scored the 2026-05-26/27 slate; no settlement run since). One settlement pass unlocks them.

**OFFLINE — from experiments (ingested into the ledger's offline section):**
- All variants under `results/professional_tennis/experiments/` (fixed-split + blocked walk-forward): CatBoost/LightGBM (one-hot & `native_cat`), recency-weighted XGB (half-lives 3/5/8/12y), NN tweaks, base_141 baselines. Tabulated from their saved metrics; where formats are heterogeneous, parse what exists and **flag gaps explicitly** (no silent omission).

## 5. Metrics

- **Discrimination:** AUC, accuracy.
- **Calibration / probabilistic:** log loss, Brier, ECE, calibration slope & intercept, reliability bins.
- **Ranking priority** (per project strategy: calibration + Kelly over raw accuracy): log loss / Brier / ECE / calibration **first**, accuracy last.
- Single shared implementation in `production/evaluation/metrics.py`, **reused by the dashboard** (refactor `dashboard/data.py` to import it — eliminate duplicate math, prevent drift).

## 6. ROI / profitability

Counterfactual backtest on the settled cohort using **odds logged at prediction time**; de-vig **recomputed from logged decimal odds** (not trusting stored implied-prob columns).

- Per model: pick = side with max edge vs de-vigged market; qualify if edge ≥ 0.02.
- **Two staking modes:** (a) **flat** unit stake per qualifying bet (pure predictive edge); (b) **Kelly** with live params (multiplier 0.18, threshold 0.02, cap 0.05).
- Report: ROI, total P&L (units and fixed-notional bankroll), n_bets, win-rate-on-bets, and max drawdown (Kelly). Fixed-notional bankroll (not compounding) for clean cross-model comparison; compounding noted as a variant.
- **Benchmarks:** ROI break-even bar = beating the vig (0%); market favorite hit-rate = predictive bar.
- Mirror the **live staking logic exactly** for an "as-run" number; also report the clean flat-stake number.
- **Realized P&L** of actually-placed bets (`all_bets.csv`) reconstructed by joining to settled outcomes — secondary (currently 0 settled bets), reported when available.

## 7. Architecture

New package **`production/evaluation/`**, one responsibility per file (~≤200 lines where feasible):

- `metrics.py` — pure metric functions (log loss, Brier, ECE, AUC, calibration slope/intercept, reliability bins). No I/O.
- `cohorts.py` — load `prediction_log.csv` + shadow log; build the authoritative `match_uid → winner` map; define tiers; assemble per-model prediction frames; intersection logic. **The only file that touches storage format** → SQLite migration (Sprint 2) swaps its loader only.
- `roi.py` — counterfactual staking simulation (flat + Kelly), de-vig, settlement, ROI/drawdown.
- `offline.py` — ingest experiment metrics from `results/professional_tennis/experiments/` into ledger rows.
- `ledger.py` — orchestrate → write **versioned CSV** (`results/professional_tennis/ledger/<date>/model_ledger.csv`) + **markdown report**.
- CLI: `python -m production.evaluation.ledger` (+ thin wrapper if convenient).

`analyze_predictions.py` (console, 444 lines) is left intact this sprint; it may later delegate to the new util. Rationale for a new package vs. extending it: it is console-shaped and single-purpose; the ledger needs a different output shape and must be reusable + unit-testable.

## 8. Pipeline audit

- **Offline smoke test** (`production/tests/`): mock the Bovada fetch + TA scraper with small fixtures; drive the orchestrator through odds → features → inference → staking → logging → settlement; assert non-crash, schema contract, prediction rows written, settlement-enrich path. Seconds, no network. Permanent regression guard.
- **Live validation:** one `python production/main.py --dry-run` (fetches odds + features + predictions and runs settlement; **no betting session / no bet logging**). **Pre-flight:** a tiny TA connectivity probe before settlement issues many requests. Watch for 429 / availability; if broken, switch to **systematic-debugging** (it worked before — find what changed). The settlement catch-up clears the May/June backlog → **unlocks the 3 shadow models**; then re-run the ledger to include all 7 live models.

## 9. Out of scope (later sprints)

SQLite migration (Sprint 2), retrain / promote a champion (Sprint 3), cloud deployment (Sprint 4). Expanding **live shadow coverage** to more variants (so more variants accumulate live data) is a small config follow-on — noted, not required here.

## 10. Risks

- **TA down/blocked** → live unlock + fresh settlement delayed; the offline ledger still ships and is the primary deliverable.
- **Heterogeneous offline experiment formats** → tolerant parsing; fall back to documented summary numbers and flag gaps.
- **Small samples** (GOLD 358, shadow-XGB 266, shadow trio 115) → report sample sizes / simple confidence intervals; avoid over-claiming on thin cohorts.

## 11. Documentation deliverables

- `docs/modeling/MODEL_LEDGER.md` — how to run, how to read, cohort definitions, current verdict.
- Update `AGENTS.md` — the ledger is the **source of truth for model performance**; how to regenerate; cohort tiers; intersection rule.
- Update pointers in `docs/production/README.md` and `docs/modeling/EXPERIMENT_WORKFLOW.md`.
