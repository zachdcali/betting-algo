# Model Evaluation Ledger + Pipeline Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned, repeatable model-evaluation ledger that scores every model (live, shadow, and offline-experiment) on clearly-labeled cohorts with calibration + ROI, and add a permanent offline smoke test plus one validated live run.

**Architecture:** New `production/evaluation/` package with single-responsibility modules (`metrics`, `cohorts`, `roi`, `offline`, `ledger`). Correctness is derived by joining each model's probability to an authoritative `match_uid → winner` map, scored on labeled cohort tiers (GOLD / COMPLETE), with cross-model verdicts on the intersection cohort. The dashboard is refactored to reuse the shared metrics. A mock-based smoke test guards the data-generating pipeline; one live `--dry-run` validates scrapers/settlement and unlocks the unsettled shadow models.

**Tech Stack:** Python 3.9, pandas, numpy, scikit-learn (already in `tennis_env`), pytest.

## Global Constraints

- **Interpreter:** `tennis_env/bin/python` (fallback `python3`). Run pytest from repo root: `tennis_env/bin/python -m pytest tests -q`.
- **Import root:** `production/` is on `sys.path`. The new package imports as `from evaluation.metrics import ...`. New tests live in root `tests/` and start with the standard preamble:
  ```python
  import sys
  from pathlib import Path
  REPO_ROOT = Path(__file__).resolve().parents[1]
  PRODUCTION_DIR = REPO_ROOT / "production"
  if str(PRODUCTION_DIR) not in sys.path:
      sys.path.insert(0, str(PRODUCTION_DIR))
  ```
- **Orientation convention:** all probabilities are `P(player1 wins)`; ground truth `y1 = 1 if actual_winner == 1 else 0`. Every model and the market are scored from player1's perspective.
- **No promotions, no model retraining, no overwriting promoted artifacts, no SQLite.** Measurement + tests + docs only.
- **No co-author line in commits.** Commit per task. Do not commit mutable churn (`prediction_log.csv`, `data/atp_rankings.csv`, `data/atp_heights.json`, `.DS_Store`, screenshots).
- **Cohort honesty:** never report a number without its cohort label + n. GOLD = `snapshot_v2 & exact_feature_snapshot & features_complete & settled`. COMPLETE = `features_complete & settled`.
- **Live params (for ROI "as-run" mode):** kelly_multiplier 0.18, edge_threshold 0.02, max_stake_fraction 0.05.

---

### Task 1: `metrics.py` — pure scoring functions

**Files:**
- Create: `production/evaluation/__init__.py` (empty)
- Create: `production/evaluation/metrics.py`
- Test: `tests/test_evaluation_metrics.py`

**Interfaces:**
- Produces (all take `y_true: np.ndarray` in {0,1} and `p: np.ndarray` = P(player1 wins)):
  - `accuracy(y_true, p) -> float`
  - `log_loss_score(y_true, p) -> float`
  - `brier_score(y_true, p) -> float`
  - `auc_score(y_true, p) -> float` (returns `nan` if one class only)
  - `ece(y_true, p, n_bins=10) -> float`
  - `calibration_slope_intercept(y_true, p) -> tuple[float, float]`
  - `reliability_table(y_true, p, n_bins=10) -> pd.DataFrame` (cols: `bin_lo, bin_hi, mean_pred, frac_pos, count`)
  - `compute_all(y_true, p, n_bins=10) -> dict` (keys: `n, accuracy, log_loss, brier, auc, ece, cal_slope, cal_intercept`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation_metrics.py
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import metrics


def test_perfect_predictions():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.99, 0.01, 0.99, 0.01])
    assert metrics.accuracy(y, p) == 1.0
    assert metrics.brier_score(y, p) < 0.001
    assert metrics.auc_score(y, p) == 1.0
    assert metrics.log_loss_score(y, p) < 0.02


def test_ece_zero_for_calibrated():
    # 100 preds at p=0.5 with exactly half positive -> perfectly calibrated bin
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.5)
    assert metrics.ece(y, p, n_bins=10) < 1e-9


def test_calibration_slope_near_one_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    slope, intercept = metrics.calibration_slope_intercept(y, p)
    assert 0.8 < slope < 1.2
    assert abs(intercept) < 0.2


def test_compute_all_keys_and_n():
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.6, 0.4, 0.7, 0.55, 0.3])
    out = metrics.compute_all(y, p)
    assert out["n"] == 5
    for k in ["accuracy", "log_loss", "brier", "auc", "ece", "cal_slope", "cal_intercept"]:
        assert k in out


def test_auc_nan_single_class():
    y = np.array([1, 1, 1])
    p = np.array([0.6, 0.7, 0.8])
    assert np.isnan(metrics.auc_score(y, p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# production/evaluation/metrics.py
"""Pure model-scoring functions. No I/O. All probs are P(player1 wins)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_EPS = 1e-15


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def accuracy(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p) >= 0.5).astype(float) == y_true))


def log_loss_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = _clip(p)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y_true) ** 2))


def auc_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(p, dtype=float)))


def ece(y_true, p, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = len(p)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)


def calibration_slope_intercept(y_true, p) -> tuple[float, float]:
    """Logistic regression of y on logit(p). slope≈1, intercept≈0 => calibrated."""
    y_true = np.asarray(y_true, dtype=int)
    p = _clip(p)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def reliability_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({
            "bin_lo": edges[b], "bin_hi": edges[b + 1],
            "mean_pred": float(p[mask].mean()) if mask.any() else float("nan"),
            "frac_pos": float(y_true[mask].mean()) if mask.any() else float("nan"),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def compute_all(y_true, p, n_bins: int = 10) -> dict:
    slope, intercept = calibration_slope_intercept(y_true, p)
    return {
        "n": int(len(np.asarray(p))),
        "accuracy": accuracy(y_true, p),
        "log_loss": log_loss_score(y_true, p),
        "brier": brier_score(y_true, p),
        "auc": auc_score(y_true, p),
        "ece": ece(y_true, p, n_bins),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_metrics.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add production/evaluation/__init__.py production/evaluation/metrics.py tests/test_evaluation_metrics.py
git commit -m "Add evaluation.metrics: pure model-scoring functions"
```

---

### Task 2: `cohorts.py` — load logs, ground truth, scored long-frame, tiers, intersection

**Files:**
- Create: `production/evaluation/cohorts.py`
- Test: `tests/test_evaluation_cohorts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (reads CSVs).
- Produces:
  - `build_ground_truth(pred_log: pd.DataFrame) -> pd.Series` — index `match_uid`, value `y1∈{0,1}`; dedup on `match_uid`; drop conflicting winners.
  - `build_scored_frame(pred_log: pd.DataFrame, shadow_log: pd.DataFrame | None) -> pd.DataFrame` — long format, one row per `(match_uid, model)`. Columns: `match_uid, model, family, p1_prob, p1_odds_decimal, p2_odds_decimal, y1, is_gold, is_complete`. Models: `nn, xgb, rf, market` (from `prediction_log`), `shadow_xgboost, shadow_catboost, shadow_lightgbm, shadow_nn` (from shadow log, joined to ground truth). Only settled rows with non-null prob are included.
  - `MODEL_PROB_COLS: dict[str,str]` — `{"nn":"model_p1_prob","xgb":"xgb_p1_prob","rf":"rf_p1_prob","market":"market_p1_prob"}`.
  - `intersection_uids(scored: pd.DataFrame, models: list[str], tier_col: str) -> set` — match_uids present for *all* listed models within the tier.

- [ ] **Step 1: Write the failing test** (synthetic fixtures; no real files)

```python
# tests/test_evaluation_cohorts.py
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import cohorts


def _pred_log():
    return pd.DataFrame([
        # settled gold row, all models present
        dict(match_uid="m1", actual_winner=1, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.7, xgb_p1_prob=0.65, rf_p1_prob=0.6, market_p1_prob=0.62,
             p1_odds_decimal=1.7, p2_odds_decimal=2.2),
        # settled complete-but-legacy row, NN+market only (xgb/rf null)
        dict(match_uid="m2", actual_winner=2, features_complete=True,
             logging_quality="legacy_backfilled", rescore_quality="legacy_fallback_match",
             model_p1_prob=0.4, xgb_p1_prob=np.nan, rf_p1_prob=np.nan, market_p1_prob=0.45,
             p1_odds_decimal=2.3, p2_odds_decimal=1.6),
        # unsettled row -> excluded
        dict(match_uid="m3", actual_winner=np.nan, features_complete=True,
             logging_quality="snapshot_v2", rescore_quality="exact_feature_snapshot",
             model_p1_prob=0.5, xgb_p1_prob=0.5, rf_p1_prob=0.5, market_p1_prob=0.5,
             p1_odds_decimal=2.0, p2_odds_decimal=1.8),
    ])


def test_ground_truth_orientation_and_dedup():
    gt = cohorts.build_ground_truth(_pred_log())
    assert gt.loc["m1"] == 1   # player1 won
    assert gt.loc["m2"] == 0   # player2 won
    assert "m3" not in gt.index  # unsettled excluded


def test_scored_frame_models_and_tiers():
    scored = cohorts.build_scored_frame(_pred_log(), None)
    # m1 has nn,xgb,rf,market ; m2 has nn,market only
    m1 = scored[scored.match_uid == "m1"]
    assert set(m1.model) == {"nn", "xgb", "rf", "market"}
    assert bool(m1.is_gold.iloc[0]) is True
    m2 = scored[scored.match_uid == "m2"]
    assert set(m2.model) == {"nn", "market"}
    assert bool(m2.is_gold.iloc[0]) is False
    assert bool(m2.is_complete.iloc[0]) is True
    # y1 propagated
    assert scored[(scored.match_uid == "m1") & (scored.model == "nn")].y1.iloc[0] == 1


def test_intersection_excludes_partial_coverage():
    scored = cohorts.build_scored_frame(_pred_log(), None)
    inter = cohorts.intersection_uids(scored, ["nn", "xgb", "rf", "market"], "is_complete")
    assert inter == {"m1"}   # m2 lacks xgb/rf
```

- [ ] **Step 2: Run test to verify it fails**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_cohorts.py -q`
Expected: FAIL — `No module named 'evaluation.cohorts'` (or attribute errors).

- [ ] **Step 3: Write minimal implementation**

```python
# production/evaluation/cohorts.py
"""Load logs, derive authoritative ground truth, assemble a long scored frame."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

MODEL_PROB_COLS = {
    "nn": "model_p1_prob", "xgb": "xgb_p1_prob",
    "rf": "rf_p1_prob", "market": "market_p1_prob",
}
SHADOW_FAMILIES = ["xgboost", "catboost", "lightgbm", "nn"]


def load_prediction_log(prod_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(prod_dir, "prediction_log.csv"), low_memory=False)


def load_shadow_log(prod_dir: str) -> pd.DataFrame | None:
    p = os.path.join(prod_dir, "logs", "performance_v1_shadow_predictions.csv")
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else None


def build_ground_truth(pred_log: pd.DataFrame) -> pd.Series:
    s = pred_log[pred_log["actual_winner"].notna()].copy()
    s["y1"] = (s["actual_winner"].astype(float) == 1).astype(int)
    # drop match_uids whose winner conflicts across rows, then dedup
    nun = s.groupby("match_uid")["y1"].nunique()
    good = nun[nun == 1].index
    s = s[s["match_uid"].isin(good)].drop_duplicates("match_uid")
    return s.set_index("match_uid")["y1"]


def _tier_flags(pred_log: pd.DataFrame) -> pd.DataFrame:
    f = pred_log.copy()
    f["is_complete"] = f.get("features_complete").astype("boolean").fillna(False)
    f["is_gold"] = (
        f["is_complete"]
        & (f.get("logging_quality") == "snapshot_v2")
        & (f.get("rescore_quality") == "exact_feature_snapshot")
    )
    return f[["match_uid", "is_complete", "is_gold"]].drop_duplicates("match_uid")


def build_scored_frame(pred_log: pd.DataFrame, shadow_log: pd.DataFrame | None) -> pd.DataFrame:
    gt = build_ground_truth(pred_log)
    tiers = _tier_flags(pred_log).set_index("match_uid")
    settled = pred_log[pred_log["match_uid"].isin(gt.index)].drop_duplicates("match_uid").set_index("match_uid")
    rows = []
    for model, col in MODEL_PROB_COLS.items():
        sub = settled[settled[col].notna()]
        for uid, r in sub.iterrows():
            rows.append(dict(
                match_uid=uid, model=model, family=model,
                p1_prob=float(r[col]),
                p1_odds_decimal=r.get("p1_odds_decimal"),
                p2_odds_decimal=r.get("p2_odds_decimal"),
                y1=int(gt.loc[uid]),
                is_gold=bool(tiers.loc[uid, "is_gold"]),
                is_complete=bool(tiers.loc[uid, "is_complete"]),
            ))
    if shadow_log is not None and "model_family" in shadow_log.columns:
        sh = shadow_log[shadow_log["match_uid"].isin(gt.index)]
        for _, r in sh.iterrows():
            uid = r["match_uid"]
            if pd.isna(r.get("shadow_p1_prob")):
                continue
            rows.append(dict(
                match_uid=uid, model=f"shadow_{r['model_family']}", family=r["model_family"],
                p1_prob=float(r["shadow_p1_prob"]),
                p1_odds_decimal=r.get("p1_odds_decimal"),
                p2_odds_decimal=r.get("p2_odds_decimal"),
                y1=int(gt.loc[uid]),
                is_gold=bool(tiers.loc[uid, "is_gold"]) if uid in tiers.index else False,
                is_complete=bool(tiers.loc[uid, "is_complete"]) if uid in tiers.index else False,
            ))
    return pd.DataFrame(rows)


def intersection_uids(scored: pd.DataFrame, models: list[str], tier_col: str) -> set:
    sub = scored[scored[tier_col] & scored["model"].isin(models)]
    counts = sub.groupby("match_uid")["model"].nunique()
    need = len(set(models))
    return set(counts[counts == need].index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_cohorts.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add production/evaluation/cohorts.py tests/test_evaluation_cohorts.py
git commit -m "Add evaluation.cohorts: ground truth, scored frame, tiers, intersection"
```

---

### Task 3: `roi.py` — de-vig + counterfactual staking (flat + Kelly)

**Files:**
- Create: `production/evaluation/roi.py`
- Test: `tests/test_evaluation_roi.py`

**Interfaces:**
- Consumes: a scored sub-frame (rows for one model) with `p1_prob, p1_odds_decimal, p2_odds_decimal, y1`.
- Produces:
  - `devig_two_way(o1: float, o2: float) -> tuple[float, float]` — fair (p1, p2) from decimal odds via implied-prob normalization.
  - `simulate(df: pd.DataFrame, mode: str = "flat", edge_threshold: float = 0.02, kelly_mult: float = 0.18, cap: float = 0.05, bankroll: float = 1000.0) -> dict` — keys: `mode, n_candidates, n_bets, win_rate, total_staked, pnl, roi, ending_bankroll, max_drawdown`. `flat` stakes 1 unit per qualifying bet; `kelly` stakes `kelly_mult * kelly_fraction * bankroll` capped at `cap*bankroll`, fixed (non-compounding) notional.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation_roi.py
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import roi


def test_devig_normalizes_to_one():
    p1, p2 = roi.devig_two_way(1.5, 2.5)
    assert abs((p1 + p2) - 1.0) < 1e-9
    assert p1 > p2  # shorter odds -> higher prob


def test_flat_roi_winning_edge():
    # model loves player1 (0.8) vs fair ~0.5; p1 always wins -> positive ROI
    df = pd.DataFrame([
        dict(p1_prob=0.8, p1_odds_decimal=2.0, p2_odds_decimal=2.0, y1=1),
        dict(p1_prob=0.8, p1_odds_decimal=2.0, p2_odds_decimal=2.0, y1=1),
    ])
    out = roi.simulate(df, mode="flat")
    assert out["n_bets"] == 2
    assert out["roi"] > 0
    assert out["pnl"] > 0


def test_no_bet_when_edge_below_threshold():
    df = pd.DataFrame([dict(p1_prob=0.5, p1_odds_decimal=2.0, p2_odds_decimal=2.0, y1=1)])
    out = roi.simulate(df, mode="flat", edge_threshold=0.02)
    assert out["n_bets"] == 0
    assert out["roi"] == 0.0


def test_kelly_caps_stake():
    df = pd.DataFrame([dict(p1_prob=0.99, p1_odds_decimal=5.0, p2_odds_decimal=1.2, y1=1)])
    out = roi.simulate(df, mode="kelly", kelly_mult=1.0, cap=0.05, bankroll=1000.0)
    assert out["total_staked"] <= 0.05 * 1000.0 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_roi.py -q`
Expected: FAIL — `No module named 'evaluation.roi'`.

- [ ] **Step 3: Write minimal implementation**

```python
# production/evaluation/roi.py
"""Counterfactual staking simulation at logged odds. Player1-perspective probs."""
from __future__ import annotations
import numpy as np
import pandas as pd


def devig_two_way(o1: float, o2: float) -> tuple[float, float]:
    q1, q2 = 1.0 / o1, 1.0 / o2
    s = q1 + q2
    return q1 / s, q2 / s


def _kelly_fraction(p: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * p - (1 - p)) / b
    return max(0.0, f)


def simulate(df: pd.DataFrame, mode: str = "flat", edge_threshold: float = 0.02,
             kelly_mult: float = 0.18, cap: float = 0.05, bankroll: float = 1000.0) -> dict:
    n_candidates = 0
    staked, pnl, wins, n_bets = 0.0, 0.0, 0, 0
    equity, peak, max_dd = bankroll, bankroll, 0.0
    for _, r in df.iterrows():
        o1, o2 = r.get("p1_odds_decimal"), r.get("p2_odds_decimal")
        if pd.isna(o1) or pd.isna(o2) or o1 <= 1 or o2 <= 1:
            continue
        n_candidates += 1
        fair1, fair2 = devig_two_way(float(o1), float(o2))
        p1 = float(r["p1_prob"])
        edge1, edge2 = p1 - fair1, (1 - p1) - fair2
        if edge1 >= edge2:
            side_p, side_odds, edge, won = p1, float(o1), edge1, (r["y1"] == 1)
        else:
            side_p, side_odds, edge, won = 1 - p1, float(o2), edge2, (r["y1"] == 0)
        if edge < edge_threshold:
            continue
        if mode == "flat":
            stake = 1.0
        else:
            stake = min(kelly_mult * _kelly_fraction(side_p, side_odds) * bankroll, cap * bankroll)
        if stake <= 0:
            continue
        n_bets += 1
        staked += stake
        profit = stake * (side_odds - 1.0) if won else -stake
        pnl += profit
        wins += int(won)
        equity += profit
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "mode": mode, "n_candidates": n_candidates, "n_bets": n_bets,
        "win_rate": (wins / n_bets) if n_bets else 0.0,
        "total_staked": staked, "pnl": pnl,
        "roi": (pnl / staked) if staked else 0.0,
        "ending_bankroll": bankroll + pnl, "max_drawdown": max_dd,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_roi.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add production/evaluation/roi.py tests/test_evaluation_roi.py
git commit -m "Add evaluation.roi: de-vig + flat/Kelly counterfactual staking"
```

---

### Task 4: `offline.py` — ingest offline experiment metrics

**Files:**
- Create: `production/evaluation/offline.py`
- Test: `tests/test_evaluation_offline.py`

**Interfaces:**
- Produces: `discover_experiment_metrics(experiments_root: str) -> pd.DataFrame` — one row per experiment artifact, columns: `source ("offline"), experiment, family, split, accuracy, auc, log_loss, brier, ece, path`. Missing metrics are `NaN`, never dropped silently.

- [ ] **Step 1 (investigation, not a test): determine the actual on-disk format.**

Run: `ls -R results/professional_tennis/experiments/ | head -60` and inspect one metrics file (look for `metrics.json`, `results.json`, or a summary CSV). Note the JSON keys / CSV columns used for accuracy/auc/logloss/brier/ece. Record findings in a comment at the top of `offline.py`.

- [ ] **Step 2: Write the failing test** (fixture mirroring the discovered format — example assumes per-experiment `metrics.json`)

```python
# tests/test_evaluation_offline.py
import sys, json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pandas as pd
from evaluation import offline


def test_discovers_and_tolerates_missing(tmp_path):
    exp = tmp_path / "2026-04-25" / "xgboost" / "performance_v1__xgb_depth5_recency_hl_12y"
    exp.mkdir(parents=True)
    (exp / "metrics.json").write_text(json.dumps({
        "test_accuracy": 0.6713, "test_auc": 0.7312, "test_log_loss": 0.604, "test_brier": 0.209
        # ece intentionally absent
    }))
    df = offline.discover_experiment_metrics(str(tmp_path))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["family"] == "xgboost"
    assert abs(row["accuracy"] - 0.6713) < 1e-9
    assert pd.isna(row["ece"])  # missing -> NaN, not dropped
    assert row["source"] == "offline"
```

- [ ] **Step 3: Implement `discover_experiment_metrics`** to walk `experiments_root`, read each metrics file in the discovered format, map keys tolerantly (`test_accuracy|accuracy`, `test_auc|auc`, `test_log_loss|log_loss|logloss`, `test_brier|brier`, `ece`), and infer `family`/`experiment`/`split` from the path segments. Use the real format found in Step 1; if multiple formats exist, handle each branch. Code must read the actual files — no hardcoded values.

- [ ] **Step 4: Run test to verify it passes**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_offline.py -q`
Expected: PASS. Then run against the real dir from a Python shell and confirm row count > 0; log how many experiments had missing metrics.

- [ ] **Step 5: Commit**

```bash
git add production/evaluation/offline.py tests/test_evaluation_offline.py
git commit -m "Add evaluation.offline: ingest offline experiment metrics"
```

---

### Task 5: `ledger.py` — assemble + write CSV + markdown report + CLI

**Files:**
- Create: `production/evaluation/ledger.py`
- Test: `tests/test_evaluation_ledger.py`

**Interfaces:**
- Consumes: `cohorts.build_scored_frame`, `cohorts.intersection_uids`, `metrics.compute_all`, `roi.simulate`, `offline.discover_experiment_metrics`.
- Produces:
  - `build_live_ledger(scored: pd.DataFrame) -> pd.DataFrame` — one row per `(model, tier)` for tiers `gold` and `complete`, with all metrics + flat ROI + kelly ROI. Plus an `intersection` block per tier (metrics recomputed on the all-model intersection cohort).
  - `write_outputs(live: pd.DataFrame, offline_df: pd.DataFrame, out_dir: str, report_path: str, run_date: str) -> None` — writes `<out_dir>/model_ledger.csv` and a markdown report.
  - `main(argv=None)` — CLI: `--prod-dir`, `--experiments-root`, `--out-dir`, `--report`, `--run-date` (no `Date.now()`; date passed in / defaults via `datetime` at runtime is fine here since this is a normal script, not a workflow).

- [ ] **Step 1: Write the failing test** (uses the synthetic pred_log from Task 2's pattern)

```python
# tests/test_evaluation_ledger.py
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import cohorts, ledger


def _scored():
    rows = []
    rng = np.random.default_rng(1)
    for i in range(60):
        uid = f"m{i}"
        y = int(rng.uniform() < 0.5)
        for model, skill in [("nn", 0.55), ("xgb", 0.75), ("market", 0.7)]:
            p = skill if y == 1 else (1 - skill)
            p = min(max(p + rng.normal(0, 0.05), 0.02), 0.98)
            rows.append(dict(match_uid=uid, model=model, family=model, p1_prob=p,
                             p1_odds_decimal=1.9, p2_odds_decimal=1.9, y1=y,
                             is_gold=True, is_complete=True))
    return pd.DataFrame(rows)


def test_build_live_ledger_has_row_per_model_and_metrics():
    live = ledger.build_live_ledger(_scored())
    assert {"nn", "xgb", "market"} <= set(live["model"])
    for col in ["accuracy", "log_loss", "brier", "auc", "ece", "roi_flat", "roi_kelly", "n", "tier"]:
        assert col in live.columns
    # xgb is more skillful than nn -> lower log loss on gold
    g = live[live.tier == "gold"].set_index("model")
    assert g.loc["xgb", "log_loss"] < g.loc["nn", "log_loss"]


def test_write_outputs(tmp_path):
    live = ledger.build_live_ledger(_scored())
    offline_df = pd.DataFrame(columns=["source", "experiment", "family", "split",
                                       "accuracy", "auc", "log_loss", "brier", "ece", "path"])
    out_dir = tmp_path / "ledger"
    report = tmp_path / "MODEL_LEDGER.md"
    ledger.write_outputs(live, offline_df, str(out_dir), str(report), "2026-06-21")
    assert (out_dir / "model_ledger.csv").exists()
    assert report.exists()
    assert "Model Ledger" in report.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_ledger.py -q`
Expected: FAIL — `No module named 'evaluation.ledger'`.

- [ ] **Step 3: Implement** `build_live_ledger` (group scored by model within each tier, call `metrics.compute_all` on `y1` vs `p1_prob`, call `roi.simulate` twice for flat+kelly, emit `tier∈{gold,complete}`; also compute an intersection block restricting to `intersection_uids(scored, all_models, tier_col)` labeled `tier=gold_intersection`/`complete_intersection`), `write_outputs` (CSV + a markdown report that ranks models by log_loss then ROI, with cohort sizes and an explicit "Verdict" placeholder section filled from the data), and `main` (argparse + `datetime.now().strftime("%Y-%m-%d")` default run-date). Keep file focused; push any helper formatting into small local functions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `tennis_env/bin/python -m pytest tests/test_evaluation_ledger.py -q`
Expected: PASS (2 passed). Then run the full new suite: `tennis_env/bin/python -m pytest tests/test_evaluation_*.py -q` (expect all green).

- [ ] **Step 5: Commit**

```bash
git add production/evaluation/ledger.py tests/test_evaluation_ledger.py
git commit -m "Add evaluation.ledger: assemble ledger + write CSV/markdown + CLI"
```

---

### Task 6: Refactor `dashboard/data.py` to reuse `evaluation.metrics` (DRY)

**Files:**
- Modify: `dashboard/data.py` (the `build_metrics_summary` computation, ~lines 446–504)
- Test: `tests/test_dashboard_smoke.py` (existing — must still pass)

- [ ] **Step 1:** Read `dashboard/data.py` around `build_metrics_summary`. Confirm how it puts `production/` on `sys.path` (add the standard preamble if missing so `from evaluation import metrics` resolves).
- [ ] **Step 2:** Replace the inline accuracy/AUC/Brier/log-loss/ECE math with calls to `metrics.compute_all(y_true, p)`, mapping results into the existing return structure (keep output keys identical so the dashboard renders unchanged). Do not change the dashboard's public output shape.
- [ ] **Step 3:** Run: `tennis_env/bin/python -m pytest tests/test_dashboard_smoke.py -q` → Expected: PASS. Also run the whole suite `tennis_env/bin/python -m pytest tests -q` and confirm no regressions.
- [ ] **Step 4: Commit**

```bash
git add dashboard/data.py
git commit -m "Refactor dashboard to reuse evaluation.metrics (remove duplicate math)"
```

---

### Task 7: Generate the real ledger + write the verdict doc

**Files:**
- Create (generated): `results/professional_tennis/ledger/2026-06-21/model_ledger.csv`
- Create: `docs/modeling/MODEL_LEDGER.md`

- [ ] **Step 1:** Run the CLI on real logs:
  `cd production && ../tennis_env/bin/python -m evaluation.ledger --prod-dir . --experiments-root ../results/professional_tennis/experiments --out-dir ../results/professional_tennis/ledger/2026-06-21 --report ../docs/modeling/MODEL_LEDGER.md`
- [ ] **Step 2:** Inspect `model_ledger.csv`. Sanity checks: GOLD n≈358, COMPLETE n≈1162; NN/XGB/RF/market/shadow_xgboost present; shadow_catboost/lightgbm/nn absent (0 settled — expected, will be unlocked in Task 10). Confirm the rankings make sense.
- [ ] **Step 3:** Ensure `MODEL_LEDGER.md` contains: cohort definitions + sizes, the per-model table (ranked by log loss then ROI), the intersection verdict, and a plain-English "which model should drive bets and are we beating the vig" conclusion. Note explicitly that the shadow trio is pending settlement.
- [ ] **Step 4: Commit** (commit the doc + the dated CSV ledger as an intentional artifact; do NOT commit `prediction_log.csv`)

```bash
git add docs/modeling/MODEL_LEDGER.md results/professional_tennis/ledger/2026-06-21/model_ledger.csv
git commit -m "Generate first model ledger and document verdict"
```

---

### Task 8: Offline end-to-end pipeline smoke test

**Files:**
- Create: `tests/test_pipeline_smoke.py`

**Interfaces:** mocks `odds.fetch_bovada.fetch_bovada_tennis_odds` and the TA scraper / feature calculator so no network is hit; drives the orchestrator through prediction + a settlement-enrich path on a tmp copy of a tiny `prediction_log.csv`.

- [ ] **Step 1:** Read `production/main.py` `LiveBettingOrchestrator` to find the smallest seam to drive end-to-end without network (likely: monkeypatch `fetch_bovada_tennis_odds` to return a 2-row odds DataFrame, monkeypatch the TA feature builder to return canned 141-feature rows, point logging at a tmp dir). Identify exact patch targets.
- [ ] **Step 2: Write the test** that: patches those targets, runs the predict path on a 2-match fixture, and asserts (a) it returns without exception, (b) prediction rows are written with NN/XGB/RF probabilities present, (c) the 141-feature schema contract holds. Add a second test that feeds one already-completed match into the settlement-enrich function and asserts the row gets `actual_winner`/`settled_at` without recomputing inference.
- [ ] **Step 3:** Run: `tennis_env/bin/python -m pytest tests/test_pipeline_smoke.py -q` → iterate until PASS. Then full suite green.
- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_smoke.py
git commit -m "Add offline end-to-end pipeline smoke test (mocked Bovada/TA)"
```

---

### Task 9: Live validation run + TA diagnosis (operational, supervised)

**Files:** none committed (operational). Capture console output to `scratchpad/` for the summary.

- [ ] **Step 1: TA connectivity probe (cheap).** From `production/`, run a tiny script that fetches ONE Tennis Abstract page via the existing `scraping/ta_scraper.py` client and prints HTTP status + whether content parsed. If it 429s/blocks/times out, STOP and switch to **superpowers:systematic-debugging** (compare against the known-good past behavior; check headers/rate-limit/delay/UA changes) before any bulk run.
- [ ] **Step 2: Settlement-only catch-up (throttled).** Run `auto_settle.py` (uncapped catch-up with the gentle 8s pacing already in place) and watch for 429s. This settles the May/June backlog and unlocks the shadow trio.
- [ ] **Step 3: Prediction dry-run.** Run `python production/main.py --dry-run --skip-auto-settle` (settlement already done in Step 2; `--dry-run` = no betting session/bet logging). Confirm: odds fetched, features extracted, predictions logged, shadow rows logged. Capture counts.
- [ ] **Step 4:** Record outcomes (settled count, any TA issues + fix, fresh prediction counts) in the scratchpad summary for the final report. No commit (these mutate generated logs only).

---

### Task 10: Re-run ledger with newly-settled data + finalize docs

**Files:**
- Update (generated): `results/professional_tennis/ledger/2026-06-21/model_ledger.csv` (or a new dated dir if the run lands on a new date)
- Modify: `docs/modeling/MODEL_LEDGER.md`, `AGENTS.md`, `docs/production/README.md`, `docs/modeling/EXPERIMENT_WORKFLOW.md`

- [ ] **Step 1:** Re-run the ledger CLI (Task 7 Step 1). Confirm the shadow trio (catboost/lightgbm/nn) now have settled rows and appear in the live ledger.
- [ ] **Step 2:** Update `MODEL_LEDGER.md` with the refreshed numbers + final verdict (all 7+ models).
- [ ] **Step 3:** Update `AGENTS.md`: add the evaluation ledger as the **source of truth for model performance**, document the regenerate command, the GOLD/COMPLETE tier definitions, and the intersection rule. Add pointers in `docs/production/README.md` and `docs/modeling/EXPERIMENT_WORKFLOW.md`.
- [ ] **Step 4: Commit**

```bash
git add docs/modeling/MODEL_LEDGER.md AGENTS.md docs/production/README.md docs/modeling/EXPERIMENT_WORKFLOW.md results/professional_tennis/ledger
git commit -m "Refresh ledger with settled shadow models; document evaluation source of truth"
```

---

## Self-Review

**Spec coverage:** Goals 1–5 → Tasks 1–7 (ledger/metrics/ROI/offline/CSV+md), Task 6 (DRY dashboard), Tasks 8–9 (smoke + live validation), Task 10 (unlock shadows + docs). Cohort honesty → Task 2 + Task 7. ROI flat+Kelly → Task 3. Offline variants → Task 4. Source-of-truth docs → Task 10. All spec sections map to a task.

**Placeholder scan:** Code-bearing tasks (1,2,3,5) carry complete implementations + tests. Tasks 4, 8 require on-disk/seam inspection first (explicit investigation steps) because the real format/seam must be read, not guessed — the code is then written against what's found, with concrete fixtures shown. Tasks 6, 9, 10 are mechanical/operational and specify exact files, commands, and assertions.

**Type consistency:** `compute_all` keys (`accuracy, log_loss, brier, auc, ece, cal_slope, cal_intercept, n`) are consumed unchanged by `build_live_ledger`. Scored-frame columns (`match_uid, model, family, p1_prob, p1_odds_decimal, p2_odds_decimal, y1, is_gold, is_complete`) are produced by `cohorts.build_scored_frame` and consumed by `roi.simulate` (`p1_prob, p1_odds_decimal, p2_odds_decimal, y1`) and `ledger.build_live_ledger`. `intersection_uids(scored, models, tier_col)` signature matches its ledger call site. Consistent.
