import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import ledger


def _scored():
    rows = []
    rng = np.random.default_rng(1)
    for i in range(80):
        uid = f"m{i}"
        y = int(rng.uniform() < 0.5)
        for model, skill in [("nn", 0.55), ("xgb", 0.78), ("rf", 0.7), ("market", 0.72)]:
            base = skill if y == 1 else (1 - skill)
            p = min(max(base + rng.normal(0, 0.05), 0.02), 0.98)
            rows.append(dict(match_uid=uid, model=model, family=model, p1_prob=p,
                             p1_odds_decimal=1.9, p2_odds_decimal=1.9, y1=y,
                             is_gold=True, is_complete=True))
    return pd.DataFrame(rows)


def test_build_live_ledger_columns_and_ranking():
    live = ledger.build_live_ledger(_scored())
    assert {"nn", "xgb", "rf", "market"} <= set(live["model"])
    for col in ["accuracy", "log_loss", "brier", "auc", "ece",
                "roi_flat", "roi_kelly", "n", "tier"]:
        assert col in live.columns
    # gold tier present, plus intersection tier
    assert "gold" in set(live["tier"])
    assert "gold_intersection" in set(live["tier"])
    # xgb is the most skillful synthetic model -> lower log loss than nn on gold
    g = live[live.tier == "gold"].set_index("model")
    assert g.loc["xgb", "log_loss"] < g.loc["nn", "log_loss"]


def test_write_outputs(tmp_path):
    live = ledger.build_live_ledger(_scored())
    offline_df = pd.DataFrame(columns=["source", "run_date", "experiment", "family",
                                       "feature_set", "feature_mode", "split", "n_features",
                                       "accuracy", "auc", "log_loss", "brier", "ece", "path"])
    out_dir = tmp_path / "ledger"
    report = tmp_path / "MODEL_LEDGER.md"
    ledger.write_outputs(live, offline_df, str(out_dir), str(report), "2026-06-21")
    assert (out_dir / "model_ledger.csv").exists()
    assert report.exists()
    text = report.read_text()
    assert "Model Ledger" in text
    assert "2026-06-21" in text
