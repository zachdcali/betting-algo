import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import ledger, metrics


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
                "roi_flat", "roi_flat_kalshi", "n_bets_flat_kalshi",
                "kalshi_since", "roi_kelly", "n", "tier"]:
        assert col in live.columns
    # gold tier present, plus intersection tier
    assert "gold" in set(live["tier"])
    assert "gold_intersection" in set(live["tier"])
    # xgb is the most skillful synthetic model -> lower log loss than nn on gold
    g = live[live.tier == "gold"].set_index("model")
    assert g.loc["xgb", "log_loss"] < g.loc["nn", "log_loss"]


def test_live_ledger_discloses_source_start_before_first_kalshi_bet():
    scored = _scored()
    scored["kalshi_p1_ask"] = float("nan")
    scored["kalshi_p2_ask"] = float("nan")
    scored["kalshi_observation_at"] = pd.NaT
    scored.attrs["kalshi_logging_start"] = "2026-07-17T01:02:03+00:00"
    live = ledger.build_live_ledger(scored)
    assert set(live["n_bets_flat_kalshi"]) == {0}
    assert set(live["kalshi_since"]) == {"2026-07-17T01:02:03+00:00"}


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
    assert "# Model Ledger snapshot — 2026-06-21" in text
    assert "Dated full ledger snapshot, including live metrics and offline experiments" in text
    assert "may lag the public dashboard" in text
    assert "manifest-pinned accepted sync" in text


def test_all_model_intersection_is_dynamic_and_excludes_partial_shadow_coverage():
    scored = _scored()
    shadow_rows = []
    for uid in ["m0", "m1", "m2"]:
        base = scored[(scored["match_uid"] == uid) & (scored["model"] == "nn")].iloc[0]
        shadow_rows.append({
            **base.to_dict(), "model": "shadow_cat_v1", "family": "catboost",
            "p1_prob": 0.64,
        })
    for uid in ["m0", "m1"]:
        base = scored[(scored["match_uid"] == uid) & (scored["model"] == "nn")].iloc[0]
        shadow_rows.append({
            **base.to_dict(), "model": "shadow_lgbm_v1", "family": "lightgbm",
            "p1_prob": 0.63,
        })
    scored = pd.concat([scored, pd.DataFrame(shadow_rows)], ignore_index=True)

    live = ledger.build_live_ledger(
        scored,
        active_shadow_models=["shadow_cat_v1", "shadow_lgbm_v1"],
    )
    all_common = live[live["tier"] == "gold_all_model_intersection"]

    assert set(all_common["model"]) == {
        "nn", "xgb", "rf", "market", "shadow_cat_v1", "shadow_lgbm_v1",
    }
    assert set(all_common["n"]) == {2}


def test_all_model_intersection_ignores_retired_shadow_history():
    scored = _scored()
    shadow_rows = []
    for uid in ["m0", "m1", "m2"]:
        base = scored[(scored["match_uid"] == uid) & (scored["model"] == "nn")].iloc[0]
        shadow_rows.extend([
            {**base.to_dict(), "model": "shadow_active", "p1_prob": 0.62},
            {**base.to_dict(), "model": "shadow_retired", "p1_prob": 0.61},
        ])
    scored = pd.concat([scored, pd.DataFrame(shadow_rows)], ignore_index=True)

    live = ledger.build_live_ledger(
        scored, active_shadow_models=["shadow_active"],
    )
    all_common = live[live["tier"] == "gold_all_model_intersection"]

    assert "shadow_active" in set(all_common["model"])
    assert "shadow_retired" not in set(all_common["model"])
    assert "shadow_retired" in set(live.loc[live["tier"] == "gold", "model"])


def test_market_timing_models_only_appear_in_dedicated_tiers():
    scored = _scored()
    timing_rows = []
    for uid in ["m0", "m1"]:
        base = scored[(scored["match_uid"] == uid) & (scored["model"] == "market")].iloc[0]
        for model in ["market_open", "market_close"]:
            timing_rows.append({**base.to_dict(), "model": model})
    scored = pd.concat([scored, pd.DataFrame(timing_rows)], ignore_index=True)

    live = ledger.build_live_ledger(scored)

    assert live[
        live["tier"].isin(["gold", "complete"])
        & live["model"].isin(["market_open", "market_close"])
    ].empty
    assert set(live.loc[live["tier"] == "settled_market_timing", "model"]) == {
        "market_open", "market_close",
    }


def test_settled_market_timing_uses_valid_results_without_feature_tier_filtering():
    scored = _scored()
    timing_rows = []
    for uid, is_gold in [("m0", True), ("m1", False)]:
        base = scored[(scored["match_uid"] == uid) & (scored["model"] == "market")].iloc[0]
        for model, probability in [("market_open", 0.48), ("market_close", 0.52)]:
            timing_rows.append({
                **base.to_dict(), "model": model, "family": "market",
                "p1_prob": probability, "is_gold": is_gold,
                "is_complete": is_gold,
            })
    scored = pd.concat([scored, pd.DataFrame(timing_rows)], ignore_index=True)

    live = ledger.build_live_ledger(scored)
    settled = live[live["tier"] == "settled_market_timing"]
    gold = live[live["tier"] == "gold_market_timing"]

    assert set(settled["model"]) == {"market_open", "market_close"}
    assert set(settled["n"]) == {2}
    assert set(gold["n"]) == {1}
    calibration = ledger.build_calibration_ledger(scored, live)
    settled_bins = calibration[calibration["tier"] == "settled_market_timing"]
    assert set(settled_bins.groupby("model")["count"].sum()) == {2}


def test_calibration_ledger_exactly_materializes_authoritative_reliability_bins():
    scored = _scored()
    live = ledger.build_live_ledger(scored)

    calibration = ledger.build_calibration_ledger(scored, live)
    actual = (
        calibration[(calibration["model"] == "nn") & (calibration["tier"] == "gold")]
        .sort_values("bin_index")
        .reset_index(drop=True)
    )
    nn_gold = scored[scored["is_gold"] & scored["model"].eq("nn")]
    expected = metrics.reliability_table(
        nn_gold["y1"].values, nn_gold["p1_prob"].values
    ).reset_index(names="bin_index")

    pd.testing.assert_frame_equal(
        actual[["bin_index", "bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"]],
        expected[["bin_index", "bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"]],
        check_dtype=False,
    )
    assert int(actual["count"].sum()) == int(
        live[(live["model"] == "nn") & (live["tier"] == "gold")].iloc[0]["n"]
    )
