"""
Re-score all logged matches using the SURFACE_FIX models.

Uses the already-logged feature vectors (no re-scraping needed).
Compares new SURFACE_FIX predictions against the original predictions
stored in prediction_log.csv and match_features_export.csv.
"""

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

try:
    from models.nn_runtime import TennisNet
    from models.registry_utils import resolve_artifact_path
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .models.nn_runtime import TennisNet
    from .models.registry_utils import resolve_artifact_path

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "..", "results", "professional_tennis")
LOGS = os.path.join(BASE, "logs")


# ── 1. Load all feature logs ──────────────────────────────────────────────────
print("=" * 60)
print("LOADING LOGGED FEATURES")
print("=" * 60)

dfs = []
for path in sorted(glob.glob(os.path.join(LOGS, "features_*.csv"))):
    try:
        df = pd.read_csv(path)
        df["_source_file"] = os.path.basename(path)
        dfs.append(df)
    except Exception as e:
        print(f"  skip {path}: {e}")

all_features = pd.concat(dfs, ignore_index=True)
print(f"Total rows across all feature logs: {len(all_features)}")

# Important: do NOT deduplicate by match_id. Older feature logs used the per-run
# row index as match_id, so the same integer appears across many unrelated runs.
if "feature_snapshot_id" in all_features.columns:
    before = len(all_features)
    all_features = all_features.sort_values("_source_file").drop_duplicates(
        subset=["feature_snapshot_id"], keep="last"
    )
    print(f"After dedup by feature_snapshot_id: {len(all_features)} unique snapshots (removed {before - len(all_features)})")
else:
    print("No feature_snapshot_id column found — keeping all feature rows and using fallback matching")

print(f"Date range from timestamps: {all_features['timestamp'].min()} → {all_features['timestamp'].max()}"
      if "timestamp" in all_features.columns else "")


# ── 2. Load model feature list ────────────────────────────────────────────────
xgb_model_path = resolve_artifact_path("xgboost") or os.path.join(RESULTS, "XGBoost", "xgboost_model_SURFACE_FIX.json")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(xgb_model_path))
MODEL_FEATURES = list(xgb_model.feature_names_in_)
print(f"\nModel expects {len(MODEL_FEATURES)} features")

# Check coverage
missing_from_log = [f for f in MODEL_FEATURES if f not in all_features.columns]
if missing_from_log:
    print(f"  WARNING — features missing from logs (will be filled 0): {missing_from_log}")
for f in missing_from_log:
    all_features[f] = 0.0


# ── 3. Build feature matrix ───────────────────────────────────────────────────
X = all_features[MODEL_FEATURES].fillna(0.0).astype(float)
print(f"Feature matrix shape: {X.shape}")


# ── 4. XGBoost predictions ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RUNNING SURFACE_FIX MODELS")
print("=" * 60)

xgb_probs = xgb_model.predict_proba(X)[:, 1]
print(f"XGBoost: min={xgb_probs.min():.3f}  max={xgb_probs.max():.3f}  mean={xgb_probs.mean():.3f}")


# ── 5. Neural Network predictions ────────────────────────────────────────────
nn_model_path = resolve_artifact_path("nn") or os.path.join(RESULTS, "Neural_Network", "neural_network_model_SURFACE_FIX.pth")
nn_scaler_path = resolve_artifact_path("nn", "scaler_file") or os.path.join(RESULTS, "Neural_Network", "scaler_SURFACE_FIX.pkl")

with open(nn_scaler_path, "rb") as f:
    scaler = pickle.load(f)

nn_model = TennisNet(input_size=len(MODEL_FEATURES))
nn_model.load_state_dict(torch.load(nn_model_path, map_location="cpu"))
nn_model.eval()

X_scaled = scaler.transform(X.values)
with torch.no_grad():
    nn_probs = nn_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

print(f"Neural Net: min={nn_probs.min():.3f}  max={nn_probs.max():.3f}  mean={nn_probs.mean():.3f}")


# ── 6. Random Forest predictions ─────────────────────────────────────────────
rf_path = resolve_artifact_path("random_forest") or os.path.join(RESULTS, "Random_Forest", "random_forest_model_SURFACE_FIX.pkl")
with open(rf_path, "rb") as f:
    rf_model = pickle.load(f)

rf_probs = rf_model.predict_proba(X)[:, 1]
print(f"Random Forest: min={rf_probs.min():.3f}  max={rf_probs.max():.3f}  mean={rf_probs.mean():.3f}")


# ── 7. Ensemble (equal weight) ────────────────────────────────────────────────
ensemble_probs = (xgb_probs + nn_probs + rf_probs) / 3.0
print(f"Ensemble:      min={ensemble_probs.min():.3f}  max={ensemble_probs.max():.3f}  mean={ensemble_probs.mean():.3f}")


# ── 8. Build results table ────────────────────────────────────────────────────
meta_cols = [c for c in ["match_id", "player1_raw", "player2_raw", "event",
                          "timestamp", "meta_surface_input", "meta_round_input",
                          "_source_file", "_has_defaulted_features"]
             if c in all_features.columns]

results = all_features[meta_cols].copy()
results["surface_fix_xgb_p1"]  = xgb_probs
results["surface_fix_nn_p1"]   = nn_probs
results["surface_fix_rf_p1"]   = rf_probs
results["surface_fix_ensemble_p1"] = ensemble_probs

# Flag extreme predictions (>90% or <10%)
results["xgb_extreme"] = (xgb_probs > 0.90) | (xgb_probs < 0.10)
results["nn_extreme"]  = (nn_probs  > 0.90) | (nn_probs  < 0.10)

print(f"\nExtreme XGBoost predictions (>90% or <10%): {results['xgb_extreme'].sum()}")
print(f"Extreme NN predictions (>90% or <10%):      {results['nn_extreme'].sum()}")


# ── 9. Join with prediction_log for comparison ────────────────────────────────
pred_log_path = os.path.join(BASE, "prediction_log.csv")
if os.path.exists(pred_log_path):
    pred_log = pd.read_csv(pred_log_path)
    print(f"\nPrediction log: {len(pred_log)} rows")

    def norm(s):
        return str(s).lower().strip().replace("-", " ").replace(".", "")

    results["_p1n"] = results["player1_raw"].apply(norm) if "player1_raw" in results.columns else ""
    results["_p2n"] = results["player2_raw"].apply(norm) if "player2_raw" in results.columns else ""
    results["_feature_ts"] = pd.to_datetime(results.get("timestamp"), errors="coerce")
    if "meta_match_date" in results.columns:
        results["_feature_match_date"] = pd.to_datetime(results["meta_match_date"], errors="coerce")
    else:
        results["_feature_match_date"] = pd.NaT

    pred_log["_p1n"] = pred_log["p1"].apply(norm)
    pred_log["_p2n"] = pred_log["p2"].apply(norm)
    pred_log["_logged_ts"] = pd.to_datetime(
        pred_log["odds_scraped_at"].fillna(pred_log["logged_at"]),
        errors="coerce",
    )
    pred_log["_pred_match_date"] = pd.to_datetime(pred_log["match_date"], errors="coerce")

    merged = pred_log.copy()
    for col in results.columns:
        if col not in merged.columns:
            merged[col] = np.nan

    matched_indices = set()

    if "feature_snapshot_id" in pred_log.columns and "feature_snapshot_id" in results.columns:
        exact = pred_log.merge(
            results,
            on="feature_snapshot_id",
            how="left",
            suffixes=("_orig", "_new"),
        )
        has_exact = exact["surface_fix_xgb_p1"].notna()
        merged.loc[has_exact, exact.columns] = exact.loc[has_exact, exact.columns]
        matched_indices.update(exact[has_exact].index.tolist())

    for idx, row in merged.iterrows():
        if idx in matched_indices and pd.notna(merged.at[idx, "surface_fix_xgb_p1"]):
            continue

        candidates = results[
            (results["_p1n"] == row["_p1n"]) &
            (results["_p2n"] == row["_p2n"])
        ].copy()

        if candidates.empty:
            continue

        if pd.notna(row["_pred_match_date"]) and candidates["_feature_match_date"].notna().any():
            date_diff = (candidates["_feature_match_date"] - row["_pred_match_date"]).abs()
            candidates = candidates[date_diff <= pd.Timedelta(days=3)]
            if candidates.empty:
                continue

        if pd.notna(row["_logged_ts"]) and candidates["_feature_ts"].notna().any():
            candidates["_delta"] = (candidates["_feature_ts"] - row["_logged_ts"]).abs()
            best = candidates.sort_values("_delta").iloc[0]
        else:
            best = candidates.iloc[-1]

        for col in results.columns:
            merged.at[idx, col] = best.get(col)

    if "model_p1_prob" in merged.columns and "surface_fix_xgb_p1" in merged.columns:
        matched = merged["surface_fix_xgb_p1"].notna()
        print(f"Matched {matched.sum()} / {len(merged)} prediction-log rows to feature data")

        comparison = merged[matched][[
            "p1", "p2",
            "model_p1_prob", "surface_fix_xgb_p1", "surface_fix_nn_p1",
            "surface_fix_ensemble_p1", "actual_winner", "model_correct",
        ]].copy()
        comparison["xgb_delta"] = comparison["surface_fix_xgb_p1"] - comparison["model_p1_prob"]
        comparison["xgb_pred_correct"] = (
            ((comparison["surface_fix_xgb_p1"] > 0.5) & (comparison["actual_winner"] == 1)) |
            ((comparison["surface_fix_xgb_p1"] < 0.5) & (comparison["actual_winner"] == 2))
        )

        settled = comparison["actual_winner"].notna()
        print("\n" + "=" * 60)
        print("COMPARISON: Original vs SURFACE_FIX (settled matches only)")
        print("=" * 60)
        if settled.sum() > 0:
            orig_acc  = comparison.loc[settled, "model_correct"].mean()
            new_acc   = comparison.loc[settled, "xgb_pred_correct"].mean()
            delta_avg = comparison.loc[settled, "xgb_delta"].abs().mean()
            print(f"Settled matches compared: {settled.sum()}")
            print(f"Original model accuracy:  {orig_acc:.3f} ({orig_acc*100:.1f}%)")
            print(f"SURFACE_FIX XGB accuracy: {new_acc:.3f} ({new_acc*100:.1f}%)")
            print(f"Avg abs probability shift: {delta_avg:.4f}")
            print(f"\nLargest probability shifts (top 10):")
            top_shifts = comparison.loc[settled].reindex(
                comparison.loc[settled, "xgb_delta"].abs().nlargest(10).index
            )
            for _, row in top_shifts.iterrows():
                print(f"  {row['p1']} vs {row['p2']}: orig={row['model_p1_prob']:.3f} → new={row['surface_fix_xgb_p1']:.3f} (Δ{row['xgb_delta']:+.3f})")
        else:
            print("No settled matches found in joined data — showing probability distribution only")
            print(comparison[["p1","p2","model_p1_prob","surface_fix_xgb_p1","xgb_delta"]].to_string())

        comparison.to_csv(os.path.join(BASE, "logs", "rescore_comparison.csv"), index=False)
        print(f"\nComparison saved to: production/logs/rescore_comparison.csv")


# ── 10. Save full rescore results ─────────────────────────────────────────────
out_path = os.path.join(LOGS, "rescore_surface_fix_all.csv")
results.to_csv(out_path, index=False)
print(f"\nFull rescore saved to: {out_path}")

print("\n" + "=" * 60)
print("PROBABILITY DISTRIBUTION SUMMARY (all logged matches)")
print("=" * 60)
for name, probs in [("XGBoost", xgb_probs), ("Neural Net", nn_probs),
                     ("Random Forest", rf_probs), ("Ensemble", ensemble_probs)]:
    buckets = {
        "<10%":  (probs < 0.10).sum(),
        "10-25%": ((probs >= 0.10) & (probs < 0.25)).sum(),
        "25-40%": ((probs >= 0.25) & (probs < 0.40)).sum(),
        "40-60%": ((probs >= 0.40) & (probs < 0.60)).sum(),
        "60-75%": ((probs >= 0.60) & (probs < 0.75)).sum(),
        "75-90%": ((probs >= 0.75) & (probs < 0.90)).sum(),
        ">90%":   (probs >= 0.90).sum(),
    }
    print(f"\n{name}:")
    for bucket, count in buckets.items():
        pct = count / len(probs) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>7}: {count:4d} ({pct:5.1f}%) {bar}")
