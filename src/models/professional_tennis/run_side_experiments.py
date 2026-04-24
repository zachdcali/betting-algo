from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from experiment_utils import (
    build_blocked_windows,
    build_fixed_split,
    compute_metrics,
    flatten_metrics,
    load_ml_ready_df,
    make_experiment_dir,
    save_json,
    split_xy,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOSTER_FAMILIES = {"catboost", "lightgbm"}


class TennisLogitsNet(nn.Module):
    def __init__(self, input_size: int, hidden_dims: List[int], dropouts: List[float]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size
        for hidden, drop in zip(hidden_dims, dropouts):
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(drop)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


def choose_scaler(kind: str):
    if kind == "robust":
        return RobustScaler()
    return StandardScaler()


def train_nn_config(split_payload: Dict, config: Dict, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = choose_scaler(config["scaler"])
    X_train = scaler.fit_transform(split_payload["train_X"].values)
    X_val = scaler.transform(split_payload["val_X"].values)
    X_test = scaler.transform(split_payload["test_X"].values)

    batch_size = config.get("batch_size", 1024)
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(split_payload["train_y"])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(split_payload["val_y"])),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(split_payload["test_y"])),
        batch_size=batch_size,
        shuffle=False,
    )

    model = TennisLogitsNet(
        input_size=X_train.shape[1],
        hidden_dims=config["hidden_dims"],
        dropouts=config["dropouts"],
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    best_state = None
    best_val_loss = float("inf")
    patience = config.get("patience", 8)
    patience_counter = 0
    history = []

    for epoch in range(config.get("max_epochs", 40)):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = model(batch_x)
                val_losses.append(criterion(logits, batch_y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(loader: DataLoader) -> np.ndarray:
        probs = []
        model.eval()
        with torch.no_grad():
            for batch_x, _ in loader:
                logits = model(batch_x.to(DEVICE))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
        return np.asarray(probs, dtype=float)

    val_prob = predict(val_loader)
    test_prob = predict(test_loader)
    val_metrics = compute_metrics(split_payload["val_y"], val_prob)
    test_metrics = compute_metrics(split_payload["test_y"], test_prob)

    torch.save(model.state_dict(), output_dir / "model.pth")
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    summary = {
        "family": "nn",
        "config": config,
        "split_label": split_payload["label"],
        "device": str(DEVICE),
        "best_val_loss": best_val_loss,
        "epochs_run": len(history),
        **flatten_metrics("val", val_metrics),
        **flatten_metrics("test", test_metrics),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def train_xgb_config(split_payload: Dict, config: Dict, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        early_stopping_rounds=config.get("early_stopping_rounds", 50),
        **config["params"],
    )
    model.fit(
        split_payload["train_X"],
        split_payload["train_y"],
        eval_set=[(split_payload["val_X"], split_payload["val_y"])],
        verbose=False,
    )
    val_prob = model.predict_proba(split_payload["val_X"])[:, 1]
    test_prob = model.predict_proba(split_payload["test_X"])[:, 1]
    val_metrics = compute_metrics(split_payload["val_y"], val_prob)
    test_metrics = compute_metrics(split_payload["test_y"], test_prob)

    model.save_model(output_dir / "model.json")
    summary = {
        "family": "xgboost",
        "config": config,
        "split_label": split_payload["label"],
        "best_iteration": getattr(model, "best_iteration", None),
        **flatten_metrics("val", val_metrics),
        **flatten_metrics("test", test_metrics),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _require_catboost():
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is not installed. Install requirements.txt or run "
            "`tennis_env/bin/pip install catboost`."
        ) from exc
    return CatBoostClassifier, Pool


def _require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is not installed. Install requirements.txt or run "
            "`tennis_env/bin/pip install lightgbm`."
        ) from exc
    return lgb


def _as_native_cat_slug(feature_mode: str, slug: str) -> str:
    return f"{slug}__{feature_mode}"


def _catboost_pool(Pool, X: pd.DataFrame, y: np.ndarray, cat_features: List[str]):
    if cat_features:
        X = X.copy()
        for col in cat_features:
            X[col] = X[col].fillna("Unknown").astype(str)
        return Pool(X, y, cat_features=cat_features)
    return Pool(X, y)


def train_catboost_config(split_payload: Dict, config: Dict, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    CatBoostClassifier, Pool = _require_catboost()
    cat_features = list(split_payload.get("categorical_features", []))
    model = CatBoostClassifier(
        random_seed=42,
        loss_function="Logloss",
        eval_metric="Logloss",
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=config.get("early_stopping_rounds", 75),
        **config["params"],
    )
    train_pool = _catboost_pool(Pool, split_payload["train_X"], split_payload["train_y"], cat_features)
    val_pool = _catboost_pool(Pool, split_payload["val_X"], split_payload["val_y"], cat_features)
    test_pool = _catboost_pool(Pool, split_payload["test_X"], split_payload["test_y"], cat_features)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
    val_prob = model.predict_proba(val_pool)[:, 1]
    test_prob = model.predict_proba(test_pool)[:, 1]
    val_metrics = compute_metrics(split_payload["val_y"], val_prob)
    test_metrics = compute_metrics(split_payload["test_y"], test_prob)

    model.save_model(str(output_dir / "model.cbm"))
    summary = {
        "family": "catboost",
        "config": config,
        "split_label": split_payload["label"],
        "feature_mode": split_payload.get("feature_mode", "one_hot"),
        "categorical_features": cat_features,
        "best_iteration": model.get_best_iteration(),
        **flatten_metrics("val", val_metrics),
        **flatten_metrics("test", test_metrics),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _prepare_lightgbm_frames(split_payload: Dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cat_features = list(split_payload.get("categorical_features", []))
    frames = [
        split_payload["train_X"].copy(),
        split_payload["val_X"].copy(),
        split_payload["test_X"].copy(),
    ]
    for col in cat_features:
        train_values = frames[0][col].fillna("Unknown").astype(str)
        categories = sorted(train_values.unique().tolist())
        if "Unknown" not in categories:
            categories.append("Unknown")
        if "__unseen__" not in categories:
            categories.append("__unseen__")
        dtype = CategoricalDtype(categories=categories)
        train_known = set(categories)
        for frame in frames:
            values = frame[col].fillna("Unknown").astype(str)
            values = values.where(values.isin(train_known), "__unseen__")
            frame[col] = values.astype(dtype)
    return frames[0], frames[1], frames[2]


def train_lightgbm_config(split_payload: Dict, config: Dict, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    lgb = _require_lightgbm()
    cat_features = list(split_payload.get("categorical_features", []))
    train_X, val_X, test_X = _prepare_lightgbm_frames(split_payload)
    model = lgb.LGBMClassifier(
        random_state=42,
        objective="binary",
        n_jobs=-1,
        verbosity=-1,
        **config["params"],
    )
    callbacks = [
        lgb.early_stopping(config.get("early_stopping_rounds", 75), verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        train_X,
        split_payload["train_y"],
        eval_set=[(val_X, split_payload["val_y"])],
        eval_metric="binary_logloss",
        categorical_feature=cat_features or "auto",
        callbacks=callbacks,
    )
    val_prob = model.predict_proba(val_X)[:, 1]
    test_prob = model.predict_proba(test_X)[:, 1]
    val_metrics = compute_metrics(split_payload["val_y"], val_prob)
    test_metrics = compute_metrics(split_payload["test_y"], test_prob)

    model.booster_.save_model(str(output_dir / "model.txt"))
    summary = {
        "family": "lightgbm",
        "config": config,
        "split_label": split_payload["label"],
        "feature_mode": split_payload.get("feature_mode", "one_hot"),
        "categorical_features": cat_features,
        "best_iteration": getattr(model, "best_iteration_", None),
        **flatten_metrics("val", val_metrics),
        **flatten_metrics("test", test_metrics),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def run_fixed_experiments(df: pd.DataFrame) -> pd.DataFrame:
    split = split_xy(build_fixed_split(df))

    nn_configs = [
        {
            "slug": "nn_logits_128_64_32_lowdrop",
            "hidden_dims": [128, 64, 32],
            "dropouts": [0.10, 0.10, 0.05],
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 1024,
            "max_epochs": 40,
            "patience": 8,
            "scaler": "standard",
        },
        {
            "slug": "nn_logits_96_48_lowdrop",
            "hidden_dims": [96, 48],
            "dropouts": [0.10, 0.05],
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 1024,
            "max_epochs": 40,
            "patience": 8,
            "scaler": "standard",
        },
        {
            "slug": "nn_logits_128_64_robust",
            "hidden_dims": [128, 64],
            "dropouts": [0.08, 0.05],
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "batch_size": 1024,
            "max_epochs": 45,
            "patience": 10,
            "scaler": "robust",
        },
    ]

    xgb_configs = [
        {
            "slug": "xgb_depth4_slow_regularized",
            "params": {
                "n_estimators": 1200,
                "max_depth": 4,
                "learning_rate": 0.02,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "min_child_weight": 3,
                "reg_lambda": 3.0,
                "reg_alpha": 0.5,
                "gamma": 0.5,
            },
        },
        {
            "slug": "xgb_depth5_balanced_regularized",
            "params": {
                "n_estimators": 1000,
                "max_depth": 5,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "min_child_weight": 3,
                "reg_lambda": 3.0,
                "reg_alpha": 0.5,
                "gamma": 0.0,
            },
        },
        {
            "slug": "xgb_depth6_medium",
            "params": {
                "n_estimators": 800,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.90,
                "min_child_weight": 1,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "gamma": 0.0,
            },
        },
    ]

    rows = []
    for config in nn_configs:
        out = make_experiment_dir("nn", config["slug"])
        rows.append(train_nn_config(split, config, out))
    for config in xgb_configs:
        out = make_experiment_dir("xgboost", config["slug"])
        rows.append(train_xgb_config(split, config, out))

    return pd.DataFrame(rows)


def get_booster_configs(family: str, blocked: bool = False) -> List[Dict]:
    if family == "catboost":
        return [
            {
                "slug": "cat_depth6_screening",
                "params": {
                    "iterations": 320 if blocked else 450,
                    "depth": 6,
                    "learning_rate": 0.07,
                    "l2_leaf_reg": 5.0,
                    "random_strength": 1.0,
                    "bootstrap_type": "Bernoulli",
                    "subsample": 0.85,
                    "rsm": 0.85,
                    "border_count": 128,
                },
                "early_stopping_rounds": 50,
            },
        ]
    if family == "lightgbm":
        return [
            {
                "slug": "lgbm_leaves31_regularized",
                "params": {
                    "n_estimators": 700 if blocked else 1000,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_child_samples": 60,
                    "subsample": 0.85,
                    "subsample_freq": 1,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 4.0,
                    "reg_alpha": 0.3,
                },
                "early_stopping_rounds": 50,
            },
        ]
    raise ValueError(f"Unknown booster family: {family}")


def train_booster_config(family: str, split_payload: Dict, config: Dict, output_dir: Path) -> Dict:
    if family == "catboost":
        return train_catboost_config(split_payload, config, output_dir)
    if family == "lightgbm":
        return train_lightgbm_config(split_payload, config, output_dir)
    raise ValueError(f"Unknown booster family: {family}")


def run_booster_fixed_experiments(
    df: pd.DataFrame,
    families: List[str],
    feature_modes: List[str],
) -> pd.DataFrame:
    rows = []
    fixed_split = build_fixed_split(df)
    for feature_mode in feature_modes:
        split = split_xy(fixed_split, feature_mode=feature_mode)
        for family in families:
            for config in get_booster_configs(family):
                slug = _as_native_cat_slug(feature_mode, config["slug"])
                out = make_experiment_dir(family, slug)
                rows.append(train_booster_config(family, split, config, out))
    return pd.DataFrame(rows)


def run_booster_blocked_eval(
    df: pd.DataFrame,
    families: List[str],
    feature_modes: List[str],
) -> pd.DataFrame:
    splits = build_blocked_windows(
        df,
        train_years=6,
        val_years=1,
        test_years=1,
        start_year=2010,
        end_test_year=2024,
        step_years=2,
    )

    rows = []
    for window in splits:
        for feature_mode in feature_modes:
            payload = split_xy(window, feature_mode=feature_mode)
            for family in families:
                config = get_booster_configs(family, blocked=True)[0]
                slug = _as_native_cat_slug(feature_mode, f"{config['slug']}__{window.label}")
                out = make_experiment_dir(family, slug)
                rows.append(train_booster_config(family, payload, config, out))
    return pd.DataFrame(rows)


def run_blocked_eval(df: pd.DataFrame) -> pd.DataFrame:
    splits = build_blocked_windows(
        df,
        train_years=6,
        val_years=1,
        test_years=1,
        start_year=2010,
        end_test_year=2024,
        step_years=2,
    )

    nn_config = {
        "slug": "nn_logits_128_64_32_lowdrop",
        "hidden_dims": [128, 64, 32],
        "dropouts": [0.10, 0.10, 0.05],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 1024,
        "max_epochs": 35,
        "patience": 7,
        "scaler": "standard",
    }
    xgb_config = {
        "slug": "xgb_depth5_balanced_regularized",
        "params": {
            "n_estimators": 1000,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 3,
            "reg_lambda": 3.0,
            "reg_alpha": 0.5,
            "gamma": 0.0,
        },
    }

    rows = []
    for split in splits:
        payload = split_xy(split)
        nn_out = make_experiment_dir("nn", f"{nn_config['slug']}__{split.label}")
        xgb_out = make_experiment_dir("xgboost", f"{xgb_config['slug']}__{split.label}")
        rows.append(train_nn_config(payload, nn_config, nn_out))
        rows.append(train_xgb_config(payload, xgb_config, xgb_out))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run organized side experiments for tennis models.")
    parser.add_argument(
        "--mode",
        choices=["fixed", "blocked", "all"],
        default="all",
        help="Experiment mode to run.",
    )
    parser.add_argument(
        "--include-boosters",
        action="store_true",
        help="Also run CatBoost and LightGBM side experiments.",
    )
    parser.add_argument(
        "--only-boosters",
        action="store_true",
        help="Run only CatBoost/LightGBM experiments for the selected mode.",
    )
    parser.add_argument(
        "--booster-families",
        default="catboost,lightgbm",
        help="Comma-separated booster families to run: catboost,lightgbm.",
    )
    parser.add_argument(
        "--booster-feature-mode",
        choices=["one_hot", "native_cat", "both"],
        default="both",
        help="Feature representation for CatBoost/LightGBM experiments.",
    )
    args = parser.parse_args()

    df = load_ml_ready_df()
    summaries: List[pd.DataFrame] = []
    if args.only_boosters:
        args.include_boosters = True

    if not args.only_boosters and args.mode in {"fixed", "all"}:
        summaries.append(run_fixed_experiments(df))
    if not args.only_boosters and args.mode in {"blocked", "all"}:
        summaries.append(run_blocked_eval(df))
    if args.include_boosters:
        booster_families = [family.strip() for family in args.booster_families.split(",") if family.strip()]
        unknown_families = sorted(set(booster_families) - BOOSTER_FAMILIES)
        if unknown_families:
            raise ValueError(f"Unknown booster families: {unknown_families}")
        feature_modes = (
            ["one_hot", "native_cat"]
            if args.booster_feature_mode == "both"
            else [args.booster_feature_mode]
        )
        if args.mode in {"fixed", "all"}:
            summaries.append(run_booster_fixed_experiments(df, booster_families, feature_modes))
        if args.mode in {"blocked", "all"}:
            summaries.append(run_booster_blocked_eval(df, booster_families, feature_modes))

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        batch_parts = [args.mode]
        if args.only_boosters:
            batch_parts.append("boosters_only")
        elif args.include_boosters:
            batch_parts.append("with_boosters")
        if args.include_boosters:
            batch_parts.append(args.booster_feature_mode)
            batch_parts.append("-".join(sorted(booster_families)))
        out_dir = make_experiment_dir("summaries", f"batch_{'_'.join(batch_parts)}")
        combined.to_csv(out_dir / "summary.csv", index=False)
        print("\n=== Experiment Summary ===")
        display_cols = [
            col for col in [
                "family", "feature_mode", "split_label", "config", "val_accuracy", "val_auc", "val_log_loss",
                "test_accuracy", "test_auc", "test_log_loss", "test_brier", "test_ece"
            ] if col in combined.columns
        ]
        print(combined[display_cols].to_string(index=False))
        print(f"\nSaved combined summary to: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
