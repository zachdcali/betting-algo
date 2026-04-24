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
    args = parser.parse_args()

    df = load_ml_ready_df()
    summaries: List[pd.DataFrame] = []
    if args.mode in {"fixed", "all"}:
        summaries.append(run_fixed_experiments(df))
    if args.mode in {"blocked", "all"}:
        summaries.append(run_blocked_eval(df))

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        out_dir = make_experiment_dir("summaries", f"batch_{args.mode}")
        combined.to_csv(out_dir / "summary.csv", index=False)
        print("\n=== Experiment Summary ===")
        display_cols = [
            col for col in [
                "family", "split_label", "config", "val_accuracy", "val_auc", "val_log_loss",
                "test_accuracy", "test_auc", "test_log_loss", "test_brier", "test_ece"
            ] if col in combined.columns
        ]
        print(combined[display_cols].to_string(index=False))
        print(f"\nSaved combined summary to: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
