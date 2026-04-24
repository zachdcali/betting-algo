from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "data" / "JeffSackmann" / "jeffsackmann_ml_ready_SURFACE_FIX.csv"
EXPERIMENTS_ROOT = REPO_ROOT / "results" / "professional_tennis" / "experiments"


EXACT_141_FEATURES: List[str] = [
    'P2_WinStreak_Current', 'P1_WinStreak_Current', 'P2_Surface_Matches_30d', 'Height_Diff',
    'P1_Surface_Matches_30d', 'Player2_Height', 'P1_Matches_30d', 'P2_Matches_30d',
    'P2_Surface_Experience', 'P2_Form_Trend_30d', 'Player1_Height', 'P1_Form_Trend_30d',
    'Round_R16', 'Surface_Transition_Flag', 'P1_Surface_Matches_90d', 'P1_Surface_Experience',
    'Rank_Diff', 'Round_R32', 'Rank_Points_Diff', 'P2_Level_WinRate_Career',
    'P2_Surface_Matches_90d', 'P1_Level_WinRate_Career', 'P2_Level_Matches_Career',
    'P2_WinRate_Last10_120d', 'Round_QF', 'Level_25', 'P1_Round_WinRate_Career',
    'P1_Surface_WinRate_90d', 'Round_Q1', 'Player1_Rank', 'P1_Level_Matches_Career',
    'P2_Round_WinRate_Career', 'draw_size', 'P1_WinRate_Last10_120d', 'Age_Diff',
    'Level_15', 'Player1_Rank_Points', 'Handedness_Matchup_RL', 'Player2_Rank',
    'Avg_Age', 'P1_Country_RUS', 'Player2_Age', 'P2_vs_Lefty_WinRate',
    'Round_F', 'Surface_Clay', 'P2_Sets_14d', 'Rank_Momentum_Diff_30d',
    'H2H_P2_Wins', 'Player2_Rank_Points', 'Player1_Age', 'P2_Rank_Volatility_90d',
    'P1_Days_Since_Last', 'Grass_Season', 'P1_Semifinals_WinRate', 'Level_A',
    'Level_D', 'P1_Country_USA', 'P1_Country_GBR', 'P1_Country_FRA',
    'P2_Matches_14d', 'P2_Country_USA', 'P2_Country_ITA', 'Round_Q2',
    'P2_Surface_WinRate_90d', 'P1_Hand_L', 'P2_Hand_L', 'P1_Country_ITA',
    'P2_Rust_Flag', 'P1_Rank_Change_90d', 'P1_Country_AUS', 'P1_Hand_U',
    'P1_Hand_R', 'Round_RR', 'Avg_Height', 'P1_Sets_14d',
    'P2_Country_Other', 'Round_SF', 'P1_vs_Lefty_WinRate', 'Indoor_Season',
    'Avg_Rank', 'P1_Rust_Flag', 'Avg_Rank_Points', 'Level_F',
    'Round_R64', 'P2_Country_CZE', 'P2_Hand_R', 'Surface_Hard',
    'P1_Matches_14d', 'Surface_Carpet', 'Round_R128', 'P1_Country_SRB',
    'P2_Hand_U', 'P1_Rank_Volatility_90d', 'Level_M', 'P2_Country_ESP',
    'Handedness_Matchup_LR', 'P1_Country_CZE', 'P2_Country_SUI', 'Surface_Grass',
    'H2H_Total_Matches', 'Level_O', 'P1_Hand_A', 'P1_Finals_WinRate',
    'Rank_Momentum_Diff_90d', 'P2_Finals_WinRate', 'Round_Q4', 'Level_G',
    'Round_ER', 'Level_S', 'Round_BR', 'Round_Q3', 'Rank_Ratio',
    'P1_Country_SUI', 'Clay_Season', 'P1_Country_GER', 'P2_Rank_Change_30d',
    'P1_Country_ESP', 'P2_Hand_A', 'H2H_Recent_P1_Advantage', 'P2_Country_AUS',
    'P2_Country_SRB', 'P2_Country_GBR', 'P2_Country_ARG', 'Handedness_Matchup_RR',
    'P1_Rank_Change_30d', 'P2_Country_GER', 'Handedness_Matchup_LL', 'P2_Country_RUS',
    'P1_Country_ARG', 'Level_C', 'P2_Semifinals_WinRate', 'P2_Days_Since_Last',
    'P1_Peak_Age', 'P2_Peak_Age', 'H2H_P1_WinRate', 'P1_Country_Other',
    'H2H_P1_Wins', 'P1_BigMatch_WinRate', 'P2_Rank_Change_90d', 'P2_BigMatch_WinRate',
    'P2_Country_FRA'
]


@dataclass
class DataSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label: str


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if mask.any():
            ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * (mask.sum() / len(y_true))
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": ece_score(y_true, y_prob),
    }


def load_ml_ready_df(min_year: int = 1990) -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    df["year"] = df["tourney_date"].dt.year
    df = df[df["year"] >= min_year].copy()
    df = df.dropna(subset=[col for col in ["Player1_Rank", "Player2_Rank"] if col in df.columns]).copy()
    return df


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EXACT_141_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    return out


def build_fixed_split(df: pd.DataFrame) -> DataSplit:
    return DataSplit(
        train_df=df[df["tourney_date"] < "2022-01-01"].copy(),
        val_df=df[(df["tourney_date"] >= "2022-01-01") & (df["tourney_date"] < "2023-01-01")].copy(),
        test_df=df[df["tourney_date"] >= "2023-01-01"].copy(),
        label="fixed_2022_val_2023plus_test",
    )


def build_blocked_windows(
    df: pd.DataFrame,
    train_years: int,
    val_years: int,
    test_years: int,
    start_year: int,
    end_test_year: int,
    step_years: Optional[int] = None,
) -> List[DataSplit]:
    splits: List[DataSplit] = []
    step = step_years or (train_years + val_years + test_years)
    train_start = start_year
    while True:
        train_end = train_start + train_years - 1
        val_start = train_end + 1
        val_end = val_start + val_years - 1
        test_start = val_end + 1
        test_end = test_start + test_years - 1
        if test_end > end_test_year:
            break

        train_mask = (df["year"] >= train_start) & (df["year"] <= train_end)
        val_mask = (df["year"] >= val_start) & (df["year"] <= val_end)
        test_mask = (df["year"] >= test_start) & (df["year"] <= test_end)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        if len(train_df) and len(val_df) and len(test_df):
            splits.append(
                DataSplit(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    label=f"train_{train_start}_{train_end}__val_{val_start}_{val_end}__test_{test_start}_{test_end}",
                )
            )
        train_start += step
    return splits


def split_xy(split: DataSplit) -> Dict[str, pd.DataFrame | np.ndarray]:
    out: Dict[str, pd.DataFrame | np.ndarray] = {"label": split.label}
    for prefix, frame in (("train", split.train_df), ("val", split.val_df), ("test", split.test_df)):
        prepared = prepare_feature_frame(frame)
        X = prepared[EXACT_141_FEATURES].astype(float)
        medians = None
        if prefix == "train":
            medians = X.median()
            out["_train_medians"] = medians
        else:
            medians = out["_train_medians"]
        X = X.fillna(medians)
        y = prepared["Player1_Wins"].astype(int).values
        out[f"{prefix}_X"] = X
        out[f"{prefix}_y"] = y
        out[f"{prefix}_n"] = len(prepared)
    return out


def make_experiment_dir(family: str, experiment_slug: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = EXPERIMENTS_ROOT / stamp / family / experiment_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}

