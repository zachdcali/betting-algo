from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from performance_features import PERFORMANCE_FEATURES, PERFORMANCE_FEATURE_SET
from performance_features import add_performance_features
from preprocess import get_round_day_offset


REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURE_SET_ROOT = REPO_ROOT / "results" / "professional_tennis" / "feature_sets"
JEFF_ROOT = REPO_ROOT / "data" / "JeffSackmann"
MASTER_PATH = JEFF_ROOT / "jeffsackmann_master_combined.csv"
CANONICAL_141_PATH = JEFF_ROOT / "jeffsackmann_ml_ready_SURFACE_FIX.csv"


def build_performance_v1(output_dir: Path | None = None) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d")
    output_dir = output_dir or FEATURE_SET_ROOT / PERFORMANCE_FEATURE_SET / stamp
    if output_dir.exists() and any(output_dir.iterdir()):
        output_dir = FEATURE_SET_ROOT / PERFORMANCE_FEATURE_SET / f"{stamp}__run_{datetime.now().strftime('%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "jeffsackmann_ml_ready_performance_v1.csv"
    ml_df = build_performance_v1_frame()
    ml_df.to_csv(dataset_path, index=False)

    manifest = {
        "feature_set": PERFORMANCE_FEATURE_SET,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "rows": int(len(ml_df)),
        "feature_count": int(141 + len(PERFORMANCE_FEATURES)),
        "performance_feature_count": len(PERFORMANCE_FEATURES),
        "performance_features": PERFORMANCE_FEATURES,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved {PERFORMANCE_FEATURE_SET} dataset to: {dataset_path}")
    print(f"Saved manifest to: {output_dir / 'manifest.json'}")
    return dataset_path


def build_performance_v1_frame() -> pd.DataFrame:
    """
    Append performance_v1 columns to the existing canonical 141-feature dataset.

    This avoids recomputing the expensive legacy temporal features. The raw
    orientation/date steps mirror preprocess.py, then row alignment is checked
    against the canonical ML-ready CSV before concatenating columns.
    """
    print("Loading canonical 141-feature dataset...")
    canonical = pd.read_csv(CANONICAL_141_PATH, low_memory=False)

    print("Building performance_v1 columns from raw Sackmann aggregate...")
    raw = _prepare_oriented_raw_frame()
    perf = add_performance_features(raw)

    _assert_alignment(canonical, perf)
    return pd.concat([canonical.reset_index(drop=True), perf[PERFORMANCE_FEATURES].reset_index(drop=True)], axis=1)


def _prepare_oriented_raw_frame() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PATH, low_memory=False)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")

    np.random.seed(42)
    swap_mask = np.random.random(len(df)) < 0.5

    df["Player1_ID"] = df["winner_id"].copy()
    df["Player1_Name"] = df["winner_name"].copy()
    df["Player2_ID"] = df["loser_id"].copy()
    df["Player2_Name"] = df["loser_name"].copy()
    df["Player1_Wins"] = 1

    df.loc[swap_mask, "Player1_ID"] = df.loc[swap_mask, "loser_id"]
    df.loc[swap_mask, "Player1_Name"] = df.loc[swap_mask, "loser_name"]
    df.loc[swap_mask, "Player2_ID"] = df.loc[swap_mask, "winner_id"]
    df.loc[swap_mask, "Player2_Name"] = df.loc[swap_mask, "winner_name"]
    df.loc[swap_mask, "Player1_Wins"] = 0

    qual_rounds_per_tourney = df.groupby("tourney_id")["round"].apply(
        lambda rounds: sum(
            1
            for r in rounds.unique()
            if str(r).upper().startswith("Q") and str(r).upper() not in ("QF",)
        )
    ).to_dict()
    df["_num_qual_rounds"] = df["tourney_id"].map(qual_rounds_per_tourney).fillna(0).astype(int)
    df["inferred_match_date"] = df.apply(
        lambda row: row["tourney_date"]
        + pd.Timedelta(
            days=get_round_day_offset(
                row.get("tourney_level"),
                row.get("draw_size"),
                row.get("round"),
                tourney_date=row["tourney_date"],
                num_qual_rounds=row.get("_num_qual_rounds") or None,
            )
        ),
        axis=1,
    )
    df.drop(columns=["_num_qual_rounds"], inplace=True)

    needed = [
        "tourney_date",
        "inferred_match_date",
        "tourney_name",
        "match_num",
        "surface",
        "round",
        "best_of",
        "score",
        "minutes",
        "Player1_ID",
        "Player1_Name",
        "Player2_ID",
        "Player2_Name",
        "Player1_Wins",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_bpSaved",
        "l_bpFaced",
    ]
    return df[needed].copy()


def _assert_alignment(canonical: pd.DataFrame, perf: pd.DataFrame) -> None:
    if len(canonical) != len(perf):
        raise RuntimeError(f"Row count mismatch: canonical={len(canonical)} performance={len(perf)}")

    canonical_keys = canonical[["tourney_date", "tourney_name", "round", "Player1_Name", "Player2_Name", "Player1_Wins"]].copy()
    perf_keys = perf[["tourney_date", "tourney_name", "round", "Player1_Name", "Player2_Name", "Player1_Wins"]].copy()
    canonical_keys["tourney_date"] = pd.to_datetime(canonical_keys["tourney_date"]).dt.strftime("%Y-%m-%d")
    perf_keys["tourney_date"] = pd.to_datetime(perf_keys["tourney_date"]).dt.strftime("%Y-%m-%d")

    left = canonical_keys.fillna("__NA__").astype(str).reset_index(drop=True)
    right = perf_keys.fillna("__NA__").astype(str).reset_index(drop=True)
    mismatch = (left != right).any(axis=1)
    if mismatch.any():
        first = int(mismatch.idxmax())
        raise RuntimeError(
            "Performance feature rows do not align with canonical dataset at "
            f"row {first}: canonical={left.iloc[first].to_dict()} performance={right.iloc[first].to_dict()}"
        )
    print("Row alignment check passed against canonical 141-feature dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build versioned side feature-set datasets.")
    parser.add_argument(
        "--feature-set",
        choices=[PERFORMANCE_FEATURE_SET],
        default=PERFORMANCE_FEATURE_SET,
        help="Feature set to build.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/professional_tennis/feature_sets/<feature-set>/<date>.",
    )
    args = parser.parse_args()

    if args.feature_set == PERFORMANCE_FEATURE_SET:
        build_performance_v1(args.output_dir)


if __name__ == "__main__":
    main()
