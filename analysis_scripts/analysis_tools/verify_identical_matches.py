#!/usr/bin/env python3
"""
verify_identical_matches.py — STRICT score + rank verification, then align to ML (leak-free)

Pipeline:
  1) Load row-aligned matched files:
       data/JeffSackmann/jeffsackmann_exact_matched_final.csv
       data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv
  2) Parse scores and keep ONLY rows where:
       - score sets match exactly, AND
       - winner/loser ranks match exactly (numeric equality)
  3) Keep only rows with usable odds (AvgW/AvgL > 1.01)
  4) Align each kept row to ML-ready leak-free features by (tourney_date + Jeff winner/loser names)
     -> set Player1_Wins = 1 if (P1,P2) == (winner,loser), else 0 if reversed
  5) Enforce numeric-feature completeness on ML, keep odds aligned
  6) Save:
       analysis_scripts/verified/ml_verified.csv          (all ML features + Player1_Wins)
       analysis_scripts/verified/tennis_verified.csv      (Date,Winner,Loser,AvgW,AvgL)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

# -------------------------
# Paths (repo root inferred)
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

JEFF_MATCHED_PATH   = REPO_ROOT / "data" / "JeffSackmann" / "jeffsackmann_exact_matched_final.csv"
TENNIS_MATCHED_PATH = REPO_ROOT / "data" / "Tennis-Data.co.uk" / "tennis_data_exact_matched_final.csv"
ML_READY_PATH       = REPO_ROOT / "data" / "JeffSackmann" / "jeffsackmann_ml_ready_LEAK_FREE.csv"

OUT_DIR = REPO_ROOT / "analysis_scripts" / "verified"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ML     = OUT_DIR / "ml_verified.csv"
OUT_TENNIS = OUT_DIR / "tennis_verified.csv"

# Optional year filter like (2023, 2024); set to None to disable
YEAR_RANGE = None  # e.g., (2023, 2024)

# -------------------------
# Helpers
# -------------------------
def norm(s: str) -> str:
    return str(s).strip()

def parse_jeff_score(score_str):
    """Parse Jeff Sackmann score string -> list of (w,l) set tuples or None."""
    if pd.isna(score_str):
        return None
    s = str(score_str).strip()
    if any(x in s.upper() for x in ['RET', 'W/O', 'DEF', 'WALKOVER']):  # abandonments etc.
        return None
    sets = re.findall(r'(\d+)-(\d+)', s)
    return [tuple(x) for x in sets] if len(sets) >= 2 else None

def parse_tennis_data_score(row):
    """Parse Tennis-Data W1/L1...W5/L5 -> list of (w,l) set tuples or None."""
    sets = []
    for i in range(1, 6):
        w_col, l_col = f"W{i}", f"L{i}"
        if pd.notna(row.get(w_col)) and pd.notna(row.get(l_col)):
            try:
                w = int(float(row[w_col]))
                l = int(float(row[l_col]))
                sets.append((str(w), str(l)))
            except (ValueError, TypeError):
                break
        else:
            break
    return sets if len(sets) >= 2 else None

# -------------------------
# Main
# -------------------------
def main():
    print("\n================================================================================")
    print("VERIFY IDENTICAL MATCHES — strict scores + ranks, then align to leak-free ML")
    print("================================================================================")

    # 1) Load files
    print("1) Loading matched & ML files...")
    for p in [JEFF_MATCHED_PATH, TENNIS_MATCHED_PATH, ML_READY_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    jeff   = pd.read_csv(JEFF_MATCHED_PATH, low_memory=False)
    tennis = pd.read_csv(TENNIS_MATCHED_PATH, low_memory=False)
    ml     = pd.read_csv(ML_READY_PATH, low_memory=False)

    need_jeff = {"tourney_date","winner_name","loser_name","winner_rank","loser_rank","score"}
    need_ten  = {"Date","Winner","Loser","WRank","LRank","AvgW","AvgL","W1","L1","W2","L2"}
    need_ml   = {"tourney_date","Player1_Name","Player2_Name"}
    miss = need_jeff - set(jeff.columns);  assert not miss, f"Jeff missing: {miss}"
    miss = need_ten  - set(tennis.columns);assert not miss, f"Tennis missing: {miss}"
    miss = need_ml   - set(ml.columns);    assert not miss, f"ML missing: {miss}"

    # Normalize + types
    jeff["tourney_date"] = pd.to_datetime(jeff["tourney_date"], errors="coerce")
    tennis["Date"]       = pd.to_datetime(tennis["Date"], errors="coerce")
    ml["tourney_date"]   = pd.to_datetime(ml["tourney_date"], errors="coerce")

    for c in ["winner_name","loser_name"]:
        jeff[c] = jeff[c].astype(str).map(norm)
    for c in ["Winner","Loser"]:
        tennis[c] = tennis[c].astype(str).map(norm)
    for c in ["Player1_Name","Player2_Name"]:
        ml[c] = ml[c].astype(str).map(norm)

    if YEAR_RANGE:
        y0, y1 = YEAR_RANGE
        jeff   = jeff[(jeff["tourney_date"].dt.year>=y0)&(jeff["tourney_date"].dt.year<=y1)].reset_index(drop=True)
        tennis = tennis[(tennis["Date"].dt.year>=y0)&(tennis["Date"].dt.year<=y1)].reset_index(drop=True)
        ml     = ml[(ml["tourney_date"].dt.year>=y0)&(ml["tourney_date"].dt.year<=y1)].reset_index(drop=True)

    if len(jeff) != len(tennis):
        print(f"⚠️  Warning: matched files lengths differ (Jeff={len(jeff):,}, Tennis={len(tennis):,}). "
              "Assuming same sort order; strict filter will use shared indices only.")

    # 2) Parse scores and build strict mask (score + both ranks equal)
    print("2) Parsing scores & enforcing strict score+rank equality...")
    jeff["parsed_score"]   = jeff["score"].apply(parse_jeff_score)
    tennis["parsed_score"] = tennis.apply(parse_tennis_data_score, axis=1)

    # Rank columns to numeric
    for col in ["winner_rank","loser_rank"]:
        jeff[col] = pd.to_numeric(jeff[col], errors="coerce")
    for col in ["WRank","LRank","AvgW","AvgL"]:
        tennis[col] = pd.to_numeric(tennis[col], errors="coerce")

    # Build strict mask on the intersection of indices
    shared_idx = sorted(set(jeff.index).intersection(set(tennis.index)))
    strict_mask = []
    for i in shared_idx:
        jrow = jeff.loc[i]
        trow = tennis.loc[i]
        # scores equal?
        score_ok = (jrow["parsed_score"] is not None and
                    trow["parsed_score"] is not None and
                    jrow["parsed_score"] == trow["parsed_score"])
        # ranks equal?
        try:
            rank_ok = (not pd.isna(jrow["winner_rank"]) and not pd.isna(jrow["loser_rank"]) and
                       not pd.isna(trow["WRank"]) and not pd.isna(trow["LRank"]) and
                       float(jrow["winner_rank"]) == float(trow["WRank"]) and
                       float(jrow["loser_rank"])  == float(trow["LRank"]))
        except Exception:
            rank_ok = False
        strict_mask.append(score_ok and rank_ok)

    kept_idx = [idx for idx, ok in zip(shared_idx, strict_mask) if ok]
    print(f"   ✅ Strict score+rank matches: {len(kept_idx):,}")

    if len(kept_idx) == 0:
        raise RuntimeError("No strict matches found. Check inputs/years/row alignment.")

    jeff_s   = jeff.loc[kept_idx].reset_index(drop=True)
    tennis_s = tennis.loc[kept_idx].reset_index(drop=True)

    # 3) Filter to usable odds
    print("3) Filtering to usable odds (AvgW/AvgL > 1.01)...")
    odds_mask = (
        tennis_s["AvgW"].notna() & tennis_s["AvgL"].notna() &
        (tennis_s["AvgW"] > 1.01) & (tennis_s["AvgL"] > 1.01)
    )
    jeff_s   = jeff_s.loc[odds_mask].reset_index(drop=True)
    tennis_s = tennis_s.loc[odds_mask].reset_index(drop=True)
    print(f"   ✅ After odds filter: {len(jeff_s):,}")

    # 4) Align to ML by (tourney_date + Jeff winner/loser names)
    print("4) Aligning to ML leak-free by (date + Jeff winner/loser names)...")
    ml_by_date = {d: g for d, g in ml.groupby(ml["tourney_date"])}

    ml_rows = []
    p1wins  = []
    misses  = 0

    for i in range(len(jeff_s)):
        d   = jeff_s.at[i, "tourney_date"]
        win = jeff_s.at[i, "winner_name"]
        los = jeff_s.at[i, "loser_name"]

        g = ml_by_date.get(d)
        if g is None:
            misses += 1
            continue

        hits = g[(g["Player1_Name"]==win) & (g["Player2_Name"]==los)]
        if len(hits) == 0:
            hits = g[(g["Player1_Name"]==los) & (g["Player2_Name"]==win)]
            if len(hits) == 0:
                misses += 1
                continue
            ml_rows.append(hits.iloc[0].copy())
            p1wins.append(0)
        else:
            ml_rows.append(hits.iloc[0].copy())
            p1wins.append(1)

    if len(ml_rows) == 0:
        raise RuntimeError("Could not align any strict-matched rows to ML file (name/date mismatch?).")

    if misses:
        print(f"   ⚠️  ML aligns missed on {misses:,} of {len(jeff_s):,} rows (dropping those).")

    ml_aligned     = pd.DataFrame(ml_rows).reset_index(drop=True)
    tennis_aligned = tennis_s.iloc[:len(ml_aligned)].copy().reset_index(drop=True)  # one tennis row per ML success
    ml_aligned["Player1_Wins"] = p1wins

    # 5) Enforce numeric-feature completeness on ML, keep tennis aligned
    print("5) Enforcing numeric-feature completeness on ML...")
    id_like = {"tourney_date","Player1_Name","Player2_Name","Player1_Wins"}
    feat_cols = [c for c in ml_aligned.columns if c not in id_like]
    numeric_cols = [c for c in feat_cols if ml_aligned[c].dtype.name in ["int64","float64","bool"]]

    valid = ~ml_aligned[numeric_cols].isnull().any(axis=1)
    if valid.sum() == 0:
        raise RuntimeError("After numeric completeness filter, zero ML rows remain.")
    if valid.sum() != len(ml_aligned):
        print(f"   ⚠️  Dropping {len(ml_aligned)-valid.sum():,} rows with missing numeric features.")

    ml_final     = ml_aligned.loc[valid].reset_index(drop=True)
    tennis_final = tennis_aligned.loc[valid].reset_index(drop=True)

    print(f"   ✅ Final aligned rows: {len(ml_final):,}")

    # 6) Save
    print("6) Saving outputs...")
    ml_final.to_csv(OUT_ML, index=False)
    tennis_final[["Date","Winner","Loser","AvgW","AvgL"]].to_csv(OUT_TENNIS, index=False)

    print(f"   📄 ML verified:     {OUT_ML}")
    print(f"   📄 Tennis verified: {OUT_TENNIS}")
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
