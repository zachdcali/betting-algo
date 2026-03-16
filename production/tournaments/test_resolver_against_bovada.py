import pandas as pd
import sys
from pathlib import Path
from resolve_tournament import TournamentResolver
from fallback_heuristics import get_fallback_tournament_meta

ROOT = Path(__file__).resolve().parents[2]
BOVADA = ROOT / "production" / "odds" / "data" / "bovada_tennis_latest.csv"
MAP = ROOT / "data" / "tournaments_map.csv"

def main():
    assert BOVADA.exists(), f"Missing {BOVADA}"
    assert MAP.exists(), f"Missing {MAP}"

    df = pd.read_csv(BOVADA)
    events = sorted(set(df["event"].dropna()))
    # fetch_bovada.py already filters for men's events, so all events are men's
    mens = events

    print("BOVADA MEN'S EVENTS:")
    for e in mens: 
        print(f"  - {e}")
    
    print(f"\nTotal events: {len(events)}")
    print(f"Men's events: {len(mens)} (all events are pre-filtered)")

    r = TournamentResolver(str(MAP))

    rows = []
    learned_aliases = []
    print(f"\nTESTING FUZZY RESOLVER WITH AUTO-LEARNING:")
    print("=" * 80)
    
    for ev in mens:
        res = r.resolve_soft(ev)
        if res is None:
            # Use fallback heuristics for unmatched events
            meta = get_fallback_tournament_meta(ev)
            rows.append({
                "event": ev, 
                "matched": False, 
                "match_name": "", 
                "score": 0.0, 
                "surface": meta.surface or "", 
                "level": meta.level, 
                "draw": meta.draw_size,
                "learned": False,
                "status": "AMBIGUOUS_OR_UNMATCHED",
                "confidence": meta.confidence
            })
            print(f"❌ {ev:<40} → FALLBACK (level:{meta.level}, draw:{meta.draw_size}, surface:{meta.surface or 'unknown'})")
        else:
            meta, score = res
            from resolve_tournament import level_hint_from_title
            idx, _ = r.best_match(ev, level_hint_from_title(ev))
            match_name = r.df.iloc[idx]["canonical_name"]
            
            # Auto-learn aliases for fuzzy matches (score < 1.0)
            learned = False
            if score < 1.0:
                if r.learn_alias(ev, idx, score, save_to_csv=True):
                    learned = True
                    learned_aliases.append({
                        "event": ev,
                        "canonical": match_name,
                        "score": score
                    })
                    print(f"🎓 LEARNED_ALIAS: {ev} → {match_name}")
                    
                    # Re-resolve to confirm it's now exact (score=1.0)
                    re_res = r.resolve_soft(ev)
                    if re_res and re_res[1] == 1.0:
                        print(f"   ✓ Confirmed: now resolves with score=1.0")
                        score = 1.0  # update for display
            
            # Determine resolution status
            status = "RESOLVED" if score >= 0.6 else "LOW_CONFIDENCE"
            confidence = "exact" if score == 1.0 else "fuzzy"
            
            rows.append({
                "event": ev, 
                "matched": True, 
                "match_name": match_name, 
                "score": round(score, 3),
                "surface": meta.surface, 
                "level": meta.level, 
                "draw": meta.draw_size,
                "learned": learned,
                "status": status,
                "confidence": confidence
            })
            
            status = "🎓" if learned else "✅"
            print(f"{status} {ev:<40} → {match_name:<30} (score:{score:.3f}, {meta.surface}, {meta.level}, Draw:{meta.draw_size})")

    out = pd.DataFrame(rows).sort_values(["matched", "score"], ascending=[False, False])
    
    # Summary table
    print(f"\nSUMMARY TABLE:")
    print("=" * 80)
    print(out[["event", "matched", "match_name", "score", "surface", "level", "draw", "status", "confidence", "learned"]].to_string(index=False))

    # Stats
    matched_count = len(out[out["matched"]])
    total_count = len(out)
    learned_count = len(learned_aliases)
    resolved_count = len(out[out["status"] == "RESOLVED"])
    fallback_count = len(out[out["status"] == "AMBIGUOUS_OR_UNMATCHED"])
    low_conf_count = len(out[out["status"] == "LOW_CONFIDENCE"])
    
    print(f"\nSTATS:")
    print(f"Total events: {total_count}")
    print(f"Matched: {matched_count}/{total_count} ({matched_count/total_count*100:.1f}%)")
    print(f"  - RESOLVED (score ≥0.6): {resolved_count} ({resolved_count/total_count*100:.1f}%)")
    print(f"  - LOW_CONFIDENCE: {low_conf_count} ({low_conf_count/total_count*100:.1f}%)")
    print(f"  - FALLBACK: {fallback_count} ({fallback_count/total_count*100:.1f}%)")
    print(f"Auto-learned aliases: {learned_count}")
    print(f"\n📊 RECOMMENDATION: Only ingest RESOLVED events for model training")
    
    if learned_aliases:
        print(f"\nLEARNED ALIASES:")
        for alias in learned_aliases:
            print(f"  {alias['event']} → {alias['canonical']} (score: {alias['score']:.3f})")

    # Save artifacts
    OUT_OK = ROOT / "data" / "resolver_matches.csv"
    OUT_BAD = ROOT / "data" / "resolver_unmatched.csv"
    
    matched_df = out[out["matched"]]
    unmatched_df = out[~out["matched"]]
    
    if not matched_df.empty:
        matched_df.to_csv(OUT_OK, index=False)
        print(f"✅ Matches saved → {OUT_OK}")
    
    if not unmatched_df.empty:
        unmatched_df.to_csv(OUT_BAD, index=False)
        print(f"⚠️  Unmatched saved → {OUT_BAD}")
    else:
        print(f"🎉 All events matched successfully!")
        
    # Show conflicts for matched tournaments (optional)
    conflicts_path = ROOT / "data" / "tournaments_conflicts.csv"
    if conflicts_path.exists() and matched_count > 0:
        print(f"\n📊 CONFLICT CHECK:")
        print("=" * 50)
        conflicts_df = pd.read_csv(conflicts_path)
        matched_names = set(out[out["matched"]]["match_name"])
        relevant_conflicts = conflicts_df[conflicts_df["canonical_name"].isin(matched_names)]
        
        if not relevant_conflicts.empty:
            print("⚠️  Today's tournaments with historical surface/level/draw conflicts:")
            for _, row in relevant_conflicts.iterrows():
                print(f"  {row['canonical_name']}:")
                print(f"    Surface: {row['surface_hist']}")
                print(f"    Level: {row['level_hist']}")
                print(f"    Draw: {row['draw_hist']}")
        else:
            print("✅ No conflicts in today's tournament slate")

if __name__ == "__main__":
    main()