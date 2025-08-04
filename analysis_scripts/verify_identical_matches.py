#!/usr/bin/env python3
"""
Verify that matched games are truly identical by checking scores and rankings
"""
import pandas as pd
import re

def parse_jeff_score(score_str):
    """Parse Jeff Sackmann score format"""
    if pd.isna(score_str):
        return None
    
    score_str = str(score_str).strip()
    # Remove common patterns
    if any(x in score_str.upper() for x in ['RET', 'W/O', 'DEF', 'WALKOVER']):
        return None
    
    # Extract set scores using regex
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if len(sets) >= 2:  # At least 2 sets
        return sets
    return None

def parse_tennis_data_score(row):
    """Parse Tennis-Data score from W1,L1,W2,L2 columns"""
    sets = []
    for i in range(1, 6):  # Up to 5 sets
        w_col = f'W{i}'
        l_col = f'L{i}'
        if pd.notna(row.get(w_col)) and pd.notna(row.get(l_col)):
            try:
                w_score = int(float(row[w_col]))
                l_score = int(float(row[l_col]))
                sets.append((str(w_score), str(l_score)))
            except (ValueError, TypeError):
                continue
    
    if len(sets) >= 2:  # At least 2 sets
        return sets
    return None

def scores_match(jeff_sets, tennis_sets):
    """Check if two score sets match"""
    if jeff_sets is None or tennis_sets is None:
        return False
    
    if len(jeff_sets) != len(tennis_sets):
        return False
    
    for i in range(len(jeff_sets)):
        if jeff_sets[i] != tennis_sets[i]:
            return False
    
    return True

print("VERIFYING IDENTICAL MATCHES WITH SCORES AND RANKINGS")
print("="*60)

# Load the matched datasets
jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv')
tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv')

print(f"Loaded datasets: {len(jeff_matched)} matches each")

# Parse scores for both datasets
print("\n1. Parsing scores...")
jeff_matched['parsed_score'] = jeff_matched['score'].apply(parse_jeff_score)
tennis_matched['parsed_score'] = tennis_matched.apply(parse_tennis_data_score, axis=1)

# Check score matches
print("\n2. Checking score matches...")
score_matches = 0
score_mismatches = 0
score_unparseable = 0
sample_matches = []
sample_mismatches = []

for i in range(len(jeff_matched)):
    jeff_row = jeff_matched.iloc[i]
    tennis_row = tennis_matched.iloc[i]
    
    jeff_score = jeff_row['parsed_score']
    tennis_score = tennis_row['parsed_score']
    
    if jeff_score is None or tennis_score is None:
        score_unparseable += 1
    elif scores_match(jeff_score, tennis_score):
        score_matches += 1
        if len(sample_matches) < 5:
            sample_matches.append((jeff_row, tennis_row, jeff_score, tennis_score))
    else:
        score_mismatches += 1
        if len(sample_mismatches) < 5:
            sample_mismatches.append((jeff_row, tennis_row, jeff_score, tennis_score))

print(f"   Score matches: {score_matches:,}")
print(f"   Score mismatches: {score_mismatches:,}")
print(f"   Unparseable scores: {score_unparseable:,}")

if sample_matches:
    print(f"\n   Sample score matches:")
    for jeff_row, tennis_row, jeff_score, tennis_score in sample_matches:
        print(f"     {jeff_row['winner_name']} def. {jeff_row['loser_name']}")
        print(f"       Jeff: {jeff_row['score']} -> {jeff_score}")
        print(f"       Tennis: W1={tennis_row.get('W1')},L1={tennis_row.get('L1')},W2={tennis_row.get('W2')},L2={tennis_row.get('L2')} -> {tennis_score}")
        print()

if sample_mismatches:
    print(f"\n   Sample score mismatches:")
    for jeff_row, tennis_row, jeff_score, tennis_score in sample_mismatches:
        print(f"     {jeff_row['winner_name']} def. {jeff_row['loser_name']}")
        print(f"       Jeff: {jeff_row['score']} -> {jeff_score}")
        print(f"       Tennis: W1={tennis_row.get('W1')},L1={tennis_row.get('L1')},W2={tennis_row.get('W2')},L2={tennis_row.get('L2')} -> {tennis_score}")
        print()

# Check ATP rankings
print("\n3. Checking ATP ranking matches...")
ranking_matches = 0
ranking_mismatches = 0
ranking_missing = 0
sample_rank_matches = []
sample_rank_mismatches = []

for i in range(len(jeff_matched)):
    jeff_row = jeff_matched.iloc[i]
    tennis_row = tennis_matched.iloc[i]
    
    # Get rankings
    jeff_winner_rank = jeff_row.get('winner_rank')
    jeff_loser_rank = jeff_row.get('loser_rank')
    tennis_winner_rank = tennis_row.get('WRank')
    tennis_loser_rank = tennis_row.get('LRank')
    
    # Convert to numeric
    try:
        jeff_winner_rank = int(float(jeff_winner_rank)) if pd.notna(jeff_winner_rank) else None
        jeff_loser_rank = int(float(jeff_loser_rank)) if pd.notna(jeff_loser_rank) else None
        tennis_winner_rank = int(float(tennis_winner_rank)) if pd.notna(tennis_winner_rank) else None
        tennis_loser_rank = int(float(tennis_loser_rank)) if pd.notna(tennis_loser_rank) else None
    except (ValueError, TypeError):
        ranking_missing += 1
        continue
    
    if (jeff_winner_rank is None or jeff_loser_rank is None or 
        tennis_winner_rank is None or tennis_loser_rank is None):
        ranking_missing += 1
        continue
    
    if jeff_winner_rank == tennis_winner_rank and jeff_loser_rank == tennis_loser_rank:
        ranking_matches += 1
        if len(sample_rank_matches) < 5:
            sample_rank_matches.append((jeff_row, tennis_row))
    else:
        ranking_mismatches += 1
        if len(sample_rank_mismatches) < 5:
            sample_rank_mismatches.append((jeff_row, tennis_row))

print(f"   Ranking matches: {ranking_matches:,}")
print(f"   Ranking mismatches: {ranking_mismatches:,}")
print(f"   Missing rankings: {ranking_missing:,}")

if sample_rank_matches:
    print(f"\n   Sample ranking matches:")
    for jeff_row, tennis_row in sample_rank_matches:
        print(f"     {jeff_row['winner_name']} (#{jeff_row['winner_rank']}) def. {jeff_row['loser_name']} (#{jeff_row['loser_rank']})")
        print(f"       vs {tennis_row['Winner']} (#{tennis_row['WRank']}) def. {tennis_row['Loser']} (#{tennis_row['LRank']})")
        print()

if sample_rank_mismatches:
    print(f"\n   Sample ranking mismatches:")
    for jeff_row, tennis_row in sample_rank_mismatches:
        print(f"     {jeff_row['winner_name']} (#{jeff_row['winner_rank']}) def. {jeff_row['loser_name']} (#{jeff_row['loser_rank']})")
        print(f"       vs {tennis_row['Winner']} (#{tennis_row['WRank']}) def. {tennis_row['Loser']} (#{tennis_row['LRank']})")
        print()

# Overall verification
print(f"\n" + "="*60)
print(f"VERIFICATION SUMMARY")
print(f"="*60)
print(f"Total matches checked: {len(jeff_matched):,}")
print(f"")
print(f"Score verification:")
print(f"  Exact score matches: {score_matches:,} ({score_matches/len(jeff_matched)*100:.1f}%)")
print(f"  Score mismatches: {score_mismatches:,} ({score_mismatches/len(jeff_matched)*100:.1f}%)")
print(f"  Unparseable scores: {score_unparseable:,} ({score_unparseable/len(jeff_matched)*100:.1f}%)")
print(f"")
print(f"Ranking verification:")
print(f"  Exact ranking matches: {ranking_matches:,} ({ranking_matches/len(jeff_matched)*100:.1f}%)")
print(f"  Ranking mismatches: {ranking_mismatches:,} ({ranking_mismatches/len(jeff_matched)*100:.1f}%)")
print(f"  Missing rankings: {ranking_missing:,} ({ranking_missing/len(jeff_matched)*100:.1f}%)")

# Calculate high-confidence matches
high_confidence = 0
for i in range(len(jeff_matched)):
    jeff_score = jeff_matched.iloc[i]['parsed_score']
    tennis_score = tennis_matched.iloc[i]['parsed_score']
    
    # Check if this match has both score and ranking verification
    has_score_match = jeff_score is not None and tennis_score is not None and scores_match(jeff_score, tennis_score)
    
    # Check rankings for this specific match
    jeff_row = jeff_matched.iloc[i]
    tennis_row = tennis_matched.iloc[i]
    try:
        jeff_w_rank = int(float(jeff_row['winner_rank'])) if pd.notna(jeff_row['winner_rank']) else None
        jeff_l_rank = int(float(jeff_row['loser_rank'])) if pd.notna(jeff_row['loser_rank']) else None
        tennis_w_rank = int(float(tennis_row['WRank'])) if pd.notna(tennis_row['WRank']) else None
        tennis_l_rank = int(float(tennis_row['LRank'])) if pd.notna(tennis_row['LRank']) else None
        
        has_rank_match = (jeff_w_rank == tennis_w_rank and jeff_l_rank == tennis_l_rank and 
                         all(x is not None for x in [jeff_w_rank, jeff_l_rank, tennis_w_rank, tennis_l_rank]))
    except:
        has_rank_match = False
    
    if has_score_match or has_rank_match:
        high_confidence += 1

print(f"")
print(f"HIGH-CONFIDENCE IDENTICAL MATCHES:")
print(f"  Matches with score OR ranking verification: {high_confidence:,} ({high_confidence/len(jeff_matched)*100:.1f}%)")
print(f"")

if high_confidence >= 1000:
    print(f"✅ VERIFICATION PASSED: {high_confidence:,} matches are verified as identical")
    print(f"✅ Safe to proceed with model comparison on these matches")
else:
    print(f"⚠️  VERIFICATION CONCERN: Only {high_confidence:,} matches verified as identical")
    print(f"⚠️  May need stricter matching criteria")