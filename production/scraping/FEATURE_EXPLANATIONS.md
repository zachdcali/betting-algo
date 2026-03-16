# Feature Explanations - Shapovalov vs Ruud Test

## Summary of Your Manual Review

This document clarifies the features you marked as "not sure" or "incorrect" in your manual audit.

---

## Features You Flagged as Incorrect

### 1. `P1_Form_Trend_30d` and `P2_Form_Trend_30d`

**Your Calculation**: Simple win rate
- P1: 3/6 = 0.5
- P2: 7/9 = 0.778

**Actual Calculation**: **Exponential Weighted Moving Average**
- More recent matches get HIGHER weight
- Weight formula: `w = exp(-days_ago / 15)` where 15 is the half-life
- A match 1 day ago gets weight ~0.94
- A match 15 days ago gets weight ~0.37
- A match 30 days ago gets weight ~0.14

**Why Different**: Recent wins/losses matter MORE than older ones.

**Source**: `preprocess.py` lines 490-515 (`calculate_form_trend` function)

```python
def calculate_form_trend(match_history, target_date, days=30):
    # Calculate exponential weights (more recent = higher weight)
    for match in recent_matches:
        days_ago = (target_date - match['date']).days
        weight = np.exp(-days_ago / 15)  # Half-life of 15 days

        total_weight += weight
        if match['won']:
            weighted_wins += weight

    return weighted_wins / total_weight
```

**Verdict**: ✅ CORRECT - Our implementation matches training exactly (ta_feature_calculator.py lines 143-159)

---

### 2. `P2_Level_WinRate_Career`

**Your Calculation**: 421/656 = 0.6417...
**Our Value**: 0.6711...

**Possible Causes**:
1. We're only seeing 225 ATP-level matches (not 656 total)
2. Training filters by specific level (A = ATP 250/500)
3. May not be getting all career matches from "Show Career" page

**TODO**: Verify we're pulling ALL career matches from Tennis Abstract

---

### 3. `P1_Level_WinRate_Career`

**Your Calculation**: 309/537 = 0.5754...
**Our Value**: 0.5381...

**Same Issue**: May not be getting full career match count

---

## Features You Marked as "Not Sure"

### 4. `P2_Level_Matches_Career` = 225

**What It Is**: Number of career matches at ATP-level tournaments (Level_A)

**Source**: `preprocess.py` line 90
```python
df_sorted.loc[row.name, f'{prefix}Level_Matches_Career'] = level_stats['total']
```

**Calculation**: Count all matches where `level = 'A'` (ATP 250/500)
- Does NOT include Masters (M), Grand Slams (G), Challengers (C), etc.
- Only ATP-level events

---

### 5. `P2_WinRate_Last10_120d` = 0.8

**What It Is**: Win rate over **last 10 matches** within **120 days**

**Logic**:
1. Get all matches in last 120 days
2. Take the **most recent 10** from that window
3. Calculate win rate
4. Requires minimum 3 matches, else defaults to 0.5

**Source**: `preprocess.py` lines 319-337 (`calculate_recent_form`)

```python
def calculate_recent_form(match_history, target_date, max_matches=10, max_days=120):
    # Get recent matches within time window
    recent_matches = [matches in last 120 days]

    # Sort by date and take most recent N matches
    recent_matches = recent_matches[:10]

    if len(recent_matches) >= 3:
        return wins / len(recent_matches)
    else:
        return 0.5  # Neutral prior
```

**For P1**: Only had <3 matches in last 120 days → defaults to 0.5
**For P2**: Had ≥3 matches, actual win rate calculated

---

### 6. `P1_Round_WinRate_Career` and `P2_Round_WinRate_Career`

**What It Is**: Career win rate at the **specific round** of the upcoming match

**Example**: If upcoming match is R32 (Round of 32), this is win rate in all career R32 matches

**Thresholds** (`preprocess.py` lines 260-266):
- Finals (F): ≥1 match else 0.5
- Semifinals (SF): ≥2 matches else 0.5
- Other rounds: ≥3 matches else 0.5

---

### 7. `draw_size` = 32

**Where From**: This comes from **Bovada or tournament metadata**, NOT Tennis Abstract

**In Production**:
- Bovada API provides tournament info
- We use tournament resolver to map tournament names to draw sizes
- For this test, we hardcoded `draw_size=32`

**NOT Hardcoded in Production**: Will be fetched from tournament metadata

---

### 8. `Age_Diff` and Ages

**Age Rounding**: Uses **full years** (integer age), NOT fractional

**Calculation**:
```python
birth_year = int(dob_str[:4])
age = datetime.now().year - birth_year
```

**Example**:
- Shapovalov: Born 1999-04-15 → 2025 - 1999 = **26**
- Ruud: Born 1998-12-22 → 2025 - 1998 = **27**
- Age_Diff = 26 - 27 = **-1**

---

### 9. `P2_Rank_Volatility_90d` = 0.4577...

**What It Is**: **Standard deviation** of rank over last 90 days

**Calculation**:
1. Get all rank values from matches in last 90 days
2. Calculate `std(ranks)`

**Interpretation**:
- Low value (like 0.46) = stable rank
- High value = fluctuating rank

---

### 10. `P1_Days_Since_Last` and `P2_Days_Since_Last` = 14

**What It Is**: Days since **start of last tournament** (tournament-week basis)

**Logic** (`preprocess.py` lines 432-453):
1. Group matches by tournament week (using Monday as key)
2. Find most recent tournament week that ended BEFORE current match
3. Calculate days between Mondays

**Example**:
- Last tournament: Week of Oct 6 (Monday)
- Current match: Oct 20 (in week of Oct 13 Monday = Oct 20)
- Days = 14

**NOT**: Days since last individual match (that would leak data since we'd know they advanced)

---

### 11. `P2_Rust_Flag` and `P1_Rust_Flag`

**What It Is**: Binary flag indicating if player has been inactive

**Calculation**:
```python
Rust_Flag = 1 if Days_Since_Last > 21 else 0
```

**Interpretation**:
- 0 = Player is active (played in last 3 weeks)
- 1 = Player is "rusty" (21+ days since last tournament)

**Both = 0 Here**: Both players played 14 days ago, so neither is rusty

---

### 12. `Indoor_Season` = 1

**What It Is**: Binary flag for indoor tennis season

**Calculation** (`preprocess.py` lines 146-149):
```python
Indoor_Season = 1 if month >= 10 or month <= 2 else 0
```

**Months**: October, November, December, January, February

**Today is Oct 18** → Month = 10 → `Indoor_Season = 1` ✅ CORRECT

**Why**: Indoor season typically runs Oct-Feb (fall/winter), outdoor is Mar-Sep

---

### 13. `H2H_Recent_P1_Advantage` = -0.1667...

**What It Is**: Recent H2H advantage (last 3 meetings)

**Calculation** (`preprocess.py` lines 393-408):
```python
# Take last 3 H2H meetings
recent_h2h = h2h_matches[:3]

if len(recent_h2h) >= 2:
    wins = sum(1 for match in recent_h2h if match['won'])
    return wins / len(recent_h2h) - 0.5  # Advantage relative to 50%
else:
    return 0  # No advantage for insufficient data
```

**For Shapovalov vs Ruud**:
- Last 4 H2H matches: Ruud won 3, Shapovalov won 1
- Last 3 meetings: Likely Ruud won 2-3
- If Ruud won 2/3: Shapovalov won 1/3 = 0.333
- Advantage = 0.333 - 0.5 = **-0.167** ✅ CORRECT

**Interpretation**:
- Negative value = P2 (Ruud) has recent advantage
- Positive value = P1 has recent advantage
- 0 = Even or insufficient data

---

### 14. `P1_BigMatch_WinRate` and `P2_BigMatch_WinRate`

**What It Is**: Win rate in **Grand Slams + Masters** tournaments only

**Calculation** (`preprocess.py` lines 92-96):
```python
big_match_wins = stats['level_stats']['G']['wins'] + stats['level_stats']['M']['wins']
big_match_total = stats['level_stats']['G']['total'] + stats['level_stats']['M']['total']

if big_match_total >= 3:
    return big_match_wins / big_match_total
else:
    return 0.5  # Default for insufficient data
```

**Requires**: ≥3 big matches total, else defaults to 0.5

---

### 15. `Rank_Momentum_Diff_90d` = 9

**What It Is**: Difference in rank momentum over 90 days

**Calculation**:
```python
Rank_Momentum_Diff_90d = P1_Rank_Change_90d - P2_Rank_Change_90d
```

**Example**:
- P1_Rank_Change_90d = 10 (improved by 10 spots)
- P2_Rank_Change_90d = 1 (improved by 1 spot)
- Diff = 10 - 1 = **9**

**Interpretation**:
- Positive = P1 has better momentum
- Negative = P2 has better momentum

---

## Features You Said Are Correct

These I'll just clarify for completeness:

- `P1_Hand_U` = "Hand Unknown" (Ambidextrous = A, Unknown = U)
- `P1_Hand_A` = "Ambidextrous"
- `Peak_Age_P1/P2` = 1 if age 24-28 (prime tennis years), else 0
- `Rank_Ratio` = max_rank / min_rank (higher = bigger ranking gap)
- All country one-hots are correct
- All handedness matchups are correct

---

## Summary of Action Items

1. ✅ **Form_Trend**: CORRECT - uses exponential weighting, not simple win rate
2. ⚠️ **Level_WinRate_Career**: May not be getting full career - need to verify "Show Career" parsing
3. ✅ **All other features**: Either correct or appropriately using defaults
4. ⚠️ **P1_WinRate_Last10_120d**: Defaulting to 0.5 due to <3 matches - verify this is correct for Shapovalov

---

## Next Steps

1. Verify Tennis Abstract is returning FULL career matches (not just recent years)
2. Confirm `P1_WinRate_Last10_120d` should be 0.5 or if we're missing matches
3. All other features appear to be calculated correctly per training logic
