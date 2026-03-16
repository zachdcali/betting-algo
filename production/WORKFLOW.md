# 🎾 Live Tennis Betting System - Complete Workflow

This document explains exactly how the production betting system works, step by step.

## 📋 **System Overview**

The system follows ChatGPT's modular design with complete bet tracking:

```
[Fetch Odds] → [Extract Features] → [Model Inference] → [Calculate Stakes] → [Log Bets] → [Settle Results]
```

## 🔄 **Daily Workflow**

### **1. Morning Setup (9 AM)**
```bash
cd /Users/zachdodson/Documents/betting-algo/production

# Check current bankroll and pending bets
python settle_bets.py --show-pending

# Run the main pipeline
python main.py --config config.json --bankroll 1000
```

**What happens:**
- ✅ Scrapes Bovada for ATP/Challenger/ITF matches (handles "Show More" buttons)
- ✅ Filters for Men's professional tennis only
- ✅ Extracts 143 features using UTR match data
- ✅ Runs NN-143 model inference with calibration
- ✅ Calculates Kelly stakes using block betting strategy
- ✅ **Logs all bets to tracking system** 📝
- ✅ Shows both dollar amounts and percentages

### **2. Review Bet Slips**
The system outputs detailed bet recommendations:
```
📋 BETTING SUMMARY - 2025-08-24 14:30:00
================================================================================
Bankroll: $1000.00
Total stakes: $85.40 (8.5% of bankroll)  
Number of bets: 4
Potential profit: $165.20
Average edge: 3.2%

BET DETAILS:
--------------------------------------------------------------------------------
Novak Djokovic vs Rafael Nadal
  Bet on: Novak Djokovic @ 2.10
  Stake: $28.50 (2.9% of bankroll)  ← Shows both $ and %
  Edge: 4.1% | Potential: $31.35
  Event: ATP Masters 1000 - Indian Wells

📊 Session tracking:
   Session ID: session_20250824_143000
   Total pending bets: 4
   Use 'python settle_bets.py session_20250824_143000' to settle results later
```

### **3. Place Actual Bets** 
**Manual step:** Use the bet slips to place actual bets on your sportsbook.

### **4. Evening Settlement (After matches finish)**
```bash
# Interactive settlement of all pending bets
python settle_bets.py --interactive

# OR settle a specific session
python settle_bets.py session_20250824_143000 --interactive
```

**What happens:**
- Shows each pending bet
- Prompts for win/loss result
- **Automatically updates bankroll** 💰
- **Tracks P&L and win rate** 📊
- **Maintains complete betting history** 📈

## 🔧 **Key Features Explained**

### **✅ Comprehensive Bet Logging**
Every bet is logged with:
- Bet ID, session, timestamps
- Match details, stakes, odds, edges
- Model probabilities vs market probabilities
- Settlement results and P&L
- Complete bankroll history

### **✅ Block Betting Strategy**
- Groups matches by day/event
- Allocates Kelly stakes proportionally within each block
- Prevents over-leveraging
- Scales down if total > bankroll

### **✅ ATP/Challenger/ITF Only**
- Filters for: "ATP", "Challenger", "ITF", "Men's"
- Blocks: "Women", "WTA", "Exhibition", "UTR", "Doubles"

### **✅ Multiple "Show More" Handling**
- Clicks all visible "Show More" buttons
- Repeats until no more buttons found
- Handles tournaments with many matches (like US Open R1)

### **✅ Player Name Matching**
- Uses existing UTR player mapping with name variants
- Handles special characters (like ø, é, etc.)
- Fallback to fuzzy matching for new players

## 📁 **File Structure & Purpose**

```
production/
├── main.py                    # 🎯 Main orchestrator - run this
├── settle_bets.py             # 💰 Bet settlement script  
├── config.json                # ⚙️ Configuration
│
├── odds/
│   └── fetch_bovada.py        # 🕷️ Bovada scraper (ATP/Challenger/ITF only)
│
├── features/ 
│   └── extract_features.py    # 🔧 143-feature extraction from UTR data
│
├── models/
│   └── inference.py           # 🎯 NN-143 model + calibration
│
├── utils/
│   ├── stake_calculator.py    # 💵 Kelly stakes + block betting
│   ├── bet_tracker.py         # 📊 Complete bet logging system
│   └── name_matcher.py        # 🔍 Player name normalization
│
└── logs/                      # 📁 All output data
    ├── all_bets.csv          # Complete bet history
    ├── bankroll_history.csv  # Bankroll tracking
    ├── betting_sessions.csv  # Session summaries
    └── bet_slips_latest.csv  # Latest recommendations
```

## 🚀 **Command Reference**

### **Main Pipeline**
```bash
# Basic run
python main.py

# With custom bankroll  
python main.py --bankroll 2000

# More conservative Kelly
python main.py --kelly-multiplier 0.10

# Dry run (no bet logging)
python main.py --dry-run
```

### **Bet Settlement**
```bash
# Show all pending bets
python settle_bets.py --show-pending

# Interactive settlement
python settle_bets.py --interactive  

# Session summary
python settle_bets.py session_20250824_143000

# Settle specific bet
python settle_bets.py --bet-id bet_20250824_143000_1
```

### **Testing**
```bash
# Test all components
python test_system.py

# Test individual modules
cd odds && python fetch_bovada.py
cd models && python inference.py
```

## 🎯 **Expected Daily Results**

**Good day:**
- 5-15 ATP/Challenger matches found
- 2-8 profitable betting opportunities (>2% edge)
- $50-200 total stakes on $1000 bankroll
- Clear bet slips with dollar amounts and percentages

**Slow day:**
- 0-5 matches found (off-season)
- 0-2 betting opportunities
- May need to wait for bigger tournaments

## ⚠️ **Important Notes**

1. **Uses same NN-143 model** as your variance analysis (placeholder until you retrain)
2. **Block betting** matches your `allocate_block_stakes` logic exactly
3. **Complete audit trail** - every bet logged and tracked
4. **No automatic bet placement** - generates slips for manual execution
5. **Special character handling** - normalizes player names properly
6. **Handles large tournaments** - clicks all "Show More" buttons

## 🔄 **Integration with Your Analysis**

The production system uses:
- ✅ **Same 143 features** as your backtests
- ✅ **Same Kelly methodology** as `variance_bootstrap.py`  
- ✅ **Same block betting** as `deterministic_backtest.py`
- ✅ **Same NN-143 model** (placeholder until retrained)
- ✅ **Same edge thresholds** as your analysis

Once you get your optimal `k` value from the variance analysis, just update `config.json`:
```json
{
  "kelly_multiplier": 0.18  // Your optimal k value
}
```

## 🎉 **Ready for Production!**

The system is designed to handle everything ChatGPT recommended:
- ✅ Modular components for easy debugging  
- ✅ Complete bet logging and settlement
- ✅ Block betting for realistic tournament scenarios
- ✅ ATP/Challenger/ITF filtering with "Show More" handling
- ✅ Player name matching with existing UTR data
- ✅ Dollar amounts + percentages for clarity
- ✅ Full audit trail and P&L tracking

Just run `python main.py` and you're live! 🚀