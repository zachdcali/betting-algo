# Tennis Match Prediction System

A machine learning system for predicting professional ATP/Challenger tennis match outcomes. Trained on 900K+ historical matches (1990–2024) with strict temporal validation to prevent data leakage.

## Performance

All models evaluated on a held-out chronological test set. The ATP ranking baseline predicts based on current rank only; closing betting odds represent the market ceiling.

### Apples-to-Apples (identical 1,525-match sample with closing odds)

| Method | Accuracy |
|--------|----------|
| Closing Betting Odds | 68.13% |
| **XGBoost** | **65.11%** |
| **Neural Network** | **65.38%** |
| **Random Forest** | **64.13%** |
| ATP Ranking Baseline | 63.21% |

### Full Test Set (~56K matches)

| Model | Accuracy | AUC-ROC | vs ATP Baseline |
|-------|----------|---------|-----------------|
| XGBoost | 66.51% | 0.7285 | +1.83 pp |
| Neural Network | 66.38% | 0.7268 | +1.70 pp |
| Random Forest | 66.08% | 0.7246 | +1.40 pp |
| ATP Baseline | 64.68% | — | — |

![Apples-to-Apples Accuracy](analysis_scripts/backtests/apples_to_apples.png)

## Architecture

Three model types trained on identical leak-free data with chronological train/test splits:

- **Neural Network** — 5-layer MLP (128→64→32→16→1) with dropout and sigmoid output
- **XGBoost** — gradient boosted trees
- **Random Forest** — bagged ensemble

Features span current rankings, recent form, surface performance, head-to-head records, career stats, and match context (surface, level, round, draw size). Bayesian smoothing applied to rate-based features to handle sparse data robustly.

## Data Sources

- **Historical match data**: [Jeff Sackmann's tennis_atp](https://github.com/JeffSackmann/tennis_atp) (1990–2024, 900K+ matches)
- **Live player data**: Tennis Abstract (current rankings, match history, player profiles)
- **Live odds**: Bovada

## Repository Structure

```
src/models/professional_tennis/   # Training scripts (preprocess, train_nn, train_xgb, train_rf)
production/                       # Live pipeline (odds scraping, feature calc, inference)
  main.py                         # End-to-end orchestrator
  odds/                           # Bovada odds scraper
  features/                       # Live feature computation
  models/                         # Model inference
  utils/                          # Kelly staking, bet tracking
analysis_scripts/                 # Backtesting and evaluation
results/professional_tennis/      # Evaluation outputs (model weights kept local)
```

## Live Pipeline

The `production/` pipeline runs daily:
1. Scrape upcoming ATP/Challenger matches and odds (Bovada)
2. Fetch current player stats from Tennis Abstract
3. Compute 141 features per matchup
4. Run inference with the neural network
5. Calculate edges vs implied market probabilities
6. Apply fractional Kelly staking
7. Log predictions for later accuracy tracking

## Reproducibility

```bash
python -m venv tennis_env
source tennis_env/bin/activate
pip install -r requirements.txt

# Preprocess and train
python src/models/professional_tennis/preprocess.py
python src/models/professional_tennis/train_nn_143.py
python src/models/professional_tennis/train_xgb.py
python src/models/professional_tennis/train_rf.py

# Run live pipeline
cd production && python main.py
```

*Data files are not committed. Model weights are kept local.*
