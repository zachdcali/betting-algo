# Tennis Match Prediction System

A machine learning system for predicting professional tennis match outcomes using temporal feature engineering and multiple model architectures.

## Project Overview

This project applies machine learning techniques to tennis match prediction, achieving measurable improvements over ATP ranking baselines through temporal feature engineering and proper validation methodologies.

### Current Status
The system has established baseline models using comprehensive tennis match data (1990-2024). Current data collection efforts focus on integrating Universal Tennis Rating (UTR) data, which preliminary analysis suggests provides ~1% accuracy improvement over ATP rankings specifically for ATP-level matches (the most competitive tier).

## Key Results

### Professional Tennis Models (684K+ matches, 1990-2025)

| Model | Accuracy | Improvement vs ATP Rankings | AUC-ROC |
|-------|----------|----------------------------|---------|
| **XGBoost** | **70.86%** | **+6.18pp** | **0.790** |
| **Neural Network** | **70.61%** | **+5.93pp** | **0.789** |
| **Random Forest** | **66.83%** | **+2.15pp** | **0.736** |
| ATP Ranking Baseline | 64.68% | - | - |

### Performance by Tournament Level

| Tournament Level | Model Accuracy | ATP Baseline | Improvement | Sample Size |
|------------------|----------------|--------------|-------------|-------------|
| **ITF Futures** | 73.9% | 66.7% | **+7.2pp** | 28,822 matches |
| **ATP-Level** | 68.4% | 62.7% | **+5.7pp** | 8,133 matches |
| **Challengers** | 67.4% | 62.4% | **+5.0pp** | 18,488 matches |

*Results show larger prediction edges at lower tournament levels, with ATP-level matches representing the most competitive prediction environment.*

## Features

### Temporal Engineering
- Ranking momentum tracking (14d, 30d, 90d windows)
- Surface-specific adaptation (recent performance on current surface)
- Head-to-head records with surface and tournament level weighting
- Fatigue metrics (days since last tournament, not match-level to prevent leakage)
- Win/loss streaks and form trends
- Seasonal patterns and player-specific tendencies

### ML Architecture
- Temporal validation (1990-2022 train, 2023+ test)
- Leakage prevention (chronological processing, tournament-based rest metrics)
- Feature importance analysis with 149 engineered features
- Comprehensive performance breakdowns (by year, surface, tournament level)

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── professional_tennis/     # Main tennis prediction models
│   │   │   ├── preprocess.py        # Temporal feature engineering
│   │   │   ├── train_rf.py         # Random Forest implementation
│   │   │   ├── train_xgb.py        # XGBoost implementation
│   │   │   └── train_nn.py         # Neural Network implementation
│   │   └── betting_odds/           # Historical odds-based models
│   └── scrapers/                   # Data collection utilities
├── results/                        # Model outputs and analyses
├── docs/                          # Detailed documentation
└── requirements.txt               # Python dependencies
```

## Technical Implementation

### Models
- XGBoost: Gradient boosting with 150 estimators, optimized for temporal patterns
- Neural Network: 4-layer architecture (128→64→32→16) with dropout regularization
- Random Forest: 100 estimators with careful hyperparameter tuning

### Key Technical Innovations
1. Chronological feature calculation to prevent data leakage
2. Tournament-based rest metrics instead of match-based (prevents progression leakage)
3. Weighted head-to-head records considering surface and tournament importance
4. Adaptive surface performance tracking recent form on specific surfaces

## Feature Importance Insights

Top contributing features across models:
1. Player1_Rank_Advantage (49.9% importance in XGBoost)
2. Rank_Diff and Rank_Ratio 
3. Recent form trends (14d, 30d windows)
4. Win streak momentum
5. Surface-specific adaptation

## Reproducibility

### Environment Setup
```bash
# Build Docker environment
docker build -t tennis-prediction .
docker run -it tennis-prediction

# Install dependencies
pip install -r requirements.txt
```

### Model Training
```bash
# Professional tennis models
cd src/models/professional_tennis/
python train_xgb.py
python train_nn.py  
python train_rf.py

# Betting odds models
cd src/models/betting_odds/
python train_rf.py
```

## Data Requirements

The system expects processed tennis match data with the following structure:
- Match metadata (date, tournament, surface, round)
- Player information (rankings, ages, countries, handedness)
- Historical performance metrics

*Note: Actual datasets are not included to protect proprietary data sources.*

## Methodology

### Validation Strategy
- Temporal split: Train on 1990-2022, test on 2023+
- No data leakage: Chronological processing with strict temporal boundaries
- Realistic baselines: ATP ranking predictions accounting for randomized player positions

### Performance Analysis
- Year-by-year breakdown showing consistent improvements
- Tournament-level analysis revealing edge opportunities
- Surface-specific performance metrics
- Statistical significance testing

---

*This project represents a professional-grade machine learning system suitable for production deployment in sports analytics environments.*