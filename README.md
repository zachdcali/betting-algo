# Tennis Match Prediction System

A comprehensive machine learning system for predicting professional tennis match outcomes using advanced temporal feature engineering and multiple model architectures. Built with rigorous data leakage prevention and validated on 684K+ matches spanning 1990-2024.

## Project Overview

This project applies cutting-edge machine learning techniques to tennis match prediction, achieving significant improvements over ATP ranking baselines through sophisticated temporal feature engineering, proper validation methodologies, and leak-free preprocessing. The system has undergone extensive validation including shuffled target tests to ensure zero data leakage.

### Current Status ✅
The system features production-ready models with excellent calibration metrics (ECE: 0.008), validated on comprehensive tennis match data. All models have been rigorously tested for data leakage and show realistic performance suitable for live betting applications.

## Key Results

### Professional Tennis Models (684K+ matches, 1990-2024)

| Model | Accuracy | Improvement vs ATP Rankings | AUC-ROC | Log Loss | ECE (Calibration) |
|-------|----------|----------------------------|---------|----------|-------------------|
| **Neural Network** | **70.69%** | **+6.01pp** | **0.789** | **0.547** | **0.008** |
| **XGBoost** | **69.17%** | **+4.49pp** | **0.766** | **0.573** | **0.015** |
| **Random Forest** | **66.85%** | **+2.17pp** | **0.736** | **0.602** | **0.028** |
| ATP Ranking Baseline | 64.68% | - | - | - | - |

*Neural Network shows the best overall performance with excellent calibration for betting applications.*

### Performance by Tournament Level

| Tournament Level | Neural Network | XGBoost | Random Forest | ATP Baseline | Sample Size |
|------------------|----------------|---------|---------------|--------------|-------------|
| **ATP-Level** | **68.2%** (+5.5pp) | **66.8%** (+4.2pp) | **64.2%** (+1.5pp) | 62.7% | 8,133 matches |
| **Challengers** | **66.4%** (+4.0pp) | **66.8%** (+4.4pp) | **64.8%** (+2.4pp) | 62.4% | 18,488 matches |
| **ITF Futures** | **74.2%** (+7.5pp) | **71.4%** (+4.8pp) | **68.9%** (+2.3pp) | 66.7% | 28,822 matches |

*Models show largest prediction edges at lower tournament levels, with ITF Futures being most predictable due to larger skill gaps.*

### Performance by Surface

| Surface | Neural Network | XGBoost | Random Forest | Sample Size |
|---------|----------------|---------|---------------|-------------|
| **Clay** | **71.1%** (+5.9pp) | **69.9%** (+4.7pp) | **67.2%** (+2.1pp) | 25,074 matches |
| **Hard** | **70.4%** (+6.1pp) | **68.6%** (+4.3pp) | **66.6%** (+2.2pp) | 28,976 matches |
| **Grass** | **68.2%** (+5.2pp) | **67.3%** (+4.4pp) | **65.2%** (+2.2pp) | 1,470 matches |
| **Carpet** | **74.1%** (+10.1pp) | **75.5%** (+11.5pp) | **69.0%** (+5.1pp) | 355 matches |

## Features & Architecture

### Advanced Temporal Engineering (63 Features)
- **Ranking momentum tracking** (14d, 30d, 90d windows) with volatility metrics
- **Surface-specific adaptation** tracking recent performance on current surface
- **Head-to-head records** with temporal weighting and surface context
- **Fatigue and rust metrics** (days since last match, rust flags)
- **Win/loss streaks** and form trends with momentum indicators
- **Tournament-level experience** and big match performance
- **Seasonal patterns** (clay season, grass season, indoor season indicators)

### Core Features (85 Total)
- **Player attributes**: Rankings, rank points, age, height, handedness
- **Match context**: Tournament level, surface, round, draw size
- **Derived metrics**: Ranking differences, ratios, advantages
- **Geographic factors**: Country-specific performance patterns

### ML Architecture Highlights
- **Leak-free preprocessing**: Randomize player positions FIRST, then calculate all features
- **Temporal validation**: 1990-2022 train, 2023-2024 test (chronological split)
- **Binary advantage exclusion**: Removed redundant features that caused over-reliance
- **Advanced regularization**: XGBoost with L1/L2 regularization, NN with dropout
- **Cross-validation monitoring**: 3-fold CV to prevent overfitting

## Technical Implementation

### Models
- **Neural Network**: 4-layer architecture (143→128→64→32→16→1) with sigmoid output
  - Dropout regularization (0.3, 0.3, 0.2, 0.2)
  - Adam optimizer with 0.001 learning rate
  - 100 epochs with early stopping capability

- **XGBoost**: Gradient boosting optimized for feature balance
  - 150 estimators, max_depth=5, learning_rate=0.1
  - L1/L2 regularization (reg_alpha=1.0, reg_lambda=1.0)
  - Column subsampling (colsample_bytree=0.6) for diversity

- **Random Forest**: Ensemble method with careful hyperparameter tuning
  - 100 estimators, max_depth=15
  - Balanced feature importance distribution

### Data Leakage Prevention ⚠️ CRITICAL
1. **"Randomize First" methodology**: Player1/Player2 assignment occurs BEFORE any feature calculation
2. **Chronological processing**: All temporal features calculated using only past information
3. **Advantage feature exclusion**: Removed binary advantages that caused model over-reliance
4. **Shuffled target validation**: Models achieve ~50% accuracy on randomized targets (confirming no leakage)

## Feature Importance Insights

### Neural Network (Permutation Importance)
1. **P1_WinStreak_Current** (4.5%) - Current form momentum
2. **P1_Surface_Matches_30d** (4.4%) - Recent surface activity
3. **P2_Surface_Matches_30d** (4.1%) - Opponent surface activity
4. **Height_Diff** (3.2%) - Physical advantage
5. **P1_Matches_30d** (3.1%) - Recent match activity

### XGBoost (Gain Importance)
1. **Rank_Diff** (18.3%) - Core ranking differential
2. **Rank_Ratio** (5.4%) - Relative ranking strength
3. **P2_Level_WinRate_Career** (3.9%) - Tournament-level experience
4. **P1_Level_WinRate_Career** (3.6%) - Player experience
5. **Rank_Points_Diff** (2.9%) - Points-based differential

*Neural Network focuses on temporal patterns while tree models emphasize static comparisons - both approaches are valid and complementary.*

## Project Structure

```
├── src/
│   ├── Models/
│   │   └── professional_tennis/     # Main tennis prediction models
│   │       ├── preprocess.py        # Leak-free temporal feature engineering
│   │       ├── train_rf.py         # Random Forest implementation
│   │       ├── train_xgb.py        # XGBoost with regularization
│   │       └── train_nn.py         # Neural Network with permutation importance
│   ├── data_processing/             # Data cleaning and preparation
│   ├── scrapers/                    # Data collection utilities
│   └── utils/                      # Helper functions and utilities
├── results/                        # Model outputs, visualizations, and analyses
│   └── professional_tennis/        # Detailed results by model type
├── data/
│   └── JeffSackmann/              # Tennis match data (Jeff Sackmann's repository)
├── analysis_scripts/              # Kelly Criterion and backtesting scripts
└── requirements.txt               # Python dependencies
```

## Reproducibility

### Environment Setup
```bash
# Using virtual environment (recommended)
python -m venv tennis_env
source tennis_env/bin/activate  # On Windows: tennis_env\Scripts\activate
pip install -r requirements.txt

# Alternative: Docker environment
docker build -t tennis-prediction .
docker run -it tennis-prediction
```

### Data Preprocessing
```bash
# Generate leak-free dataset
cd src/Models/professional_tennis/
python preprocess.py
```

### Model Training
```bash
# Train all models with leak-free features
python train_nn.py      # Neural Network (best overall performance)
python train_xgb.py     # XGBoost (balanced feature importance)
python train_rf.py      # Random Forest (baseline comparison)
```

### Validation & Testing
```bash
# Models automatically run shuffled target tests
# Expected output: ~50% accuracy on randomized targets (confirms no leakage)
```

## Data Requirements

The system processes comprehensive tennis match data from Jeff Sackmann's repository:
- **Match metadata**: Date, tournament, surface, round, draw size
- **Player information**: Rankings, rank points, ages, countries, handedness
- **Historical outcomes**: Win/loss records with temporal context
- **Tournament details**: Level classification, prize money tiers

**Data Coverage**: 916K+ matches (1967-2024), filtered to 684K+ matches (1990-2024) with complete ranking data.

## Methodology

### Validation Strategy
- **Temporal split**: Train on 1990-2022, test on 2023-2024 (no future data leakage)
- **Shuffled target test**: Confirms models achieve ~50% accuracy on randomized labels
- **Chronological processing**: All features calculated using only past information
- **Realistic baselines**: ATP ranking predictions with proper randomized player positions

### Performance Analysis
- **Year-by-year breakdown**: Consistent 2023-2024 performance
- **Tournament-level analysis**: Identifies edge opportunities at different competition levels
- **Surface-specific metrics**: Performance varies by playing surface
- **Calibration analysis**: Low ECE scores confirm probability reliability for betting

### Statistical Rigor
- **Cross-validation**: 3-fold stratified CV during training
- **Multiple metrics**: Accuracy, AUC-ROC, Log Loss, Brier Score, ECE
- **Significance testing**: Large sample sizes ensure statistical validity
- **Benchmark comparison**: Performance vs. published tennis ML research

## Betting Applications

### Model Calibration
- **Excellent ECE scores** (0.008-0.028) indicate probabilities are trustworthy
- **Low Brier scores** (0.187-0.208) show minimal prediction error
- **Consistent performance** across years and tournament levels

### Kelly Criterion Ready
Models provide well-calibrated probabilities suitable for:
- Kelly Criterion bankroll management
- Value betting identification
- Risk-adjusted position sizing

---

*This project demonstrates a complete, production-ready machine learning system for sports prediction with rigorous validation and leak-prevention methodologies. The system achieves state-of-the-art performance while maintaining statistical rigor suitable for live betting applications.*