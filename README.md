# UFC Pre-fight Prediction MVP

A comprehensive machine learning pipeline for predicting UFC fight outcomes using fighter statistics, historical performance, market odds, and advanced analytics. This system implements a sophisticated two-stage stacking approach with market-aware features and rigorous backtesting capabilities.

## 🥊 Project Overview

This project implements an end-to-end ML pipeline for UFC fight prediction, combining traditional sports analytics with modern machine learning techniques. The system features:

- **Two-Stage Market-Aware Stacking**: Skill-only models + market odds integration
- **Vig-Free Odds Processing**: Advanced betting market analysis
- **Side×Band Calibration**: Per-side, per-odds-regime probability calibration
- **Comprehensive Backtesting**: Multi-threshold EV analysis with Kelly staking
- **Advanced Filtering**: Side-specific, probability-band, and edge-capping strategies

## 🏗️ Project Structure

```
ufc-predictor/
├── pyproject.toml          # Project configuration and dependencies
├── requirements.txt         # Python package requirements
├── README.md               # This file
├── .env.example            # Environment variables template
├── data/                   # Data storage and artifacts
│   ├── raw/               # Original CSV data files
│   ├── interim/           # Cleaned and intermediate data
│   ├── processed/         # Feature matrices and train/test splits
│   ├── models/            # Saved model artifacts
│   ├── reports/           # Evaluation outputs and visualizations
│   │   ├── calibration/   # Side×band calibrators
│   │   ├── backtests/     # EV backtesting results
│   │   │   └── sweeps/    # Multi-threshold sweep results
│   │   └── selector/      # Profit meta-selectors
│   └── splits/            # Time-based CV splits
├── src/ufcml/             # Main package source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   ├── io.py              # Data input/output operations
│   ├── clean.py           # Data cleaning and preprocessing
│   ├── features.py        # Feature engineering (including form features)
│   ├── elo.py             # ELO rating system implementation
│   ├── split.py           # Time-based data splitting strategies
│   ├── train.py           # Model training pipeline (skill + market stacking)
│   ├── calibrate.py       # Basic model calibration
│   ├── calibration.py     # Advanced side×band calibration
│   ├── evaluate.py        # Model evaluation and metrics
│   ├── predict.py         # Prediction logic
│   ├── api.py             # FastAPI web service
│   └── utils.py           # Utility functions (vig-free odds, etc.)
├── scripts/                # Executable pipeline scripts
│   ├── make_clean.py      # Data cleaning pipeline
│   ├── make_features.py   # Feature engineering pipeline
│   ├── make_splits.py     # Data splitting pipeline
│   ├── train_skill.py     # Skill-only model training
│   ├── train_market.py    # Market-aware stacking training
│   ├── evaluate_all.py    # Comprehensive evaluation
│   ├── export_cv_predictions.py  # Export CV predictions for backtesting
│   ├── train_sideband_calibrators.py  # Train advanced calibrators
│   ├── train_profit_selector.py  # Train profit meta-selectors
│   ├── backtest_ev.py     # Basic EV backtesting
│   ├── backtest_sweep.py  # Multi-threshold EV sweep
│   └── predict_cli.py     # Command-line prediction tool
└── tests/                  # Test suite
    ├── test_clean.py      # Data cleaning tests
    ├── test_features.py   # Feature engineering tests
    └── test_split.py      # Data splitting tests
```

## 🚀 Pipeline Stages

### 1. Data Ingestion & Cleaning (`clean.py`)
- Load raw UFC data from CSV files
- Handle missing values and data inconsistencies
- Standardize fighter names and fight records
- Validate data integrity and schema
- **NEW**: Optionally preserve historical columns (Finish, FinishDetails) for form features

### 2. Feature Engineering (`features.py`)
- Extract fighter statistics (wins, losses, KO rate, etc.)
- Calculate rolling averages and trends
- Generate fight-specific features (weight class, venue, etc.)
- Implement time-based feature decay
- **NEW**: Short-term "form" features (recent win rate, fights past 365 days, days since last fight)
- **NEW**: Recent KO/Submission share analysis (last 3 fights)
- **NEW**: ELO rating integration with time-aware calculation

### 3. ELO Rating System (`elo.py`)
- Calculate and maintain ELO ratings for fighters
- Handle rating updates after fights with time decay
- Implement weight class and title bout adjustments
- Provide baseline prediction probabilities
- **NEW**: Time-aware ELO with K-factor adjustments

### 4. Data Splitting (`split.py`)
- Implement temporal splitting (no future data leakage)
- **NEW**: Month-based expanding window splits
- **NEW**: Configurable minimum training/validation sizes
- **NEW**: Rolling cross-validation with 3+ folds
- Maintain fight chronology integrity

### 5. Model Training (`train.py`)
- **NEW**: Two-stage market-aware stacking approach
  - **Stage A**: Skill-only models (no odds) for out-of-fold predictions
  - **Stage B**: Market models using skill predictions + odds features
- Train XGBoost models with robust hyperparameters
- Implement early stopping and cross-validation
- Handle class imbalance and feature selection
- Save trained models and metadata

### 6. Model Calibration (`calibrate.py` + `calibration.py`)
- **Basic Calibration**: Platt scaling and isotonic regression
- **NEW**: Side×Band Calibration
  - Per-side calibration (Red vs Blue fighters)
  - Market probability band-specific calibration
  - Isotonic regression within odds regimes
- Validate calibration on holdout sets
- Generate calibration plots

### 7. Evaluation (`evaluate.py`)
- Calculate comprehensive metrics (accuracy, precision, recall, F1)
- Generate confusion matrices and ROC curves
- Implement betting-specific metrics (profit/loss, ROI)
- Create detailed evaluation reports
- **NEW**: Reliability plots and calibration analysis

### 8. Advanced Backtesting (`backtest_ev.py`, `backtest_sweep.py`)
- **Basic EV Backtesting**: Single threshold analysis
- **NEW**: Multi-threshold EV sweep analysis
- **NEW**: Side filtering (Red/Blue/both)
- **NEW**: Probability band filtering
- **NEW**: Edge capping for overconfident predictions
- **NEW**: Kelly criterion staking with multipliers
- **NEW**: Calibrated probability integration

### 9. Market Analysis (`utils.py`)
- **NEW**: Vig-free odds processing
- **NEW**: American odds to implied probability conversion
- **NEW**: Overround calculation and removal
- **NEW**: Fair probability computation

## 🎯 Current System Performance

### Model Performance (Market Stacking):
- **ROC AUC**: 0.683 average across 3 CV folds
- **LogLoss**: 0.629 average
- **Brier Score**: 0.220 average
- **Accuracy**: 63.8% average

### Advanced Strategy Results (Red-Only + Calibration + Edge Capping):
- **Flat Stake**: 3.2% ROI (2% threshold) to 4.9% ROI (3% threshold)
- **Kelly Betting**: 10.0% ROI (2% threshold) to 10.1% ROI (3% threshold)
- **Optimal Configuration**: Red fighters only, 3% edge threshold, Kelly 0.5x multiplier

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository>
cd UFC_MLpredictor
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Data Pipeline
```bash
# Clean raw data
python scripts/make_clean.py

# Build features (with ELO)
python scripts/make_features.py

# Create time-based splits
python scripts/make_splits.py --n-folds 3 --group-by month --min-train-size 2500 --min-valid-size 800
```

### 3. Model Training
```bash
# Train skill-only models
python scripts/train_skill.py --n-folds 3

# Train market-aware stacking
python scripts/train_market.py --n-folds 3
```

### 4. Advanced Evaluation
```bash
# Export CV predictions for backtesting
python scripts/export_cv_predictions.py

# Train side×band calibrators
python scripts/train_sideband_calibrators.py

# Run comprehensive EV backtest sweep
python scripts/backtest_sweep.py --thresholds 0.02,0.03,0.05,0.07 --side_filter red --min_fairp_red 0.55 --max_fairp_red 0.90 --max_edge 0.20 --calibrator data/reports/calibration/sideband_calibrator.joblib
```

## 🎲 Advanced Betting Strategies

### Optimal Configuration (Based on Backtesting):
```bash
python scripts/backtest_sweep.py \
  --thresholds 0.03 \
  --kelly \
  --kelly_multipliers 0.5 \
  --side_filter red \
  --min_fairp_red 0.45 \
  --max_fairp_red 0.95 \
  --max_edge 0.20 \
  --calibrator data/reports/calibration/sideband_calibrator.joblib
```

**Expected Results**: 10.1% ROI with Kelly staking, manageable bet volume

### Strategy Components:
1. **Side Filtering**: Focus on Red fighters (historically more reliable)
2. **Edge Threshold**: 3% minimum edge for quality bets
3. **Probability Bands**: 45%-95% fair probability range
4. **Edge Capping**: 20% maximum edge (remove overconfident predictions)
5. **Kelly Staking**: 0.5x multiplier with 25% cap
6. **Calibration**: Side×band probability recalibration

## 🔧 Configuration Options

### Data Splitting:
- `--n-folds`: Number of CV folds (default: 3)
- `--group-by`: Grouping strategy (month/date)
- `--min-train-size`: Minimum training set size (default: 2500)
- `--min-valid-size`: Minimum validation set size (default: 800)

### Backtesting:
- `--thresholds`: Edge thresholds to test (e.g., "0.02,0.03,0.05")
- `--side_filter`: Betting side (both/red/blue)
- `--min_fairp_red/blue`: Probability band filters
- `--max_edge`: Maximum edge cap (default: 0.30)
- `--calibrator`: Path to side×band calibrator
- `--kelly`: Enable Kelly criterion staking
- `--kelly_multipliers`: Kelly multipliers (e.g., "1.0,0.5")

## 📊 Output Files

### Model Artifacts:
- `data/models/skill_fold_{k}.joblib`: Skill-only models per fold
- `data/models/market_fold_{k}.joblib`: Market stacking models per fold

### Evaluation Results:
- `data/reports/skill_cv_metrics.json`: Skill model performance
- `data/reports/market_cv_metrics.json`: Market stacking performance
- `data/reports/calibration/sideband_calibrator.joblib`: Advanced calibrator

### Backtesting Results:
- `data/reports/backtests/sweeps/`: Multi-threshold sweep results
- `data/reports/backtests/sweeps/summary.json`: Comprehensive sweep summary
- `data/reports/backtests/sweeps/overall_flat.csv`: Flat stake results
- `data/reports/backtests/sweeps/overall_kelly.csv`: Kelly staking results

## 🎯 Key Features

### Market-Aware Stacking:
- **Stage A**: Skill-only models generate out-of-fold predictions
- **Stage B**: Market models combine skill predictions with odds features
- **Vig-Free Integration**: Fair probabilities and implied odds
- **No Data Leakage**: Strict temporal splitting

### Advanced Calibration:
- **Side-Specific**: Separate calibration for Red vs Blue fighters
- **Band-Specific**: Calibration within market probability ranges
- **Isotonic Regression**: Non-parametric probability mapping

### Sophisticated Backtesting:
- **Multi-Threshold Analysis**: Systematic edge threshold evaluation
- **Kelly Criterion**: Optimal bet sizing with risk management
- **Advanced Filtering**: Side, probability band, and edge capping
- **Calibrated Probabilities**: Integration with side×band calibration

## 🔮 Future Enhancements

- **Real-time Predictions**: Live fight prediction API
- **Feature Importance Analysis**: SHAP values and model interpretability
- **Ensemble Methods**: Additional stacking layers and model combinations
- **Market Integration**: Real-time odds fetching and analysis
- **Risk Management**: Advanced Kelly variants and position sizing

## 📚 Technical Details

### Dependencies:
- **Core ML**: pandas, numpy, scikit-learn, xgboost
- **Web Framework**: FastAPI, uvicorn
- **Visualization**: matplotlib, shap
- **Utilities**: joblib, pydantic, optuna

### Model Architecture:
- **Base Models**: XGBoost with robust hyperparameters
- **Stacking**: Two-stage ensemble with market features
- **Calibration**: Isotonic regression with side×band specificity
- **Validation**: Time-based cross-validation with expanding windows

This system represents a state-of-the-art approach to sports prediction, combining traditional sports analytics with advanced machine learning techniques and sophisticated betting analysis tools.
