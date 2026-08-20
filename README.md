# Financial Machine Learning

## Project Description

This project is a system for financial machine learning research. It is heavily inspired by Marcos de Prado's
Advances in Financial Machine Learning, and it tries to apply those ideas in a practical software system.

While most projects that follow Advances in Financial Machine Learning provide a list of core functions related to
market data, this project aims to be a more complete software framework for the entire investment research workflow by
including data preprocessing and feature analysis across market and alternative data.

## Directory Structure

The project is organized as follows:

```text
  .
  |-- src/                          # reusable packages for research
  |   |-- data_preprocessing/   
  |   |-- strategy_modeling/    
  |   `-- model_backtesting/
  |-- notebooks/                    # interactive examples for the corresponding source packages
  |   |-- data_preprocessing/   
  |   |-- strategy_modeling/    
  |   `-- model_backtesting/
  |-- data/                         # local research storage
  |   |-- research_data/       
  |   |-- model_artifact/       
  |   `-- backtest_results/
  |-- docs/                         # documentation and API references
  `-- tests/                        # automated tests for source packages
```

## Installation

Create a virtual environment, activate it, and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Notebook Execution Order

These 22 notebooks form an AAPL 2025 research workflow. Run each notebook from its containing directory in a fresh
kernel. The first six notebooks reuse existing Parquet files and require Alpaca or FinBERT access only when an input is
missing or the stored sentiment rows no longer match the downloaded news.

| Order | Notebook |
|---:|---|
| 1 | `notebooks/data_preprocessing/market_data.ipynb` |
| 2 | `notebooks/data_preprocessing/alternative_data.ipynb` |
| 3 | `notebooks/data_preprocessing/market_structured_bars.ipynb` |
| 4 | `notebooks/data_preprocessing/market_differentiated_bars.ipynb` |
| 5 | `notebooks/data_preprocessing/market_technical_indicators.ipynb` |
| 6 | `notebooks/data_preprocessing/alternative_sentiment_scores.ipynb` |
| 7 | `notebooks/data_preprocessing/train_test_split.ipynb` |
| 8 | `notebooks/data_preprocessing/event_labeling.ipynb` |
| 9 | `notebooks/data_preprocessing/event_weights.ipynb` |
| 10 | `notebooks/data_preprocessing/prepare_the_data.ipynb` |
| 11 | `notebooks/strategy_modeling/cross_validation.ipynb` |
| 12 | `notebooks/strategy_modeling/ensemble_methods.ipynb` |
| 13 | `notebooks/strategy_modeling/feature_importance.ipynb` |
| 14 | `notebooks/strategy_modeling/hyperparameter_tuning.ipynb` |
| 15 | `notebooks/strategy_modeling/primary_model.ipynb` |
| 16 | `notebooks/strategy_modeling/meta_model.ipynb` |
| 17 | `notebooks/model_backtesting/bet_sizing.ipynb` |
| 18 | `notebooks/model_backtesting/backtest_validation.ipynb` |
| 19 | `notebooks/model_backtesting/backtest_overfitting.ipynb` |
| 20 | `notebooks/model_backtesting/backtest_synthetic.ipynb` |
| 21 | `notebooks/model_backtesting/backtest_statistics.ipynb` |
| 22 | `notebooks/model_backtesting/strategy_risk.ipynb` |
