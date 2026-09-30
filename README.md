# Financial Machine Learning

## Project Description

This project is a system for financial machine learning research. It is heavily inspired by Marcos de Prado's Advances
in Financial Machine Learning, and it tries to apply those ideas in a practical software system.

Unlike existing projects that mainly provide core market data functions inspired by Advances in Financial Machine
Learning, this project aims to provide a comprehensive framework for the entire investment research workflow. It
supports both market data and alternative data to enable a broader range of financial research applications.

## Directory Structure

The project is organized as follows:

```text
  |-- src/                          # Production code for financial research
  |-- notebooks/                    # Reproducible research experiments
  |-- tests/                        # Automated testing and validation
  |-- data/                         # Datasets, model artifacts, and results
  `-- docs/                         # Project documentation
```

## Installation

Create a virtual environment, activate it, and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Strategies

| Aspect          | Synchronous Cross-Sectional Sentiment Long-Short                                                                                                                              | Asynchronous Cross-Sectional Sentiment Long-Short                                                                                                                                                                             |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Selection       | Go long the top 20% and short the bottom 20% of securities by sentiment each day.                                                                                             | Maintain up to `K` eligible long positions and `K` eligible short positions.                                                                                                                                                  |
| Entry           | Before 6:00 a.m.: enter at the same-day open.<br>From 6:00 a.m. through 4:00 p.m.: enter at the same-day close.<br>After 4:00 p.m.: enter at the next-day open.               | When an eligible long or short position closes, select the highest-priority long or lowest-priority short candidate from the corresponding queue. Apply time decay and a limit price when determining priority and execution. |
| Exit            | Before 6:00 a.m.: liquidate at the same-day close.<br>From 6:00 a.m. through 4:00 p.m.: liquidate at the next-day close.<br>After 4:00 p.m.: liquidate at the next-day close. | Close a position when one of its triple barriers is reached.                                                                                                                                                                  |

## Notebook Execution Order

Run each from its containing directory in a fresh kernel:

| Order | Notebook                                                     |
|------:|--------------------------------------------------------------|
|     1 | `notebooks/preprocessing/market_data.ipynb`                  |
|     2 | `notebooks/preprocessing/market_structured_bars.ipynb`       |
|     3 | `notebooks/preprocessing/market_differentiated_bars.ipynb`   |
|     4 | `notebooks/preprocessing/market_technical_indicators.ipynb`  |
|     5 | `notebooks/preprocessing/alternative_data.ipynb`             |
|     6 | `notebooks/preprocessing/alternative_sentiment_scores.ipynb` |
|     7 | `notebooks/preprocessing/train_test_split.ipynb`             |
|     8 | `notebooks/preprocessing/event_labeling.ipynb`               |
|     9 | `notebooks/preprocessing/event_weights.ipynb`                |
|    10 | `notebooks/preprocessing/prepare_the_data.ipynb`             |
|    11 | `notebooks/modeling/primary_model.ipynb`                     |
|    12 | `notebooks/modeling/meta_model.ipynb`                        |
|    13 | `notebooks/backtesting/strategy_validation.ipynb`            |
|    14 | `notebooks/backtesting/bet_sizing.ipynb`                     |
|    15 | `notebooks/backtesting/portfolio_management.ipynb`           |
|    16 | `notebooks/backtesting/backtest_statistics.ipynb`            |
