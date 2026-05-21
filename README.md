# Multi Asset Investing

## Project Description

This project is a system for financial research and automated trading. It is heavily inspired by Marcos de Prado's
Advances in Financial Machine Learning, and it tries to apply those ideas in a practical software system. It also
provides basic strategies to test out, which were inspired by Lasse Heje Pedersen's Efficiently Inefficient.

While most projects that follow Advances in Financial Machine Learning provide a list of core functions and solutions to
problems given in the book, this project aims to be a more complete software framework for the entire investment
research workflow by including data preprocessing, feature analysis, strategy templates, backtesting, and execution
features.

Also, most projects do not cover financial research and automated trading in the same project, so it often leads to
re-implementing the same algorithms in different software. This project tries to reduce that duplication by providing a
shared foundation that can be used across experiments and live trading systems.

## Directory Structure

The project is organized as follows:

```text
.
|-- data_preprocessing/
|-- feature_analysis/
|-- strategy_research/
|-- model_backtesting/
|-- live_trading/
|-- REQUIREMENTS.md
|-- ARCHITECTURE.md
`-- DECISIONS.md
```

## Installation

Create a virtual environment and install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy
```

## Usage

### Data Preprocessing

- Import market, fundamental, analytic, alternative data.
- Preprocess and align the provided data to prevent leakage.
- Place the data files in the data folder.


### Feature Analysis

- Research and discover relevant features by conducting tests with Jupyter Notebooks.
- Convert findings into python files in order to link to strategies.
- Place these files in the feature_analysis folder.

### Strategy Research

- Research and discover relevant strategies by conducting tests with Jupyter Notebooks.
- Convert findings into python files in order to link to live trading.
- Place these files in the strategy_research folder.


### Model Backtesting

- Backtest strategies on historical or synthetic data.

```bash
python -m model_backtesting backtest \
  --asset btcusdt \
  --strategy trend_following \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --data-source historical
```

### Live Trading

- Choose the asset and the strategy, and run in locally in real time.

```bash
python -m live_trading trade \
  --asset btcusdt \
  --strategy trend_following \
  --broker binance \
  --mode paper
```
