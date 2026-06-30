# Financial Machine Learning

## Project Description

This project is a system for financial research and automated trading. It is heavily inspired by Marcos de Prado's
Advances in Financial Machine Learning, and it tries to apply those ideas in a practical software system. It also
provides template strategies to test.

While most projects that follow Advances in Financial Machine Learning provide a list of core functions related to
market data, this project aims to be a more complete software framework for the entire investment research workflow by
including data preprocessing, feature analysis across not only market data, but also fundamental data, analytic data,
and alternative data.

Also, most projects do not cover financial research and automated trading in the same project, so it often leads to
re-implementing the same algorithms in different software. This project tries to reduce that duplication by providing a
shared foundation that can be used across experiments and execution systems.

## Directory Structure

The project is organized as follows:

```text
  .
  |-- src/                      # reusable Python package code
  |   |-- data_preprocessing/   # data fetching, bars, labeling, sampling, feature engineering
  |   |-- strategy_modeling/    # validation, feature importance, tuning, ensembles
  |   |-- strategy_research/    # strategy research components
  |   |-- model_backtesting/    # backtesting components
  |   `-- live_trading/         # live trading components
  |-- notebooks/                # executable research notebooks by topic
  |-- docs/                     # documentation, architecture notes, and API references
  `-- data/                     # local datasets and generated artifacts
```

## Installation

Create a virtual environment, activate it, and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Tasks

- Single-variate ARIMA, GARCH
- Multi-variate ARIMA, GARCH
- VaR, PCA, State-Space, Structural Breaks, Entropy
- Earnings, Book Value, Cash Flow, Accruals, ROE, P/E, P/B
- Liquidity, Leverage, Profitability, Asset Turnover, Cash Flow, Growth
- Forecasted Earnings, Forecasted ROE, Residual Income, DCF, EV, P/E, P/B
- Economic Value Added, Earnings Quality, Accrual Quality, Cash Conversion, Accounting Distortions
- Expected Return, Volatility, Beta, Alpha, Cost of Equity, Credit Risk, Default Risk
- Recommendation with other features: “Analyzing the Analysts: When Do Recommendations Add Value?”
- News Sentiment: “Predicting Returns With Text Data”
- CNN, LSTM: “DeepLOB: Deep Convolutional Neural Networks for Limit Order Books”
- Change to Finnhub
- Create a notebook that checks if adding different features gives better results
- Create a presentation that clearly explains the concepts
