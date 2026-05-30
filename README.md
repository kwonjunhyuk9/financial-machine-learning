# Financial Machine Learning

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
  |-- src/                      # reusable Python package code
  |   |-- data_preprocessing/   # data fetching, bars, labeling, sampling, feature engineering
  |   |-- feature_analysis/     # validation, feature importance, tuning, ensembles
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
