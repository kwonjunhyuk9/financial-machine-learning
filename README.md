# Financial Machine Learning

## Project Description

This project is a system for financial research and automated trading. It is heavily inspired by Marcos de Prado's
Advances in Financial Machine Learning, and it tries to apply those ideas in a practical software system.

While most projects that follow Advances in Financial Machine Learning provide a list of core functions related to
market data, this project aims to be a more complete software framework for the entire investment research workflow by
including data preprocessing, feature analysis across not only market data, but also fundamental data, and alternative
data.

Also, most projects do not cover financial research and automated trading in the same project, so it often leads to
re-implementing the same algorithms in different software. This project tries to reduce that duplication by providing a
shared foundation that can be used across experiments and execution systems.

## Directory Structure

The project is organized as follows:

```text
  .
  |-- src/                      # reusable package code
  |   |-- data_preprocessing/   # data preprocessing components
  |   |-- strategy_modeling/    # strategy modeling components
  |   |-- model_backtesting/    # model backtesting components
  |   `-- live_trading/         # live trading components
  |-- notebooks/                # executable research notebooks
  |   |-- data_preprocessing/   # data preprocessing components
  |   |-- strategy_modeling/    # strategy modeling components
  |   |-- model_backtesting/    # model backtesting components
  |   `-- live_trading/         # live trading components
  |-- workflows/                # curated pipelines and workflows
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

- Add AGENTS, SKILLS, SUBAGENTS, and spec driven development
- Change to Finnhub and Kraken
- Check if concepts of a library are well implemented
- Modify C4 Diagram to fit the standards
- Add more recent ML models, such as CNN, LSTM
- If necessary, implement testing codes
- Run on a Mac Mini and present the results
- Create a presentation that clearly explains the concepts