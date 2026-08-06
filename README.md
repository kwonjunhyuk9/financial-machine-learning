# Financial Machine Learning

## Project Description

This project is a system for financial machine learning research. It is heavily inspired by Marcos de Prado's
Advances in Financial Machine Learning, and it tries to apply those ideas in a practical software system.

While most projects that follow Advances in Financial Machine Learning provide a list of core functions related to
market data, this project aims to be a more complete software framework for the entire investment research workflow by
including data preprocessing, feature analysis across not only market data, but also fundamental data, and alternative
data.

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
