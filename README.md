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
  |-- src/                          # reusable packages for research and execution
  |   |-- data_preprocessing/   
  |   |-- strategy_modeling/    
  |   |-- model_backtesting/    
  |   `-- live_trading/        
  |-- notebooks/                    # interactive examples for the corresponding source packages
  |   |-- data_preprocessing/   
  |   |-- strategy_modeling/    
  |   |-- model_backtesting/    
  |   `-- live_trading/         
  |-- workflows/                    # integrated strategy research workflows
  |   |-- directional_strategies/ 
  |   |-- feature_analysis/     
  |   |-- fundamental_strategies/ 
  |   `-- relative_value_strategies/ 
  |-- data/                         # local research and runtime storage
  |   |-- research_data/       
  |   |-- model_artifact/       
  |   |-- backtest_results/    
  |   `-- trading_state/        
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
