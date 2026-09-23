# System Architecture

## 1. Architecture Overview

### 1.1 Technology Stack

| Category                 | Technology                                                  |
|--------------------------|-------------------------------------------------------------|
| Language                 | Python 3.11                                                 |
| Database                 | Parquet                                                     |
| Data Science             | Pandas, NumPy, SciPy, scikit-learn, statsmodels, Matplotlib, transformers, PyTorch, FinanceToolkit |
| External Service Clients | alpaca-py                                                    |
| Documentation            | MkDocs                                                      |
| Logging                  | loguru                                                      |
| Testing                  | pytest                                                      |
| CI/CD                    | GitHub Actions                                              |

## 2. Architecture Diagrams

### 2.1 Context Diagram

```mermaid
flowchart TD
    dp["Preprocessor<br/>[Person]"]
    sm["Modeler<br/>[Person]"]
    mb["Backtester<br/>[Person]"]
    alpaca["Alpaca API<br/>[External System]"]
    system["Financial Machine Learning<br/>[Software System]"]
    dp -->|" prepares data and signals "| system
    sm -->|" develops models "| system
    mb -->|" assesses strategies "| system
    system -->|" fetches market and alternative data "| alpaca
```

### 2.2 Container Diagram

```mermaid
flowchart TD
    dp_user["Preprocessor<br/>[Person]"]
    sm_user["Modeler<br/>[Person]"]
    mb_user["Backtester<br/>[Person]"]
    alpaca["Alpaca API<br/>[External System]"]

    subgraph system["Financial Machine Learning [Software System]"]
        preprocessing_workspace["Preprocessing Workspace<br/>[Container: Jupyter notebooks]"]
        data_store[("Research Data Store<br/>[Container: Parquet files]")]
        modeling_workspace["Modeling Workspace<br/>[Container: Jupyter notebooks]"]
        model_store[("Model Artifact Store<br/>[Container: Joblib files]")]
        backtesting_workspace["Backtesting Workspace<br/>[Container: Jupyter notebooks]"]
        result_store[("Reusable Backtest Data Store<br/>[Container: Parquet files]")]
    end

    dp_user -->|" creates preparation notebooks "| preprocessing_workspace
    sm_user -->|" creates strategy workflows "| modeling_workspace
    mb_user -->|" creates backtest analyses "| backtesting_workspace
    alpaca -->|" provides market and alternative data "| preprocessing_workspace
    preprocessing_workspace -->|" writes prepared datasets "| data_store
    data_store -->|" provides features and labels "| modeling_workspace
    modeling_workspace -->|" writes model artifacts "| model_store
    data_store -->|" provides backtest data "| backtesting_workspace
    model_store -->|" provides candidate models "| backtesting_workspace
    backtesting_workspace -->|" writes reusable strategy returns and paths "| result_store
```

### 2.3 Component Diagram

```mermaid
flowchart TD
    alpaca["Alpaca API<br/>[External System]"]

    subgraph system["Financial Machine Learning [Software System]"]
        data_store[("Research Data Store<br/>[Container: Parquet files]")]
        model_store[("Model Artifact Store<br/>[Container: Joblib files]")]
        result_store[("Reusable Backtest Data Store<br/>[Container: Parquet files]")]

        subgraph preprocessing_workspace["Preprocessing Workspace [Container: Jupyter notebooks]"]
            fetch_data["Fetch Data<br/>[Component: Python module]"]
            prepare_data["Prepare Features<br/>[Component: Python module]"]
            split_events["Build Candidate Schema and Split Events<br/>[Component: Python module]"]
            label_events["Label and Weight Events<br/>[Component: Python module]"]
            clean_events["Explore and Clean Weighted Events<br/>[Component: Python module]"]
        end

        subgraph modeling_workspace["Modeling Workspace [Container: Jupyter notebooks]"]
            primary_model["Primary Model<br/>[Component: Python module]"]
            meta_model["Meta Model<br/>[Component: Python module]"]
        end

        subgraph backtesting_workspace["Backtesting Workspace [Container: Jupyter notebooks]"]
            find_settings["Find Optimal Settings<br/>[Component: Python module]"]
            strategy_validation["Strategy Validation<br/>[Component: Python module]"]
            portfolio["Portfolio Accounting<br/>[Component: Python module]"]
            review_statistics["Review Statistics<br/>[Component: Python module]"]
        end

    end

    alpaca -->|" provides market and alternative data "| fetch_data
    fetch_data --> prepare_data
    prepare_data --> split_events
    split_events --> label_events
    label_events --> clean_events
    clean_events -->|" writes prepared events with inline partitions "| data_store
    data_store -->|" provides features and labels "| primary_model
    primary_model -->|" produces side and probabilities "| meta_model
    meta_model -->|" writes trained artifacts "| model_store
    data_store -->|" provides backtest data "| find_settings
    model_store -->|" provides candidate models "| find_settings
    find_settings -->|" provides selected sizing and rule settings "| strategy_validation
    find_settings -->|" writes reusable strategy returns "| result_store
    find_settings -->|" bet_sizing.build_target_positions: final signed positions "| portfolio
    data_store -->|" exact raw trade prices "| portfolio
    portfolio -->|" writes portfolio_ledger.parquet "| result_store
    strategy_validation -->|" writes reusable path returns "| result_store
    result_store -->|" account ledger and event classification inputs "| review_statistics
```
