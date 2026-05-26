# System Architecture

## 1. Technology Stack

| Category             | Technology            |
|----------------------|-----------------------|
| Language             | Python                |
| Research Environment | Jupyter Notebook      |
| Data Source          | Alpaca API            |
| Data Storage         | Parquet               |
| Execution            | Toss API, Binance API |
| Documentation        | MkDocs                |

## 2. Architecture Diagrams

### 2.1 Context Diagram

```mermaid
flowchart LR
    dc[Data Curator]
    fa[Feature Analyst]
    st[Strategist]
    bt[Backtester]
    dm[Deployment Manager]
    ext[(Market / Fundamental / Analytics / Alternative Data Sources)]
    broker[(Broker APIs)]
    sys[Multi-Asset Investing]
    dc --> sys
    fa --> sys
    st --> sys
    bt --> sys
    dm --> sys
    ext --> sys
    sys --> broker
```

### 2.2 Container Diagram

```mermaid
flowchart LR
    ext[(External Data Sources)]
    broker[(Broker APIs)]
    parquet[(Parquet Data Store)]

    subgraph system[Multi-Asset Investing]
        dp[data_preprocessing]
        feat[feature_analysis]
        alpha[alpha_models]
        backtest[model_backtest]
        live[live_trading]
    end

    ext --> dp
    dp --> parquet
    parquet --> feat
    dp --> feat
    feat --> alpha
    alpha --> backtest
    backtest --> alpha
    backtest --> live
    live --> broker
```

### 2.3 Component Diagram

#### 2.3.1 data_preprocessing

```mermaid
flowchart TD
    subgraph dp[data_preprocessing]
        f1[fetch_market_data]
        f2[fetch_fundamental_data]
        f3[fetch_analytic_data]
        f4[fetch_alternative_data]
        s1[financial_data_structures]
        s2[financial_data_labeling]
        s3[fractionally_differentiate_features]
        s4[sample_weights]
    end

    f1 --> s1
    f2 --> s1
    f3 --> s1
    f4 --> s1
    s1 --> s2
    s1 --> s3
    s2 --> s4
```

#### 2.3.2 feature_analysis

```mermaid
flowchart TD
    subgraph fa[feature_analysis]
        ens[ensemble_methods]
        ht[hyperparameter_tuning]
        cv[cross_validation]
        fi[feature_importance]
    end

    ens --> ht
    ens --> cv
    ht --> cv
    cv --> fi
```

#### 2.3.3 strategy_research

```mermaid
flowchart TD
    subgraph am[strategy_research]
        c1[component_1]
        c2[component_2]
        c3[component_3]
    end

    c1 --> c2
    c2 --> c3
```

#### 2.3.4 model_backtesting

```mermaid
flowchart TD
    subgraph mb[model_backtesting]
        c1[component_1]
        c2[component_2]
        c3[component_3]
    end

    c1 --> c2
    c2 --> c3
```

#### 2.3.5 live_trading

```mermaid
flowchart TD
    subgraph lt[live_trading]
        c1[component_1]
        c2[component_2]
        c3[component_3]
    end

    c1 --> c2
    c2 --> c3
```
