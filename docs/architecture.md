# System Architecture

## 1. Technology Stack

| Category      | Technology |
|---------------|------------|
| Language      | Python     |
| Research      | Jupyter    |
| Data Source   | Finnhub    |
| Data Storage  | Parquet    |
| Execution     | Kraken     |
| Documentation | MkDocs     |

## 2. Architecture Diagrams

### 2.1 Context Diagram

```mermaid
flowchart LR
    da[Data Analyst]
    sm[Strategy Modeler]
    mb[Model Backtester]
    lt[Live Trader]
    ext[(External Data)]
    broker[(Broker APIs)]
    sys[Financial Machine Learning]
    da --> sys
    sm --> sys
    mb --> sys
    lt --> sys
    ext --> sys
    sys --> broker
```

### 2.2 Container Diagram

```mermaid
flowchart LR
    ext[(External Data)]
    broker[(Broker APIs)]
    parquet[(Parquet Storage)]

    subgraph system[Financial Machine Learning]
        da[data_preprocessing]
        sm[strategy_modeling]
        bt[model_backtesting]
        lt[live_trading]
    end

    ext --> da
    da --> parquet
    parquet --> sm
    sm --> bt
    bt --> lt
    lt --> broker
```

### 2.3 Component Diagram

#### 2.3.1 data preprocessing

```mermaid
flowchart TD
    ext[(External Data)]
    parquet[(Parquet Storage)]

    subgraph dp[data_preprocessing]
        f1[fetch_market_data]
        f2[fetch_fundamental_data]
        f4[fetch_alternative_data]
        s1[financial_data_structures]
        s2[financial_data_labeling]
        s3[fractionally_differentiate_features]
        s4[sample_weights]
        s5[preprocess_fundamental_data]
        s7[preprocess_alternative_data]
    end

    ext --> f1
    ext --> f2
    ext --> f4
    f1 --> s1
    s1 --> s2
    s1 --> s3
    s2 --> s4
    f2 --> s5
    f4 --> s7
    s3 --> parquet
    s4 --> parquet
    s5 --> parquet
    s7 --> parquet
```

#### 2.3.2 strategy modeling

```mermaid
flowchart TD
    parquet[("Parquet Storage")]
    model_storage[("Model Storage")]

    subgraph sm["strategy_modeling"]
        events["Event Selection<br/>CUSUM filter or rebalance schedule<br/>output: t0, t1"]

        subgraph primary_model["Primary Model"]
            primary_ens["ensemble_methods"]
            primary_ht["hyperparameter_tuning"]
            primary_cv["cross_validation"]
            primary_fi["feature_importance"]
            primary_ens --> primary_ht
            primary_ht --> primary_cv
            primary_cv --> primary_fi
        end

        subgraph meta_model["Meta Model"]
            meta_ens["ensemble_methods"]
            meta_ht["hyperparameter_tuning"]
            meta_cv["cross_validation"]
            meta_fi["feature_importance"]
            meta_ens --> meta_ht
            meta_ht --> meta_cv
            meta_cv --> meta_fi
        end

        tabular_artifacts["Tabular Artifacts<br/>events, features, labels, signals"]
        model_artifacts["Model Artifacts<br/>trained models, schemas, metrics"]
    end

    parquet --> events
    events --> primary_model
    primary_model --> meta_model
    meta_model --> tabular_artifacts
    meta_model --> model_artifacts
    tabular_artifacts --> parquet
    model_artifacts --> model_storage
```

#### 2.3.3 model backtesting

```mermaid
flowchart TD
    parquet[("Parquet Storage")]
    model_storage[("Model Storage")]
    results[("Backtest Results")]

    subgraph mb[model_backtesting]
        validation[backtest_validation]
        synthetic[backtest_synthetic]
        overfitting[backtest_overfitting]
        sizing[bet_sizing]
        statistics[backtest_statistics]
        risk[strategy_risk]
    end

    parquet --> validation
    model_storage --> validation
    parquet --> synthetic
    validation --> overfitting
    validation --> sizing
    sizing --> statistics
    synthetic --> statistics
    overfitting --> results
    statistics --> risk
    risk --> results
```

#### 2.3.4 live trading

```mermaid
flowchart TD
    parquet[("Parquet Storage")]
    model_storage[("Model Storage")]
    kraken[("Kraken API")]

    subgraph lt[live_trading]
        config[config]
        client[kraken_client]
        runner[live_runner]
        risk[risk_manager]
        orders_mgr[order_manager]
    end

    config --> client
    config --> runner
    config --> risk
    parquet --> runner
    model_storage --> runner
    client --> runner
    runner --> risk
    risk --> orders_mgr
    orders_mgr --> client
    client --> kraken
```
