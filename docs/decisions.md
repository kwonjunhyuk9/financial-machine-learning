# Decision Record

## 1. Decision Log

### 1.1 Language

Decision:

- Use Python 3.11 for source modules and notebooks.

Reason:

- It satisfies the project's declared Python requirement and supports the typing and third-party libraries used across the research and live-trading workflows.
- Its standard library includes the filesystem and SQLite interfaces used by the project's local workflows.

### 1.2 Database

Decision:

- Use Parquet for research datasets and analytical results.
- Use SQLite for live-trading orders and executions.

Reason:

- Parquet keeps market, fundamental, and news data portable and efficient for notebook and batch workflows.
- SQLite provides a lightweight local store with transactional updates for order and execution state.

### 1.3 Communication

Decision:

- Use REST for historical data retrieval, account queries, and exchange actions.
- Reserve WebSocket communication for streaming market and execution updates as the live-trading workflow expands.

Reason:

- REST matches the current request-response interactions with Alpaca and Kraken.
- WebSocket avoids repeated polling when the live runtime needs timely price or execution updates.

### 1.4 Data Science

Decision:

- Use pandas and NumPy for tabular and numerical data processing.
- Use SciPy and statsmodels for statistical distributions and time-series diagnostics.
- Use scikit-learn for classifiers, cross-validation, hyperparameter tuning, and evaluation metrics.
- Use Matplotlib for research visualizations.

Reason:

- pandas and NumPy provide the tabular and array operations used for features, labels, and backtest paths.
- SciPy and statsmodels provide the distribution functions and stationarity diagnostics used for bet sizing and fractional differentiation.
- scikit-learn provides the estimator, purged cross-validation, tuning, and metric interfaces used by strategy modeling.
- Matplotlib keeps feature, validation, and backtest plots in the same Python workflow as the analyses that produce them.

### 1.5 External Service Clients

Decision:

- Use FinanceToolkit with Financial Modeling Prep to retrieve fundamental data.
- Use alpaca-py to retrieve Alpaca market data and news.
- Use CCXT to retrieve Kraken account and market data and submit orders.

Reason:

- FinanceToolkit standardizes Financial Modeling Prep statements.
- alpaca-py provides the historical trade and news clients used by the research workflow.
- CCXT provides a consistent exchange-client interface for Kraken balance, ticker, and order operations.

### 1.6 Documentation

Decision:

- Use MkDocs for the documentation site, Material for MkDocs for presentation, and mkdocstrings for API reference pages.

Reason:

- The Markdown-based stack keeps project documentation lightweight while generating API references from Python docstrings and supporting GitHub Pages deployment.
- Material for MkDocs provides the site navigation, while mkdocstrings keeps API documentation aligned with source docstrings.

### 1.7 Logging

Decision:

- Use loguru for data-fetching and live-trading diagnostics.

Reason:

- loguru provides lightweight levels and formatted messages without a larger logging configuration layer.
- The fetcher and live-trading modules use the same logging interface for operational diagnostics.

### 1.8 Testing

Decision:

- Use pytest for automated tests of preprocessing, modeling, backtesting, and live-trading modules.

Reason:

- pytest provides concise assertions and fixture support for the project's function and class-based modules.
- Its test discovery supports the repository's separate preprocessing, modeling, backtesting, and live-trading test directories.

### 1.9 CI/CD

Decision:

- Use GitHub Actions to build and deploy the MkDocs site.

Reason:

- The workflow builds documentation on pushes to the main branch and deploys the generated static site to GitHub Pages.
- Manual workflow dispatch supports documentation rebuilds without requiring a source-code change.

### 1.10 Choosing the Data Source

Decision:

- Use Financial Modeling Prep, accessed with FinanceToolkit, for fundamental data.
- Use Alpaca for equity and cryptocurrency market data.
- Use Alpaca News API for Benzinga news content.

Reason:

- Financial Modeling Prep provides standardized financial statements and market data for fundamental factors.
- Alpaca provides the historical trade data used by the market-data workflow.
- Alpaca News API provides Benzinga content through the same integration and credentials used for market data.
