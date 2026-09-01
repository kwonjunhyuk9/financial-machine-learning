# Decision Record

## 1. Decision Log

### 1.1 Language

Decision:

- Use Python 3.11 for source modules and notebooks.

Reason:

- It satisfies the project's declared Python requirement and supports the typing and third-party libraries used across
  the research workflows.
- Its standard library includes the filesystem interfaces used by the project's local workflows.

### 1.2 Database

Decision:

- Use Parquet for research datasets and intermediate analytical results that are reused across notebooks.
- Display terminal diagnostics and evaluation results in the notebook that produces them instead of persisting duplicate
  Parquet files.

Reason:

- Parquet keeps market, news, model-input, and reusable backtest data portable and efficient for notebook and batch
  workflows.
- Keeping terminal diagnostics in notebooks avoids maintaining files that have no downstream consumer while preserving
  the analysis alongside its code and visual output.

### 1.3 Data Science

Decision:

- Use pandas and NumPy for tabular and numerical data processing.
- Use SciPy and statsmodels for statistical distributions and time-series diagnostics.
- Use scikit-learn for classifiers, cross-validation, hyperparameter tuning, and evaluation metrics.
- Use Matplotlib for research visualizations.
- Use FinanceToolkit for market technical indicators.

Reason:

- pandas and NumPy provide the tabular and array operations used for features, labels, and backtest paths.
- SciPy and statsmodels provide the distribution functions and stationarity diagnostics used for bet sizing and
  fractional differentiation.
- scikit-learn provides the estimator, purged cross-validation, tuning, and metric interfaces used by strategy modeling.
- Matplotlib keeps feature, validation, and backtest plots in the same Python workflow as the analyses that produce
  them.
- FinanceToolkit provides technical indicators from locally prepared market bars.

### 1.4 External Service Clients

Decision:

- Use alpaca-py to retrieve Alpaca market data and news.

Reason:

- alpaca-py provides the historical trade and news clients used by the research workflow.

### 1.5 Documentation

Decision:

- Use MkDocs for the documentation site, Material for MkDocs for presentation, and mkdocstrings for API reference pages.

Reason:

- The Markdown-based stack keeps project documentation lightweight while generating API references from Python
  docstrings and supporting GitHub Pages deployment.
- Material for MkDocs provides the site navigation, while mkdocstrings keeps API documentation aligned with source
  docstrings.

### 1.6 Logging

Decision:

- Use loguru for data-fetching and modeling diagnostics.

Reason:

- loguru provides lightweight levels and formatted messages without a larger logging configuration layer.
- The data-fetching and modeling modules use the same logging interface for diagnostics.

### 1.7 Testing

Decision:

- Use pytest for automated tests of preprocessing, modeling, and backtesting modules.

Reason:

- pytest provides concise assertions and fixture support for the project's function and class-based modules.
- Its test discovery supports the repository's separate preprocessing, modeling, and backtesting test directories.

### 1.8 CI/CD

Decision:

- Use GitHub Actions to build and deploy the MkDocs site.

Reason:

- The workflow builds documentation on pushes to the main branch and deploys the generated static site to GitHub Pages.
- Manual workflow dispatch supports documentation rebuilds without requiring a source-code change.
