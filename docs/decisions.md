# Decision Record

## Self-Financing Portfolio Evaluation

Decision:

- Begin independent primary-only and meta-filtered holdout accounts with USD 100,000 each and no development
  holdings, external flows, or withdrawals. Reinvest gains and losses at every event-start/end boundary.
- Average signed primary sides and signed probability-derived meta signals over active intervals `[start, end)`.
  Include zero meta signals in the mean. Discretize only the aggregated meta signal at 0.10; primary signals retain
  their mean. Dynamic forecast sizing remains a separate diagnostic, not the traded strategy.
- Generate final targets once through `bet_sizing.build_target_positions`, reusing `get_signal` for signed meta
  sizing. Portfolio accounting consumes these targets without repeating probability transformations. Keep the
  unsigned legacy event-artifact path separate for existing consumers; it is not the portfolio performance input.
- Mark accounts at raw AAPL trade prices. For duplicate timestamps retain the last price. Require exact boundary
  prices and agreement with saved raw event returns; do not interpolate missing boundaries.
- Execute once at each combined event boundary at its observed price, including boundaries where the target weight
  is unchanged. This is a zero-latency reference-price fill assumption. Hold share quantities fixed between boundaries
  and close at the final boundary. Future exit times determine realized lifetimes, not entry sizes.
- Allow fractional shares. Size against post-cost equity with target weights in `[-1, 1]`. Record actual traded
  share notional; a reversal closes the old position and opens a new one. Charge buys and sells separate configurable
  proportional one-way broker fees and slippage, defaulting to zero. For the reported backtest, apply 1 bp of each
  with no taxes or other costs. Cash interest, financing, borrow fees, and dividends are zero. Fail on nonpositive AUM.
- Keep pre/post-trade marks in the account equity path. Initial pre-entry AUM is the return baseline, so opening
  costs cannot disappear. Maximum dollar exposure includes pre- and post-trade holdings. Intermediate price drift
  can move realized leverage outside target bounds without forcing unscheduled rebalancing.
- Sample daily metrics at UTC calendar closes, carrying state over missing days and retaining first/last partial
  days. AUM is the arithmetic mean of those marks; leverage is mean absolute dollar exposure divided by mean AUM.
  Turnover is total two-way traded notional divided by elapsed calendar years and this same mean AUM.
- Derive daily account and underlying returns on the same intervals. Compound performance uses final/initial AUM;
  CAGR uses exact elapsed time. Daily Sharpe and PSR annualize at 365.25; subtract the effective 3% risk-free return
  on the entire account over each actual interval, including weekends. The annualized PSR benchmark remains one.
- Treat a flat-to-flat or same-direction episode as one trade. Same-side resizing remains within the episode;
  flips split costs by closing/opening notional. New reversal trades use AUM after the old exit cost as their entry
  capital. Sum marked PnL and assigned costs per trade; divide net PnL by entry capital for trade returns. Trade count,
  equal-weighted holding period, long ratio, hit statistics, and HHI use these episodes. Bet frequency uses the full
  account period, including flat time.
- Measure drawdown episodes on the full observed equity path from the last peak before a decline through recovery,
  or the final mark for unrecovered episodes. Report durations in years. No drawdown means zero maximum depth/duration
  and undefined episode percentiles. Undefined trade metrics and zero cost denominators remain NaN.
- Persist broker fees, slippage, and their sum as execution cost in the reusable multi-strategy ledger from Bet Sizing.
  Keep existing event-return artifacts for classification and legacy consumers, but never compound their overlapping
  returns as account performance. Apply classification metrics to the original event labels, probabilities, and sample
  weights.

Reason:

- Aggregating concurrent exposures before accounting creates one capital budget per strategy. Tracking cash and shares
  makes reinvestment, transaction costs, dollar exposure, and all investment performance metrics mutually consistent.
## Event Weight Simplification

Decision:

- Follow the event-weight policy in [Requirements](requirements.md#21-preprocessing).

Reason:

- Purged cross-validation can train on events before and after validation, so a global recency preference does not
  represent the information available at each validation time.
- Average uniqueness was a diagnostic output unused by model fitting. Removing its calculation and persisted column
  simplifies the schema without changing return attribution's concurrency adjustment.

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
