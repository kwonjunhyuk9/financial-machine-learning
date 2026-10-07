# `backtesting.portfolio_management`

The fixed-universe API is `simulate_cross_sectional(events, observations, calibration, calendar, end,
settings, evaluation_partition)`. It consumes signals from the selected partition, event-keyed price calibration,
and strictly chronological quote/clock batches,
maintains per-symbol active events and candidate queues, and returns in-memory account, fill, closed-trade,
exposure, and exclusion tables. `candidate_snapshot` ranks latest events while sizing from all active signals
within each security. `PortfolioSettings` defines shared defaults.

The research notebooks persist strategy statistics, account curves, and comparison metadata through
`portfolio_management.run_final_backtest`.

::: backtesting.portfolio_management

Scheduled strategies use `portfolio_management.simulate_synchronous` with independent book exposures and the same statistics
contract. `portfolio_management.simulate_strategy` selects execution by strategy and preserves common evaluation bounds.
The asynchronous research configuration uses `PortfolioSettings(k=10)`; synchronous books use `k=5` each. Legacy
unscoped API defaults remain available for existing callers.


Sentiment CPCV forecasts are generated and saved by the synchronous strategy-validation notebook. The asynchronous
notebook reads the same Parquet output; regenerate it after changing data, models, or validation settings.
