# `backtesting.portfolio_management`

The fixed-universe API is `simulate_cross_sectional(events, observations, calibration, calendar, end,
settings, evaluation_partition)`. It consumes signals from the selected partition, event-keyed price calibration,
and strictly chronological quote/clock batches,
maintains per-symbol active events and candidate queues, and returns in-memory account, fill, closed-trade,
exposure, and exclusion tables. `candidate_snapshot` ranks latest events while sizing from all active signals
within each security. `PortfolioSettings` defines shared defaults.

The research notebooks persist only final statistics through
`portfolio_management.run_final_backtest`.

::: backtesting.portfolio_management
