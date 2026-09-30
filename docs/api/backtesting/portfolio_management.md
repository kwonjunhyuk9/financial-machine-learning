# `backtesting.portfolio_management`

The fixed-universe API is `simulate_cross_sectional(events, observations, calibration, calendar, end,
settings)`. It consumes composite-keyed holdout signals and strictly chronological quote/clock batches,
maintains per-symbol active events and candidate queues, and returns in-memory account, fill, closed-trade,
exposure, and exclusion tables. `candidate_snapshot` ranks latest events while sizing from all active signals
within each security. `PortfolioSettings` defines shared defaults.

Legacy single-security helpers remain callable. The research notebooks use the multi-security interface and
persist only final statistics through `portfolio_management.run_final_backtest`.

Accounting assumptions are specified in the
[portfolio evaluation decision](../../decisions.md#self-financing-portfolio-evaluation).

::: backtesting.portfolio_management
