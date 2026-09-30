# `backtesting.strategy_validation`

CPCV uses the same distinct-time grouping as purged k-fold and preserves composite event identity. Price calibration is
fit on each split's training observations, and the notebook assembles each set of out-of-sample forecasts into a complete
development path for bet sizing, cross-sectional account simulation, and backtest statistics. Path results remain
in memory and do not replace the sealed-holdout result.

::: backtesting.strategy_validation
