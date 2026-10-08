# `backtesting.strategy_validation`

CPCV uses the same distinct-time grouping as purged k-fold and preserves composite event identity. Price calibration is
fit on each split's training observations, and the notebook assembles each set of out-of-sample forecasts into a complete
development path for bet sizing, cross-sectional account simulation, and backtest statistics. Path results remain
in memory and do not replace the sealed-holdout result.

`generate_cpcv_predictions` refits the frozen primary/meta configurations within each split, using inner primary OOF predictions for meta training, and saves the test-only prediction table. Inner fold and embargo settings are supplied by the notebook.

::: backtesting.strategy_validation
