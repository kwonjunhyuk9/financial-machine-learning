# `backtesting.bet_sizing`

`event_bet_signals` transforms each event probability. `average_symbol_targets` accepts only currently active events, includes zero pass signals, averages per symbol, then discretizes and divides by `2K`; it never reads future event ends.

`build_target_positions` owns final signed holdout positions. It reuses `get_signal`
for meta sizing and passes the resulting targets to portfolio accounting.

::: backtesting.bet_sizing
