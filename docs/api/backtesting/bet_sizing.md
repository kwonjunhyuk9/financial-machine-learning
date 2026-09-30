# `backtesting.bet_sizing`

`event_bet_signals` transforms each event probability. `average_symbol_targets` accepts only currently active events, includes zero pass signals, averages per symbol, then discretizes and divides by `2K`; it never reads future event ends.

::: backtesting.bet_sizing
