# `backtesting.portfolio`

This module consumes final target positions from `backtesting.bet_sizing` and
handles price validation, cash/share accounting, daily marks, and trade attribution.
It does not generate signals from model predictions.

Accounting assumptions are specified in the
[portfolio evaluation decision](../../decisions.md#self-financing-portfolio-evaluation).

::: backtesting.portfolio
