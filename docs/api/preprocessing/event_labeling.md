# `preprocessing.event_labeling`

The `horizon_volatility` event column stores the estimated volatility of returns over the configured bar horizon.
Horizontal barrier distances are this volatility multiplied by the profit-taking and stop-loss multipliers.
It replaces `target_return`; regenerate existing event artifacts from the event-labeling stage before using downstream workflows.

Labeled output places `event_end`, `vertical_barrier`, `horizon_volatility`, `raw_return`, and `direction_label`
immediately after `symbol` and `event_start`, followed by partition metadata and model features.

::: preprocessing.event_labeling
