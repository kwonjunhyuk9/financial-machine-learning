# `modeling.model_workflow`

`build_primary_model_frame` selects composite `(symbol, event_start)` keys and returns a time-sorted MultiIndex frame. Primary features use the common 50-column schema; meta appends side/confidence. Prediction writers preserve both key columns.
Primary and meta modeling share the same learning-curve, feature-importance, and evaluation helpers.

::: modeling.model_workflow
