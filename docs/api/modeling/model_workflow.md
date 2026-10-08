# `modeling.model_workflow`

`build_primary_model_frame` selects composite `(symbol, event_start)` keys and returns a time-sorted MultiIndex frame. The explicit `model_kind` selects 49 Market or 50 Sentiment primary features; meta appends side/confidence. Both use common event keys and labels. Prediction writers preserve both key columns.
Primary and meta modeling share candidate selection, tuning, OOF prediction, learning-curve, feature-importance, and evaluation helpers.

`finalize_primary_model` and `finalize_meta_model` fit on development data, predict holdout events, and save the existing prediction tables and fitted model artifacts. Evaluation displays remain in the notebooks.

::: modeling.model_workflow
