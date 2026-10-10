# `modeling.model_workflow`

`build_primary_model_frame` selects composite `(symbol, event_start)` keys and returns a time-sorted MultiIndex frame. The explicit `model_kind` selects 49 Market or 50 Sentiment primary features; meta appends side/confidence. Both use common event keys and labels. Prediction writers preserve both key columns.
Primary and meta modeling share full-grid OOF selection, learning-curve, feature-importance, and evaluation helpers.
`run_model_selection_workflow` is the model-selection entry point; the unused baseline-comparison and selected-family tuning APIs have been removed.
`run_model_selection_workflow` returns all candidate scores in `tuning`, each family winner in
`estimators`, `oof_predictions`, `configurations`, and `comparison`, and the overall winner through
`selected_name` and the `final_*` fields. It ranks pooled weighted OOF log loss then F1 for primary,
and F1 then log loss for meta; exact ties retain family and grid order.
The four notebooks use Purged Validation, Ensemble Methods, Learning Curves, Feature Importance,
and Model Evaluation sections. Diagnostics use the three family winners: 10 learning-curve points
in one row, MDI/MDA/SFI in one row per family, confusion matrices in one row, and overlaid PR/ROC
curves side by side.

`finalize_primary_model` and `finalize_meta_model` fit on development data, predict holdout events, and save the existing prediction tables and fitted model artifacts. Evaluation displays remain in the notebooks.

::: modeling.model_workflow
