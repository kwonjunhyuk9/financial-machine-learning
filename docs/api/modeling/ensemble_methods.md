# `modeling.ensemble_methods`

The four modeling notebooks search `max_depth` over `[2, 5, 10]` for every family.
Factories leave `min_samples_leaf` at scikit-learn's default of 1; it is not a factory argument or a grid dimension.
AdaBoost leaves its base tree `class_weight` at the default `None`; Bagging uses
`class_weight="balanced"` on its base trees; Random Forest uses `class_weight="balanced_subsample"`.
Event `sample_weight` is still passed to training and evaluation.

::: modeling.ensemble_methods
