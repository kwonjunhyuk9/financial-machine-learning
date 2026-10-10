# `modeling.feature_importance`

MDI maps each Bagging tree's selected features back to the full input feature set,
assigning zero to unselected features. Repeated feature selections are summed.
If all impurity importances are zero, MDI returns zero means and standard errors.
AdaBoost tree weights are normalized on a copy, leaving the fitted model unchanged.

MDA normalizes the baseline-minus-permuted score by the permuted log loss or
`1 - permuted score`. When that denominator is zero, it returns the unnormalized
score difference for that fold instead: zero for unchanged perfect predictions,
or a negative value if permutation improved the score to its maximum.

::: modeling.feature_importance
