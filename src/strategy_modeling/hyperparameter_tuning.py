from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.strategy_modeling.cross_validation import PurgedKFold


class MyPipeline(Pipeline):
    """Pipeline that forwards sample weights to the final step."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> MyPipeline:
        """Fit the pipeline while passing sample weights to the last estimator.

        Args:
            X: Training features.
            y: Training labels.
            sample_weight: Optional per-sample weights.
            **fit_params: Extra fit parameters passed to the parent pipeline.

        Returns:
            The fitted pipeline instance.
        """
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super().fit(X, y, **fit_params)


def fit_classifier_with_hyperparameter_search(
    feat: pd.DataFrame,
    lbl: pd.Series,
    t1: pd.Series,
    pipe_clf: BaseEstimator,
    param_grid: dict[str, Sequence[Any]] | list[dict[str, Any]],
    cv: int = 3,
    bagging: Sequence[int | float | None] = (0, None, 1.0),
    n_jobs: int = -1,
    pct_embargo: float = 0.0,
    **fit_params: Any,
) -> BaseEstimator:
    """Tune a classifier with grid search and purged cross-validation.

    Args:
        feat: Training features.
        lbl: Training labels.
        t1: Label end times for purged cross-validation.
        pipe_clf: Pipeline or estimator to tune.
        param_grid: Hyperparameter search space.
        cv: Number of cross-validation folds.
        bagging: Bagging configuration.
        n_jobs: Number of parallel workers for the search.
        pct_embargo: Embargo fraction applied to each fold.
        **fit_params: Extra fit parameters passed to the estimator.

    Returns:
        The best fitted estimator, optionally wrapped in a bagging pipeline.
    """
    final_step_weight = None
    if hasattr(pipe_clf, "steps"):
        final_step_weight = fit_params.get(
            pipe_clf.steps[-1][0] + "__sample_weight"
        )
    sample_weight = fit_params.get("sample_weight", final_step_weight)
    if sample_weight is None:
        scoring_weight = pd.Series(1.0, index=feat.index)
    elif isinstance(sample_weight, pd.Series):
        scoring_weight = sample_weight.reindex(feat.index)
    else:
        scoring_weight = pd.Series(sample_weight, index=feat.index)

    use_f1 = set(lbl.values) == {0, 1}

    def weighted_scorer(
        estimator: BaseEstimator,
        validation_features: pd.DataFrame,
        validation_labels: pd.Series,
    ) -> float:
        weights = scoring_weight.loc[validation_features.index].to_numpy()
        if use_f1:
            return float(f1_score(
                validation_labels,
                estimator.predict(validation_features),
                pos_label=1,
                sample_weight=weights,
                zero_division=0,
            ))
        return -float(log_loss(
            validation_labels,
            estimator.predict_proba(validation_features),
            labels=estimator.classes_,
            sample_weight=weights,
        ))

    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                      scoring=weighted_scorer, cv=inner_cv, n_jobs=n_jobs)

    gs = gs.fit(feat, lbl, **fit_params).best_estimator_

    if bagging[1] is not None and bagging[1] > 0:
        gs = BaggingClassifier(estimator=MyPipeline(gs.steps),
                               n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]),
                               max_features=float(bagging[2]),
                               n_jobs=n_jobs)
        sample_weight = fit_params.get(
            'sample_weight',
            fit_params.get(gs.estimator.steps[-1][0] + '__sample_weight')
        )
        gs = gs.fit(feat, lbl, sample_weight=sample_weight)
        gs = Pipeline([('bag', gs)])

    return gs
