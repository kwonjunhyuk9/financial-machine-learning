from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from src.modeling.purged_validation import PurgedKFold


class _ProgressGridSearchCV(GridSearchCV):
    """Evaluate configurations in order and report completion in the caller."""

    def _run_search(self, evaluate_candidates, *, callback_ctx=None):
        configurations = ParameterGrid(self.param_grid)
        search_ctx = (
            callback_ctx.subcontext(task_name="search", max_subtasks=len(configurations))
            .call_on_fit_task_begin(estimator=self)
            if callback_ctx is not None else None
        )
        for parameters in configurations:
            self._progress.set_postfix_str(str(parameters))
            configuration_ctx = (
                search_ctx.subcontext(task_name="configuration", max_subtasks=self.n_splits_,
                                      sequential_subtasks=False)
                .call_on_fit_task_begin(estimator=self)
                if search_ctx is not None else None
            )
            evaluate_candidates(
                [parameters],
                **({"callback_ctx": configuration_ctx} if configuration_ctx is not None else {}),
            )
            if configuration_ctx is not None:
                configuration_ctx.call_on_fit_task_end(estimator=self)
            self._progress.update(1)
        if search_ctx is not None:
            search_ctx.call_on_fit_task_end(estimator=self)
        self._progress.set_postfix_str("Refit best configuration")


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
    n_jobs: int = -1,
    pct_embargo: float = 0.0,
    *,
    show_progress: bool = False,
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
        n_jobs: Number of parallel workers for the search.
        pct_embargo: Embargo fraction applied to each fold.
        show_progress: Show completed configurations and the final refit.
        **fit_params: Extra fit parameters passed to the estimator.

    Returns:
        The best fitted estimator.
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
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> float:
        weights = (
            scoring_weight.loc[validation_features.index].to_numpy()
            if sample_weight is None else np.asarray(sample_weight)
        )
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

    search_class = _ProgressGridSearchCV if show_progress else GridSearchCV
    gs = search_class(estimator=pipe_clf, param_grid=param_grid,
                      scoring=weighted_scorer, cv=inner_cv, n_jobs=n_jobs)
    if not show_progress:
        return gs.fit(feat, lbl, **fit_params).best_estimator_
    with tqdm(total=len(ParameterGrid(param_grid)) + 1, desc="Grid search") as progress:
        gs._progress = progress
        fitted = gs.fit(feat, lbl, **fit_params).best_estimator_
        progress.update(1)
        return fitted
