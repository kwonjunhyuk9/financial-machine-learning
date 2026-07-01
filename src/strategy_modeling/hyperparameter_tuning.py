import numpy as np

from scipy.stats import rv_continuous
from src.strategy_modeling.cross_validation import PurgedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    """Pipeline that forwards sample weights to the final step."""

    def fit(self, X, y, sample_weight=None, **fit_params):
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
        return super(MyPipeline, self).fit(X, y, **fit_params)


def fit_classifier_with_hyperparameter_search(
        feat,
        lbl,
        t1,
        pipe_clf,
        param_grid,
        cv=3,
        bagging=[0, None, 1.],
        rndSearchIter=0,
        n_jobs=-1,
        pctEmbargo=0,
        **fit_params
):
    """Tune a classifier with grid or randomized purged cross-validation.

    Args:
        feat: Training features.
        lbl: Training labels.
        t1: Label end times for purged cross-validation.
        pipe_clf: Pipeline or estimator to tune.
        param_grid: Hyperparameter search space.
        cv: Number of cross-validation folds.
        bagging: Bagging configuration.
        rndSearchIter: Number of randomized search iterations.
        n_jobs: Number of parallel workers for the search.
        pctEmbargo: Embargo fraction applied to each fold.
        **fit_params: Extra fit parameters passed to the estimator.

    Returns:
        The best fitted estimator, optionally wrapped in a bagging pipeline.
    """
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'
    else:
        scoring = 'neg_log_loss'

    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                          scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid,
                                scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
                                n_iter=rndSearchIter)

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


class LogUniformDistribution(rv_continuous):
    """Continuous distribution with density uniform in log space."""

    def _cdf(self, x):
        """Evaluate the cumulative distribution function.

        Args:
            x: Evaluation point.

        Returns:
            The cumulative probability at ``x``.
        """
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform(a=1, b=np.exp(1)):
    """Create a log-uniform random variable.

    Args:
        a: Lower bound.
        b: Upper bound.

    Returns:
        A SciPy continuous random variable instance.
    """
    return LogUniformDistribution(a=a, b=b, name='log_uniform')
