import numpy as np

from scipy.stats import rv_continuous
from src.feature_analysis.cross_validation import PurgedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.],
                n_jobs=-1, pctEmbargo=0, **fit_params):
    """Fit a classifier pipeline with purged cross-validation.

    Args:
        feat: Training features.
        lbl: Training labels.
        t1: Label end times for purged cross-validation.
        pipe_clf: Pipeline or estimator to tune.
        param_grid: Hyperparameter search space.
        cv: Number of cross-validation folds.
        bagging: Bagging configuration.
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
    gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                      scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_

    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps),
                               n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
                               max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params[
            gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])

    return gs

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


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.],
                rndSearchIter=0, n_jobs=-1, pctEmbargo=0, **fit_params):
    """Tune a classifier with grid or randomized purged cross-validation.

    Args:
        feat: Training features.
        lbl: Training labels.
        t1: Label end times for purged cross-validation.
        pipe_clf: Pipeline or estimator to tune.
        param_grid: Hyperparameter search space.
        cv: Number of cross-validation folds.
        bagging: Bagging configuration.
        rndSearchIter: Number of randomized search iterations. Use ``0`` for grid search.
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
                          scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid,
                                scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
                                iid=False, n_iter=rndSearchIter)

    gs = gs.fit(feat, lbl, **fit_params).best_estimator_

    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps),
                               n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]),
                               max_features=float(bagging[2]),
                               n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params[
            gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])

    return gs


class logUniform_gen(rv_continuous):
    """Continuous distribution with density uniform in log space."""

    def _cdf(self, x):
        """Evaluate the cumulative distribution function."""
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1, b=np.exp(1)):
    """Create a log-uniform random variable.

    Args:
        a: Lower bound.
        b: Upper bound.

    Returns:
        A SciPy continuous random variable instance.
    """
    return logUniform_gen(a=a, b=b, name='logUniform')
