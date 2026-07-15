import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import BaseCrossValidator


def get_train_times(t1, test_times):
    """Remove training labels that overlap with the test intervals.

    Args:
        t1: Label end times indexed by observation start time.
        test_times: Test interval end times indexed by interval start time.

    Returns:
        A filtered series of training label end times.
    """
    trn = t1.copy(deep=True)

    for i, j in test_times.items():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        df1 = trn[(i <= trn) & (trn <= j)].index
        df2 = trn[(trn.index <= i) & (j <= trn)].index

        trn = trn.drop(df0.union(df1).union(df2))

    return trn


def get_embargo_times(times, pct_embargo):
    """Apply an embargo window after each observation time.

    Args:
        times: Ordered observation times.
        pct_embargo: Fraction of the sample length to embargo.

    Returns:
        A series mapping each observation time to its embargo end time.
    """
    step = int(times.shape[0] * pct_embargo)

    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = pd.concat([
            mbrg,
            pd.Series(times[-1], index=times[-step:])
        ])

    return mbrg


class PurgedKFold(BaseCrossValidator):
    """K-fold splitter that purges overlapping labels and applies embargo."""

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0):
        """Initialize the purged cross-validator.

        Args:
            n_splits: Number of folds.
            t1: Label end times indexed by observation time.
            pct_embargo: Fraction of observations to embargo after each test fold.

        Returns:
            None.

        Raises:
            ValueError: If ``t1`` is not a ``pd.Series``.
        """
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the configured number of folds.

        Args:
            X: Unused feature matrix accepted for scikit-learn compatibility.
            y: Unused target values accepted for scikit-learn compatibility.
            groups: Unused group labels accepted for scikit-learn compatibility.

        Returns:
            The configured number of cross-validation folds.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yield purged train and test index splits.

        Args:
            X: Feature matrix indexed like ``self.t1``.
            y: Unused target values.
            groups: Unused grouping labels.

        Yields:
            Train and test index arrays for each cross-validation fold.

        Raises:
            ValueError: If ``X`` and ``t1`` do not share the same index.
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)

        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]

            max_t1_index = self.t1.index.searchsorted(
                self.t1.iloc[test_indices].max()
            )

            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index
            )

            if max_t1_index < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_index + mbrg:])
                )

            yield train_indices, test_indices


def score_cross_validation(
        clf,
        X,
        y,
        sample_weight,
        scoring="neg_log_loss",
        t1=None,
        cv=None,
        cv_gen=None,
        pct_embargo=None
):
    """Evaluate a classifier under purged cross-validation.

    Args:
        clf: Estimator implementing ``fit`` and ``predict`` methods.
        X: Feature matrix.
        y: Target values.
        sample_weight: Sample weights aligned with ``X``.
        scoring: Scoring metric, either ``"neg_log_loss"`` or ``"accuracy"``.
        t1: Label end times used when ``cv_gen`` is not supplied.
        cv: Number of folds used when ``cv_gen`` is not supplied.
        cv_gen: Preconfigured cross-validation generator.
        pct_embargo: Embargo fraction used when constructing ``cv_gen``.

    Returns:
        A NumPy array of fold scores.

    Raises:
        ValueError: If ``scoring`` is not supported.
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'.")

    if cv_gen is None:
        cv_gen = PurgedKFold(
            n_splits=cv,
            t1=t1,
            pct_embargo=pct_embargo
        )

    score = []

    for train, test in cv_gen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values
        )

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])

            score_ = -log_loss(
                y.iloc[test],
                prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_
            )

        else:
            pred = fit.predict(X.iloc[test, :])

            score_ = accuracy_score(
                y.iloc[test],
                pred,
                sample_weight=sample_weight.iloc[test].values
            )

        score.append(score_)

    return np.array(score)
