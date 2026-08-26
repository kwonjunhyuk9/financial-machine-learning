from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator


def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
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


def get_embargo_times(times: pd.Index, pct_embargo: float) -> pd.Series:
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

    def __init__(
        self,
        n_splits: int = 3,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.0,
    ) -> None:
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

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the configured number of folds.

        Args:
            X: Unused feature matrix accepted for scikit-learn compatibility.
            y: Unused target values accepted for scikit-learn compatibility.
            groups: Unused group labels accepted for scikit-learn compatibility.

        Returns:
            The configured number of cross-validation folds.
        """
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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
            test_indices = indices[i:j]
            train_indices = np.setdiff1d(
                indices,
                test_indices,
                assume_unique=True,
            )

            train_starts = self.t1.index[train_indices]
            train_ends = self.t1.iloc[train_indices]
            keep = np.ones(train_indices.shape[0], dtype=bool)
            for test_start, test_end in self.t1.iloc[test_indices].items():
                overlap = (train_starts <= test_end) & (train_ends >= test_start)
                keep &= ~overlap
            train_indices = train_indices[keep]

            if mbrg:
                embargo_start = self.t1.index.searchsorted(
                    self.t1.iloc[test_indices].max(),
                    side="right",
                )
                embargo_indices = indices[
                    embargo_start:embargo_start + mbrg
                ]
                train_indices = np.setdiff1d(
                    train_indices,
                    embargo_indices,
                    assume_unique=True,
                )

            yield train_indices, test_indices
