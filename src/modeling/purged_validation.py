from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator


def _purge_train_indices(
    samples_info_sets: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    """Remove training intervals overlapping any test interval, including endpoints.

    Inputs use positions in the same start-time-sorted event series. Neither
    the series nor the index arrays are modified or reordered.
    """
    train_starts = samples_info_sets.index[train_indices]
    train_ends = samples_info_sets.iloc[train_indices]
    keep = np.ones(train_indices.shape[0], dtype=bool)

    for test_start, test_end in samples_info_sets.iloc[test_indices].items():
        overlap = (train_starts <= test_end) & (train_ends >= test_start)
        keep &= ~overlap

    return train_indices[keep]


def _embargo_train_indices(
    samples_info_sets: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    pct_embargo: float,
) -> np.ndarray:
    """Exclude positions after each contiguous test run's latest event end.

    Event starts and test positions must be sorted. Each window starts strictly
    after that run's latest end and spans ceil(N * pct_embargo) observations,
    capped at the series end. N is the full event-series length, before purging.
    Adjacent test positions form one run. Inputs are not modified or reordered.

    Raises:
        ValueError: If pct_embargo is not finite or outside [0, 1).
    """
    if not np.isfinite(pct_embargo) or not 0 <= pct_embargo < 1:
        raise ValueError("pct_embargo must be finite and in [0, 1)")

    embargo_size = int(np.ceil(len(samples_info_sets) * pct_embargo))
    if embargo_size == 0 or test_indices.size == 0:
        return train_indices

    keep = np.ones(train_indices.shape[0], dtype=bool)
    boundaries = np.flatnonzero(np.diff(test_indices) != 1) + 1
    for test_run in np.split(test_indices, boundaries):
        start = samples_info_sets.index.searchsorted(
            samples_info_sets.iloc[test_run].max(), side="right"
        )
        stop = min(start + embargo_size, len(samples_info_sets))
        keep &= ~((train_indices >= start) & (train_indices < stop))

    return train_indices[keep]


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
            t1: Label end times indexed by sorted observation start time.
            pct_embargo: Finite fraction in [0, 1) of all observations to embargo,
                rounded up, strictly after each test fold's latest label end.

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
            ValueError: If ``X`` and ``t1`` do not share the same index or
                pct_embargo is not finite or outside [0, 1).
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")

        indices = np.arange(X.shape[0])

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

            train_indices = _purge_train_indices(
                samples_info_sets=self.t1,
                train_indices=train_indices,
                test_indices=test_indices,
            )
            train_indices = _embargo_train_indices(
                samples_info_sets=self.t1,
                train_indices=train_indices,
                test_indices=test_indices,
                pct_embargo=self.pct_embargo,
            )

            yield train_indices, test_indices
