from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def combinatorial_purged_cross_validation(
        samples_info_sets,
        num_groups,
        num_test_groups,
        pct_embargo=0.0
):
    """Generate combinatorial purged cross-validation split metadata.

    Args:
        samples_info_sets: Series indexed by observation start time and valued by
        observation end time.
        num_groups: Number of contiguous groups used to partition the sorted observations.
        num_test_groups: Number of groups used as the test set in each split.
        pct_embargo: Fraction of all observations to embargo after each tested group.

    Returns:
        A frame with one row per test-group combination and these columns: ``split_num`` is
        the sequential split identifier; ``train_groups`` is the tuple of group numbers not
        selected for testing; ``test_groups`` is the tuple of group numbers selected for
        testing; ``train_indices`` contains purged and embargoed training positions;
        ``test_indices`` contains test positions.
    """
    samples_info_sets = _validate_samples_info_sets(samples_info_sets)

    if num_groups < 2:
        raise ValueError("num_groups must be greater than 1")

    if num_groups > samples_info_sets.shape[0]:
        raise ValueError("num_groups cannot exceed the number of observations")

    if num_test_groups < 1 or num_test_groups >= num_groups:
        raise ValueError("num_test_groups must be in [1, num_groups)")

    if pct_embargo < 0 or pct_embargo >= 1:
        raise ValueError("pct_embargo must be in [0, 1)")

    groups = _get_groups(
        num_observations=samples_info_sets.shape[0],
        num_groups=num_groups
    )
    embargo_size = int(np.ceil(samples_info_sets.shape[0] * pct_embargo))
    out = []

    for split_num, test_groups in enumerate(
            combinations(range(num_groups), num_test_groups)
    ):
        test_indices = np.concatenate([
            groups[group]
            for group in test_groups
        ])
        train_indices = np.setdiff1d(
            np.arange(samples_info_sets.shape[0]),
            test_indices,
            assume_unique=True
        )
        train_indices = _purge_train_indices(
            samples_info_sets=samples_info_sets,
            train_indices=train_indices,
            test_indices=test_indices
        )
        train_indices = _embargo_train_indices(
            train_indices=train_indices,
            test_groups=test_groups,
            groups=groups,
            embargo_size=embargo_size,
            num_observations=samples_info_sets.shape[0]
        )

        out.append({
            "split_num": split_num,
            "train_groups": tuple(
                group
                for group in range(num_groups)
                if group not in test_groups
            ),
            "test_groups": test_groups,
            "train_indices": tuple(train_indices.tolist()),
            "test_indices": tuple(test_indices.tolist())
        })

    return pd.DataFrame(out)


def get_combinatorial_backtest_paths(splits, num_groups):
    """Arrange CPCV split identifiers into combinatorial backtest paths.

    Args:
        splits: Output from ``combinatorial_purged_cross_validation``.
        num_groups: Number of contiguous observation groups used to make the supplied CPCV
        splits.

    Returns:
        A frame indexed by group number with one column per CPCV backtest path.

    Raises:
        ValueError: If split metadata cannot form complete paths.
    """
    splits = _validate_splits(splits=splits, num_groups=num_groups)
    num_test_groups = len(splits.iloc[0]["test_groups"])
    num_paths = comb(num_groups - 1, num_test_groups - 1)

    path_counts = pd.Series(0, index=range(num_groups))
    paths = pd.DataFrame(
        np.nan,
        index=pd.Index(range(num_groups), name="group"),
        columns=[f"path_{path}" for path in range(num_paths)]
    )

    for _, split in splits.iterrows():
        for group in split["test_groups"]:
            path_num = path_counts.loc[group]

            if path_num >= num_paths:
                raise ValueError("splits cannot form complete backtest paths")

            paths.loc[group, f"path_{path_num}"] = split["split_num"]
            path_counts.loc[group] += 1

    if paths.shape[1] != num_paths or paths.isna().any().any():
        raise ValueError("splits cannot form complete backtest paths")

    return paths.astype("int64")


def _validate_samples_info_sets(samples_info_sets):
    """Validate and sort event information intervals.

    Args:
        samples_info_sets: Candidate event interval series.

    Returns:
        The same series sorted by start time.

    Raises:
        ValueError: If event intervals are missing, duplicated, or inconsistent.
    """
    if not isinstance(samples_info_sets, pd.Series):
        raise ValueError("samples_info_sets must be a pd.Series")

    if samples_info_sets.empty:
        raise ValueError("samples_info_sets must not be empty")

    samples_info_sets = samples_info_sets.sort_index()

    if samples_info_sets.index.has_duplicates:
        raise ValueError("samples_info_sets index must not contain duplicates")

    if samples_info_sets.isna().any():
        raise ValueError("samples_info_sets must not contain missing end times")

    if (samples_info_sets < samples_info_sets.index).any():
        raise ValueError("samples_info_sets end times must be at or after start times")

    return samples_info_sets


def _get_groups(num_observations, num_groups):
    """Partition observation positions into CPCV groups without shuffling.

    Args:
        num_observations: Total number of sorted observations.
        num_groups: Number of contiguous groups to produce.

    Returns:
        A list of integer position arrays.
    """
    group_size = num_observations // num_groups
    groups = []
    start = 0

    for _ in range(num_groups - 1):
        stop = start + group_size
        groups.append(np.arange(start, stop))
        start = stop

    groups.append(np.arange(start, num_observations))

    return groups


def _purge_train_indices(samples_info_sets, train_indices, test_indices):
    """Remove training observations whose information intervals overlap tests.

    Args:
        samples_info_sets: Validated event interval series sorted by observation start time.
        train_indices: Candidate training positions before purging.
        test_indices: Test positions for the current CPCV split.

    Returns:
        Training positions that do not overlap any test interval.
    """
    train_starts = samples_info_sets.index[train_indices]
    train_ends = samples_info_sets.iloc[train_indices]
    test_starts = samples_info_sets.index[test_indices]
    test_ends = samples_info_sets.iloc[test_indices]

    keep = np.ones(train_indices.shape[0], dtype=bool)

    for test_start, test_end in zip(test_starts, test_ends):
        overlap = (train_starts <= test_end) & (train_ends >= test_start)
        keep &= ~overlap

    return train_indices[keep]


def _embargo_train_indices(
        train_indices,
        test_groups,
        groups,
        embargo_size,
        num_observations
):
    """Remove candidate training observations inside post-test embargo windows.

    Args:
        train_indices: Candidate training positions after purging.
        test_groups: Group numbers assigned to the test set in the current split.
        groups: List of contiguous group position arrays.
        embargo_size: Number of positional observations embargoed after each tested group.
        num_observations: Total number of observations in the sample.

    Returns:
        Training positions with any observation in a post-test embargo window removed.
    """
    if embargo_size == 0:
        return train_indices

    embargoed_indices = []

    for group in test_groups:
        start = groups[group][-1] + 1
        stop = min(start + embargo_size, num_observations)
        embargoed_indices.extend(range(start, stop))

    if not embargoed_indices:
        return train_indices

    return np.setdiff1d(train_indices, np.array(embargoed_indices), assume_unique=True)


def _validate_splits(splits, num_groups):
    """Validate CPCV split metadata before constructing backtest paths.

    Args:
        splits: Candidate CPCV split metadata frame.
        num_groups: Number of valid group identifiers, covering ``0`` through ``num_groups -
        1``.

    Returns:
        The original ``splits`` frame when it contains valid path metadata.

    Raises:
        ValueError: If split metadata is missing or references invalid groups.
    """
    required_columns = {"split_num", "test_groups"}

    if not isinstance(splits, pd.DataFrame):
        raise ValueError("splits must be a pd.DataFrame")

    if not required_columns.issubset(splits.columns):
        raise ValueError("splits must contain split_num and test_groups columns")

    if splits.empty:
        raise ValueError("splits must not be empty")

    test_group_sizes = splits["test_groups"].map(len)

    if test_group_sizes.nunique() != 1:
        raise ValueError("all splits must use the same number of test groups")

    for test_groups in splits["test_groups"]:
        if len(set(test_groups)) != len(test_groups):
            raise ValueError("test_groups must not contain duplicate groups")

        if min(test_groups) < 0 or max(test_groups) >= num_groups:
            raise ValueError("test_groups contains a group outside [0, num_groups)")

    return splits
