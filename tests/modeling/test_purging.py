import numpy as np
import pandas as pd
import pytest

from src.modeling.purged_validation import (
    _embargo_train_indices,
    _purge_train_indices,
)


def test_purge_removes_overlap_containment_and_touching_endpoints():
    events = pd.Series([0, 3, 8, 6, 4, 8, 6, 7], index=range(8))
    train = np.array([0, 1, 2, 4, 5, 6, 7])
    test = np.array([3])
    original = events.copy()

    result = _purge_train_indices(events, train, test)

    np.testing.assert_array_equal(result, [0, 7])
    pd.testing.assert_series_equal(events, original)
    np.testing.assert_array_equal(train, [0, 1, 2, 4, 5, 6, 7])
    np.testing.assert_array_equal(test, [3])


@pytest.mark.parametrize(
    "test,train,pct,expected",
    [
        ([1, 2], [0, 3, 4, 5, 6, 7, 8, 9], 0, [0, 3, 4, 5, 6, 7, 8, 9]),
        ([1, 2], [0, 3, 4, 5, 6, 7, 8, 9], 0.01, [0, 4, 5, 6, 7, 8, 9]),
        ([1, 2], [0, 3, 4, 5, 6, 7, 8, 9], 0.11, [0, 5, 6, 7, 8, 9]),
        ([1, 2, 6, 7], [0, 3, 4, 5, 8, 9], 0.2, [0, 5]),
        ([1, 3], [0, 2, 4, 5, 6, 7, 8, 9], 0.4, [0, 8, 9]),
        ([8], [0, 1, 9], 0.4, [0, 1]),
        ([9], [0, 1], 0.4, [0, 1]),
        ([1], [], 0.2, []),
        ([], [0, 1], 0.2, [0, 1]),
    ],
)
def test_embargo_windows(test, train, pct, expected):
    events = pd.Series(range(10), index=range(10))
    train = np.array(train, dtype=int)
    test = np.array(test, dtype=int)
    original_train, original_test = train.copy(), test.copy()
    original_events = events.copy()

    result = _embargo_train_indices(events, train, test, pct)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(test, original_test)
    pd.testing.assert_series_equal(events, original_events)


def test_embargo_uses_latest_end_and_full_sample_size_after_purging():
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    events = pd.Series(index, index=index)
    events.iloc[1] = index[5] + pd.Timedelta(hours=12)
    test = np.array([1, 2])
    train = _purge_train_indices(events, np.array([0, 3, 4, 5, 6, 7, 8, 9]), test)

    np.testing.assert_array_equal(train, [0, 6, 7, 8, 9])
    np.testing.assert_array_equal(
        _embargo_train_indices(events, train, test, 0.11), [0, 8, 9]
    )


@pytest.mark.parametrize("pct", [-0.1, 1, 1.1, np.nan, np.inf, -np.inf])
def test_embargo_rejects_invalid_fraction(pct):
    with pytest.raises(ValueError, match="pct_embargo"):
        _embargo_train_indices(pd.Series([0, 1]), np.array([1]), np.array([0]), pct)
