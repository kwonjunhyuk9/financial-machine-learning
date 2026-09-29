import numpy as np
import pandas as pd
import pytest

from src.modeling.purged_validation import (
    PurgedKFold, _embargo_train_indices, _purge_train_indices,
    event_times, index_events,
)


def test_purged_kfold_exposes_configured_number_of_splits():
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    features = pd.DataFrame({"feature": range(6)}, index=index)
    splitter = PurgedKFold(3, pd.Series(index, index=index), pct_embargo=0.1)

    assert len(list(splitter.split(features))) == 3


def test_purged_kfold_rejects_too_few_splits():
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    with pytest.raises(ValueError, match="at least 2"):
        PurgedKFold(1, pd.Series(index, index=index))


def test_purged_kfold_removes_overlapping_intervals_and_post_test_embargo():
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    features = pd.DataFrame({"feature": range(6)}, index=index)
    information_sets = pd.Series(
        index + pd.to_timedelta([0, 2, 0, 0, 0, 0], unit="D"),
        index=index,
    )
    splitter = PurgedKFold(3, information_sets, pct_embargo=0.2)

    for train, test in splitter.split(features):
        train_starts = information_sets.index[train]
        train_ends = information_sets.iloc[train]
        for test_start, test_end in information_sets.iloc[test].items():
            overlap = (train_starts <= test_end) & (train_ends >= test_start)
            assert not overlap.any()

    first_train, _ = next(splitter.split(features))
    assert 4 not in first_train


def composite_events():
    starts = pd.date_range("2025-01-02", periods=12, freq="h", tz="UTC")
    return pd.DataFrame([
        {"symbol": symbol, "event_start": time,
         "event_end": time + pd.Timedelta(minutes=70)}
        for time in starts for symbol in ["A", "B"]
    ])


def test_cv_purges_across_symbols_and_keeps_equal_times_together():
    events = index_events(composite_events())
    starts = event_times(events.index)
    for train, test in PurgedKFold(3, events.event_end, .05).split(events):
        assert not set(starts[train]) & set(starts[test])
        for position in test:
            overlap = ((starts[train] <= events.event_end.iloc[position])
                       & (events.event_end.iloc[train].to_numpy() >= starts[position]))
            assert not overlap.any()


def test_same_time_duplicate_for_same_symbol_is_rejected():
    frame = composite_events()
    with pytest.raises(ValueError, match="unique"):
        index_events(pd.concat([frame, frame.iloc[[0]]]))


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
        ([1, 2], [0, 3, 4, 5, 6, 7, 8, 9], .01, [0, 4, 5, 6, 7, 8, 9]),
        ([1, 2], [0, 3, 4, 5, 6, 7, 8, 9], .11, [0, 5, 6, 7, 8, 9]),
        ([1, 2, 6, 7], [0, 3, 4, 5, 8, 9], .2, [0, 5]),
        ([1, 3], [0, 2, 4, 5, 6, 7, 8, 9], .4, [0, 8, 9]),
        ([8], [0, 1, 9], .4, [0, 1]), ([9], [0, 1], .4, [0, 1]),
        ([1], [], .2, []), ([], [0, 1], .2, [0, 1]),
    ],
)
def test_embargo_windows(test, train, pct, expected):
    events = pd.Series(range(10), index=range(10))
    train = np.array(train, dtype=int)
    test = np.array(test, dtype=int)
    original_train, original_test, original_events = train.copy(), test.copy(), events.copy()
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
    np.testing.assert_array_equal(_embargo_train_indices(events, train, test, .11), [0, 8, 9])


@pytest.mark.parametrize("pct", [-.1, 1, 1.1, np.nan, np.inf, -np.inf])
def test_embargo_rejects_invalid_fraction(pct):
    with pytest.raises(ValueError, match="pct_embargo"):
        _embargo_train_indices(pd.Series([0, 1]), np.array([1]), np.array([0]), pct)
