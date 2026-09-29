import numpy as np
import pandas as pd
import pytest

from src.backtesting.strategy_validation import combinatorial_purged_cross_validation
from src.modeling.purged_validation import PurgedKFold, index_events


def test_cpcv_returns_one_split_per_test_group_combination():
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    samples_info_sets = pd.Series(index + pd.Timedelta(days=1), index=index)

    splits = combinatorial_purged_cross_validation(samples_info_sets, num_groups=3, num_test_groups=1)

    assert len(splits) == 3
    assert set(splits.columns) >= {"train_indices", "test_indices"}

    starts = pd.date_range("2025-01-02", periods=12, freq="h", tz="UTC")
    events = index_events(pd.DataFrame([
        {"symbol": symbol, "event_start": time,
         "event_end": time + pd.Timedelta(minutes=70)}
        for time in starts for symbol in ["A", "B"]
    ]))
    equal_time_splits = combinatorial_purged_cross_validation(events.event_end, 3, 1, .05)
    folds = list(PurgedKFold(3, events.event_end, .05).split(events))
    for row, (train, test) in zip(equal_time_splits.itertuples(), folds):
        assert row.train_indices == tuple(train)
        assert row.test_indices == tuple(test)


def test_cpcv_rejects_invalid_group_count():
    index = pd.date_range("2026-01-01", periods=3, freq="D")

    with pytest.raises(ValueError, match="greater than 1"):
        combinatorial_purged_cross_validation(pd.Series(index, index=index), 1, 1)


@pytest.mark.parametrize("pct", [0, 0.01, 0.11])
def test_cpcv_matches_purged_kfold_for_identical_test_groups(pct):
    index = pd.date_range("2026-01-01", periods=12, freq="D")
    events = pd.Series(index, index=index)
    events.iloc[0] = index[4]
    features = pd.DataFrame({"feature": range(12)}, index=index)
    splits = combinatorial_purged_cross_validation(events, 3, 1, pct)

    for (_, split), (train, test) in zip(
        splits.iterrows(), PurgedKFold(3, events, pct).split(features)
    ):
        np.testing.assert_array_equal(split["test_indices"], test)
        np.testing.assert_array_equal(split["train_indices"], train)


@pytest.mark.parametrize(
    "groups,expected", [((0, 1), (9, 10, 11)), ((0, 2), (10, 11))]
)
def test_cpcv_adjacent_and_separated_groups_use_latest_event_end(groups, expected):
    events = pd.Series(range(12), index=range(12))
    events.iloc[0] = 7
    events.iloc[4] = 8
    splits = combinatorial_purged_cross_validation(events, 6, 2, 0.01)
    split = next(row for _, row in splits.iterrows() if row["test_groups"] == groups)

    assert split["train_indices"] == expected


@pytest.mark.parametrize("pct", [-0.1, 1, 1.1, np.nan, np.inf, -np.inf])
def test_splitters_reject_invalid_embargo(pct):
    events = pd.Series(range(6), index=range(6))
    features = pd.DataFrame({"feature": range(6)})

    with pytest.raises(ValueError, match="pct_embargo"):
        combinatorial_purged_cross_validation(events, 3, 1, pct)
    with pytest.raises(ValueError, match="pct_embargo"):
        list(PurgedKFold(3, events, pct).split(features))
