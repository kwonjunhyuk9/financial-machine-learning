import pandas as pd
import pytest

from src.strategy_modeling.cross_validation import PurgedKFold, get_embargo_times


def test_purged_kfold_exposes_configured_number_of_splits():
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    features = pd.DataFrame({"feature": range(6)}, index=index)
    splitter = PurgedKFold(3, pd.Series(index, index=index), pct_embargo=0.1)

    assert len(list(splitter.split(features))) == 3
    assert len(get_embargo_times(index, pct_embargo=0.5)) == len(index)


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
