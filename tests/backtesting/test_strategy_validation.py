import numpy as np
import pandas as pd
import pytest

from src.backtesting.strategy_validation import (
    assemble_cpcv_path_predictions,
    combinatorial_purged_cross_validation,
    get_combinatorial_backtest_paths,
    get_cpcv_price_calibrations,
)
from src.modeling.purged_validation import PurgedKFold, index_events


def test_cpcv_returns_one_split_per_test_group_combination():
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
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
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")

    with pytest.raises(ValueError, match="greater than 1"):
        combinatorial_purged_cross_validation(pd.Series(index, index=index), 1, 1)


def test_cpcv_paths_select_one_prediction_per_observation():
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    splits = combinatorial_purged_cross_validation(
        pd.Series(index, index=index), num_groups=3, num_test_groups=2
    )
    predictions = pd.DataFrame([
        {
            "split_num": split.split_num,
            "observation_position": position,
            "value": split.split_num * 10 + position,
        }
        for split in splits.itertuples()
        for position in split.test_indices
    ])

    paths = assemble_cpcv_path_predictions(predictions, splits, index, num_groups=3)
    assignments = get_combinatorial_backtest_paths(splits, num_groups=3)
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

    assert set(paths) == {"path_0", "path_1"}
    for path_name, path in paths.items():
        assert path["observation_position"].tolist() == list(range(6))
        for group, positions in enumerate(groups):
            expected_split = assignments.loc[group, path_name]
            selected = path.loc[path["observation_position"].isin(positions)]
            assert selected["split_num"].eq(expected_split).all()


def test_cpcv_price_calibration_uses_training_observations_only():
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    development = pd.DataFrame({
        "entry_price": 100.0,
        "target_return": [0.01, 0.02, 0.03, 0.04, 0.50, 0.60],
        "partition": "development",
    }, index=pd.MultiIndex.from_arrays(
        [["A"] * 6, index], names=["symbol", "event_start"]
    ))
    splits = combinatorial_purged_cross_validation(
        pd.Series(index, index=index), num_groups=3, num_test_groups=1
    )
    split = splits.loc[splits["test_groups"].map(lambda groups: groups == (2,))].iloc[0]

    original = get_cpcv_price_calibrations(development, splits)
    changed = development.copy()
    changed.iloc[list(split.test_indices), changed.columns.get_loc("target_return")] = 99.0
    recalibrated = get_cpcv_price_calibrations(changed, splits)

    original_w = original.loc[original["split_num"].eq(split.split_num), "w"].item()
    changed_w = recalibrated.loc[
        recalibrated["split_num"].eq(split.split_num), "w"
    ].item()
    assert changed_w == original_w


@pytest.mark.parametrize("pct", [0, 0.01, 0.11])
def test_cpcv_matches_purged_kfold_for_identical_test_groups(pct):
    index = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
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
    starts = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    events = pd.Series(starts, index=starts)
    events.iloc[0] = starts[7]
    events.iloc[4] = starts[8]
    splits = combinatorial_purged_cross_validation(events, 6, 2, 0.01)
    split = next(row for _, row in splits.iterrows() if row["test_groups"] == groups)

    assert split["train_indices"] == expected


@pytest.mark.parametrize("pct", [-0.1, 1, 1.1, np.nan, np.inf, -np.inf])
def test_splitters_reject_invalid_embargo(pct):
    starts = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    events = pd.Series(starts, index=starts)
    features = pd.DataFrame({"feature": range(6)}, index=starts)

    with pytest.raises(ValueError, match="pct_embargo"):
        combinatorial_purged_cross_validation(events, 3, 1, pct)
    with pytest.raises(ValueError, match="pct_embargo"):
        list(PurgedKFold(3, events, pct).split(features))
