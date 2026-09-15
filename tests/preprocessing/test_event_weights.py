import numpy as np
import pandas as pd
import pytest

from src.preprocessing.event_weights import (
    WEIGHT_COLUMNS,
    build_partitioned_event_weights,
    compute_return_attribution_weights,
    count_concurrent_events,
)


def _inputs():
    starts = pd.date_range("2025-01-02", periods=6, freq="2h", tz="UTC")
    events = pd.DataFrame(
        {
            "event_start": starts,
            "event_end": starts + pd.Timedelta(hours=1),
            "direction_label": np.tile([-1, 1], 3),
            "partition": ["development"] * 3 + ["holdout"] * 3,
            "holdout_boundary": starts[3],
        }
    )
    close_index = pd.date_range(
        starts.min(),
        starts.max() + pd.Timedelta(hours=1),
        freq="h",
    )
    close = pd.Series(np.linspace(100.0, 112.0, len(close_index)), index=close_index)
    return events, close


def test_partitioned_event_weights_are_complete_and_normalized():
    events, close = _inputs()

    weighted = build_partitioned_event_weights(events, close)

    assert weighted.columns[-2:].tolist() == WEIGHT_COLUMNS
    assert WEIGHT_COLUMNS == ["return_attribution_weight", "sample_weight"]
    assert np.isfinite(weighted[WEIGHT_COLUMNS]).all().all()
    for partition in ["development", "holdout"]:
        partition_weights = weighted.loc[
            weighted["partition"].eq(partition), "sample_weight"
        ]
        assert partition_weights.sum() == pytest.approx(len(partition_weights))
        attribution = weighted.loc[
            weighted["partition"].eq(partition), "return_attribution_weight"
        ]
        base = attribution.clip(lower=attribution[attribution.gt(0)].min())
        np.testing.assert_allclose(partition_weights, base / base.mean())
    pd.testing.assert_frame_equal(weighted[events.columns], events)


def test_legacy_columns_are_removed_on_recalculation():
    events, close = _inputs()
    expected = build_partitioned_event_weights(events, close)
    legacy = expected.assign(average_uniqueness_weight=0.5, time_decay_weight=0.5)
    legacy["sample_weight"] = 99.0
    pd.testing.assert_frame_equal(build_partitioned_event_weights(legacy, close), expected)


def test_equal_attribution_has_no_time_preference():
    events, close = _inputs()
    times = close.index.insert(0, close.index[0] - pd.Timedelta(hours=1))
    close = pd.Series(np.exp(np.arange(len(times), dtype=float)), index=times)
    weighted = build_partitioned_event_weights(events, close)
    for _, partition in weighted.groupby("partition"):
        np.testing.assert_allclose(partition["sample_weight"], 1.0)


def test_zero_attribution_uses_smallest_positive_weight():
    events, close = _inputs()
    close.iloc[1] = close.iloc[0]
    weighted = build_partitioned_event_weights(events, close)
    development = weighted.loc[weighted["partition"].eq("development")]
    attribution = development["return_attribution_weight"]
    assert attribution.iloc[0] == 0.0
    base = attribution.clip(lower=attribution[attribution.gt(0)].min())
    np.testing.assert_allclose(development["sample_weight"], base / base.mean())


@pytest.mark.parametrize("partition", ["development", "holdout"])
def test_all_zero_partition_is_rejected(partition):
    events, close = _inputs()
    if partition == "development":
        close[:] = 100.0
    else:
        close.iloc[5:] = close.iloc[5]
    with pytest.raises(ValueError, match=f"{partition} return-attribution weights are all zero"):
        build_partitioned_event_weights(events, close)


def test_holdout_changes_do_not_change_development_weights():
    events, close = _inputs()
    expected = build_partitioned_event_weights(events, close)
    close.iloc[6:] *= 1.5
    actual = build_partitioned_event_weights(events, close)
    mask = events["partition"].eq("development")
    pd.testing.assert_frame_equal(actual.loc[mask], expected.loc[mask])


def test_overlapping_events_share_return_contributions():
    times = pd.date_range("2025-01-02", periods=4, freq="h", tz="UTC")
    close = pd.Series(np.exp(np.arange(4, dtype=float)), index=times)
    ends = pd.Series([times[2], times[3]], index=times[:2])
    concurrency = count_concurrent_events(times, ends, ends.index)
    np.testing.assert_allclose(concurrency, [1, 2, 2, 1])
    attribution = compute_return_attribution_weights(ends, concurrency, close, ends.index)
    np.testing.assert_allclose(attribution, [1, 2])
