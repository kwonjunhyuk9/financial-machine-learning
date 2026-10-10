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
            "symbol": "AAPL",
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
    close_index = pd.MultiIndex.from_product([["AAPL"], close_index], names=["symbol", "end"])
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
    times = close.index.get_level_values("end")
    times = times.insert(0, times[0] - pd.Timedelta(hours=1))
    times = pd.MultiIndex.from_product([["AAPL"], times], names=["symbol", "end"])
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


def test_weights_normalize_across_symbols_not_separately():
    start = pd.Timestamp("2025-01-02", tz="UTC")
    events = pd.DataFrame([
        {"symbol": symbol, "event_start": start + pd.Timedelta(hours=hour),
         "event_end": start + pd.Timedelta(hours=hour + 1),
         "partition": "development" if hour == 0 else "holdout",
         "holdout_boundary": start + pd.Timedelta(hours=2)}
        for symbol in ["A", "B"] for hour in [0, 2]
    ])
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range(start, periods=4, freq="h")],
        names=["symbol", "end"],
    )
    close = pd.Series([100., 101., 102., 103., 100., 110., 120., 130.], index=index)
    result = build_partitioned_event_weights(events, close, show_progress=True)
    pd.testing.assert_frame_equal(result, build_partitioned_event_weights(events, close))
    assert result.groupby("partition").sample_weight.mean().eq(1).all()
    assert not result.loc[result.symbol.eq("A"), "sample_weight"].eq(1).all()


@pytest.mark.parametrize("column", ["event_start", "event_end", "holdout_boundary"])
def test_weights_reject_invalid_event_timestamps(column):
    events, close = _inputs()
    events.loc[0, column] = pd.NaT
    with pytest.raises(ValueError, match="valid timestamps"):
        build_partitioned_event_weights(events, close)


def test_weights_reject_missing_symbol_and_duplicate_event_keys():
    events, close = _inputs()
    with pytest.raises(ValueError, match="missing columns.*symbol"):
        build_partitioned_event_weights(events.drop(columns="symbol"), close)
    with pytest.raises(ValueError, match="unique valid composite"):
        build_partitioned_event_weights(pd.concat([events, events.iloc[:1]]), close)


def test_weights_require_both_partitions_and_one_boundary():
    events, close = _inputs()
    with pytest.raises(ValueError, match="development and holdout"):
        build_partitioned_event_weights(events.loc[events.partition.eq("development")], close)
    events.loc[0, "holdout_boundary"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="one holdout boundary"):
        build_partitioned_event_weights(events, close)


def test_weights_reject_time_only_prices():
    events, close = _inputs()
    with pytest.raises(ValueError, match="indexed by"):
        build_partitioned_event_weights(events, close.droplevel("symbol"))


@pytest.mark.parametrize("invalid_price", [np.nan, np.inf])
def test_weights_reject_nonfinite_prices(invalid_price):
    events, close = _inputs()
    close.iloc[0] = invalid_price
    with pytest.raises(ValueError, match="finite and non-empty"):
        build_partitioned_event_weights(events, close)


def test_weights_reject_empty_prices_and_duplicate_price_keys():
    events, close = _inputs()
    with pytest.raises(ValueError, match="finite and non-empty"):
        build_partitioned_event_weights(events, close.iloc[:0])
    with pytest.raises(ValueError, match="unique valid"):
        build_partitioned_event_weights(events, pd.concat([close, close.iloc[:1]]))


def test_weights_reject_invalid_price_timestamps():
    events, close = _inputs()
    times = close.index.get_level_values("end").to_list()
    times[0] = pd.NaT
    close.index = pd.MultiIndex.from_arrays([["AAPL"] * len(times), times], names=["symbol", "end"])
    with pytest.raises(ValueError, match="unique valid"):
        build_partitioned_event_weights(events, close)
