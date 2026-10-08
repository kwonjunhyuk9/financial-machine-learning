from __future__ import annotations

import numpy as np
import pandas as pd


WEIGHT_COLUMNS = [
    "return_attribution_weight",
    "sample_weight",
]


def count_concurrent_events(
    close_idx: pd.Index,
    t1: pd.Series,
    molecule: pd.Index,
) -> pd.Series:
    """Count how many events are active at each bar in a slice.

    Args:
        close_idx: Full close-price index.
        t1: Event end times indexed by start time.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of concurrency counts over the relevant bar range.
    """
    t1 = t1.fillna(close_idx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]

    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0]:iloc[1] + 1])

    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.

    return count.loc[molecule[0]:t1[molecule].max()]


def compute_return_attribution_weights(
    t1: pd.Series,
    num_co_events: pd.Series,
    close: pd.Series,
    molecule: pd.Index,
) -> pd.Series:
    """Compute return-attribution sample weights.

    Args:
        t1: Event end times indexed by start time.
        num_co_events: Concurrency counts over the price bars.
        close: Close price series.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of absolute sample weights.
    """
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule)

    for t_in, t_out in t1.loc[wght.index].items():
        wght.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()

    return wght.abs()


def build_partitioned_event_weights(
    events: pd.DataFrame,
    close: pd.Series,
) -> pd.DataFrame:
    """Normalize floored return attribution independently within each partition.

    Args:
        events: Labeled events containing unique intervals and inline partitions.
        close: Dollar-bar close prices indexed by ``(symbol, end)``.

    Returns:
        Events with return-attribution and mean-one sample weights appended.

    Raises:
        ValueError: If required data is missing, duplicated, or cannot be weighted.
    """
    required_event_columns = {
        "symbol",
        "event_start",
        "event_end",
        "partition",
        "holdout_boundary",
    }
    missing_events = required_event_columns.difference(events.columns)
    if missing_events:
        raise ValueError(f"Events are missing columns: {sorted(missing_events)}")

    result = events.reset_index(drop=True).drop(
        columns=[*WEIGHT_COLUMNS, "average_uniqueness_weight", "time_decay_weight"],
        errors="ignore",
    ).copy()
    result["event_start"] = pd.to_datetime(
        result["event_start"], utc=True, errors="coerce"
    )
    result["event_end"] = pd.to_datetime(
        result["event_end"], utc=True, errors="coerce"
    )
    result["holdout_boundary"] = pd.to_datetime(
        result["holdout_boundary"], utc=True, errors="coerce"
    )
    if result[
        ["event_start", "event_end", "holdout_boundary"]
    ].isna().any().any():
        raise ValueError("Event metadata must contain valid timestamps.")
    if result["symbol"].isna().any() or result.duplicated(["symbol", "event_start"]).any():
        raise ValueError("Events require unique valid composite event keys")
    partitions = set(result["partition"].dropna().unique())
    if partitions != {"development", "holdout"}:
        raise ValueError("Events must contain development and holdout partitions.")
    if result["holdout_boundary"].nunique() != 1:
        raise ValueError("Events must contain one holdout boundary.")

    close_prices = close.astype(float).copy()
    if not isinstance(close.index, pd.MultiIndex) or close.index.names != ["symbol", "end"]:
        raise ValueError("Close prices must be indexed by (symbol, end)")
    times = pd.to_datetime(close.index.get_level_values("end"), utc=True, errors="coerce")
    close_prices.index = pd.MultiIndex.from_arrays(
        [close.index.get_level_values("symbol"), times], names=["symbol", "end"]
    )
    if (times.isna().any() or close_prices.index.get_level_values("symbol").isna().any()
            or close_prices.index.has_duplicates):
        raise ValueError("Close-price index must contain unique valid (symbol, end) keys.")
    close_prices = close_prices.sort_index()
    if close_prices.empty or not np.isfinite(close_prices).all():
        raise ValueError("Close prices must be finite and non-empty.")

    result["return_attribution_weight"] = np.nan
    for (symbol, partition), group in result.groupby(["symbol", "partition"]):
        prices = close_prices.xs(symbol, level="symbol").sort_index()
        group = group.sort_values("event_start")
        intervals = group.set_index("event_start")["event_end"]
        concurrency = count_concurrent_events(prices.index, intervals, intervals.index)
        attribution = compute_return_attribution_weights(intervals, concurrency, prices, intervals.index)
        result.loc[group.index, "return_attribution_weight"] = attribution.reindex(group.event_start).to_numpy()
    for partition, group in result.groupby("partition"):
        attribution = group.return_attribution_weight
        floor = attribution.loc[attribution.gt(0)].min()
        if pd.isna(floor):
            raise ValueError(f"{partition} return-attribution weights are all zero")
        weights = attribution.clip(lower=floor)
        result.loc[group.index, "sample_weight"] = weights / weights.mean()
    if result[WEIGHT_COLUMNS].isna().any().any():
        raise ValueError("Every composite event must receive complete weights")
    return result.sort_values(["event_start", "symbol"], ignore_index=True)
