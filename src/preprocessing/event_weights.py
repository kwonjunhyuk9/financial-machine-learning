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
        close: Dollar-bar close prices indexed by bar end time.

    Returns:
        Events with return-attribution and mean-one sample weights appended.

    Raises:
        ValueError: If required data is missing, duplicated, or cannot be weighted.
    """
    required_event_columns = {
        "event_start",
        "event_end",
        "partition",
        "holdout_boundary",
    }
    missing_events = required_event_columns.difference(events.columns)
    if missing_events:
        raise ValueError(f"Events are missing columns: {sorted(missing_events)}")

    weighted_input = events.drop(
        columns=[*WEIGHT_COLUMNS, "average_uniqueness_weight", "time_decay_weight"],
        errors="ignore",
    ).copy()
    weighted_input["event_start"] = pd.to_datetime(
        weighted_input["event_start"], utc=True, errors="coerce"
    )
    weighted_input["event_end"] = pd.to_datetime(
        weighted_input["event_end"], utc=True, errors="coerce"
    )
    weighted_input["holdout_boundary"] = pd.to_datetime(
        weighted_input["holdout_boundary"], utc=True, errors="coerce"
    )
    if weighted_input[
        ["event_start", "event_end", "holdout_boundary"]
    ].isna().any().any():
        raise ValueError("Event metadata must contain valid timestamps.")
    if weighted_input["event_start"].duplicated().any():
        raise ValueError("Event starts must be unique.")
    partitions = set(weighted_input["partition"].dropna().unique())
    if partitions != {"development", "holdout"}:
        raise ValueError("Events must contain development and holdout partitions.")
    if weighted_input["holdout_boundary"].nunique() != 1:
        raise ValueError("Events must contain one holdout boundary.")

    close_prices = close.astype(float).copy()
    close_prices.index = pd.to_datetime(close_prices.index, utc=True, errors="coerce")
    if close_prices.index.isna().any() or close_prices.index.duplicated().any():
        raise ValueError("Close-price index must contain unique valid timestamps.")
    close_prices = close_prices.sort_index()
    if close_prices.empty or not np.isfinite(close_prices).all():
        raise ValueError("Close prices must be finite and non-empty.")

    weight_tables = []
    for partition in ["development", "holdout"]:
        partition_events = weighted_input.loc[
            weighted_input["partition"].eq(partition)
        ].set_index("event_start")
        if partition_events.empty:
            raise ValueError(f"{partition} must contain at least one event.")

        information_sets = partition_events["event_end"]
        concurrency = count_concurrent_events(
            close_prices.index,
            information_sets,
            information_sets.index,
        )
        return_attribution = compute_return_attribution_weights(
            information_sets,
            concurrency,
            close_prices,
            information_sets.index,
        )
        positive_floor = return_attribution[return_attribution.gt(0)].min()
        if pd.isna(positive_floor):
            raise ValueError(
                f"{partition} return-attribution weights are all zero."
            )
        base_weight = return_attribution.clip(lower=positive_floor)
        sample_weight = base_weight / base_weight.mean()
        weight_tables.append(
            pd.DataFrame(
                {
                    "return_attribution_weight": return_attribution,
                    "sample_weight": sample_weight,
                }
            ).rename_axis("event_start").reset_index()
        )

    weight_table = pd.concat(weight_tables, ignore_index=True)
    weighted = weighted_input.merge(
        weight_table,
        on="event_start",
        how="left",
        validate="one_to_one",
    ).sort_values("event_start", ignore_index=True)
    if weighted[WEIGHT_COLUMNS].isna().any().any():
        raise ValueError("Every retained event must receive complete weights.")
    return weighted
