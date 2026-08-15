import numpy as np
import pandas as pd
from loguru import logger


def apply_to_molecule(function, indexed_subset, num_threads, **kwargs):
    """Apply a labeling helper over the requested pandas index subset.

    Args:
        function: Worker function that accepts an ``event_index`` keyword argument.
        indexed_subset: Tuple of ``(name, values)`` describing the subset to process.
        num_threads: Retained for API compatibility; execution is sequential here.
        **kwargs: Extra keyword arguments passed to ``function``.

    Returns:
        The worker function output for the requested event index.
    """
    _, event_index = indexed_subset
    return function(event_index=event_index, **kwargs)


def get_daily_volatility(close_prices, span=100):
    """Estimate exponentially weighted daily volatility.

    Args:
        close_prices: Close price series indexed by timestamp.
        span: Span used by the exponentially weighted standard deviation.

    Returns:
        A series of daily volatility estimates aligned to ``close_prices``.
    """
    previous_day_positions = close_prices.index.searchsorted(
        close_prices.index - pd.Timedelta(days=1)
    )
    has_previous_day = previous_day_positions > 0
    current_positions = np.arange(close_prices.shape[0])[has_previous_day]
    previous_positions = previous_day_positions[has_previous_day] - 1

    daily_returns = pd.Series(
        close_prices.iloc[current_positions].values
        / close_prices.iloc[previous_positions].values
        - 1,
        index=close_prices.index[has_previous_day],
    )
    return daily_returns.ewm(span=span).std()


def get_vertical_barriers(event_times, close_prices, num_bars=1):
    """Set a vertical barrier a fixed number of bars after each event.

    Args:
        event_times: Event start timestamps.
        close_prices: Close price series used to locate future bars.
        num_bars: Number of bars ahead to place the vertical barrier.

    Returns:
        A series mapping each eligible event start time to its vertical barrier time.
    """
    event_index = pd.DatetimeIndex(event_times)
    event_positions = close_prices.index.get_indexer(event_index)
    vertical_barrier_times = {}

    for event_time, event_position in zip(event_index, event_positions):
        if event_position < 0:
            continue
        barrier_position = event_position + num_bars
        if barrier_position < len(close_prices.index):
            vertical_barrier_times[event_time] = close_prices.index[barrier_position]

    return pd.Series(vertical_barrier_times)


def apply_profit_taking_stop_loss_on_t1(
        close_prices,
        event_table,
        barrier_multipliers,
        event_index,
):
    """Locate horizontal barrier hits before the vertical barrier.

    Args:
        close_prices: Close price series.
        event_table: Event frame containing ``vertical_barrier``, ``target_return``,
            and ``event_side``.
        barrier_multipliers: Profit-taking and stop-loss multipliers.
        event_index: Subset of event start timestamps to process.

    Returns:
        A frame with vertical-barrier, stop-loss, and profit-taking timestamps.
    """
    selected_events = event_table.loc[event_index]
    barrier_hits = pd.DataFrame(
        index=selected_events.index,
        columns=["vertical_barrier", "stop_loss", "profit_taking"],
        dtype=object,
    )
    barrier_hits["vertical_barrier"] = selected_events["vertical_barrier"]

    if barrier_multipliers[0] > 0:
        profit_taking_thresholds = (
            barrier_multipliers[0] * selected_events["target_return"]
        )
    else:
        profit_taking_thresholds = pd.Series(index=event_table.index, dtype=float)

    if barrier_multipliers[1] > 0:
        stop_loss_thresholds = (
            -barrier_multipliers[1] * selected_events["target_return"]
        )
    else:
        stop_loss_thresholds = pd.Series(index=event_table.index, dtype=float)

    final_close_time = close_prices.index[-1]
    for event_time, vertical_barrier in selected_events[
            "vertical_barrier"
    ].fillna(final_close_time).items():
        price_path = close_prices.loc[event_time:vertical_barrier]
        adjusted_returns = (
            (price_path / close_prices.loc[event_time] - 1)
            * selected_events.at[event_time, "event_side"]
        )
        barrier_hits.loc[event_time, "stop_loss"] = adjusted_returns[
            adjusted_returns < stop_loss_thresholds.loc[event_time]
        ].index.min()
        barrier_hits.loc[event_time, "profit_taking"] = adjusted_returns[
            adjusted_returns > profit_taking_thresholds.loc[event_time]
        ].index.min()

    return barrier_hits


def get_events(
        close_prices,
        event_times,
        barrier_multipliers,
        target_returns,
        minimum_target_return,
        num_threads,
        vertical_barriers=None,
        event_sides=None,
):
    """Build the event table used by triple-barrier labeling.

    Args:
        close_prices: Close price series.
        event_times: Event start times.
        barrier_multipliers: Profit-taking and stop-loss multipliers.
        target_returns: Target return series.
        minimum_target_return: Minimum target return required to keep an event.
        num_threads: Number of worker threads for the barrier search.
        vertical_barriers: Optional vertical barrier times.
        event_sides: Optional side predictions for meta-labeling.

    Returns:
        An event frame with ``event_end``, ``target_return``, and optional
        ``event_side`` columns.
    """
    selected_target_returns = target_returns.loc[event_times]
    selected_target_returns = selected_target_returns[
        selected_target_returns > minimum_target_return
    ]

    if vertical_barriers is None:
        vertical_barriers = pd.Series(pd.NaT, index=event_times)

    if event_sides is None:
        effective_event_sides = pd.Series(1.0, index=selected_target_returns.index)
        effective_barrier_multipliers = [
            barrier_multipliers[0],
            barrier_multipliers[0],
        ]
    else:
        effective_event_sides = event_sides.loc[selected_target_returns.index]
        effective_barrier_multipliers = barrier_multipliers[:2]

    event_table = pd.concat(
        {
            "vertical_barrier": vertical_barriers,
            "target_return": selected_target_returns,
            "event_side": effective_event_sides,
        },
        axis=1,
    ).dropna(subset=["target_return"])

    barrier_hits = apply_to_molecule(
        function=apply_profit_taking_stop_loss_on_t1,
        indexed_subset=("event_index", event_table.index),
        num_threads=num_threads,
        close_prices=close_prices,
        event_table=event_table,
        barrier_multipliers=effective_barrier_multipliers,
    )

    def _get_earliest_barrier_time(row):
        """Select the earliest non-missing barrier timestamp."""
        barrier_times = [value for value in row if pd.notna(value)]
        return min(barrier_times) if barrier_times else pd.NaT

    event_table["event_end"] = barrier_hits.apply(
        _get_earliest_barrier_time,
        axis=1,
    )
    event_table = event_table[["event_end", "target_return", "event_side"]]

    if event_sides is None:
        event_table = event_table.drop("event_side", axis=1)

    return event_table


def get_bins(event_table, close_prices):
    """Convert event outcomes into return and label pairs.

    Args:
        event_table: Event frame with ``event_end`` and optional ``event_side``.
        close_prices: Close price series covering event starts and ends.

    Returns:
        A frame containing ``realized_return`` and ``label`` columns.
    """
    completed_events = event_table.dropna(subset=["event_end"])
    label_table = pd.DataFrame(index=completed_events.index)
    start_prices = close_prices.reindex(completed_events.index, method="bfill")
    event_end_times = pd.DatetimeIndex(completed_events["event_end"].tolist())
    end_prices = close_prices.reindex(event_end_times, method="bfill")
    label_table["realized_return"] = (
        end_prices.to_numpy() / start_prices.to_numpy() - 1
    )

    if "event_side" in completed_events:
        label_table["realized_return"] *= completed_events["event_side"]

    label_table["label"] = np.sign(label_table["realized_return"])

    if "event_side" in completed_events:
        label_table.loc[label_table["realized_return"] <= 0, "label"] = 0

    return label_table


def drop_labels(labeled_events, minimum_frequency=0.05):
    """Remove labels whose relative frequency falls below a threshold.

    Args:
        labeled_events: Event frame containing a ``label`` column.
        minimum_frequency: Minimum class frequency required to keep a label.

    Returns:
        The filtered event frame.
    """
    while True:
        label_frequencies = labeled_events["label"].value_counts(normalize=True)
        if (
                label_frequencies.min() > minimum_frequency
                or label_frequencies.shape[0] < 3
        ):
            break
        rare_label = label_frequencies.idxmin()
        logger.debug(
            "Dropping label {} with frequency {}.",
            rare_label,
            label_frequencies.loc[rare_label],
        )
        labeled_events = labeled_events[labeled_events["label"] != rare_label]
    return labeled_events
