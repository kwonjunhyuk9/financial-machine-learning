from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from loguru import logger


def get_bar_horizon_volatility(
    close_prices: pd.Series,
    horizon_bars: int = 1_000,
    span: int = 100,
) -> pd.Series:
    """Estimate volatility for returns over a fixed bar horizon.

    Args:
        close_prices: Close price series indexed by timestamp.
        horizon_bars: Number of bars spanned by each return observation.
        span: Span used by the exponentially weighted standard deviation.

    Returns:
        A volatility series aligned to ``close_prices``.
    """
    horizon_returns = close_prices.pct_change(
        periods=horizon_bars,
        fill_method=None,
    )
    return horizon_returns.ewm(
        span=span,
        adjust=True,
    ).std(
        bias=False,
    )


def get_vertical_barriers(
    event_times: pd.Index | Sequence[pd.Timestamp],
    close_prices: pd.Series,
    num_bars: int = 1,
) -> pd.Series:
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
    close_prices: pd.Series,
    event_table: pd.DataFrame,
    barrier_multipliers: Sequence[float],
    event_index: pd.Index,
) -> pd.DataFrame:
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
    close_prices: pd.Series,
    event_times: pd.Index,
    barrier_multipliers: Sequence[float],
    target_returns: pd.Series,
    minimum_target_return: float,
    vertical_barriers: pd.Series | None = None,
    event_sides: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the event table used by triple-barrier labeling.

    Args:
        close_prices: Close price series.
        event_times: Event start times.
        barrier_multipliers: Profit-taking and stop-loss multipliers.
        target_returns: Target return series.
        minimum_target_return: Minimum target return required to keep an event.
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

    barrier_hits = apply_profit_taking_stop_loss_on_t1(
        event_index=event_table.index,
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


def get_bins(event_table: pd.DataFrame, close_prices: pd.Series) -> pd.DataFrame:
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


def drop_labels(
    labeled_events: pd.DataFrame,
    minimum_frequency: float = 0.05,
) -> pd.DataFrame:
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


def build_labeled_event_data(
        candidate_split: pd.DataFrame,
        dollar_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add triple-barrier outcomes to a pre-split event feature schema.

    Missing feature values are preserved for a later cleaning stage and do not
    affect labeling eligibility. Labeling parameters and retained classes are
    derived from development candidates only.

    Args:
        candidate_split: Integrated feature rows with fixed partition metadata.
        dollar_bars: Dollar bars containing completed timestamps and close prices.

    Returns:
        The 60-column labeled event data and final partition manifest.

    Raises:
        ValueError: If the input schema or fixed partition contract is invalid.
    """
    candidate_metadata = {
        "event_start",
        "symbol",
        "partition",
        "holdout_boundary",
    }
    required_features = {
        "mean_sentiment_score",
        "fractionally_differenced_log_close",
    }
    missing_candidates = (
        candidate_metadata | required_features
    ).difference(candidate_split.columns)
    missing_bars = {"end", "close"}.difference(dollar_bars.columns)
    if missing_candidates:
        raise ValueError(
            f"Candidate split is missing columns: {sorted(missing_candidates)}"
        )
    if missing_bars:
        raise ValueError(f"Dollar bars are missing columns: {sorted(missing_bars)}")

    candidates = candidate_split.copy()
    candidates["event_start"] = pd.to_datetime(
        candidates["event_start"],
        utc=True,
        errors="coerce",
    )
    candidates["holdout_boundary"] = pd.to_datetime(
        candidates["holdout_boundary"],
        utc=True,
        errors="coerce",
    )
    bars = dollar_bars.copy()
    bars["end"] = pd.to_datetime(bars["end"], utc=True, errors="coerce")
    if candidates[["event_start", "holdout_boundary"]].isna().any().any():
        raise ValueError("Candidate split timestamps must be valid.")
    if bars["end"].isna().any():
        raise ValueError("Dollar bar end times must be valid.")
    if candidates["event_start"].duplicated().any():
        raise ValueError("Candidate event_start must be unique.")
    if bars["end"].duplicated().any():
        raise ValueError("Dollar bar end times must be unique.")

    partitions = set(candidates["partition"].dropna().unique())
    if partitions != {"development", "holdout"}:
        raise ValueError("Candidate split must contain development and holdout.")
    boundaries = candidates["holdout_boundary"].drop_duplicates()
    if len(boundaries) != 1:
        raise ValueError("Candidate split must contain one holdout boundary.")
    holdout_boundary = boundaries.item()
    development_starts = candidates.loc[
        candidates["partition"].eq("development"),
        "event_start",
    ]
    holdout_starts = candidates.loc[
        candidates["partition"].eq("holdout"),
        "event_start",
    ]
    if not development_starts.lt(holdout_boundary).all() or not holdout_starts.ge(
        holdout_boundary
    ).all():
        raise ValueError("Candidate partitions must respect holdout_boundary.")

    feature_columns = [
        column
        for column in candidates.columns
        if column not in candidate_metadata
    ]
    if len(feature_columns) != 53:
        raise ValueError("Candidate split must contain exactly 53 model features.")

    candidates = candidates.sort_values("event_start", kind="stable")
    candidate_indexed = candidates.set_index("event_start")
    partition_by_start = candidate_indexed["partition"]
    close = bars.sort_values("end").set_index("end")["close"].astype(float)
    bar_horizon_volatility = get_bar_horizon_volatility(
        close_prices=close,
        horizon_bars=1_000,
        span=100,
    )
    vertical_barriers = get_vertical_barriers(
        candidate_indexed.index,
        close,
        num_bars=1_000,
    ).rename("vertical_barrier")
    target_returns = bar_horizon_volatility.reindex(candidate_indexed.index)
    eligible = target_returns.dropna().index.intersection(vertical_barriers.index)
    development_eligible = eligible.intersection(
        partition_by_start[partition_by_start.eq("development")].index
    )
    if development_eligible.empty:
        raise ValueError("No development candidates are eligible for labeling.")
    minimum_target = float(
        target_returns.loc[development_eligible].quantile(0.25)
    )

    events = get_events(
        close_prices=close,
        event_times=eligible,
        barrier_multipliers=[1.0, 1.0],
        target_returns=target_returns,
        minimum_target_return=minimum_target,
        vertical_barriers=vertical_barriers,
    )
    labels = get_bins(events, close)
    labeled = events.join(vertical_barriers).join(labels).rename(
        columns={"realized_return": "raw_return", "label": "direction_label"}
    )
    development_labeled = labeled.loc[
        labeled.index.intersection(development_eligible)
    ]
    retained_development = drop_labels(
        development_labeled.rename(columns={"direction_label": "label"}),
        minimum_frequency=0.10,
    )
    retained_labels = set(retained_development["label"].astype("int8"))
    if retained_labels != {-1, 1}:
        raise ValueError("Development labels must retain classes -1 and 1.")

    directional = labeled[
        labeled["direction_label"].isin(retained_labels)
    ].copy()
    directional["direction_label"] = directional["direction_label"].astype(
        "int8"
    )
    directional["partition"] = partition_by_start.reindex(directional.index)
    overlap = (
        directional["partition"].eq("development")
        & directional["event_end"].ge(holdout_boundary)
    )
    directional.loc[overlap, "partition"] = "holdout_overlap_purged"
    partition_manifest = directional.reset_index(names="event_start")[
        ["event_start", "event_end", "partition"]
    ]
    partition_manifest["holdout_boundary"] = holdout_boundary

    retained = directional[
        directional["partition"].isin(["development", "holdout"])
    ].drop(columns="partition")
    model_data = retained.join(
        candidate_indexed.loc[:, ["symbol", *feature_columns]]
    ).rename_axis("event_start").reset_index()
    metadata_columns = [
        "event_start",
        "symbol",
        "event_end",
        "vertical_barrier",
        "target_return",
        "raw_return",
        "direction_label",
    ]
    model_data = model_data.loc[
        :, [*metadata_columns, *feature_columns]
    ].sort_values("event_start", ignore_index=True)

    if model_data.shape[1] != 60:
        raise ValueError("Labeled event data must contain exactly 60 columns.")
    development_ends = partition_manifest.loc[
        partition_manifest["partition"].eq("development"),
        "event_end",
    ]
    if not development_ends.lt(holdout_boundary).all():
        raise ValueError("Development events must end before the holdout boundary.")
    return model_data, partition_manifest
