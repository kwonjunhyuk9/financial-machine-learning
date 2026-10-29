from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_bar_horizon_volatility(
        close_prices: pd.Series,
        horizon_bars: int = 1_000,
        span: int = 100,
) -> pd.Series:
    """Estimate volatility for returns over a fixed bar horizon.

    Args:
        close_prices: Closing prices indexed by timestamp.
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
        close_prices: Closing prices.
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
) -> pd.DataFrame:
    """Locate horizontal barrier hits before the vertical barrier.

    Args:
        close_prices: Closing prices.
        event_table: Event frame containing ``vertical_barrier`` and ``horizon_volatility``.
        barrier_multipliers: Profit-taking and stop-loss multipliers.

    Returns:
        A frame with vertical-barrier, stop-loss, and profit-taking timestamps.
    """
    barrier_hits = pd.DataFrame(
        index=event_table.index,
        columns=["vertical_barrier", "stop_loss", "profit_taking"],
        dtype=object,
    )
    barrier_hits["vertical_barrier"] = event_table["vertical_barrier"]

    if barrier_multipliers[0] > 0:
        profit_taking_thresholds = (
                barrier_multipliers[0] * event_table["horizon_volatility"]
        )
    else:
        profit_taking_thresholds = pd.Series(index=event_table.index, dtype=float)

    if barrier_multipliers[1] > 0:
        stop_loss_thresholds = (
                -barrier_multipliers[1] * event_table["horizon_volatility"]
        )
    else:
        stop_loss_thresholds = pd.Series(index=event_table.index, dtype=float)

    final_close_time = close_prices.index[-1]
    for event_time, vertical_barrier in event_table[
        "vertical_barrier"
    ].fillna(final_close_time).items():
        price_path = close_prices.loc[event_time:vertical_barrier]
        returns = price_path / close_prices.loc[event_time] - 1
        barrier_hits.loc[event_time, "stop_loss"] = returns[
            returns < stop_loss_thresholds.loc[event_time]
            ].index.min()
        barrier_hits.loc[event_time, "profit_taking"] = returns[
            returns > profit_taking_thresholds.loc[event_time]
            ].index.min()

    return barrier_hits


def get_events(
        close_prices: pd.Series,
        event_times: pd.Index,
        barrier_multipliers: Sequence[float],
        horizon_volatility: pd.Series,
        minimum_horizon_volatility: float,
        vertical_barriers: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the event table used by triple-barrier labeling.

    Args:
        close_prices: Closing prices.
        event_times: Event start times.
        barrier_multipliers: Profit-taking and stop-loss multipliers.
        horizon_volatility: Estimated return volatility over the configured bar horizon.
        minimum_horizon_volatility: Minimum volatility required to keep an event.
        vertical_barriers: Vertical barrier timestamps.

    Returns:
        An event frame with ``event_end`` and ``horizon_volatility`` columns.
    """
    selected_horizon_volatility = horizon_volatility.loc[event_times]
    selected_horizon_volatility = selected_horizon_volatility[
        selected_horizon_volatility > minimum_horizon_volatility
        ]

    if vertical_barriers is None:
        vertical_barriers = pd.Series(pd.NaT, index=event_times)

    event_table = pd.concat(
        {
            "vertical_barrier": vertical_barriers,
            "horizon_volatility": selected_horizon_volatility,
        },
        axis=1,
    ).dropna(subset=["horizon_volatility"])

    barrier_hits = apply_profit_taking_stop_loss_on_t1(
        close_prices=close_prices,
        event_table=event_table,
        barrier_multipliers=barrier_multipliers,
    )

    def _get_earliest_barrier_time(row):
        """Select the earliest non-missing barrier timestamp."""
        barrier_times = [value for value in row if pd.notna(value)]
        return min(barrier_times) if barrier_times else pd.NaT

    event_table["event_end"] = barrier_hits.apply(
        _get_earliest_barrier_time,
        axis=1,
    )
    return event_table[["event_end", "horizon_volatility"]]


def get_bins(event_table: pd.DataFrame, close_prices: pd.Series) -> pd.DataFrame:
    """Convert event outcomes into return and label pairs.

    Args:
        event_table: Event frame with ``event_end``.
        close_prices: Closing prices.

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

    label_table["label"] = np.sign(label_table["realized_return"])

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
        labeled_events = labeled_events[labeled_events["label"] != rare_label]
    return labeled_events


def build_labeled_event_data(
        candidate_split: pd.DataFrame,
        dollar_bars: pd.DataFrame,
        *,
        return_horizon_bars: int,
        vertical_horizon_bars: int,
        volatility_span: int,
        minimum_volatility_quantile: float,
        barrier_multipliers: Sequence[float],
        minimum_label_frequency: float,
        show_progress: bool = False,
) -> pd.DataFrame:
    """Label candidate events using the triple-barrier method.

    Args:
        candidate_split: Integrated feature rows with fixed partition metadata.
        dollar_bars: Dollar bars containing completed timestamps and close prices.
        return_horizon_bars: Return horizon in bars.
        vertical_horizon_bars: Vertical barrier horizon in bars.
        volatility_span: Exponentially weighted volatility span.
        minimum_volatility_quantile: Volatility quantile for event selection.
        barrier_multipliers: Profit-taking and stop-loss multipliers.
        minimum_label_frequency: Minimum class frequency.
        show_progress: Whether to display progress.

    Returns:
        Labeled event data with inline partition metadata.
    """
    if candidate_split["symbol"].nunique() > 1:
        if candidate_split.duplicated(["symbol", "event_start"]).any():
            raise ValueError("Duplicate composite event keys")
        outputs, exclusions = [], []
        groups = candidate_split.groupby("symbol", sort=False)
        with tqdm(groups, total=len(groups), desc="Event labeling", disable=not show_progress) as progress:
            for symbol, group in progress:
                progress.set_postfix_str(symbol)
                if set(group.partition) != {"development", "holdout"}:
                    exclusions.append({"symbol": symbol, "events": len(group),
                                       "reason": "both partitions required for symbol calibration"})
                    continue
                labeled = _build_symbol_labeled_event_data(
                    group, dollar_bars.loc[dollar_bars.symbol.eq(symbol)],
                    return_horizon_bars=return_horizon_bars, vertical_horizon_bars=vertical_horizon_bars,
                    volatility_span=volatility_span, minimum_volatility_quantile=minimum_volatility_quantile,
                    barrier_multipliers=barrier_multipliers, minimum_label_frequency=minimum_label_frequency,
                )
                outputs.append(labeled)
                exclusions.append({"symbol": symbol, "events": len(group) - len(labeled),
                                   "reason": "label eligibility or holdout purge"})
        if not outputs:
            raise ValueError("No symbols eligible for development-calibrated labeling")
        result = pd.concat(outputs, ignore_index=True).sort_values(["event_start", "symbol"])
        result.attrs["exclusions"] = exclusions
        return result

    with tqdm(total=1, desc="Event labeling", disable=not show_progress) as progress:
        progress.set_postfix_str(
            ", ".join(candidate_split["symbol"].dropna().astype(str).unique())
        )
        result = _build_symbol_labeled_event_data(
            candidate_split, dollar_bars,
            return_horizon_bars=return_horizon_bars,
            vertical_horizon_bars=vertical_horizon_bars,
            volatility_span=volatility_span,
            minimum_volatility_quantile=minimum_volatility_quantile,
            barrier_multipliers=barrier_multipliers,
            minimum_label_frequency=minimum_label_frequency,
        )
        progress.update(1)
    return result


def _build_symbol_labeled_event_data(
        candidate_split: pd.DataFrame,
        dollar_bars: pd.DataFrame,
        *,
        return_horizon_bars: int,
        vertical_horizon_bars: int,
        volatility_span: int,
        minimum_volatility_quantile: float,
        barrier_multipliers: Sequence[float],
        minimum_label_frequency: float,
) -> pd.DataFrame:
    """Label one symbol using development-only calibration."""
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

    candidates = candidates.sort_values("event_start", kind="stable")
    candidate_indexed = candidates.set_index("event_start")
    partition_by_start = candidate_indexed["partition"]
    close = bars.sort_values("end").set_index("end")["close"].astype(float)
    bar_horizon_volatility = get_bar_horizon_volatility(
        close_prices=close,
        horizon_bars=return_horizon_bars,
        span=volatility_span,
    )
    vertical_barriers = get_vertical_barriers(
        candidate_indexed.index,
        close,
        num_bars=vertical_horizon_bars,
    ).rename("vertical_barrier")
    horizon_volatility = bar_horizon_volatility.reindex(candidate_indexed.index)
    eligible = horizon_volatility.dropna().index.intersection(vertical_barriers.index)
    development_eligible = eligible.intersection(
        partition_by_start[partition_by_start.eq("development")].index
    )
    if development_eligible.empty:
        raise ValueError("No development candidates are eligible for labeling.")
    minimum_horizon_volatility = float(
        horizon_volatility.loc[development_eligible].quantile(minimum_volatility_quantile)
    )

    events = get_events(
        close_prices=close,
        event_times=eligible,
        barrier_multipliers=barrier_multipliers,
        horizon_volatility=horizon_volatility,
        minimum_horizon_volatility=minimum_horizon_volatility,
        vertical_barriers=vertical_barriers,
    )
    labels = get_bins(events, close)
    labeled = events.copy()
    labeled.insert(1, "vertical_barrier", vertical_barriers)
    labeled = labeled.join(labels).rename(
        columns={"realized_return": "raw_return", "label": "primary_label"}
    )
    development_labeled = labeled.loc[
        labeled.index.intersection(development_eligible)
    ]
    retained_development = drop_labels(
        development_labeled.rename(columns={"primary_label": "label"}),
        minimum_frequency=minimum_label_frequency,
    )
    retained_labels = set(retained_development["label"].astype("int8"))

    directional = labeled[
        labeled["primary_label"].isin(retained_labels)
    ].copy()
    directional["primary_label"] = directional["primary_label"].astype(
        "int8"
    )
    directional["holdout_boundary"] = holdout_boundary
    directional["partition"] = partition_by_start.reindex(directional.index)
    overlap = (
            directional["partition"].eq("development")
            & directional["event_end"].ge(holdout_boundary)
    )
    retained = directional.loc[~overlap].copy()
    model_data = retained.join(
        candidate_indexed[feature_columns]
    ).rename_axis("event_start").reset_index()
    model_data.insert(0, "symbol", candidates["symbol"].iloc[0])
    model_data = model_data.sort_values("event_start", ignore_index=True)

    return model_data
