from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from scipy.stats import norm


def get_signal(
    events: pd.DataFrame,
    step_size: float,
    prob: pd.Series,
    pred: pd.Series,
    num_classes: int,
) -> pd.Series:
    """Compute discretized bet sizes from classification probabilities.

    Args:
        events: Event metadata indexed by event start time, optionally with ``side``.
        step_size: Discretization interval for bet sizes.
        prob: Predicted probability for the selected class.
        pred: Predicted side for standard labeling, or take/pass prediction for
            meta-labeling.
        num_classes: Number of possible classes in the classifier.

    Returns:
        A series of discretized bet sizes.
    """
    if prob.shape[0] == 0:
        return pd.Series(dtype="float64")

    signal = (prob - 1.0 / num_classes) / (prob * (1.0 - prob)) ** 0.5
    signal = pred * (2 * norm.cdf(signal) - 1)

    if "side" in events:
        signal *= events.loc[signal.index, "side"]

    active_signals = signal.to_frame("signal").join(events[["t1"]], how="left")
    averaged_signals = average_active_signals(active_signals)
    discretized_signal = discretize_signal(
        signal=averaged_signals,
        step_size=step_size
    )

    return discretized_signal


def average_active_signals(signals: pd.DataFrame) -> pd.Series:
    """Average all active bet signals at each event boundary.

    Args:
        signals: Frame with ``signal`` values and ``t1`` event end times.

    Returns:
        A series indexed by signal evaluation times.
    """
    time_points = signals.index.union(pd.Index(signals["t1"].dropna())).sort_values()

    out = average_active_signals_for_times(signals=signals, time_points=time_points)

    return out


def average_active_signals_for_times(
    signals: pd.DataFrame,
    time_points: Sequence[pd.Timestamp],
) -> pd.Series:
    """Average active signals for the supplied evaluation times.

    Args:
        signals: Frame with ``signal`` values and ``t1`` event end times.
        time_points: Evaluation times.

    Returns:
        A series of average active signals.
    """
    out = pd.Series(dtype="float64")

    for loc in time_points:
        active_idx = (
                (signals.index <= loc)
                & ((loc < signals["t1"]) | pd.isnull(signals["t1"]))
        )

        active_events = signals[active_idx].index

        if len(active_events) > 0:
            out[loc] = signals.loc[active_events, "signal"].mean()
        else:
            out[loc] = 0

    return out


def discretize_signal(signal: pd.Series, step_size: float) -> pd.Series:
    """Discretize a signal series into bounded bet sizes.

    Args:
        signal: Raw bet size signal.
        step_size: Discretization interval.

    Returns:
        A series clipped to the interval ``[-1, 1]``.
    """
    discretized_signal = (signal / step_size).round() * step_size
    discretized_signal[discretized_signal > 1] = 1
    discretized_signal[discretized_signal < -1] = -1

    return discretized_signal


def bet_size(w: float, price_divergence: float) -> float:
    """Compute dynamic bet size from forecast-market price divergence.

    Args:
        w: Calibration coefficient.
        price_divergence: Difference between forecast and market prices.

    Returns:
        A continuous bet size in the interval ``(-1, 1)``.
    """
    return price_divergence * (w + price_divergence ** 2) ** -0.5


def get_target_position(
    w: float,
    forecast_price: float,
    market_price: float,
    max_position: int,
) -> int:
    """Compute the target position implied by the dynamic bet size.

    Args:
        w: Calibration coefficient.
        forecast_price: Forecast price.
        market_price: Current market price.
        max_position: Maximum absolute position size.

    Returns:
        Integer target position.
    """
    return int(bet_size(w, forecast_price - market_price) * max_position)


def inverse_price(
    forecast_price: float,
    w: float,
    bet_size_value: float,
) -> float:
    """Compute the price implied by a target bet size.

    Args:
        forecast_price: Forecast price.
        w: Calibration coefficient.
        bet_size_value: Target bet size.

    Returns:
        Implied price for the target bet size.
    """
    return forecast_price - bet_size_value * (w / (1 - bet_size_value ** 2)) ** 0.5


def limit_price(
    target_position: int,
    current_position: int,
    forecast_price: float,
    w: float,
    max_position: int,
) -> float | None:
    """Compute the limit price for moving from current to target position.

    Args:
        target_position: Desired target position.
        current_position: Current position.
        forecast_price: Forecast price.
        w: Calibration coefficient.
        max_position: Maximum absolute position size.

    Returns:
        Average limit price for the required order, or ``None`` if no order is needed.

    Raises:
        ValueError: If an intermediate position reaches ``max_position``.
    """
    if target_position == current_position:
        return None

    step = 1 if target_position > current_position else -1
    positions = range(
        current_position + step,
        target_position + step,
        step
    )

    prices = []

    for position in positions:
        if abs(position) >= max_position:
            raise ValueError("position must stay within (-max_position, max_position)")

        prices.append(inverse_price(
            forecast_price=forecast_price,
            w=w,
            bet_size_value=position / float(max_position)
        ))

    return sum(prices) / len(prices)


def get_w(price_divergence: float, bet_size_value: float) -> float:
    """Calibrate ``w`` from a price divergence and target bet size.

    Args:
        price_divergence: Difference between forecast and market prices.
        bet_size_value: Target bet size, where ``0 < bet_size_value < 1``.

    Returns:
        Calibration coefficient.
    """
    return price_divergence ** 2 * (bet_size_value ** -2 - 1)
