from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


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


def probability_bet_size(probability: pd.Series) -> pd.Series:
    """Transform binary take probabilities before any averaging, including 0/1."""
    if not probability.between(0, 1).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    values = probability.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (values - 0.5) / np.sqrt(values * (1 - values))
    return pd.Series(2 * norm.cdf(z) - 1, index=probability.index)


def event_bet_signals(events: pd.DataFrame) -> pd.Series:
    """Return signed signals; pass events remain zero-valued active observations."""
    if not events.primary_side.isin([-1, 1]).all() or not events.meta_action.isin([0, 1]).all():
        raise ValueError("Invalid primary side or meta action")
    if not events.loc[events.meta_action.eq(1), "meta_probability"].ge(0.5).all():
        raise ValueError("Take probabilities must be at least 0.5")
    return probability_bet_size(events.meta_probability) * events.primary_side * events.meta_action


def average_symbol_targets(active_events: pd.DataFrame, k: int = 5,
                           step_size: float = 0.10) -> pd.Series:
    """Average only within a symbol, then discretize and apply the 1/(2K) cap.

    Input contains only events known to be active at the observation time. This
    helper never reads future event_end values or averages across securities.
    """
    if k < 1 or not 0 < step_size <= 1:
        raise ValueError("Invalid K or discretization step")
    if active_events.empty:
        return pd.Series(dtype=float, name="target_weight")
    frame = active_events.copy()
    frame["signal"] = event_bet_signals(frame)
    average = frame.groupby("symbol", sort=True).signal.mean()
    return (discretize_signal(average, step_size) / (2 * k)).rename("target_weight")
