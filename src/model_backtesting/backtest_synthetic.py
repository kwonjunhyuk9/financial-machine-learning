from itertools import product

import numpy as np
import pandas as pd


def synthetic_trading_rule_experiment(
        forecasts,
        half_lives,
        profit_taking_range,
        stop_loss_range,
        sigma=1.0,
        num_iterations=10000,
        max_holding_period=100,
        random_state=0
):
    """Run the AFML synthetic trading-rule experiment.

    Args:
        forecasts: Long-run equilibrium values to test.
        half_lives: Half-life values to test.
        profit_taking_range: Non-negative profit-taking thresholds.
        stop_loss_range: Non-negative stop-loss magnitudes.
        sigma: Shock volatility.
        num_iterations: Number of synthetic paths per process parameter set.
        max_holding_period: Vertical barrier, measured in observations.
        random_state: Seed for the random number generator.

    Returns:
        A frame with one row per process and trading-rule combination.
    """
    outputs = []

    for forecast, half_life in product(forecasts, half_lives):
        result = synthetic_trading_rule_sharpe_ratios(
            forecast=forecast,
            half_life=half_life,
            sigma=sigma,
            num_iterations=num_iterations,
            max_holding_period=max_holding_period,
            profit_taking_range=profit_taking_range,
            stop_loss_range=stop_loss_range,
            random_state=random_state
        )
        result.insert(0, "forecast", forecast)
        result.insert(1, "half_life", half_life)
        result.insert(2, "sigma", sigma)
        result.insert(3, "max_holding_period", max_holding_period)
        outputs.append(result)

    return pd.concat(outputs, ignore_index=True)


def synthetic_trading_rule_sharpe_ratios(
        forecast,
        half_life,
        profit_taking_range,
        stop_loss_range,
        sigma=1.0,
        num_iterations=10000,
        max_holding_period=100,
        random_state=0
):
    """Compute synthetic trading-rule Sharpe ratios.

    Args:
        forecast: Long-run equilibrium value of the synthetic price process.
        half_life: Half-life of the Ornstein-Uhlenbeck process.
        profit_taking_range: Non-negative profit-taking thresholds.
        stop_loss_range: Non-negative stop-loss magnitudes.
        sigma: Shock volatility.
        num_iterations: Number of synthetic paths to simulate.
        max_holding_period: Vertical barrier, measured in observations.
        random_state: Seed for the random number generator.

    Returns:
        A frame with one row per trading rule and columns for profit-taking threshold,
        stop-loss magnitude, terminal outcome mean, terminal outcome standard deviation,
        and non-annualized Sharpe ratio.
    """
    num_iterations = int(num_iterations)
    max_holding_period = int(max_holding_period)

    if half_life <= 0:
        raise ValueError("half_life must be positive")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")

    if max_holding_period <= 0:
        raise ValueError("max_holding_period must be positive")

    profit_taking_range = np.atleast_1d(
        np.asarray(profit_taking_range, dtype="float64")
    )
    stop_loss_range = np.atleast_1d(np.asarray(stop_loss_range, dtype="float64"))

    if (profit_taking_range < 0).any() or (stop_loss_range < 0).any():
        raise ValueError("thresholds must be non-negative")

    rng = np.random.default_rng(random_state)
    phi = 2 ** (-1.0 / half_life)
    paths = np.empty((num_iterations, max_holding_period))
    prices = np.zeros(num_iterations)

    for step in range(max_holding_period):
        shocks = rng.normal(0.0, sigma, num_iterations)
        prices = (1.0 - phi) * forecast + phi * prices + shocks
        paths[:, step] = prices

    out = []

    for profit_taking, stop_loss in product(profit_taking_range, stop_loss_range):
        touched = (paths > profit_taking) | (paths < -stop_loss)
        exit_positions = np.where(
            touched.any(axis=1),
            touched.argmax(axis=1),
            max_holding_period - 1
        )
        outcomes = paths[np.arange(num_iterations), exit_positions]
        mean = outcomes.mean()
        std = outcomes.std(ddof=0)

        out.append({
            "profit_taking": profit_taking,
            "stop_loss": stop_loss,
            "mean": mean,
            "std": std,
            "sharpe_ratio": np.nan if std == 0 else mean / std
        })

    return pd.DataFrame(out)
