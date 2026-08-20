from __future__ import annotations

import numpy as np
from scipy import stats


def implied_precision(
    stop_loss: float | np.ndarray,
    profit_taking: float | np.ndarray,
    frequency: float | np.ndarray,
    target_sharpe: float | np.ndarray,
) -> np.ndarray:
    """Compute the minimum precision required to reach a target Sharpe ratio.

    Args:
        stop_loss: Average losing bet outcome.
        profit_taking: Average winning bet outcome.
        frequency: Number of bets per year.
        target_sharpe: Target annualized Sharpe ratio.

    Returns:
        The minimum precision rate required to reach ``target_sharpe``.
    """
    stop_loss = np.asarray(stop_loss, dtype="float64")
    profit_taking = np.asarray(profit_taking, dtype="float64")
    frequency = np.asarray(frequency, dtype="float64")
    target_sharpe = np.asarray(target_sharpe, dtype="float64")

    spread = profit_taking - stop_loss
    a = (frequency + target_sharpe ** 2) * spread ** 2
    b = (2 * frequency * stop_loss - target_sharpe ** 2 * spread) * spread
    c = frequency * stop_loss ** 2

    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def implied_betting_frequency(
    stop_loss: float | np.ndarray,
    profit_taking: float | np.ndarray,
    precision: float | np.ndarray,
    target_sharpe: float | np.ndarray,
) -> np.ndarray:
    """Compute the betting frequency required to reach a target Sharpe ratio.

    Args:
        stop_loss: Average losing bet outcome.
        profit_taking: Average winning bet outcome.
        precision: Probability of a winning bet.
        target_sharpe: Target annualized Sharpe ratio.

    Returns:
        The number of bets per year required to reach ``target_sharpe``.
    """
    stop_loss = np.asarray(stop_loss, dtype="float64")
    profit_taking = np.asarray(profit_taking, dtype="float64")
    precision = np.asarray(precision, dtype="float64")
    target_sharpe = np.asarray(target_sharpe, dtype="float64")

    spread = profit_taking - stop_loss
    frequency = (
        (target_sharpe * spread) ** 2
        * precision
        * (1 - precision)
        / (spread * precision + stop_loss) ** 2
    )
    sharpe = (
        (spread * precision + stop_loss)
        / (spread * np.sqrt(precision * (1 - precision)))
        * np.sqrt(frequency)
    )

    return np.where(np.isclose(sharpe, target_sharpe), frequency, np.nan)


def mix_gaussians(
    mean_1: float,
    mean_2: float,
    sigma_1: float,
    sigma_2: float,
    probability_1: float,
    num_observations: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw observations from a two-Gaussian mixture.

    Args:
        mean_1: Mean of the first Gaussian component.
        mean_2: Mean of the second Gaussian component.
        sigma_1: Standard deviation of the first Gaussian component.
        sigma_2: Standard deviation of the second Gaussian component.
        probability_1: Probability of drawing from the first component.
        num_observations: Number of observations to draw.
        random_state: Seed for the random number generator.

    Returns:
        A shuffled array of mixture observations.
    """
    rng = np.random.default_rng(random_state)
    num_observations = int(num_observations)
    num_1 = int(num_observations * probability_1)
    num_2 = num_observations - num_1

    observations = np.concatenate([
        rng.normal(mean_1, sigma_1, size=num_1),
        rng.normal(mean_2, sigma_2, size=num_2)
    ])
    rng.shuffle(observations)

    return observations


def probability_of_strategy_failure(
    returns: np.ndarray,
    frequency: float,
    target_sharpe: float,
) -> float | np.ndarray:
    """Estimate the probability that a strategy fails to reach a target Sharpe ratio.

    Args:
        returns: Bet outcomes.
        frequency: Number of bets per year.
        target_sharpe: Target annualized Sharpe ratio.

    Returns:
        The probability that precision falls below the target-implied precision.
    """
    returns = np.asarray(returns, dtype="float64")
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns <= 0]

    profit_taking = positive_returns.mean()
    stop_loss = negative_returns.mean()
    precision = positive_returns.shape[0] / float(returns.shape[0])
    threshold_precision = implied_precision(
        stop_loss=stop_loss,
        profit_taking=profit_taking,
        frequency=frequency,
        target_sharpe=target_sharpe
    )

    return stats.norm.cdf(
        threshold_precision,
        loc=precision,
        scale=precision * (1 - precision)
    )
