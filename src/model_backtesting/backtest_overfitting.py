from itertools import combinations
from functools import partial

import numpy as np
import pandas as pd


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Compute the annualized Sharpe ratio for each strategy column.

    Args:
        returns: Strategy returns with observations in rows and strategies in columns.
        risk_free_rate: Per-period risk-free return.
        periods_per_year: Number of return periods in one year.

    Returns:
        A series of Sharpe ratios indexed by strategy name.
    """
    if isinstance(risk_free_rate, pd.Series):
        excess_returns = returns.sub(risk_free_rate, axis=0)
    else:
        excess_returns = returns - risk_free_rate

    out = excess_returns.mean() / excess_returns.std(ddof=1)

    if periods_per_year is not None:
        out *= periods_per_year ** 0.5

    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def get_sharpe_ratio_metric(annual_risk_free_rate=0.0, periods_per_year=252):
    """Create a Sharpe ratio metric function from annual assumptions.

    Args:
        annual_risk_free_rate: Annualized risk-free return to convert into a per-period rate
        before computing excess returns.
        periods_per_year: Number of return periods in one year.

    Returns:
        A callable that accepts a strategy return frame and returns annualized excess Sharpe
        ratios by strategy.

    Raises:
        ValueError: If ``periods_per_year`` is not positive.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    risk_free_rate = annual_risk_free_rate / periods_per_year

    return partial(
        sharpe_ratio,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year
    )


def combinatorial_symmetric_cross_validation(
        returns,
        num_partitions=16,
        metric_func=sharpe_ratio
):
    """Run combinatorially symmetric cross-validation.

    Args:
        returns: Strategy returns with observations in rows and strategies in columns.
        num_partitions: Even number of equal-sized row partitions.
        metric_func: Function that scores a return frame by strategy and returns one value
        per column.

    Returns:
        A frame with one row per CSCV split.
    """
    returns = _validate_returns(returns=returns, num_partitions=num_partitions)
    partitions = np.array_split(np.arange(returns.shape[0]), num_partitions)
    split_size = num_partitions // 2

    out = []

    for train_partitions in combinations(range(num_partitions), split_size):
        test_partitions = tuple(
            partition
            for partition in range(num_partitions)
            if partition not in train_partitions
        )

        train_indices = np.concatenate([
            partitions[partition]
            for partition in train_partitions
        ])
        test_indices = np.concatenate([
            partitions[partition]
            for partition in test_partitions
        ])

        train_metrics = _get_strategy_metrics(
            returns=returns.iloc[train_indices],
            metric_func=metric_func
        )
        test_metrics = _get_strategy_metrics(
            returns=returns.iloc[test_indices],
            metric_func=metric_func
        )

        best_strategy = train_metrics.idxmax()
        relative_rank = _relative_rank(metrics=test_metrics, strategy=best_strategy)
        logit = _logit(relative_rank)

        out.append({
            "train_partitions": train_partitions,
            "test_partitions": test_partitions,
            "best_strategy": best_strategy,
            "is_metric": train_metrics[best_strategy],
            "oos_metric": test_metrics[best_strategy],
            "relative_rank": relative_rank,
            "logit": logit
        })

    return pd.DataFrame(out)


def probability_of_backtest_overfitting(
        returns,
        num_partitions=16,
        metric_func=sharpe_ratio,
        threshold=0.0
):
    """Estimate the probability of backtest overfitting from CSCV logits.

    Args:
        returns: Strategy returns with observations in rows and strategies in columns.
        num_partitions: Even number of equal-sized row partitions.
        metric_func: Function that scores a return frame by strategy and returns one value
        per column.
        threshold: Logit threshold used to classify underperformance.

    Returns:
        The PBO estimate.

    Raises:
        ValueError: If CSCV produces no valid logits.
    """
    cscv = combinatorial_symmetric_cross_validation(
        returns=returns,
        num_partitions=num_partitions,
        metric_func=metric_func
    )
    logits = cscv["logit"].dropna()

    if logits.empty:
        raise ValueError("CSCV produced no valid logits")

    pbo = (logits <= threshold).mean()

    return pbo


def _validate_returns(returns, num_partitions):
    """Validate and coerce the CSCV performance matrix.

    Args:
        returns: Candidate performance matrix.
        num_partitions: Requested number of equal-sized row partitions.

    Returns:
        The return matrix coerced to ``float64``.

    Raises:
        ValueError: If the return matrix or partition count is invalid.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pd.DataFrame")

    if returns.shape[1] < 2:
        raise ValueError("returns must contain at least two strategy columns")

    if num_partitions < 2 or num_partitions % 2 != 0:
        raise ValueError("num_partitions must be an even integer greater than 1")

    if num_partitions > returns.shape[0]:
        raise ValueError("num_partitions cannot exceed the number of observations")

    if returns.shape[0] % num_partitions != 0:
        raise ValueError("returns rows must divide evenly into num_partitions")

    returns = returns.astype("float64")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns must contain only finite values")

    return returns


def _relative_rank(metrics, strategy):
    """Compute a strategy's relative rank in the open interval ``(0, 1)``.

    Args:
        metrics: Strategy performance statistics.
        strategy: Strategy label whose rank is evaluated.

    Returns:
        The strategy rank divided by ``N + 1``, where ``N`` is the number of strategies.
    """
    rank = metrics.rank(method="average")[strategy]

    return rank / (metrics.shape[0] + 1)


def _logit(value):
    """Compute the CSCV logit transform.

    Args:
        value: Relative rank value, typically in the interval ``(0, 1)``.

    Returns:
        ``log(value / (1 - value))``, or ``NaN`` when ``value`` is missing.
    """
    if pd.isna(value):
        return np.nan

    return np.log(value / (1 - value))


def _get_strategy_metrics(returns, metric_func):
    """Evaluate strategy metrics and align them to the return columns.

    Args:
        returns: Strategy return matrix for one CSCV sample.
        metric_func: Function that scores a return frame by strategy.

    Returns:
        A ``float64`` series indexed like ``returns.columns``.

    Raises:
        ValueError: If ``metric_func`` produces no valid metrics.
    """
    metrics = metric_func(returns)

    if not isinstance(metrics, pd.Series):
        metrics = pd.Series(metrics, index=returns.columns)

    metrics = metrics.reindex(returns.columns).astype("float64")

    if metrics.dropna().empty:
        raise ValueError("metric_func produced no valid strategy metrics")

    return metrics
