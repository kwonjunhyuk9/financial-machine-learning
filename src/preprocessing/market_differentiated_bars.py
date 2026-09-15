from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def get_weights_fixed_width(
    differencing_order: float,
    weight_cutoff: float,
) -> np.ndarray:
    """Compute fixed-width fractional differencing weights.

    Args:
        differencing_order: Fractional differencing order.
        weight_cutoff: Absolute weight cutoff threshold.

    Returns:
        A column vector of weights ordered from oldest to newest.
    """
    weights = [1.0]
    lag = 1
    while True:
        weight = (
            -weights[-1] / lag * (differencing_order - lag + 1)
        )
        if abs(weight) < weight_cutoff:
            break
        weights.append(weight)
        lag += 1
    return np.array(weights[::-1]).reshape(-1, 1)


def fractional_difference_fixed_width(
    series_frame: pd.DataFrame,
    differencing_order: float,
    weight_cutoff: float = 1e-5,
) -> pd.DataFrame:
    """Apply fixed-width fractional differencing to each column.

    Args:
        series_frame: Input time series frame.
        differencing_order: Fractional differencing order.
        weight_cutoff: Weight cutoff threshold used to set the window width.

    Returns:
        A frame of fractionally differenced series.
    """
    weights = get_weights_fixed_width(differencing_order, weight_cutoff)
    width = len(weights) - 1

    differentiated_columns = {}
    for column_name in series_frame.columns:
        filled_series = series_frame[[column_name]].ffill().dropna()
        differentiated_series = pd.Series()
        for position in range(width, filled_series.shape[0]):
            start_index = filled_series.index[position - width]
            end_index = filled_series.index[position]
            if not np.isfinite(series_frame.loc[end_index, column_name]):
                continue
            differentiated_series.loc[end_index] = np.dot(
                weights.T,
                filled_series.loc[start_index:end_index],
            )[0, 0]
        differentiated_columns[column_name] = differentiated_series.copy(
            deep=True
        )
    return pd.concat(differentiated_columns, axis=1)


def evaluate_fractional_differencing_orders(
    log_price_series: pd.Series | pd.DataFrame,
    weight_cutoff: float = 0.01,
    differencing_orders: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate stationarity across fractional differencing orders.

    Args:
        log_price_series: Time-indexed Series or single-column DataFrame of log
            prices.
        weight_cutoff: Weight cutoff passed to fixed-width differencing.
        differencing_orders: Optional iterable of fractional differencing orders.

    Returns:
        A DataFrame with ADF statistics and correlations by differencing order.

    Raises:
        ValueError: If ``log_price_series`` is not a Series or single-column
            DataFrame.
    """
    from statsmodels.tsa.stattools import adfuller

    diagnostics = pd.DataFrame(
        columns=[
            'adf_statistic',
            'p_value',
            'used_lags',
            'n_observations',
            'critical_value_5pct',
            'correlation',
        ]
    )
    if isinstance(log_price_series, pd.Series):
        log_price_frame = log_price_series.to_frame(
            name=log_price_series.name or 'log_close'
        )
    else:
        log_price_frame = log_price_series.copy()
    if log_price_frame.shape[1] != 1:
        raise ValueError(
            'log_price_series must be a Series or single-column DataFrame'
        )
    column_name = log_price_frame.columns[0]
    if differencing_orders is None:
        differencing_orders = np.linspace(0, 1, 11)

    log_prices = log_price_frame[[column_name]].dropna()

    for differencing_order in differencing_orders:
        differentiated_prices = fractional_difference_fixed_width(
            log_prices,
            differencing_order,
            weight_cutoff=weight_cutoff,
        )
        correlation = np.corrcoef(
            log_prices.loc[differentiated_prices.index, column_name],
            differentiated_prices[column_name],
        )[0, 1]
        adf_result = adfuller(
            differentiated_prices[column_name],
            maxlag=1,
            regression='c',
            autolag=None,
        )
        diagnostics.loc[differencing_order] = (
            list(adf_result[:4])
            + [adf_result[4]['5%']]
            + [correlation]
        )

    return diagnostics
