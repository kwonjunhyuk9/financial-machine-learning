import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_weights(differencing_order, num_weights):
    """Compute fractional differencing weights.

    Args:
        differencing_order: Fractional differencing order.
        num_weights: Number of weights to generate.

    Returns:
        A column vector of weights ordered from oldest to newest.
    """
    weights = [1.0]
    for lag in range(1, num_weights):
        next_weight = (
            -weights[-1] / lag * (differencing_order - lag + 1)
        )
        weights.append(next_weight)
    return np.array(weights[::-1]).reshape(-1, 1)


def get_weights_fixed_width(differencing_order, weight_cutoff):
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


def plot_weights(differencing_order_range, num_plots, num_weights):
    """Plot fractional differencing weights over a range of orders.

    Args:
        differencing_order_range: Inclusive lower and upper differencing orders.
        num_plots: Number of curves to plot.
        num_weights: Number of weights per curve.

    Returns:
        None.
    """
    weights_frame = pd.DataFrame()
    for differencing_order in np.linspace(
        differencing_order_range[0],
        differencing_order_range[1],
        num_plots,
    ):
        weights = get_weights(differencing_order, num_weights=num_weights)
        order_weights = pd.DataFrame(
            weights,
            index=range(weights.shape[0])[::-1],
            columns=[differencing_order],
        )
        weights_frame = weights_frame.join(order_weights, how='outer')
    axis = weights_frame.plot()
    axis.legend(loc='upper left')
    plt.show()
    return


def fractional_difference(
    series_frame,
    differencing_order,
    weight_loss_threshold=0.01,
):
    """Apply expanding-window fractional differencing to each column.

    Args:
        series_frame: Input time series frame.
        differencing_order: Fractional differencing order.
        weight_loss_threshold: Cumulative weight-loss threshold used to skip
            early observations.

    Returns:
        A frame of fractionally differenced series.
    """
    weights = get_weights(differencing_order, series_frame.shape[0])

    cumulative_weights = np.cumsum(abs(weights))
    cumulative_weights /= cumulative_weights[-1]
    num_skipped = cumulative_weights[
        cumulative_weights > weight_loss_threshold
    ].shape[0]

    differentiated_columns = {}
    for column_name in series_frame.columns:
        filled_series = series_frame[[column_name]].ffill().dropna()
        differentiated_series = pd.Series()
        for position in range(num_skipped, filled_series.shape[0]):
            index = filled_series.index[position]
            if not np.isfinite(series_frame.loc[index, column_name]):
                continue
            differentiated_series.loc[index] = np.dot(
                weights[-(position + 1):, :].T,
                filled_series.loc[:index],
            )[0, 0]
        differentiated_columns[column_name] = differentiated_series.copy(
            deep=True
        )
    return pd.concat(differentiated_columns, axis=1)


def fractional_difference_fixed_width(
    series_frame,
    differencing_order,
    weight_cutoff=1e-5,
):
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


def plot_min_ffd(
    log_price_series,
    weight_cutoff=0.01,
    differencing_orders=None,
):
    """Plot stationarity diagnostics across fractional differencing orders.

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

    diagnostics[['adf_statistic', 'correlation']].plot(
        secondary_y='adf_statistic'
    )
    plt.axhline(
        diagnostics['critical_value_5pct'].mean(),
        linewidth=1,
        color='r',
        linestyle='dotted',
    )
    plt.show()
    return diagnostics
