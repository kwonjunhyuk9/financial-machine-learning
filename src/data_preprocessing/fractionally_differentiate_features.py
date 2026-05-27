import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_weights(d, size):
    """Compute fractional differencing weights.

    Args:
        d: Fractional differencing order.
        size: Number of weights to generate.

    Returns:
        A column vector of weights ordered from oldest to newest.
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def get_weights_fixed_width(d, thres):
    """Compute fixed-width fractional differencing weights.

    Args:
        d: Fractional differencing order.
        thres: Absolute weight cutoff threshold.

    Returns:
        A column vector of weights ordered from oldest to newest.
    """
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] / k * (d - k + 1)
        if abs(weight) < thres:
            break
        weights.append(weight)
        k += 1
    return np.array(weights[::-1]).reshape(-1, 1)


def plot_weights(dRange, nPlots, size):
    """Plot fractional differencing weights over a range of orders.

    Args:
        dRange: Inclusive lower and upper bounds for ``d``.
        nPlots: Number of curves to plot.
        size: Number of weights per curve.
    """
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = get_weights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left')
    plt.show()
    return


def fractional_difference(series, d, thres=.01):
    """Apply expanding-window fractional differencing to each column.

    Args:
        series: Input time series frame.
        d: Fractional differencing order.
        thres: Cumulative weight-loss threshold used to skip early observations.

    Returns:
        A frame of fractionally differenced series.
    """
    w = get_weights(d, series.shape[0])

    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]

    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].ffill().dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue
            df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def fractional_difference_fixed_width(series, d, thres=1e-5):
    """Apply fixed-width fractional differencing to each column.

    Args:
        series: Input time series frame.
        d: Fractional differencing order.
        thres: Weight cutoff threshold used to set the window width.

    Returns:
        A frame of fractionally differenced series.
    """
    w = get_weights_fixed_width(d, thres)
    width = len(w) - 1

    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].ffill().dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def plot_min_ffd(series, thres=.01, d_values=None):
    """Plot stationarity diagnostics across fractional differencing orders.

    Args:
        series: Time-indexed Series or single-column DataFrame of prices.
        thres: Weight cutoff threshold passed to fixed-width differencing.
        d_values: Optional iterable of fractional differencing orders.

    Returns:
        A DataFrame with ADF statistics and correlations by differencing order.
    """
    from statsmodels.tsa.stattools import adfuller

    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    if isinstance(series, pd.Series):
        df0 = series.to_frame(name=series.name or 'close')
    else:
        df0 = series.copy()
    if df0.shape[1] != 1:
        raise ValueError('series must be a Series or single-column DataFrame')
    column = df0.columns[0]
    d_values = np.linspace(0, 1, 11) if d_values is None else d_values

    df1 = np.log(df0[[column]]).dropna()

    for d in d_values:
        df2 = fractional_difference_fixed_width(df1, d, thres=thres)
        corr = np.corrcoef(df1.loc[df2.index, column], df2[column])[0, 1]
        adf_result = adfuller(df2[column], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(adf_result[:4]) + [adf_result[4]['5%']] + [corr]

    out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.show()
    return out
