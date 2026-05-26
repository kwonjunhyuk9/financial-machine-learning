import numpy as np
import pandas as pd


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
    mpl.show()
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
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
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
    w = getWeights_FFD(d, thres)
    width = len(w) - 1

    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def plot_min_ffd():
    """Plot stationarity diagnostics across fractional differencing orders."""
    from statsmodels.tsa.stattools import adfuller

    path, instName = './', 'ES1_Index_Method12'
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = pd.read_csv(path + instName + '.csv', index_col=0, parse_dates=True)

    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0[['Close']]).resample('1D').last()
        df2 = fractional_difference_fixed_width(df1, d, thres=.01)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        df2 = adfuller(df2['Close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]

    out.to_csv(path + instName + '_testMinFFD.csv')
    out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    mpl.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    mpl.savefig(path + instName + '_testMinFFD.png')
    return
