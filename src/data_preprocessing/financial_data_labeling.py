import numpy as np
import pandas as pd


def apply_to_molecule(func, pdObj, numThreads, **kwargs):
    """Apply a labeling helper over the requested pandas index subset.

    Args:
        func: Worker function that accepts a ``molecule`` keyword argument.
        pdObj: Tuple of ``(name, values)`` describing the subset to process.
        numThreads: Retained for API compatibility; execution is sequential here.
        **kwargs: Extra keyword arguments passed to ``func``.

    Returns:
        The worker function output for the requested ``molecule``.
    """
    _, molecule = pdObj
    return func(molecule=molecule, **kwargs)


def get_daily_volatility(close, span0=100):
    """Estimate exponentially weighted daily volatility.

    Args:
        close: Close price series indexed by timestamp.
        span0: Span used by the exponentially weighted standard deviation.

    Returns:
        A series of daily volatility estimates aligned to ``close``.
    """
    positions = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    valid = positions > 0
    current_positions = np.arange(close.shape[0])[valid]
    previous_positions = positions[valid] - 1

    returns = pd.Series(
        close.iloc[current_positions].values / close.iloc[previous_positions].values - 1,
        index=close.index[valid],
    )
    returns = returns.ewm(span=span0).std()
    return returns


def get_vertical_barriers(tEvents, close, numBars=1):
    """Set a vertical barrier a fixed number of bars after each event.

    Args:
        tEvents: Event start timestamps.
        close: Close price series used to locate future bars.
        numBars: Number of bars ahead to place the vertical barrier.

    Returns:
        A series mapping each eligible event start time to its vertical barrier time.
    """
    event_index = pd.DatetimeIndex(tEvents)
    positions = close.index.get_indexer(event_index)
    barrier_times = {}

    for event_time, position in zip(event_index, positions):
        if position < 0:
            continue
        barrier_position = position + numBars
        if barrier_position < len(close.index):
            barrier_times[event_time] = close.index[barrier_position]

    return pd.Series(barrier_times)


def apply_profit_taking_stop_loss_on_t1(close, events, ptSl, molecule):
    """Locate horizontal barrier hits before the vertical barrier.

    Args:
        close: Close price series.
        events: Event frame containing ``t1``, ``trgt``, and ``side``.
        ptSl: Profit-taking and stop-loss multipliers.
        molecule: Subset of event indices to process.

    Returns:
        A frame with the earliest stop-loss and profit-taking hit times.
    """
    events_ = events.loc[molecule]
    out = pd.DataFrame(index=events_.index, columns=['t1', 'sl', 'pt'], dtype=object)
    out['t1'] = events_['t1']

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)

    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()

    return out


def get_events(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """Build the event table used by triple-barrier labeling.

    Args:
        close: Close price series.
        tEvents: Event start times.
        ptSl: Profit-taking and stop-loss multipliers.
        trgt: Target return series.
        minRet: Minimum target return required to keep an event.
        numThreads: Number of worker threads for the barrier search.
        t1: Optional vertical barrier times.
        side: Optional side predictions for meta-labeling.

    Returns:
        An event frame with targets, barrier times, and optional side information.
    """
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]

    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])

    df0 = apply_to_molecule(
        func=apply_profit_taking_stop_loss_on_t1,
        pdObj=('molecule', events.index),
        numThreads=numThreads,
        close=close,
        events=events,
        ptSl=ptSl_
    )

    def _earliest_timestamp(row):
        timestamps = [value for value in row if pd.notna(value)]
        return min(timestamps) if timestamps else pd.NaT

    events['t1'] = df0.apply(_earliest_timestamp, axis=1)

    if side is None:
        events = events.drop('side', axis=1)

    return events


def get_bins(events, close):
    """Convert event outcomes into return and label pairs.

    Args:
        events: Event frame with vertical barrier times and optional side values.
        close: Close price series covering event starts and ends.

    Returns:
        A frame containing realized returns and discrete labels.
    """
    events_ = events.dropna(subset=['t1'])
    out = pd.DataFrame(index=events_.index)
    start_prices = close.reindex(events_.index, method='bfill')
    end_index = pd.DatetimeIndex(events_['t1'].tolist())
    end_prices = close.reindex(end_index, method='bfill')
    out['ret'] = end_prices.to_numpy() / start_prices.to_numpy() - 1

    if 'side' in events_:
        out['ret'] *= events_['side']

    out['bin'] = np.sign(out['ret'])

    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0

    return out


def drop_labels(events, minPct=.05):
    """Remove labels whose relative frequency falls below a threshold.

    Args:
        events: Event frame containing a ``bin`` column.
        minPct: Minimum class frequency required to keep a label.

    Returns:
        The filtered event frame.
    """
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events
