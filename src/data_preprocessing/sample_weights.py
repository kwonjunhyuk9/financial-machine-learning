import numpy as np
import pandas as pd


def count_concurrent_events(closeIdx, t1, molecule):
    """Count how many events are active at each bar in a slice.

    Args:
        closeIdx: Full close-price index.
        t1: Event end times indexed by start time.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of concurrency counts over the relevant bar range.
    """
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]

    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])

    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1.

    return count.loc[molecule[0]:t1[molecule].max()]


def compute_average_uniqueness_weights(t1, numCoEvents, molecule):
    """Compute average uniqueness weights for a slice of events.

    Args:
        t1: Event end times indexed by start time.
        numCoEvents: Concurrency counts over the price bars.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of average uniqueness weights.
    """
    wght = pd.Series(index=molecule)

    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()

    return wght


def build_indicator_matrix(barIx, t1):
    """Build an indicator matrix mapping bars to active events.

    Args:
        barIx: Bar index.
        t1: Event end times indexed by start time.

    Returns:
        A binary indicator matrix with one column per event.
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.iteritems()):
        indM.loc[t0:t1_, i] = 1.
    return indM


def compute_average_uniqueness(indM):
    """Compute per-event average uniqueness from an indicator matrix.

    Args:
        indM: Indicator matrix with bars on rows and events on columns.

    Returns:
        A series of average uniqueness values.
    """
    c = indM.sum(axis=1)
    u = indM.div(c, axis=0)
    avgU = u[u > 0].mean()
    return avgU


def sequential_bootstrap(indM, sLength=None):
    """Sample event indices with sequential bootstrap.

    Args:
        indM: Indicator matrix with bars on rows and events on columns.
        sLength: Desired sample length. Defaults to the number of events.

    Returns:
        A list of sampled event indices.
    """
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]
            avgU.loc[i] = compute_average_uniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


def generate_random_t1(numObs, numBars, maxH):
    """Generate random event horizons for simulation.

    Args:
        numObs: Number of events.
        numBars: Number of bars in the simulated sample.
        maxH: Maximum event horizon in bars.

    Returns:
        A sorted series of random event end times.
    """
    t1 = pd.Series()
    for i in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


def run_monte_carlo_trial(numObs, numBars, maxH):
    """Compare standard and sequential bootstrap uniqueness in one trial.

    Args:
        numObs: Number of events.
        numBars: Number of bars in the simulated sample.
        maxH: Maximum event horizon in bars.

    Returns:
        A dictionary with standard and sequential uniqueness statistics.
    """
    t1 = generate_random_t1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = build_indicator_matrix(barIx, t1)

    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = compute_average_uniqueness(indM[phi]).mean()

    phi = sequential_bootstrap(indM)
    seqU = compute_average_uniqueness(indM[phi]).mean()

    return {'stdU': stdU, 'seqU': seqU}


def build_monte_carlo_jobs(numObs=10, numBars=100, maxH=5, numIters=1E6, numThreads=24):
    """Build Monte Carlo job specifications.

    Args:
        numObs: Number of events per trial.
        numBars: Number of bars per trial.
        maxH: Maximum event horizon in bars.
        numIters: Number of trials to schedule.
        numThreads: Unused thread count placeholder.
    """
    jobs = []
    for i in range(int(numIters)):
        job = {'func': run_monte_carlo_trial, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)


def compute_sample_weights(t1, numCoEvents, close, molecule):
    """Compute return-attribution sample weights.

    Args:
        t1: Event end times indexed by start time.
        numCoEvents: Concurrency counts over the price bars.
        close: Close price series.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of absolute sample weights.
    """
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule)

    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()

    return wght.abs()


def apply_time_decay(tW, clfLastW=1.):
    """Apply piecewise-linear decay to sample weights.

    Args:
        tW: Base weight series.
        clfLastW: Weight assigned to the oldest observation.

    Returns:
        A decayed weight series.
    """
    clfW = tW.sort_index().cumsum()

    if clfLastW >= 0:
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])

    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0

    print(const, slope)
    return clfW
