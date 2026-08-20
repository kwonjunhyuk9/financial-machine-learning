import numpy as np
import pandas as pd
from loguru import logger


WEIGHT_COLUMNS = [
    "average_uniqueness_weight",
    "return_attribution_weight",
    "time_decay_weight",
    "sample_weight",
]


def count_concurrent_events(close_idx, t1, molecule):
    """Count how many events are active at each bar in a slice.

    Args:
        close_idx: Full close-price index.
        t1: Event end times indexed by start time.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of concurrency counts over the relevant bar range.
    """
    t1 = t1.fillna(close_idx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]

    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0]:iloc[1] + 1])

    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.

    return count.loc[molecule[0]:t1[molecule].max()]


def compute_average_uniqueness_weights(t1, num_co_events, molecule):
    """Compute average uniqueness weights for a slice of events.

    Args:
        t1: Event end times indexed by start time.
        num_co_events: Concurrency counts over the price bars.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of average uniqueness weights.
    """
    wght = pd.Series(index=molecule)

    for t_in, t_out in t1.loc[wght.index].items():
        wght.loc[t_in] = (1. / num_co_events.loc[t_in:t_out]).mean()

    return wght


def compute_return_attribution_weights(t1, num_co_events, close, molecule):
    """Compute return-attribution sample weights.

    Args:
        t1: Event end times indexed by start time.
        num_co_events: Concurrency counts over the price bars.
        close: Close price series.
        molecule: Slice of event start times to evaluate.

    Returns:
        A series of absolute sample weights.
    """
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule)

    for t_in, t_out in t1.loc[wght.index].items():
        wght.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()

    return wght.abs()


def apply_time_decay(t_w, clf_last_w=1.):
    """Apply piecewise-linear decay to sample weights.

    Args:
        t_w: Base weight series.
        clf_last_w: Weight assigned to the oldest observation.

    Returns:
        A decayed weight series.
    """
    clf_w = t_w.sort_index().cumsum()

    if clf_last_w >= 0:
        slope = (1. - clf_last_w) / clf_w.iloc[-1]
    else:
        slope = 1. / ((clf_last_w + 1) * clf_w.iloc[-1])

    const = 1. - slope * clf_w.iloc[-1]
    clf_w = const + slope * clf_w
    clf_w[clf_w < 0] = 0

    logger.debug("Applied time decay with slope {} and intercept {}.", slope, const)
    return clf_w


def build_partitioned_event_weights(
    events: pd.DataFrame,
    partition_manifest: pd.DataFrame,
    close: pd.Series,
) -> pd.DataFrame:
    """Calculate event weights independently within each fixed partition.

    Args:
        events: Labeled events containing unique start and end timestamps.
        partition_manifest: Fixed event partitions keyed by event start.
        close: Dollar-bar close prices indexed by bar end time.

    Returns:
        Events with four partition-local weight columns appended.

    Raises:
        ValueError: If required data is missing, duplicated, or cannot be weighted.
    """
    required_event_columns = {"event_start", "event_end"}
    required_manifest_columns = {"event_start", "partition"}
    missing_events = required_event_columns.difference(events.columns)
    missing_manifest = required_manifest_columns.difference(partition_manifest.columns)
    if missing_events:
        raise ValueError(f"Events are missing columns: {sorted(missing_events)}")
    if missing_manifest:
        raise ValueError(
            f"Partition manifest is missing columns: {sorted(missing_manifest)}"
        )

    weighted_input = events.drop(columns=WEIGHT_COLUMNS, errors="ignore").copy()
    weighted_input["event_start"] = pd.to_datetime(
        weighted_input["event_start"], utc=True, errors="coerce"
    )
    weighted_input["event_end"] = pd.to_datetime(
        weighted_input["event_end"], utc=True, errors="coerce"
    )
    manifest = partition_manifest.copy()
    manifest["event_start"] = pd.to_datetime(
        manifest["event_start"], utc=True, errors="coerce"
    )
    if weighted_input[["event_start", "event_end"]].isna().any().any():
        raise ValueError("Event intervals must contain valid timestamps.")
    if manifest["event_start"].isna().any():
        raise ValueError("Manifest event_start must contain valid timestamps.")
    if weighted_input["event_start"].duplicated().any():
        raise ValueError("Event starts must be unique.")
    if manifest["event_start"].duplicated().any():
        raise ValueError("Manifest event starts must be unique.")

    close_prices = close.astype(float).copy()
    close_prices.index = pd.to_datetime(close_prices.index, utc=True, errors="coerce")
    if close_prices.index.isna().any() or close_prices.index.duplicated().any():
        raise ValueError("Close-price index must contain unique valid timestamps.")
    close_prices = close_prices.sort_index()
    if close_prices.empty or not np.isfinite(close_prices).all():
        raise ValueError("Close prices must be finite and non-empty.")

    event_starts = set(weighted_input["event_start"])
    active_starts = set(
        manifest.loc[
            manifest["partition"].isin(["development", "holdout"]),
            "event_start",
        ]
    )
    if event_starts != active_starts:
        raise ValueError("Events must match active development and holdout rows.")

    weight_tables = []
    for partition in ["development", "holdout"]:
        partition_starts = manifest.loc[
            manifest["partition"].eq(partition), "event_start"
        ]
        partition_events = weighted_input.loc[
            weighted_input["event_start"].isin(partition_starts)
        ].set_index("event_start")
        if partition_events.empty:
            raise ValueError(f"{partition} must contain at least one event.")

        information_sets = partition_events["event_end"]
        concurrency = count_concurrent_events(
            close_prices.index,
            information_sets,
            information_sets.index,
        )
        uniqueness = compute_average_uniqueness_weights(
            information_sets,
            concurrency,
            information_sets.index,
        )
        return_attribution = compute_return_attribution_weights(
            information_sets,
            concurrency,
            close_prices,
            information_sets.index,
        )
        positive_floor = return_attribution[return_attribution.gt(0)].min()
        if pd.isna(positive_floor):
            raise ValueError(
                f"{partition} return-attribution weights are all zero."
            )
        base_weight = return_attribution.clip(lower=positive_floor)
        time_decay = apply_time_decay(base_weight, clf_last_w=0.50)
        sample_weight = base_weight * time_decay
        sample_weight *= len(sample_weight) / sample_weight.sum()
        weight_tables.append(
            pd.DataFrame(
                {
                    "average_uniqueness_weight": uniqueness,
                    "return_attribution_weight": return_attribution,
                    "time_decay_weight": time_decay,
                    "sample_weight": sample_weight,
                }
            ).rename_axis("event_start").reset_index()
        )

    weight_table = pd.concat(weight_tables, ignore_index=True)
    weighted = weighted_input.merge(
        weight_table,
        on="event_start",
        how="left",
        validate="one_to_one",
    ).sort_values("event_start", ignore_index=True)
    if weighted[WEIGHT_COLUMNS].isna().any().any():
        raise ValueError("Every retained event must receive complete weights.")
    return weighted


def build_indicator_matrix(bar_ix, t1):
    """Build an indicator matrix mapping bars to active events.

    Args:
        bar_ix: Bar index.
        t1: Event end times indexed by start time.

    Returns:
        A binary indicator matrix with one column per event.
    """
    ind_m = pd.DataFrame(0, index=bar_ix, columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.items()):
        ind_m.loc[t0:t1_, i] = 1.
    return ind_m


def compute_average_uniqueness(ind_m):
    """Compute per-event average uniqueness from an indicator matrix.

    Args:
        ind_m: Indicator matrix with bars on rows and events on columns.

    Returns:
        A series of average uniqueness values.
    """
    c = ind_m.sum(axis=1)
    u = ind_m.div(c, axis=0)
    avg_u = u[u > 0].mean()
    return avg_u


def sequential_bootstrap(ind_m, s_length=None):
    """Sample event indices with sequential bootstrap.

    Args:
        ind_m: Indicator matrix with bars on rows and events on columns.
        s_length: Desired sample length.

    Returns:
        A list of sampled event indices.
    """
    if s_length is None:
        s_length = ind_m.shape[1]
    phi = []
    while len(phi) < s_length:
        avg_u = pd.Series(dtype=float)
        for i in ind_m:
            ind_m_ = ind_m[phi + [i]]
            avg_u.loc[i] = compute_average_uniqueness(ind_m_).iloc[-1]
        prob = avg_u / avg_u.sum()
        phi += [np.random.choice(ind_m.columns, p=prob)]
    return phi


def generate_random_t1(num_obs, num_bars, max_h):
    """Generate random event horizons for simulation.

    Args:
        num_obs: Number of events.
        num_bars: Number of bars in the simulated sample.
        max_h: Maximum event horizon in bars.

    Returns:
        A sorted series of random event end times.
    """
    t1 = pd.Series(dtype=int)
    for i in range(num_obs):
        ix = np.random.randint(0, num_bars)
        val = ix + np.random.randint(1, max_h)
        t1.loc[ix] = val
    return t1.sort_index()


def run_monte_carlo_trial(num_obs, num_bars, max_h):
    """Compare standard and sequential bootstrap uniqueness in one trial.

    Args:
        num_obs: Number of events.
        num_bars: Number of bars in the simulated sample.
        max_h: Maximum event horizon in bars.

    Returns:
        A dictionary with standard and sequential uniqueness statistics.
    """
    t1 = generate_random_t1(num_obs, num_bars, max_h)
    bar_ix = range(t1.max() + 1)
    ind_m = build_indicator_matrix(bar_ix, t1)

    phi = np.random.choice(ind_m.columns, size=ind_m.shape[1])
    std_u = compute_average_uniqueness(ind_m[phi]).mean()

    phi = sequential_bootstrap(ind_m)
    seq_u = compute_average_uniqueness(ind_m[phi]).mean()

    return {'std_u': std_u, 'seq_u': seq_u}


def build_monte_carlo_jobs(num_obs=10, num_bars=100, max_h=5, num_iters=1E6, num_threads=24):
    """Build Monte Carlo job specifications.

    Args:
        num_obs: Number of events per trial.
        num_bars: Number of bars per trial.
        max_h: Maximum event horizon in bars.
        num_iters: Number of trials to schedule.
        num_threads: Unused thread count placeholder.

    Returns:
        None.
    """
    jobs = []
    for i in range(int(num_iters)):
        job = {
            'func': run_monte_carlo_trial,
            'num_obs': num_obs,
            'num_bars': num_bars,
            'max_h': max_h,
        }
        jobs.append(job)
