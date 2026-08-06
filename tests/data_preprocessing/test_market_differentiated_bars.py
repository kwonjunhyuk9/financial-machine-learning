import numpy as np
import pandas as pd

from src.data_preprocessing.market_differentiated_bars import (
    get_weights,
    get_weights_fixed_width,
    plot_min_ffd,
)


def test_get_weights_retains_current_observation_weight():
    weights = get_weights(differencing_order=0.5, num_weights=4)

    assert weights.shape == (4, 1)
    assert weights[-1, 0] == 1.0


def test_fixed_width_weights_stop_at_threshold():
    weights = get_weights_fixed_width(
        differencing_order=0.5,
        weight_cutoff=0.1,
    )

    assert weights[-1, 0] == 1.0
    assert len(weights) > 1


def test_min_ffd_diagnostic_uses_descriptive_column_names(monkeypatch):
    random_generator = np.random.default_rng(seed=0)
    log_prices = random_generator.normal(scale=0.01, size=200).cumsum()
    log_price_series = pd.Series(log_prices, name='log_close')
    monkeypatch.setattr(pd.DataFrame, 'plot', lambda self, **kwargs: None)
    monkeypatch.setattr(
        'matplotlib.pyplot.axhline',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    diagnostics = plot_min_ffd(
        log_price_series,
        weight_cutoff=0.01,
        differencing_orders=[0.0],
    )

    assert diagnostics.columns.tolist() == [
        'adf_statistic',
        'p_value',
        'used_lags',
        'n_observations',
        'critical_value_5pct',
        'correlation',
    ]
