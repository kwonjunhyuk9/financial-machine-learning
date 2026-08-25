import numpy as np
import pandas as pd

from src.data_preprocessing.market_differentiated_bars import (
    evaluate_fractional_differencing_orders,
    get_weights_fixed_width,
)


def test_fixed_width_weights_stop_at_threshold():
    weights = get_weights_fixed_width(
        differencing_order=0.5,
        weight_cutoff=0.1,
    )

    assert weights[-1, 0] == 1.0
    assert len(weights) > 1


def test_evaluate_orders_uses_descriptive_column_names():
    random_generator = np.random.default_rng(seed=0)
    log_prices = random_generator.normal(scale=0.01, size=200).cumsum()
    log_price_series = pd.Series(log_prices, name='log_close')

    diagnostics = evaluate_fractional_differencing_orders(
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
