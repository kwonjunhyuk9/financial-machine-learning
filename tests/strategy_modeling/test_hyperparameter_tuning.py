import numpy as np

from src.strategy_modeling.hyperparameter_tuning import log_uniform


def test_log_uniform_distribution_respects_bounds():
    distribution = log_uniform(1, np.e)

    assert distribution.cdf(1) == 0.0
    assert distribution.cdf(np.e) == 1.0


def test_log_uniform_cdf_is_monotonic_inside_its_support():
    distribution = log_uniform(1, np.e ** 2)

    assert distribution.cdf(np.e) == 0.5
