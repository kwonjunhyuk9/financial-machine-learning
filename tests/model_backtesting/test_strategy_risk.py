from src.model_backtesting.strategy_risk import implied_betting_frequency, implied_precision, mix_gaussians


def test_implied_precision_is_a_probability():
    precision = implied_precision(-1.0, 2.0, frequency=100, target_sharpe=1.0)

    assert 0.0 <= precision <= 1.0


def test_frequency_and_mixture_outputs_have_requested_shape():
    frequency = implied_betting_frequency(-1.0, 2.0, precision=0.6, target_sharpe=1.0)
    returns = mix_gaussians(0.01, -0.01, 0.1, 0.1, 0.5, num_observations=20)

    assert frequency > 0
    assert returns.shape == (20,)
