from src.model_backtesting.backtest_synthetic import synthetic_trading_rule_sharpe_ratios


def test_synthetic_sharpe_ratios_are_reproducible_with_a_seed():
    kwargs = {
        "forecast": 100.0,
        "half_life": 10.0,
        "profit_taking_range": [1.0, 2.0],
        "stop_loss_range": [1.0],
        "num_iterations": 50,
        "max_holding_period": 10,
        "random_state": 7,
    }

    first = synthetic_trading_rule_sharpe_ratios(**kwargs)
    second = synthetic_trading_rule_sharpe_ratios(**kwargs)

    assert first.equals(second)
    assert len(first) == 2


def test_synthetic_sharpe_ratios_reject_invalid_half_life():
    import pytest

    with pytest.raises(ValueError, match="half_life"):
        synthetic_trading_rule_sharpe_ratios(100, 0, [1], [1])
