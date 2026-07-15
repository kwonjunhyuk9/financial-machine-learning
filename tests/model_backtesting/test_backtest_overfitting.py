import pandas as pd

from src.model_backtesting.backtest_overfitting import get_sharpe_ratio_metric, sharpe_ratio


def test_sharpe_ratio_returns_one_value_per_strategy():
    returns = pd.DataFrame({"a": [0.01, -0.01], "b": [0.02, 0.01]})

    ratios = sharpe_ratio(returns, periods_per_year=1)

    assert ratios.index.tolist() == ["a", "b"]
    assert ratios["b"] > ratios["a"]


def test_metric_factory_applies_annual_risk_free_rate():
    metric = get_sharpe_ratio_metric(annual_risk_free_rate=0.0, periods_per_year=1)
    result = metric(pd.DataFrame({"strategy": [0.01, 0.02]}))

    assert result.index.tolist() == ["strategy"]
