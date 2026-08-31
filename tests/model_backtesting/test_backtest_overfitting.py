import pandas as pd

from src.model_backtesting.backtest_overfitting import get_sharpe_ratio_metric, sharpe_ratio
from src.model_backtesting.backtest_statistics import Efficiency


def test_sharpe_ratio_returns_one_value_per_strategy():
    returns = pd.DataFrame({"a": [0.01, -0.01], "b": [0.02, 0.01]})

    ratios = sharpe_ratio(returns, periods_per_year=1)

    assert ratios.index.tolist() == ["a", "b"]
    assert ratios["b"] > ratios["a"]


def test_metric_factory_applies_annual_risk_free_rate():
    metric = get_sharpe_ratio_metric(annual_risk_free_rate=0.0, periods_per_year=1)
    result = metric(pd.DataFrame({"strategy": [0.01, 0.02]}))

    assert result.index.tolist() == ["strategy"]


def test_sharpe_ratio_matches_efficiency_for_each_strategy():
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "variable": [0.01, -0.02, 0.03, 0.01],
            "constant": [0.01, 0.01, 0.01, 0.01],
        },
        index=index,
    )
    risk_free_rate = pd.Series([0.001, 0.002, 0.001, 0.002], index=index)

    annualized = sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=12,
    )
    non_annualized = sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=None,
    )
    expected_annualized = pd.Series({
        column: Efficiency.annualized_sharpe_ratio(
            returns[column],
            risk_free_rate=risk_free_rate,
            periods_per_year=12,
        )
        for column in returns
    })
    expected_non_annualized = pd.Series({
        column: Efficiency.sharpe_ratio(
            returns[column],
            risk_free_rate=risk_free_rate,
        )
        for column in returns
    })

    pd.testing.assert_series_equal(annualized, expected_annualized)
    pd.testing.assert_series_equal(non_annualized, expected_non_annualized)
    assert pd.isna(sharpe_ratio(
        returns[["constant"]],
        risk_free_rate=0.0,
        periods_per_year=12,
    )["constant"])
