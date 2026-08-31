import pandas as pd
import pytest

from src.model_backtesting.backtest_statistics import (
    Efficiency,
    GeneralCharacteristics,
    Performance,
    compute_strategy_returns,
)


def test_strategy_returns_default_to_zero_execution_costs():
    predictions = pd.DataFrame(
        {
            "raw_return": [0.02, -0.01],
            "primary_side": [1, -1],
            "meta_action": [1, 0],
            "bet_size": [0.5, 0.8],
        }
    )

    returns = compute_strategy_returns(predictions)

    assert returns.loc[0, "primary_only_gross_return"] == pytest.approx(0.02)
    assert returns.loc[0, "primary_only_total_cost"] == 0.0
    assert returns.loc[0, "primary_only_net_return"] == pytest.approx(0.02)
    assert returns.loc[0, "meta_filtered_gross_return"] == pytest.approx(0.01)
    assert returns.loc[0, "meta_filtered_total_cost"] == 0.0
    assert returns.loc[0, "meta_filtered_net_return"] == pytest.approx(0.01)
    assert returns.loc[1, "meta_filtered_net_return"] == 0.0


def test_performance_and_aum_statistics_use_series_means():
    pnl = pd.Series([1.0, -0.5, 2.0])
    aum = pd.Series([100.0, 200.0])

    assert Performance.pnl(pnl) == 2.5
    assert GeneralCharacteristics.average_aum(aum) == 150.0


def test_efficiency_sharpe_ratio_handles_constant_returns():
    assert pd.isna(Efficiency.sharpe_ratio(pd.Series([0.01, 0.01])))
