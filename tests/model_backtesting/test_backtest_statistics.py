import pandas as pd

from src.model_backtesting.backtest_statistics import Efficiency, GeneralCharacteristics, Performance


def test_performance_and_aum_statistics_use_series_means():
    pnl = pd.Series([1.0, -0.5, 2.0])
    aum = pd.Series([100.0, 200.0])

    assert Performance.pnl(pnl) == 2.5
    assert GeneralCharacteristics.average_aum(aum) == 150.0


def test_efficiency_sharpe_ratio_handles_constant_returns():
    assert pd.isna(Efficiency.sharpe_ratio(pd.Series([0.01, 0.01])))
