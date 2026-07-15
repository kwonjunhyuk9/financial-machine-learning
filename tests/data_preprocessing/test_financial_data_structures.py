import pandas as pd

from src.data_preprocessing.financial_data_structures import get_cusum_events, get_tick_bars


def test_get_tick_bars_aggregates_each_threshold_window():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
            "price": [100.0, 101.0, 102.0, 103.0],
            "size": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = get_tick_bars(trades, threshold=2)

    assert result.ohlcv["close"].tolist() == [101.0, 103.0]


def test_get_cusum_events_detects_positive_and_negative_threshold_crossings():
    index = pd.date_range("2026-01-01", periods=4, freq="D")

    events = get_cusum_events(pd.Series([0.0, 0.6, 0.1, -0.7], index=index), threshold=0.5)

    assert events.tolist() == [index[1], index[3]]
