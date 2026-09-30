import pandas as pd
import pytest

from src.preprocessing.market_structured_bars import get_dollar_bars


def _make_trades(num_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="min"),
            "price": [100.0 + idx for idx in range(num_rows)],
            "size": [1.0] * num_rows,
        }
    )


def test_get_dollar_bars_aggregates_each_threshold_window():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
            "price": [100.0, 100.0, 100.0, 100.0],
            "size": [1.0, 2.0, 1.0, 2.0],
        }
    )

    result = get_dollar_bars(trades, threshold=300.0)

    assert result.ohlcv["dollar_value"].tolist() == [300.0, 300.0]
    assert result.ohlcv["ticks"].tolist() == [2, 2]


def test_threshold_bars_exclude_incomplete_trailing_window():
    result = get_dollar_bars(_make_trades(4), threshold=303.0)

    assert result.ohlcv["ticks"].tolist() == [3]


@pytest.mark.parametrize("threshold", [0, -1])
def test_dollar_bars_require_positive_threshold(threshold):
    with pytest.raises(ValueError, match="Threshold must be positive"):
        get_dollar_bars(_make_trades(4), threshold=threshold)
