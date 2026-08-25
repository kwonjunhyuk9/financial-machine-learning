from pathlib import Path

import pandas as pd
import pytest

from src.data_preprocessing.market_structured_bars import (
    get_dollar_bars,
    get_tick_bars,
    get_volume_bars,
    save_structured_bar_result,
)


STRUCTURED_BAR_COLUMNS = [
    "end",
    "start",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_value",
    "ticks",
    "buy_volume",
    "sell_volume",
]


def _make_trades(num_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="min"),
            "price": [100.0 + idx for idx in range(num_rows)],
            "size": [1.0] * num_rows,
        }
    )


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


def test_get_volume_bars_aggregates_each_threshold_window():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
            "price": [100.0, 101.0, 102.0, 103.0],
            "size": [1.0, 2.0, 1.0, 2.0],
        }
    )

    result = get_volume_bars(trades, threshold=3.0)

    assert result.ohlcv["volume"].tolist() == [3.0, 3.0]
    assert result.ohlcv["close"].tolist() == [101.0, 103.0]


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
    result = get_tick_bars(_make_trades(4), threshold=3)

    assert result.ohlcv["ticks"].tolist() == [3]


@pytest.mark.parametrize(
    "bar_function",
    [get_tick_bars, get_volume_bars, get_dollar_bars],
)
@pytest.mark.parametrize("threshold", [0, -1])
def test_threshold_bar_functions_require_positive_threshold(
    bar_function,
    threshold,
):
    with pytest.raises(ValueError, match="Threshold must be positive"):
        bar_function(_make_trades(4), threshold=threshold)


def test_save_structured_bar_result_writes_complete_ohlcv(tmp_path: Path):
    result = get_tick_bars(_make_trades(4), threshold=2)
    output_path = tmp_path / "features" / "tick_bar.parquet"

    saved_path = save_structured_bar_result(result, output_path)
    saved = pd.read_parquet(saved_path)

    assert saved_path == output_path
    assert saved.columns.tolist() == STRUCTURED_BAR_COLUMNS
    assert saved["end"].tolist() == result.ohlcv.index.tolist()
    assert saved["dollar_value"].tolist() == result.ohlcv["dollar_value"].tolist()
    assert saved["ticks"].tolist() == result.ohlcv["ticks"].tolist()
    assert saved["buy_volume"].tolist() == result.ohlcv["buy_volume"].tolist()
    assert saved["sell_volume"].tolist() == result.ohlcv["sell_volume"].tolist()


def test_save_structured_bar_result_preserves_empty_schema(tmp_path: Path):
    result = get_tick_bars(_make_trades(4), threshold=10)

    saved_path = save_structured_bar_result(result, tmp_path / "empty.parquet")
    saved = pd.read_parquet(saved_path)

    assert saved.empty
    assert saved.columns.tolist() == STRUCTURED_BAR_COLUMNS
