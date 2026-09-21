from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.market_structured_bars import (
    estimate_dollar_bar_threshold,
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


def test_estimate_dollar_bar_threshold_uses_median_daily_dollar_value():
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 09:30:00+00:00",
                    "2026-01-01 09:31:00+00:00",
                    "2026-01-02 09:30:00+00:00",
                    "2026-01-03 09:30:00+00:00",
                ]
            ),
            "price": [100.0, 100.0, 100.0, 100.0],
            "size": [1.0, 2.0, 6.0, 9.0],
        }
    )

    threshold = estimate_dollar_bar_threshold(
        trades,
        target_minutes=1,
        session_minutes=300,
    )

    assert threshold == pytest.approx(2.0)


def test_estimate_dollar_bar_threshold_scales_with_target_and_session_minutes():
    trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00:00+00:00"]),
            "price": [144.0],
            "size": [10.0],
        }
    )

    one_minute_stock = estimate_dollar_bar_threshold(
        trades,
        target_minutes=1,
        session_minutes=390,
    )
    five_minute_stock = estimate_dollar_bar_threshold(
        trades,
        target_minutes=5,
        session_minutes=390,
    )
    one_minute_bitcoin = estimate_dollar_bar_threshold(
        trades,
        target_minutes=1,
        session_minutes=1_440,
    )

    assert five_minute_stock == pytest.approx(one_minute_stock * 5)
    assert one_minute_bitcoin == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("target_minutes", "session_minutes", "message"),
    [
        (0, 390, "target_minutes must be positive"),
        (-1, 390, "target_minutes must be positive"),
        (1, 0, "session_minutes must be positive"),
        (1, -390, "session_minutes must be positive"),
    ],
)
def test_estimate_dollar_bar_threshold_requires_positive_durations(
    target_minutes,
    session_minutes,
    message,
):
    with pytest.raises(ValueError, match=message):
        estimate_dollar_bar_threshold(
            _make_trades(1),
            target_minutes=target_minutes,
            session_minutes=session_minutes,
        )


def test_estimate_dollar_bar_threshold_requires_trades():
    empty_trades = pd.DataFrame(columns=["timestamp", "price", "size"])

    with pytest.raises(ValueError, match="Trades must not be empty"):
        estimate_dollar_bar_threshold(
            empty_trades,
            target_minutes=1,
            session_minutes=390,
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
