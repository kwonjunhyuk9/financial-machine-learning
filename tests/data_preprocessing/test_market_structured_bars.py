from pathlib import Path

import pandas as pd
import pytest

from src.data_preprocessing import market_structured_bars
from src.data_preprocessing.market_structured_bars import (
    get_dollar_imbalance_bars,
    get_dollar_run_bars,
    get_tick_bars,
    get_tick_imbalance_bars,
    get_tick_run_bars,
    get_volume_imbalance_bars,
    get_volume_run_bars,
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


def test_imbalance_bars_do_not_use_future_values_during_warm_up():
    trades = _make_trades(10)

    result = get_tick_imbalance_bars(
        trades,
        exp_num_ticks_init=20,
        num_prev_bars=3,
        expected_imbalance_window=10,
    )

    assert result.ohlcv.empty


def test_imbalance_bar_endpoints_are_unchanged_by_future_observations():
    prefix = _make_trades(30)
    extended = _make_trades(40)
    parameters = {
        "exp_num_ticks_init": 3,
        "num_prev_bars": 2,
        "expected_imbalance_window": 4,
    }

    prefix_result = get_tick_imbalance_bars(prefix, **parameters)
    extended_result = get_tick_imbalance_bars(extended, **parameters)
    prefix_end = prefix["timestamp"].iloc[-1].tz_localize("UTC")
    extended_prefix_endpoints = extended_result.ohlcv.index[
        extended_result.ohlcv.index <= prefix_end
    ]

    assert extended_prefix_endpoints.tolist() == prefix_result.ohlcv.index.tolist()


def test_run_bars_support_one_sided_activity():
    trades = _make_trades(20)

    result = get_tick_run_bars(
        trades,
        exp_num_ticks_init=3,
        num_prev_bars=2,
        expected_imbalance_window=4,
    )

    assert result.ohlcv["ticks"].tolist() == [10, 10]


@pytest.mark.parametrize(
    "bar_function",
    [
        get_tick_imbalance_bars,
        get_volume_imbalance_bars,
        get_dollar_imbalance_bars,
        get_tick_run_bars,
        get_volume_run_bars,
        get_dollar_run_bars,
    ],
)
def test_adaptive_bar_functions_accept_separate_expectation_windows(bar_function):
    result = bar_function(
        _make_trades(30),
        exp_num_ticks_init=3,
        num_prev_bars=2,
        expected_imbalance_window=4,
    )

    assert not result.ohlcv.empty


@pytest.mark.parametrize("bar_function", [get_tick_imbalance_bars, get_tick_run_bars])
@pytest.mark.parametrize("parameter", ["exp_num_ticks_init", "num_prev_bars", "expected_imbalance_window"])
@pytest.mark.parametrize("invalid_value", [0, -1, 1.5, True])
def test_adaptive_bar_parameters_must_be_positive_integers(bar_function, parameter, invalid_value):
    parameters = {
        "exp_num_ticks_init": 3,
        "num_prev_bars": 2,
        "expected_imbalance_window": 4,
    }
    parameters[parameter] = invalid_value

    with pytest.raises(ValueError, match="positive integers"):
        bar_function(_make_trades(10), **parameters)


@pytest.mark.parametrize("keyword", ["expected_num_ticks_init", "expected_window"])
def test_old_adaptive_bar_keywords_are_not_supported(keyword):
    with pytest.raises(TypeError):
        get_tick_imbalance_bars(_make_trades(10), **{keyword: 3})


def test_imbalance_bars_apply_bar_and_observation_windows_separately(monkeypatch):
    calls: list[tuple[tuple[float, ...], int]] = []

    def record_ewma(values: list[float], span: int) -> float:
        calls.append((tuple(values), span))
        return float(values[-1])

    monkeypatch.setattr(market_structured_bars, "_ewma", record_ewma)
    prepared = pd.DataFrame({"signed_tick": [1.0] * 4})

    indices = market_structured_bars._compute_imbalance_bar_end_indices(
        prepared,
        "signed_tick",
        exp_num_ticks_init=2,
        num_prev_bars=3,
        expected_imbalance_window=4,
        min_exp_num_ticks=1,
    )

    assert indices == [1, 3]
    assert ((2, 2), 3) in calls
    assert ((1.0, 1.0, 1.0, 1.0), 4) in calls


def test_run_bars_apply_bar_and_observation_windows_separately(monkeypatch):
    calls: list[tuple[tuple[float, ...], int]] = []

    def record_ewma(values: list[float], span: int) -> float:
        calls.append((tuple(values), span))
        return float(values[-1])

    monkeypatch.setattr(market_structured_bars, "_ewma", record_ewma)
    prepared = pd.DataFrame(
        {
            "signed_tick": [1.0] * 4,
            "tick_sign": [1.0] * 4,
        }
    )

    indices = market_structured_bars._compute_run_bar_end_indices(
        prepared,
        "signed_tick",
        exp_num_ticks_init=2,
        num_prev_bars=3,
        expected_imbalance_window=4,
        min_exp_num_ticks=1,
    )

    assert indices == [1, 3]
    assert ((2, 2), 3) in calls
    assert ((1.0, 1.0, 1.0, 1.0), 4) in calls
