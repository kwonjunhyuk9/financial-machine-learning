from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoTradesRequest,
    StockBarsRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame
from src.preprocessing import market_data
from src.preprocessing.market_data import ResearchPaths


def test_build_output_path_normalizes_symbols():
    path = market_data._build_output_path(
        symbols=["BRK/B"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2),
        output_dir=Path("output"),
    )

    assert path == Path("output/brk-b_2026-01-01_2026-01-02.parquet")


def test_normalize_trade_frame_rejects_missing_price_column():
    trades = pd.DataFrame({"timestamp": ["2026-01-01"], "symbol": ["AAPL"], "size": [1]})

    with pytest.raises(ValueError, match="price"):
        market_data._normalize_trade_frame(trades)


def test_normalize_minute_frame_uses_close_price_and_volume_as_size():
    bars = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T14:31:00Z"],
            "symbol": ["AAPL"],
            "open": [100],
            "high": [102],
            "low": [99],
            "close": [101],
            "volume": [250],
            "trade_count": [12],
            "vwap": [100.5],
        }
    )

    result = market_data._normalize_minute_frame(bars)

    assert result.columns.tolist() == ["timestamp", "symbol", "price", "size"]
    assert result.loc[0, "price"] == 101.0
    assert result.loc[0, "size"] == 250.0
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_normalize_minute_frame_rejects_missing_close_column():
    bars = pd.DataFrame(
        {"timestamp": ["2026-01-01"], "symbol": ["AAPL"], "volume": [250]}
    )

    with pytest.raises(ValueError, match="close"):
        market_data._normalize_minute_frame(bars)


@pytest.mark.parametrize(
    ("asset_class", "data_type", "method_name", "request_type"),
    [
        ("stock", "tick", "get_stock_trades", StockTradesRequest),
        ("stock", "1min", "get_stock_bars", StockBarsRequest),
        ("crypto", "tick", "get_crypto_trades", CryptoTradesRequest),
        ("crypto", "1min", "get_crypto_bars", CryptoBarsRequest),
    ],
)
def test_fetch_alpaca_historical_data_dispatches_request(
    monkeypatch,
    asset_class,
    data_type,
    method_name,
    request_type,
):
    trades = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "symbol": ["AAPL", "AAPL"],
            "price": [100, 101],
            "size": [1, 2],
            "id": ["trade-1", "trade-2"],
            "exchange": ["V", "V"],
        }
    )
    bars = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "symbol": ["AAPL", "AAPL"],
            "close": [100, 101],
            "volume": [10, 20],
        }
    )
    response = SimpleNamespace(df=trades if data_type == "tick" else bars)
    stock_client = Mock()
    crypto_client = Mock()
    getattr(stock_client, method_name, Mock()).return_value = response
    getattr(crypto_client, method_name, Mock()).return_value = response

    monkeypatch.setattr(market_data, "_get_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(
        market_data,
        "StockHistoricalDataClient",
        Mock(return_value=stock_client),
    )
    monkeypatch.setattr(
        market_data,
        "CryptoHistoricalDataClient",
        Mock(return_value=crypto_client),
    )

    result = market_data.fetch_alpaca_historical_data(
        symbols=["AAPL"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        asset_class=asset_class,
        data_type=data_type,
    )

    client = stock_client if asset_class == "stock" else crypto_client
    request = getattr(client, method_name).call_args.args[0]
    assert isinstance(request, request_type)
    assert result.columns.tolist() == ["timestamp", "symbol", "price", "size"]
    assert result["timestamp"].tolist() == [pd.Timestamp("2026-01-01T00:00:00Z")]
    if data_type == "1min":
        assert request.timeframe.value == TimeFrame.Minute.value


def test_fetch_alpaca_historical_data_rejects_invalid_data_type():
    with pytest.raises(ValueError, match="data_type"):
        market_data.fetch_alpaca_historical_data(
            symbols=["AAPL"],
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 2),
            asset_class="stock",
            data_type="day",
        )


def test_fetch_alpaca_historical_data_rejects_invalid_asset_class():
    with pytest.raises(ValueError, match="asset_class"):
        market_data.fetch_alpaca_historical_data(
            symbols=["AAPL"],
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 2),
            asset_class="option",
            data_type="tick",
        )


def write_universe(tmp_path, symbols=None, column="symbol"):
    destination = tmp_path / "data/preprocessing/sp500_2025.csv"
    destination.parent.mkdir(parents=True)
    symbols = symbols if symbols is not None else [f"S{i:03}" for i in range(503)]
    pd.DataFrame({column: symbols}).to_csv(destination, index=False)
    return destination


def test_load_manifest_reads_fixed_universe_csv(tmp_path):
    write_universe(tmp_path)
    manifest = market_data.load_manifest(ResearchPaths(tmp_path))
    assert manifest.columns.tolist() == ["symbol"]
    assert len(manifest) == manifest.symbol.nunique() == 503


def test_load_manifest_rejects_wrong_column(tmp_path):
    write_universe(tmp_path, column="ticker")
    with pytest.raises(ValueError, match="only the symbol column"):
        market_data.load_manifest(ResearchPaths(tmp_path))


@pytest.mark.parametrize(
    "symbols",
    [
        [f"S{i:03}" for i in range(502)],
        [f"S{i:03}" for i in range(502)] + ["S000"],
        [f"S{i:03}" for i in range(502)] + [None],
    ],
)
def test_load_manifest_rejects_invalid_symbol_list(tmp_path, symbols):
    write_universe(tmp_path, symbols=symbols)
    with pytest.raises(ValueError):
        market_data.load_manifest(ResearchPaths(tmp_path))


def test_manifest_hash_changes_with_csv_content(tmp_path):
    destination = write_universe(tmp_path)
    paths = ResearchPaths(tmp_path)
    original = market_data.manifest_hash(paths)
    destination.write_text(destination.read_text().replace("S000", "X000"))
    assert market_data.manifest_hash(paths) != original
