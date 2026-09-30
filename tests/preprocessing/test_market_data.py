from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from alpaca.data.requests import (
    CryptoTradesRequest,
    StockTradesRequest,
)
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


@pytest.mark.parametrize(
    ("asset_class", "method_name", "request_type"),
    [
        ("stock", "get_stock_trades", StockTradesRequest),
        ("crypto", "get_crypto_trades", CryptoTradesRequest),
    ],
)
def test_fetch_alpaca_historical_data_dispatches_request(
    monkeypatch,
    asset_class,
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
    response = SimpleNamespace(df=trades)
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
    )

    client = stock_client if asset_class == "stock" else crypto_client
    request = getattr(client, method_name).call_args.args[0]
    assert isinstance(request, request_type)
    assert result.columns.tolist() == ["timestamp", "symbol", "price", "size"]
    assert result["timestamp"].tolist() == [pd.Timestamp("2026-01-01T00:00:00Z")]
def test_fetch_alpaca_historical_data_rejects_invalid_asset_class():
    with pytest.raises(ValueError, match="asset_class"):
        market_data.fetch_alpaca_historical_data(
            symbols=["AAPL"],
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 2),
            asset_class="option",
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


def test_sessions_filters_cached_calendar_to_2025(tmp_path):
    paths = ResearchPaths(tmp_path)
    paths.universe.mkdir(parents=True)
    pd.DataFrame(
        {
            "session": ["2024-12-31", "2025-01-02", "2026-01-02"],
            "open": pd.to_datetime(
                ["2024-12-31T14:30Z", "2025-01-02T14:30Z", "2026-01-02T14:30Z"]
            ),
            "close": pd.to_datetime(
                ["2024-12-31T21:00Z", "2025-01-02T21:00Z", "2026-01-02T21:00Z"]
            ),
        }
    ).to_parquet(paths.universe / "sessions.parquet", index=False)

    result = market_data.sessions(paths)

    assert result["session"].tolist() == ["2025-01-02"]


def test_read_raw_ignores_pre_2025_partitions(tmp_path):
    paths = ResearchPaths(tmp_path)
    directory = paths.raw("AAPL", "tick")
    directory.mkdir(parents=True)
    for date, price in [("2024-12-31", 99.0), ("2025-01-02", 100.0)]:
        path = directory / f"{date}.parquet"
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(f"{date}T15:00Z")],
                "symbol": ["AAPL"],
                "price": [price],
                "size": [1.0],
            }
        ).to_parquet(path, index=False)
        path.with_suffix(".json").write_text("{}")

    result = market_data.read_raw(paths, "AAPL", "tick")

    assert result["price"].tolist() == [100.0]


def test_collect_raw_rejects_removed_minute_data_kind(tmp_path):
    with pytest.raises(ValueError, match="tick.*news"):
        market_data.collect_raw(ResearchPaths(tmp_path), "1min")


def test_research_dates_keep_preparation_inside_2025():
    assert market_data.DATA_START == pd.Timestamp("2025-01-01", tz="UTC")
    assert market_data.RESEARCH_START == pd.Timestamp("2025-02-01", tz="UTC")
    assert market_data.END == pd.Timestamp("2026-01-01", tz="UTC")


def test_raw_partitions_use_data_start_for_ticks_and_research_start_for_news(
    tmp_path,
):
    paths = ResearchPaths(tmp_path)
    for kind in ["tick", "news"]:
        directory = paths.raw("AAPL", kind)
        directory.mkdir(parents=True)
        for date in ["2025-01-02", "2025-02-03"]:
            (directory / f"{date}.parquet").touch()

    tick_names = [path.name for path in market_data.raw_partitions(paths, "AAPL", "tick")]
    news_names = [path.name for path in market_data.raw_partitions(paths, "AAPL", "news")]

    assert tick_names == ["2025-01-02.parquet", "2025-02-03.parquet"]
    assert news_names == ["2025-02-03.parquet"]


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
