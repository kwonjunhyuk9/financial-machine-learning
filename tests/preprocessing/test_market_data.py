from datetime import datetime, timezone
import json
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockTradesRequest
from src.preprocessing import market_data
from src.preprocessing.market_data import ResearchPaths


def test_normalize_trade_frame_rejects_missing_price_column():
    trades = pd.DataFrame({"timestamp": ["2026-01-01"], "symbol": ["AAPL"], "size": [1]})

    with pytest.raises(ValueError, match="price"):
        market_data._normalize_trade_frame(trades)


def test_fetch_alpaca_historical_data_requests_sip_stock_trades(monkeypatch):
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
    stock_client.get_stock_trades.return_value = response

    monkeypatch.setattr(market_data, "_get_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(
        market_data,
        "StockHistoricalDataClient",
        Mock(return_value=stock_client),
    )
    result = market_data.fetch_alpaca_historical_data(
        symbols=["AAPL"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    request = stock_client.get_stock_trades.call_args.args[0]
    assert isinstance(request, StockTradesRequest)
    assert request.feed == DataFeed.SIP
    assert result.columns.tolist() == ["timestamp", "symbol", "price", "size"]
    assert result["timestamp"].tolist() == [pd.Timestamp("2026-01-01T00:00:00Z")]


def write_universe(tmp_path, symbols=None, column="symbol"):
    destination = tmp_path / "data/preprocessing/universe/sp500_2025.csv"
    destination.parent.mkdir(parents=True)
    symbols = symbols if symbols is not None else [f"S{i:03}" for i in range(503)]
    pd.DataFrame({column: symbols}).to_csv(destination, index=False)
    return destination


def configure_tick_collection(monkeypatch, symbols):
    market_open = pd.Timestamp("2025-01-02T14:30:00Z")
    market_close = pd.Timestamp("2025-01-02T21:00:00Z")
    monkeypatch.setattr(
        market_data,
        "load_manifest",
        lambda unused_paths: pd.DataFrame({"symbol": symbols}),
    )
    monkeypatch.setattr(
        market_data,
        "sessions",
        lambda unused_paths: pd.DataFrame(
            {"session": ["2025-01-02"], "open": [market_open], "close": [market_close]}
        ),
    )
    monkeypatch.setattr(market_data, "manifest_hash", lambda unused_paths: "universe")
    return market_open, market_close


def test_research_paths_use_preprocessing_data_store(tmp_path):
    paths = ResearchPaths(tmp_path)

    assert paths.data == tmp_path / "data/preprocessing"
    assert paths.universe == tmp_path / "data/preprocessing/universe"
    assert paths.raw("AAPL", "tick") == tmp_path / "data/preprocessing/market/AAPL/raw/tick"
    assert paths.event("model") == (
        tmp_path
        / "data/preprocessing/events/sp500_model_events_2025-02-01_2025-12-31.parquet"
    )


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

    assert directory == tmp_path / "data/preprocessing/market/AAPL/raw/tick"
    assert result["price"].tolist() == [100.0]


def test_collect_raw_rejects_removed_minute_data_kind(tmp_path):
    with pytest.raises(ValueError, match="tick.*news"):
        market_data.collect_raw(ResearchPaths(tmp_path), "1min")


def test_collect_raw_rejects_nonpositive_worker_count(tmp_path):
    with pytest.raises(ValueError, match="max_workers"):
        market_data.collect_raw(ResearchPaths(tmp_path), "tick", max_workers=0)


def test_collect_raw_bounds_parallel_symbols_and_preserves_order(tmp_path, monkeypatch):
    paths = ResearchPaths(tmp_path)
    symbols = ["A", "B", "C"]
    configure_tick_collection(monkeypatch, symbols)
    barrier = Barrier(3)
    lock = Lock()
    active = 0
    peak = 0

    def fetch(*, symbols, start, end):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if symbols[0] != "SPY":
                barrier.wait(timeout=2)
            return pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp(start)],
                    "symbol": symbols,
                    "price": [100.0],
                    "size": [1.0],
                }
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(market_data, "fetch_alpaca_historical_data", fetch)

    result = market_data.collect_raw(paths, "tick", max_workers=3)

    assert peak == 3
    assert result["symbol"].tolist() == ["A", "B", "C", "SPY"]


def test_collect_raw_uses_same_executor_path_with_one_worker(tmp_path, monkeypatch):
    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["A", "B"])
    calls = []

    def fetch(*, symbols, start, end):
        calls.append(symbols[0])
        return pd.DataFrame(
            columns=["timestamp", "symbol", "price", "size"]
        )

    monkeypatch.setattr(market_data, "fetch_alpaca_historical_data", fetch)

    result = market_data.collect_raw(paths, "tick", max_workers=1)

    assert calls == ["A", "B", "SPY"]
    assert result["symbol"].tolist() == calls


def test_collect_raw_uses_one_worker_for_news(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["A", "B"])
    monkeypatch.setattr(
        market_data, "RESEARCH_START", pd.Timestamp("2025-02-01", tz="UTC")
    )
    monkeypatch.setattr(market_data, "END", pd.Timestamp("2025-02-02", tz="UTC"))
    calls = []

    def fetch(*, symbols, start, end):
        calls.append(symbols[0])
        return pd.DataFrame({"created_at": [pd.Timestamp(start)]})

    monkeypatch.setattr(alternative_data, "fetch_alpaca_news", fetch)
    monkeypatch.setattr(
        alternative_data,
        "filter_symbol_news",
        lambda frame, unused_symbol: frame,
    )

    result = market_data.collect_raw(paths, "news", max_workers=1)

    assert calls == ["A", "B"]
    assert result["symbol"].tolist() == calls


def test_collect_raw_propagates_failure_and_keeps_completed_partition(
    tmp_path, monkeypatch
):
    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["DONE", "FAIL"])

    def fetch(*, symbols, start, end):
        if symbols == ["FAIL"]:
            raise RuntimeError("download failed")
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(start)],
                "symbol": symbols,
                "price": [100.0],
                "size": [1.0],
            }
        )

    monkeypatch.setattr(market_data, "fetch_alpaca_historical_data", fetch)

    with pytest.raises(RuntimeError, match="download failed"):
        market_data.collect_raw(paths, "tick", max_workers=1)

    completed = paths.raw("DONE", "tick") / "2025-01-02.parquet"
    assert completed.exists()
    assert completed.with_suffix(".json").exists()
    assert market_data.read_raw(paths, "DONE", "tick")["symbol"].tolist() == ["DONE"]


def test_collect_raw_reuses_completed_partitions_in_preprocessing_store(
    tmp_path, monkeypatch
):
    paths = ResearchPaths(tmp_path)
    write_universe(tmp_path)
    market_open = pd.Timestamp("2025-01-02T14:30:00Z")
    market_close = pd.Timestamp("2025-01-02T21:00:00Z")
    identity = market_data.manifest_hash(paths)
    monkeypatch.setattr(
        market_data,
        "load_manifest",
        lambda unused_paths: pd.DataFrame({"symbol": ["A"]}),
    )
    monkeypatch.setattr(
        market_data,
        "sessions",
        lambda unused_paths: pd.DataFrame(
            {"session": ["2025-01-02"], "open": [market_open], "close": [market_close]}
        ),
    )
    fetch = Mock(side_effect=AssertionError("completed partitions must not be fetched"))
    monkeypatch.setattr(market_data, "fetch_alpaca_historical_data", fetch)

    for symbol in ["A", "SPY"]:
        destination = paths.raw(symbol, "tick") / "2025-01-02.parquet"
        market_data.save_frame(pd.DataFrame(), destination)
        destination.with_suffix(".json").write_text(
            json.dumps(
                {
                    "version": market_data.VERSION,
                    "universe": identity,
                    "kind": "tick",
                    "feed": "sip",
                    "start": str(market_open),
                    "end": str(market_close),
                    "request_symbol": symbol,
                    "rows": 0,
                }
            )
        )

    result = market_data.collect_raw(paths, "tick", max_workers=3)

    fetch.assert_not_called()
    assert result.to_dict("records") == [
        {"symbol": "A", "kind": "tick", "rows": 0},
        {"symbol": "SPY", "kind": "tick", "rows": 0},
    ]


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
