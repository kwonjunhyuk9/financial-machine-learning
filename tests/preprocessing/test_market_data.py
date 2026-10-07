from datetime import datetime, timezone
import json
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from alpaca.data.enums import DataFeed
from alpaca.data.models.trades import TradeSet
from alpaca.data.requests import StockTradesRequest
from src.preprocessing import market_data
from src.preprocessing.market_data import ResearchPaths


def test_normalize_trade_frame_rejects_missing_price_column():
    trades = pd.DataFrame({"timestamp": ["2026-01-01"], "symbol": ["AAPL"], "size": [1]})

    with pytest.raises(ValueError, match="price"):
        market_data._normalize_trade_frame(trades)


def test_fetch_alpaca_historical_data_handles_empty_response(monkeypatch):
    stock_client = Mock()
    stock_client.get_stock_trades.return_value = TradeSet({})
    monkeypatch.setattr(market_data, "_get_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(
        market_data, "StockHistoricalDataClient", Mock(return_value=stock_client)
    )

    result = market_data.fetch_alpaca_historical_data(
        symbols=["AAPL"],
        start=datetime(2025, 1, 2, tzinfo=timezone.utc),
        end=datetime(2025, 1, 3, tzinfo=timezone.utc),
    )

    assert result.empty
    assert result.columns.tolist() == ["timestamp", "symbol", "price", "size"]
    assert isinstance(result["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_fetch_alpaca_historical_data_requests_sip_stock_trades(monkeypatch):
    stock_client = Mock()
    stock_client.get_stock_trades.return_value = {
        "AAPL": [
            {"t": "2026-01-01T00:00:00Z", "p": 100, "s": 1},
            {"t": "2026-01-02T00:00:00Z", "p": 101, "s": 2},
        ]
    }

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


def configure_raw_tick_client(monkeypatch, calls, failure_symbol=None):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._session = SimpleNamespace(close=Mock())

        def get(self, unused_path, data):
            symbol = data["symbols"]
            calls.append(symbol)
            if symbol == failure_symbol:
                raise RuntimeError("download failed")
            return {
                "trades": {
                    symbol: [{
                        "t": data["start"], "p": 100.0, "s": 1,
                    }]
                }
            }

    monkeypatch.setattr(market_data, "_get_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", FakeClient)


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
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._session = SimpleNamespace(close=Mock())

        def get(self, unused_path, data):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if data["symbols"] != "SPY":
                    barrier.wait(timeout=2)
                return {"trades": {data["symbols"]: [{
                    "t": data["start"], "p": 100.0, "s": 1,
                }]}}
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(market_data, "_get_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", FakeClient)

    result = market_data.collect_raw(paths, "tick", max_workers=3)

    assert peak == 3
    assert result["symbol"].tolist() == ["A", "B", "C", "SPY"]


def test_collect_raw_uses_same_executor_path_with_one_worker(tmp_path, monkeypatch):
    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["A", "B"])
    calls = []
    configure_raw_tick_client(monkeypatch, calls)

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
        calls.append(symbols)
        return pd.DataFrame({"created_at": [pd.Timestamp(start)]})

    monkeypatch.setattr(alternative_data, "fetch_alpaca_news", fetch)
    monkeypatch.setattr(
        alternative_data,
        "filter_symbol_news",
        lambda frame, unused_symbol: frame,
    )

    result = market_data.collect_raw(paths, "news", max_workers=1)

    assert calls == [["A", "B"]]
    assert result["symbol"].tolist() == ["A", "B"]


def test_collect_raw_propagates_failure_and_keeps_completed_partition(
    tmp_path, monkeypatch
):
    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["DONE", "FAIL"])

    calls = []
    configure_raw_tick_client(monkeypatch, calls, failure_symbol="FAIL")

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


def test_normalize_raw_trades_preserves_precision_and_order():
    raw = {
        "AAPL": [
            {"t": "2025-01-02T14:30:00.123456789Z", "p": 2, "s": 1},
            {"t": "2025-01-02T14:30:00.123456999Z", "p": 3, "s": 1},
            {"t": "2025-01-02T14:35:00Z", "p": 4, "s": 1},
        ]
    }

    result = market_data._normalize_raw_trade_frame(raw)

    assert result["price"].tolist() == [2.0, 3.0, 4.0]
    assert result["timestamp"].tolist()[0] == pd.Timestamp(
        "2025-01-02T14:30:00.123456Z"
    )
    assert result["timestamp"].dtype == "datetime64[us, UTC]"


def test_write_tick_partition_streams_pages_and_writes_empty_results(tmp_path):
    start = pd.Timestamp("2025-01-02T14:30:00Z")
    end = start + pd.Timedelta(minutes=5)

    class Client:
        def __init__(self):
            self.calls = 0
            self._session = SimpleNamespace(close=Mock())

        def get(self, unused_path, data):
            self.calls += 1
            if self.calls == 1:
                return {
                    "trades": {"AAPL": [
                        {"t": "2025-01-02T14:30:00.123456789Z", "p": 2, "s": 1},
                    ]},
                    "next_page_token": "next",
                }
            return {
                "trades": {"AAPL": [
                    {"t": "2025-01-02T14:31:00Z", "p": 3, "s": 1},
                ]},
                "next_page_token": None,
            }

    destination = tmp_path / "2025-01-02.parquet"
    rows = market_data._write_tick_partition(
        Client(), "AAPL", start.to_pydatetime(), end.to_pydatetime(), destination
    )

    assert rows == 2
    assert pd.read_parquet(destination)["price"].tolist() == [2.0, 3.0]
    assert not destination.with_suffix(".pending.parquet").exists()


def test_write_tick_partition_writes_empty_result_with_schema(tmp_path):
    start = pd.Timestamp("2025-01-02T14:30:00Z")
    destination = tmp_path / "2025-01-02.parquet"

    class Client:
        def get(self, unused_path, data):
            return {"trades": {}, "next_page_token": None}

    rows = market_data._write_tick_partition(
        Client(), "AAPL", start.to_pydatetime(),
        (start + pd.Timedelta(minutes=5)).to_pydatetime(), destination
    )

    frame = pd.read_parquet(destination)
    assert rows == 0
    assert frame.empty
    assert frame.columns.tolist() == ["timestamp", "symbol", "price", "size"]


def test_write_tick_partition_failure_keeps_old_file_and_cleans_temporary(tmp_path):
    start = pd.Timestamp("2025-01-02T14:30:00Z")
    destination = tmp_path / "2025-01-02.parquet"
    destination.write_bytes(b"old complete file")

    class Client:
        def get(self, unused_path, data):
            raise RuntimeError("page failed")

    with pytest.raises(RuntimeError, match="page failed"):
        market_data._write_tick_partition(
            Client(), "AAPL", start.to_pydatetime(),
            (start + pd.Timedelta(minutes=5)).to_pydatetime(), destination
        )

    assert destination.read_bytes() == b"old complete file"
    assert not destination.with_suffix(".pending.parquet").exists()


def test_write_tick_partition_rejects_reverse_page_order(tmp_path):
    start = pd.Timestamp("2025-01-02T14:30:00Z")
    destination = tmp_path / "2025-01-02.parquet"

    class Client:
        def __init__(self):
            self.calls = 0

        def get(self, unused_path, data):
            self.calls += 1
            timestamp = "2025-01-02T14:31:00Z" if self.calls == 1 else "2025-01-02T14:30:00Z"
            return {
                "trades": {"AAPL": [{"t": timestamp, "p": 2, "s": 1}]},
                "next_page_token": "next" if self.calls == 1 else None,
            }

    with pytest.raises(ValueError, match="not ordered"):
        market_data._write_tick_partition(
            Client(), "AAPL", start.to_pydatetime(),
            (start + pd.Timedelta(minutes=5)).to_pydatetime(), destination
        )

    assert not destination.exists()


def test_collect_raw_reuses_one_client_per_symbol_across_dates(tmp_path, monkeypatch):
    paths = ResearchPaths(tmp_path)
    opens = pd.to_datetime(["2025-01-02T14:30Z", "2025-01-03T14:30Z"], utc=True)
    closes = pd.to_datetime(["2025-01-02T21:00Z", "2025-01-03T21:00Z"], utc=True)
    monkeypatch.setattr(market_data, "load_manifest", lambda unused: pd.DataFrame({"symbol": ["A"]}))
    monkeypatch.setattr(
        market_data, "sessions",
        lambda unused: pd.DataFrame({"session": ["a", "b"], "open": opens, "close": closes}),
    )
    monkeypatch.setattr(market_data, "manifest_hash", lambda unused: "universe")
    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self._session = SimpleNamespace(close=Mock())
            clients.append(self)

        def get(self, unused_path, data):
            self.calls += 1
            return {"trades": {}, "next_page_token": None}

    monkeypatch.setattr(market_data, "_get_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", Client)

    result = market_data.collect_raw(paths, "tick", max_workers=2)

    assert result["rows"].tolist() == [0, 0]
    assert len(clients) == 2
    assert [client.calls for client in clients] == [2, 2]
    assert all(client._session.close.call_count == 1 for client in clients)


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


def test_news_batch_preserves_raw_and_reselects_without_requests(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["A", "B"])
    start = pd.Timestamp("2025-02-01", tz="UTC")
    end = start + pd.Timedelta(days=1)
    monkeypatch.setattr(market_data, "RESEARCH_START", start)
    monkeypatch.setattr(market_data, "END", end)
    news = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "symbols": ["A", "B", "A,B", "A", "A"],
        "url": ["https://benzinga.com/news/25/02/123/a"] * 3
               + ["https://example.com/a", "https://benzinga.com/news/25/02/123/a"],
        "created_at": [start] * 4 + [end],
        "content": ["body"] * 5,
    })
    fetch = Mock(return_value=news)
    monkeypatch.setattr(alternative_data, "fetch_alpaca_news", fetch)
    # Selected legacy partitions cannot substitute for missing unfiltered responses.
    for symbol in ["A", "B"]:
        legacy = paths.raw(symbol, "news") / "2025-02-01.parquet"
        market_data.save_frame(news.iloc[:0], legacy)
        legacy.with_suffix(".json").write_text(json.dumps({
            "version": market_data.VERSION, "universe": market_data.manifest_hash(paths),
            "kind": "news", "feed": "benzinga", "start": str(start), "end": str(end),
            "request_symbol": symbol, "rows": 0,
        }))
    result = market_data.collect_raw(paths, "news")
    assert fetch.call_args.kwargs["symbols"] == ["A", "B"]
    assert result.rows.tolist() == [1, 1]
    source = paths.data / "alternative/raw/news/2025-02-01.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(source), news)
    for symbol in ["A", "B"]:
        expected = alternative_data.filter_symbol_news(news, symbol)
        expected = expected.loc[expected.created_at.lt(end)].copy()
        expected["symbol"] = symbol
        pd.testing.assert_frame_equal(market_data.read_raw(paths, symbol, "news"), expected)
    market_data.collect_raw(paths, "news")
    assert fetch.call_count == 1
    # A legacy selection record is regenerated from the preserved response.
    record = paths.raw("A", "news") / "2025-02-01.json"
    saved = json.loads(record.read_text())
    saved.pop("filter_version")
    record.write_text(json.dumps(saved))
    market_data.collect_raw(paths, "news")
    assert "filter_version" in json.loads(record.read_text())
    # Source changes invalidate selections without a network request.
    market_data.save_frame(news.iloc[:0], source)
    assert market_data.collect_raw(paths, "news").rows.tolist() == [0, 0]
    assert fetch.call_count == 1
    # Missing completion record requires recollection.
    source.with_suffix(".json").unlink()
    market_data.collect_raw(paths, "news")
    assert fetch.call_count == 2
    metadata = json.loads(source.with_suffix(".json").read_text())
    metadata["universe"] = "wrong"
    source.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Cache metadata mismatch"):
        market_data.collect_raw(paths, "news")


def test_news_selection_failure_keeps_raw_for_resume(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["A", "B"])
    monkeypatch.setattr(market_data, "RESEARCH_START", pd.Timestamp("2025-02-01", tz="UTC"))
    monkeypatch.setattr(market_data, "END", pd.Timestamp("2025-02-02", tz="UTC"))
    news = pd.DataFrame(columns=["symbols", "url", "created_at"])
    fetch = Mock(return_value=news)
    monkeypatch.setattr(alternative_data, "fetch_alpaca_news", fetch)
    original = alternative_data.filter_symbol_news

    def select(frame, symbol):
        if symbol == "B":
            raise RuntimeError("selection failed")
        return original(frame, symbol)

    monkeypatch.setattr(alternative_data, "filter_symbol_news", select)
    with pytest.raises(RuntimeError, match="selection failed"):
        market_data.collect_raw(paths, "news")
    assert (paths.raw("A", "news") / "2025-02-01.json").exists()
    monkeypatch.setattr(alternative_data, "filter_symbol_news", original)
    assert market_data.collect_raw(paths, "news").rows.tolist() == [0, 0]
    assert fetch.call_count == 1


def test_news_workers_process_dates_and_sum_in_manifest_order(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path)
    configure_tick_collection(monkeypatch, ["B", "A"])
    monkeypatch.setattr(market_data, "RESEARCH_START", pd.Timestamp("2025-02-01", tz="UTC"))
    monkeypatch.setattr(market_data, "END", pd.Timestamp("2025-02-03", tz="UTC"))
    barrier = Barrier(2)
    calls = []

    def fetch(*, symbols, start, end):
        calls.append((symbols, start))
        barrier.wait(timeout=2)
        return pd.DataFrame({
            "symbols": ["A"], "created_at": [pd.Timestamp(start)],
            "url": ["https://benzinga.com/news/25/02/123/a"],
        })

    monkeypatch.setattr(alternative_data, "fetch_alpaca_news", fetch)
    result = market_data.collect_raw(paths, "news", max_workers=2)
    assert len(calls) == 2
    assert all(symbols == ["B", "A"] for symbols, unused in calls)
    assert result.symbol.tolist() == ["B", "A"]
    assert result.rows.tolist() == [0, 2]
