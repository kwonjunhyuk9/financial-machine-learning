import json
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.preprocessing import market_data
from src.preprocessing.market_data import ResearchPaths


@pytest.mark.parametrize("tape,conditions,accepted", [
    ("A", [" "], True), ("B", [" "], True), ("C", ["@"], True),
    ("A", ["@"], False), ("C", [" "], False),
    ("A", [" ", "I"], True), ("B", [" ", "I"], True),
    ("C", ["@", "I"], True), ("C", ["I"], True),
    ("A", ["I"], True), ("B", ["I"], True),
    ("A", ["@", "I"], False), ("C", [" ", "I"], False),
    ("C", ["@", "I", "Q"], False), ("C", ["@", "I", "M"], False),
    ("C", ["@", "I", "F"], False), ("C", ["@", "I", "T"], False),
    ("C", ["Q"], False), ("C", ["M"], False), ("C", ["6"], False),
])
def test_raw_regular_and_odd_lot_filter(tape, conditions, accepted):
    raw = {"A": [{"t": "2025-01-02T14:30Z", "p": 100, "s": 5,
                  "c": conditions, "z": tape}]}
    result = market_data._normalize_raw_trade_frame(raw)
    assert len(result) == int(accepted)
    assert result.columns.tolist() == ["timestamp", "symbol", "price", "size"]
    if accepted:
        assert result[["price", "size"]].values.tolist() == [[100., 5.]]


@pytest.mark.parametrize("conditions,tape", [
    (None, "C"), ([], "C"), ("@", "C"), (["?"], "C"),
    ([None], "C"), (["@"], None), (["@"], "unknown"),
])
def test_unknown_trade_metadata_fails(conditions, tape):
    with pytest.raises(ValueError, match="Unknown or missing"):
        market_data._normalize_raw_trade_frame({"A": [
            {"t": "2025-01-02T14:30Z", "p": 100, "s": 5, "c": conditions, "z": tape},
        ]})


def test_odd_lot_contributes_to_dollar_threshold_volume_and_prices():
    from src.preprocessing.market_structured_bars import get_dollar_bars

    frame = market_data._normalize_raw_trade_frame({"A": [
        {"t": "2025-01-02T14:30:00Z", "p": 100., "s": 1., "c": ["@"], "z": "C"},
        {"t": "2025-01-02T14:30:01Z", "p": 110., "s": 1., "c": ["@", "I"], "z": "C"},
    ]})
    bars = get_dollar_bars(frame, threshold=200)
    assert bars[["open", "high", "low", "close", "volume", "dollar_value"]].values.tolist() == [
        [100., 110., 100., 110., 2., 210.],
    ]


def test_invalid_conditions_do_not_replace_a_completed_tick_file(tmp_path):
    destination = tmp_path / "ticks.parquet"
    destination.write_bytes(b"old completed partition")
    client = SimpleNamespace(get=Mock(return_value={"trades": {"A": [
        {"t": "2025-01-02T14:30Z", "p": 100, "s": 1, "z": "C"},
    ]}}))
    with pytest.raises(ValueError, match="Unknown or missing trade conditions"):
        market_data._write_tick_partition(client, "A", pd.Timestamp("2025-01-02T14:30Z"),
                                          pd.Timestamp("2025-01-02T21:00Z"), destination)
    assert destination.read_bytes() == b"old completed partition"
    assert not destination.with_suffix(".pending.parquet").exists()


def write_universe(tmp_path, symbols=None, column="symbol"):
    destination = tmp_path / "data/preprocessing/universe/sp500_2025.csv"
    destination.parent.mkdir(parents=True)
    symbols = symbols if symbols is not None else [f"S{i:03}" for i in range(503)]
    pd.DataFrame({column: symbols}).to_csv(destination, index=False)
    return destination


def configure_tick_collection(monkeypatch, symbols):
    monkeypatch.setattr(market_data, "collect_split_records", Mock())
    market_open = pd.Timestamp("2025-01-02T14:30:00Z")
    market_close = pd.Timestamp("2025-01-02T21:00:00Z")
    monkeypatch.setattr(
        market_data,
        "load_manifest",
        lambda unused_paths, **kwargs: pd.DataFrame({"symbol": symbols}),
    )
    monkeypatch.setattr(
        market_data,
        "sessions",
        lambda unused_paths, **kwargs: pd.DataFrame(
            {"session": ["2025-01-02"], "open": [market_open], "close": [market_close]}
        ),
    )
    monkeypatch.setattr(market_data, "manifest_hash", lambda unused_paths, **kwargs: "universe")
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
                        "t": data["start"], "p": 100.0, "s": 1, "c": ["@"], "z": "C",
                    }]
                }
            }

    monkeypatch.setattr(market_data, "_get_alpaca_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", FakeClient)


def test_research_paths_use_preprocessing_data_store(tmp_path):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))

    assert paths.data == tmp_path / "data/preprocessing"
    assert paths.universe == tmp_path / "data/preprocessing/universe"
    assert paths.raw("AAPL", "tick") == tmp_path / "data/preprocessing/market/stock/AAPL/raw/tick"
    assert paths.event("prepared") == (
        tmp_path
        / "data/preprocessing/events/sp500_events_prepared_2025-02-01_2025-12-31.parquet"
    )


@pytest.mark.parametrize("count", [1, 502, 503])
def test_load_manifest_reads_fixed_universe_csv(tmp_path, count):
    write_universe(tmp_path, symbols=[f"S{i:03}" for i in range(count)])
    manifest = market_data.load_manifest(tmp_path / "data/preprocessing/universe/sp500_2025.csv")
    assert manifest.columns.tolist() == ["symbol"]
    assert len(manifest) == manifest.symbol.nunique() == count


def test_load_manifest_rejects_wrong_column(tmp_path):
    write_universe(tmp_path, column="ticker")
    with pytest.raises(ValueError, match="only the symbol column"):
        market_data.load_manifest(tmp_path / "data/preprocessing/universe/sp500_2025.csv")


def test_sessions_filters_cached_calendar_to_2025(tmp_path):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
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

    (paths.universe / "sessions.json").write_text(json.dumps({"start": "2024-01-01T00:00Z", "end": "2027-01-01T00:00Z"}))
    result = market_data.sessions(paths, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))

    assert result["session"].tolist() == ["2025-01-02"]


def test_read_raw_ignores_pre_2025_partitions(tmp_path):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
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
        path.with_suffix(".json").write_text(json.dumps({}))

    result = market_data.read_raw(paths, "AAPL", "tick", start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))

    assert directory == tmp_path / "data/preprocessing/market/stock/AAPL/raw/tick"
    assert result["price"].tolist() == [100.0]


def test_collect_raw_rejects_removed_minute_data_kind(tmp_path):
    with pytest.raises(ValueError, match="tick.*news"):
        market_data.collect_raw(ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC")), "1min", start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1)


def test_collect_raw_rejects_nonpositive_worker_count(tmp_path):
    with pytest.raises(ValueError, match="max_workers"):
        market_data.collect_raw(ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC")), "tick", max_workers=0, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv")


def test_collect_raw_bounds_parallel_symbols_and_preserves_order(tmp_path, monkeypatch):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
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
                    "t": data["start"], "p": 100.0, "s": 1, "c": ["@"], "z": "C",
                }]}}
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(market_data, "_get_alpaca_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", FakeClient)

    result = market_data.collect_raw(paths, "tick", max_workers=3, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", show_progress=True)

    assert result.columns.tolist() == ["symbol", "output_rows", "processed_files", "cached_files"]
    assert result.processed_files.tolist() == [1, 1, 1, 1]
    assert result.cached_files.tolist() == [0, 0, 0, 0]
    assert peak == 3
    assert result["symbol"].tolist() == ["A", "B", "C", "SPY"]


def test_collect_raw_uses_same_executor_path_with_one_worker(tmp_path, monkeypatch):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    configure_tick_collection(monkeypatch, ["A", "B"])
    calls = []
    configure_raw_tick_client(monkeypatch, calls)

    result = market_data.collect_raw(paths, "tick", max_workers=1, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv")

    assert calls == ["A", "B", "SPY"]
    assert result["symbol"].tolist() == calls


def test_collect_raw_uses_one_worker_for_news(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    configure_tick_collection(monkeypatch, ["A", "B"])
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

    result = market_data.collect_raw(paths, "news", max_workers=1, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2025-02-02", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv")

    assert calls == [["A", "B"]]
    assert result["symbol"].tolist() == ["A", "B"]


def test_collect_raw_propagates_failure_and_keeps_completed_partition(
    tmp_path, monkeypatch
):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    configure_tick_collection(monkeypatch, ["DONE", "FAIL"])

    calls = []
    configure_raw_tick_client(monkeypatch, calls, failure_symbol="FAIL")

    with pytest.raises(RuntimeError, match="download failed"):
        market_data.collect_raw(paths, "tick", max_workers=1, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv")

    completed = paths.raw("DONE", "tick") / "2025-01-02.parquet"
    assert completed.exists()
    assert completed.with_suffix(".json").exists()
    assert market_data.read_raw(paths, "DONE", "tick", start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))["symbol"].tolist() == ["DONE"]


def test_collect_raw_reuses_completed_partitions_in_preprocessing_store(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(market_data, "collect_split_records", Mock())
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    write_universe(tmp_path)
    market_open = pd.Timestamp("2025-01-02T14:30:00Z")
    market_close = pd.Timestamp("2025-01-02T21:00:00Z")
    identity = market_data.manifest_hash(tmp_path / "data/preprocessing/universe/sp500_2025.csv")
    monkeypatch.setattr(
        market_data,
        "load_manifest",
        lambda unused_paths, **kwargs: pd.DataFrame({"symbol": ["A"]}),
    )
    monkeypatch.setattr(
        market_data,
        "sessions",
        lambda unused_paths, **kwargs: pd.DataFrame(
            {"session": ["2025-01-02"], "open": [market_open], "close": [market_close]}
        ),
    )
    fetch = Mock(side_effect=AssertionError("completed partitions must not be fetched"))
    monkeypatch.setattr(market_data, "_write_tick_partition", fetch)

    for symbol in ["A", "SPY"]:
        destination = paths.raw(symbol, "tick") / "2025-01-02.parquet"
        market_data.save_frame(pd.DataFrame(), destination)
        destination.with_suffix(".json").write_text(
            json.dumps(
                {

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

    result = market_data.collect_raw(paths, "tick", max_workers=3, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv")

    fetch.assert_not_called()
    assert result.to_dict("records") == [
        {"symbol": "A", "output_rows": 0,
         "processed_files": 0, "cached_files": 1},
        {"symbol": "SPY", "output_rows": 0,
         "processed_files": 0, "cached_files": 1},
    ]


def test_research_paths_require_explicit_boundaries(tmp_path):
    with pytest.raises(TypeError, match="start.*end"):
        ResearchPaths(tmp_path)


@pytest.mark.parametrize("start,end,period", [
    ("2025-02-01", "2026-01-01", "2025-02-01_2025-12-31"),
    ("2024-02-01", "2024-03-01", "2024-02-01_2024-02-29"),
    ("2023-02-01", "2023-03-01", "2023-02-01_2023-02-28"),
])
@pytest.mark.parametrize("stage", ["candidates", "labeled", "weighted", "prepared"])
def test_research_paths_derive_event_period(tmp_path, start, end, period, stage):
    paths = ResearchPaths(
        tmp_path, start=pd.Timestamp(start, tz="UTC"), end=pd.Timestamp(end, tz="UTC"),
    )
    assert paths.period == period
    assert paths.event(stage) == (
        tmp_path / "data/preprocessing/events" / f"sp500_events_{stage}_{period}.parquet"
    )


@pytest.mark.parametrize("name", ["start", "end"])
@pytest.mark.parametrize("invalid", [
    pd.Timestamp("2025-06-01"),
    pd.Timestamp("2025-06-01", tz="Asia/Seoul"),
    pd.Timestamp("2025-06-01 01:00", tz="UTC"),
    pd.NaT,
    "2025-06-01",
])
def test_research_paths_reject_invalid_boundaries(tmp_path, name, invalid):
    boundaries = {
        "start": pd.Timestamp("2025-02-01", tz="UTC"),
        "end": pd.Timestamp("2026-01-01", tz="UTC"),
    }
    boundaries[name] = invalid
    with pytest.raises(ValueError, match=f"{name} must be a UTC midnight"):
        ResearchPaths(tmp_path, **boundaries)


@pytest.mark.parametrize("end", ["2025-02-01", "2025-01-01"])
def test_research_paths_require_increasing_boundaries(tmp_path, end):
    with pytest.raises(ValueError, match="start must be before end"):
        ResearchPaths(
            tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
            end=pd.Timestamp(end, tz="UTC"),
        )


def test_research_paths_model_kind_and_required_boundaries(tmp_path):
    boundaries = {
        "start": pd.Timestamp("2025-02-01", tz="UTC"),
        "end": pd.Timestamp("2026-01-01", tz="UTC"),
    }
    assert ResearchPaths(tmp_path, "market", **boundaries) == ResearchPaths(
        tmp_path, model_kind="market", **boundaries,
    )
    with pytest.raises(ValueError, match="model_kind"):
        ResearchPaths(tmp_path, "unknown", **boundaries)
    for missing in boundaries:
        with pytest.raises(TypeError, match=missing):
            ResearchPaths(tmp_path, **{k: v for k, v in boundaries.items() if k != missing})


def test_normalize_raw_trades_preserves_precision_and_order():
    raw = {
        "AAPL": [
            {"t": "2025-01-02T14:30:00.123456789Z", "p": 2, "s": 1, "c": ["@"], "z": "C"},
            {"t": "2025-01-02T14:30:00.123456999Z", "p": 3, "s": 1, "c": ["@"], "z": "C"},
            {"t": "2025-01-02T14:35:00Z", "p": 4, "s": 1, "c": ["@"], "z": "C"},
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

    requests = []

    class Client:
        def __init__(self):
            self.calls = 0
            self._session = SimpleNamespace(close=Mock())

        def get(self, unused_path, data):
            requests.append((unused_path, data.copy()))
            self.calls += 1
            if self.calls == 1:
                return {
                    "trades": {"AAPL": [
                        {"t": "2025-01-02T14:30:00.123456789Z", "p": 2, "s": 1, "c": ["@"], "z": "C"},
                    ]},
                    "next_page_token": "next",
                }
            return {
                "trades": {"AAPL": [
                    {"t": "2025-01-02T14:31:00Z", "p": 3, "s": 1, "c": ["@"], "z": "C"},
                ]},
                "next_page_token": None,
            }

    destination = tmp_path / "2025-01-02.parquet"
    rows = market_data._write_tick_partition(
        Client(), "AAPL", start.to_pydatetime(), end.to_pydatetime(), destination
    )

    request_path, params = requests[0]
    assert request_path == "/stocks/trades"
    assert params["feed"] == "sip"
    assert params["symbols"] == "AAPL"
    assert pd.Timestamp(params["start"]) == start
    assert pd.Timestamp(params["end"]) == end
    assert params["sort"] == "asc"
    assert requests[1][1]["page_token"] == "next"
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
                "trades": {"AAPL": [{"t": timestamp, "p": 2, "s": 1, "c": ["@"], "z": "C"}]},
                "next_page_token": "next" if self.calls == 1 else None,
            }

    with pytest.raises(ValueError, match="not ordered"):
        market_data._write_tick_partition(
            Client(), "AAPL", start.to_pydatetime(),
            (start + pd.Timedelta(minutes=5)).to_pydatetime(), destination
        )

    assert not destination.exists()


def test_collect_raw_reuses_one_client_per_symbol_across_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, "collect_split_records", Mock())
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    opens = pd.to_datetime(["2025-01-02T14:30Z", "2025-01-03T14:30Z"], utc=True)
    closes = pd.to_datetime(["2025-01-02T21:00Z", "2025-01-03T21:00Z"], utc=True)
    monkeypatch.setattr(market_data, "load_manifest", lambda unused, **kwargs: pd.DataFrame({"symbol": ["A"]}))
    monkeypatch.setattr(
        market_data, "sessions",
        lambda unused, **kwargs: pd.DataFrame({"session": ["a", "b"], "open": opens, "close": closes}),
    )
    monkeypatch.setattr(market_data, "manifest_hash", lambda unused, **kwargs: "universe")
    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self._session = SimpleNamespace(close=Mock())
            clients.append(self)

        def get(self, unused_path, data):
            self.calls += 1
            return {"trades": {}, "next_page_token": None}

    monkeypatch.setattr(market_data, "_get_alpaca_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", Client)

    result = market_data.collect_raw(paths, "tick", max_workers=2, start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv")

    assert result["output_rows"].tolist() == [0, 0]
    assert len(clients) == 2
    assert [client.calls for client in clients] == [2, 2]
    assert all(client._session.close.call_count == 1 for client in clients)


def test_raw_partitions_use_data_start_for_ticks_and_research_start_for_news(
    tmp_path,
):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    for kind in ["tick", "news"]:
        directory = paths.raw("AAPL", kind)
        directory.mkdir(parents=True)
        for date in ["2025-01-02", "2025-02-03"]:
            (directory / f"{date}.parquet").touch()

    tick_names = [path.name for path in market_data.raw_partitions(paths, "AAPL", "tick", start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))]
    news_names = [path.name for path in market_data.raw_partitions(paths, "AAPL", "news", start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))]

    assert tick_names == ["2025-01-02.parquet", "2025-02-03.parquet"]
    assert news_names == ["2025-02-03.parquet"]


@pytest.mark.parametrize(
    "symbols",
    [
        [],
        [f"S{i:03}" for i in range(502)] + ["S000"],
        [f"S{i:03}" for i in range(502)] + [None],
        ["AAPL", ""],
    ],
)
def test_load_manifest_rejects_invalid_symbol_list(tmp_path, symbols):
    write_universe(tmp_path, symbols=symbols)
    with pytest.raises(ValueError):
        market_data.load_manifest(tmp_path / "data/preprocessing/universe/sp500_2025.csv")


def test_manifest_hash_changes_with_csv_content(tmp_path):
    destination = write_universe(tmp_path)
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    original = market_data.manifest_hash(tmp_path / "data/preprocessing/universe/sp500_2025.csv")
    destination.write_text(destination.read_text().replace("S000", "X000"))
    assert market_data.manifest_hash(tmp_path / "data/preprocessing/universe/sp500_2025.csv") != original


def test_news_batch_preserves_raw_and_reselects_without_requests(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    configure_tick_collection(monkeypatch, ["A", "B"])
    start = pd.Timestamp("2025-02-01", tz="UTC")
    end = start + pd.Timedelta(days=1)
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
            "universe": market_data.manifest_hash(tmp_path / "data/preprocessing/universe/sp500_2025.csv"),
            "kind": "news", "feed": "benzinga", "start": str(start), "end": str(end),
            "request_symbol": symbol, "rows": 0,
        }))
    result = market_data.collect_raw(paths, "news", start=start, end=end, manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1)
    assert fetch.call_args.kwargs["symbols"] == ["A", "B"]
    assert result.columns.tolist() == ["symbol", "output_rows", "processed_files", "cached_files"]
    assert result.output_rows.tolist() == [1, 1]
    source = paths.news_source / "2025-02-01.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(source), news)
    for symbol in ["A", "B"]:
        expected = alternative_data.filter_symbol_news(news, symbol)
        expected = expected.loc[expected.created_at.lt(end)].copy()
        expected["symbol"] = symbol
        pd.testing.assert_frame_equal(market_data.read_raw(paths, symbol, "news", start=start, end=end), expected)
    market_data.collect_raw(paths, "news", start=start, end=end, manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1)
    assert fetch.call_count == 1
    # A selection without its source identity is regenerated from the preserved response.
    record = paths.raw("A", "news") / "2025-02-01.json"
    saved = json.loads(record.read_text())
    saved.pop("source")
    record.write_text(json.dumps(saved))
    market_data.collect_raw(paths, "news", start=start, end=end, manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1)
    assert "source" in json.loads(record.read_text())
    # Source changes invalidate selections without a network request.
    market_data.save_frame(news.iloc[:0], source)
    assert market_data.collect_raw(paths, "news", start=start, end=end, manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1).output_rows.tolist() == [0, 0]
    assert fetch.call_count == 1
    # Missing completion record requires recollection.
    source.with_suffix(".json").unlink()
    market_data.collect_raw(paths, "news", start=start, end=end, manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1)
    assert fetch.call_count == 2
    metadata = json.loads(source.with_suffix(".json").read_text())
    metadata["universe"] = "wrong"
    source.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Cache metadata mismatch"):
        market_data.collect_raw(paths, "news", start=start, end=end, manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1)


def test_news_selection_failure_keeps_raw_for_resume(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    configure_tick_collection(monkeypatch, ["A", "B"])
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
        market_data.collect_raw(paths, "news", start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2025-02-02", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1)
    assert (paths.raw("A", "news") / "2025-02-01.json").exists()
    monkeypatch.setattr(alternative_data, "filter_symbol_news", original)
    assert market_data.collect_raw(paths, "news", start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2025-02-02", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv", max_workers=1).output_rows.tolist() == [0, 0]
    assert fetch.call_count == 1


def test_news_workers_process_dates_and_sum_in_manifest_order(tmp_path, monkeypatch):
    from src.preprocessing import alternative_data

    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    configure_tick_collection(monkeypatch, ["B", "A"])
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
    result = market_data.collect_raw(paths, "news", max_workers=2, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2025-02-03", tz="UTC"), manifest_path=tmp_path / "data/preprocessing/universe/sp500_2025.csv")
    assert len(calls) == 2
    assert all(symbols == ["B", "A"] for symbols, unused in calls)
    assert result.symbol.tolist() == ["B", "A"]
    assert result.output_rows.tolist() == [0, 2]


def test_feature_reuse_requires_current_settings_metadata(tmp_path):
    path = tmp_path / "feature.parquet"
    identity = {"settings": {"threshold": 100}}
    market_data.save_feature(pd.DataFrame({"value": [1]}), path, identity)
    assert market_data.reusable_feature(path, identity)
    changed = {**identity, "settings": {"threshold": 200}}
    with pytest.raises(ValueError, match="settings changed"):
        market_data.reusable_feature(path, changed)
    metadata = path.with_suffix(".json")
    legacy = {"computation_settings": identity["settings"]}
    metadata.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="settings unavailable"):
        market_data.reusable_feature(path, identity)
    assert json.loads(metadata.read_text()) == legacy


@pytest.fixture
def paths(tmp_path):
    return market_data.ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
                                     end=pd.Timestamp("2025-04-01", tz="UTC"))


def test_collect_split_pages_use_ex_date_and_cache_empty_snapshots(paths, monkeypatch):
    opens = pd.to_datetime(["2025-02-03T14:30Z", "2025-03-10T13:30Z"], utc=True)
    schedule = pd.DataFrame({"open": opens})
    requests, clients = [], []
    record = {"id": "split-1", "symbol": "A", "ex_date": "2025-03-10",
              "process_date": "2025-04-15", "old_rate": 1, "new_rate": 10}

    class Client:
        def __init__(self, **kwargs):
            self._session = SimpleNamespace(close=Mock())
            clients.append(self)

        def get(self, path, data, **kwargs):
            requests.append((path, data.copy(), kwargs))
            if data["symbols"] == "A" and "page_token" not in data:
                return {"corporate_actions": {"forward_splits": [record]}, "next_page_token": "p2"}
            return {"corporate_actions": {}, "next_page_token": None}

    monkeypatch.setattr(market_data, "_get_alpaca_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", Client)
    kwargs = dict(start=paths.start, end=paths.end, schedule=schedule)
    market_data.collect_split_records(paths, ["A", "SPY"], **kwargs)
    events = market_data.load_split_events(paths, "A", start=paths.start, end=paths.end)
    assert events == [{"effective_at": str(opens[1]), "ratio": 10.}]
    assert requests[0][0] == "/corporate-actions"
    assert requests[0][2] == {"api_version": "v1"}
    assert requests[0][1]["start"] == "1900-01-01"
    assert requests[0][1]["types"] == "forward_split,reverse_split"
    assert requests[1][1]["page_token"] == "p2"
    assert market_data.load_split_events(paths, "SPY", start=paths.start, end=paths.end) == []
    assert json.loads(market_data.split_path(paths, "A").read_text())["records"] == [record]
    assert clients[0]._session.close.call_count == 1
    market_data.collect_split_records(paths, ["A", "SPY"], **kwargs)
    assert len(requests) == 3
    with pytest.raises(ValueError, match="snapshot mismatch"):
        market_data.load_split_events(paths, "A", start=paths.start - pd.Timedelta(days=1), end=paths.end)


def test_missing_and_failed_split_collection_never_mean_no_splits(paths, monkeypatch):
    with pytest.raises(FileNotFoundError, match="Missing split snapshot"):
        market_data.load_split_events(paths, "A", start=paths.start, end=paths.end)
    client = SimpleNamespace(get=Mock(side_effect=RuntimeError("permission denied")),
                             _session=SimpleNamespace(close=Mock()))
    monkeypatch.setattr(market_data, "_get_alpaca_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(market_data, "StockHistoricalDataClient", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="permission denied"):
        market_data.collect_split_records(paths, ["A"], start=paths.start, end=paths.end,
                                            schedule=pd.DataFrame({"open": []}))
    assert not market_data.split_path(paths, "A").exists()
    client._session.close.assert_called_once()


@pytest.mark.parametrize("change,error", [
    ({"ex_date": "2025-02-02"}, "covered session"),
    ({"old_rate": 0}, "Invalid split ratio"),
    ({"new_rate": float("nan")}, "Invalid split ratio"),
    ({"symbol": "B"}, "Unexpected split symbol"),
])
def test_split_records_fail_on_bad_data(paths, change, error):
    record = {"symbol": "A", "ex_date": "2025-02-03", "old_rate": 1, "new_rate": 2, **change}
    schedule = pd.DataFrame({"open": pd.to_datetime(["2025-02-03T14:30Z"], utc=True)})
    with pytest.raises(ValueError, match=error):
        market_data._split_events([record], "A", schedule, paths.start, paths.end)


def test_duplicate_split_and_out_of_coverage_events(paths):
    record = {"symbol": "A", "ex_date": "2025-02-03", "old_rate": 1, "new_rate": 2}
    schedule = pd.DataFrame({"open": pd.to_datetime(["2025-02-03T14:30Z"], utc=True)})
    with pytest.raises(ValueError, match="Duplicate split ex-date"):
        market_data._split_events([record, record], "A", schedule, paths.start, paths.end)
    record["ex_date"] = "2025-04-02"
    assert market_data._split_events([record], "A", schedule, paths.start, paths.end) == []


def test_tick_collection_uses_session_rows_and_clips_request_bounds(tmp_path, monkeypatch):
    configure_tick_collection(monkeypatch, ["A"])
    configure_raw_tick_client(monkeypatch, [])
    schedule = pd.DataFrame({
        "session": ["2025-01-02", "2025-01-03", "2025-01-06"],
        "open": pd.to_datetime(["2025-01-02T14:30Z", "2025-01-03T14:30Z", "2025-01-06T14:30Z"]),
        "close": pd.to_datetime(["2025-01-02T21:00Z", "2025-01-03T18:00Z", "2025-01-06T21:00Z"]),
    })
    monkeypatch.setattr(market_data, "sessions", lambda *args, **kwargs: schedule)
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-01-02", tz="UTC"),
                          end=pd.Timestamp("2025-01-07", tz="UTC"))
    start = schedule.open.iloc[0] + pd.Timedelta(minutes=15)
    end = schedule.close.iloc[-1] - pd.Timedelta(hours=1)
    requests = []

    def write_partition(client, symbol, request_start, request_end, path):
        requests.append((symbol, pd.Timestamp(request_start), pd.Timestamp(request_end), path.stem))
        market_data.save_frame(pd.DataFrame({"symbol": [symbol]}), path)
        return 1

    monkeypatch.setattr(market_data, "_write_tick_partition", write_partition)
    report = market_data.collect_raw(paths, "tick", start=start, end=end,
                                    manifest_path=tmp_path / "universe.csv", max_workers=1)

    expected = [
        (start, schedule.close.iloc[0], "2025-01-02"),
        (schedule.open.iloc[1], schedule.close.iloc[1], "2025-01-03"),
        (schedule.open.iloc[2], end, "2025-01-06"),
    ]
    for symbol in ("A", "SPY"):
        assert [request[1:] for request in requests if request[0] == symbol] == expected
        for request_start, request_end, day in expected:
            metadata = json.loads((paths.raw(symbol, "tick") / f"{day}.json").read_text())
            assert metadata["start"] == str(request_start)
            assert metadata["end"] == str(request_end)
    assert report.symbol.tolist() == ["A", "SPY"]
    assert report.output_rows.tolist() == [3, 3]


@pytest.mark.parametrize("contents", [None, "{", '{"settings": {}}'])
def test_feature_metadata_requires_existing_readable_json(tmp_path, contents):
    path = tmp_path / "feature.parquet"
    if contents is not None:
        path.with_suffix(".json").write_text(contents)
    if contents is None:
        with pytest.raises(ValueError, match="Feature metadata unavailable"):
            market_data.require_feature_metadata(path)
    elif contents == "{":
        with pytest.raises(json.JSONDecodeError):
            market_data.require_feature_metadata(path)
    else:
        market_data.require_feature_metadata(path)


def test_feature_reuse_still_rejects_changed_input_files(tmp_path):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
                          end=pd.Timestamp("2026-01-01", tz="UTC"))
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(manifest, index=False)
    source = tmp_path / "source.parquet"
    output = tmp_path / "feature.parquet"
    market_data.save_frame(pd.DataFrame({"close": [100.]}), source)
    identity = market_data.feature_identity(paths, [source], manifest_path=manifest,
                                            settings={"window": 14})
    market_data.save_feature(pd.DataFrame({"feature": [1.]}), output, identity)
    assert market_data.reusable_feature(output, identity)
    market_data.save_frame(pd.DataFrame({"close": [100., 101.]}), source)
    changed = market_data.feature_identity(paths, [source], manifest_path=manifest,
                                           settings={"window": 14})
    with pytest.raises(ValueError, match="inputs/settings changed"):
        market_data.reusable_feature(output, changed)
