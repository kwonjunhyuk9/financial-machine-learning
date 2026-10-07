from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import Sort, StockTradesRequest

VERSION = "sp500-fixed-2025-v3"
EXPECTED_SECURITIES = 503
PERIOD = "2025-02-01_2025-12-31"
DATA_START = pd.Timestamp("2025-01-01", tz="UTC")
RESEARCH_START = pd.Timestamp("2025-02-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
_TRADE_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("symbol", pa.large_string()),
    ("price", pa.float64()),
    ("size", pa.float64()),
])


@dataclass(frozen=True)
class ResearchPaths:
    root: Path
    model_kind: str | None = None

    def __post_init__(self):
        if self.model_kind not in (None, "market", "sentiment"):
            raise ValueError("model_kind must be market or sentiment")

    @property
    def data(self) -> Path:
        return self.root / "data/preprocessing"

    @property
    def universe(self) -> Path:
        return self.data / "universe"

    @property
    def artifacts(self) -> Path:
        directory = self.root / "data/model_artifact"
        return directory / self.model_kind if self.model_kind else directory

    def event(self, stage: str) -> Path:
        name = "event_candidates" if stage == "candidates" else f"{stage}_events"
        return self.data / "events" / f"sp500_{name}_{PERIOD}.parquet"

    def feature(self, symbol: str, name: str) -> Path:
        kind = "alternative" if name == "sentiment_scores" else "market"
        return self.data / kind / symbol / "features" / f"{name}.parquet"

    def raw(self, symbol: str, kind: str) -> Path:
        parent = "alternative" if kind == "news" else "market"
        if symbol == "SPY":
            return self.data / "benchmark/SPY/raw" / kind
        return self.data / parent / symbol / "raw" / kind


def _get_credentials() -> tuple[str, str]:
    """Read Alpaca market data credentials from the environment."""
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "Data requires ALPACA_API_KEY/ALPACA_SECRET_KEY or "
            "APCA_API_KEY_ID/APCA_API_SECRET_KEY."
        )
    return api_key, secret_key


def _normalize_trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize an SDK trade DataFrame into the project's tabular schema."""
    frame = trades.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names="timestamp")
    else:
        frame = frame.reset_index(drop=False)
    preferred = ["timestamp", "symbol", "price", "size"]
    if frame.empty:
        frame = frame.reindex(columns=preferred)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame
    if "timestamp" not in frame.columns:
        raise ValueError(
            "Trade data must include a timestamp column after normalization. "
            f"Columns: {frame.columns.tolist()}"
        )
    if "symbol" not in frame.columns:
        raise ValueError("Trade data must include a symbol column after normalization.")
    if "price" not in frame.columns:
        raise ValueError("Trade data must include a price column after normalization.")
    if "size" not in frame.columns:
        raise ValueError("Trade data must include a size column after normalization.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["price"] = frame["price"].astype(float)
    frame["size"] = frame["size"].astype(float)
    return frame.loc[:, preferred].sort_values(
        ["timestamp", "symbol"], kind="stable"
    ).reset_index(drop=True)


def _normalize_raw_trade_frame(
    trades: dict[str, list[dict]] | pd.DataFrame,
) -> pd.DataFrame:
    """Normalize raw trades while preserving the SDK's microsecond precision."""
    if hasattr(trades, "df"):
        return _normalize_trade_frame(trades.df)
    if isinstance(trades, pd.DataFrame):
        return _normalize_trade_frame(trades)
    frame = pd.DataFrame(
        [(trade["t"], symbol, trade["p"], trade["s"])
         for symbol, records in trades.items() for trade in records],
        columns=_TRADE_SCHEMA.names,
    )
    frame["timestamp"] = (
        pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
        .dt.floor("us").astype("datetime64[us, UTC]")
    )
    frame["symbol"] = frame["symbol"].astype(str)
    frame["price"] = frame["price"].astype(float)
    frame["size"] = frame["size"].astype(float)
    return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def fetch_alpaca_historical_data(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch SIP stock trades from Alpaca.

    Args:
        symbols: Symbols to request.
        start: Inclusive request start time.
        end: Exclusive result end time.
    Returns:
        A normalized trade DataFrame.
    """
    api_key, secret_key = _get_credentials()
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key, raw_data=True)
    request = StockTradesRequest(
        symbol_or_symbols=list(symbols),
        start=start,
        end=end,
        feed=DataFeed.SIP,
        sort=Sort.ASC,
    )
    try:
        market_data = _normalize_raw_trade_frame(client.get_stock_trades(request))
    finally:
        client._session.close()

    end_timestamp = pd.to_datetime(end, utc=True)
    return market_data.loc[market_data["timestamp"] < end_timestamp].reset_index(drop=True)


def _write_tick_partition(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    path: Path,
) -> int:
    """Stream ordered SIP pages into one atomically published daily partition."""
    params = StockTradesRequest(
        symbol_or_symbols=[symbol], start=start, end=end, feed=DataFeed.SIP, sort=Sort.ASC,
    ).to_request_fields()
    params["limit"] = 10_000
    end_timestamp = pd.to_datetime(end, utc=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.parquet")
    count = 0
    previous = None
    try:
        with pq.ParquetWriter(temporary, _TRADE_SCHEMA, compression="snappy") as writer:
            while True:
                page = client.get("/stocks/trades", data=params)
                frame = _normalize_raw_trade_frame(page.get("trades", {}))
                frame = frame.loc[frame["timestamp"] < end_timestamp]
                if not frame.empty:
                    if previous is not None and frame["timestamp"].iloc[0] < previous:
                        raise ValueError(f"Trade pages are not ordered: {symbol}")
                    previous = frame["timestamp"].iloc[-1]
                writer.write_table(pa.Table.from_pandas(
                    frame, schema=_TRADE_SCHEMA, preserve_index=False,
                ))
                count += len(frame)
                token = page.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    """Atomically publish a complete parquet table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sessions(paths: ResearchPaths) -> pd.DataFrame:
    """Load or cache the 2025 exchange calendar, including early closes."""
    destination = paths.universe / "sessions.parquet"
    if destination.exists():
        cached = pd.read_parquet(destination)
        cached["open"] = pd.to_datetime(cached["open"], utc=True)
        cached["close"] = pd.to_datetime(cached["close"], utc=True)
        return cached.loc[
            cached["open"].ge(DATA_START) & cached["open"].lt(END)
        ].reset_index(drop=True)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    key, secret = _get_credentials()
    calendar = TradingClient(key, secret).get_calendar(
        GetCalendarRequest(start=DATA_START.date(), end=END.date())
    )
    rows = []
    for day in calendar:
        def utc(value):
            stamp = pd.Timestamp(value)
            localized = stamp.tz_localize("America/New_York") if stamp.tzinfo is None else stamp
            return localized.tz_convert("UTC")

        rows.append({"session": str(day.date), "open": utc(day.open), "close": utc(day.close)})
    result = pd.DataFrame(rows).sort_values("open").reset_index(drop=True)
    result = result.loc[
        result["open"].ge(DATA_START) & result["open"].lt(END)
    ].reset_index(drop=True)
    save_frame(result, destination)
    return result


def load_manifest(paths: ResearchPaths) -> pd.DataFrame:
    manifest = pd.read_csv(
        paths.universe / "sp500_2025.csv",
        dtype="string",
        keep_default_na=False,
        skip_blank_lines=False,
    )
    if manifest.columns.tolist() != ["symbol"]:
        raise ValueError("The fixed universe CSV must contain only the symbol column")
    if len(manifest) != EXPECTED_SECURITIES or manifest.symbol.nunique() != EXPECTED_SECURITIES:
        raise ValueError(f"The fixed universe CSV must contain {EXPECTED_SECURITIES} distinct symbols")
    if manifest.symbol.isna().any() or manifest.symbol.eq("").any():
        raise ValueError("Universe symbols must be complete")
    return manifest


def manifest_hash(paths: ResearchPaths) -> str:
    return sha256((paths.universe / "sp500_2025.csv").read_bytes()).hexdigest()


NEWS_RAW_VERSION = "news-batch-v1"
NEWS_FILTER_VERSION = "single-symbol-benzinga-v1"


def _collect_news(paths: ResearchPaths, symbols: list[str], max_workers: int) -> pd.DataFrame:
    """Save daily batch responses before producing symbol partitions."""
    from src.preprocessing.alternative_data import fetch_alpaca_news, filter_symbol_news

    identity = manifest_hash(paths)
    days = pd.date_range(RESEARCH_START, END, inclusive="left", freq="D")

    def collect_day(start: pd.Timestamp) -> list[int]:
        end = start + pd.Timedelta(days=1)
        source = paths.data / "alternative/raw/news" / f"{start.date()}.parquet"
        record = source.with_suffix(".json")
        expected = {
            "version": NEWS_RAW_VERSION, "universe": identity, "kind": "news",
            "feed": "benzinga", "start": str(start), "end": str(end),
            "request_symbols": symbols,
        }
        if source.exists() and record.exists():
            saved = json.loads(record.read_text())
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Cache metadata mismatch: {source}")
            news = pd.read_parquet(source)
        else:
            news = fetch_alpaca_news(
                symbols=symbols, start=start.to_pydatetime(), end=end.to_pydatetime()
            )
            save_frame(news, source)
            record.write_text(json.dumps({**expected, "rows": len(news)}, indent=2))

        stat = source.stat()
        source_identity = [stat.st_size, stat.st_mtime_ns]
        counts = []
        for symbol in symbols:
            path = paths.raw(symbol, "news") / source.name
            metadata = path.with_suffix(".json")
            selected_expected = {
                "version": VERSION, "universe": identity, "kind": "news",
                "feed": "benzinga", "start": str(start), "end": str(end),
                "request_symbol": symbol,
            }
            selection = {"filter_version": NEWS_FILTER_VERSION, "source": source_identity}
            if path.exists() and metadata.exists():
                saved = json.loads(metadata.read_text())
                if any(saved.get(key) != value for key, value in selected_expected.items()):
                    raise ValueError(f"Cache metadata mismatch: {path}")
                if all(saved.get(key) == value for key, value in selection.items()):
                    counts.append(saved["rows"])
                    continue
            frame = filter_symbol_news(news, symbol)
            created = pd.to_datetime(frame.created_at, utc=True)
            frame = frame.loc[created.ge(start) & created.lt(end)].copy()
            frame["symbol"] = symbol
            save_frame(frame, path)
            metadata.write_text(json.dumps(
                {**selected_expected, **selection, "rows": len(frame)}, indent=2
            ))
            counts.append(len(frame))
        return counts

    totals = [0] * len(symbols)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(collect_day, day) for day in days]
    try:
        for future in as_completed(futures):
            totals = [total + count for total, count in zip(totals, future.result())]
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return pd.DataFrame({"symbol": symbols, "kind": "news", "rows": totals})


def collect_raw(
    paths: ResearchPaths,
    kind: str,
    *,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Collect daily partitions with bounded parallelism.

    Tick tasks reuse one client per symbol and publish streamed daily files only
    after every page succeeds. Matching completed partitions are reused.

    Args:
        paths: Project data paths.
        kind: Either ``"tick"`` or ``"news"``.
        max_workers: Concurrent symbols for ticks or concurrent days for news.
    Returns:
        Row counts in manifest order, with SPY last for tick data.
    """
    if kind not in {"tick", "news"}:
        raise ValueError("kind must be either 'tick' or 'news'")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    manifest = load_manifest(paths)
    if kind == "news":
        return _collect_news(paths, list(manifest.symbol), max_workers)
    schedule = sessions(paths)
    symbols = list(manifest.symbol) + ["SPY"]
    identity = manifest_hash(paths)
    days = schedule.loc[
        schedule.open.ge(DATA_START) & schedule.open.lt(END), "open"
    ]

    def collect_symbol(symbol: str) -> dict[str, object]:
        count = 0
        client = None
        try:
            for stamp in days:
                day = pd.Timestamp(stamp).normalize()
                session = schedule.loc[schedule.open.dt.normalize().eq(day)].iloc[0]
                start, end = session.open, session.close
                path = paths.raw(symbol, kind) / f"{day.date()}.parquet"
                record = path.with_suffix(".json")
                expected = {
                    "version": VERSION, "universe": identity, "kind": kind,
                    "feed": "sip",
                    "start": str(start), "end": str(end), "request_symbol": symbol,
                }
                if path.exists() and record.exists():
                    saved = json.loads(record.read_text())
                    if any(saved.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"Cache metadata mismatch: {path}")
                    count += saved["rows"]
                    continue
                # An orphaned marker must not certify a replacement after failure.
                record.unlink(missing_ok=True)
                if client is None:
                    api_key, secret_key = _get_credentials()
                    client = StockHistoricalDataClient(
                        api_key=api_key, secret_key=secret_key, raw_data=True,
                    )
                temporary_record = record.with_suffix(".pending.json")
                try:
                    rows = _write_tick_partition(
                        client, symbol, start.to_pydatetime(), end.to_pydatetime(), path,
                    )
                    temporary_record.write_text(json.dumps({**expected, "rows": rows}, indent=2))
                    temporary_record.replace(record)
                except BaseException:
                    temporary_record.unlink(missing_ok=True)
                    raise
                count += rows
        finally:
            if client is not None:
                client._session.close()
        return {"symbol": symbol, "kind": kind, "rows": count}

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(collect_symbol, symbol): position
        for position, symbol in enumerate(symbols)
    }
    counts: list[dict[str, object] | None] = [None] * len(symbols)
    try:
        for future in as_completed(futures):
            counts[futures[future]] = future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return pd.DataFrame(counts)


def raw_partitions(paths: ResearchPaths, symbol: str, kind: str) -> list[Path]:
    """Return raw partitions inside the configured data or research period."""
    lower_bound = RESEARCH_START if kind == "news" else DATA_START
    return [
        path
        for path in sorted(paths.raw(symbol, kind).glob("*.parquet"))
        if lower_bound.date() <= pd.Timestamp(path.stem).date() < END.date()
    ]


def read_raw(paths: ResearchPaths, symbol: str, kind: str) -> pd.DataFrame:
    files = raw_partitions(paths, symbol, kind)
    if not files:
        raise FileNotFoundError(f"No completed {kind} partitions for {symbol}")
    for file in files:
        if not file.with_suffix(".json").exists():
            raise ValueError(f"Incomplete partition: {file}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def feature_identity(paths: ResearchPaths, dependencies: list[Path]) -> dict:
    """Record inputs so incompatible features cannot be silently reused."""
    inputs = {}
    for source in dependencies:
        if not source.exists():
            raise FileNotFoundError(source)
        stat = source.stat()
        inputs[str(source.relative_to(paths.root))] = [stat.st_size, stat.st_mtime_ns]
    return {"version": VERSION, "universe": manifest_hash(paths), "inputs": inputs}


def reusable_feature(path: Path, identity: dict) -> bool:
    if not path.exists():
        return False
    metadata = path.with_suffix(".json")
    if not metadata.exists() or json.loads(metadata.read_text()) != identity:
        raise ValueError(f"Feature inputs/version changed: {path}; deliberately rebuild this stage")
    return True


def save_feature(frame: pd.DataFrame, path: Path, identity: dict) -> None:
    save_frame(frame, path)
    path.with_suffix(".json").write_text(json.dumps(identity, indent=2))
