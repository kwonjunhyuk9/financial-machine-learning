from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import Sort, StockTradesRequest

VERSION = "sp500-fixed-2025-v3"
_TRADE_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("symbol", pa.large_string()),
    ("price", pa.float64()),
    ("size", pa.float64()),
])


@dataclass(frozen=True)
class ResearchPaths:
    """Project paths with an explicitly supplied event-file period."""

    root: Path
    period: str = field(kw_only=True)
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
        directory = self.root / "data/modeling"
        return directory / self.model_kind if self.model_kind else directory

    def event(self, stage: str) -> Path:
        name = "event_candidates" if stage == "candidates" else f"{stage}_events"
        return self.data / "events" / f"sp500_{name}_{self.period}.parquet"

    @property
    def news_source(self) -> Path:
        """Daily batch news responses for the stock universe."""
        return self.data / "alternative/stock/raw/news"

    def _symbol_directory(self, parent: str, symbol: str, asset_class: str | None) -> Path:
        asset_class = ("etf" if symbol == "SPY" else "stock") if asset_class is None else asset_class
        if asset_class not in ("stock", "etf", "crypto"):
            raise ValueError("asset_class must be stock, etf, or crypto")
        return self.data / parent / asset_class / symbol

    def feature(self, symbol: str, name: str, *, asset_class: str | None = None) -> Path:
        """Feature path; SPY defaults to ETF and other symbols to stock."""
        kind = "alternative" if name == "sentiment_scores" else "market"
        return self._symbol_directory(kind, symbol, asset_class) / "features" / f"{name}.parquet"

    def raw(self, symbol: str, kind: str, *, asset_class: str | None = None) -> Path:
        """Raw path; explicitly select asset_class for other ETFs or crypto."""
        parent = "alternative" if kind == "news" else "market"
        return self._symbol_directory(parent, symbol, asset_class) / "raw" / kind


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


def sessions(paths: ResearchPaths, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load or cache the explicitly requested exchange calendar."""
    if start >= end:
        raise ValueError("Calendar start must precede end")
    destination = paths.universe / "sessions.parquet"
    metadata = destination.with_suffix(".json")
    coverage = json.loads(metadata.read_text()) if metadata.exists() else None
    if (destination.exists() and coverage is not None
            and pd.Timestamp(coverage["start"]) <= start
            and pd.Timestamp(coverage["end"]) >= end):
        cached = pd.read_parquet(destination)
        cached["open"] = pd.to_datetime(cached["open"], utc=True)
        cached["close"] = pd.to_datetime(cached["close"], utc=True)
        return cached.loc[
            cached["close"].ge(start) & cached["open"].lt(end)
        ].reset_index(drop=True)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    key, secret = _get_credentials()
    calendar = TradingClient(key, secret).get_calendar(
        GetCalendarRequest(start=start.date(), end=end.date())
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
        result["close"].ge(start) & result["open"].lt(end)
    ].reset_index(drop=True)
    save_frame(result, destination)
    metadata.write_text(json.dumps({"start": str(start), "end": str(end)}, indent=2))
    return result


def load_manifest(manifest_path: Path, *, expected_securities: int) -> pd.DataFrame:
    manifest = pd.read_csv(
        manifest_path,
        dtype="string",
        keep_default_na=False,
        skip_blank_lines=False,
    )
    if manifest.columns.tolist() != ["symbol"]:
        raise ValueError("The fixed universe CSV must contain only the symbol column")
    if len(manifest) != expected_securities or manifest.symbol.nunique() != expected_securities:
        raise ValueError(f"The fixed universe CSV must contain {expected_securities} distinct symbols")
    if manifest.symbol.isna().any() or manifest.symbol.eq("").any():
        raise ValueError("Universe symbols must be complete")
    return manifest


def manifest_hash(manifest_path: Path) -> str:
    return sha256(manifest_path.read_bytes()).hexdigest()


NEWS_RAW_VERSION = "news-batch-v1"
NEWS_FILTER_VERSION = "single-symbol-benzinga-v1"


def _collect_news(paths: ResearchPaths, symbols: list[str], max_workers: int, *,
                  start: pd.Timestamp, end: pd.Timestamp, manifest_path: Path,
                  show_progress: bool = False) -> pd.DataFrame:
    """Save daily batch responses before producing symbol partitions."""
    from src.preprocessing.alternative_data import fetch_alpaca_news, filter_symbol_news

    identity = manifest_hash(manifest_path)
    days = pd.date_range(start, end, inclusive="left", freq="D")

    def collect_day(start: pd.Timestamp) -> list[dict[str, int]]:
        request_end = min(start + pd.Timedelta(days=1), end)
        source = paths.news_source / f"{start.date()}.parquet"
        record = source.with_suffix(".json")
        expected = {
            "version": NEWS_RAW_VERSION, "universe": identity, "kind": "news",
            "feed": "benzinga", "start": str(start), "end": str(request_end),
            "request_symbols": symbols,
        }
        if source.exists() and record.exists():
            saved = json.loads(record.read_text())
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Cache metadata mismatch: {source}")
            news = pd.read_parquet(source)
        else:
            news = fetch_alpaca_news(
                symbols=symbols, start=start.to_pydatetime(), end=request_end.to_pydatetime()
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
                "feed": "benzinga", "start": str(start), "end": str(request_end),
                "request_symbol": symbol,
            }
            selection = {"filter_version": NEWS_FILTER_VERSION, "source": source_identity}
            if path.exists() and metadata.exists():
                saved = json.loads(metadata.read_text())
                if any(saved.get(key) != value for key, value in selected_expected.items()):
                    raise ValueError(f"Cache metadata mismatch: {path}")
                if all(saved.get(key) == value for key, value in selection.items()):
                    counts.append({"rows": saved["rows"], "cached_partitions": 1,
                                   "processed_partitions": 0})
                    continue
            frame = filter_symbol_news(news, symbol)
            created = pd.to_datetime(frame.created_at, utc=True)
            frame = frame.loc[created.ge(start) & created.lt(request_end)].copy()
            frame["symbol"] = symbol
            save_frame(frame, path)
            metadata.write_text(json.dumps(
                {**selected_expected, **selection, "rows": len(frame)}, indent=2
            ))
            counts.append({"rows": len(frame), "cached_partitions": 0,
                           "processed_partitions": 1})
        return counts

    totals = [{"symbol": symbol, "kind": "news", "rows": 0,
               "processed_partitions": 0, "cached_partitions": 0} for symbol in symbols]
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(collect_day, day): day for day in days}
    try:
        with tqdm(total=len(futures), desc="News download", disable=not show_progress) as progress:
            for future in as_completed(futures):
                counts = future.result()
                for total, count in zip(totals, counts):
                    for key, value in count.items():
                        total[key] += value
                progress.set_postfix_str(str(futures[future].date()))
                progress.update(1)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return pd.DataFrame(totals)


def collect_raw(
    paths: ResearchPaths,
    kind: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    manifest_path: Path,
    expected_securities: int,
    max_workers: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Collect daily partitions with bounded parallelism.

    Tick tasks reuse one client per symbol and publish streamed daily files only
    after every page succeeds. Matching completed partitions are reused.

    Args:
        paths: Project data paths.
        kind: Either ``"tick"`` or ``"news"``.
        start: Inclusive collection start in UTC.
        end: Exclusive collection end in UTC.
        manifest_path: Notebook-selected universe CSV.
        expected_securities: Required number of distinct universe symbols.
        max_workers: Concurrent symbols for ticks or concurrent days for news.
        show_progress: Show one overall progress bar for completed work units.
    Returns:
        Row counts in manifest order, with SPY last for tick data.
    """
    if kind not in {"tick", "news"}:
        raise ValueError("kind must be either 'tick' or 'news'")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if start >= end:
        raise ValueError("Collection start must precede end")
    manifest = load_manifest(manifest_path, expected_securities=expected_securities)
    if kind == "news":
        return _collect_news(paths, list(manifest.symbol), max_workers,
                             start=start, end=end, manifest_path=manifest_path,
                             show_progress=show_progress)
    schedule = sessions(paths, start=start, end=end)
    symbols = list(manifest.symbol) + ["SPY"]
    identity = manifest_hash(manifest_path)
    days = schedule["open"]

    def collect_symbol(symbol: str) -> dict[str, object]:
        count = 0
        processed_partitions = cached_partitions = 0
        client = None
        try:
            for stamp in days:
                day = pd.Timestamp(stamp).normalize()
                session = schedule.loc[schedule.open.dt.normalize().eq(day)].iloc[0]
                request_start, request_end = max(session.open, start), min(session.close, end)
                path = paths.raw(symbol, kind) / f"{day.date()}.parquet"
                record = path.with_suffix(".json")
                expected = {
                    "version": VERSION, "universe": identity, "kind": kind,
                    "feed": "sip",
                    "start": str(request_start), "end": str(request_end), "request_symbol": symbol,
                }
                if path.exists() and record.exists():
                    saved = json.loads(record.read_text())
                    if any(saved.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"Cache metadata mismatch: {path}")
                    count += saved["rows"]
                    cached_partitions += 1
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
                        client, symbol, request_start.to_pydatetime(), request_end.to_pydatetime(), path,
                    )
                    temporary_record.write_text(json.dumps({**expected, "rows": rows}, indent=2))
                    temporary_record.replace(record)
                except BaseException:
                    temporary_record.unlink(missing_ok=True)
                    raise
                count += rows
                processed_partitions += 1
        finally:
            if client is not None:
                client._session.close()
        return {"symbol": symbol, "kind": kind, "rows": count,
                "processed_partitions": processed_partitions, "cached_partitions": cached_partitions}

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(collect_symbol, symbol): position
        for position, symbol in enumerate(symbols)
    }
    counts: list[dict[str, object] | None] = [None] * len(symbols)
    try:
        with tqdm(total=len(futures), desc="Trade download", disable=not show_progress) as progress:
            for future in as_completed(futures):
                counts[futures[future]] = future.result()
                progress.set_postfix_str(symbols[futures[future]])
                progress.update(1)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return pd.DataFrame(counts)


def raw_partitions(paths: ResearchPaths, symbol: str, kind: str, *,
                   start: pd.Timestamp, end: pd.Timestamp,
                   asset_class: str | None = None) -> list[Path]:
    """Return raw partitions in the explicitly requested interval."""
    return [
        path
        for path in sorted(paths.raw(symbol, kind, asset_class=asset_class).glob("*.parquet"))
        if pd.Timestamp(path.stem, tz="UTC") < end
        and pd.Timestamp(path.stem, tz="UTC") + pd.Timedelta(days=1) > start
    ]


def read_raw(paths: ResearchPaths, symbol: str, kind: str, *,
             start: pd.Timestamp, end: pd.Timestamp,
             asset_class: str | None = None) -> pd.DataFrame:
    files = raw_partitions(paths, symbol, kind, start=start, end=end, asset_class=asset_class)
    if not files:
        raise FileNotFoundError(f"No completed {kind} partitions for {symbol}")
    for file in files:
        if not file.with_suffix(".json").exists():
            raise ValueError(f"Incomplete partition: {file}")
    result = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    timestamp = "created_at" if kind == "news" else "timestamp"
    times = pd.to_datetime(result[timestamp], utc=True)
    return result.loc[times.ge(start) & times.lt(end)].reset_index(drop=True)


def feature_identity(paths: ResearchPaths, dependencies: list[Path], *,
                     manifest_path: Path, settings: dict) -> dict:
    """Record inputs and explicit computation settings for cache compatibility."""
    inputs = {}
    for source in dependencies:
        if not source.exists():
            raise FileNotFoundError(source)
        stat = source.stat()
        inputs[str(source.relative_to(paths.root))] = [stat.st_size, stat.st_mtime_ns]
    return {"version": VERSION, "universe": manifest_hash(manifest_path), "inputs": inputs,
            "settings": json.loads(json.dumps(settings, default=str))}


def reusable_feature(path: Path, identity: dict) -> bool:
    if not path.exists():
        return False
    metadata = path.with_suffix(".json")
    if not metadata.exists():
        raise ValueError(f"Feature settings unavailable: {path}; deliberately rebuild this stage")
    saved = json.loads(metadata.read_text())
    if "settings" not in saved:
        raise ValueError(f"Feature settings unavailable: {path}; deliberately rebuild this stage")
    if saved != identity:
        raise ValueError(f"Feature inputs/version/settings changed: {path}; deliberately rebuild this stage")
    return True


def save_feature(frame: pd.DataFrame, path: Path, identity: dict) -> None:
    save_frame(frame, path)
    path.with_suffix(".json").write_text(json.dumps(identity, indent=2))
