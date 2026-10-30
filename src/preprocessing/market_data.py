from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from math import isfinite

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import Sort, StockTradesRequest

from src.preprocessing import _get_alpaca_credentials

_TRADE_CONDITIONS = {
    "A": set(" BCEFHIKLMNOPQRTUVXZ45679"),
    "B": set(" BCEFHIKLMNOPQRTUVXZ45679"),
    "C": set("@ABCDFGHIKLMNOPQRTUVWXYZ45679"),
}
_TRADE_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("symbol", pa.large_string()),
    ("conditions", pa.list_(pa.string())),
    ("exchange", pa.large_string()),
    ("trade_id", pa.int64()),
    ("tape", pa.large_string()),
    ("price", pa.float64()),
    ("size", pa.float64()),
])


@dataclass(frozen=True)
class ResearchPaths:
    """Locate research data and model artifacts."""

    root: Path
    start: pd.Timestamp = field(kw_only=True)
    end: pd.Timestamp = field(kw_only=True)
    model_kind: str | None = None

    def __post_init__(self):
        """Validate the model kind and research interval after initialization.

        Raises:
            ValueError: The model kind is unsupported, a boundary is not a UTC
                midnight Timestamp, or the interval is not strictly increasing.
        """
        if self.model_kind not in (None, "market", "sentiment"):
            raise ValueError("model_kind must be market or sentiment")
        for name, boundary in (("start", self.start), ("end", self.end)):
            if (
                not isinstance(boundary, pd.Timestamp)
                or pd.isna(boundary)
                or str(boundary.tz) != "UTC"
                or boundary != boundary.normalize()
            ):
                raise ValueError(f"{name} must be a UTC midnight pandas Timestamp")
        if self.start >= self.end:
            raise ValueError("start must be before end")

    @property
    def period(self) -> str:
        """Return the research period identifier.

        Returns:
            Research period identifier.
        """
        return f"{self.start:%Y-%m-%d}_{self.end - pd.Timedelta(days=1):%Y-%m-%d}"

    @property
    def data(self) -> Path:
        """Return the preprocessing data directory.

        Returns:
            Preprocessing data directory.
        """
        return self.root / "data/preprocessing"

    @property
    def universe(self) -> Path:
        """Return the universe data directory.

        Returns:
            Universe data directory.
        """
        return self.data / "universe"

    @property
    def artifacts(self) -> Path:
        """Return the modeling artifact directory.

        Returns:
            Modeling artifact directory.
        """
        directory = self.root / "data/modeling"
        return directory / self.model_kind if self.model_kind else directory

    def event(self, stage: str) -> Path:
        """Return the event artifact path for a stage.

        Args:
            stage: Preprocessing stage name.

        Returns:
            Event artifact path.
        """
        return self.data / "events" / f"sp500_events_{stage}_{self.period}.parquet"

    @property
    def news_source(self) -> Path:
        """Return the raw news response directory.

        Returns:
            Raw news response directory.
        """
        return self.data / "alternative/stock/raw/news"

    def _symbol_directory(self, parent: str, symbol: str, asset_class: str | None) -> Path:
        asset_class = ("etf" if symbol == "SPY" else "stock") if asset_class is None else asset_class
        if asset_class not in ("stock", "etf", "crypto"):
            raise ValueError("asset_class must be stock, etf, or crypto")
        return self.data / parent / asset_class / symbol

    def feature(self, symbol: str, name: str, *, asset_class: str | None = None) -> Path:
        """Return the feature artifact path for a symbol.

        Args:
            symbol: Ticker symbol.
            name: Feature name.
            asset_class: Asset class.

        Returns:
            Feature artifact path.
        """
        kind = "alternative" if name == "sentiment_scores" else "market"
        return self._symbol_directory(kind, symbol, asset_class) / "features" / f"{name}.parquet"

    def raw(self, symbol: str, kind: str, *, asset_class: str | None = None) -> Path:
        """Return the raw data directory for a symbol.

        Args:
            symbol: Ticker symbol.
            kind: Raw data category.
            asset_class: Asset class.

        Returns:
            Raw data directory.
        """
        parent = "alternative" if kind == "news" else "market"
        return self._symbol_directory(parent, symbol, asset_class) / "raw" / kind


def _is_eligible_trade(conditions: object, tape: object) -> bool:
    """Apply Alpaca volume and trade-count eligibility rules.

    Args:
        conditions: Nonempty sequence of trade condition codes, including arrays read from
            Parquet.
        tape: Supported equity tape A, B, or C.

    Returns:
        Whether every condition permits volume updates; Q, M, and 9 exclude the record.

    Raises:
        ValueError: Conditions or tape are missing or unsupported.
    """
    if not isinstance(tape, str) or tape not in _TRADE_CONDITIONS:
        raise ValueError(f"Unknown or missing trade tape: {tape!r}")
    if (not isinstance(conditions, (list, tuple, np.ndarray)) or len(conditions) == 0
            or any(not isinstance(code, str) or code not in _TRADE_CONDITIONS[tape]
                   for code in conditions)):
        raise ValueError(f"Unknown or missing trade conditions: {conditions!r}")
    return not any(code in {"Q", "M", "9"} for code in conditions)


def filter_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of trades eligible for Alpaca volume and trade counts.

    Args:
        trades: Raw trades with conditions and tape columns.

    Returns:
        Eligible records, retaining their order, index, and columns.

    Raises:
        ValueError: Required metadata is missing or unsupported.
    """
    missing = {"conditions", "tape"}.difference(trades.columns)
    if missing:
        raise ValueError(f"Trades are missing required columns: {sorted(missing)}")
    eligible = [_is_eligible_trade(conditions, tape)
                for conditions, tape in zip(trades["conditions"], trades["tape"])]
    return trades.loc[np.asarray(eligible, dtype=bool)].copy()


def _normalize_raw_trade_frame(
    trades: dict[str, list[dict]],
) -> pd.DataFrame:
    """Normalize response records without filtering their trade conditions.

    Args:
        trades: API records grouped by symbol, including original c, x, i, and z metadata when
            available.

    Returns:
        Trade rows sorted stably by timestamp and symbol, with UTC microsecond timestamps,
        nullable integer trade IDs, and unfiltered condition metadata.
    """
    frame = pd.DataFrame(
        [(trade["t"], symbol, trade.get("c"), trade.get("x"), trade.get("i"),
          trade.get("z"), trade["p"], trade["s"])
         for symbol, records in trades.items() for trade in records],
        columns=_TRADE_SCHEMA.names,
        dtype=object,
    )
    frame["timestamp"] = (
        pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
        .dt.floor("us").astype("datetime64[us, UTC]")
    )
    frame["symbol"] = frame["symbol"].astype(str)
    frame["trade_id"] = frame["trade_id"].astype("Int64")
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
    """Stream ordered SIP pages into an atomically published daily partition.

    Failed writes remove the pending file and preserve the previously completed destination.

    Args:
        client: Alpaca historical client returning raw response dictionaries.
        symbol: Requested ticker.
        start: Inclusive request start.
        end: Exclusive partition end.
        path: Destination Parquet file.

    Returns:
        Number of saved records, retaining all returned trade conditions before the end
        boundary.

    Raises:
        ValueError: Trade timestamps move backwards across pages.
    """
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
    """Atomically publish a complete parquet table.

    Args:
        frame: Input observations.
        path: Artifact path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sessions(paths: ResearchPaths, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load or cache the explicitly requested exchange calendar.

    Args:
        paths: Research data and artifact paths.
        start: Interval start timestamp.
        end: Interval end timestamp.

    Returns:
        Trading-session dates and market hours.
    """
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

    key, secret = _get_alpaca_credentials()
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


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the research universe manifest.

    Args:
        manifest_path: Universe manifest path.

    Returns:
        Research universe symbols.
    """
    manifest = pd.read_csv(
        manifest_path,
        dtype="string",
        keep_default_na=False,
        skip_blank_lines=False,
    )
    if manifest.columns.tolist() != ["symbol"]:
        raise ValueError("The fixed universe CSV must contain only the symbol column")
    if manifest.empty:
        raise ValueError("The fixed universe CSV must contain at least one symbol")
    if manifest.symbol.duplicated().any():
        raise ValueError("Universe symbols must be distinct")
    if manifest.symbol.isna().any() or manifest.symbol.eq("").any():
        raise ValueError("Universe symbols must be complete")
    return manifest


def manifest_hash(manifest_path: Path) -> str:
    """Compute the universe manifest hash.

    Args:
        manifest_path: Universe manifest path.

    Returns:
        SHA-256 digest of the universe manifest.
    """
    return sha256(manifest_path.read_bytes()).hexdigest()


def _collect_news(paths: ResearchPaths, symbols: list[str], max_workers: int, *,
                  start: pd.Timestamp, end: pd.Timestamp, manifest_path: Path,
                  show_progress: bool = False) -> pd.DataFrame:
    """Save daily batch responses before producing symbol partitions.

    Args:
        paths: Research paths for source responses and symbol partitions.
        symbols: Requested ticker symbols.
        max_workers: Maximum number of concurrent day requests.
        start: Inclusive UTC start.
        end: Exclusive UTC end.
        manifest_path: Universe manifest used to identify cached data.
        show_progress: Whether to display download progress.

    Returns:
        Per-symbol row counts and processed/cached partition counts.

    Raises:
        ValueError: Existing source cache metadata disagrees with the request.
    """
    from src.preprocessing.alternative_data import fetch_alpaca_news, filter_symbol_news

    identity = manifest_hash(manifest_path)
    days = pd.date_range(start, end, inclusive="left", freq="D")

    def collect_day(start: pd.Timestamp) -> list[dict[str, int]]:
        """Collect one day and derive symbol-specific news partitions.

        Args:
            start: Inclusive UTC day boundary; the request is capped at the outer collection end.

        Returns:
            Per-symbol row and cache counters in the requested symbol order.

        Raises:
            ValueError: Existing daily source metadata disagrees with the request.
        """
        request_end = min(start + pd.Timedelta(days=1), end)
        source = paths.news_source / f"{start.date()}.parquet"
        record = source.with_suffix(".json")
        expected = {
            "universe": identity, "kind": "news",
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
                "universe": identity, "kind": "news",
                "feed": "benzinga", "start": str(start), "end": str(request_end),
                "request_symbol": symbol,
            }
            selection = {"source": source_identity}
            if path.exists() and metadata.exists():
                saved = json.loads(metadata.read_text())
                if any(saved.get(key) != value for key, value in selected_expected.items()):
                    raise ValueError(f"Cache metadata mismatch: {path}")
                if all(saved.get(key) == value for key, value in selection.items()):
                    counts.append({"output_rows": saved["rows"], "processed_files": 0,
                                   "cached_files": 1})
                    continue
            frame = filter_symbol_news(news, symbol)
            created = pd.to_datetime(frame.created_at, utc=True)
            frame = frame.loc[created.ge(start) & created.lt(request_end)].copy()
            frame["symbol"] = symbol
            save_frame(frame, path)
            metadata.write_text(json.dumps(
                {**selected_expected, **selection, "rows": len(frame)}, indent=2
            ))
            counts.append({"output_rows": len(frame), "processed_files": 1,
                           "cached_files": 0})
        return counts

    totals = [{"symbol": symbol, "output_rows": 0,
               "processed_files": 0, "cached_files": 0} for symbol in symbols]
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


def split_path(paths, symbol: str) -> Path:
    """Return the split snapshot path for a symbol.

    Args:
        paths: Research data and artifact paths.
        symbol: Ticker symbol.

    Returns:
        Stock-split snapshot path.
    """
    return paths.universe / "splits" / f"{symbol}.json"


def _split_events(records: list[dict], symbol: str, schedule: pd.DataFrame,
                  start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Resolve split ex-dates to actual trading-session opens.

    Args:
        records: Corporate-action split records for the requested symbol.
        symbol: Expected ticker symbol.
        schedule: Trading-session calendar with timezone-aware open times.
        start: Inclusive UTC coverage start.
        end: Exclusive UTC coverage end.

    Returns:
        Chronological effective_at timestamps and new-to-old share ratios for covered ex-dates.

    Raises:
        ValueError: A symbol, session, ratio, or duplicate ex-date is invalid.
    """
    opens = {stamp.tz_convert("America/New_York").date(): stamp
             for stamp in pd.to_datetime(schedule["open"], utc=True)}
    events, seen = [], set()
    for record in records:
        if record["symbol"] != symbol:
            raise ValueError(f"Unexpected split symbol: {record['symbol']}")
        ex_date = pd.Timestamp(record["ex_date"]).date()
        day = pd.Timestamp(ex_date, tz="America/New_York").tz_convert("UTC")
        if not start.normalize() <= day < end:
            continue
        if ex_date not in opens:
            raise ValueError(f"Split ex-date is not a covered session: {symbol} {ex_date}")
        old, new = float(record["old_rate"]), float(record["new_rate"])
        if not all(isfinite(rate) and rate > 0 for rate in (old, new)):
            raise ValueError(f"Invalid split ratio: {symbol} {ex_date}")
        if ex_date in seen:
            raise ValueError(f"Duplicate split ex-date: {symbol} {ex_date}")
        seen.add(ex_date)
        ratio = new / old
        if not isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Invalid split ratio: {symbol} {ex_date}")
        events.append({"effective_at": str(opens[ex_date]), "ratio": ratio})
    return sorted(events, key=lambda event: pd.Timestamp(event["effective_at"]))


def collect_split_records(paths, symbols: list[str], *, start: pd.Timestamp,
                          end: pd.Timestamp, schedule: pd.DataFrame) -> None:
    """Collect stock-split snapshots for the research universe.

    Args:
        paths: Research data and artifact paths.
        symbols: Ticker symbols.
        start: Interval start timestamp.
        end: Interval end timestamp.
        schedule: Trading-session schedule.
    """
    # Include the warmup and reference date in one unit-conversion interval.
    coverage_start, coverage_end = min(start, paths.start), max(end, paths.start + pd.Timedelta(days=1))
    if coverage_start < start or coverage_end > end:
        schedule = sessions(paths, start=coverage_start, end=coverage_end)
    expected = {"start": str(coverage_start),
                "end": str(coverage_end), "reference": str(paths.start)}
    client = None
    try:
        for symbol in dict.fromkeys(symbols):
            path = split_path(paths, symbol)
            if path.exists():
                load_split_events(paths, symbol, start=start, end=end)
                continue
            if client is None:
                key, secret = _get_alpaca_credentials()
                client = StockHistoricalDataClient(api_key=key, secret_key=secret, raw_data=True)
            params = {"symbols": symbol, "types": "forward_split,reverse_split",
                      "start": "1900-01-01",
                      "end": str(max(pd.Timestamp.now(tz="UTC").date(), coverage_end.date())),
                      "limit": 1000, "sort": "asc"}
            records, tokens = [], set()
            while True:
                page = client.get("/corporate-actions", data=params, api_version="v1")
                actions = page["corporate_actions"]
                for kind in ("forward_splits", "reverse_splits"):
                    records.extend(actions.get(kind, []))
                token = page.get("next_page_token")
                if not token:
                    break
                if token in tokens:
                    raise ValueError("Repeated corporate-action pagination token")
                tokens.add(token)
                params["page_token"] = token
            events = _split_events(records, symbol, schedule, coverage_start, coverage_end)
            payload = {**expected, "symbol": symbol, "records": records, "events": events,
                       "fetched_at": str(pd.Timestamp.now(tz="UTC"))}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".pending.json")
            try:
                temporary.write_text(json.dumps(payload, indent=2))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        if client is not None:
            client._session.close()


def load_split_events(paths, symbol: str, *, start: pd.Timestamp,
                      end: pd.Timestamp) -> list[dict]:
    """Load collected stock-split events for a symbol.

    Args:
        paths: Research data and artifact paths.
        symbol: Ticker symbol.
        start: Interval start timestamp.
        end: Interval end timestamp.

    Returns:
        Stock-split effective timestamps and ratios.
    """
    path = split_path(paths, symbol)
    if not path.exists():
        raise FileNotFoundError(f"Missing split snapshot: {path}; run market-data collection first")
    saved = json.loads(path.read_text())
    if (saved.get("symbol") != symbol
            or saved.get("reference") != str(paths.start)
            or pd.Timestamp(saved["start"]) > min(start, paths.start)
            or pd.Timestamp(saved["end"]) < max(end, paths.start + pd.Timedelta(days=1))):
        raise ValueError(f"Split snapshot mismatch: {path}; deliberately refresh this snapshot")
    return saved["events"]


def collect_raw(
    paths: ResearchPaths,
    kind: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    manifest_path: Path,
    max_workers: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Collect daily partitions with bounded parallelism.

    Args:
        paths: Project data paths.
        kind: Either ``"tick"`` or ``"news"``.
        start: Inclusive collection start in UTC.
        end: Exclusive collection end in UTC.
        manifest_path: Notebook-selected universe CSV.
        max_workers: Concurrent symbols for ticks or concurrent days for news.
        show_progress: Whether to display progress.

    Returns:
        Per-symbol output row and processed and cached file counts.
    """
    if kind not in {"tick", "news"}:
        raise ValueError("kind must be either 'tick' or 'news'")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if start >= end:
        raise ValueError("Collection start must precede end")
    manifest = load_manifest(manifest_path)
    if kind == "news":
        return _collect_news(paths, list(manifest.symbol), max_workers,
                             start=start, end=end, manifest_path=manifest_path,
                             show_progress=show_progress)
    schedule = sessions(paths, start=start, end=end)
    symbols = list(manifest.symbol) + ["SPY"]
    collect_split_records(paths, symbols, start=start, end=end, schedule=schedule)
    identity = manifest_hash(manifest_path)

    def collect_symbol(symbol: str) -> dict[str, object]:
        """Collect or reuse daily raw tick partitions for one symbol.

        Args:
            symbol: Ticker requested using the outer collection bounds and universe identity.

        Returns:
            The symbol and its saved-row, processed-partition, and cached-partition counts.

        Raises:
            ValueError: Cached request metadata or the ordering of downloaded pages is invalid.
        """
        count = 0
        processed_partitions = cached_partitions = 0
        client = None
        try:
            for session in schedule.itertuples(index=False):
                day = pd.Timestamp(session.open).normalize()
                request_start, request_end = max(session.open, start), min(session.close, end)
                path = paths.raw(symbol, kind) / f"{day.date()}.parquet"
                record = path.with_suffix(".json")
                expected = {
                    "universe": identity, "kind": kind,
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
                    api_key, secret_key = _get_alpaca_credentials()
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
        return {"symbol": symbol, "output_rows": count,
                "processed_files": processed_partitions, "cached_files": cached_partitions}

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
    """Return raw partitions in the explicitly requested interval.

    Args:
        paths: Research data and artifact paths.
        symbol: Ticker symbol.
        kind: Raw data category.
        start: Interval start timestamp.
        end: Interval end timestamp.
        asset_class: Asset class.

    Returns:
        Raw data partition paths.
    """
    return [
        path
        for path in sorted(paths.raw(symbol, kind, asset_class=asset_class).glob("*.parquet"))
        if pd.Timestamp(path.stem, tz="UTC") < end
        and pd.Timestamp(path.stem, tz="UTC") + pd.Timedelta(days=1) > start
    ]


def read_raw(paths: ResearchPaths, symbol: str, kind: str, *,
             start: pd.Timestamp, end: pd.Timestamp,
             asset_class: str | None = None) -> pd.DataFrame:
    """Load raw observations for a symbol and interval.

    Args:
        paths: Research data and artifact paths.
        symbol: Ticker symbol.
        kind: Raw data category.
        start: Interval start timestamp.
        end: Interval end timestamp.
        asset_class: Asset class.

    Returns:
        Raw observations within the requested interval.
    """
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
    """Describe feature inputs and computation settings.

    Args:
        paths: Research data and artifact paths.
        dependencies: Input artifact paths.
        manifest_path: Universe manifest path.
        settings: Feature computation settings.

    Returns:
        Universe hash, dependency metadata, and computation settings.
    """
    inputs = {}
    for source in dependencies:
        if not source.exists():
            raise FileNotFoundError(source)
        stat = source.stat()
        inputs[str(source.relative_to(paths.root))] = [stat.st_size, stat.st_mtime_ns]
    return {"universe": manifest_hash(manifest_path), "inputs": inputs,
            "settings": json.loads(json.dumps(settings, default=str))}


def reusable_feature(path: Path, identity: dict) -> bool:
    """Check whether a feature artifact matches its input metadata.

    Args:
        path: Artifact path.
        identity: Feature input and configuration metadata.

    Returns:
        Whether a matching feature artifact exists.
    """
    if not path.exists():
        return False
    metadata = path.with_suffix(".json")
    if not metadata.exists():
        raise ValueError(f"Feature settings unavailable: {path}; deliberately rebuild this stage")
    saved = json.loads(metadata.read_text())
    if "settings" not in saved:
        raise ValueError(f"Feature settings unavailable: {path}; deliberately rebuild this stage")
    saved["settings"].pop("trade_filter", None)
    if saved != identity:
        raise ValueError(f"Feature inputs/settings changed: {path}; deliberately rebuild this stage")
    return True


def require_feature_metadata(path: Path) -> None:
    """Require readable metadata for a completed feature artifact.

    Args:
        path: Artifact path.
    """
    metadata = path.with_suffix(".json")
    if not metadata.exists():
        raise ValueError(f"Feature metadata unavailable: {path}; deliberately rebuild this stage")
    json.loads(metadata.read_text())


def save_feature(frame: pd.DataFrame, path: Path, identity: dict) -> None:
    """Save a feature artifact and its metadata.

    Args:
        frame: Input observations.
        path: Artifact path.
        identity: Feature input and configuration metadata.
    """
    save_frame(frame, path)
    path.with_suffix(".json").write_text(json.dumps(identity, indent=2))
