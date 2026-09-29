from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from alpaca.data.enums import CryptoFeed, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoTradesRequest,
    StockBarsRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/preprocessing/market/data"
MarketDataType = Literal["tick", "1min"]

PERIOD = "2025-01-01_2025-12-31"
VERSION = "sp500-fixed-2025-v2"
UNIVERSE_URL = (
    "https://en.wikipedia.org/w/index.php?"
    "title=List_of_S%26P_500_companies&oldid=1265285344"
)
EXPECTED_COMPANIES = 500
EXPECTED_SECURITIES = 503
START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
WARMUP = pd.Timestamp("2023-10-01", tz="UTC")


@dataclass(frozen=True)
class ResearchPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data/research_data"

    @property
    def universe(self) -> Path:
        return self.data / "universe"

    @property
    def artifacts(self) -> Path:
        return self.root / "data/model_artifact"

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
    """Normalize Alpaca trades into the project's tabular schema."""
    frame = trades.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names="timestamp")
    else:
        frame = frame.reset_index(drop=False)

    preferred = ["timestamp", "symbol", "price", "size"]
    if frame.empty:
        return frame.reindex(columns=preferred)

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

    frame = frame.loc[:, preferred]
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    return frame


def _normalize_minute_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize Alpaca minute bars into close-price and size rows."""
    frame = bars.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names="timestamp")
    else:
        frame = frame.reset_index(drop=False)

    preferred = ["timestamp", "symbol", "price", "size"]
    if frame.empty:
        return frame.reindex(columns=preferred)

    if "timestamp" not in frame.columns:
        raise ValueError(
            "Minute data must include a timestamp column after normalization. "
            f"Columns: {frame.columns.tolist()}"
        )
    if "symbol" not in frame.columns:
        raise ValueError("Minute data must include a symbol column after normalization.")
    if "close" not in frame.columns:
        raise ValueError("Minute data must include a close column after normalization.")
    if "volume" not in frame.columns:
        raise ValueError("Minute data must include a volume column after normalization.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["close"] = frame["close"].astype(float)
    frame["volume"] = frame["volume"].astype(float)

    if "vwap" in frame:
        frame["dollar_value"] = frame["vwap"].astype(float) * frame["volume"]
        preferred = [*preferred, "dollar_value"]
    frame = frame.rename(columns={"close": "price", "volume": "size"})
    frame = frame.loc[:, preferred]
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    return frame


def _build_output_path(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Build a deterministic parquet path for a market dataset."""
    slug = "_".join(symbols).replace("/", "-").replace(":", "-").replace(" ", "").lower()
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    return output_dir / f"{slug}_{start_str}_{end_str}.parquet"


def fetch_alpaca_historical_data(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    asset_class: str,
    data_type: MarketDataType,
    stock_feed: str = "iex",
    crypto_feed: str = "us",
) -> pd.DataFrame:
    """Fetch historical tick or one-minute data from Alpaca.

    Args:
        symbols: Symbols to request.
        start: Inclusive request start time.
        end: Exclusive result end time.
        asset_class: Either ``"crypto"`` or ``"stock"``.
        data_type: Either ``"tick"`` or ``"1min"``.
        stock_feed: Stock market data feed name.
        crypto_feed: Crypto market data feed name.

    Returns:
        A normalized tick or one-minute DataFrame.

    Raises:
        ValueError: If ``asset_class`` or ``data_type`` is unsupported.
    """
    if data_type not in ("tick", "1min"):
        raise ValueError("data_type must be either 'tick' or '1min'.")

    if asset_class == "crypto":
        client = CryptoHistoricalDataClient()
        if data_type == "tick":
            request = CryptoTradesRequest(
                symbol_or_symbols=list(symbols),
                start=start,
                end=end,
            )
            response = client.get_crypto_trades(
                request,
                feed=CryptoFeed(crypto_feed.lower()),
            )
        else:
            request = CryptoBarsRequest(
                symbol_or_symbols=list(symbols),
                start=start,
                end=end,
                timeframe=TimeFrame.Minute,
            )
            response = client.get_crypto_bars(
                request,
                feed=CryptoFeed(crypto_feed.lower()),
            )
    elif asset_class == "stock":
        api_key, secret_key = _get_credentials()
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        if data_type == "tick":
            request = StockTradesRequest(
                symbol_or_symbols=list(symbols),
                start=start,
                end=end,
                feed=DataFeed(stock_feed.lower()),
            )
            response = client.get_stock_trades(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=list(symbols),
                start=start,
                end=end,
                timeframe=TimeFrame.Minute,
                feed=DataFeed(stock_feed.lower()),
            )
            response = client.get_stock_bars(request)
    else:
        raise ValueError("asset_class must be either 'crypto' or 'stock'.")

    if data_type == "tick":
        market_data = _normalize_trade_frame(response.df)
    else:
        market_data = _normalize_minute_frame(response.df)

    end_timestamp = pd.to_datetime(end, utc=True)
    return market_data.loc[market_data["timestamp"] < end_timestamp].reset_index(drop=True)


def save_alpaca_historical_data(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    asset_class: str,
    data_type: MarketDataType,
    output_path: Path | None = None,
    stock_feed: str = "iex",
    crypto_feed: str = "us",
) -> Path:
    """Fetch Alpaca historical market data and save it to parquet.

    Args:
        symbols: Symbols to request.
        start: Inclusive request start time.
        end: Exclusive result end time.
        asset_class: Either ``"crypto"`` or ``"stock"``.
        data_type: Either ``"tick"`` or ``"1min"``.
        output_path: Explicit output path.
        stock_feed: Stock market data feed name.
        crypto_feed: Crypto market data feed name.

    Returns:
        The parquet path written to disk.
    """
    market_data = fetch_alpaca_historical_data(
        symbols=symbols,
        start=start,
        end=end,
        asset_class=asset_class,
        data_type=data_type,
        stock_feed=stock_feed,
        crypto_feed=crypto_feed,
    )
    destination = output_path or _build_output_path(
        symbols=symbols,
        start=start,
        end=end,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    market_data.to_parquet(destination, index=False)
    logger.info(
        "Saved {} historical {} rows to {}.",
        len(market_data),
        data_type,
        destination,
    )
    return destination


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    """Atomically publish a complete parquet table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sessions(paths: ResearchPaths) -> pd.DataFrame:
    """Load or cache the exchange calendar, including early closes."""
    destination = paths.universe / "sessions.parquet"
    if destination.exists():
        return pd.read_parquet(destination)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    key, secret = _get_credentials()
    calendar = TradingClient(key, secret).get_calendar(
        GetCalendarRequest(start=WARMUP.date(), end=END.date())
    )
    rows = []
    for day in calendar:
        def utc(value):
            stamp = pd.Timestamp(value)
            localized = stamp.tz_localize("America/New_York") if stamp.tzinfo is None else stamp
            return localized.tz_convert("UTC")

        rows.append({"session": str(day.date), "open": utc(day.open), "close": utc(day.close)})
    result = pd.DataFrame(rows).sort_values("open").reset_index(drop=True)
    save_frame(result, destination)
    return result


def load_manifest(paths: ResearchPaths) -> pd.DataFrame:
    manifest = pd.read_parquet(paths.universe / "sp500_2025_manifest.parquet")
    if manifest.columns.tolist() != ["symbol"]:
        raise ValueError("The frozen manifest must contain only the symbol column; rebuild this stage")
    if len(manifest) != EXPECTED_SECURITIES or manifest.symbol.nunique() != EXPECTED_SECURITIES:
        raise ValueError(f"The frozen manifest must contain {EXPECTED_SECURITIES} distinct symbols")
    if manifest.symbol.isna().any() or manifest.symbol.eq("").any():
        raise ValueError("Universe symbols must be complete")
    return manifest


def manifest_hash(paths: ResearchPaths) -> str:
    return sha256((paths.universe / "sp500_2025_manifest.parquet").read_bytes()).hexdigest()


def prepare_universe(paths: ResearchPaths) -> pd.DataFrame:
    """Create the fixed opening-2025 security list from a pinned web revision."""
    destination = paths.universe / "sp500_2025_manifest.parquet"
    if destination.exists():
        return load_manifest(paths)
    tables = pd.read_html(
        UNIVERSE_URL,
        storage_options={"User-Agent": "financial-machine-learning/0.0 (educational research)"},
    )
    if not tables:
        raise ValueError("Pinned Wikipedia revision contains no tables")
    members = tables[0]
    required = {"Symbol", "CIK"}
    if not required.issubset(members):
        raise ValueError(f"Pinned universe table requires {sorted(required)}")
    if members["CIK"].nunique() != EXPECTED_COMPANIES:
        raise ValueError(f"Pinned universe must contain {EXPECTED_COMPANIES} companies")
    symbols = members["Symbol"].astype("string").str.strip()
    if len(symbols) != EXPECTED_SECURITIES or symbols.nunique() != EXPECTED_SECURITIES:
        raise ValueError(f"Pinned universe must contain {EXPECTED_SECURITIES} distinct securities")
    if symbols.isna().any() or symbols.eq("").any():
        raise ValueError("Pinned universe symbols must be complete")
    save_frame(pd.DataFrame({"symbol": symbols.sort_values().to_numpy()}), destination)
    return load_manifest(paths)


def collect_raw(paths: ResearchPaths, kind: str) -> pd.DataFrame:
    """Collect daily market or news partitions with completion records."""
    from src.preprocessing.alternative_data import fetch_alpaca_news, filter_symbol_news

    manifest = load_manifest(paths)
    schedule = sessions(paths)
    symbols = list(manifest.symbol) + (["SPY"] if kind == "1min" else [])
    identity = manifest_hash(paths)
    counts = []
    for symbol in symbols:
        count = 0
        days = (pd.date_range(START, END, inclusive="left", freq="D") if kind == "news"
                else schedule.loc[schedule.open.lt(END), "open"])
        for stamp in days:
            day = pd.Timestamp(stamp).normalize()
            if kind == "news":
                start, end = day, day + pd.Timedelta(days=1)
            else:
                session = schedule.loc[schedule.open.dt.normalize().eq(day)].iloc[0]
                start, end = session.open, session.close
            path = paths.raw(symbol, kind) / f"{day.date()}.parquet"
            record = path.with_suffix(".json")
            expected = {
                "version": VERSION, "universe": identity, "kind": kind,
                "feed": "benzinga" if kind == "news" else "sip",
                "start": str(start), "end": str(end), "request_symbol": symbol,
            }
            if path.exists() and record.exists():
                saved = json.loads(record.read_text())
                if any(saved.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Cache metadata mismatch: {path}")
                count += saved["rows"]
                continue
            if kind == "news":
                frame = fetch_alpaca_news(
                    symbols=[symbol], start=start.to_pydatetime(), end=end.to_pydatetime()
                )
                frame = filter_symbol_news(frame, symbol)
                frame = frame.loc[frame.created_at.ge(start) & frame.created_at.lt(end)].copy()
                frame["symbol"] = symbol
            else:
                frame = fetch_alpaca_historical_data(
                    symbols=[symbol], start=start.to_pydatetime(), end=end.to_pydatetime(),
                    asset_class="stock", data_type=kind, stock_feed="sip",
                )
                frame["symbol"] = symbol
            save_frame(frame, path)
            record.write_text(json.dumps({**expected, "rows": len(frame)}, indent=2))
            count += len(frame)
        counts.append({"symbol": symbol, "kind": kind, "rows": count})
    return pd.DataFrame(counts)


def read_raw(paths: ResearchPaths, symbol: str, kind: str) -> pd.DataFrame:
    files = sorted(paths.raw(symbol, kind).glob("*.parquet"))
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
