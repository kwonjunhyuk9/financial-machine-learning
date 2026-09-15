from __future__ import annotations

import os
from datetime import datetime
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/research_data/market/data"
MarketDataType = Literal["tick", "1min"]


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
