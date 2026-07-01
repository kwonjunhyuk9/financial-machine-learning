from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.enums import CryptoFeed, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoTradesRequest, StockTradesRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/raw/alpaca"

load_dotenv()


def _get_credentials() -> tuple[str, str]:
    """Read Alpaca market data credentials from the environment.

    Args:
        No arguments.

    Returns:
        A tuple of API key and secret key.

    Raises:
        ValueError: If Alpaca credentials are not configured.
    """
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "Data requires ALPACA_API_KEY/ALPACA_SECRET_KEY or "
            "APCA_API_KEY_ID/APCA_API_SECRET_KEY."
        )
    return api_key, secret_key


def _normalize_trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Convert Alpaca trade data into the schema used by this project.

    Args:
        trades: DataFrame returned by Alpaca's ``TradeSet.df`` property.

    Returns:
        A normalized trade DataFrame sorted by timestamp.

    Raises:
        ValueError: If required trade columns are missing.
    """
    frame = trades.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names="timestamp")
    else:
        frame = frame.reset_index(drop=False)

    preferred = ["timestamp", "symbol", "price", "size", "id", "exchange"]
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

    existing = [column for column in preferred if column in frame.columns]
    frame = frame.loc[:, existing]
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    return frame


def _build_output_path(
        *,
        asset_class: str,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Build a deterministic output path for a trade dataset.

    Args:
        asset_class: Asset class name.
        symbols: Symbols included in the request.
        start: Inclusive request start time.
        end: Inclusive request end time.
        output_dir: Root directory for saved trade data.

    Returns:
        A parquet file path.
    """
    slug = "_".join(symbols).replace("/", "-").replace(":", "-").replace(" ", "").lower()
    start_str = start.strftime("%Y%m%dT%H%M%SZ")
    end_str = end.strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"{asset_class}_{slug}_trades_{start_str}_{end_str}.parquet"


def fetch_alpaca_historical_trades(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        asset_class: str,
        stock_feed: str = "iex",
        crypto_feed: str = "us",
) -> pd.DataFrame:
    """Fetch historical trades from Alpaca and normalize the result.

    Args:
        symbols: Symbols to request.
        start: Inclusive request start time.
        end: Inclusive request end time.
        asset_class: Either ``"crypto"`` or ``"stock"``.
        stock_feed: Stock market data feed name.
        crypto_feed: Crypto market data feed name.

    Returns:
        A normalized trade DataFrame.

    Raises:
        ValueError: If ``asset_class`` is not ``"crypto"`` or ``"stock"``.
    """
    if asset_class == "crypto":
        client = CryptoHistoricalDataClient()
        request = CryptoTradesRequest(
            symbol_or_symbols=list(symbols),
            start=start,
            end=end,
        )
        response = client.get_crypto_trades(
            request,
            feed=CryptoFeed(crypto_feed.lower()),
        )
    elif asset_class == "stock":
        api_key, secret_key = _get_credentials()
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        request = StockTradesRequest(
            symbol_or_symbols=list(symbols),
            start=start,
            end=end,
            feed=DataFeed(stock_feed.lower()),
        )
        response = client.get_stock_trades(request)
    else:
        raise ValueError("asset_class must be either 'crypto' or 'stock'.")

    return _normalize_trade_frame(response.df)


def save_alpaca_historical_trades(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        asset_class: str,
        output_path: Path | None = None,
        stock_feed: str = "iex",
        crypto_feed: str = "us",
) -> Path:
    """Fetch Alpaca historical trades and save them to parquet.

    Args:
        symbols: Symbols to request.
        start: Inclusive request start time.
        end: Inclusive request end time.
        asset_class: Either ``"crypto"`` or ``"stock"``.
        output_path: Explicit output path.
        stock_feed: Stock market data feed name.
        crypto_feed: Crypto market data feed name.

    Returns:
        The parquet path written to disk.
    """
    trades = fetch_alpaca_historical_trades(
        symbols=symbols,
        start=start,
        end=end,
        asset_class=asset_class,
        stock_feed=stock_feed,
        crypto_feed=crypto_feed,
    )
    destination = output_path or _build_output_path(
        asset_class=asset_class,
        symbols=symbols,
        start=start,
        end=end,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(destination, index=False)
    return destination
