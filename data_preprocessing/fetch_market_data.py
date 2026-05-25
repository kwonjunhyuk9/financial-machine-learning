from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.enums import CryptoFeed, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoTradesRequest, StockTradesRequest


DEFAULT_OUTPUT_DIR = Path("data/alpaca/trades")


load_dotenv()


def _parse_datetime(value: str) -> datetime:
    """Parse a user-provided datetime into a timezone-aware UTC datetime.

    Args:
        value: ISO-like datetime string.

    Returns:
        A UTC-aware datetime object.
    """
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _default_start_end() -> tuple[datetime, datetime]:
    """Return a conservative default lookback window for trade downloads.

    Returns:
        A tuple of start and end datetimes in UTC.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=2)
    return start, end


def _get_stock_credentials() -> tuple[str, str]:
    """Read Alpaca stock market data credentials from the environment.

    Returns:
        A tuple of API key and secret key.

    Raises:
        ValueError: If the expected environment variables are not set.
    """
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "Stock data requires ALPACA_API_KEY/ALPACA_SECRET_KEY or "
            "APCA_API_KEY_ID/APCA_API_SECRET_KEY."
        )
    return api_key, secret_key


def _normalize_trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Convert Alpaca trade data into the schema used by this project.

    Args:
        trades: DataFrame returned by Alpaca's ``TradeSet.df`` property.

    Returns:
        A normalized trade DataFrame sorted by timestamp.
    """
    frame = trades.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names="timestamp")
    else:
        frame = frame.reset_index(drop=False)

    if "timestamp" not in frame.columns:
        raise ValueError("Trade data must include a timestamp column after normalization.")
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

    preferred = ["timestamp", "symbol", "price", "size", "id", "exchange", "conditions", "tape"]
    existing = [column for column in preferred if column in frame.columns]
    remaining = [column for column in frame.columns if column not in existing]
    frame = frame.loc[:, existing + remaining]
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
    return output_dir / asset_class / f"{slug}_{start_str}_{end_str}.parquet"


def fetch_alpaca_historical_trades(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    asset_class: str = "crypto",
    limit: int | None = None,
    stock_feed: str = "iex",
    crypto_feed: str = "us",
) -> pd.DataFrame:
    """Fetch historical trades from Alpaca and normalize the result.

    Args:
        symbols: Symbols to request.
        start: Inclusive request start time.
        end: Inclusive request end time.
        asset_class: Either ``"crypto"`` or ``"stock"``.
        limit: Optional maximum number of returned trades.
        stock_feed: Stock market data feed name.
        crypto_feed: Crypto market data feed name.

    Returns:
        A normalized trade DataFrame.
    """
    if asset_class == "crypto":
        client = CryptoHistoricalDataClient()
        request = CryptoTradesRequest(
            symbol_or_symbols=list(symbols),
            start=start,
            end=end,
            limit=limit,
        )
        response = client.get_crypto_trades(
            request,
            feed=CryptoFeed(crypto_feed.lower()),
        )
    elif asset_class == "stock":
        api_key, secret_key = _get_stock_credentials()
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        request = StockTradesRequest(
            symbol_or_symbols=list(symbols),
            start=start,
            end=end,
            limit=limit,
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
    asset_class: str = "crypto",
    limit: int | None = None,
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
        limit: Optional maximum number of returned trades.
        output_path: Explicit output path. If omitted, a deterministic path is used.
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
        limit=limit,
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for historical trade downloads."""
    start, end = _default_start_end()

    parser = argparse.ArgumentParser(description="Fetch Alpaca historical trades and save them to parquet.")
    parser.add_argument("--asset-class", choices=["crypto", "stock"], default="crypto")
    parser.add_argument("--symbol", dest="symbols", action="append", help="Symbol to request. Repeat for multiple.")
    parser.add_argument("--start", default=start.isoformat(), help="UTC start datetime, for example 2026-05-25T00:00:00Z.")
    parser.add_argument("--end", default=end.isoformat(), help="UTC end datetime, for example 2026-05-25T02:00:00Z.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum number of trades to request.")
    parser.add_argument("--output-path", type=Path, default=None, help="Explicit parquet output path.")
    parser.add_argument("--stock-feed", default="iex", help="Stock data feed, such as iex or sip.")
    parser.add_argument("--crypto-feed", default="us", help="Crypto data feed.")
    return parser


def main() -> None:
    """Fetch historical trades from Alpaca and save them under ``data/``."""
    parser = build_parser()
    args = parser.parse_args()

    symbols = args.symbols or ["BTC/USD"]
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    output_path = save_alpaca_historical_trades(
        symbols=symbols,
        start=start,
        end=end,
        asset_class=args.asset_class,
        limit=args.limit,
        output_path=args.output_path,
        stock_feed=args.stock_feed,
        crypto_feed=args.crypto_feed,
    )
    print(output_path)


if __name__ == "__main__":
    main()
