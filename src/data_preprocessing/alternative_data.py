from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/research_data/alternative"


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


def _normalize_news_frame(news: pd.DataFrame) -> pd.DataFrame:
    """Normalize Alpaca news into a parquet-friendly table."""
    frame = news.reset_index()
    columns = [
        "id",
        "headline",
        "source",
        "url",
        "summary",
        "created_at",
        "updated_at",
        "symbols",
        "author",
        "content",
    ]
    if frame.empty:
        return frame.reindex(columns=columns)
    frame["symbols"] = frame["symbols"].map(
        lambda symbols: ",".join(symbols) if isinstance(symbols, list) else symbols,
    )
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True)
    return (
        frame.reindex(columns=columns)
        .sort_values("created_at", kind="stable")
        .reset_index(drop=True)
    )


def _build_output_path(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
) -> Path:
    """Build a deterministic parquet path for a news dataset."""
    slug = "_".join(symbols).replace("/", "-").replace(" ", "").lower()
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    return DEFAULT_OUTPUT_DIR / f"{slug}_{start_str}_{end_str}.parquet"


def fetch_alpaca_news(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
) -> pd.DataFrame:
    """Fetch Alpaca news for alternative-data sentiment analysis.

    Args:
        symbols: Ticker symbols used to filter news.
        start: Inclusive request start time.
        end: Inclusive request end time.

    Returns:
        A normalized news DataFrame.
    """
    api_key, secret_key = _get_credentials()
    client = NewsClient(api_key=api_key, secret_key=secret_key)
    request = NewsRequest(
        symbols=",".join(symbols),
        start=start,
        end=end,
        include_content=True,
    )
    return _normalize_news_frame(client.get_news(request).df)


def save_alpaca_news(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        output_path: Path | None = None,
) -> Path:
    """Fetch Alpaca news and save it as parquet.

    Args:
        symbols: Ticker symbols used to filter news.
        start: Inclusive request start time.
        end: Inclusive request end time.
        output_path: Optional parquet destination.

    Returns:
        The parquet path written to disk.
    """
    news = fetch_alpaca_news(symbols=symbols, start=start, end=end)
    destination = output_path or _build_output_path(symbols=symbols, start=start, end=end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    news.to_parquet(destination, index=False)
    logger.info("Saved {} news rows to {}.", len(news), destination)
    return destination
