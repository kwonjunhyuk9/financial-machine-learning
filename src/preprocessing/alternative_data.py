from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence
from urllib.parse import urlparse

import pandas as pd

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

from src.preprocessing import _get_alpaca_credentials

_NEWS_AND_ANALYST_RATINGS_PATH = re.compile(
    r"^/(?:news|analyst-ratings|analyst-stock-ratings)"
    r"(?:/[^/]+)*/\d{2}/\d{2}/[^/]+(?:/.*)?$"
)


def _parse_symbols(value: object) -> set[str]:
    """Normalize collection-valued or comma-separated news symbols.

    Args:
        value: A symbol value or a collection whose items may contain comma-separated symbols.

    Returns:
        Distinct uppercase symbols with surrounding whitespace and empty entries removed.
    """
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return {
        symbol.strip().upper()
        for item in values
        for symbol in str(item).split(",")
        if symbol.strip()
    }


def _is_news_or_analyst_ratings_url(value: object) -> bool:
    """Recognize supported Benzinga article URLs.

    Args:
        value: URL to check, converted to a string.

    Returns:
        Whether the host is benzinga.com, optionally prefixed by www, and the dated path is a
        supported news or analyst-ratings article.
    """
    parsed = urlparse(str(value))
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host == "benzinga.com" and bool(
        _NEWS_AND_ANALYST_RATINGS_PATH.fullmatch(parsed.path)
    )


def filter_symbol_news(news: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Select Benzinga News and Analyst Ratings articles for a symbol.

    Args:
        news: News articles.
        symbol: Ticker symbol.

    Returns:
        Selected news articles.
    """
    required_columns = {"symbols", "url"}
    missing_columns = required_columns.difference(news.columns)
    if missing_columns:
        raise ValueError(f"News data is missing columns: {sorted(missing_columns)}")

    if news.empty:
        return news.reset_index(drop=True)

    eligible = news["symbols"].map(_parse_symbols).eq({symbol.upper()})
    eligible &= news["url"].map(_is_news_or_analyst_ratings_url)
    if "source" in news:
        eligible &= news["source"].str.lower().eq("benzinga")
    return news.loc[eligible].reset_index(drop=True)


def _normalize_news_frame(news: pd.DataFrame) -> pd.DataFrame:
    """Normalize Alpaca news for Parquet storage.

    Args:
        news: Alpaca news frame whose index is restored as columns.

    Returns:
        Fixed news columns sorted by created_at, with UTC timestamps and list-valued symbols
        serialized as comma-separated strings. Empty input retains the column schema.
    """
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
        Normalized news articles.
    """
    api_key, secret_key = _get_alpaca_credentials()
    client = NewsClient(api_key=api_key, secret_key=secret_key)
    request = NewsRequest(
        symbols=",".join(symbols),
        start=start,
        end=end,
        include_content=True,
    )
    response = client.get_news(request)
    frame = response.df if response.data["news"] else pd.DataFrame()
    return _normalize_news_frame(frame)
