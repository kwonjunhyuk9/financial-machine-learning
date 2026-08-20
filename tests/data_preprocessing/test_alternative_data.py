from datetime import datetime

import pandas as pd
import pytest

from src.data_preprocessing.alternative_data import _build_output_path
from src.data_preprocessing.alternative_data import _normalize_news_frame
from src.data_preprocessing.alternative_data import filter_aapl_high_information_news


def test_build_output_path_uses_readable_date_range():
    path = _build_output_path(
        symbols=["AAPL"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2),
    )

    assert path.name == "aapl_2026-01-01_2026-01-02.parquet"


def test_normalize_news_frame_serializes_symbols_and_sorts_time():
    news = pd.DataFrame(
        {
            "id": [2, 1],
            "headline": ["later", "earlier"],
            "source": ["source", "source"],
            "url": ["u2", "u1"],
            "summary": ["s2", "s1"],
            "created_at": ["2026-01-02", "2026-01-01"],
            "updated_at": ["2026-01-02", "2026-01-01"],
            "symbols": [["MSFT", "AAPL"], ["AAPL"]],
            "author": ["a", "a"],
            "content": ["c2", "c1"],
        }
    )

    frame = _normalize_news_frame(news)

    assert frame["id"].tolist() == [1, 2]
    assert frame.loc[1, "symbols"] == "MSFT,AAPL"


def test_normalize_news_frame_preserves_empty_schema():
    frame = _normalize_news_frame(pd.DataFrame())

    assert frame.columns.tolist() == [
        "id", "headline", "source", "url", "summary", "created_at",
        "updated_at", "symbols", "author", "content",
    ]


@pytest.mark.parametrize(
    "url",
    [
        "https://www.benzinga.com/news/25/01/42900001/apple-update",
        "https://benzinga.com/news/earnings/25/01/42900002/apple-results",
        "https://benzinga.com/news/legal/25/01/42900003/apple-case",
        "https://benzinga.com/news/contracts/25/01/42900004/apple-contract",
        "https://benzinga.com/news/buybacks/25/01/42900005/apple-buyback",
        "https://benzinga.com/news/stock-split/25/01/42900006/apple-split",
        "https://benzinga.com/analyst-ratings/price-target/25/01/42900007/apple-target",
        "https://benzinga.com/analyst-ratings/upgrades/25/01/42900008/apple-upgrade",
        "https://benzinga.com/analyst-ratings/downgrades/25/01/42900009/apple-cut",
        "https://benzinga.com/analyst-ratings/reiteration/25/01/42900010/apple-rating",
        "https://benzinga.com/analyst-stock-ratings/price-target/25/01/42900011/apple-target",
        "https://benzinga.com/analyst-stock-ratings/upgrades/25/01/42900012/apple-upgrade",
        "https://benzinga.com/analyst-stock-ratings/downgrades/25/01/42900013/apple-cut",
        "https://benzinga.com/analyst-stock-ratings/reiteration/25/01/42900014/apple-rating",
        "https://benzinga.com/markets/guidance/25/01/42900015/apple-guidance",
    ],
)
def test_filter_aapl_high_information_news_accepts_allowed_urls(url):
    news = pd.DataFrame({"symbols": [" aapl "], "url": [url]})

    filtered = filter_aapl_high_information_news(news)

    assert len(filtered) == 1


@pytest.mark.parametrize(
    ("symbols", "url"),
    [
        (
            "AAPL,MSFT",
            "https://benzinga.com/news/25/01/42900001/apple-update",
        ),
        (
            "AAPL",
            "https://example.com/news/25/01/42900001/apple-update",
        ),
        (
            "AAPL",
            "https://benzinga.com/general/25/01/42900001/apple-update",
        ),
        (
            "AAPL",
            "https://benzinga.com/news/earnings/25/01/",
        ),
    ],
)
def test_filter_aapl_high_information_news_rejects_ineligible_rows(symbols, url):
    news = pd.DataFrame({"symbols": [symbols], "url": [url]})

    assert filter_aapl_high_information_news(news).empty


def test_filter_aapl_high_information_news_preserves_schema_order_and_created_at():
    created_at = pd.to_datetime(
        ["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"]
    )
    news = pd.DataFrame(
        {
            "id": [2, 1],
            "created_at": created_at,
            "symbols": [["AAPL"], {"aapl"}],
            "url": [
                "https://benzinga.com/news/25/01/42900002/later",
                "https://benzinga.com/markets/guidance/25/01/42900001/earlier",
            ],
            "headline": ["later", "earlier"],
        },
        index=[9, 4],
    )

    filtered = filter_aapl_high_information_news(news)

    assert filtered.columns.tolist() == news.columns.tolist()
    assert filtered["id"].tolist() == [2, 1]
    assert filtered["created_at"].tolist() == list(created_at)
    assert filtered.index.tolist() == [0, 1]


def test_filter_aapl_high_information_news_preserves_empty_input():
    news = pd.DataFrame(columns=["id", "symbols", "url", "created_at"])

    filtered = filter_aapl_high_information_news(news)

    assert filtered.empty
    assert filtered.columns.tolist() == news.columns.tolist()
