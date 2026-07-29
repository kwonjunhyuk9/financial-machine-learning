from datetime import datetime

import pandas as pd

from src.data_preprocessing.alternative_data import _build_output_path
from src.data_preprocessing.alternative_data import _normalize_news_frame


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
