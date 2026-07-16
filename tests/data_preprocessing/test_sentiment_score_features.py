import pandas as pd
import pytest

from src.data_preprocessing.sentiment_score_features import score_sentiment_features


def test_score_sentiment_features_adds_finbert_scores():
    news = pd.DataFrame(
        {
            "headline": ["Earnings beat expectations", None],
            "summary": ["Shares rise after results", None],
        },
    )
    predictions = [
        [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.1},
        ],
    ]

    features = score_sentiment_features(
        news,
        classifier=lambda *args, **kwargs: predictions,
    )

    assert features.loc[0, "sentiment_positive"] == 0.8
    assert features.loc[0, "sentiment_score"] == pytest.approx(0.7)
    assert pd.isna(features.loc[1, "sentiment_score"])


def test_score_sentiment_features_requires_text_columns():
    with pytest.raises(ValueError, match="text columns"):
        score_sentiment_features(pd.DataFrame({"headline": ["news"]}))
