from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from transformers import pipeline

FINBERT_MODEL = "ProsusAI/finbert"


def score_sentiment_features(
        news: pd.DataFrame,
        *,
        text_columns: Sequence[str] = ("headline", "summary"),
        batch_size: int = 16,
        classifier: Callable[..., list[Any]] | None = None,
) -> pd.DataFrame:
    """Add FinBERT sentiment probabilities and score to news rows.

    Args:
        news: News rows containing the selected text columns.
        text_columns: Ordered text columns combined for each article.
        batch_size: Number of articles scored in one model batch.
        classifier: Optional text-classification callable for testing or reuse.

    Returns:
        A copy of ``news`` with positive, negative, neutral, and sentiment-score columns.

    Raises:
        ValueError: If text columns are missing or the batch size is invalid.
    """
    missing_columns = set(text_columns).difference(news.columns)
    if missing_columns:
        raise ValueError(f"News data is missing text columns: {sorted(missing_columns)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    features = news.copy()
    text = (
        features.loc[:, text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )
    score_columns = ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]
    features.loc[:, score_columns] = float("nan")

    valid_text = text.ne("")
    if not valid_text.any():
        features["sentiment_score"] = float("nan")
        return features

    classifier = classifier or pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        top_k=None,
    )
    predictions = classifier(
        text.loc[valid_text].tolist(),
        batch_size=batch_size,
        truncation=True,
    )
    probabilities = [
        {
            str(prediction["label"]).lower(): float(prediction["score"])
            for prediction in article_predictions
        }
        for article_predictions in predictions
    ]
    scores = pd.DataFrame(probabilities, index=features.index[valid_text]).reindex(
        columns=["positive", "negative", "neutral"],
        fill_value=0.0,
    )
    features.loc[valid_text, score_columns] = scores.rename(
        columns={
            "positive": "sentiment_positive",
            "negative": "sentiment_negative",
            "neutral": "sentiment_neutral",
        },
    )
    features["sentiment_score"] = (
        features["sentiment_positive"] - features["sentiment_negative"]
    )
    return features
