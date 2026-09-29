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
        batch_size: Number of articles scored in one models batch.
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


def build_sentiment_features(paths) -> pd.DataFrame:
    """Build resumable sentiment features for every fixed-universe symbol."""
    from src.preprocessing.market_data import (
        feature_identity, load_manifest, read_raw, reusable_feature, save_feature,
    )

    report = []
    for symbol in load_manifest(paths).symbol:
        news = read_raw(paths, symbol, "news").drop_duplicates("id").sort_values("id")
        output = paths.feature(symbol, "sentiment_scores")
        identity = feature_identity(paths, sorted(paths.raw(symbol, "news").glob("*.json")))
        cached = reusable_feature(output, identity)
        if cached and pd.read_parquet(output, columns=["id"]).id.tolist() == news.id.tolist():
            report.append({"symbol": symbol, "status": "cached", "rows": len(news)})
            continue
        if news.empty:
            result = news.assign(
                sentiment_positive=pd.Series(dtype=float),
                sentiment_negative=pd.Series(dtype=float),
                sentiment_neutral=pd.Series(dtype=float),
                sentiment_score=pd.Series(dtype=float),
            )
        else:
            result = score_sentiment_features(news, batch_size=16)
        save_feature(result, output, identity)
        report.append({"symbol": symbol, "rows": len(result)})
    return pd.DataFrame(report)
