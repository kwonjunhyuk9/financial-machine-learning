from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm

def score_sentiment_features(
        news: pd.DataFrame,
        *,
        model_name: str,
        text_columns: Sequence[str] = ("headline", "summary"),
        batch_size: int = 16,
        classifier: Callable[..., list[Any]] | None = None,
) -> pd.DataFrame:
    """Add sentiment probabilities and score using the selected model.

    Args:
        news: News rows containing the selected text columns.
        model_name: Model and tokenizer identifier used when creating a classifier.
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
        model=model_name,
        tokenizer=model_name,
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


def build_sentiment_features(paths, *, manifest_path, expected_securities: int,
                             start: pd.Timestamp, end: pd.Timestamp, model_name: str,
                             text_columns: Sequence[str], batch_size: int,
                             show_progress: bool = False) -> pd.DataFrame:
    """Build resumable sentiment features, lazily reusing one classifier.

    Args:
        show_progress: Show one overall progress bar with the current symbol,
            including symbols whose results are reused from cache.

    The classifier is loaded only when non-empty text requires inference.
    Returns the complete per-symbol report with row counts and cache status.
    """
    from src.preprocessing.market_data import (
        feature_identity, load_manifest, raw_partitions, read_raw,
        reusable_feature, save_feature,
    )

    classifier = None

    def classify(*args, **kwargs):
        nonlocal classifier
        if classifier is None:
            classifier = pipeline(
                "text-classification", model=model_name, tokenizer=model_name, top_k=None,
            )
        return classifier(*args, **kwargs)

    report = []
    symbols = load_manifest(manifest_path, expected_securities=expected_securities).symbol
    with tqdm(symbols, desc="Sentiment scores", disable=not show_progress) as progress:
        for symbol in progress:
            progress.set_postfix_str(symbol)
            news = read_raw(paths, symbol, "news", start=start, end=end).drop_duplicates("id").sort_values("id")
            output = paths.feature(symbol, "sentiment_scores")
            identity = feature_identity(
                paths,
                [
                    path.with_suffix(".json")
                    for path in raw_partitions(paths, symbol, "news", start=start, end=end)
                ],
                manifest_path=manifest_path,
                settings={"start": start, "end": end, "model_name": model_name,
                          "text_columns": list(text_columns), "batch_size": batch_size},
            )
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
                result = score_sentiment_features(news, model_name=model_name,
                                                  text_columns=text_columns, batch_size=batch_size,
                                                  classifier=classify)
            save_feature(result, output, identity)
            report.append({"symbol": symbol, "status": "processed", "rows": len(result)})
    return pd.DataFrame(report)
