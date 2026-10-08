from io import StringIO
from unittest.mock import Mock

import pandas as pd
import pytest
from tqdm import tqdm

from src.preprocessing import alternative_sentiment_scores as sentiment
from src.preprocessing import market_data
from src.preprocessing.alternative_sentiment_scores import score_sentiment_features


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
        model_name="ProsusAI/finbert",
    )

    assert features.loc[0, "sentiment_positive"] == 0.8
    assert features.loc[0, "sentiment_score"] == pytest.approx(0.7)
    assert pd.isna(features.loc[1, "sentiment_score"])


def test_score_sentiment_features_requires_text_columns():
    with pytest.raises(ValueError, match="text columns"):
        score_sentiment_features(pd.DataFrame({"headline": ["news"]}), model_name="ProsusAI/finbert")


@pytest.fixture
def sentiment_build(tmp_path, monkeypatch):
    symbols = ["AAPL", "MSFT"]
    paths = market_data.ResearchPaths(tmp_path, period="2025")
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": symbols}).to_csv(manifest, index=False)
    news = pd.DataFrame({"id": [2, 1, 1], "headline": ["Growth", "Profit", "Profit"],
                         "summary": ["Up", "Up", "Up"]})
    monkeypatch.setattr(market_data, "read_raw", lambda *args, **kwargs: news.copy())
    monkeypatch.setattr(market_data, "raw_partitions", lambda *args, **kwargs: [])
    classifier = Mock(side_effect=lambda texts, **kwargs: [
        [{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.1},
         {"label": "neutral", "score": 0.1}] for text in texts
    ])
    factory = Mock(return_value=classifier)
    monkeypatch.setattr(sentiment, "pipeline", factory)
    bars = []

    def progress(*args, **kwargs):
        bar = tqdm(*args, file=StringIO(), **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(sentiment, "tqdm", progress)
    kwargs = dict(paths=paths, manifest_path=manifest, expected_securities=2,
                  start=pd.Timestamp("2025-01-01", tz="UTC"),
                  end=pd.Timestamp("2026-01-01", tz="UTC"),
                  model_name="ProsusAI/finbert", text_columns=["headline", "summary"],
                  batch_size=16)
    return kwargs, news, factory, classifier, bars


def test_build_reuses_classifier_and_closes_single_progress_bar(sentiment_build):
    kwargs, news, factory, classifier, bars = sentiment_build
    report = sentiment.build_sentiment_features(**kwargs, show_progress=True)

    factory.assert_called_once_with("text-classification", model="ProsusAI/finbert",
                                    tokenizer="ProsusAI/finbert", top_k=None)
    assert classifier.call_count == 2
    assert report.rows.tolist() == [2, 2]
    for call in classifier.call_args_list:
        assert call.args == (["Profit Up", "Growth Up"],)
        assert call.kwargs == {"batch_size": 16, "truncation": True}
    for symbol in report.symbol:
        saved = pd.read_parquet(kwargs["paths"].feature(symbol, "sentiment_scores"))
        assert saved.id.tolist() == [1, 2]
        assert saved.sentiment_score.tolist() == pytest.approx([0.7, 0.7])
    assert len(bars) == 1
    assert bars[0].n == bars[0].total == 2
    assert bars[0].postfix == "MSFT"
    assert bars[0].disable  # tqdm closes the bar by disabling it.


def test_build_all_cached_skips_model_and_counts_progress(sentiment_build):
    kwargs, news, factory, classifier, bars = sentiment_build
    sentiment.build_sentiment_features(**kwargs)
    factory.reset_mock()
    classifier.reset_mock()
    bars.clear()

    report = sentiment.build_sentiment_features(**kwargs, show_progress=True)

    factory.assert_not_called()
    classifier.assert_not_called()
    assert report.status.tolist() == ["cached", "cached"]
    assert len(bars) == 1
    assert bars[0].n == 2
    assert bars[0].disable


@pytest.mark.parametrize("empty_rows", [True, False])
def test_build_empty_news_or_text_does_not_load_model(sentiment_build, empty_rows):
    kwargs, news, factory, classifier, bars = sentiment_build
    if empty_rows:
        news.drop(news.index, inplace=True)
    else:
        news.loc[:, ["headline", "summary"]] = ""

    report = sentiment.build_sentiment_features(**kwargs)

    factory.assert_not_called()
    assert report.rows.tolist() == ([0, 0] if empty_rows else [2, 2])
    for symbol in report.symbol:
        saved = pd.read_parquet(kwargs["paths"].feature(symbol, "sentiment_scores"))
        assert saved.sentiment_score.isna().all()


def test_build_closes_progress_on_failure(sentiment_build):
    kwargs, news, factory, classifier, bars = sentiment_build
    classifier.side_effect = RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        sentiment.build_sentiment_features(**kwargs, show_progress=True)

    assert len(bars) == 1
    assert bars[0].disable
