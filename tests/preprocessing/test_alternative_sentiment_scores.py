import json
from io import StringIO
from unittest.mock import Mock

import pandas as pd
import pytest
from tqdm import tqdm

from src.preprocessing import alternative_sentiment_scores as sentiment
from src.preprocessing import market_data
from src.preprocessing.alternative_sentiment_scores import score_sentiment_features


def test_combine_news_text_handles_missing_text_and_preserves_index():
    news = pd.DataFrame(
        {
            "headline": [" Earnings beat", "Title", None, None, "", 123],
            "summary": ["Shares rise ", None, "Summary", None, "", "Result"],
        },
        index=pd.Index([9, 3, 7, 2, 5, 1], name="article"),
    )
    original = news.copy(deep=True)

    text = sentiment.combine_news_text(news)

    assert text.tolist() == ["Earnings beat Shares rise", "Title", "Summary", "", "", "123 Result"]
    pd.testing.assert_index_equal(text.index, news.index)
    pd.testing.assert_frame_equal(news, original)


def test_combine_news_text_uses_selected_column_order():
    news = pd.DataFrame({"headline": ["Title"], "summary": ["Summary"]})

    text = sentiment.combine_news_text(news, text_columns=["summary", "headline"])

    assert text.tolist() == ["Summary Title"]


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
    paths = market_data.ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": symbols}).to_csv(manifest, index=False)
    news = pd.DataFrame({"id": [2, 1, 1], "headline": ["Growth", "Profit", "Profit"],
                         "summary": ["Up", "Up", "Discarded"],
                         "created_at": pd.to_datetime([
                             "2025-02-01 00:01:00Z", "2025-02-01 00:00:00Z", "2025-02-01 00:00:00Z",
                         ])})
    monkeypatch.setattr(market_data, "read_raw", lambda paths, symbol, *args, **kwargs: news.assign(symbol=symbol))
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
    kwargs = dict(paths=paths, manifest_path=manifest,
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
    assert report.columns.tolist() == ["symbol", "output_rows", "processed_files", "cached_files"]
    assert report.output_rows.tolist() == [2, 2]
    assert report.processed_files.tolist() == [1, 1]
    assert report.cached_files.tolist() == [0, 0]
    for call in classifier.call_args_list:
        assert call.args == (["Profit Up", "Growth Up"],)
        assert call.kwargs == {"batch_size": 16, "truncation": True}
    for symbol in report.symbol:
        saved = pd.read_parquet(kwargs["paths"].feature(symbol, "sentiment_scores"))
        assert saved.columns.tolist() == sentiment.SENTIMENT_COLUMNS
        assert saved.symbol.tolist() == [symbol, symbol]
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
    assert report.processed_files.tolist() == [0, 0]
    assert report.cached_files.tolist() == [1, 1]
    assert report.output_rows.tolist() == [2, 2]
    assert len(bars) == 1
    assert bars[0].n == 2
    assert bars[0].disable


def test_build_deduplicates_before_inference_per_symbol(sentiment_build):
    kwargs, news, factory, classifier, bars = sentiment_build
    news.loc[len(news)] = {
        "id": 3, "headline": "Profit", "summary": "Duplicate content",
        "created_at": pd.Timestamp("2025-02-01", tz="UTC"),
    }
    news.loc[len(news)] = {
        "id": 4, "headline": "Profit", "summary": "Later",
        "created_at": pd.Timestamp("2025-02-01 00:02:00", tz="UTC"),
    }
    news.sort_values("id", ascending=False, kind="stable", inplace=True)
    original = news.copy(deep=True)

    report = sentiment.build_sentiment_features(**kwargs)

    assert report.output_rows.tolist() == [3, 3]
    for call in classifier.call_args_list:
        assert call.args == (["Profit Up", "Growth Up", "Profit Later"],)
    for symbol in report.symbol:
        saved = pd.read_parquet(kwargs["paths"].feature(symbol, "sentiment_scores"))
        assert saved.id.tolist() == [1, 2, 4]
        assert saved.symbol.tolist() == [symbol] * 3
    pd.testing.assert_frame_equal(news, original)


def test_build_rejects_legacy_cache_without_loading_model(sentiment_build):
    kwargs, news, factory, classifier, bars = sentiment_build
    sentiment.build_sentiment_features(**kwargs)
    metadata = kwargs["paths"].feature("AAPL", "sentiment_scores").with_suffix(".json")
    identity = json.loads(metadata.read_text())
    assert identity["settings"].pop("deduplication") == [["id"], ["created_at", "headline"]]
    assert identity["settings"].pop("columns") == sentiment.SENTIMENT_COLUMNS
    metadata.write_text(json.dumps(identity))
    factory.reset_mock()
    classifier.reset_mock()

    with pytest.raises(ValueError, match="deliberately rebuild"):
        sentiment.build_sentiment_features(**kwargs)

    factory.assert_not_called()
    classifier.assert_not_called()


@pytest.mark.parametrize("empty_rows", [True, False])
def test_build_empty_news_or_text_does_not_load_model(sentiment_build, empty_rows):
    kwargs, news, factory, classifier, bars = sentiment_build
    if empty_rows:
        news.drop(news.index, inplace=True)
    else:
        news.loc[:, ["headline", "summary"]] = ""

    report = sentiment.build_sentiment_features(**kwargs)

    factory.assert_not_called()
    assert report.output_rows.tolist() == ([0, 0] if empty_rows else [2, 2])
    for symbol in report.symbol:
        saved = pd.read_parquet(kwargs["paths"].feature(symbol, "sentiment_scores"))
        assert saved.columns.tolist() == sentiment.SENTIMENT_COLUMNS
        assert saved.sentiment_score.isna().all()


def test_build_closes_progress_on_failure(sentiment_build):
    kwargs, news, factory, classifier, bars = sentiment_build
    classifier.side_effect = RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        sentiment.build_sentiment_features(**kwargs, show_progress=True)

    assert len(bars) == 1
    assert bars[0].disable


@pytest.mark.parametrize("index", [pd.Index([], dtype=int, name="article"),
                                  pd.DatetimeIndex([], tz="UTC", name="created_at")])
def test_combine_empty_news_preserves_index(index):
    news = pd.DataFrame(columns=["headline", "summary"], index=index)
    text = sentiment.combine_news_text(news)
    pd.testing.assert_series_equal(text, pd.Series(index=index, dtype=str))


def test_combine_empty_news_still_requires_selected_columns():
    with pytest.raises(KeyError, match="summary"):
        sentiment.combine_news_text(pd.DataFrame(columns=["headline"]))


@pytest.mark.parametrize("row_count", [0, 2])
@pytest.mark.parametrize("supplied_classifier", [False, True])
def test_empty_news_or_text_returns_float_scores_without_inference(monkeypatch, row_count, supplied_classifier):
    news = pd.DataFrame({"headline": ["  "] * row_count, "summary": [None] * row_count},
                        index=pd.Index(range(row_count), name="article"))
    original = news.copy()
    classifier = Mock()
    factory = Mock(return_value=classifier)
    monkeypatch.setattr(sentiment, "pipeline", factory)

    result = sentiment.score_sentiment_features(
        news, model_name="unused", classifier=classifier if supplied_classifier else None,
    )

    expected = news.assign(sentiment_positive=float("nan"), sentiment_negative=float("nan"),
                           sentiment_neutral=float("nan"), sentiment_score=float("nan"))
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(news, original)
    factory.assert_not_called()
    classifier.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_empty_news_still_requires_positive_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        sentiment.score_sentiment_features(pd.DataFrame(columns=["headline", "summary"]),
                                           model_name="unused", batch_size=batch_size)


def test_empty_news_still_requires_text_columns():
    with pytest.raises(ValueError, match="text columns"):
        sentiment.score_sentiment_features(pd.DataFrame(columns=["headline"]), model_name="unused")


def test_empty_news_builder_still_validates_batch_size(sentiment_build):
    kwargs, news, factory, classifier, bars = sentiment_build
    news.drop(news.index, inplace=True)
    kwargs["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        sentiment.build_sentiment_features(**kwargs)
    factory.assert_not_called()
    classifier.assert_not_called()
