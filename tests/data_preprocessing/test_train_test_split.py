import pandas as pd
import pytest

from src.data_preprocessing.train_test_split import (
    build_event_candidates,
    build_event_feature_schema,
    chronological_train_test_split,
)


def test_build_event_candidates_aligns_and_aggregates_news():
    bar_ends = pd.to_datetime(
        ["2026-01-01 14:30:00Z", "2026-01-01 14:31:00Z"]
    )
    news = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                [
                    "2026-01-01 14:29:30Z",
                    "2026-01-01 14:29:30Z",
                    "2026-01-01 14:30:01Z",
                ]
            ),
            "headline": ["first", "first", "second"],
            "sentiment_score": [0.2, 0.9, -0.4],
        }
    )

    candidates = build_event_candidates(news, bar_ends)

    assert candidates.columns.tolist() == [
        "event_start",
        "news_count",
        "mean_sentiment_score",
    ]
    assert candidates["event_start"].tolist() == list(bar_ends)
    assert candidates["news_count"].tolist() == [1, 1]
    assert candidates["mean_sentiment_score"].tolist() == pytest.approx([0.2, -0.4])


def test_build_event_feature_schema_preserves_rows_and_missing_values():
    event_starts = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    candidates = pd.DataFrame(
        {
            "event_start": event_starts,
            "news_count": [1, 2],
            "mean_sentiment_score": [0.2, -0.4],
        }
    )
    fractional = pd.DataFrame(
        {
            "end": event_starts[:1],
            "fractionally_differenced_log_close": [0.1],
        }
    )
    technical_columns = [f"technical_{index}" for index in range(51)]
    technical = pd.DataFrame(
        {
            "start": event_starts - pd.Timedelta(minutes=1),
            "end": event_starts,
            "symbol": "AAPL",
            **{
                column: [float(index), float(index + 1)]
                for index, column in enumerate(technical_columns)
            },
        }
    )
    technical.loc[1, "technical_0"] = float("nan")

    schema = build_event_feature_schema(candidates, fractional, technical)

    assert schema.columns.tolist() == [
        "event_start",
        "symbol",
        "news_count",
        "mean_sentiment_score",
        "fractionally_differenced_log_close",
        *technical_columns,
    ]
    assert schema.shape == (2, 56)
    assert schema["event_start"].tolist() == list(event_starts)
    assert schema["fractionally_differenced_log_close"].isna().tolist() == [
        False,
        True,
    ]
    assert pd.isna(schema.loc[1, "technical_0"])


def test_chronological_train_test_split_is_deterministic_and_stable():
    starts = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    candidates = pd.DataFrame(
        {
            "event_start": starts[::-1],
            "news_count": range(10),
            "mean_sentiment_score": range(10),
        }
    )

    development, holdout, manifest = chronological_train_test_split(candidates)

    assert development["event_start"].tolist() == list(starts[:8])
    assert holdout["event_start"].tolist() == list(starts[8:])
    assert manifest.columns.tolist() == [
        "event_start",
        "partition",
        "holdout_boundary",
    ]
    assert manifest["partition"].value_counts().to_dict() == {
        "development": 8,
        "holdout": 2,
    }
    assert manifest["holdout_boundary"].nunique() == 1
    assert manifest["holdout_boundary"].iloc[0] == starts[8]


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_chronological_train_test_split_rejects_invalid_fraction(test_size):
    candidates = pd.DataFrame(
        {"event_start": pd.date_range("2026-01-01", periods=10, tz="UTC")}
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        chronological_train_test_split(candidates, test_size=test_size)


def test_chronological_train_test_split_rejects_duplicate_starts():
    event_start = pd.Timestamp("2026-01-01", tz="UTC")
    candidates = pd.DataFrame({"event_start": [event_start, event_start]})

    with pytest.raises(ValueError, match="unique"):
        chronological_train_test_split(candidates)


def test_chronological_train_test_split_requires_two_nonempty_partitions():
    candidates = pd.DataFrame(
        {"event_start": [pd.Timestamp("2026-01-01", tz="UTC")]}
    )

    with pytest.raises(ValueError, match="both partitions non-empty"):
        chronological_train_test_split(candidates)
