import pandas as pd
import pytest

import src.data_preprocessing.event_labeling as event_labeling
from src.data_preprocessing.event_labeling import (
    build_labeled_event_data,
    drop_labels,
    get_bins,
    get_events,
    get_vertical_barriers,
)


def test_get_vertical_barriers_selects_future_bars():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    close_prices = pd.Series(range(4), index=index)

    barriers = get_vertical_barriers(
        event_times=index[:2],
        close_prices=close_prices,
        num_bars=2,
    )

    assert barriers.to_dict() == {index[0]: index[2], index[1]: index[3]}


def test_get_events_selects_earliest_barrier_without_event_sides():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    close_prices = pd.Series([100.0, 102.0, 99.0, 101.0], index=index)
    target_returns = pd.Series(0.01, index=index[:2])
    vertical_barriers = pd.Series({index[0]: index[3], index[1]: index[3]})

    events = get_events(
        close_prices=close_prices,
        event_times=index[:2],
        barrier_multipliers=[1.0, 1.0],
        target_returns=target_returns,
        minimum_target_return=0.0,
        num_threads=1,
        vertical_barriers=vertical_barriers,
    )

    assert events.columns.tolist() == ["event_end", "target_return"]
    assert events.loc[index[0], "event_end"] == index[1]
    assert events.loc[index[1], "event_end"] == index[2]


def test_get_bins_and_drop_labels_create_direction_labels():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    close_prices = pd.Series([100.0, 110.0, 90.0, 100.0], index=index)
    events = pd.DataFrame(
        {"event_end": [index[1], index[2], index[3]]},
        index=index[:3],
    )

    labels = get_bins(event_table=events, close_prices=close_prices)
    filtered = drop_labels(
        pd.DataFrame({"label": [1, 1, 0, -1]}),
        minimum_frequency=0.3,
    )

    assert labels["label"].tolist() == [1.0, -1.0, 1.0]
    assert 0 not in filtered["label"].tolist()


def test_get_bins_creates_binary_meta_labels_with_event_sides():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    close_prices = pd.Series([100.0, 110.0, 90.0], index=index)
    events = pd.DataFrame(
        {
            "event_end": [index[1], index[2]],
            "event_side": [1.0, -1.0],
        },
        index=index[:2],
    )

    labels = get_bins(event_table=events, close_prices=close_prices)

    assert labels["realized_return"].tolist() == pytest.approx([0.1, 2 / 11])
    assert labels["label"].tolist() == [1.0, 1.0]


def test_build_labeled_event_data_preserves_missing_features(monkeypatch):
    starts = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    technical_columns = [f"technical_{index}" for index in range(51)]
    candidate_split = pd.DataFrame(
        {
            "event_start": starts,
            "symbol": "AAPL",
            "partition": ["development"] * 8 + ["holdout"] * 2,
            "holdout_boundary": starts[8],
            "news_count": 1,
            "mean_sentiment_score": 0.1,
            "fractionally_differenced_log_close": 0.2,
            **{column: 1.0 for column in technical_columns},
        }
    )
    candidate_split.loc[3, "technical_0"] = float("nan")
    dollar_bars = pd.DataFrame({"end": starts, "close": range(100, 110)})

    monkeypatch.setattr(
        event_labeling,
        "get_daily_volatility",
        lambda close_prices, span: pd.Series(0.01, index=close_prices.index),
    )
    monkeypatch.setattr(
        event_labeling,
        "get_vertical_barriers",
        lambda event_times, close_prices, num_bars: pd.Series(
            pd.DatetimeIndex(event_times) + pd.Timedelta(minutes=30),
            index=event_times,
        ),
    )

    def fake_get_events(**kwargs):
        event_times = pd.DatetimeIndex(kwargs["event_times"])
        return pd.DataFrame(
            {
                "event_end": event_times + pd.Timedelta(minutes=30),
                "target_return": 0.01,
            },
            index=event_times,
        )

    def fake_get_bins(event_table, close_prices):
        return pd.DataFrame(
            {
                "realized_return": [0.01, -0.01] * 5,
                "label": [1.0, -1.0] * 5,
            },
            index=event_table.index,
        )

    monkeypatch.setattr(event_labeling, "get_events", fake_get_events)
    monkeypatch.setattr(event_labeling, "get_bins", fake_get_bins)

    model_data, manifest = build_labeled_event_data(
        candidate_split,
        dollar_bars,
    )

    assert model_data.shape == (10, 61)
    assert pd.isna(
        model_data.loc[model_data["event_start"].eq(starts[3]), "technical_0"]
    ).item()
    assert manifest["partition"].value_counts().to_dict() == {
        "development": 8,
        "holdout": 2,
    }
