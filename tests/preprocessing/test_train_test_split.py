import json
from pathlib import Path

import pandas as pd
import pytest
from pyarrow import ArrowInvalid

from src.preprocessing.train_test_split import (
    build_event_candidates,
    build_event_feature_schema,
    build_partitioned_candidates,
    chronological_train_test_split,
    load_existing_holdout_boundary,
)


def test_load_existing_holdout_boundary_returns_none_for_missing_file(tmp_path):
    assert load_existing_holdout_boundary(tmp_path / "candidates.parquet") is None


def test_load_existing_holdout_boundary_restores_saved_timestamp(tmp_path):
    path = tmp_path / "candidates.parquet"
    boundary = pd.Timestamp("2025-10-01", tz="UTC")
    pd.DataFrame({"holdout_boundary": [boundary, boundary]}).to_parquet(path)

    assert load_existing_holdout_boundary(path) == boundary


@pytest.mark.parametrize(
    "boundaries",
    [
        [],
        [pd.NaT],
        [pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC")],
    ],
)
def test_load_existing_holdout_boundary_rejects_nonunique_boundary(tmp_path, boundaries):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame({"holdout_boundary": boundaries}).to_parquet(path)

    with pytest.raises(ValueError, match="one immutable boundary"):
        load_existing_holdout_boundary(path)


def test_load_existing_holdout_boundary_propagates_read_error(tmp_path):
    path = tmp_path / "candidates.parquet"
    path.write_bytes(b"not a parquet file")

    with pytest.raises(ArrowInvalid):
        load_existing_holdout_boundary(path)


@pytest.mark.parametrize("restore_boundary", [False, True])
def test_build_partitioned_candidates_matches_notebook_assembly(tmp_path, restore_boundary):
    starts = pd.date_range("2025-02-01", periods=5, freq="D", tz="UTC")
    features = ["mean_sentiment_score", "fractionally_differenced_log_close", "Relative Strength Index"]
    candidates = pd.DataFrame({
        "symbol": ["A", "B"] * len(starts),
        "event_start": starts.repeat(2),
        **{feature: [float(i) for i in range(10)] for feature in features},
    }).iloc[::-1].reset_index(drop=True)
    candidates.loc[0, "mean_sentiment_score"] = float("nan")
    original = candidates.copy(deep=True)
    boundary = None
    if restore_boundary:
        path = tmp_path / "candidates.parquet"
        pd.DataFrame({"holdout_boundary": [starts[3]]}).to_parquet(path)
        boundary = load_existing_holdout_boundary(path)

    expected_development, expected_holdout, manifest = chronological_train_test_split(
        candidates, test_size=0.20, holdout_boundary=boundary,
    )
    expected = candidates.merge(
        manifest, on=["symbol", "event_start"], how="left", validate="one_to_one",
    )


    development, holdout, partitioned = build_partitioned_candidates(
        candidates, test_size=0.20, holdout_boundary=boundary,
    )

    pd.testing.assert_frame_equal(development, expected_development)
    pd.testing.assert_frame_equal(holdout, expected_holdout)
    pd.testing.assert_frame_equal(
        partitioned.sort_index(axis=1), expected.sort_index(axis=1),
    )
    pd.testing.assert_frame_equal(candidates, original)


def test_build_event_candidates_aligns_and_aggregates_news():
    bar_ends = pd.to_datetime(
        ["2026-01-01 14:30:00Z", "2026-01-01 14:31:00Z"]
    )
    news = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                [
                    "2026-01-01 14:29:30Z",
                    "2026-01-01 14:30:01Z",
                ]
            ),
            "sentiment_score": [0.2, -0.4],
        }
    )

    candidates = build_event_candidates(news, bar_ends)

    assert set(candidates.columns) == {
        "event_start",
        "mean_sentiment_score",
    }
    assert candidates["event_start"].tolist() == list(bar_ends)
    assert candidates["mean_sentiment_score"].tolist() == pytest.approx([0.2, -0.4])


def test_build_event_candidates_keeps_distinct_news_at_the_same_time():
    bar_ends = pd.to_datetime(["2026-01-01 14:30:00Z"])
    news = pd.DataFrame({
        "created_at": pd.to_datetime(["2026-01-01 14:29:30Z"] * 2),
        "sentiment_score": [0.2, 0.8],
    })

    candidates = build_event_candidates(news, bar_ends)

    assert candidates.mean_sentiment_score.tolist() == pytest.approx([0.5])


def test_build_event_feature_schema_preserves_rows_and_missing_values():
    event_starts = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    candidates = pd.DataFrame(
        {
            "event_start": event_starts,
            "mean_sentiment_score": [0.2, -0.4],
        }
    )
    fractional = pd.DataFrame(
        {
            "end": event_starts[:1],
            "fractionally_differenced_log_close": [0.1],
        }
    )
    technical_columns = ["Relative Strength Index"]
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
    technical.loc[1, "Relative Strength Index"] = float("nan")

    schema = build_event_feature_schema(candidates, fractional, technical)

    assert len(schema) == 2
    assert schema["event_start"].tolist() == list(event_starts)
    assert schema["fractionally_differenced_log_close"].isna().tolist() == [
        False,
        True,
    ]
    assert pd.isna(schema.loc[1, "Relative Strength Index"])


def test_chronological_train_test_split_is_deterministic_and_stable():
    starts = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    candidates = pd.DataFrame(
        {
            "symbol": "AAPL",
            "event_start": starts[::-1],
            "mean_sentiment_score": range(10),
        }
    )

    development, holdout, manifest = chronological_train_test_split(candidates, test_size=0.20, holdout_boundary=None)

    assert development["event_start"].tolist() == list(starts[:8])
    assert holdout["event_start"].tolist() == list(starts[8:])
    assert set(manifest.columns) == {
        "symbol",
        "event_start",
        "holdout_boundary",
        "partition",
    }
    assert manifest["partition"].value_counts().to_dict() == {
        "development": 8,
        "holdout": 2,
    }
    assert manifest["holdout_boundary"].nunique() == 1
    assert manifest["holdout_boundary"].iloc[0] == starts[8]


def test_chronological_train_test_split_uses_explicit_boundary_between_events():
    starts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    candidates = pd.DataFrame({"symbol": "AAPL", "event_start": starts})
    boundary = starts[1] + pd.Timedelta(minutes=30)

    development, holdout, manifest = chronological_train_test_split(
        candidates,
        holdout_boundary=boundary,
        test_size=0.20,
    )

    assert development["event_start"].tolist() == list(starts[:2])
    assert holdout["event_start"].tolist() == list(starts[2:])
    assert manifest["holdout_boundary"].eq(boundary).all()


def test_split_notebook_uses_calendar_boundary_instead_of_stored_boundary(tmp_path, monkeypatch):
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks/preprocessing/train_test_split.ipynb"
    notebook = json.loads(notebook_path.read_text())
    directory = tmp_path / "notebooks/preprocessing"
    directory.mkdir(parents=True)
    monkeypatch.chdir(directory)
    namespace = {}
    exec("".join(notebook["cells"][2]["source"]), namespace)
    path = namespace["candidate_path"]
    path.parent.mkdir(parents=True)
    pd.DataFrame({"holdout_boundary": [pd.Timestamp("2025-10-01", tz="UTC")]}).to_parquet(path)
    starts = pd.to_datetime([
        "2025-02-01T00:00:00Z", "2025-12-31T23:59:59Z",
        "2026-01-01T00:00:00Z", "2026-06-30T23:59:59Z",
    ])
    namespace["candidate_schema"] = pd.DataFrame({"symbol": "AAPL", "event_start": starts})

    for index in (5, 6):
        exec("".join(notebook["cells"][index]["source"]), namespace)

    assert namespace["paths"].start == pd.Timestamp("2025-02-01", tz="UTC")
    assert namespace["paths"].end == pd.Timestamp("2026-07-01", tz="UTC")
    assert namespace["development"].event_start.tolist() == list(starts[:2])
    assert namespace["holdout"].event_start.tolist() == list(starts[2:])
    saved = pd.read_parquet(path)
    assert saved.holdout_boundary.eq(pd.Timestamp("2026-01-01", tz="UTC")).all()


@pytest.mark.parametrize(
    "boundary",
    [
        pd.Timestamp("2025-12-31", tz="UTC"),
        pd.Timestamp("2026-01-03", tz="UTC"),
    ],
)
def test_chronological_train_test_split_rejects_empty_explicit_partition(boundary):
    candidates = pd.DataFrame(
        {"symbol": "AAPL", "event_start": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")}
    )

    with pytest.raises(ValueError, match="both partitions non-empty"):
        chronological_train_test_split(
            candidates,
            holdout_boundary=boundary,
            test_size=0.20,
        )


def test_chronological_train_test_split_rejects_invalid_explicit_boundary():
    candidates = pd.DataFrame(
        {"symbol": "AAPL", "event_start": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")}
    )

    with pytest.raises(ValueError, match="valid timestamp"):
        chronological_train_test_split(
            candidates,
            holdout_boundary=pd.NaT,
            test_size=0.20,
        )


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_chronological_train_test_split_rejects_invalid_fraction(test_size):
    candidates = pd.DataFrame(
        {"symbol": "AAPL", "event_start": pd.date_range("2026-01-01", periods=10, tz="UTC")}
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        chronological_train_test_split(candidates, test_size=test_size, holdout_boundary=None)


def test_chronological_train_test_split_rejects_duplicate_starts():
    event_start = pd.Timestamp("2026-01-01", tz="UTC")
    candidates = pd.DataFrame({"symbol": "AAPL", "event_start": [event_start, event_start]})

    with pytest.raises(ValueError, match="unique"):
        chronological_train_test_split(candidates, test_size=0.20, holdout_boundary=None)


def test_chronological_train_test_split_requires_two_nonempty_partitions():
    candidates = pd.DataFrame(
        {"symbol": "AAPL", "event_start": [pd.Timestamp("2026-01-01", tz="UTC")]}
    )

    with pytest.raises(ValueError, match="both partitions non-empty"):
        chronological_train_test_split(candidates, test_size=0.20, holdout_boundary=None)


def test_shared_holdout_time_is_frozen_and_same_time_is_not_split():
    starts = pd.date_range("2025-01-02", periods=12, freq="h", tz="UTC")
    events = pd.DataFrame([
        {"symbol": symbol, "event_start": time,
         "event_end": time + pd.Timedelta(minutes=70)}
        for time in starts for symbol in ["A", "B"]
    ])
    development, holdout, manifest = chronological_train_test_split(events, test_size=0.20, holdout_boundary=None)
    assert not set(development.event_start) & set(holdout.event_start)
    assert not manifest.duplicated(["symbol", "event_start"]).any()
    boundary = manifest.holdout_boundary.iloc[0]
    _, _, reduced = chronological_train_test_split(events.iloc[1:], holdout_boundary=boundary, test_size=0.20)
    assert reduced.holdout_boundary.eq(boundary).all()


def test_chronological_train_test_split_requires_symbol():
    candidates = pd.DataFrame({"event_start": pd.date_range("2026-01-01", periods=4, tz="UTC")})
    with pytest.raises(ValueError, match="symbol and event_start"):
        chronological_train_test_split(candidates, test_size=0.20, holdout_boundary=None)
