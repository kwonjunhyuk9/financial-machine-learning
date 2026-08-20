import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing.event_weights import (
    WEIGHT_COLUMNS,
    build_partitioned_event_weights,
)
from src.data_preprocessing.prepare_the_data import (
    build_tree_preprocessing,
    get_event_feature_groups,
    prepare_weighted_event_data,
)


def _inputs():
    starts = pd.date_range("2025-01-02", periods=8, freq="2h", tz="UTC")
    technical = {
        f"technical_{index}": np.linspace(index, index + 1, len(starts))
        for index in range(51)
    }
    events = pd.DataFrame(
        {
            "event_start": starts,
            "symbol": "AAPL",
            "event_end": starts + pd.Timedelta(hours=1),
            "vertical_barrier": starts + pd.Timedelta(hours=2),
            "target_return": 0.01,
            "raw_return": np.linspace(-0.02, 0.03, len(starts)),
            "direction_label": np.tile([-1, 1], 4),
            "mean_sentiment_score": np.linspace(-0.8, 0.8, len(starts)),
            "fractionally_differenced_log_close": np.linspace(1.3, 1.5, len(starts)),
            **technical,
        }
    )
    boundary = starts[4]
    manifest = pd.DataFrame(
        {
            "event_start": starts,
            "event_end": events["event_end"],
            "partition": ["development"] * 4 + ["holdout"] * 4,
            "holdout_boundary": boundary,
        }
    )
    close_index = pd.date_range(
        starts.min(),
        starts.max() + pd.Timedelta(hours=1),
        freq="h",
    )
    close = pd.Series(np.linspace(100.0, 116.0, len(close_index)), index=close_index)
    weighted = build_partitioned_event_weights(events, manifest, close)
    return weighted, manifest, close


def test_complete_weighted_data_is_preserved_without_reweighting():
    weighted, manifest, close = _inputs()

    prepared, prepared_manifest, report = prepare_weighted_event_data(
        weighted,
        manifest,
        close,
    )

    pd.testing.assert_frame_equal(prepared, weighted)
    pd.testing.assert_frame_equal(prepared_manifest, manifest)
    assert len(report) == 53
    invalid_values = report[
        ["missing_values", "infinite_values", "invalid_rows"]
    ].sum().sum()
    assert invalid_values == 0


def test_invalid_feature_rows_are_recorded_dropped_and_reweighted():
    weighted, manifest, close = _inputs()
    weighted.loc[0, "mean_sentiment_score"] = np.nan
    weighted.loc[1, "technical_0"] = np.nan
    weighted.loc[4, "fractionally_differenced_log_close"] = np.inf

    prepared, prepared_manifest, report = prepare_weighted_event_data(
        weighted,
        manifest,
        close,
    )

    expected_starts = weighted.loc[2:, "event_start"].drop(index=4).tolist()
    assert prepared["event_start"].tolist() == expected_starts
    assert prepared_manifest.loc[0, "partition"] == "development_feature_missing"
    assert prepared_manifest.loc[1, "partition"] == "development_feature_missing"
    assert prepared_manifest.loc[4, "partition"] == "holdout_feature_missing"
    assert prepared.shape[1] == weighted.shape[1]
    feature_columns = [
        column
        for columns in get_event_feature_groups(prepared).values()
        for column in columns
    ]
    assert np.isfinite(prepared[feature_columns]).all().all()
    assert (
        report.set_index("feature").loc[
            "mean_sentiment_score", "missing_values"
        ]
        == 1
    )
    assert report.set_index("feature").loc["technical_0", "missing_values"] == 1
    assert report.set_index("feature").loc[
        "fractionally_differenced_log_close", "infinite_values"
    ] == 1
    for partition in ["development", "holdout"]:
        partition_starts = prepared_manifest.loc[
            prepared_manifest["partition"].eq(partition), "event_start"
        ]
        assert prepared.loc[
            prepared["event_start"].isin(partition_starts), "sample_weight"
        ].sum() == pytest.approx(len(partition_starts))


def test_tree_preprocessing_is_an_identity_placeholder():
    weighted, _, _ = _inputs()
    feature_groups = get_event_feature_groups(weighted)
    feature_columns = [
        column
        for columns in feature_groups.values()
        for column in columns
    ]
    features = weighted[feature_columns]

    transformed = build_tree_preprocessing(feature_groups).fit_transform(features)

    pd.testing.assert_frame_equal(transformed, features)


def test_prepare_requires_complete_53_feature_schema():
    weighted, manifest, close = _inputs()
    weighted = weighted.drop(columns="technical_50")

    with pytest.raises(ValueError, match="51 technical"):
        prepare_weighted_event_data(weighted, manifest, close)
