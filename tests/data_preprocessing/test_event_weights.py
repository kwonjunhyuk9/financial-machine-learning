import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing.event_weights import (
    WEIGHT_COLUMNS,
    apply_time_decay,
    build_indicator_matrix,
    build_partitioned_event_weights,
)


def test_build_indicator_matrix_marks_active_events():
    matrix = build_indicator_matrix(range(4), pd.Series({0: 1, 1: 3}))

    assert matrix.to_numpy().tolist() == [[1, 0], [1, 1], [0, 1], [0, 1]]


def test_time_decay_scales_oldest_weight_to_requested_level():
    weights = apply_time_decay(pd.Series([1.0, 2.0]), clf_last_w=0.5)

    assert weights.iloc[-1] == 1.0
    assert weights.iloc[0] == 2 / 3


def test_partitioned_event_weights_are_complete_and_normalized():
    starts = pd.date_range("2025-01-02", periods=6, freq="2h", tz="UTC")
    events = pd.DataFrame(
        {
            "event_start": starts,
            "event_end": starts + pd.Timedelta(hours=1),
            "direction_label": np.tile([-1, 1], 3),
        }
    )
    manifest = pd.DataFrame(
        {
            "event_start": starts,
            "partition": ["development"] * 3 + ["holdout"] * 3,
        }
    )
    close_index = pd.date_range(
        starts.min(),
        starts.max() + pd.Timedelta(hours=1),
        freq="h",
    )
    close = pd.Series(np.linspace(100.0, 112.0, len(close_index)), index=close_index)

    weighted = build_partitioned_event_weights(events, manifest, close)

    assert weighted.columns[-4:].tolist() == WEIGHT_COLUMNS
    assert np.isfinite(weighted[WEIGHT_COLUMNS]).all().all()
    for partition in ["development", "holdout"]:
        partition_starts = manifest.loc[
            manifest["partition"].eq(partition), "event_start"
        ]
        partition_weights = weighted.loc[
            weighted["event_start"].isin(partition_starts), "sample_weight"
        ]
        assert partition_weights.sum() == pytest.approx(len(partition_starts))
