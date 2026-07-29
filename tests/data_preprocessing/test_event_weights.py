import pandas as pd

from src.data_preprocessing.event_weights import apply_time_decay, build_indicator_matrix


def test_build_indicator_matrix_marks_active_events():
    matrix = build_indicator_matrix(range(4), pd.Series({0: 1, 1: 3}))

    assert matrix.to_numpy().tolist() == [[1, 0], [1, 1], [0, 1], [0, 1]]


def test_time_decay_scales_oldest_weight_to_requested_level():
    weights = apply_time_decay(pd.Series([1.0, 2.0]), clf_last_w=0.5)

    assert weights.iloc[-1] == 1.0
    assert weights.iloc[0] == 2 / 3
