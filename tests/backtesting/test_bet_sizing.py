import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.backtesting.bet_sizing import (
    average_symbol_targets,
    bet_size,
    discretize_signal,
    event_bet_signals,
    get_target_position,
)


def test_discretize_signal_rounds_and_clips_positions():
    signal = pd.Series([-1.2, -0.2, 0.3, 1.4])

    assert discretize_signal(signal, step_size=0.25).tolist() == [-1.0, -0.25, 0.25, 1.0]


def test_dynamic_bet_size_and_position_follow_forecast_direction():
    size = bet_size(w=4.0, price_divergence=2.0)

    assert 0 < size < 1
    assert get_target_position(4.0, 102.0, 100.0, 10) > 0


def probability_for_signal(signal):
    z = norm.ppf((signal + 1) / 2)
    return 0.5 + z / (2 * np.sqrt(1 + z * z))


def test_average_is_per_symbol_pass_is_zero_and_discretization_is_last():
    events = pd.DataFrame({"symbol": ["A", "A", "B"], "primary_side": [1, 1, -1],
                           "meta_action": [1, 1, 1],
                           "meta_probability": [probability_for_signal(.8), probability_for_signal(.4), 1.]})
    targets = average_symbol_targets(events)
    assert targets["A"] == pytest.approx(.06)
    assert targets["B"] == pytest.approx(-.10)
    events = pd.concat([events, pd.DataFrame({"symbol": ["A"], "primary_side": [1],
                                             "meta_action": [0], "meta_probability": [.1]})])
    assert average_symbol_targets(events)["A"] == pytest.approx(.04)


def test_opposing_signals_cancel_and_probability_boundaries_are_defined():
    frame = pd.DataFrame({"symbol": ["A", "A", "B"], "primary_side": [1, -1, 1],
                          "meta_action": [1, 1, 1], "meta_probability": [1., 1., .5]})
    assert average_symbol_targets(frame).eq(0).all()
    assert event_bet_signals(frame).tolist() == [1., -1., 0.]
    assert average_symbol_targets(frame.iloc[:0]).empty


def test_meta_pass_reduces_existing_symbol_mean_without_replacing_old_event():
    frame = pd.DataFrame({"symbol": ["A", "A"], "primary_side": [1, 1],
                          "meta_action": [1, 0], "meta_probability": [1., .1]})
    assert average_symbol_targets(frame)["A"] == pytest.approx(.05)
    assert average_symbol_targets(frame.iloc[[1]])["A"] == 0
