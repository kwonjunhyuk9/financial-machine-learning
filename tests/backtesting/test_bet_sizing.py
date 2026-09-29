import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.backtesting.bet_sizing import (
    average_symbol_targets,
    average_active_signals,
    average_active_signals_for_times,
    bet_size,
    build_target_positions,
    discretize_signal,
    get_signal,
    get_target_position,
    event_bet_signals,
)


def test_get_signal_is_bounded_and_respects_pass_decision():
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    events = pd.DataFrame({"t1": index + pd.Timedelta(hours=1)}, index=index)
    probability = pd.Series([0.50, 0.60, 0.90], index=index)
    action = pd.Series([1, 0, 1], index=index)

    sizes = get_signal(events, 0.1, probability, action, num_classes=2).loc[index]

    assert sizes.between(0.0, 1.0).all()
    assert sizes.iloc[0] == 0.0
    assert sizes.iloc[1] == 0.0
    assert sizes.iloc[2] > 0.0


def test_discretize_signal_rounds_and_clips_positions():
    signal = pd.Series([-1.2, -0.2, 0.3, 1.4])

    assert discretize_signal(signal, step_size=0.25).tolist() == [-1.0, -0.25, 0.25, 1.0]


def test_dynamic_bet_size_and_position_follow_forecast_direction():
    size = bet_size(w=4.0, price_divergence=2.0)

    assert 0 < size < 1
    assert get_target_position(4.0, 102.0, 100.0, 10) > 0


def test_signed_overlap_and_zero_meta_signals():
    times = pd.date_range("2025-01-01", periods=5, tz="UTC")
    events = pd.DataFrame({
        "partition": ["development", "holdout", "holdout", "holdout"],
        "primary_side": [1, 1, -1, 1],
        "meta_action": [1, 1, 0, 1],
        "meta_probability": [1.0, 1.0, 0.1, 1.0],
        "event_end": [times[4], times[3], times[3], times[4]],
    }, index=times[:4])

    targets = build_target_positions(events)

    assert targets.index[0] == times[1]
    assert targets.primary_only.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert targets.meta_filtered.tolist() == [1.0, 0.5, 1.0, 0.0]


def test_same_side_overlap_is_averaged_not_summed():
    times = pd.date_range("2025-01-01", periods=4, tz="UTC")
    events = pd.DataFrame({
        "partition": ["holdout", "holdout"], "primary_side": [1, 1],
        "meta_action": [1, 1], "meta_probability": [1.0, 1.0],
        "event_end": [times[2], times[3]],
    }, index=times[:2])
    targets = build_target_positions(events)
    assert targets.primary_only.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_signal_interfaces_agree_with_direction_and_probability_boundaries():
    times = pd.date_range("2025-01-01", periods=4, tz="UTC")
    events = pd.DataFrame({
        "t1": times[1:], "side": [1, 1, -1],
    }, index=times[:3])
    probabilities = pd.Series([0.0, 0.5, 1.0], index=events.index)
    actions = pd.Series([0, 1, 1], index=events.index)
    signals = pd.DataFrame({"signal": [0.0, 0.0, -1.0], "t1": times[1:]}, index=events.index)
    averaged = average_active_signals(signals)
    explicit = average_active_signals_for_times(signals, times)
    pd.testing.assert_series_equal(averaged, explicit)
    result = get_signal(events, 0.1, probabilities, actions, 2)
    pd.testing.assert_series_equal(result, discretize_signal(averaged, 0.1))
    assert np.isfinite(result).all()


@pytest.mark.parametrize("field,value", [
    ("primary_side", 0), ("meta_action", 2), ("meta_probability", np.nan),
])
def test_build_targets_retains_input_validation(field, value):
    times = pd.date_range("2025-01-01", periods=2, tz="UTC")
    events = pd.DataFrame({
        "partition": ["holdout"], "primary_side": [1], "meta_action": [1],
        "meta_probability": [0.8], "event_end": [times[1]],
    }, index=times[:1])
    events[field] = value
    with pytest.raises(ValueError):
        build_target_positions(events)


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
