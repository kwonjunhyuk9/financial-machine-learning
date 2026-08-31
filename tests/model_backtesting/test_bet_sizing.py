import pandas as pd

from src.model_backtesting.bet_sizing import (
    bet_size,
    discretize_signal,
    get_signal,
    get_target_position,
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
