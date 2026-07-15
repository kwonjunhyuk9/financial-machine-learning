import pandas as pd

from src.model_backtesting.bet_sizing import bet_size, discretize_signal, get_target_position


def test_discretize_signal_rounds_and_clips_positions():
    signal = pd.Series([-1.2, -0.2, 0.3, 1.4])

    assert discretize_signal(signal, step_size=0.25).tolist() == [-1.0, -0.25, 0.25, 1.0]


def test_dynamic_bet_size_and_position_follow_forecast_direction():
    size = bet_size(w=4.0, price_divergence=2.0)

    assert 0 < size < 1
    assert get_target_position(4.0, 102.0, 100.0, 10) > 0
