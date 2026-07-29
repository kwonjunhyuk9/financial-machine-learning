from src.data_preprocessing.market_differentiated_bars import get_weights, get_weights_fixed_width


def test_get_weights_retains_current_observation_weight():
    weights = get_weights(d=0.5, size=4)

    assert weights.shape == (4, 1)
    assert weights[-1, 0] == 1.0


def test_fixed_width_weights_stop_at_threshold():
    weights = get_weights_fixed_width(d=0.5, thres=0.1)

    assert weights[-1, 0] == 1.0
    assert len(weights) > 1
