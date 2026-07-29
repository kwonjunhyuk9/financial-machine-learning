import pandas as pd

from src.data_preprocessing.event_labeling import drop_labels, get_bins, get_vertical_barriers


def test_get_vertical_barriers_selects_future_bars():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    barriers = get_vertical_barriers(index[:2], pd.Series(range(4), index=index), num_bars=2)

    assert barriers.to_dict() == {index[0]: index[2], index[1]: index[3]}


def test_get_bins_and_drop_labels_create_stable_binary_labels():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    close = pd.Series([100.0, 110.0, 90.0, 100.0], index=index)
    events = pd.DataFrame({"t1": [index[1], index[2], index[3]]}, index=index[:3])

    labels = get_bins(events, close)
    filtered = drop_labels(pd.DataFrame({"bin": [1, 1, 0, -1]}), min_pct=0.3)

    assert labels["bin"].tolist() == [1.0, -1.0, 1.0]
    assert 0 not in filtered["bin"].tolist()
