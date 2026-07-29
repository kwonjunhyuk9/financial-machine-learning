import pandas as pd

from src.data_preprocessing.market_technical_indicators import (
    collect_market_technical_indicators,
)


class FakeTechnicals:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.data = pd.DataFrame(
            {"AAPL": [1.0]},
            index=pd.Index(["2025-01-01"], name="date"),
        )

    def get_moving_average_convergence_divergence(self, **kwargs):
        self.calls.append(("macd", kwargs))
        return self.data, self.data

    def __getattr__(self, name):
        def collect(**kwargs):
            self.calls.append((name, kwargs))
            return self.data

        return collect


class FakeToolkit:
    def __init__(self):
        self.technicals = FakeTechnicals()


def test_collect_market_technical_indicators_combines_selected_indicators():
    toolkit = FakeToolkit()

    features = collect_market_technical_indicators(
        toolkit,
        period="weekly",
        close_column="Close",
        window=10,
    )

    assert features.loc[0, "relative_strength_index"] == 1.0
    assert features.loc[0, "macd"] == 1.0
    assert features.loc[0, "average_true_range"] == 1.0
    calls = dict(toolkit.technicals.calls)
    assert calls["get_relative_strength_index"]["window"] == 10
    assert calls["get_average_true_range"]["close_column"] == "Close"
