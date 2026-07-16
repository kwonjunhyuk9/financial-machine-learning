import pandas as pd

from src.data_preprocessing.fundamental_ratio_features import (
    collect_fundamental_ratio_features,
)


class FakeRatios:
    def __init__(self):
        self.calls: list[tuple[str, int | None]] = []
        self.data = pd.DataFrame(
            {"2025Q1": [1.0]},
            index=pd.Index(["AAPL"], name="ticker"),
        )

    def __getattr__(self, name):
        def collect(*, trailing):
            self.calls.append((name, trailing))
            return self.data

        return collect


class FakeToolkit:
    def __init__(self):
        self.ratios = FakeRatios()


def test_collect_fundamental_ratio_features_combines_selected_ratios():
    toolkit = FakeToolkit()

    features = collect_fundamental_ratio_features(toolkit, trailing=4)

    assert features.loc[0, "ticker"] == "AAPL"
    assert features.loc[0, "price_to_book"] == 1.0
    assert features.loc[0, "market_cap"] == 1.0
    assert len(toolkit.ratios.calls) == 8
    assert {trailing for _, trailing in toolkit.ratios.calls} == {4}
