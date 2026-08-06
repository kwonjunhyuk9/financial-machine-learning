import pandas as pd
import pytest

from src.data_preprocessing import market_technical_indicators


class FakeTechnicals:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.data = pd.DataFrame(
            {
                ("Relative Strength Index", "AAPL"): [51.0],
                ("Average True Range", "AAPL"): [1.5],
            },
            index=pd.Index(["2025-01-01"], name="date"),
        )

    def collect_all_indicators(self, **kwargs):
        self.calls.append(kwargs)
        return self.data


class FakeToolkit:
    def __init__(self):
        self.technicals = FakeTechnicals()


def test_collect_market_technical_indicators_collects_all_indicators():
    toolkit = FakeToolkit()

    features = market_technical_indicators.collect_market_technical_indicators(
        toolkit,
        period="weekly",
        close_column="Close",
        window=10,
    )

    pd.testing.assert_frame_equal(features, toolkit.technicals.data)
    assert toolkit.technicals.calls == [
        {
            "period": "weekly",
            "close_column": "Close",
            "window": 10,
        }
    ]


def test_build_output_path_replaces_dollar_bar_name(tmp_path):
    path = market_technical_indicators._build_output_path(
        tmp_path / "aapl_dollar_bar_2025-01-01_2025-12-31.parquet"
    )

    assert path == (
        tmp_path / "aapl_technical_indicators_2025-01-01_2025-12-31.parquet"
    )


def test_save_market_technical_indicators_writes_identifier_and_feature_columns(
        monkeypatch,
        tmp_path,
):
    source = tmp_path / "aapl_dollar_bar_2025-01-01_2025-12-31.parquet"
    dollar_bars = pd.DataFrame(
        {
            "start": pd.to_datetime(
                ["2025-01-02T14:30:00Z", "2025-01-02T14:30:01Z"]
            ),
            "end": pd.to_datetime(
                ["2025-01-02T14:30:01Z", "2025-01-02T14:30:02Z"]
            ),
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000.0, 1_100.0],
        }
    )
    dollar_bars.to_parquet(source, index=False)
    technical_arguments = {}
    indicator_calls = []

    class FakeTechnicalsController:
        def __init__(self, **kwargs):
            technical_arguments.update(kwargs)

        def collect_all_indicators(self, **kwargs):
            indicator_calls.append(kwargs)
            return pd.DataFrame(
                {
                    "Relative Strength Index": [None, 55.0],
                    "Average True Range": [None, 2.0],
                }
            )

    monkeypatch.setattr(
        market_technical_indicators,
        "Technicals",
        FakeTechnicalsController,
    )

    saved_path = market_technical_indicators.save_market_technical_indicators(
        data_path=source,
        window=10,
    )

    features = pd.read_parquet(saved_path)
    assert saved_path.name == "aapl_technical_indicators_2025-01-01_2025-12-31.parquet"
    assert features.columns.tolist() == [
        "start",
        "end",
        "symbol",
        "Relative Strength Index",
        "Average True Range",
    ]
    assert features["symbol"].tolist() == ["AAPL", "AAPL"]
    historical_data = technical_arguments["historical_data"]["daily"]
    assert historical_data.columns.tolist() == [
        ("Open", "AAPL"),
        ("High", "AAPL"),
        ("Low", "AAPL"),
        ("Close", "AAPL"),
        ("Adj Close", "AAPL"),
        ("Volume", "AAPL"),
        ("Return", "AAPL"),
        ("Cumulative Return", "AAPL"),
    ]
    assert indicator_calls == [
        {
            "period": "daily",
            "close_column": "Adj Close",
            "window": 10,
        }
    ]


def test_save_market_technical_indicators_rejects_multiple_symbols(tmp_path):
    source = tmp_path / "mixed_dollar_bar_2025.parquet"
    pd.DataFrame(
        {
            "start": ["2025-01-01", "2025-01-01"],
            "end": ["2025-01-02", "2025-01-02"],
            "symbol": ["AAPL", "MSFT"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    ).to_parquet(source, index=False)

    with pytest.raises(ValueError, match="exactly one symbol"):
        market_technical_indicators.save_market_technical_indicators(data_path=source)
