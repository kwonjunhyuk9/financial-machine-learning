import pandas as pd
import pytest

from src.preprocessing import market_technical_indicators
from src.preprocessing.market_technical_indicators import (
    EXCLUDED_TECHNICAL_FEATURES,
    MODEL_FEATURES,
    TECHNICAL_FEATURES,
    require_features,
)


@pytest.mark.parametrize(
    ("source_name", "expected_name"),
    [
        (
            "aapl_dollar_bar_2025-01-01_2025-12-31.parquet",
            "aapl_dollar_bar_technical_2025-01-01_2025-12-31.parquet",
        ),
        (
            "aapl_tick_bar_2025-01-01_2025-12-31.parquet",
            "aapl_tick_bar_technical_2025-01-01_2025-12-31.parquet",
        ),
        (
            "aapl_volume_bar_2025-01-01_2025-12-31.parquet",
            "aapl_volume_bar_technical_2025-01-01_2025-12-31.parquet",
        ),
        ("apple_dollar_bar.parquet", "apple_dollar_bar_technical.parquet"),
    ],
)
def test_build_output_path_preserves_source_bar_name(
        tmp_path,
        source_name,
        expected_name,
):
    path = market_technical_indicators._build_output_path(
        tmp_path / source_name
    )

    assert path == tmp_path / expected_name


def test_save_market_technical_indicators_writes_identifier_and_feature_columns(
        monkeypatch,
        tmp_path,
):
    source = tmp_path / "aapl_dollar_bar_2025-01-01_2025-12-31.parquet"
    dollar_bars = pd.DataFrame(
        {
            "start": pd.to_datetime(
                [
                    "2025-01-02T14:30:00Z",
                    "2025-01-02T14:30:01Z",
                    "2025-01-02T14:30:02Z",
                    "2025-01-02T14:30:03Z",
                ]
            ),
            "end": pd.to_datetime(
                [
                    "2025-01-02T14:30:01Z",
                    "2025-01-02T14:30:02Z",
                    "2025-01-02T14:30:03Z",
                    "2025-01-02T14:30:04Z",
                ]
            ),
            "symbol": ["AAPL"] * 4,
            "open": [100.0, 101.0, 102.0, 101.0],
            "high": [102.0, 102.0, 102.0, 101.0],
            "low": [99.0, 100.0, 102.0, 99.0],
            "close": [101.0, 102.0, 102.0, 100.0],
            "volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0],
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
            data = {
                feature: [float(index)] * 4
                for index, feature in enumerate(TECHNICAL_FEATURES)
            }
            data.update(
                {
                    "On-Balance Volume": [-999.0] * 4,
                    "Accumulation/Distribution Line": [-998.0] * 4,
                    "Chaikin Oscillator": [-997.0] * 4,
                    **{
                        feature: [1.0] * 4
                        for feature in EXCLUDED_TECHNICAL_FEATURES
                    },
                }
            )
            return pd.DataFrame(data)

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
    assert saved_path.name == (
        "aapl_dollar_bar_technical_2025-01-01_2025-12-31.parquet"
    )
    assert features.columns.tolist() == [
        "start",
        "end",
        "symbol",
        *TECHNICAL_FEATURES,
    ]
    assert features.shape[1] == 51
    assert features["symbol"].tolist() == ["AAPL"] * 4
    assert features["On-Balance Volume"].tolist() == [-999.0] * 4
    assert features["Accumulation/Distribution Line"].tolist() == [-998.0] * 4
    assert features["Chaikin Oscillator"].tolist() == [-997.0] * 4
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


def test_feature_names_not_only_count_are_enforced():
    assert len(TECHNICAL_FEATURES) == 48 and len(MODEL_FEATURES) == 50
    wrong = list(MODEL_FEATURES)
    wrong[-1] = "wrong_feature"
    with pytest.raises(ValueError, match="schema mismatch"):
        require_features(wrong)
