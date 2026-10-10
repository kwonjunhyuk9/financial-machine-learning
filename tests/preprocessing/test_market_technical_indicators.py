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

        def get_breadth(self, feature, value, **kwargs):
            indicator_calls.append(kwargs)
            return pd.DataFrame({"AAPL": [value] * 4})

        def get_on_balance_volume(self, **kwargs):
            return self.get_breadth("On-Balance Volume", -999.0, **kwargs)

        def get_accumulation_distribution_line(self, **kwargs):
            return self.get_breadth("Accumulation/Distribution Line", -998.0, **kwargs)

        def get_chaikin_oscillator(self, **kwargs):
            return self.get_breadth("Chaikin Oscillator", -997.0, **kwargs)

        def collect_category(self, features, **kwargs):
            indicator_calls.append(kwargs)
            return pd.DataFrame({feature: [float(TECHNICAL_FEATURES.index(feature))] * 4
                                 for feature in features})

        def collect_momentum_indicators(self, **kwargs):
            return self.collect_category(TECHNICAL_FEATURES[3:25], **kwargs)

        def collect_overlap_indicators(self, **kwargs):
            return self.collect_category(TECHNICAL_FEATURES[25:37], **kwargs)

        def collect_volatility_indicators(self, **kwargs):
            return self.collect_category(TECHNICAL_FEATURES[37:], **kwargs)

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
        {"period": "daily", "close_column": "Adj Close"},
    ] * 3 + [
        {"period": "daily", "close_column": "Adj Close", "window": 10},
    ] * 3



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


@pytest.mark.parametrize("case", ["empty", "single", "up", "down", "flat", "reversals", "missing", "infinite"])
def test_array_sar_matches_financetoolkit(case):
    import numpy as np
    from financetoolkit.technicals.overlap_model import get_parabolic_sar

    size = 0 if case == "empty" else 1 if case == "single" else 200
    values = 100 + np.arange(size, dtype=float) * 0.5
    if case == "down":
        values = values[::-1]
    elif case == "flat":
        values[:] = 100
    elif case in {"reversals", "missing", "infinite"}:
        values = 100 + np.random.default_rng(10).normal(size=size).cumsum()
    index = pd.date_range("2025-01-01", periods=size, freq="min", tz="UTC", name="end")
    high = pd.Series(values + 0.3, index=index)
    low = pd.Series(values - 0.3, index=index)
    if case == "missing":
        high.iloc[[0, 15, 50]] = np.nan
        low.iloc[[20, 70]] = np.nan
    elif case == "infinite":
        high.iloc[15] = np.inf
        low.iloc[20] = -np.inf
    with np.errstate(invalid="ignore"):
        expected = get_parabolic_sar(high, low)
        actual = market_technical_indicators._parabolic_sar(high, low)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("case", ["random", "flat", "reversals"])
def test_saved_48_features_match_native_toolkit_without_excluded_calculations(
    tmp_path, monkeypatch, case,
):
    import numpy as np
    from financetoolkit.technicals.technicals_controller import Technicals
    from unittest.mock import Mock

    size = 300
    rng = np.random.default_rng(9)
    close = 100 + rng.normal(scale=0.4, size=size).cumsum()
    if case == "flat":
        close[:] = 100
    elif case == "reversals":
        close = 100 + 5 * np.sin(np.arange(size) / 8)
    end = pd.date_range("2025-01-01", periods=size, freq="min", tz="UTC")
    bars = pd.DataFrame({"start": end - pd.Timedelta(minutes=1), "end": end,
                         "symbol": "TEST", "open": close - 0.1, "high": close + 0.4,
                         "low": close - 0.4, "close": close,
                         "volume": rng.integers(100, 10000, size).astype(float)})
    source = tmp_path / "bars.parquet"
    bars.to_parquet(source, index=False)
    references = {}
    skipped = []

    def controller(**kwargs):
        native = Technicals(**kwargs)
        expected = native.collect_all_indicators(period="daily", close_column="Adj Close", window=14)
        expected = market_technical_indicators._select_native_technical_features(expected)
        expected.columns = expected.columns.get_level_values(0)
        references["expected"] = expected.loc[:, list(TECHNICAL_FEATURES)].reset_index(drop=True)
        for method in ("get_trin", "get_new_highs_new_lows", "get_advancers_decliners",
                       "get_mcclellan_oscillator", "get_parabolic_sar"):
            fail = Mock(side_effect=AssertionError(f"unexpected native call: {method}"))
            setattr(native, method, fail)
            skipped.append(fail)
        return native

    monkeypatch.setattr(market_technical_indicators, "Technicals", controller)
    destination = market_technical_indicators.save_market_technical_indicators(
        data_path=source, log_saved=False,
    )
    actual = pd.read_parquet(destination)
    pd.testing.assert_frame_equal(actual[list(TECHNICAL_FEATURES)], references["expected"], check_exact=True)
    pd.testing.assert_frame_equal(actual[["start", "end", "symbol"]], bars[["start", "end", "symbol"]])
    assert actual.columns.tolist() == ["start", "end", "symbol", *TECHNICAL_FEATURES]
    for fail in skipped:
        fail.assert_not_called()
