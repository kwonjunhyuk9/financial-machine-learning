from datetime import datetime

import pandas as pd
import pytest

from src.data_preprocessing import fundamental_ratios


def test_build_output_path_uses_symbol_and_readable_date_range():
    path = fundamental_ratios._build_output_path(
        symbol="BRK/B",
        start=datetime(2025, 1, 1),
        end=datetime(2025, 12, 31),
    )

    assert path.name == "brk-b_2025-01-01_2025-12-31.parquet"


def test_get_api_key_requires_configured_fmp_key(monkeypatch):
    monkeypatch.setattr(fundamental_ratios, "load_dotenv", lambda: None)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FINANCIAL_MODELING_PREP_API_KEY"):
        fundamental_ratios._get_api_key()


def test_collect_fundamental_ratios_collects_all_ratios_and_transposes(monkeypatch):
    ratio_data = pd.DataFrame(
        {
            "2025Q1": [1.1, 0.2],
            "2025Q2": [1.2, 0.3],
        },
        index=pd.Index(["Current Ratio", "Return on Equity"], name="ratio"),
    )
    toolkit_arguments = {}
    ratio_calls = []

    class FakeRatios:
        def collect_all_ratios(self, *, trailing):
            ratio_calls.append(trailing)
            return ratio_data

    class FakeToolkit:
        def __init__(self, **kwargs):
            toolkit_arguments.update(kwargs)
            self.ratios = FakeRatios()

    monkeypatch.setattr(fundamental_ratios, "Toolkit", FakeToolkit)
    monkeypatch.setattr(fundamental_ratios, "_get_api_key", lambda: "key")

    features = fundamental_ratios.collect_fundamental_ratios(
        symbol="aapl",
        start=datetime(2025, 1, 1),
        end=datetime(2025, 12, 31),
    )

    assert features.to_dict(orient="list") == {
        "period": ["2025Q1", "2025Q2"],
        "symbol": ["AAPL", "AAPL"],
        "Current Ratio": [1.1, 1.2],
        "Return on Equity": [0.2, 0.3],
    }
    assert toolkit_arguments == {
        "tickers": ["AAPL"],
        "api_key": "key",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "quarterly": True,
        "enforce_source": "FinancialModelingPrep",
        "progress_bar": False,
    }
    assert ratio_calls == [None]


def test_collect_fundamental_ratios_rejects_empty_symbol():
    with pytest.raises(ValueError, match="symbol"):
        fundamental_ratios.collect_fundamental_ratios(
            symbol=" ",
            start=datetime(2025, 1, 1),
            end=datetime(2025, 12, 31),
        )


def test_save_fundamental_ratios_writes_parquet(monkeypatch, tmp_path):
    expected = pd.DataFrame(
        {
            "period": ["2025Q1"],
            "symbol": ["AAPL"],
            "Current Ratio": [1.1],
        }
    )
    destination = tmp_path / "fundamental_ratios.parquet"
    monkeypatch.setattr(
        fundamental_ratios,
        "collect_fundamental_ratios",
        lambda **kwargs: expected,
    )

    saved_path = fundamental_ratios.save_fundamental_ratios(
        symbol="AAPL",
        start=datetime(2025, 1, 1),
        end=datetime(2025, 12, 31),
        output_path=destination,
    )

    assert saved_path == destination
    pd.testing.assert_frame_equal(pd.read_parquet(saved_path), expected)
