from datetime import datetime

import pandas as pd
import pytest

from src.data_preprocessing import fundamental_data
from src.data_preprocessing.fundamental_data import _build_output_path


def test_build_output_path_uses_readable_date_range():
    path = _build_output_path(
        symbols=["AAPL"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2),
    )

    assert path.name == "aapl_2026-01-01_2026-01-02.parquet"


def test_get_api_key_requires_configured_fmp_key(monkeypatch):
    monkeypatch.setattr(fundamental_data, "load_dotenv", lambda: None)
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FINANCIAL_MODELING_PREP_API_KEY"):
        fundamental_data._get_api_key()


def test_fetch_fmp_financial_statements_combines_statement_types(monkeypatch):
    statement = pd.DataFrame({"2025Q1": [1.0]}, index=pd.Index(["Revenue"], name="Line Item"))

    class FakeToolkit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_income_statement(self):
            return statement

        def get_balance_sheet_statement(self):
            return statement

        def get_cash_flow_statement(self):
            return statement

    monkeypatch.setattr(fundamental_data, "Toolkit", FakeToolkit)
    monkeypatch.setattr(fundamental_data, "_get_api_key", lambda: "key")

    result = fundamental_data.fetch_fmp_financial_statements(
        symbols=["AAPL"],
        start=datetime(2025, 1, 1),
        end=datetime(2025, 12, 31),
    )

    assert result["statement_type"].tolist() == [
        "income_statement",
        "balance_sheet",
        "cash_flow_statement",
    ]
    assert result["Line Item"].tolist() == ["Revenue", "Revenue", "Revenue"]
