from datetime import datetime

import pytest

from src.data_preprocessing import fetch_fundamental_data
from src.data_preprocessing.fetch_fundamental_data import _build_output_path


def test_build_output_path_uses_fundamental_dataset_prefix():
    path = _build_output_path(
        symbols=["AAPL"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2),
    )

    assert path.name == "fundamental_aapl_20260101T000000Z_20260102T000000Z.parquet"


def test_get_identity_requires_configured_edgar_identity(monkeypatch):
    monkeypatch.setattr(fetch_fundamental_data, "load_dotenv", lambda: None)
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)

    with pytest.raises(ValueError, match="EDGAR_IDENTITY"):
        fetch_fundamental_data._get_identity()
