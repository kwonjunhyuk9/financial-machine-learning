from datetime import datetime
from pathlib import Path

from src.data_preprocessing.market_data import _build_output_path
from src.data_preprocessing.market_data import _normalize_trade_frame


def test_build_output_path_normalizes_symbols():
    path = _build_output_path(
        asset_class="stock",
        symbols=["BRK/B"],
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2),
        output_dir=Path("output"),
    )

    assert path == Path("output/brk-b_2026-01-01_2026-01-02.parquet")


def test_normalize_trade_frame_rejects_missing_price_column():
    import pandas as pd
    import pytest

    trades = pd.DataFrame({"timestamp": ["2026-01-01"], "symbol": ["AAPL"], "size": [1]})

    with pytest.raises(ValueError, match="price"):
        _normalize_trade_frame(trades)
