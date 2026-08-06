from __future__ import annotations

from pathlib import Path

import pandas as pd
from financetoolkit import Toolkit
from financetoolkit.technicals.technicals_controller import Technicals
from loguru import logger


REQUIRED_DOLLAR_BAR_COLUMNS = {
    "start",
    "end",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def collect_market_technical_indicators(
        toolkit: Toolkit,
        *,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
) -> pd.DataFrame:
    """Collect all FinanceToolkit market technical indicators.

    Args:
        toolkit: FinanceToolkit instance with historical price data.
        period: Historical-data frequency accepted by FinanceToolkit.
        close_column: Price column used by close-based indicators.
        window: Lookback window for applicable technical indicators.

    Returns:
        FinanceToolkit technical indicators indexed by date.
    """
    return toolkit.technicals.collect_all_indicators(
        period=period,
        close_column=close_column,
        window=window,
    )


def _build_output_path(data_path: Path) -> Path:
    """Build the default technical-indicator path beside its dollar bars."""
    marker = "_dollar_bar_"
    if marker not in data_path.stem:
        raise ValueError("data_path must identify a dollar-bar parquet file.")
    output_stem = data_path.stem.replace(marker, "_technical_indicators_", 1)
    return data_path.with_name(f"{output_stem}.parquet")


def save_market_technical_indicators(
        *,
        data_path: Path,
        window: int = 14,
        output_path: Path | None = None,
) -> Path:
    """Calculate dollar-bar technical indicators and save them as parquet.

    Args:
        data_path: Dollar-bar parquet source.
        window: Lookback window for applicable technical indicators.
        output_path: Optional parquet destination.

    Returns:
        The parquet path written to disk.
    """
    source = Path(data_path)
    dollar_bars = pd.read_parquet(source)
    missing_columns = REQUIRED_DOLLAR_BAR_COLUMNS.difference(dollar_bars.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dollar-bar data is missing required columns: {missing}.")

    symbols = dollar_bars["symbol"].dropna().astype(str).str.strip().str.upper().unique()
    if len(symbols) != 1:
        raise ValueError("Dollar-bar data must contain exactly one symbol.")
    symbol = symbols[0]

    historical_data = dollar_bars.set_index("end")[
        ["open", "high", "low", "close", "volume"]
    ].rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    historical_data["Adj Close"] = historical_data["Close"]
    historical_data["Return"] = historical_data["Adj Close"].pct_change(
        fill_method=None
    )
    historical_data["Cumulative Return"] = (
        1 + historical_data["Return"].fillna(0)
    ).cumprod()
    historical_data = historical_data[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Return",
            "Cumulative Return",
        ]
    ]
    historical_data.columns = pd.MultiIndex.from_product(
        [historical_data.columns, [symbol]]
    )

    empty_data = pd.DataFrame()
    technicals = Technicals(
        tickers=[symbol],
        historical_data={
            "intraday": empty_data,
            "daily": historical_data,
            "weekly": empty_data,
            "monthly": empty_data,
            "quarterly": empty_data,
            "yearly": empty_data,
        },
        rounding=4,
    )
    indicators = technicals.collect_all_indicators(
        period="daily",
        close_column="Adj Close",
        window=window,
    )
    identifiers = dollar_bars[["start", "end", "symbol"]].reset_index(drop=True)
    features = pd.concat(
        [identifiers, indicators.reset_index(drop=True)],
        axis=1,
    )

    destination = Path(output_path) if output_path else _build_output_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    logger.info("Saved {} technical-indicator rows to {}.", len(features), destination)
    return destination
