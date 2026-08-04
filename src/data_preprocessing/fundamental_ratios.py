from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from financetoolkit import Toolkit
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/research_data/fundamental/features"


def _get_api_key() -> str:
    """Read the Financial Modeling Prep API key from the environment."""
    load_dotenv()
    api_key = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
    if not api_key:
        raise ValueError("FINANCIAL_MODELING_PREP_API_KEY must be configured.")
    return api_key


def _build_output_path(*, symbol: str, start: datetime, end: datetime) -> Path:
    """Build a deterministic parquet path for fundamental ratio features."""
    slug = symbol.replace("/", "-").replace(" ", "").lower()
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    return DEFAULT_OUTPUT_DIR / f"{slug}_{start_str}_{end_str}.parquet"


def collect_fundamental_ratios(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    trailing: int | None = None,
) -> pd.DataFrame:
    """Collect all quarterly FinanceToolkit ratios for one symbol.

    Args:
        symbol: Ticker symbol to fetch.
        start: Inclusive fundamental-ratio start date.
        end: Inclusive fundamental-ratio end date.
        trailing: Number of quarterly periods used for trailing calculations.

    Returns:
        Fundamental ratios with reporting periods as rows and ratios as columns.
    """
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must not be empty.")

    toolkit = Toolkit(
        tickers=[normalized_symbol],
        api_key=_get_api_key(),
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        quarterly=True,
        enforce_source="FinancialModelingPrep",
        progress_bar=False,
    )
    ratios = toolkit.ratios.collect_all_ratios(trailing=trailing)
    features = ratios.transpose()
    features.index.name = "period"
    features.insert(0, "symbol", normalized_symbol)
    return features.reset_index()


def save_fundamental_ratios(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    trailing: int | None = None,
    output_path: Path | None = None,
) -> Path:
    """Collect all fundamental ratios for one symbol and save them as parquet.

    Args:
        symbol: Ticker symbol to fetch.
        start: Inclusive fundamental-ratio start date.
        end: Inclusive fundamental-ratio end date.
        trailing: Number of quarterly periods used for trailing calculations.
        output_path: Optional parquet destination.

    Returns:
        The parquet path written to disk.
    """
    features = collect_fundamental_ratios(
        symbol=symbol,
        start=start,
        end=end,
        trailing=trailing,
    )
    destination = output_path or _build_output_path(
        symbol=symbol,
        start=start,
        end=end,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    logger.info("Saved {} fundamental ratio periods to {}.", len(features), destination)
    return destination
