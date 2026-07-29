from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv
from financetoolkit import Toolkit
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/research_data/fundamental"


def _get_api_key() -> str:
    """Read the Financial Modeling Prep API key from the environment."""
    load_dotenv()
    api_key = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
    if not api_key:
        raise ValueError("FINANCIAL_MODELING_PREP_API_KEY must be configured.")
    return api_key


def _build_output_path(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
) -> Path:
    """Build a deterministic parquet path for fundamental data."""
    slug = "_".join(symbols).replace("/", "-").replace(" ", "").lower()
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    return DEFAULT_OUTPUT_DIR / f"{slug}_{start_str}_{end_str}.parquet"


def _statement_frame(*, statement: pd.DataFrame, statement_type: str) -> pd.DataFrame:
    """Convert one FinanceToolkit statement into a parquet-friendly table."""
    frame = statement.reset_index()
    if frame.empty:
        return frame
    frame.insert(0, "statement_type", statement_type)
    return frame


def fetch_fmp_financial_statements(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
) -> pd.DataFrame:
    """Fetch quarterly FMP financial statements through FinanceToolkit.

    Args:
        symbols: Ticker symbols to fetch.
        start: Inclusive financial-statement start date.
        end: Inclusive financial-statement end date.

    Returns:
        Standardized income, balance-sheet, and cash-flow statement rows.
    """
    toolkit = Toolkit(
        tickers=list(symbols),
        api_key=_get_api_key(),
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        quarterly=True,
        enforce_source="FinancialModelingPrep",
        progress_bar=False,
    )
    statements = (
        ("income_statement", toolkit.get_income_statement()),
        ("balance_sheet", toolkit.get_balance_sheet_statement()),
        ("cash_flow_statement", toolkit.get_cash_flow_statement()),
    )
    frames = [
        _statement_frame(statement=statement, statement_type=statement_type)
        for statement_type, statement in statements
        if not statement.empty
    ]
    if not frames:
        return pd.DataFrame(columns=["statement_type"])
    return pd.concat(frames, ignore_index=True)


def save_fmp_financial_statements(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        output_path: Path | None = None,
) -> Path:
    """Fetch FMP financial statements and save them as parquet.

    Args:
        symbols: Ticker symbols to fetch.
        start: Inclusive financial-statement start date.
        end: Inclusive financial-statement end date.
        output_path: Optional parquet destination.

    Returns:
        The parquet path written to disk.
    """
    statements = fetch_fmp_financial_statements(symbols=symbols, start=start, end=end)
    destination = output_path or _build_output_path(symbols=symbols, start=start, end=end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    statements.to_parquet(destination, index=False)
    logger.info("Saved {} financial statement rows to {}.", len(statements), destination)
    return destination
