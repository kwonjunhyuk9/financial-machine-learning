from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv
from edgar import Company, set_identity
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/research_data/fundamental"


def _get_identity() -> str:
    """Read the SEC EDGAR identity from the environment."""
    load_dotenv()
    identity = os.getenv("EDGAR_IDENTITY")
    if not identity:
        raise ValueError("EDGAR_IDENTITY must contain your name and email address.")
    return identity


def _build_output_path(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
) -> Path:
    """Build a deterministic parquet path for fundamental data."""
    slug = "_".join(symbols).replace("/", "-").replace(" ", "").lower()
    start_str = start.strftime("%Y%m%dT%H%M%SZ")
    end_str = end.strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"fundamental_{slug}_{start_str}_{end_str}.parquet"


def _statement_frame(
        *,
        statement: object,
        symbol: str,
        filing: object,
        statement_type: str,
) -> pd.DataFrame:
    """Add filing metadata to one EdgarTools financial statement."""
    frame = statement.to_dataframe(view="summary", include_unit=True, include_point_in_time=True)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    frame.insert(0, "statement_type", statement_type)
    frame.insert(0, "accession", filing.accession_no)
    frame.insert(0, "report_date", getattr(filing, "report_date", None))
    frame.insert(0, "filing_date", filing.filing_date)
    frame.insert(0, "form_type", filing.form)
    frame.insert(0, "symbol", symbol)
    return frame


def fetch_edgar_financial_statements(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
) -> pd.DataFrame:
    """Fetch 10-K and 10-Q financial statements from SEC EDGAR.

    Args:
        symbols: Ticker symbols to fetch from SEC EDGAR.
        start: Inclusive filing-date boundary.
        end: Inclusive filing-date boundary.

    Returns:
        Standardized financial-statement rows with filing metadata.
    """
    set_identity(_get_identity())
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        filings = Company(symbol).get_filings(
            form=["10-K", "10-Q"],
            filing_date=(start.date().isoformat(), end.date().isoformat()),
            amendments=False,
            is_xbrl=True,
        )
        for filing in filings:
            report = filing.obj()
            financials = getattr(report, "financials", None)
            if financials is None:
                continue
            for statement_type, get_statement in (
                ("income_statement", financials.income_statement),
                ("balance_sheet", financials.balance_sheet),
                ("cash_flow_statement", financials.cashflow_statement),
            ):
                statement = get_statement()
                if statement is None:
                    continue
                frame = _statement_frame(
                    statement=statement,
                    symbol=symbol,
                    filing=filing,
                    statement_type=statement_type,
                )
                if not frame.empty:
                    frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "form_type",
                "filing_date",
                "report_date",
                "accession",
                "statement_type",
                "concept",
                "label",
                "standard_concept",
            ],
        )
    return pd.concat(frames, ignore_index=True)


def save_edgar_financial_statements(
        *,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        output_path: Path | None = None,
) -> Path:
    """Fetch SEC 10-K and 10-Q financial statements and save them as parquet.

    Args:
        symbols: Ticker symbols to fetch from SEC EDGAR.
        start: Inclusive filing-date boundary.
        end: Inclusive filing-date boundary.
        output_path: Optional parquet destination.

    Returns:
        The parquet path written to disk.
    """
    statements = fetch_edgar_financial_statements(symbols=symbols, start=start, end=end)
    destination = output_path or _build_output_path(symbols=symbols, start=start, end=end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    statements.to_parquet(destination, index=False)
    logger.info("Saved {} financial statement rows to {}.", len(statements), destination)
    return destination
