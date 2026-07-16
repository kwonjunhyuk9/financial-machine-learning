from __future__ import annotations

import pandas as pd
from financetoolkit import Toolkit


def collect_fundamental_ratio_features(
        toolkit: Toolkit,
        *,
        trailing: int | None = 4,
) -> pd.DataFrame:
    """Collect representative FinanceToolkit fundamental ratio features.

    Args:
        toolkit: FinanceToolkit instance with financial statements and price data.
        trailing: Number of quarterly periods used for trailing calculations.

    Returns:
        Fundamental features indexed by ticker and reporting period.
    """
    ratios = toolkit.ratios
    feature_data = {
        "price_to_book": ratios.get_price_to_book_ratio(trailing=trailing),
        "return_on_equity": ratios.get_return_on_equity(trailing=trailing),
        "operating_margin": ratios.get_operating_margin(trailing=trailing),
        "debt_to_equity": ratios.get_debt_to_equity_ratio(trailing=trailing),
        "asset_turnover": ratios.get_asset_turnover_ratio(trailing=trailing),
        "free_cash_flow_yield": ratios.get_free_cash_flow_yield(trailing=trailing),
        "dividend_yield": ratios.get_dividend_yield(trailing=trailing),
        "market_cap": ratios.get_market_cap(trailing=trailing),
    }
    return _combine_feature_data(feature_data)


def _combine_feature_data(feature_data: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    """Combine FinanceToolkit feature tables into ticker-period rows."""
    features = pd.concat(
        [_to_feature_series(data, name) for name, data in feature_data.items()],
        axis=1,
    )
    return features.reset_index()


def _to_feature_series(data: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    """Convert one FinanceToolkit result to a named feature series."""
    if isinstance(data, pd.Series):
        return data.rename(name)
    return data.stack(future_stack=True).rename(name)
