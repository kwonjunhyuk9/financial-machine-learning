from __future__ import annotations

import pandas as pd
from financetoolkit import Toolkit


def collect_technical_indicator_features(
        toolkit: Toolkit,
        *,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
) -> pd.DataFrame:
    """Collect representative FinanceToolkit technical indicator features.

    Args:
        toolkit: FinanceToolkit instance with historical price data.
        period: Historical-data frequency accepted by FinanceToolkit.
        close_column: Price column used by close-based indicators.
        window: Lookback window for RSI, moving average, and ATR.

    Returns:
        Technical features indexed by date and ticker.
    """
    technicals = toolkit.technicals
    macd, macd_signal = technicals.get_moving_average_convergence_divergence(
        period=period,
        close_column=close_column,
    )
    feature_data = {
        "relative_strength_index": technicals.get_relative_strength_index(
            period=period,
            close_column=close_column,
            window=window,
        ),
        "moving_average": technicals.get_moving_average(
            period=period,
            close_column=close_column,
            window=window,
        ),
        "macd": macd,
        "macd_signal": macd_signal,
        "average_true_range": technicals.get_average_true_range(
            period=period,
            close_column=close_column,
            window=window,
        ),
    }
    return _combine_feature_data(feature_data)


def _combine_feature_data(feature_data: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    """Combine FinanceToolkit feature tables into date-ticker rows."""
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
