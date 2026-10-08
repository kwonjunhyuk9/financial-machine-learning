from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from financetoolkit.technicals.technicals_controller import Technicals
from loguru import logger


TECHNICAL_FEATURES = (
    'On-Balance Volume', 'Accumulation/Distribution Line',
    'Chaikin Oscillator', 'Money Flow Index', 'Williams %R',
    'Aroon Indicator Up', 'Aroon Indicator Down', 'Commodity Channel Index',
    'Relative Vigor Index', 'Force Index', 'Ultimate Oscillator',
    'Percentage Price Oscillator', 'Detrended Price Oscillator',
    'Average Directional Index', 'Chande Momentum Oscillator',
    'Ichimoku Conversion Line', 'Ichimoku Base Line', 'Ichimoku Leading Span A',
    'Ichimoku Leading Span B', 'Stochastic %K', 'Stochastic %D', 'MACD Line',
    'MACD Signal Line', 'Relative Strength Index', 'Balance of Power',
    'Simple Moving Average (SMA)', 'Exponential Moving Average (EMA)',
    'Double Exponential Moving Average (DEMA)', 'TRIX',
    'Triangular Moving Average', 'Weighted Moving Average (WMA)',
    'Hull Moving Average (HMA)', 'Volume Weighted Average Price (VWAP)',
    'Parabolic SAR', 'Pivot Point', 'Pivot Point Resistance 1',
    'Pivot Point Support 1', 'Bollinger Band Upper', 'Bollinger Band Middle',
    'Bollinger Band Lower', 'True Range', 'Average True Range',
    'Keltner Channel Upper', 'Keltner Channel Middle', 'Keltner Channel Lower',
    'Donchian Channel Upper', 'Donchian Channel Middle', 'Donchian Channel Lower',
)
EXCLUDED_TECHNICAL_FEATURES = (
    "TRIN",
    "New Highs - New Lows",
    "Advancers - Decliners",
    "McClellan Oscillator",
)
MODEL_FEATURES = (
    "mean_sentiment_score", "fractionally_differenced_log_close",
    *TECHNICAL_FEATURES,
)


def require_features(columns, expected=MODEL_FEATURES):
    """Reject missing or unexpected feature names."""
    if set(columns) != set(expected):
        raise ValueError(
            f"Feature schema mismatch: missing={set(expected) - set(columns)}, "
            f"unexpected={set(columns) - set(expected)}"
        )


def _select_native_technical_features(indicators: pd.DataFrame) -> pd.DataFrame:
    """Keep the FinanceToolkit columns valid on single-security dollar bars."""
    indicator_names = indicators.columns.get_level_values(0)
    selected = indicators.loc[
        :, ~indicator_names.isin(EXCLUDED_TECHNICAL_FEATURES)
    ]
    require_features(
        selected.columns.get_level_values(0),
        TECHNICAL_FEATURES,
    )
    return selected


REQUIRED_BAR_COLUMNS = {
    "start",
    "end",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def _build_output_path(data_path: Path) -> Path:
    """Build the default technical-indicator path beside its source bars."""
    period_pattern = re.compile(
        r"_(\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2})$"
    )
    period_match = period_pattern.search(data_path.stem)
    if period_match:
        output_stem = (
            f"{data_path.stem[:period_match.start()]}_technical"
            f"{period_match.group(0)}"
        )
    else:
        output_stem = f"{data_path.stem}_technical"
    return data_path.with_name(f"{output_stem}.parquet")


def save_market_technical_indicators(
        *,
        data_path: Path,
        window: int = 14,
        output_path: Path | None = None,
) -> Path:
    """Calculate bar-data technical indicators and save them as parquet.

    Args:
        data_path: Single-symbol OHLCV bar parquet source.
        window: Lookback window for applicable technical indicators.
        output_path: Optional parquet destination.

    Returns:
        The parquet path written to disk.
    """
    source = Path(data_path)
    bars = pd.read_parquet(source)
    missing_columns = REQUIRED_BAR_COLUMNS.difference(bars.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Bar data is missing required columns: {missing}.")

    symbols = bars["symbol"].dropna().astype(str).str.strip().str.upper().unique()
    if len(symbols) != 1:
        raise ValueError("Bar data must contain exactly one symbol.")
    symbol = symbols[0]

    historical_data = bars.set_index("end")[
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
    indicators = _select_native_technical_features(indicators).reset_index(
        drop=True
    )
    if isinstance(indicators.columns, pd.MultiIndex):
        indicators.columns = indicators.columns.get_level_values(0)
    indicators = indicators.loc[:, list(TECHNICAL_FEATURES)]

    identifiers = bars[["start", "end", "symbol"]].reset_index(drop=True)
    features = pd.concat(
        [identifiers, indicators],
        axis=1,
    )

    destination = Path(output_path) if output_path else _build_output_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    logger.info("Saved {} technical-indicator rows to {}.", len(features), destination)
    return destination


def build_technical_features(paths, *, manifest_path: Path, expected_securities: int,
                             window: int) -> pd.DataFrame:
    """Build technical features for every fixed-universe symbol."""
    from src.preprocessing.market_data import feature_identity, load_manifest, reusable_feature

    report = []
    for symbol in load_manifest(manifest_path, expected_securities=expected_securities).symbol:
        output = paths.feature(symbol, "technical")
        identity = feature_identity(
            paths,
            [paths.feature(symbol, "dollar_bars")],
            manifest_path=manifest_path, settings={"window": window},
        )
        if not reusable_feature(output, identity):
            save_market_technical_indicators(
                data_path=paths.feature(symbol, "dollar_bars"),
                output_path=output,
                window=window,
            )
            output.with_suffix(".json").write_text(json.dumps(identity, indent=2))
        report.append({"symbol": symbol, "status": "ready"})
    return pd.DataFrame(report)
