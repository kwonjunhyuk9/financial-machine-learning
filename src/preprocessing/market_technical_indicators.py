from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pyarrow.parquet as pq
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


def _parabolic_sar(prices_high: pd.Series, prices_low: pd.Series) -> pd.Series:
    """Match FinanceToolkit's default SAR recurrence using array access."""
    high = prices_high.to_numpy()
    low = prices_low.to_numpy()
    sar = np.empty(len(high), dtype=float)
    if not len(high):
        return pd.Series(sar, index=prices_high.index)

    uptrend = True
    af = 0.02
    extreme_point = high[0]
    sar[0] = low[0]
    for i in range(1, len(high)):
        prior_sar = sar[i - 1]
        if uptrend:
            current_sar = prior_sar + af * (extreme_point - prior_sar)
            current_sar = min(current_sar, low[i - 1], low[max(i - 2, 0)])
            if low[i] < current_sar:
                uptrend = False
                current_sar = extreme_point
                extreme_point = low[i]
                af = 0.02
            elif high[i] > extreme_point:
                extreme_point = high[i]
                af = min(af + 0.02, 0.2)
        else:
            current_sar = prior_sar - af * (prior_sar - extreme_point)
            current_sar = max(current_sar, high[i - 1], high[max(i - 2, 0)])
            if high[i] > current_sar:
                uptrend = True
                current_sar = extreme_point
                extreme_point = high[i]
                af = 0.02
            elif low[i] < extreme_point:
                extreme_point = low[i]
                af = min(af + 0.02, 0.2)
        sar[i] = current_sar
    return pd.Series(sar, index=prices_high.index)


def save_market_technical_indicators(
        *,
        data_path: Path,
        window: int = 14,
        output_path: Path | None = None,
        log_saved: bool = True,
) -> Path:
    """Calculate bar-data technical indicators and save them as parquet.

    Args:
        data_path: Single-symbol OHLCV bar parquet source.
        window: Lookback window for applicable technical indicators.
        output_path: Optional parquet destination.
        log_saved: Emit a debug diagnostic for the saved artifact.

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
    # Override only this controller instance; keep the installed toolkit unchanged.
    def get_parabolic_sar(*, period, close_column):
        return _parabolic_sar(
            historical_data["High"][symbol], historical_data["Low"][symbol],
        ).to_frame(symbol).round(4)

    technicals.get_parabolic_sar = get_parabolic_sar
    kwargs = {"period": "daily", "close_column": "Adj Close"}
    breadth = pd.concat({
        "On-Balance Volume": technicals.get_on_balance_volume(**kwargs)[symbol],
        "Accumulation/Distribution Line": technicals.get_accumulation_distribution_line(**kwargs)[symbol],
        "Chaikin Oscillator": technicals.get_chaikin_oscillator(**kwargs)[symbol],
    }, axis=1)
    indicators = pd.concat([
        breadth,
        technicals.collect_momentum_indicators(**kwargs, window=window),
        technicals.collect_overlap_indicators(**kwargs, window=window),
        technicals.collect_volatility_indicators(**kwargs, window=window),
    ], axis=1).round(4)
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
    if log_saved:
        logger.debug("Saved {} technical-indicator rows to {}.", len(features), destination)
    return destination


def build_technical_features(paths, *, manifest_path: Path, expected_securities: int,
                             window: int,
                             show_progress: bool = False) -> pd.DataFrame:
    """Build technical features for every fixed-universe symbol.

    Args:
        show_progress: Show one overall progress bar, including cached symbols.
    """
    from src.preprocessing.market_data import feature_identity, load_manifest, reusable_feature

    report = []
    symbols = load_manifest(manifest_path, expected_securities=expected_securities).symbol
    with tqdm(symbols, desc="Technical indicators", disable=not show_progress) as progress:
        for symbol in progress:
            progress.set_postfix_str(symbol)
            output = paths.feature(symbol, "technical")
            identity = feature_identity(
                paths,
                [paths.feature(symbol, "dollar_bars")],
                manifest_path=manifest_path, settings={"window": window},
            )
            cached = reusable_feature(output, identity)
            if not cached:
                save_market_technical_indicators(
                    data_path=paths.feature(symbol, "dollar_bars"),
                    output_path=output,
                    window=window,
                    log_saved=False,
                )
                output.with_suffix(".json").write_text(json.dumps(identity, indent=2))
            report.append({"symbol": symbol, "status": "cached" if cached else "processed",
                           "rows": None if cached else pq.read_metadata(output).num_rows})
    return pd.DataFrame(report)
