from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from financetoolkit import Toolkit
from financetoolkit.technicals.technicals_controller import Technicals
from loguru import logger


TECHNICAL_FEATURES = (
    'McClellan Oscillator', 'Advancers - Decliners', 'On-Balance Volume',
    'Accumulation/Distribution Line', 'Chaikin Oscillator', 'TRIN',
    'New Highs - New Lows', 'Money Flow Index', 'Williams %R',
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
MODEL_FEATURES = (
    "mean_sentiment_score", "fractionally_differenced_log_close",
    *TECHNICAL_FEATURES,
)
BREADTH_FEATURES = (
    "Advancers - Decliners", "TRIN", "McClellan Oscillator",
    "New Highs - New Lows",
)


def require_features(columns, expected=MODEL_FEATURES):
    """Reject missing or unexpected feature names."""
    if set(columns) != set(expected):
        raise ValueError(
            f"Feature schema mismatch: missing={set(expected) - set(columns)}, "
            f"unexpected={set(columns) - set(expected)}"
        )


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


def collect_market_technical_indicators(
        toolkit: Toolkit,
        *,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 14,
) -> pd.DataFrame:
    """Collect supported FinanceToolkit market technical indicators.

    Args:
        toolkit: FinanceToolkit instance with historical price data.
        period: Historical-data frequency accepted by FinanceToolkit.
        close_column: Price column used by close-based indicators.
        window: Lookback window for applicable technical indicators.

    Returns:
        Supported FinanceToolkit technical indicators indexed by date.
    """
    indicators = toolkit.technicals.collect_all_indicators(
        period=period,
        close_column=close_column,
        window=window,
    )
    indicator_names = indicators.columns.get_level_values(0)
    return indicators.loc[:, indicator_names != "TRIN"]


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
        market_breadth: pd.DataFrame | None = None,
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
    indicator_names = indicators.columns.get_level_values(0)
    indicators = indicators.loc[:, indicator_names != "TRIN"].reset_index(drop=True)
    if isinstance(indicators.columns, pd.MultiIndex):
        indicators.columns = indicators.columns.get_level_values(0)

    price_change = bars["close"].diff()
    price_direction = price_change.gt(0).astype(int) - price_change.lt(0).astype(int)
    indicators["On-Balance Volume"] = (
        price_direction * bars["volume"]
    ).cumsum().reset_index(drop=True)

    price_range = bars["high"] - bars["low"]
    money_flow_multiplier = (
        (bars["close"] - bars["low"])
        - (bars["high"] - bars["close"])
    ) / price_range
    money_flow_multiplier = money_flow_multiplier.mask(price_range.eq(0), 0.0)
    indicators["Accumulation/Distribution Line"] = (
        money_flow_multiplier * bars["volume"]
    ).cumsum().reset_index(drop=True)

    adl = indicators["Accumulation/Distribution Line"]
    indicators["Chaikin Oscillator"] = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()
    if market_breadth is not None:
        aligned = pd.merge_asof(bars[["end"]].sort_values("end"),
            market_breadth.rename(columns={"end": "breadth_time"}).sort_values("breadth_time"),
            left_on="end", right_on="breadth_time", direction="backward")
        stale = (aligned.end.dt.tz_convert("America/New_York").dt.date
                 != aligned.breadth_time.dt.tz_convert("America/New_York").dt.date)
        aligned.loc[stale, ["Advancers - Decliners", "TRIN"]] = float("nan")
        for column in BREADTH_FEATURES:
            indicators[column] = aligned[column].to_numpy()
        require_features(indicators.columns, TECHNICAL_FEATURES)
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


def intraday_breadth(prices: pd.DataFrame, cumulative_volume: pd.DataFrame,
                     previous_close: pd.Series, universe_size: int = 500) -> pd.DataFrame:
    """Calculate breadth without treating absent observations as unchanged."""
    valid = prices.notna() & cumulative_volume.notna() & previous_close.notna()
    up = prices.gt(previous_close, axis=1) & valid
    down = prices.lt(previous_close, axis=1) & valid
    advances, declines = up.sum(axis=1), down.sum(axis=1)
    up_volume = cumulative_volume.where(up, 0).sum(axis=1)
    down_volume = cumulative_volume.where(down, 0).sum(axis=1)
    coverage = valid.sum(axis=1) / universe_size
    trin = (advances / declines.replace(0, np.nan)) / (up_volume / down_volume.replace(0, np.nan))
    result = pd.DataFrame({
        "Advancers - Decliners": advances - declines,
        "TRIN": trin.replace([np.inf, -np.inf], np.nan),
        "coverage": coverage,
    })
    result.loc[coverage.lt(0.95), ["Advancers - Decliners", "TRIN"]] = np.nan
    return result


def build_breadth(paths) -> pd.DataFrame:
    """Use completed minute snapshots and lagged daily market-wide features."""
    from src.preprocessing.market_data import END, load_manifest, sessions

    symbols = list(load_manifest(paths).symbol)
    daily_closes, daily_net, output = [], [], []
    for session in sessions(paths).itertuples():
        if session.open >= END:
            continue
        index = pd.date_range(session.open + pd.Timedelta(minutes=1), session.close, freq="min")
        prices, volumes = {}, {}
        for symbol in symbols:
            file = paths.raw(symbol, "1min") / f"{session.open.date()}.parquet"
            if not file.exists():
                continue
            data = pd.read_parquet(file)
            if data.empty:
                continue
            data["timestamp"] += pd.Timedelta(minutes=1)
            minute = data.set_index("timestamp").reindex(index)
            prices[symbol] = minute.price
            volumes[symbol] = minute["size"].cumsum(skipna=False)
        price = pd.DataFrame(prices, index=index).reindex(columns=symbols)
        volume = pd.DataFrame(volumes, index=index).reindex(columns=symbols)
        previous = daily_closes[-1] if daily_closes else pd.Series(np.nan, index=symbols)
        breadth = intraday_breadth(price, volume, previous, len(symbols))
        net_history = pd.Series(daily_net, dtype=float)
        oscillator = np.nan
        if len(net_history) >= 39 and net_history.tail(39).notna().all():
            oscillator = (net_history.ewm(span=19, adjust=False).mean().iloc[-1]
                          - net_history.ewm(span=39, adjust=False).mean().iloc[-1])
        new_high_low, high_low_coverage = np.nan, 0.0
        if len(daily_closes) >= 253:
            history = pd.DataFrame(daily_closes[-253:-1])
            valid = history.notna().all() & previous.notna()
            high_low_coverage = float(valid.mean())
            if high_low_coverage >= 0.95:
                new_high_low = int(
                    (previous.gt(history.max()) & valid).sum()
                    - (previous.lt(history.min()) & valid).sum()
                )
        breadth["McClellan Oscillator"] = oscillator
        breadth["New Highs - New Lows"] = new_high_low
        breadth["high_low_coverage"] = high_low_coverage
        output.append(breadth.rename_axis("end").reset_index())
        daily_closes.append(price.iloc[-1])
        daily_net.append(breadth["Advancers - Decliners"].iloc[-1])
    return pd.concat(output, ignore_index=True)


def build_technical_features(paths) -> pd.DataFrame:
    """Build technical features for every fixed-universe symbol."""
    from src.preprocessing.market_data import feature_identity, load_manifest, reusable_feature

    breadth = build_breadth(paths)
    report, dependencies = [], []
    for member in load_manifest(paths).symbol:
        dependencies.extend(sorted(paths.raw(member, "1min").glob("*.json")))
    for symbol in load_manifest(paths).symbol:
        output = paths.feature(symbol, "technical")
        identity = feature_identity(paths, [*dependencies, paths.feature(symbol, "dollar_bars")])
        if not reusable_feature(output, identity):
            save_market_technical_indicators(
                data_path=paths.feature(symbol, "dollar_bars"),
                output_path=output,
                market_breadth=breadth,
            )
            output.with_suffix(".json").write_text(json.dumps(identity, indent=2))
        report.append({"symbol": symbol, "status": "ready"})
    result = pd.DataFrame(report)
    result.attrs["breadth_coverage"] = breadth[["end", "coverage", "high_low_coverage"]]
    return result
