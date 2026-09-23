"""Self-financing portfolios evaluated at observed prices and traded at boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_event_prices(predictions: pd.DataFrame, prices: pd.Series) -> None:
    """Require exact holdout boundary prices consistent with saved raw returns."""
    _validate_index(prices.index, "prices")
    events = predictions.loc[predictions["partition"].eq("holdout")]
    starts = prices.reindex(events.index).to_numpy()
    ends = prices.reindex(pd.DatetimeIndex(events["event_end"])).to_numpy()
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("Every event boundary must have an exact observed price")
    if not np.allclose(
        ends / starts - 1, events["raw_return"], rtol=1e-9, atol=1e-12,
    ):
        raise ValueError("Observed event prices do not reproduce raw_return")


def simulate_portfolio(
    prices: pd.Series,
    target_positions: pd.Series,
    initial_aum: float = 100_000.0,
    broker_fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Mark a fractional-share cash account at every price, trade at boundaries.

    Targets are fractions of post-cost AUM. Prices must include every boundary;
    no interpolation or future prices are used. The final target must be zero.
    Cash earns no interest, and borrow fees and dividends are excluded. The first
    ``aum_before`` is the initial capital, before any opening transaction cost.
    """
    _validate_index(prices.index, "prices")
    _validate_index(target_positions.index, "target_positions")
    if prices.index.tz != target_positions.index.tz:
        raise ValueError("Prices and targets must use the same timezone")
    if not np.isfinite(initial_aum) or initial_aum <= 0:
        raise ValueError("initial_aum must be positive and finite")
    for name, value in {
        "broker_fee_bps": broker_fee_bps,
        "slippage_bps": slippage_bps,
    }.items():
        if not np.isfinite(value) or not 0 <= value < 10_000:
            raise ValueError(f"{name} must be in [0, 10000)")
    if broker_fee_bps + slippage_bps >= 10_000:
        raise ValueError("Total one-way execution cost must be below 10000 bps")
    if not target_positions.between(-1, 1).all():
        raise ValueError("target_positions must be finite and in [-1, 1]")
    if target_positions.iloc[-1] != 0:
        raise ValueError("The final target must close the portfolio")
    prices = prices.loc[target_positions.index[0]:target_positions.index[-1]]
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be positive and finite")
    boundaries = prices.index.get_indexer(target_positions.index)
    if (boundaries < 0).any():
        raise ValueError("Every target boundary must have an exact observed price")

    price = prices.to_numpy(dtype=float)
    size = len(prices)
    quantity = np.zeros(size)
    cash = np.zeros(size)
    target = np.zeros(size)
    traded_value = np.zeros(size)
    broker_fee = np.zeros(size)
    slippage_cost = np.zeros(size)
    execution_cost = np.zeros(size)
    old_quantity, old_cash = 0.0, float(initial_aum)
    broker_fee_rate = broker_fee_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    execution_cost_rate = broker_fee_rate + slippage_rate
    stops = np.append(boundaries[1:], size)
    for i, stop, weight in zip(boundaries, stops, target_positions, strict=True):
        equity = old_cash + old_quantity * price[i]
        if equity <= 0:
            raise ValueError("AUM must stay positive")
        old_value = old_quantity * price[i]
        direction = np.sign(weight * equity - old_value)
        # Solve x = weight * (equity - rate * abs(x - old_value)).
        new_value = weight * (
            equity + execution_cost_rate * direction * old_value
        ) / (
            1 + weight * execution_cost_rate * direction
        )
        traded_value[i] = abs(new_value - old_value)
        broker_fee[i] = broker_fee_rate * traded_value[i]
        slippage_cost[i] = slippage_rate * traded_value[i]
        execution_cost[i] = broker_fee[i] + slippage_cost[i]
        old_cash -= new_value - old_value + execution_cost[i]
        old_quantity = new_value / price[i]
        quantity[i:stop] = old_quantity
        cash[i:stop] = old_cash
        target[i:stop] = weight

    quantity_before = np.r_[0.0, quantity[:-1]]
    position_value = quantity * price
    aum = cash + position_value
    aum_before = aum + execution_cost
    if not np.isfinite(aum).all() or (aum <= 0).any():
        raise ValueError("AUM must stay positive and finite at every observed price")
    ledger = pd.DataFrame({
        "price": price,
        "target_position": target,
        "quantity_before": quantity_before,
        "quantity": quantity,
        "cash": cash,
        "aum_before": aum_before,
        "aum": aum,
        "position_value_before": quantity_before * price,
        "position_value": position_value,
        "traded_value": traded_value,
        "broker_fee": broker_fee,
        "slippage_cost": slippage_cost,
        "execution_cost": execution_cost,
        "gross_pnl": quantity_before * np.r_[0.0, np.diff(price)],
        "is_boundary": prices.index.isin(target_positions.index),
    }, index=prices.index)
    return ledger.rename_axis("timestamp")


def daily_portfolio(ledger: pd.DataFrame) -> pd.DataFrame:
    """Sample UTC calendar closes, retaining partial days and initial entry cost.

    Missing days carry the last known state forward. Period starts and ends are
    explicit so risk-free accrual uses the actual elapsed time, including weekends.
    """
    ledger = ledger.tz_convert("UTC")
    daily = ledger[["aum", "price", "position_value"]].resample("D").last().ffill()
    ends = daily.index + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    daily.index = pd.DatetimeIndex([
        min(time, ledger.index[-1]) for time in ends
    ], name="timestamp")
    daily["period_start"] = pd.Series(
        [ledger.index[0], *daily.index[:-1]], index=daily.index,
    )
    previous_aum = daily["aum"].shift(1)
    previous_aum.iloc[0] = ledger["aum_before"].iloc[0]
    previous_price = daily["price"].shift(1)
    previous_price.iloc[0] = ledger["price"].iloc[0]
    daily["net_return"] = daily["aum"] / previous_aum - 1
    daily["underlying_return"] = daily["price"] / previous_price - 1
    return daily


def portfolio_trades(ledger: pd.DataFrame) -> pd.DataFrame:
    """Attribute net PnL to flat-to-flat or same-direction position episodes.

    Same-side resizing stays in one trade. Flips split execution costs by closing
    and opening notional; the new trade's entry AUM is after the old exit cost.
    """
    boundaries = ledger.loc[ledger["is_boundary"]].copy()
    boundaries["cumulative_gross_pnl"] = ledger["gross_pnl"].cumsum()
    previous_gross = 0.0
    trade = None
    records = []
    for time, row in boundaries.iterrows():
        gross = row["cumulative_gross_pnl"] - previous_gross
        previous_gross = row["cumulative_gross_pnl"]
        old_side = np.sign(row["quantity_before"])
        new_side = np.sign(row["quantity"])
        if trade is not None:
            trade["gross_pnl"] += gross
        if old_side == new_side:
            if trade is not None:
                trade["execution_cost"] += row["execution_cost"]
            continue
        closing_cost = 0.0
        if old_side != 0:
            closing_cost = row["execution_cost"] * (
                abs(row["quantity_before"])
                / abs(row["quantity"] - row["quantity_before"])
            )
            trade["execution_cost"] += closing_cost
            trade["event_end"] = time
            trade["net_pnl"] = trade["gross_pnl"] - trade["execution_cost"]
            trade["net_return"] = trade["net_pnl"] / trade["entry_aum"]
            records.append(trade)
            trade = None
        if new_side != 0:
            trade = {
                "event_start": time,
                "side": new_side,
                "entry_aum": row["aum_before"] - closing_cost,
                "gross_pnl": 0.0,
                "execution_cost": row["execution_cost"] - closing_cost,
            }
    columns = [
        "event_start", "event_end", "side", "entry_aum", "gross_pnl",
        "execution_cost", "net_pnl", "net_return",
    ]
    result = pd.DataFrame(records, columns=columns).set_index("event_start")
    result.index = pd.DatetimeIndex(result.index, tz=ledger.index.tz)
    result["event_end"] = pd.to_datetime(result["event_end"], utc=True).dt.tz_convert(
        ledger.index.tz,
    )
    return result


def portfolio_equity(ledger: pd.DataFrame) -> pd.Series:
    """Include pre-entry capital and pre/post-trade marks for drawdown analysis."""
    values = np.column_stack([ledger["aum_before"], ledger["aum"]]).ravel()
    return pd.Series(values, index=ledger.index.repeat(2), name="aum")


def _validate_index(index: pd.Index, name: str) -> None:
    if (
        not isinstance(index, pd.DatetimeIndex) or index.tz is None
        or index.empty or index.hasnans or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError(f"{name} needs a nonempty, unique, sorted timezone index")
