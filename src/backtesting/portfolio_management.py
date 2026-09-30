"""Self-financing portfolios evaluated at observed prices and traded at boundaries."""

from __future__ import annotations

from heapq import merge
from itertools import groupby
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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


# Fixed-universe account. The original single-security helpers above remain
# available to callers; the research notebooks use this event-driven interface.
from dataclasses import dataclass
from collections.abc import Iterable

from src.backtesting.bet_sizing import (
    average_symbol_targets, get_target_position, limit_price, get_w,
)


@dataclass(frozen=True)
class PortfolioSettings:
    k: int = 5
    step_size: float = 0.10
    initial_aum: float = 100_000.0
    broker_fee_bps: float = 1.0
    slippage_bps: float = 1.0
    half_life_minutes: float = 390.0


def elapsed_session_minutes(start: pd.Timestamp, end: pd.Timestamp,
                            calendar: pd.DataFrame) -> float:
    """Count only regular-session minutes, respecting holidays and early closes."""
    relevant = calendar.loc[calendar.close.gt(start) & calendar.open.lt(end)]
    return sum(max(0.0, (min(row.close, end) - max(row.open, start)).total_seconds() / 60)
               for row in relevant.itertuples())


def calibrate_price_sizing(development: pd.DataFrame) -> pd.DataFrame:
    """Freeze per-symbol median forecast divergence using development only."""
    if not development.partition.eq("development").all():
        raise ValueError("Price calibration must not include holdout")
    rows = []
    for symbol, group in development.groupby("symbol"):
        divergence = (group.entry_price * group.target_return).abs()
        value = float(divergence.median())
        valid = np.isfinite(value) and value > 0
        rows.append({"symbol": symbol, "w": get_w(value, 0.5) if valid else np.nan,
                     "reason": "" if valid else "no positive development divergence"})
    return pd.DataFrame(rows)


def candidate_snapshot(active: pd.DataFrame, prices: dict[str, float],
                       calibration: pd.Series, now: pd.Timestamp,
                       calendar: pd.DataFrame, settings: PortfolioSettings) -> pd.DataFrame:
    """Rank representative events while sizing with all active events per symbol."""
    columns = ["symbol", "target_weight", "direction", "limit", "score", "eligible", "event_start"]
    if active.empty:
        return pd.DataFrame(columns=columns).set_index("symbol")
    targets = average_symbol_targets(active, settings.k, settings.step_size)
    latest = active.sort_values(["event_start", "symbol"]).drop_duplicates("symbol", keep="last")
    ranked = latest.sort_values(["mean_sentiment_score", "symbol"], ascending=[False, True])
    tail = len(ranked) // 5
    longs = set(ranked.head(tail).symbol) if tail >= settings.k else set()
    shorts = set(ranked.tail(tail).symbol) if tail >= settings.k else set()
    rows = []
    for row in latest.itertuples():
        target = float(targets[row.symbol])
        direction = int(np.sign(target))
        price = prices.get(row.symbol, np.nan)
        w = calibration.get(row.symbol, np.nan)
        ceiling, score = np.nan, -np.inf
        if direction and np.isfinite(w) and w > 0 and np.isfinite(price):
            forecast = row.entry_price * (1 + row.primary_side * row.target_return)
            step = get_target_position(w, forecast, price, 100)
            step = int(np.clip(step, -99, 99))
            if np.sign(step) == direction:
                ceiling = limit_price(step, 0, forecast, w, 100)
                age = elapsed_session_minutes(row.event_start, now, calendar)
                score = direction * (ceiling - price) / price * 2 ** (-age / settings.half_life_minutes)
        selected = row.symbol in (longs if direction > 0 else shorts)
        eligible = bool(direction and selected and row.primary_side == direction and row.meta_action == 1 and score > 0)
        rows.append({"symbol": row.symbol, "target_weight": target, "direction": direction,
                     "limit": ceiling, "score": score, "eligible": eligible, "event_start": row.event_start})
    return pd.DataFrame(rows).set_index("symbol")


def simulate_cross_sectional(
    events: pd.DataFrame,
    observations: Iterable[tuple[pd.Timestamp, pd.DataFrame]],
    calibration: pd.DataFrame,
    calendar: pd.DataFrame,
    end: pd.Timestamp,
    settings: PortfolioSettings = PortfolioSettings(),
) -> dict[str, pd.DataFrame]:
    """Simulate one cash account using prices strictly after each decision.

    observations yields chronological raw trade batches, plus empty minute-clock
    batches. Replacement orders require contemporaneous executable quotes for
    both legs; they never close the incumbent speculatively. Ledger rows are
    retained at minute/event/trade boundaries, not persisted to disk.
    """
    from src.backtesting.bet_sizing import event_bet_signals
    if (settings.initial_aum <= 0 or settings.k < 1 or not 0 < settings.step_size <= 1
            or min(settings.broker_fee_bps, settings.slippage_bps) < 0):
        raise ValueError("Positive initial assets and K are required")
    events = events.loc[events.partition.eq("holdout")].sort_values(["event_start", "symbol"]).copy()
    if events.empty or events.duplicated(["symbol", "event_start"]).any():
        raise ValueError("Nonempty unique holdout event keys are required")
    event_bet_signals(events)
    if events[["event_start", "vertical_barrier", "target_return"]].isna().any().any():
        raise ValueError("Signals require barriers and causal volatility")
    w = calibration.set_index("symbol").w.copy()
    missing = sorted(set(events.symbol) - set(w.dropna().index))
    exclusions = [{"symbol": symbol, "reason": "missing development price calibration"} for symbol in missing]
    stream = list(events.to_dict("records"))
    event_position = 0
    active: dict[tuple, dict] = {}
    prices, quantity, pending, applied = {}, {}, {}, {}
    cash = settings.initial_aum
    fee_rate, slip_rate = settings.broker_fee_bps / 10_000, settings.slippage_bps / 10_000
    records, fills, exposures = [], [], []
    realized_cost = {}
    lot_cash = {}
    closed = []
    previous_time = None
    previous_minute = None
    liquidation_time = end - pd.Timedelta(minutes=1)
    liquidating = False
    cutoff = events.event_start.min()
    cumulative_fee = cumulative_slip = cumulative_turnover = 0.0

    def aum():
        return cash + sum(q * prices[s] for s, q in quantity.items() if q)

    def fill(symbol, weight, time, reason):
        nonlocal cash, cumulative_fee, cumulative_slip, cumulative_turnover
        price = prices[symbol]
        old = quantity.get(symbol, 0.0)
        wealth = aum()
        if wealth <= 0:
            raise ValueError("Account equity exhausted")
        # Solve target quantity against post-cost equity. Costs include adverse
        # execution slippage and broker fees assessed at the execution price.
        low, high = 0.0, wealth
        for _ in range(60):
            post = (low + high) / 2
            q = weight * post / price
            delta = q - old
            execution = price * (1 + np.sign(delta) * slip_rate)
            cost = abs(delta) * price * slip_rate + abs(delta * execution) * fee_rate
            if post + cost > wealth:
                high = post
            else:
                low = post
        new = weight * ((low + high) / 2) / price if weight else 0.0
        delta = new - old
        if abs(delta) < 1e-12:
            return
        execution = price * (1 + np.sign(delta) * slip_rate)
        fee, slip = abs(delta * execution) * fee_rate, abs(delta) * price * slip_rate
        cash -= delta * execution + fee
        quantity[symbol] = new
        lot_cash[symbol] = lot_cash.get(symbol, 0.0) - delta * execution - fee
        realized_cost[symbol] = realized_cost.get(symbol, 0.0) + fee + slip
        cumulative_fee += fee
        cumulative_slip += slip
        cumulative_turnover += abs(delta * price)
        fills.append({"timestamp": time, "symbol": symbol, "quantity_change": delta,
                      "price": price, "execution_price": execution, "broker_fee": fee,
                      "slippage": slip, "traded_value": abs(delta * price), "reason": reason})
        if not new:
            closed.append({"timestamp": time, "symbol": symbol,
                           "net_pnl": lot_cash.pop(symbol, 0.0), "execution_cost": realized_cost.pop(symbol, 0.0)})

    for now, quotes in observations:
        now = pd.Timestamp(now)
        if previous_time is not None and now <= previous_time:
            raise ValueError("Observation batches must have strictly increasing times")
        previous_time = now
        if now > end:
            break
        touched = False
        fresh = {}
        completed_bars = set(quotes.loc[quotes.bar_end, "symbol"]) if "bar_end" in quotes else set()
        executable = quotes.loc[~quotes.bar_end] if "bar_end" in quotes else quotes
        if not executable.empty:
            if not np.isfinite(executable.price).all() or not executable.price.gt(0).all():
                raise ValueError("Executable prices must be finite and positive")
            fresh = executable.groupby("symbol", sort=False).price.last().to_dict()
            prices.update(fresh)
        if now < cutoff:
            continue
        # Expire individual events from observed barriers, never event_end.
        for key, event in list(active.items()):
            symbol = event["symbol"]
            hit = symbol in fresh and abs(prices[symbol] / event["entry_price"] - 1) >= event["target_return"]
            if symbol in completed_bars and "remaining_bars" in event:
                event["remaining_bars"] -= 1
            vertical_hit = (event["remaining_bars"] <= 0 if "remaining_bars" in event
                            else now >= event["vertical_barrier"])
            if vertical_hit or hit:
                del active[key]
                pending.pop(symbol, None)
                touched = True
        if now >= liquidation_time and not liquidating:
            liquidating = True
            pending.clear()
            for symbol, q in quantity.items():
                if q:
                    pending[symbol] = {"decision": now, "weight": 0.0, "limit": np.nan, "reason": "research_end"}
            touched = True
        # Only orders decided before now may consume this batch.
        for symbol, order in list(pending.items()):
            if symbol not in fresh or now <= order["decision"]:
                continue
            weight, victim = order["weight"], order.get("victim")
            old = quantity.get(symbol, 0.0)
            increasing = order["reason"] in {"signal_entry", "replacement_entry"}
            if increasing:
                execution = prices[symbol] * (1 + np.sign(weight) * slip_rate)
                if np.sign(weight) * (execution - order["limit"]) > 0:
                    continue
            if victim and not quantity.get(victim, 0):
                occupied = sum(q * weight > 0 for q in quantity.values())
                if occupied >= settings.k:
                    pending.pop(symbol, None)
                    continue
                victim = None
            if victim:
                if victim not in fresh:
                    continue
                fill(victim, 0.0, now, "replacement_exit")
                pending.pop(victim, None)
                applied[victim] = 0.0
            if old and weight and np.sign(old) != np.sign(weight):
                fill(symbol, 0.0, now, "direction_exit")
            fill(symbol, weight, now, order["reason"])
            applied[symbol] = weight
            pending.pop(symbol, None)
            touched = True
        while event_position < len(stream) and stream[event_position]["event_start"] <= now:
            event = stream[event_position].copy()
            event_position += 1
            symbol = event["symbol"]
            if event["event_start"] != now or symbol not in fresh:
                raise ValueError(f"Missing exact observed event-start price: {symbol} {event['event_start']}")
            event["entry_price"] = prices[symbol]
            if "horizon_bars" in event:
                event["remaining_bars"] = int(event["horizon_bars"])
            pending.pop(symbol, None)
            active[(symbol, event["event_start"])] = event
            touched = True
        minute = now.floor("min")
        minute_changed = minute != previous_minute
        previous_minute = minute
        if touched or minute_changed:
            table = candidate_snapshot(pd.DataFrame(active.values()), prices, w, now, calendar, settings)
            desired = table.target_weight.to_dict() if not table.empty and not liquidating else {}
            for symbol in list(pending):
                order = pending[symbol]
                entry = order["reason"] in {"signal_entry", "replacement_entry"}
                if (order["weight"] != desired.get(symbol, 0.0) or entry):
                    pending.pop(symbol)
            # Reductions and zero targets never require candidate eligibility.
            for symbol, q in list(quantity.items()):
                if not q:
                    continue
                target = desired.get(symbol, 0.0)
                previous = applied.get(symbol, 0.0)
                if target == previous:
                    continue
                if target == 0 or np.sign(target) != np.sign(q) or abs(target) < abs(previous):
                    reduced = 0.0 if target * q <= 0 else target
                    if symbol not in pending or pending[symbol]["weight"] != reduced:
                        pending[symbol] = {"decision": now, "weight": reduced, "limit": np.nan, "reason": "research_end" if liquidating else "signal_reduction"}
            if liquidating:
                table = table.iloc[0:0]
            ordered = table.loc[table.eligible].reset_index().sort_values(
                ["score", "event_start", "symbol"], ascending=[False, False, True]) if not table.empty else table
            reserved = {s for s, order in pending.items() if order["weight"]}
            victims = {order.get("victim") for order in pending.values()}
            for row in ordered.itertuples(index=False):
                symbol, target = row.symbol, row.target_weight
                if symbol in pending or (quantity.get(symbol, 0) and applied.get(symbol) == target):
                    continue
                if quantity.get(symbol, 0) * target < 0:
                    continue  # wait for existing direction to close
                held = [s for s, q in quantity.items() if q * row.direction > 0]
                queued = [s for s in reserved if pending[s]["weight"] * row.direction > 0 and s not in held and not pending[s].get("victim")]
                victim = None
                if symbol not in held and len(held) + len(queued) >= settings.k:
                    available = [s for s in held if s not in victims and s not in pending]
                    if not available:
                        continue
                    victim = min(available, key=lambda s: (table.loc[s, "score"] if s in table.index else -np.inf, s))
                    incumbent_score = table.loc[victim, "score"] if victim in table.index else -np.inf
                    if row.score <= incumbent_score:
                        continue
                pending[symbol] = {"decision": now, "weight": target, "limit": row.limit,
                                   "victim": victim, "reason": "replacement_entry" if victim else "signal_entry"}
                reserved.add(symbol)
                victims.add(victim)
            wealth = aum()
            gross = sum(abs(q * prices[s]) for s, q in quantity.items() if q)
            net = sum(q * prices[s] for s, q in quantity.items() if q)
            records.append({"timestamp": now, "cash": cash, "aum": wealth, "gross_exposure": gross / wealth,
                            "net_exposure": net / wealth, "broker_fee": cumulative_fee,
                            "slippage": cumulative_slip, "traded_value": cumulative_turnover})
            for symbol, q in quantity.items():
                if q:
                    exposures.append({"timestamp": now, "symbol": symbol, "quantity": q,
                                      "position_value": q * prices[symbol], "weight": q * prices[symbol] / wealth})
    if any(quantity.values()):
        raise ValueError("Research end has unliquidated positions; supply liquidation quotes before end")
    if event_position != len(stream):
        raise ValueError("Market stream ended before all holdout events were observed")
    return {"ledger": pd.DataFrame(records), "trades": pd.DataFrame(fills),
            "closed_trades": pd.DataFrame(closed), "exposures": pd.DataFrame(exposures),
            "exclusions": pd.DataFrame(exclusions)}


def load_prediction_events(paths) -> pd.DataFrame:
    """Join model events to their one-to-one meta predictions."""
    events = pd.read_parquet(paths.event("model"))
    predictions = pd.read_parquet(paths.artifacts / "meta_predictions.parquet")
    keys = ["symbol", "event_start"]
    required = [*keys, "primary_side", "meta_action", "meta_probability"]
    if len(events) != len(predictions):
        raise ValueError("Prediction and event row counts differ")
    result = events.merge(predictions[required], on=keys, validate="one_to_one")
    if len(result) != len(events):
        raise ValueError("Predictions do not cover the exact event keys")
    result["horizon_bars"] = 1000
    return result.sort_values(["event_start", "symbol"])


def prepare_calibration(paths) -> pd.DataFrame:
    """Calibrate development price sizing from observed dollar-bar prices."""
    from src.preprocessing.market_data import load_manifest, save_frame

    events = load_prediction_events(paths)
    development = events.loc[events.partition.eq("development")].copy()
    tables = []
    for symbol, group in development.groupby("symbol"):
        bars = pd.read_parquet(paths.feature(symbol, "dollar_bars")).set_index("end")
        table = group.copy()
        table["entry_price"] = bars.close.reindex(group.event_start).to_numpy()
        tables.append(table)
    result = calibrate_price_sizing(pd.concat(tables, ignore_index=True))
    missing = set(load_manifest(paths).symbol) - set(result.symbol)
    result = pd.concat([
        result,
        pd.DataFrame([{"symbol": symbol, "w": np.nan, "reason": "no development events"}
                      for symbol in sorted(missing)]),
    ], ignore_index=True)
    result["holdout_boundary"] = events.holdout_boundary.iloc[0]
    save_frame(result, paths.artifacts / "price_calibration.parquet")
    return result


def _trade_rows(file):
    """Yield trade rows while bounding memory to one Arrow batch."""
    for batch in pq.ParquetFile(file).iter_batches(
        batch_size=8192, columns=["timestamp", "symbol", "price"]
    ):
        for row in batch.to_pandas().itertuples(index=False, name=None):
            yield (*row, False)


def observation_stream(paths, start: pd.Timestamp, end: pd.Timestamp):
    """Merge raw trades, dollar-bar completions, and minute clocks."""
    from src.preprocessing.market_data import load_manifest, sessions

    symbols = list(load_manifest(paths).symbol)
    for session in sessions(paths).itertuples():
        if session.close < start or session.open > end:
            continue
        streams = []
        for symbol in symbols:
            file = paths.raw(symbol, "tick") / f"{session.open.date()}.parquet"
            if file.exists():
                if not file.with_suffix(".json").exists():
                    raise ValueError(f"Incomplete tick partition {file}")
                streams.append(_trade_rows(file))
            bars_file = paths.feature(symbol, "dollar_bars")
            if bars_file.exists():
                bar_ends = pd.read_parquet(
                    bars_file, columns=["end"],
                    filters=[("end", ">=", session.open), ("end", "<=", session.close)],
                ).end
                streams.append(iter([(time, symbol, np.nan, True) for time in bar_ends]))
        clocks = ((time, "", np.nan, False)
                  for time in pd.date_range(session.open, session.close, freq="min"))
        for timestamp, rows in groupby(
            merge(*streams, clocks, key=lambda row: row[0]), key=lambda row: row[0]
        ):
            quotes = [(symbol, price, bar_end)
                      for _, symbol, price, bar_end in rows if symbol]
            yield timestamp, pd.DataFrame(quotes, columns=["symbol", "price", "bar_end"])


def benchmark_returns(paths, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Compute close-to-close SPY returns from the final trade of each day."""
    from src.preprocessing.market_data import raw_partitions

    points = []
    for file in raw_partitions(paths, "SPY", "tick"):
        data = pd.read_parquet(file, columns=["timestamp", "price"])
        if data.empty:
            continue
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.loc[data.timestamp.le(end)]
        if not data.empty:
            points.append(data)
    if not points:
        raise ValueError("SPY benchmark prices unavailable")
    frame = pd.concat(points).set_index("timestamp").sort_index()
    anchor = frame.loc[frame.index <= start].tail(1)
    after = frame.loc[frame.index > start]
    if anchor.empty or after.empty:
        raise ValueError("SPY does not cover the evaluation interval")
    daily = pd.concat([anchor, after]).price.resample("D").last().dropna()
    returns = daily.pct_change()
    returns.iloc[0] = 0.0
    return returns


def summarize_account(result: dict[str, pd.DataFrame], settings: PortfolioSettings,
                      spy_returns: pd.Series) -> pd.DataFrame:
    ledger = result["ledger"].set_index("timestamp")
    if ledger.empty:
        raise ValueError("No account observations")
    daily = ledger.aum.resample("D").last().dropna()
    returns = daily.pct_change()
    returns.iloc[0] = daily.iloc[0] / settings.initial_aum - 1
    years = (ledger.index[-1] - ledger.index[0]).total_seconds() / (365.25 * 86400)
    final = ledger.iloc[-1]
    net_return = final.aum / settings.initial_aum - 1
    equity = pd.concat([pd.Series([settings.initial_aum]), ledger.aum.reset_index(drop=True)])
    trades = result["closed_trades"]
    pnl = trades.net_pnl if not trades.empty else pd.Series(dtype=float)
    cost, turnover = final.broker_fee + final.slippage, final.traded_value
    rows = []

    def add(section, metric, value, unit="ratio", symbol=""):
        rows.append({"section": section, "metric": metric, "symbol": symbol,
                     "value": float(value), "unit": unit})

    metrics = {
        "net_return": net_return,
        "cagr": (1 + net_return) ** (1 / years) - 1 if years > 0 else np.nan,
        "annualized_sharpe": returns.mean() / returns.std() * np.sqrt(252)
        if returns.std() > 0 else np.nan,
        "maximum_drawdown": float((1 - equity / equity.cummax()).max()),
        "hit_ratio": float(pnl.gt(0).mean()) if len(pnl) else np.nan,
        "turnover": turnover / settings.initial_aum,
        "broker_fees_per_turnover": final.broker_fee / turnover if turnover else np.nan,
        "slippage_per_turnover": final.slippage / turnover if turnover else np.nan,
        "dollar_performance_per_turnover": (final.aum - settings.initial_aum) / turnover
        if turnover else np.nan,
        "return_on_execution_cost": (final.aum - settings.initial_aum) / cost if cost else np.nan,
    }
    for metric, value in metrics.items():
        add("portfolio", metric, value)
    for metric, value in {
        "final_aum": final.aum, "net_pnl": final.aum - settings.initial_aum,
        "gross_pnl": final.aum - settings.initial_aum + cost,
        "average_hit": pnl.loc[pnl.gt(0)].mean(), "average_miss": pnl.loc[pnl.lt(0)].mean(),
        "broker_fees": final.broker_fee, "slippage": final.slippage,
        "execution_cost": cost,
    }.items():
        add("portfolio", metric, value, "USD")
    durations = ledger.index.to_series().shift(-1).sub(
        ledger.index.to_series()
    ).dt.total_seconds().fillna(0)
    for column in ["gross_exposure", "net_exposure"]:
        add("portfolio", "mean_" + column,
            (ledger[column] * durations).sum() / durations.sum() if durations.sum() else np.nan)
        add("portfolio", "max_" + column, ledger[column].max())
    if not result["exposures"].empty:
        exposure = result["exposures"].pivot(
            index="timestamp", columns="symbol", values="weight"
        ).reindex(ledger.index).fillna(0)
        for symbol in exposure:
            add("exposure", "mean_absolute_weight",
                (exposure[symbol].abs() * durations).sum() / durations.sum(), symbol=symbol)
            add("exposure", "max_absolute_weight", exposure[symbol].abs().max(), symbol=symbol)
    add("benchmark", "net_return", (1 + spy_returns).prod() - 1, symbol="SPY")
    add("benchmark", "correlation", returns.corr(spy_returns), symbol="SPY")
    for metric, value in {
        "initial_aum": settings.initial_aum, "K": settings.k,
        "one_way_broker_fee_bps": settings.broker_fee_bps,
        "one_way_slippage_bps": settings.slippage_bps,
        "cash_interest": 0, "borrow_fee": 0,
    }.items():
        add("assumption", metric, value, "setting")
    add("coverage", "excluded_price_calibrations", len(result["exclusions"]), "count")
    add("coverage", "closed_positions", len(result["closed_trades"]), "count")
    return pd.DataFrame(rows)


def run_final_backtest(paths, settings: PortfolioSettings = PortfolioSettings()):
    """Run the holdout account and persist only final statistics."""
    from src.modeling.model_workflow import score_binary_predictions
    from src.modeling.purged_validation import index_events
    from src.preprocessing.market_data import END, save_frame, sessions

    events = load_prediction_events(paths)
    calendar = sessions(paths)
    end = calendar.loc[calendar.open.lt(END), "close"].max()
    start = events.loc[events.partition.eq("holdout"), "event_start"].min()
    result = simulate_cross_sectional(
        events, observation_stream(paths, start, end),
        pd.read_parquet(paths.artifacts / "price_calibration.parquet"),
        calendar, end, settings,
    )
    if result["ledger"].empty:
        raise ValueError("No account ledger was generated")
    account_start = result["ledger"].timestamp.min()
    stats = summarize_account(result, settings, benchmark_returns(paths, account_start, end))
    predictions = index_events(pd.read_parquet(paths.artifacts / "meta_predictions.parquet"))
    holdout = predictions.loc[predictions.partition.eq("holdout")]
    scores = score_binary_predictions(
        holdout.meta_label, holdout.meta_action, holdout.meta_probability,
        holdout.sample_weight, class_labels=[0, 1], positive_label=1,
    )
    extra = pd.DataFrame([
        {"section": "classification", "metric": name, "symbol": "",
         "value": value, "unit": "score"}
        for name, value in scores.items() if np.isscalar(value)
    ])
    stats = pd.concat([stats, extra], ignore_index=True)
    save_frame(stats, paths.root / "data/backtest_results/backtest_statistics.parquet")
    return stats, result
