"""Self-financing portfolios evaluated at observed prices and traded at boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from heapq import merge
from itertools import groupby

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.backtesting.backtest_statistics import (
    ClassificationScores,
    Efficiency,
    GeneralCharacteristics,
    ImplementationShortfall,
    Performance,
    Runs,
)
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


def _weight_to_position(weight: float, k: int) -> int:
    """Map a capped portfolio weight to the open AFML position interval."""
    return int(np.clip(np.rint(weight * 2 * k * 100), -99, 99))


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
    """Rank events using calibration keyed by ``(symbol, event_start)``."""
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
        w = calibration.get((row.symbol, row.event_start), np.nan)
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
    evaluation_partition: str = "holdout",
) -> dict[str, pd.DataFrame]:
    """Simulate one cash account using prices strictly after each decision.

    observations yields chronological raw trade batches, plus empty minute-clock
    batches. Replacement orders require contemporaneous executable quotes for
    both legs; they never close the incumbent speculatively. Ledger rows are
    retained at minute/event/trade boundaries, not persisted to disk. Calibration
    is keyed by ``(symbol, event_start)``, and ``evaluation_partition`` selects
    the event partition used for the account.
    """
    from src.backtesting.bet_sizing import event_bet_signals
    if (settings.initial_aum <= 0 or settings.k < 1 or not 0 < settings.step_size <= 1
            or min(settings.broker_fee_bps, settings.slippage_bps) < 0):
        raise ValueError("Positive initial assets and K are required")
    events = events.loc[events.partition.eq(evaluation_partition)].sort_values(
        ["event_start", "symbol"]
    ).copy()
    if events.empty or events.duplicated(["symbol", "event_start"]).any():
        raise ValueError("Nonempty unique evaluation event keys are required")
    event_bet_signals(events)
    if events[["event_start", "vertical_barrier", "target_return"]].isna().any().any():
        raise ValueError("Signals require barriers and causal volatility")
    calibration_columns = {"symbol", "event_start", "w"}
    if not calibration_columns.issubset(calibration.columns):
        raise ValueError("Calibration requires symbol, event_start, and w")
    if calibration.duplicated(["symbol", "event_start"]).any():
        raise ValueError("Calibration event keys must be unique")
    w = calibration.set_index(["symbol", "event_start"])["w"].copy()
    event_keys = pd.MultiIndex.from_frame(events[["symbol", "event_start"]])
    missing = events.loc[w.reindex(event_keys).isna().to_numpy(), ["symbol", "event_start"]]
    exclusions = missing.assign(
        reason="missing development price calibration"
    ).to_dict("records")
    stream = list(events.to_dict("records"))
    event_position = 0
    active: dict[tuple, dict] = {}
    prices, quantity, pending, applied = {}, {}, {}, {}
    cash = settings.initial_aum
    fee_rate, slip_rate = settings.broker_fee_bps / 10_000, settings.slippage_bps / 10_000
    records, fills, exposures = [], [], []
    realized_cost = {}
    lot_cash = {}
    episodes = {}
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
        if not old and new:
            episodes[symbol] = {
                "event_start": time,
                "side": float(np.sign(new)),
                "entry_aum": wealth,
            }
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
            net_pnl = lot_cash.pop(symbol, 0.0)
            execution_cost = realized_cost.pop(symbol, 0.0)
            episode = episodes.pop(symbol)
            closed.append({
                **episode,
                "event_end": time,
                "timestamp": time,
                "symbol": symbol,
                "gross_pnl": net_pnl + execution_cost,
                "execution_cost": execution_cost,
                "net_pnl": net_pnl,
                "net_return": net_pnl / episode["entry_aum"],
            })

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
                order_limit = row.limit
                current_weight = applied.get(symbol, 0.0)
                if (quantity.get(symbol, 0) * target > 0
                        and abs(target) > abs(current_weight)):
                    current_position = _weight_to_position(current_weight, settings.k)
                    target_position = _weight_to_position(target, settings.k)
                    if current_position != target_position:
                        event = active[(symbol, row.event_start)]
                        forecast = event["entry_price"] * (
                            1 + event["primary_side"] * event["target_return"]
                        )
                        order_limit = limit_price(
                            target_position,
                            current_position,
                            forecast,
                            w[(symbol, row.event_start)],
                            100,
                        )
                pending[symbol] = {"decision": now, "weight": target, "limit": order_limit,
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
    closed_columns = [
        "event_start", "event_end", "timestamp", "symbol", "side", "entry_aum",
        "gross_pnl", "execution_cost", "net_pnl", "net_return",
    ]
    return {"ledger": pd.DataFrame(records), "trades": pd.DataFrame(fills),
            "closed_trades": pd.DataFrame(closed, columns=closed_columns),
            "exposures": pd.DataFrame(exposures),
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


def load_event_entry_prices(paths, events: pd.DataFrame) -> pd.DataFrame:
    """Attach dollar-bar closes observed at each event start."""
    indexed = "event_start" not in events.columns
    result = events.reset_index() if indexed else events.copy()
    result["entry_price"] = np.nan
    for symbol, indices in result.groupby("symbol", sort=False).groups.items():
        bars = pd.read_parquet(paths.feature(symbol, "dollar_bars")).set_index("end")
        starts = result.loc[indices, "event_start"]
        result.loc[indices, "entry_price"] = bars.close.reindex(starts).to_numpy()
    return result.set_index(events.index.names) if indexed else result


def prepare_calibration(paths) -> pd.DataFrame:
    """Calibrate development price sizing from observed dollar-bar prices."""
    from src.preprocessing.market_data import load_manifest, save_frame

    events = load_prediction_events(paths)
    development = load_event_entry_prices(
        paths, events.loc[events.partition.eq("development")]
    )
    result = calibrate_price_sizing(development)
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


def _daily_account(ledger: pd.DataFrame, initial_aum: float) -> pd.DataFrame:
    """Sample calendar-day account closes with explicit return intervals."""
    daily = ledger[["aum"]].resample("D").last().ffill()
    ends = daily.index + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    daily.index = pd.DatetimeIndex(
        [min(time, ledger.index[-1]) for time in ends],
        name="timestamp",
    )
    daily["period_start"] = pd.Series(
        [ledger.index[0], *daily.index[:-1]],
        index=daily.index,
    )
    previous_aum = daily["aum"].shift(1)
    previous_aum.iloc[0] = initial_aum
    daily["net_return"] = daily["aum"] / previous_aum - 1.0
    return daily


def _position_values(result: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align per-symbol dollar positions to the account observation clock."""
    exposures = result["exposures"]
    if exposures.empty:
        return pd.DataFrame({"_flat": 0.0}, index=index)
    return exposures.pivot(
        index="timestamp",
        columns="symbol",
        values="position_value",
    ).reindex(index).fillna(0.0)


def _closed_trade_inputs(
    closed_trades: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return aligned position, end-time, and return series for closed episodes."""
    starts = pd.DatetimeIndex(closed_trades["event_start"])
    positions = pd.Series(closed_trades["side"].to_numpy(), index=starts)
    event_end = pd.Series(
        pd.DatetimeIndex(closed_trades["event_end"]),
        index=starts,
    )
    returns = pd.Series(closed_trades["net_return"].to_numpy(), index=starts)
    return positions, event_end, returns


def summarize_account(result: dict[str, pd.DataFrame], settings: PortfolioSettings,
                      spy_returns: pd.Series) -> pd.DataFrame:
    """Compute the complete normalized statistics contract for one account."""
    ledger = result["ledger"].set_index("timestamp")
    if ledger.empty:
        raise ValueError("No account observations")
    daily = _daily_account(ledger, settings.initial_aum)
    position_values = _position_values(result, ledger.index)
    closed = result["closed_trades"]
    start = GeneralCharacteristics.start(ledger.index)
    end = GeneralCharacteristics.end(ledger.index)
    final = ledger.iloc[-1]
    increments = ledger[["broker_fee", "slippage", "traded_value"]].diff()
    increments.iloc[0] = ledger[["broker_fee", "slippage", "traded_value"]].iloc[0]
    net_dollar_performance = pd.Series(
        [final.aum - settings.initial_aum],
        dtype="float64",
    )
    equity = pd.concat([
        pd.Series([settings.initial_aum], index=pd.DatetimeIndex([start])),
        ledger["aum"],
    ])
    rows = []

    def add(section, metric, value=np.nan, unit="ratio", entity="final_account",
            timestamp=pd.NaT):
        rows.append({
            "section": section,
            "metric": metric,
            "entity": entity,
            "value": float(value),
            "timestamp": timestamp,
            "unit": unit,
        })

    add("general_characteristics", "start", unit="timestamp", timestamp=start)
    add("general_characteristics", "end", unit="timestamp", timestamp=end)
    general = {
        "average_aum": GeneralCharacteristics.average_aum(daily["aum"]),
        "leverage": GeneralCharacteristics.leverage(position_values, ledger["aum"]),
        "maximum_dollar_position_size": (
            GeneralCharacteristics.maximum_dollar_position_size(position_values)
        ),
        "annualized_turnover": GeneralCharacteristics.annualized_turnover(
            increments["traded_value"], daily["aum"], time_range=(start, end),
        ),
    }
    account_returns = daily["net_return"].copy()
    account_returns.index = account_returns.index.normalize()
    benchmark = spy_returns.copy()
    benchmark.index = benchmark.index.normalize()
    general["correlation_to_underlying"] = (
        GeneralCharacteristics.correlation_to_underlying(account_returns, benchmark)
    )
    if closed.empty:
        general.update({
            "ratio_of_longs": np.nan,
            "frequency_of_bets": 0.0,
            "average_holding_period": np.nan,
        })
        positions = event_end = trade_returns = None
    else:
        positions, event_end, trade_returns = _closed_trade_inputs(closed)
        general.update({
            "ratio_of_longs": GeneralCharacteristics.ratio_of_longs(positions),
            "frequency_of_bets": GeneralCharacteristics.frequency_of_bets(
                positions, event_end, time_range=(start, end),
            ),
            "average_holding_period": GeneralCharacteristics.average_holding_period(
                positions, event_end,
            ),
        })
    general_units = {
        "average_aum": "USD",
        "maximum_dollar_position_size": "USD",
        "frequency_of_bets": "count/year",
        "average_holding_period": "days",
    }
    for metric in [
        "average_aum", "leverage", "maximum_dollar_position_size",
        "ratio_of_longs", "frequency_of_bets", "average_holding_period",
        "annualized_turnover", "correlation_to_underlying",
    ]:
        add("general_characteristics", metric, general[metric], general_units.get(metric, "ratio"))

    net_pnl = final.aum - settings.initial_aum
    performance = {
        "pnl": Performance.pnl(net_dollar_performance),
        "annualized_rate_of_return": Performance.annualized_rate_of_return(
            daily["net_return"], start=start, end=end,
        ),
    }
    if closed.empty:
        performance.update({
            "pnl_from_long_positions": 0.0,
            "pnl_from_short_positions": 0.0,
            "hit_ratio": np.nan,
            "average_return_from_hits": np.nan,
            "average_return_from_misses": np.nan,
        })
    else:
        performance.update({
            "pnl_from_long_positions": Performance.pnl_from_long_positions(
                closed["net_pnl"], closed["side"],
            ),
            "pnl_from_short_positions": Performance.pnl_from_short_positions(
                closed["net_pnl"], closed["side"],
            ),
            "hit_ratio": Performance.hit_ratio(trade_returns),
            "average_return_from_hits": Performance.average_return_from_hits(trade_returns),
            "average_return_from_misses": Performance.average_return_from_misses(trade_returns),
        })
    for metric, value in performance.items():
        add("performance", metric, value, "USD" if "pnl" in metric else "ratio")

    runs = {
        "hhi_positive_returns": Runs.hhi_positive_returns(trade_returns)
        if trade_returns is not None else np.nan,
        "hhi_negative_returns": Runs.hhi_negative_returns(trade_returns)
        if trade_returns is not None else np.nan,
        "hhi_time_between_bets": Runs.hhi_time_between_bets(trade_returns)
        if trade_returns is not None else np.nan,
        "percentile_drawdown": Runs.percentile_drawdown(equity),
        "percentile_time_under_water": Runs.percentile_time_under_water(equity),
    }
    for metric, value in runs.items():
        add("runs", metric, value, "years" if "time_under_water" in metric else "ratio")

    shortfall = {
        "broker_fees_per_turnover": ImplementationShortfall.broker_fees_per_turnover(
            increments["broker_fee"], increments["traded_value"],
        ),
        "average_slippage_per_turnover": (
            ImplementationShortfall.average_slippage_per_turnover(
                increments["slippage"], increments["traded_value"],
            )
        ),
        "dollar_performance_per_turnover": (
            ImplementationShortfall.dollar_performance_per_turnover(
                net_dollar_performance, increments["traded_value"],
            )
        ),
        "return_on_execution_costs": ImplementationShortfall.return_on_execution_costs(
            net_dollar_performance,
            increments["broker_fee"] + increments["slippage"],
        ),
    }
    for metric, value in shortfall.items():
        add("implementation_shortfall", metric, value)

    efficiency = Efficiency.portfolio_statistics(
        daily["net_return"],
        daily["period_start"],
        annual_risk_free_rate=0.03,
        periods_per_year=365.25,
        annualized_benchmark_sharpe_ratio=1.0,
    )
    for metric, value in efficiency.items():
        name = "annualized_sharpe_ratio" if metric == "annualized_sharpe" else metric
        add("efficiency", name, value)

    if not result["exposures"].empty:
        exposure = result["exposures"].pivot(
            index="timestamp", columns="symbol", values="weight"
        ).reindex(ledger.index).fillna(0)
        durations = ledger.index.to_series().shift(-1).sub(
            ledger.index.to_series()
        ).dt.total_seconds().fillna(0)
        for symbol in exposure:
            add("exposure", "mean_absolute_weight",
                (exposure[symbol].abs() * durations).sum() / durations.sum()
                if durations.sum() else np.nan, entity=symbol)
            add("exposure", "max_absolute_weight", exposure[symbol].abs().max(), entity=symbol)
    add("benchmark", "net_return", (1 + spy_returns).prod() - 1, entity="SPY")
    for metric, value in {
        "initial_aum": settings.initial_aum, "K": settings.k,
        "one_way_broker_fee_bps": settings.broker_fee_bps,
        "one_way_slippage_bps": settings.slippage_bps,
        "cash_interest": 0, "borrow_fee": 0,
    }.items():
        add("assumption", metric, value, "setting")
    add("coverage", "excluded_price_calibrations", len(result["exclusions"]), "count")
    add("coverage", "closed_positions", len(result["closed_trades"]), "count")
    statistics = pd.DataFrame(rows)
    statistics["timestamp"] = pd.to_datetime(statistics["timestamp"], utc=True)
    if not np.isclose(closed["net_pnl"].sum() if not closed.empty else 0.0, net_pnl):
        raise ValueError("Closed-position PnL does not reconcile with final account AUM")
    return statistics


def run_final_backtest(paths, settings: PortfolioSettings = PortfolioSettings()):
    """Run the holdout account and persist its complete statistics contract."""
    from src.modeling.purged_validation import index_events
    from src.preprocessing.market_data import END, save_frame, sessions

    events = load_prediction_events(paths)
    saved_calibration = pd.read_parquet(paths.artifacts / "price_calibration.parquet")
    event_calibration = events[["symbol", "event_start"]].merge(
        saved_calibration,
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    calendar = sessions(paths)
    end = calendar.loc[calendar.open.lt(END), "close"].max()
    start = events.loc[events.partition.eq("holdout"), "event_start"].min()
    result = simulate_cross_sectional(
        events, observation_stream(paths, start, end),
        event_calibration,
        calendar, end, settings,
    )
    if result["ledger"].empty:
        raise ValueError("No account ledger was generated")
    account_start = result["ledger"].timestamp.min()
    stats = summarize_account(result, settings, benchmark_returns(paths, account_start, end))
    predictions = index_events(pd.read_parquet(paths.artifacts / "meta_predictions.parquet"))
    holdout = predictions.loc[predictions.partition.eq("holdout")]
    classification = {
        "primary": {
            "y_true": holdout["direction_label"],
            "y_pred": holdout["primary_side"],
            "probabilities": np.column_stack([
                1.0 - holdout["primary_probability"],
                holdout["primary_probability"],
            ]),
            "labels": [-1, 1],
        },
        "meta": {
            "y_true": holdout["meta_label"],
            "y_pred": holdout["meta_action"],
            "probabilities": np.column_stack([
                1.0 - holdout["meta_probability"],
                holdout["meta_probability"],
            ]),
            "labels": [0, 1],
        },
    }
    rows = []
    for entity, inputs in classification.items():
        values = {
            "accuracy": ClassificationScores.accuracy(
                inputs["y_true"], inputs["y_pred"], holdout["sample_weight"],
            ),
            "precision": ClassificationScores.precision(
                inputs["y_true"], inputs["y_pred"], sample_weight=holdout["sample_weight"],
            ),
            "recall": ClassificationScores.recall(
                inputs["y_true"], inputs["y_pred"], sample_weight=holdout["sample_weight"],
            ),
            "f1_score": ClassificationScores.f1_score(
                inputs["y_true"], inputs["y_pred"], sample_weight=holdout["sample_weight"],
            ),
            "negative_log_loss": ClassificationScores.negative_log_loss(
                inputs["y_true"], inputs["probabilities"], labels=inputs["labels"],
                sample_weight=holdout["sample_weight"],
            ),
        }
        rows.extend({
            "section": "classification_scores",
            "metric": metric,
            "entity": entity,
            "value": float(value),
            "timestamp": pd.NaT,
            "unit": "score",
        } for metric, value in values.items())
    extra = pd.DataFrame(rows)
    extra["timestamp"] = pd.to_datetime(extra["timestamp"], utc=True)
    stats = pd.concat([stats, extra], ignore_index=True)
    section_order = [
        "general_characteristics",
        "performance",
        "runs",
        "implementation_shortfall",
        "efficiency",
        "classification_scores",
        "exposure",
        "benchmark",
        "assumption",
        "coverage",
    ]
    stats["section"] = pd.Categorical(
        stats["section"],
        categories=section_order,
        ordered=True,
    )
    stats = stats.sort_values("section", kind="stable").reset_index(drop=True)
    stats["section"] = stats["section"].astype("string")
    save_frame(stats, paths.root / "data/backtest_results/backtest_statistics.parquet")
    return stats, result
