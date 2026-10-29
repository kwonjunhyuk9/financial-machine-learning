"""Self-financing portfolios executed on completed dollar-bar closes.

Prices and quantities use research-start share units. Splits are already
absorbed by preprocessing; positions must not receive a second split adjustment.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from heapq import heapify, heappop, heappush, merge
from itertools import groupby
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from src.backtesting.bet_sizing import (
    get_target_position, limit_price, get_w,
)


class BarStream:
    """Scan end times for counts/terminal bars, then merge bounded price batches."""

    def __init__(self, paths, start, end, symbols):
        self.start, self.end = start, end
        self.sources = []
        self.count = 0
        for symbol in sorted(set(symbols)):
            file = paths.feature(symbol, "dollar_bars")
            if not file.exists():
                raise ValueError(f"Missing dollar bars for {symbol}; run dollar-bar preprocessing: {file}")
            previous, count = None, 0
            with pq.ParquetFile(file) as parquet:
                for batch in parquet.iter_batches(batch_size=4096, columns=["end"]):
                    column = batch.column(0)
                    if column.null_count:
                        raise ValueError(f"Nonmissing bar end times are required: {symbol}")
                    times = column.cast("int64").to_numpy()
                    # Arrow timestamp units can differ; normalize to nanoseconds.
                    unit = column.type.unit
                    times = times * {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[unit]
                    times = times[(times >= start.value) & (times <= end.value)]
                    if not len(times):
                        continue
                    if (previous is not None and times[0] <= previous) or np.any(times[1:] <= times[:-1]):
                        raise ValueError(f"Unique increasing bar end times per symbol are required: {symbol}")
                    previous = int(times[-1])
                    count += len(times)
            self.count += count
            if count:
                self.sources.append((symbol, file, previous))

    def __len__(self):
        return self.count

    def _spool(self, file, destination):
        # Parquet decoders can retain an entire row group. Close each decoder
        # before merging symbols; IPC readers retain only a single small batch.
        with pq.ParquetFile(file) as parquet:
            schema = pa.schema([parquet.schema_arrow.field(name) for name in ("end", "close")])
            with pa.OSFile(str(destination), "wb") as sink, ipc.new_file(sink, schema) as writer:
                for batch in parquet.iter_batches(batch_size=4096, columns=["end", "close"]):
                    column = batch.column(0)
                    times = column.cast("int64").to_numpy()
                    times = times * {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[column.type.unit]
                    selected = batch.filter((times >= self.start.value) & (times <= self.end.value))
                    if selected.num_rows:
                        writer.write_batch(selected)

    def _rows(self, symbol, file, last, destination):
        self._spool(file, destination)
        with pa.OSFile(str(destination), "rb") as source:
            reader = ipc.open_file(source)
            for index in range(reader.num_record_batches):
                frame = reader.get_batch(index).to_pandas()
                for now, price in frame.itertuples(index=False, name=None):
                    if not isfinite(price) or price <= 0:
                        raise ValueError("Bar prices must be finite and positive")
                    yield now, symbol, price, now.value == last

    def batches(self, clocks=()):
        with TemporaryDirectory(prefix="portfolio-bars-") as directory:
            yield from self._merge(directory, clocks)

    def _merge(self, directory, clocks):
        streams = [self._rows(*source, Path(directory) / f"{index}.arrow")
                   for index, source in enumerate(self.sources)]
        boundaries = ((now, "", np.nan, False) for now in sorted(set(clocks)))
        try:
            for now, rows in groupby(merge(*streams, boundaries, key=lambda row: row[0]),
                                     key=lambda row: row[0]):
                fresh, terminal = {}, set()
                for _, symbol, price, last in rows:
                    if symbol:
                        fresh[symbol] = price
                        if last:
                            terminal.add(symbol)
                yield now, fresh, terminal
        finally:
            for stream in streams:
                stream.close()


class SessionClock:
    """Map UTC timestamps to cumulative regular-session minutes."""

    def __init__(self, calendar):
        sessions = calendar.sort_values("open")
        self.opens = pd.DatetimeIndex(sessions.open).as_unit("ns").asi8
        self.closes = pd.DatetimeIndex(sessions.close).as_unit("ns").asi8
        self.minutes = np.r_[0., np.cumsum((self.closes - self.opens) / 60e9)]

    def at(self, timestamp):
        value = pd.Timestamp(timestamp).value
        index = bisect_right(self.opens, value) - 1
        if index < 0:
            return 0.0
        return float(self.minutes[index] + (min(value, self.closes[index]) - self.opens[index]) / 60e9)


@dataclass(frozen=True)
class Candidate:
    symbol: str
    target_weight: float
    direction: int
    limit: float
    score: float
    eligible: bool
    event_start: pd.Timestamp

    @property
    def rank(self):
        return -self.score, -self.event_start.value, self.symbol


class CandidateBook:
    """Keep registered candidates and their fixed limits across selection rounds."""

    def __init__(self, settings, clock, end):
        self.settings = settings
        self.clock = clock
        self.origin = clock.at(end)
        self.targets = {}
        self.latest = {}
        self.rows = {}

    def discard(self, symbol):
        """Remove a candidate without restoring it from its still-active signals."""
        self.rows.pop(symbol, None)

    def update_signal(self, symbol, event):
        """Replace a symbol's target with its latest event, or remove it on expiry."""
        if event is None:
            self.targets.pop(symbol, None)
            self.latest.pop(symbol, None)
            return
        signal = event["event_signal"]
        discretized = min(1., max(-1., float(np.rint(signal / self.settings.step_size) * self.settings.step_size)))
        self.targets[symbol] = discretized / (2 * self.settings.max_positions_per_side)
        self.latest[symbol] = event

    def refresh(self, symbol, price):
        """Register a new event snapshot; retain this limit until another event."""
        self.rows.pop(symbol, None)
        event = self.latest.get(symbol)
        if event is not None:
            target = self.targets[symbol]
            direction = (target > 0) - (target < 0)
            w = event["price_w"]
            ceiling, advantage = np.nan, -np.inf
            score = event["meta_pass_probability"] * event["ranking_decay"]
            if direction and isfinite(w) and w > 0 and isfinite(price):
                forecast = event["forecast_price"]
                step = max(-99, min(99, get_target_position(w, forecast, price, 100)))
                if (step > 0) - (step < 0) == direction:
                    ceiling = limit_price(step, 0, forecast, w, 100)
                    advantage = direction * (ceiling - price) / price
            eligible = bool(direction and event["primary_prediction"] == direction
                            and event["meta_prediction"] == 1 and advantage > 0 and score > 0)
            row = Candidate(symbol, target, direction, ceiling, score, eligible, event["event_start"])
            self.rows[symbol] = row

    def top(self, direction, count, excluded):
        if count <= 0:
            return []
        rows = (row for row in self.rows.values()
                if row.eligible and row.direction == direction and row.symbol not in excluded)
        return sorted(rows, key=lambda row: row.rank)[:count]


class SynchronousPriceIndex:
    """Share binary-searchable prices with a bounded 32-symbol mmap cache."""

    def __init__(self, start, end, *, paths=None, prices=None):
        self.start, self.end = start, end
        self.paths, self.prices = paths, prices
        self.directory = TemporaryDirectory(prefix="synchronous-prices-")
        self.files, self.cache = {}, OrderedDict()
        self.counts = {"loads": 0, "next": 0, "marks": 0, "terminal": 0}
        self.preparation_seconds = 0.0
        self.mark_time, self.mark_cache = None, {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        while self.cache:
            self._evict()
        self.directory.cleanup()

    def _evict(self):
        _, arrays = self.cache.popitem(last=False)
        for array in arrays:
            array._mmap.close()

    def _load(self, symbol):
        begun = monotonic()
        if self.paths is not None:
            file = self.paths.feature(symbol, "dollar_bars")
            if not file.exists():
                raise ValueError(f"Missing dollar bars for {symbol}; run dollar-bar preprocessing: {file}")
            frame = pd.read_parquet(file, columns=["end", "close"]).rename(
                columns={"end": "timestamp", "close": "price"})
        else:
            frame = self.prices.loc[self.prices.symbol.eq(symbol), ["timestamp", "price"]]
        if frame.timestamp.isna().any():
            raise ValueError(f"Nonmissing bar end times are required: {symbol}")
        frame = frame.loc[frame.timestamp.between(self.start, self.end)]
        times = pd.DatetimeIndex(frame.timestamp).as_unit("ns").asi8
        values = frame.price.to_numpy(dtype=float)
        if np.any(times[1:] <= times[:-1]):
            raise ValueError(f"Unique increasing bar end times per symbol are required: {symbol}")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Bar prices must be finite and positive")
        prefix = Path(self.directory.name) / str(len(self.files))
        names = str(prefix) + "-time.npy", str(prefix) + "-price.npy"
        np.save(names[0], times)
        np.save(names[1], values)
        self.files[symbol] = names
        self.counts["loads"] += 1
        self.preparation_seconds += monotonic() - begun

    def _arrays(self, symbol):
        if symbol in self.cache:
            self.cache.move_to_end(symbol)
            return self.cache[symbol]
        if symbol not in self.files:
            self._load(symbol)
        if len(self.cache) == 32:
            self._evict()
        arrays = tuple(np.load(name, mmap_mode="r") for name in self.files[symbol])
        self.cache[symbol] = arrays
        return arrays

    def next_after(self, symbol, now):
        self.counts["next"] += 1
        times, prices = self._arrays(symbol)
        position = int(np.searchsorted(times, now.value, side="right"))
        if position == len(times):
            return None
        return pd.Timestamp(int(times[position]), tz="UTC"), float(prices[position])

    def last_at_or_before(self, symbol, now):
        self.counts["marks"] += 1
        if self.mark_time != now:
            self.mark_time, self.mark_cache = now, {}
        if symbol not in self.mark_cache:
            times, prices = self._arrays(symbol)
            position = int(np.searchsorted(times, now.value, side="right")) - 1
            self.mark_cache[symbol] = (None if position < 0 else
                (pd.Timestamp(int(times[position]), tz="UTC"), float(prices[position])))
        return self.mark_cache[symbol]

    def terminal(self, symbol):
        self.counts["terminal"] += 1
        times, _ = self._arrays(symbol)
        return pd.Timestamp(int(times[-1]), tz="UTC") if len(times) else None


class SynchronousScheduler:
    """Dispatch only decisions, fills, expiries and relevant terminal bars."""

    def __init__(self, accounts, prices, decisions, start, end):
        self.keys = list(accounts)
        self.accounts = list(accounts.values())
        self.prices = prices
        self.decisions = set(decisions)
        self.heap, self.active, self.versions = [], {}, {}
        self.order_keys = [set() for _ in self.accounts]
        self.primed = [set() for _ in self.accounts]
        self.calls = self.fills = 0
        for now in self.decisions | {start, end}:
            self._schedule(-1, ("clock", str(now.value)), now)

    def _schedule(self, account, token, now):
        key = account, token
        value = now.value
        if key in self.active and self.active[key][0] == value:
            return
        version = self.versions.get(key, 0) + 1
        self.versions[key] = version
        self.active[key] = value, version
        heappush(self.heap, (value, account, token, version))

    def _prime(self, index, account, symbol, now):
        if symbol in self.primed[index]:
            return None
        self.primed[index].add(symbol)
        last = self.prices.terminal(symbol)
        if last is None:
            account.unavailable.add(symbol)
            account.terminated.add(symbol)
        elif last <= now:
            account.terminated.add(symbol)
            return last
        else:
            self._schedule(index, ("terminal", symbol), last)
        return None

    def _orders(self, index, account, now):
        desired = set()
        for group, book in account.books.items():
            for symbol, order in book["pending"].items():
                token = "expiry", group, symbol
                desired.add(token)
                self._schedule(index, token, order["exit"])
                token = "entry", group, symbol
                if (index, token) in self.active:
                    desired.add(token)
                else:
                    quote = self.prices.next_after(symbol, order["decision"])
                    if quote is not None and quote[0] < order["exit"]:
                        desired.add(token)
                        self._schedule(index, token, quote[0])
            for symbol, position in book["positions"].items():
                token = "exit", group, symbol
                if (index, token) in self.active:
                    desired.add(token)
                else:
                    quote = self.prices.next_after(symbol, position["exit"])
                    if quote is not None:
                        desired.add(token)
                        self._schedule(index, token, quote[0])
                if position["exit"] > now:
                    token = "exit_decision", group, symbol
                    desired.add(token)
                    self._schedule(index, token, position["exit"])
        for token in self.order_keys[index] - desired:
            self.active.pop((index, token), None)
        self.order_keys[index] = desired
        if len(self.heap) > 4 * len(self.active) + 64:
            self.heap = [(value, account_id, token, version)
                         for (account_id, token), (value, version) in self.active.items()]
            heapify(self.heap)

    def run(self, show_progress=False):
        begun = monotonic()
        with tqdm(total=len(self.decisions), desc="Synchronous decisions", unit="decision",
                  disable=not show_progress) as progress:
            last_refresh = monotonic()
            progress.set_postfix_str(f"accounts={len(self.accounts)}; fills=0", refresh=False)
            while self.heap:
                value = self.heap[0][0]
                now = pd.Timestamp(value, tz="UTC")
                tasks, clock = {}, False
                while self.heap and self.heap[0][0] == value:
                    _, index, token, version = heappop(self.heap)
                    key = index, token
                    if self.active.get(key) != (value, version):
                        continue
                    del self.active[key]
                    if index == -1:
                        clock = True
                    else:
                        tasks.setdefault(index, []).append(token)
                if clock:
                    for index in range(len(self.accounts)):
                        tasks.setdefault(index, [])
                for index in sorted(tasks):
                    account = self.accounts[index]
                    tokens = tasks[index]
                    terminal = {token[1] for token in tokens if token[0] == "terminal"}
                    for group in account.books:
                        for row in account.rounds.get((now, group), ()):
                            last = self._prime(index, account, row["symbol"], now)
                            if last == now:
                                terminal.add(row["symbol"])
                    symbols = terminal | {token[2] for token in tokens if token[0] in {"entry", "exit"}}
                    fresh = {}
                    for symbol in symbols:
                        quote = self.prices.last_at_or_before(symbol, now)
                        if quote is not None and quote[0] == now:
                            fresh[symbol] = quote[1]
                    held = {symbol for book in account.books.values() for symbol in book["positions"]}
                    for symbol in held:
                        quote = self.prices.last_at_or_before(symbol, now)
                        if quote is not None:
                            account.marks[symbol] = quote[1]
                    previous_fills = len(account.trades)
                    account.observe(now, fresh, set(fresh), terminal)
                    account._record(now)
                    self.fills += len(account.trades) - previous_fills
                    self.calls += 1
                    self._orders(index, account, now)
                if clock and now in self.decisions:
                    progress.update(1)
                if show_progress and monotonic() - last_refresh >= .5:
                    progress.set_postfix_str(f"accounts={len(self.accounts)}; fills={self.fills}; {now}")
                    last_refresh = monotonic()
            progress.set_postfix_str(f"accounts={len(self.accounts)}; fills={self.fills}; {now}", refresh=False)
            progress.set_description("Finalizing synchronous accounts")
            finished = monotonic()
            results = {key: account.finish() for key, account in zip(self.keys, self.accounts)}
        self.timing = {"total_seconds": monotonic() - begun,
                       "finalize_seconds": monotonic() - finished,
                       "price_preparation_seconds": self.prices.preparation_seconds}
        return results


@dataclass(frozen=True)
class PortfolioSettings:
    """Portfolio settings with a position limit per direction across all books."""

    max_positions_per_side: int = 10
    step_size: float = 0.10
    initial_aum: float = 100_000.0
    broker_fee_bps: float = 1.0
    slippage_bps: float = 1.0
    half_life_minutes: float = 390.0
    horizon_bars: int = 1000
    horizontal_multipliers: tuple[float, float] = (1.0, 1.0)


def calibrate_price_sizing(development: pd.DataFrame) -> pd.DataFrame:
    """Freeze per-symbol median forecast divergence using development only."""
    if not development.partition.eq("development").all():
        raise ValueError("Price calibration must not include holdout")
    rows = []
    for symbol, group in development.groupby("symbol"):
        divergence = (group.entry_price * group.horizon_volatility).abs()
        value = float(divergence.median())
        valid = np.isfinite(value) and value > 0
        rows.append({"symbol": symbol, "w": get_w(value, 0.5) if valid else np.nan,
                     "reason": "" if valid else "no positive development divergence"})
    return pd.DataFrame(rows)


def simulate_cross_sectional(
    events: pd.DataFrame,
    observations: Iterable[tuple[pd.Timestamp, pd.DataFrame]],
    calibration: pd.DataFrame,
    calendar: pd.DataFrame,
    end: pd.Timestamp,
    settings: PortfolioSettings,
    evaluation_partition: str = "holdout",
) -> dict[str, pd.DataFrame]:
    """Consume completed bars with symbol, price and is_last columns.

    Every row is an executable bar close and ages its symbol's active events.
    A new event supersedes all older signals for its symbol. Each admitted
    direction shares half of AUM equally, capped at 1/K per symbol. Unused
    capacity remains in cash.
    Persistent candidates are registered on new events with fixed price limits.
    Only initial admission, event exits, and completed liquidations select new
    symbols. Other events update orders; limit violations remove candidates
    without replacement. Removed candidates require a new event to return.
    is_last marks the final bar inside the evaluation interval, forcing liquidation.
    """
    account = _CrossSectionalAccount(events, calibration, calendar, end, settings, evaluation_partition)
    for now, quotes in observations:
        if now > end:
            break
        _validate_bar_prices(quotes)
        if "is_last" not in quotes or not pd.api.types.is_bool_dtype(quotes.is_last) or quotes.is_last.isna().any():
            raise ValueError("Completed bar observations require boolean is_last")
        if quotes.symbol.duplicated().any():
            raise ValueError("Duplicate symbol at bar end")
        account.observe(now, dict(zip(quotes.symbol, quotes.price)),
                        set(quotes.symbol), set(quotes.loc[quotes.is_last, "symbol"]))
    return account.finish()


def _begin_observation(account, now):
    previous = account.previous_time
    if previous is not None:
        if now <= previous:
            raise ValueError("Observation batches must have strictly increasing times")
        if now.value // 60_000_000_000 != previous.value // 60_000_000_000:
            account._record(previous)
    account.previous_time = now


class _CrossSectionalAccount:
    """Independent account state consuming a shared, validated market stream."""

    def __init__(self, events, calibration, calendar, end, settings, evaluation_partition, clock=None):
        from src.backtesting.bet_sizing import _validate_event_bet_inputs, event_bet_signals
        if (settings.initial_aum <= 0 or settings.max_positions_per_side < 1 or not 0 < settings.step_size <= 1
                or min(settings.broker_fee_bps, settings.slippage_bps) < 0):
            raise ValueError("Positive initial assets and max_positions_per_side are required")
        events = events.loc[events.partition.eq(evaluation_partition)].sort_values(
            ["event_start", "symbol"]
        ).copy()
        if events.empty or events.duplicated(["symbol", "event_start"]).any():
            raise ValueError("Nonempty unique evaluation event keys are required")
        if "event_signal" in events:
            _validate_event_bet_inputs(events)
        else:
            events["event_signal"] = event_bet_signals(events)
        if events[["event_start", "vertical_barrier", "horizon_volatility"]].isna().any().any():
            raise ValueError("Signals require barriers and causal volatility")
        calibration_columns = {"symbol", "event_start", "w"}
        if not calibration_columns.issubset(calibration.columns):
            raise ValueError("Calibration requires symbol, event_start, and w")
        if calibration.duplicated(["symbol", "event_start"]).any():
            raise ValueError("Calibration event keys must be unique")
        self.w = calibration.set_index(["symbol", "event_start"])["w"].copy()
        event_keys = pd.MultiIndex.from_frame(events[["symbol", "event_start"]])
        missing = events.loc[self.w.reindex(event_keys).isna().to_numpy(), ["symbol", "event_start"]]
        self.exclusions = missing.assign(
            reason="missing development price calibration"
        ).to_dict("records")
        self.clock = clock if clock is not None else SessionClock(calendar)
        self.candidates = CandidateBook(settings, self.clock, end)
        columns = ["symbol", "event_start", "vertical_barrier", "horizon_volatility",
                   "primary_prediction", "meta_prediction", "meta_pass_probability", "event_signal"]
        if "horizon_bars" in events:
            columns.append("horizon_bars")
        self.stream = events[columns].to_dict("records")
        calibration_values = self.w.to_dict()
        for event in self.stream:
            event["price_w"] = calibration_values.get((event["symbol"], event["event_start"]), np.nan)
            event["ranking_decay"] = 2 ** ((self.clock.at(event["event_start"]) - self.candidates.origin)
                                          / settings.half_life_minutes)
        self.event_position = 0
        self.active: dict[tuple, dict] = {}
        self.prices, self.quantity, self.pending, self.applied = {}, {}, {}, {}
        self.position_limits = {}
        self.waiting = set()
        self.initial_selection_done = False
        self.cash = settings.initial_aum
        self.fee_rate, self.slip_rate = settings.broker_fee_bps / 10_000, settings.slippage_bps / 10_000
        self.records, self.fills, self.exposures = [], [], []
        self.realized_cost = {}
        self.lot_cash = {}
        self.episodes = {}
        self.closed = []
        self.previous_time = None
        self.terminated = set()
        self.cutoff = events.event_start.min()
        self.cumulative_fee = self.cumulative_slip = self.cumulative_turnover = 0.0

        self.settings = settings
        self.end = end
        self.active_by_symbol: dict[str, tuple[str, pd.Timestamp]] = {}
        self.expiries = []

    def aum(self):
        return self.cash + sum(q * self.prices[s] for s, q in self.quantity.items() if q)

    def _target_quantity(self, symbol, weight):
        price = self.prices[symbol]
        old = self.quantity.get(symbol, 0.0)
        wealth = self.aum()
        if wealth <= 0:
            raise ValueError("Account equity exhausted")
        if weight == 0:
            return 0.0
        if self.fee_rate == 0 and self.slip_rate == 0:
            return weight * wealth / price
        # Solve target quantity against post-cost equity. Costs include adverse
        # execution slippage and broker fees assessed at the execution price.
        low, high = 0.0, wealth
        for _ in range(60):
            post = (low + high) / 2
            q = weight * post / price
            delta = q - old
            execution = price * (1 + np.sign(delta) * self.slip_rate)
            cost = abs(delta) * price * self.slip_rate + abs(delta * execution) * self.fee_rate
            if post + cost > wealth:
                high = post
            else:
                low = post
        return weight * ((low + high) / 2) / price

    def _increasing(self, symbol, weight, *, new_quantity=None):
        old = self.quantity.get(symbol, 0.0)
        new = self._target_quantity(symbol, weight) if new_quantity is None else new_quantity
        return old * new < 0 or abs(new) > abs(old) + 1e-12

    def _reducing(self, symbol, weight):
        old = self.quantity.get(symbol, 0.0)
        new = self._target_quantity(symbol, weight)
        return old * new < 0 or abs(new) < abs(old) - 1e-12

    def _violates_limit(self, symbol, order, *, new_quantity=None):
        new = self._target_quantity(symbol, order["weight"]) if new_quantity is None else new_quantity
        if not self._increasing(symbol, order["weight"], new_quantity=new):
            return False
        delta = new - self.quantity.get(symbol, 0.0)
        direction = np.sign(delta)
        execution = self.prices[symbol] * (1 + direction * self.slip_rate)
        return not isfinite(order["limit"]) or direction * (execution - order["limit"]) > 0

    def fill(self, symbol, weight, time, reason, *, new_quantity=None):
        price = self.prices[symbol]
        old = self.quantity.get(symbol, 0.0)
        wealth = self.aum()
        new = self._target_quantity(symbol, weight) if new_quantity is None else new_quantity
        delta = new - old
        if abs(delta) < 1e-12:
            return
        if not old and new:
            self.position_limits[symbol] = self.pending[symbol]["limit"]
            self.episodes[symbol] = {
                "event_start": time,
                "side": float(np.sign(new)),
                "entry_aum": wealth,
            }
        execution = price * (1 + np.sign(delta) * self.slip_rate)
        fee, slip = abs(delta * execution) * self.fee_rate, abs(delta) * price * self.slip_rate
        self.cash -= delta * execution + fee
        if new:
            self.quantity[symbol] = new
        else:
            self.quantity.pop(symbol, None)
            self.position_limits.pop(symbol, None)
        self.lot_cash[symbol] = self.lot_cash.get(symbol, 0.0) - delta * execution - fee
        self.realized_cost[symbol] = self.realized_cost.get(symbol, 0.0) + fee + slip
        self.cumulative_fee += fee
        self.cumulative_slip += slip
        self.cumulative_turnover += abs(delta * price)
        self.fills.append({"timestamp": time, "symbol": symbol, "quantity_change": delta,
                      "price": price, "execution_price": execution, "broker_fee": fee,
                      "slippage": slip, "traded_value": abs(delta * price), "reason": reason})
        if not new:
            net_pnl = self.lot_cash.pop(symbol, 0.0)
            execution_cost = self.realized_cost.pop(symbol, 0.0)
            episode = self.episodes.pop(symbol)
            self.closed.append({
                **episode,
                "event_end": time,
                "timestamp": time,
                "symbol": symbol,
                "gross_pnl": net_pnl + execution_cost,
                "execution_cost": execution_cost,
                "net_pnl": net_pnl,
                "net_return": net_pnl / episode["entry_aum"],
            })

    def observe(self, now: pd.Timestamp, fresh: dict[str, float], completed_bars: set[str],
                terminal: set[str] = frozenset()):
        """Process a batch and report whether symbol subscriptions may have changed."""
        now = pd.Timestamp(now)
        _begin_observation(self, now)
        if now > self.end:
            return
        fills_before = len(self.fills)
        pending_before = len(self.pending)
        changed = set(terminal)
        slot_returned = False
        event_ended = False
        self.prices.update(fresh)
        self.terminated.update(terminal)
        for symbol in sorted(terminal):
            self.pending.pop(symbol, None)
            self.candidates.discard(symbol)
            self.waiting.discard(symbol)
            key = self.active_by_symbol.pop(symbol, None)
            if key is not None:
                del self.active[key]
            if self.quantity.get(symbol, 0):
                self.fill(symbol, 0.0, now, "research_end")
                slot_returned = True
        if now < self.cutoff:
            return
        if not fresh and not completed_bars:
            self._record(now)
            return
        # Visit only changed symbols and due time-based expiries.
        affected = set()
        for symbol in fresh.keys() | completed_bars:
            key = self.active_by_symbol.get(symbol)
            if key is not None:
                affected.add(key)
        while self.expiries and self.expiries[0][0] <= now:
            _, key = heappop(self.expiries)
            if key in self.active:
                affected.add(key)
        for key in affected:
            event = self.active[key]
            symbol = event["symbol"]
            hit = False
            if symbol in fresh:
                observed_return = self.prices[symbol] / event["entry_price"] - 1
                upper, lower = self.settings.horizontal_multipliers
                volatility = event["horizon_volatility"]
                hit = ((upper > 0 and observed_return > upper * volatility)
                       or (lower > 0 and observed_return < -lower * volatility))
            if symbol in completed_bars and "remaining_bars" in event:
                event["remaining_bars"] -= 1
            vertical_hit = (event["remaining_bars"] <= 0 if "remaining_bars" in event
                            else now >= event["vertical_barrier"])
            if vertical_hit or hit:
                event_ended = True
                changed.add(symbol)
                del self.active[key]
                del self.active_by_symbol[symbol]
        registered = set()
        while self.event_position < len(self.stream) and self.stream[self.event_position]["event_start"] <= now:
            event = self.stream[self.event_position].copy()
            self.event_position += 1
            symbol = event["symbol"]
            if event["event_start"] != now or symbol not in fresh:
                raise ValueError(f"Missing exact observed event-start price: {symbol} {event['event_start']}")
            if symbol in self.terminated:
                continue
            event["entry_price"] = self.prices[symbol]
            event["forecast_price"] = event["entry_price"] * (
                1 + event["primary_prediction"] * event["horizon_volatility"])
            changed.add(symbol)
            registered.add(symbol)
            if "horizon_bars" in event:
                event["remaining_bars"] = int(event["horizon_bars"])
            key = (symbol, event["event_start"])
            # A fresh prediction supersedes the previous prediction for this symbol.
            old_key = self.active_by_symbol.pop(symbol, None)
            if old_key is not None:
                del self.active[old_key]
            self.active[key] = event
            self.active_by_symbol[symbol] = key
            if "remaining_bars" not in event:
                heappush(self.expiries, (event["vertical_barrier"], key))
            elif event["remaining_bars"] <= 0:
                # Match expiry on the observation after registration, even without a bar.
                heappush(self.expiries, (now, key))
        for symbol in changed:
            key = self.active_by_symbol.get(symbol)
            self.candidates.update_signal(symbol, self.active.get(key))
        # Only new events register candidates. Expiries can invalidate an existing
        # row, but must not resurrect a previously discarded event or older signal.
        for symbol in changed:
            if symbol in registered:
                self.candidates.refresh(symbol, self.prices[symbol])
            row = self.candidates.rows.get(symbol)
            if (row is not None and self.quantity.get(symbol, 0.0) * row.direction > 0
                    and isfinite(row.limit)):
                self.position_limits[symbol] = row.limit
            event = self.candidates.latest.get(symbol)
            if row is not None and (not row.eligible or event is None
                    or row.event_start != event["event_start"]
                    or row.direction * self.candidates.targets.get(symbol, 0.0) <= 0):
                self.candidates.discard(symbol)

        removed = False
        for symbol in fresh:
            row = self.candidates.rows.get(symbol)
            if row is not None and not self.quantity.get(symbol, 0.0):
                execution = self.prices[symbol] * (1 + row.direction * self.slip_rate)
                if row.direction * (execution - row.limit) > 0:
                    self.candidates.discard(symbol)
                    self.waiting.discard(symbol)
                    self.pending.pop(symbol, None)
                    removed = True
            order = self.pending.get(symbol)
            if order is not None and symbol not in registered and self._violates_limit(symbol, order):
                self.pending.pop(symbol)
                self.candidates.discard(symbol)
                self.waiting.discard(symbol)
                removed = True
        if changed or removed:
            self._select_orders(now, allow_selection=event_ended)

        # Expired signals cannot execute an old order before reselection.
        # Only orders decided before now may consume this batch.
        for symbol, order in list(self.pending.items()):
            if symbol not in fresh or now <= order["decision"]:
                continue
            weight = order["weight"]
            old = self.quantity.get(symbol, 0.0)
            if old and weight and np.sign(old) != np.sign(weight):
                self.fill(symbol, 0.0, now, "direction_exit")
                self.applied[symbol] = 0.0
                slot_returned = True
                continue
            new = self._target_quantity(symbol, weight)
            if self._increasing(symbol, weight, new_quantity=new):
                continue
            self.fill(symbol, weight, now, "signal_reduction", new_quantity=new)
            slot_returned |= bool(old and not self.quantity.get(symbol, 0.0))
            self.applied[symbol] = weight
            self.pending.pop(symbol, None)
        # A reduction on an unobserved symbol blocks increases account-wide.
        for symbol, order in list(self.pending.items()):
            if symbol not in fresh or now <= order["decision"]:
                continue
            if any(self._reducing(s, pending["weight"]) for s, pending in self.pending.items()):
                break
            new = self._target_quantity(symbol, order["weight"])
            if self._violates_limit(symbol, order, new_quantity=new):
                continue
            self.fill(symbol, order["weight"], now, "signal_entry", new_quantity=new)
            self.applied[symbol] = order["weight"]
            self.pending.pop(symbol, None)
        if slot_returned:
            self._select_orders(now, allow_selection=True)
        if len(self.fills) != fills_before or terminal:
            self._record(now)
        return bool(changed or removed or slot_returned or len(self.pending) != pending_before)

    def _select_orders(self, now, *, allow_selection=False):
        """Update orders; select new symbols only initially or after event exits.

        Equal directional weights are capped at 1/K per symbol. Vacant capacity
        stays in cash; probability changes alone do not resize positions.
        """
        signals = self.candidates.targets
        held = {symbol for symbol, q in self.quantity.items() if q}
        selected = {symbol for symbol in held
                    if self.quantity[symbol] * signals.get(symbol, 0.0) > 0}
        vacant = {direction: max(0, self.settings.max_positions_per_side
                  - sum(self.quantity[symbol] * direction > 0 for symbol in held))
                  for direction in (-1, 1)}
        if allow_selection or not self.initial_selection_done:
            for direction in (-1, 1):
                selected.update(row.symbol for row in self.candidates.top(direction, vacant[direction], held))
            if selected - held:
                self.initial_selection_done = True
        else:
            # Registrations update the queue, not the selected set. A canceled
            # reservation stays vacant until an event exit permits selection.
            selected.update(symbol for symbol, order in self.pending.items()
                            if symbol not in held and symbol in self.candidates.rows
                            and self.candidates.rows[symbol].eligible
                            and order["weight"] * signals.get(symbol, 0.0) > 0)

        self.waiting = {row.symbol for direction in (-1, 1)
                        for row in self.candidates.top(direction, self.settings.max_positions_per_side,
                                                       held | selected)}
        for symbol in set(self.candidates.rows) - held - selected - self.waiting:
            self.candidates.discard(symbol)

        desired = {}
        for direction in (-1, 1):
            symbols = [symbol for symbol in selected if signals[symbol] * direction > 0]
            if symbols:
                weight = direction * min(1 / self.settings.max_positions_per_side, 0.5 / len(symbols))
                desired.update(dict.fromkeys(symbols, weight))
        planned = {}
        for symbol in sorted(held | selected):
            target = desired.get(symbol, 0.0)
            previous = self.applied.get(symbol, 0.0)
            if target == previous:
                continue
            new = self._target_quantity(symbol, target)
            increasing = self._increasing(symbol, target, new_quantity=new)
            if symbol in held:
                price_limit = self.position_limits.get(symbol, np.nan)
            else:
                row = self.candidates.rows.get(symbol)
                if row is None or not row.eligible:
                    continue
                price_limit = row.limit
            order = {"decision": now, "weight": target, "limit": price_limit,
                     "reason": "signal_entry" if increasing else "signal_reduction"}
            if not self._violates_limit(symbol, order, new_quantity=new):
                planned[symbol] = order
        # Preserve both decision time and insertion order for unchanged orders.
        pending = {}
        for symbol, order in self.pending.items():
            replacement = planned.pop(symbol, None)
            if replacement is None:
                continue
            unchanged = (order["weight"] == replacement["weight"]
                         and (order["weight"] == 0 or order["limit"] == replacement["limit"]))
            pending[symbol] = order if unchanged else replacement
        pending.update(planned)
        self.pending = pending

    def _record(self, now):
        if now < self.cutoff or (self.records and self.records[-1]["timestamp"] == now):
            return
        wealth = self.aum()
        gross = sum(abs(q * self.prices[s]) for s, q in self.quantity.items() if q)
        net = sum(q * self.prices[s] for s, q in self.quantity.items() if q)
        self.records.append({"timestamp": now, "cash": self.cash, "aum": wealth, "gross_exposure": gross / wealth,
                        "net_exposure": net / wealth, "broker_fee": self.cumulative_fee,
                        "slippage": self.cumulative_slip, "traded_value": self.cumulative_turnover})
        for symbol, q in self.quantity.items():
            if q:
                self.exposures.append({"timestamp": now, "symbol": symbol, "quantity": q,
                                  "position_value": q * self.prices[symbol], "weight": q * self.prices[symbol] / wealth})


    def finish(self) -> dict[str, pd.DataFrame]:
        if self.previous_time is not None:
            self._record(self.previous_time)
        if any(self.quantity.values()):
            raise ValueError("Research end has unliquidated positions; supply liquidation quotes before end")
        if self.event_position != len(self.stream):
            raise ValueError("Market stream ended before all holdout events were observed")
        closed_columns = [
            "event_start", "event_end", "timestamp", "symbol", "side", "entry_aum",
            "gross_pnl", "execution_cost", "net_pnl", "net_return",
        ]
        return {"ledger": pd.DataFrame(self.records), "trades": pd.DataFrame(self.fills),
                "closed_trades": pd.DataFrame(self.closed, columns=closed_columns),
                "exposures": pd.DataFrame(self.exposures),
                "exclusions": pd.DataFrame(self.exclusions)}


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


def _validate_bar_prices(prices):
    if not np.isfinite(prices.price).all() or not prices.price.gt(0).all():
        raise ValueError("Bar prices must be finite and positive")


def load_bar_prices(paths, start, end, symbols):
    """Read completed dollar bars once per symbol, never raw trades."""
    frames = []
    for symbol in sorted(set(symbols)):
        file = paths.feature(symbol, "dollar_bars")
        if not file.exists():
            raise ValueError(f"Missing dollar bars for {symbol}; run dollar-bar preprocessing: {file}")
        frame = pd.read_parquet(file, columns=["end", "close"],
                                filters=[("end", ">=", start), ("end", "<=", end)])
        frame = frame.rename(columns={"end": "timestamp", "close": "price"})
        frame["symbol"] = symbol
        frames.append(frame)
    prices = (pd.concat(frames, ignore_index=True) if frames else
              pd.DataFrame(columns=["timestamp", "price", "symbol"]))
    return _prepare_bar_prices(prices, start, end)


def _prepare_bar_prices(prices, start, end):
    if prices.timestamp.isna().any():
        raise ValueError("Nonmissing bar end times are required")
    prices = prices.loc[prices.timestamp.between(start, end)].copy()
    if prices.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Unique nonmissing bar end times per symbol are required")
    _validate_bar_prices(prices)
    prices = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    prices["is_last"] = ~prices.symbol.duplicated(keep="last")
    return prices


def _bar_batches(prices, clocks=()):
    """Merge decision clocks and bar closes without constructing per-time frames."""
    if isinstance(prices, BarStream):
        yield from prices.batches(clocks)
        return
    rows = iter(prices[["timestamp", "symbol", "price", "is_last"]].itertuples(index=False, name=None))
    row = next(rows, None)
    clock = iter(sorted(set(clocks)))
    boundary = next(clock, None)
    while row is not None or boundary is not None:
        now = boundary if row is None else row[0] if boundary is None else min(row[0], boundary)
        fresh, terminal = {}, set()
        while row is not None and row[0] == now:
            _, symbol, price, last = row
            fresh[symbol] = price
            if last:
                terminal.add(symbol)
            row = next(rows, None)
        if boundary == now:
            boundary = next(clock, None)
        yield now, fresh, terminal


def observation_stream(paths, start: pd.Timestamp, end: pd.Timestamp, *,
                       symbols: Sequence[str]):
    """Yield completed dollar-bar closes with symbol, price and is_last columns."""
    prices = BarStream(paths, start, end, symbols)
    for now, fresh, terminal in _bar_batches(prices):
        yield now, pd.DataFrame([{"symbol": symbol, "price": price, "is_last": symbol in terminal}
                                 for symbol, price in fresh.items()])


class _AsynchronousDispatcher:
    """Share price marks and dispatch only bars that can change an account."""

    def __init__(self, accounts):
        self.accounts = list(accounts.values())
        self.prices, self.subscribers = {}, {}
        self.symbols = [set() for _ in self.accounts]
        registrations = {}
        for index, account in enumerate(self.accounts):
            account.prices = self.prices
            for event in account.stream:
                registrations.setdefault(event["event_start"].value, set()).add(index)
        self.registrations = sorted(registrations.items())
        self.registration_position = 0
        self.expiries, self.next_expiry = [], {}
        self.previous_time = None

    def observe(self, now, fresh, completed, terminal):
        previous = self.previous_time
        if previous is not None and now.value // 60_000_000_000 != previous.value // 60_000_000_000:
            # Flush with the previous marks, even when the last bar was unrelated.
            for account in self.accounts:
                account._record(previous)
        self.prices.update(fresh)
        tasks = set()
        if not fresh or terminal:
            tasks.update(range(len(self.accounts)))
        if fresh:
            for symbol in fresh.keys() | completed:
                tasks.update(self.subscribers.get(symbol, ()))
            while (self.registration_position < len(self.registrations)
                   and self.registrations[self.registration_position][0] <= now.value):
                tasks.update(self.registrations[self.registration_position][1])
                self.registration_position += 1
            while self.expiries and self.expiries[0][0] <= now.value:
                value, index = heappop(self.expiries)
                if self.next_expiry.get(index) == value:
                    self.next_expiry.pop(index)
                    tasks.add(index)
        for index in sorted(tasks):
            account = self.accounts[index]
            account.previous_time = previous
            if account.observe(now, fresh, completed, terminal):
                symbols = account.active_by_symbol.keys() | account.pending.keys() | account.candidates.rows.keys()
                for symbol in self.symbols[index] - symbols:
                    self.subscribers[symbol].remove(index)
                    if not self.subscribers[symbol]:
                        del self.subscribers[symbol]
                for symbol in symbols - self.symbols[index]:
                    self.subscribers.setdefault(symbol, set()).add(index)
                self.symbols[index] = symbols
            if account.expiries:
                value = account.expiries[0][0].value
                if self.next_expiry.get(index) != value:
                    self.next_expiry[index] = value
                    heappush(self.expiries, (value, index))
            else:
                self.next_expiry.pop(index, None)
        self.previous_time = now


def _run_asynchronous_accounts(accounts, prices, start, end, settings, *, clocks=(), show_progress=False):
    """Dispatch a shared bar stream to asynchronous accounts with row-level progress."""
    dispatcher = _AsynchronousDispatcher(accounts)
    with tqdm(total=len(prices), desc="Dollar bars", unit="bar", disable=not show_progress) as progress:
        progress.set_postfix_str(f"accounts={len(accounts)}; {start}", refresh=False)
        last_refresh = monotonic()
        progress.set_description("Preparing Arrow batches" if isinstance(prices, BarStream) else "Dollar bars")
        with closing(_bar_batches(prices, clocks)) as observations:
            first = True
            for now, fresh, terminal in observations:
                if first:
                    progress.set_description("Dollar bars")
                    first = False
                completed = set(fresh)
                dispatcher.observe(now, fresh, completed, terminal)
                progress.update(len(fresh))
                if show_progress and monotonic() - last_refresh >= 0.5:
                    progress.set_postfix_str(f"accounts={len(accounts)}; {now}")
                    last_refresh = monotonic()
        for account in accounts.values():
            account.previous_time = dispatcher.previous_time
        del dispatcher
        progress.set_description("Finalizing accounts")
        results = {}
        for key in list(accounts):
            account = accounts.pop(key)
            result = account.finish()
            del account
            ledger = result["ledger"]
            if ledger.empty or ledger.iloc[0].timestamp > start:
                initial = {"timestamp": start, "cash": settings.initial_aum, "aum": settings.initial_aum,
                           "gross_exposure": 0.0, "net_exposure": 0.0,
                           "broker_fee": 0.0, "slippage": 0.0, "traded_value": 0.0}
                ledger = pd.concat([pd.DataFrame([initial]), ledger], ignore_index=True)
            if ledger.iloc[-1].timestamp < end:
                ledger = pd.concat([ledger, ledger.tail(1).assign(timestamp=end)], ignore_index=True)
            result["ledger"] = ledger
            results[key] = result
        return results


def _simulate_asynchronous_accounts(paths, prepared, calendar, start, end, settings,
                                    evaluation_partition, *, symbols, show_progress=False):
    """Share completed dollar bars between independent family/path accounts."""
    accounts = {}
    clock = SessionClock(calendar)
    for key, item in prepared.items():
        events = item["events"]
        selected = events.loc[events.partition.eq(evaluation_partition)
                              & events.event_start.between(start, end)].copy()
        accounts[key] = _CrossSectionalAccount(
            selected, item["calibration"], calendar, end, settings, evaluation_partition, clock,
        )
    if not accounts:
        return {}
    if show_progress:
        tqdm.write("Preparing dollar bars")
    prices = BarStream(paths, start, end, symbols)
    return _run_asynchronous_accounts(accounts, prices, start, end, settings,
                         clocks=_recording_clocks(calendar, start, end), show_progress=show_progress)


def benchmark_returns(paths, start: pd.Timestamp, end: pd.Timestamp, *,
                      data_start: pd.Timestamp, benchmark_symbol: str) -> pd.Series:
    """Compute daily returns from the last completed dollar-bar close."""
    prices = load_bar_prices(paths, data_start, end, [benchmark_symbol])
    frame = prices.set_index("timestamp").sort_index()
    anchor = frame.loc[frame.index <= start].tail(1)
    after = frame.loc[frame.index > start]
    if anchor.empty or after.empty:
        raise ValueError(f"{benchmark_symbol} does not cover the evaluation interval")
    daily = pd.concat([anchor, after]).price.resample("D").last().dropna()
    returns = daily.pct_change()
    returns.iloc[0] = 0.0
    return returns


def entry_schedule(calendar: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Map each session boundary to its information cutoff and scheduled exit."""
    rows = []
    calendar = calendar.sort_values("open").reset_index(drop=True)
    for i, session in calendar.iterrows():
        for group, entry, exit_time in (
            ("open", session.open, session.close),
            ("close", session.close, calendar.iloc[i + 1].close if i + 1 < len(calendar) else pd.NaT),
        ):
            rows.append({"group": group, "entry": entry,
                         "cutoff": entry - pd.Timedelta(minutes=1), "exit": exit_time})
    schedule = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    return schedule.loc[schedule.entry.le(end)].reset_index(drop=True)


def _synchronous_book_limit(max_positions_per_side: int) -> int:
    """Split the total directional limit equally between open and close books."""
    if (isinstance(max_positions_per_side, bool)
            or not isinstance(max_positions_per_side, (int, np.integer))
            or max_positions_per_side < 2 or max_positions_per_side % 2):
        raise ValueError("Synchronous max_positions_per_side must be a positive even integer")
    return max_positions_per_side // 2


def select_synchronous(events: pd.DataFrame, schedule: pd.DataFrame,
                       max_positions_per_side: int = 10) -> pd.DataFrame:
    """Rank latest symbol signals, using half the total directional limit per book."""
    book_limit = _synchronous_book_limit(max_positions_per_side)
    if events.duplicated(["symbol", "event_start"]).any():
        raise ValueError("Unique event keys are required")
    if not events.meta_pass_probability.between(0, 1).all():
        raise ValueError("Meta probabilities must be between zero and one")
    events = events.sort_values(["event_start", "symbol"]).copy()
    positions = pd.DatetimeIndex(schedule.cutoff).searchsorted(events.event_start, side="left")
    valid = positions < len(schedule)
    events = events.loc[valid].copy()
    events["round"] = positions[valid]
    latest = events.drop_duplicates(["round", "symbol"], keep="last")
    selected = latest.loc[latest.meta_prediction.eq(1) & latest.primary_prediction.isin([-1, 1])]
    selected = selected.sort_values(["meta_pass_probability", "symbol"], ascending=[False, True])
    selected = selected.groupby(["round", "primary_prediction"], sort=False).head(book_limit).copy()
    selected["target_weight"] = selected.primary_prediction / (
        2 * selected.groupby(["round", "primary_prediction"]).symbol.transform("size")
    )
    return selected.merge(schedule.rename_axis("round").reset_index(), on="round", validate="many_to_one")


class _SynchronousAccount:
    """Two independent books, with delayed fills and flat-before-reentry rounds."""

    def __init__(self, events, calendar, start, end, settings, evaluation_partition, selected=None):
        _synchronous_book_limit(settings.max_positions_per_side)
        if settings.initial_aum <= 0 or min(settings.broker_fee_bps, settings.slippage_bps) < 0:
            raise ValueError("Positive equity and nonnegative costs are required")
        if start >= end:
            raise ValueError("Ordered evaluation bounds are required")
        if selected is None:
            schedule = entry_schedule(calendar, end)
            signals = events.loc[events.partition.eq(evaluation_partition) & events.event_start.between(start, end)]
            selected = select_synchronous(signals, schedule, settings.max_positions_per_side)
        selected = selected.loc[selected.exit.notna() & selected.exit.le(end)
                                & selected.entry.between(start, end)]
        columns = ["symbol", "entry", "exit", "target_weight"]
        self.rounds = {(entry, group): frame[columns].to_dict("records")
                       for (entry, group), frame in selected.groupby(["entry", "group"])}
        self.books = {group: {"cash": settings.initial_aum / 2, "positions": {},
                             "waiting": [], "pending": {}}
                      for group in ("open", "close")}
        self.marks, self.terminated = {}, set()
        self.unavailable = set()
        self.records, self.group_records, self.trades = [], [], []
        self.closed, self.exposures, self.exclusions = [], [], []
        self.fee_total = self.slip_total = self.turnover = 0.0
        self.fee_rate, self.slip_rate = settings.broker_fee_bps / 10000, settings.slippage_bps / 10000
        self.previous_time = None

    def fill(self, group, symbol, delta, price, now, reason):
        execution = price * (1 + np.sign(delta) * self.slip_rate)
        fee, slip = abs(delta * execution) * self.fee_rate, abs(delta * price) * self.slip_rate
        cash_change = -delta * execution - fee
        self.books[group]["cash"] += cash_change
        self.fee_total += fee
        self.slip_total += slip
        self.turnover += abs(delta * price)
        self.trades.append({"timestamp": now, "group": group, "symbol": symbol,
                       "quantity_change": delta, "price": price, "execution_price": execution,
                       "broker_fee": fee, "slippage": slip, "traded_value": abs(delta * price),
                       "reason": reason})
        return cash_change, fee + slip

    def observe(self, now, fresh, completed_bars, terminal=frozenset()):
        now = pd.Timestamp(now)
        if self.previous_time is not None and now <= self.previous_time:
            raise ValueError("Observation batches must have strictly increasing times")
        self.previous_time = now
        fills_before = len(self.trades)
        self.marks.update(fresh)
        self.terminated.update(terminal)
        for group, book in self.books.items():
            positions, pending = book["positions"], book["pending"]
            for symbol, order in list(pending.items()):
                if order["exit"] <= now or symbol in self.terminated:
                    self.exclusions.append({"timestamp": now, "group": group, "symbol": symbol,
                                            "reason": "entry expired or final bar reached"})
                    del pending[symbol]
            book["waiting"] = [row for row in book["waiting"] if row["exit"] > now]
            for symbol, position in list(positions.items()):
                if symbol not in fresh or (symbol not in terminal and now <= position["exit"]):
                    continue
                reason = "research_end" if symbol in terminal else "scheduled_exit"
                flow, cost = self.fill(group, symbol, -position["quantity"], fresh[symbol], now, reason)
                pnl = position["entry_flow"] + flow
                costs = position["entry_cost"] + cost
                self.closed.append({"group": group, "symbol": symbol, "event_start": position["entry"],
                                    "event_end": now, "timestamp": now, "side": np.sign(position["quantity"]),
                                    "entry_aum": position["entry_aum"], "gross_pnl": pnl + costs,
                                    "execution_cost": costs, "net_pnl": pnl,
                                    "net_return": pnl / position["entry_aum"]})
                del positions[symbol]
            if (now, group) in self.rounds:
                book["waiting"] = self.rounds[(now, group)]
            if not positions and not pending and book["waiting"]:
                equity = book["cash"]
                if equity <= 0:
                    raise ValueError("Book equity exhausted")
                for row in book["waiting"]:
                    symbol = row["symbol"]
                    if symbol in self.terminated:
                        self.exclusions.append({"timestamp": now, "group": group, "symbol": symbol,
                                                "reason": "no dollar bars in evaluation interval"
                                                if symbol in self.unavailable else "final bar reached"})
                        continue
                    pending[symbol] = {"decision": now, "exit": row["exit"],
                                       "notional": equity * row["target_weight"], "equity": equity}
                book["waiting"] = []
            for symbol, order in list(pending.items()):
                if symbol not in fresh or now <= order["decision"]:
                    continue
                quantity = order["notional"] / fresh[symbol]
                flow, cost = self.fill(group, symbol, quantity, fresh[symbol], now, "scheduled_entry")
                positions[symbol] = {"quantity": quantity, "entry": now, "exit": order["exit"],
                                     "entry_flow": flow, "entry_cost": cost, "entry_aum": order["equity"]}
                del pending[symbol]
        if len(self.trades) != fills_before or terminal or not fresh:
            self._record(now)

    def _record(self, now):
        if self.records and self.records[-1]["timestamp"] == now:
            return
        total_cash = total_aum = gross = net = 0.0
        values = []
        for group, book in self.books.items():
            group_net = sum(p["quantity"] * self.marks[s] for s, p in book["positions"].items())
            group_aum = book["cash"] + group_net
            self.group_records.append({"timestamp": now, "group": group, "cash": book["cash"], "aum": group_aum})
            total_cash += book["cash"]
            total_aum += group_aum
            for symbol, position in book["positions"].items():
                value = position["quantity"] * self.marks[symbol]
                gross += abs(value)
                net += value
                values.append({"timestamp": now, "group": group, "symbol": symbol,
                               "quantity": position["quantity"], "position_value": value})
        if total_aum <= 0:
            raise ValueError("Account equity exhausted")
        self.exposures.extend({**row, "weight": row["position_value"] / total_aum} for row in values)
        self.records.append({"timestamp": now, "cash": total_cash, "aum": total_aum,
                        "gross_exposure": gross / total_aum, "net_exposure": net / total_aum,
                        "broker_fee": self.fee_total, "slippage": self.slip_total, "traded_value": self.turnover})

    def finish(self):
        if self.previous_time is not None:
            self._record(self.previous_time)
        if any(book["positions"] for book in self.books.values()):
            raise ValueError("Evaluation ended with open positions")
        return {"ledger": pd.DataFrame(self.records), "group_ledger": pd.DataFrame(self.group_records),
                "trades": pd.DataFrame(self.trades),
                "closed_trades": pd.DataFrame(self.closed, columns=["group", "symbol", "event_start", "event_end",
                    "timestamp", "side", "entry_aum", "gross_pnl", "execution_cost", "net_pnl", "net_return"]),
                "exposures": pd.DataFrame(self.exposures, columns=["timestamp", "group", "symbol", "quantity", "position_value", "weight"]),
                "exclusions": pd.DataFrame(self.exclusions)}


def _recording_clocks(calendar, start, end):
    return {start, end, *calendar.loc[calendar.close.between(start, end), "close"]}


def _synchronous_clocks(calendar, start, end):
    schedule = entry_schedule(calendar, end)
    return set(schedule.loc[schedule.entry.between(start, end), "entry"])


def simulate_synchronous(events: pd.DataFrame, prices: pd.DataFrame,
                         calendar: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                         settings: PortfolioSettings, evaluation_partition: str = "holdout",
                         *, selected: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Decide at session boundaries; fill strictly later completed bar closes.

    prices contains timestamp, symbol and price for every completed dollar bar.
    Terminal flags are derived from the last bar per symbol within the interval.
    """
    account = _SynchronousAccount(events, calendar, start, end, settings, evaluation_partition, selected)
    # Explicit frames may interleave symbols, but each symbol must remain chronological.
    with SynchronousPriceIndex(start, end, prices=prices) as index:
        decisions = _synchronous_clocks(calendar, start, end)
        return SynchronousScheduler({0: account}, index, decisions, start, end).run()[0]


def _simulate_synchronous_accounts(paths, prepared, calendar, start, end, settings,
                                   evaluation_partition, *, symbols, show_progress=False):
    accounts = {key: _SynchronousAccount(item["events"], calendar, start, end, settings,
                                         evaluation_partition, item.get("selected"))
                for key, item in prepared.items()}
    if not accounts:
        return {}
    if show_progress:
        tqdm.write("Preparing synchronous price index")
    decisions = _synchronous_clocks(calendar, start, end)
    with SynchronousPriceIndex(start, end, paths=paths) as index:
        return SynchronousScheduler(accounts, index, decisions, start, end).run(show_progress)


STRATEGIES = ("market_synchronous", "sentiment_synchronous", "sentiment_asynchronous")


def simulate_strategy(paths, events: pd.DataFrame, calendar: pd.DataFrame,
                      start: pd.Timestamp, end: pd.Timestamp, strategy: str,
                      settings: PortfolioSettings, calibration: pd.DataFrame | None = None,
                      evaluation_partition: str = "holdout",
                      bar_prices: pd.DataFrame | None = None, *, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Execute on explicit common bounds, retaining initial idle cash."""
    if strategy not in STRATEGIES or paths.model_kind != strategy.split("_")[0]:
        raise ValueError("Strategy and model artifact family must agree")
    selected = events.loc[events.partition.eq(evaluation_partition)
                          & events.event_start.between(start, end)].copy()
    if strategy.endswith("asynchronous"):
        key = (paths.model_kind, evaluation_partition)
        return _simulate_asynchronous_accounts(
            paths, {key: {"events": selected, "calibration": calibration}}, calendar, start, end,
            settings, evaluation_partition, symbols=symbols,
        )[key]
    if bar_prices is None:
        key = (paths.model_kind, evaluation_partition)
        return _simulate_synchronous_accounts(
            paths, {key: {"events": selected}}, calendar, start, end, settings,
            evaluation_partition, symbols=symbols,
        )[key]
    return simulate_synchronous(selected, bar_prices, calendar, start, end, settings, evaluation_partition)
