from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score
from sklearn.metrics import recall_score


def compute_strategy_returns(
        predictions: pd.DataFrame,
        one_way_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Compute primary-only and meta-filtered event returns with explicit costs.

    A completed event trade pays the one-way cost once on entry and once on exit.

    Args:
        predictions: Event outcomes, primary sides, meta actions, and bet sizes.
        one_way_cost_bps: Slippage charged for each entry or exit in basis points.

    Returns:
        Positions plus gross, entry, exit, total-cost, and net returns for both strategies.

    Raises:
        ValueError: If required columns or bounded decision values are invalid.
    """
    required = {"raw_return", "primary_side", "meta_action", "bet_size"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing return columns: {sorted(missing)}")
    if one_way_cost_bps < 0:
        raise ValueError("one_way_cost_bps must be non-negative")
    if not predictions["primary_side"].isin([-1, 1]).all():
        raise ValueError("primary_side must contain only -1 and 1")
    if not predictions["meta_action"].isin([0, 1]).all():
        raise ValueError("meta_action must contain only 0 and 1")
    if not predictions["bet_size"].between(0.0, 1.0).all():
        raise ValueError("bet_size must be in [0, 1]")

    out = predictions.copy()
    out["primary_only_position"] = out["primary_side"].astype("float64")
    out["meta_filtered_position"] = (
        out["primary_side"]
        * out["meta_action"]
        * out["bet_size"]
    ).astype("float64")
    one_way_rate = one_way_cost_bps / 10_000.0

    for strategy in ["primary_only", "meta_filtered"]:
        position = out[f"{strategy}_position"]
        out[f"{strategy}_gross_return"] = position * out["raw_return"]
        out[f"{strategy}_entry_cost"] = position.abs() * one_way_rate
        out[f"{strategy}_exit_cost"] = position.abs() * one_way_rate
        out[f"{strategy}_total_cost"] = (
            out[f"{strategy}_entry_cost"] + out[f"{strategy}_exit_cost"]
        )
        out[f"{strategy}_net_return"] = (
            out[f"{strategy}_gross_return"] - out[f"{strategy}_total_cost"]
        )

    return out


class GeneralCharacteristics:
    """Namespace for general backtest characteristics."""

    @staticmethod
    def start(event_start: pd.Index | pd.Series | pd.DataFrame) -> Any:
        """Return the earliest event-start timestamp.

        Args:
            event_start: Series, frame, or index with event-start timestamps.

        Returns:
            Earliest event-start timestamp.
        """
        event_start = _as_index(event_start)

        return event_start.min()

    @staticmethod
    def end(
        event_start: pd.Index | pd.Series | pd.DataFrame,
        event_end: pd.Series | None = None,
    ) -> Any:
        """Return the latest event-start or event-end timestamp.

        Args:
            event_start: Series, frame, or index with event-start timestamps.
            event_end: Optional event-end timestamps aligned to ``event_start``.

        Returns:
            Latest event-end timestamp when ``event_end`` is provided,
            otherwise the latest event-start timestamp.
        """
        event_start = _as_index(event_start)

        if event_end is None:
            return event_start.max()

        event_start, event_end = _validate_event_times(event_start, event_end)

        return event_end.max()

    @staticmethod
    def average_aum(aum: pd.Series | np.ndarray) -> float:
        """Compute the average assets under management.

        Args:
            aum: Assets under management by timestamp.

        Returns:
            Average absolute AUM.
        """
        aum = _as_series(aum, name="aum").abs()

        return aum.mean()

    @staticmethod
    def leverage(
        position_values: pd.Series | pd.DataFrame | np.ndarray,
        aum: pd.Series | np.ndarray,
    ) -> float:
        """Compute average gross dollar position size divided by average AUM.

        Args:
            position_values: Dollar position values as a series or frame.
            aum: Assets under management by timestamp.

        Returns:
            Average leverage.
        """
        gross_position = _gross_position_values(position_values)
        aum = _as_series(aum, name="aum").abs()
        df0 = pd.concat([gross_position.rename("position"), aum.rename("aum")], axis=1)
        df0 = df0.dropna()

        return df0["position"].mean() / df0["aum"].mean()

    @staticmethod
    def maximum_dollar_position_size(
        position_values: pd.Series | pd.DataFrame | np.ndarray,
    ) -> float:
        """Return the maximum gross dollar position size.

        Args:
            position_values: Dollar position values as a series or frame.

        Returns:
            Maximum gross dollar position size.
        """
        gross_position = _gross_position_values(position_values)

        return gross_position.max()

    @staticmethod
    def ratio_of_longs(positions: pd.Series | np.ndarray) -> float:
        """Compute the fraction of non-flat positions that are long.

        Args:
            positions: Position series where positive values are long.

        Returns:
            Fraction of active positions that are long.
        """
        positions = _as_series(positions, name="positions")
        active = positions[positions != 0]

        if active.empty:
            return np.nan

        return (active > 0).mean()

    @staticmethod
    def frequency_of_bets(
        positions: pd.Series,
        event_end: pd.Series,
        time_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> float:
        """Compute the annualized number of active event bets.

        Args:
            positions: Event positions indexed by event-start timestamp.
            event_end: Event-end timestamps aligned to ``positions``.
            time_range: Optional full account span, including time spent flat.

        Returns:
            Annualized number of independent bets.
        """
        positions, event_start, event_end = _validate_event_positions(
            positions,
            event_end,
        )
        start, end = event_start.min(), event_end.max()
        if time_range is not None:
            if time_range[0] > start or time_range[1] < end:
                raise ValueError("time_range must contain every trade")
            start, end = time_range
        elapsed_seconds = (end - start).total_seconds()
        active_bets = positions.ne(0.0).sum()

        if active_bets == 0:
            return 0.0

        if elapsed_seconds == 0:
            return np.nan

        elapsed_years = elapsed_seconds / (365.25 * 24 * 60 * 60)

        return float(active_bets / elapsed_years)

    @staticmethod
    def average_holding_period(
        positions: pd.Series,
        event_end: pd.Series,
    ) -> float:
        """Compute the equal-weighted average event holding period in days.

        Args:
            positions: Event positions indexed by event-start timestamp.
            event_end: Event-end timestamps aligned to ``positions``.

        Returns:
            Equal-weighted average holding period across active event bets.
        """
        positions, event_start, event_end = _validate_event_positions(
            positions,
            event_end,
        )
        active = positions.ne(0.0)

        if not active.any():
            return np.nan

        holding_days = (
            event_end - event_start.to_series(index=event_start)
        ).dt.total_seconds() / (24 * 60 * 60)

        return holding_days[active].mean()

    @staticmethod
    def annualized_turnover(
        traded_value: pd.Series | np.ndarray,
        aum: pd.Series | np.ndarray,
        time_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> float:
        """Compute annual traded dollar value divided by average AUM.

        Args:
            traded_value: Dollar value traded by timestamp.
            aum: Assets under management by timestamp.
            time_range: Explicit common evaluation span when AUM is sampled daily.

        Returns:
            Annualized turnover.
        """
        traded_value = _as_series(traded_value, name="traded_value").abs()
        aum = _as_series(aum, name="aum").abs()
        if time_range is None:
            years = _elapsed_years(traded_value.index)
        else:
            start, end = time_range
            years = (end - start).total_seconds() / (365.25 * 24 * 60 * 60)
            if years < 0:
                raise ValueError("time_range end must not precede start")
            if not isinstance(traded_value.index, pd.DatetimeIndex) or not isinstance(
                aum.index, pd.DatetimeIndex,
            ):
                raise ValueError("Explicit time_range requires datetime inputs")
            if not (
                traded_value.index.to_series().between(start, end).all()
                and aum.index.to_series().between(start, end).all()
            ):
                raise ValueError("Turnover and AUM must lie within time_range")

        if years == 0:
            return np.nan

        return traded_value.sum() / years / aum.mean()

    @staticmethod
    def correlation_to_underlying(
        strategy_returns: pd.Series | np.ndarray,
        underlying_returns: pd.Series | np.ndarray,
    ) -> float:
        """Compute correlation between strategy and underlying returns.

        Args:
            strategy_returns: Strategy return series.
            underlying_returns: Underlying investment universe return series.

        Returns:
            Pearson correlation coefficient.
        """
        strategy_returns = _as_series(strategy_returns, name="strategy_returns")
        underlying_returns = _as_series(underlying_returns, name="underlying_returns")

        return strategy_returns.corr(underlying_returns)


class Performance:
    """Namespace for performance statistics."""

    @staticmethod
    def pnl(pnl: pd.Series | np.ndarray) -> float:
        """Compute total PnL.

        Args:
            pnl: Profit-and-loss series.

        Returns:
            Total profit and loss.
        """
        pnl = _as_series(pnl, name="pnl")

        return pnl.sum()

    @staticmethod
    def pnl_from_long_positions(
            pnl: pd.Series | np.ndarray,
            positions: pd.Series | np.ndarray,
    ) -> float:
        """Compute PnL generated while the strategy is long.

        Args:
            pnl: Profit-and-loss series.
            positions: Position series where positive values are long.

        Returns:
            Total PnL generated by long positions.
        """
        pnl = _as_series(pnl, name="pnl")
        positions = _as_series(positions, name="positions")
        df0 = pd.concat([pnl.rename("pnl"), positions.rename("positions")], axis=1)
        df0 = df0.dropna()

        return df0.loc[df0["positions"] > 0, "pnl"].sum()

    @staticmethod
    def pnl_from_short_positions(
            pnl: pd.Series | np.ndarray,
            positions: pd.Series | np.ndarray,
    ) -> float:
        """Compute PnL generated while the strategy is short.

        Args:
            pnl: Profit-and-loss series.
            positions: Position series where negative values are short.

        Returns:
            Total PnL generated by short positions.
        """
        pnl = _as_series(pnl, name="pnl")
        positions = _as_series(positions, name="positions")
        df0 = pd.concat([pnl.rename("pnl"), positions.rename("positions")], axis=1)
        df0 = df0.dropna()

        return df0.loc[df0["positions"] < 0, "pnl"].sum()

    @staticmethod
    def annualized_rate_of_return(
            returns: pd.Series | np.ndarray,
            periods_per_year: float | None = None,
            start: pd.Timestamp | None = None,
            end: pd.Timestamp | None = None,
    ) -> float:
        """Compute annualized time-weighted rate of return.

        Args:
            returns: Return series.
            periods_per_year: Number of return observations per year. If ``None``,
                elapsed calendar time is inferred from the index.
            start: Optional explicit start timestamp for elapsed-time annualization.
            end: Optional explicit end timestamp for elapsed-time annualization.

        Returns:
            Annualized time-weighted rate of return.
        """
        returns = _as_series(returns, name="returns")
        cumulative_return = (1.0 + returns).prod()

        if (start is None) != (end is None):
            raise ValueError("start and end must be provided together")

        if start is not None and end is not None:
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            if end < start:
                raise ValueError("end must not precede start")
            years = (end - start).total_seconds() / (365.25 * 24 * 60 * 60)
        elif periods_per_year is None:
            years = _elapsed_years(returns.index)
        else:
            years = returns.shape[0] / periods_per_year

        if years == 0:
            return np.nan

        return cumulative_return ** (1.0 / years) - 1.0

    @staticmethod
    def hit_ratio(bet_returns: pd.Series | np.ndarray) -> float:
        """Compute the fraction of bets with positive returns.

        Args:
            bet_returns: Return series by bet.

        Returns:
            Fraction of bets with positive returns.
        """
        bet_returns = _as_series(bet_returns, name="bet_returns")

        if bet_returns.empty:
            return np.nan

        return (bet_returns > 0).mean()

    @staticmethod
    def average_return_from_hits(bet_returns: pd.Series | np.ndarray) -> float:
        """Compute average return from profitable bets.

        Args:
            bet_returns: Return series by bet.

        Returns:
            Average return among profitable bets.
        """
        bet_returns = _as_series(bet_returns, name="bet_returns")
        hits = bet_returns[bet_returns > 0]

        if hits.empty:
            return np.nan

        return hits.mean()

    @staticmethod
    def average_return_from_misses(bet_returns: pd.Series | np.ndarray) -> float:
        """Compute average return from losing bets.

        Args:
            bet_returns: Return series by bet.

        Returns:
            Average return among losing bets.
        """
        bet_returns = _as_series(bet_returns, name="bet_returns")
        misses = bet_returns[bet_returns < 0]

        if misses.empty:
            return np.nan

        return misses.mean()


class Runs:
    """Namespace for runs and drawdown statistics."""

    @staticmethod
    def hhi_positive_returns(bet_returns: pd.Series | np.ndarray) -> float:
        """Compute HHI concentration for non-negative bet returns.

        Args:
            bet_returns: Return series by bet.

        Returns:
            Normalized HHI concentration of non-negative returns.
        """
        bet_returns = _as_series(bet_returns, name="bet_returns")

        return _hhi(bet_returns[bet_returns >= 0])

    @staticmethod
    def hhi_negative_returns(bet_returns: pd.Series | np.ndarray) -> float:
        """Compute HHI concentration for negative bet returns.

        Args:
            bet_returns: Return series by bet.

        Returns:
            Normalized HHI concentration of negative returns.
        """
        bet_returns = _as_series(bet_returns, name="bet_returns")

        return _hhi(bet_returns[bet_returns < 0])

    @staticmethod
    def hhi_time_between_bets(
        bet_returns: pd.Series,
        freq: str = "ME",
    ) -> float:
        """Compute HHI concentration of bets across time buckets.
 
        Args:
            bet_returns: Return series by bet.
            freq: Pandas resampling frequency used to count bets.

        Returns:
            Normalized HHI concentration of bet counts across time buckets.
        """
        bet_returns = _as_series(bet_returns, name="bet_returns")
        bet_counts = bet_returns.resample(freq).count()

        return _hhi(bet_counts)

    @staticmethod
    def drawdown(series: pd.Series, dollars: bool = False) -> pd.Series:
        """Compute the drawdown series.

        Args:
            series: Return index or dollar performance series.
            dollars: Whether to compute drawdowns in dollars rather than returns.

        Returns:
            Drawdown series indexed by high-watermark timestamp.
        """
        drawdown, _ = _drawdown_time_under_water(series=series, dollars=dollars)

        return drawdown

    @staticmethod
    def time_under_water(series: pd.Series, dollars: bool = False) -> pd.Series:
        """Compute the time-under-water series in years.

        Args:
            series: Return index or dollar performance series.
            dollars: Whether ``series`` is dollar performance.

        Returns:
            Time-under-water series in years.
        """
        _, time_under_water = _drawdown_time_under_water(
            series=series,
            dollars=dollars
        )

        return time_under_water

    @staticmethod
    def percentile_drawdown(
        series: pd.Series,
        q: float = 0.95,
        dollars: bool = False,
    ) -> float:
        """Compute a drawdown percentile.

        Args:
            series: Return index or dollar performance series.
            q: Quantile to compute.
            dollars: Whether to compute drawdowns in dollars rather than returns.

        Returns:
            Requested drawdown percentile.
        """
        drawdown = Runs.drawdown(series=series, dollars=dollars)

        if drawdown.empty:
            return np.nan

        return drawdown.quantile(q)

    @staticmethod
    def percentile_time_under_water(
        series: pd.Series,
        q: float = 0.95,
        dollars: bool = False,
    ) -> float:
        """Compute a time-under-water percentile.

        Args:
            series: Return index or dollar performance series.
            q: Quantile to compute.
            dollars: Whether ``series`` is dollar performance.

        Returns:
            Requested time-under-water percentile.
        """
        time_under_water = Runs.time_under_water(series=series, dollars=dollars)

        if time_under_water.empty:
            return np.nan

        return time_under_water.quantile(q)


class ImplementationShortfall:
    """Namespace for implementation shortfall statistics."""

    @staticmethod
    def broker_fees_per_turnover(
        broker_fees: Sequence[float] | np.ndarray | pd.Series,
        turnover: Sequence[float] | np.ndarray | pd.Series,
    ) -> float:
        """Compute broker fees divided by turnover.

        Args:
            broker_fees: Broker fee observations.
            turnover: Turnover observations.

        Returns:
            Broker fees per unit of turnover.
        """
        return _safe_divide(np.sum(broker_fees), np.sum(turnover))

    @staticmethod
    def average_slippage_per_turnover(
        slippage: Sequence[float] | np.ndarray | pd.Series,
        turnover: Sequence[float] | np.ndarray | pd.Series,
    ) -> float:
        """Compute average slippage divided by turnover.

        Args:
            slippage: Slippage cost observations.
            turnover: Turnover observations.

        Returns:
            Slippage per unit of turnover.
        """
        return _safe_divide(np.sum(slippage), np.sum(turnover))

    @staticmethod
    def dollar_performance_per_turnover(
        dollar_performance: Sequence[float] | np.ndarray | pd.Series,
        turnover: Sequence[float] | np.ndarray | pd.Series,
    ) -> float:
        """Compute dollar performance divided by total turnover.

        Args:
            dollar_performance: Dollar performance observations.
            turnover: Turnover observations.

        Returns:
            Dollar performance per unit of turnover.
        """
        return _safe_divide(np.sum(dollar_performance), np.sum(turnover))

    @staticmethod
    def return_on_execution_costs(
        dollar_performance: Sequence[float] | np.ndarray | pd.Series,
        execution_costs: Sequence[float] | np.ndarray | pd.Series,
    ) -> float:
        """Compute dollar performance divided by total execution costs.

        Args:
            dollar_performance: Dollar performance observations.
            execution_costs: Execution cost observations.

        Returns:
            Return on execution costs.
        """
        return _safe_divide(np.sum(dollar_performance), np.sum(execution_costs))


class Efficiency:
    """Namespace for return-risk efficiency statistics."""

    @staticmethod
    def portfolio_statistics(
        returns: pd.Series,
        period_start: pd.Series,
        annual_risk_free_rate: float = 0.03,
        periods_per_year: float = 365.25,
        annualized_benchmark_sharpe_ratio: float = 1.0,
    ) -> dict[str, float]:
        """Compute daily account Sharpe and PSR with full-capital risk-free accrual.

        Returns are indexed by period end; ``period_start`` has the same index.
        Unlike event metrics, the benchmark is charged to the entire account.
        """
        if not isinstance(returns, pd.Series) or not isinstance(
            period_start, pd.Series,
        ):
            raise ValueError("returns and period_start must be pandas Series")
        if not returns.index.equals(period_start.index):
            raise ValueError("period_start must have the same index as returns")
        if not pd.api.types.is_datetime64_any_dtype(period_start.dtype):
            raise ValueError("period_start must contain datetimes")
        if period_start.isna().any() or not np.isfinite(returns).all():
            raise ValueError("Portfolio observations must not contain missing values")
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive and finite")
        starts = pd.DatetimeIndex(period_start)
        # Reuse event accrual validation with full-account exposure.
        excess = _excess_returns(
            pd.Series(returns.to_numpy(), index=starts),
            pd.Series(returns.index, index=starts),
            pd.Series(1.0, index=starts),
            annual_risk_free_rate,
        )
        sharpe = _safe_divide(excess.mean(), excess.std(ddof=1))
        return {
            "sharpe_ratio": sharpe,
            "annualized_sharpe": sharpe * periods_per_year ** 0.5,
            "probabilistic_sharpe_ratio": _probabilistic_sharpe_ratio(
                excess, annualized_benchmark_sharpe_ratio / periods_per_year ** 0.5,
            ),
        }

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        event_end: pd.Series,
        positions: pd.Series,
        annual_risk_free_rate: float = 0.03,
    ) -> float:
        """Compute the non-annualized Sharpe ratio.

        Args:
            returns: Event return series indexed by event start time.
            event_end: Event end times aligned to returns.
            positions: Bounded strategy positions aligned to returns.
            annual_risk_free_rate: Effective annual risk-free return.

        Returns:
            Non-annualized Sharpe ratio.
        """
        excess_returns = _excess_returns(
            returns=returns,
            event_end=event_end,
            positions=positions,
            annual_risk_free_rate=annual_risk_free_rate,
        )
        std = excess_returns.std(ddof=1)

        return _safe_divide(excess_returns.mean(), std)

    @staticmethod
    def annualized_sharpe_ratio(
        returns: pd.Series,
        event_end: pd.Series,
        positions: pd.Series,
        annual_risk_free_rate: float = 0.03,
        periods_per_year: int = 252,
    ) -> float:
        """Compute the annualized Sharpe ratio.

        Args:
            returns: Event return series indexed by event start time.
            event_end: Event end times aligned to returns.
            positions: Bounded strategy positions aligned to returns.
            annual_risk_free_rate: Effective annual risk-free return.
            periods_per_year: Number of return observations per year.

        Returns:
            Annualized Sharpe ratio.
        """
        sharpe_ratio = Efficiency.sharpe_ratio(
            returns=returns,
            event_end=event_end,
            positions=positions,
            annual_risk_free_rate=annual_risk_free_rate,
        )

        return sharpe_ratio * periods_per_year ** 0.5

    @staticmethod
    def probabilistic_sharpe_ratio(
        returns: pd.Series,
        event_end: pd.Series,
        positions: pd.Series,
        annualized_benchmark_sharpe_ratio: float = 1.0,
        periods_per_year: int = 252,
        annual_risk_free_rate: float = 0.03,
    ) -> float:
        """Compute the probabilistic Sharpe ratio.

        Args:
            returns: Event return series indexed by event start time.
            event_end: Event end times aligned to returns.
            positions: Bounded strategy positions aligned to returns.
            annualized_benchmark_sharpe_ratio: Annualized benchmark Sharpe ratio.
            periods_per_year: Number of return observations per year.
            annual_risk_free_rate: Effective annual risk-free return.

        Returns:
            Probability that the observed Sharpe ratio exceeds the benchmark.
        """
        excess_returns = _excess_returns(
            returns=returns,
            event_end=event_end,
            positions=positions,
            annual_risk_free_rate=annual_risk_free_rate,
        )
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        benchmark_sharpe_ratio = (
            annualized_benchmark_sharpe_ratio / periods_per_year ** 0.5
        )

        return _probabilistic_sharpe_ratio(
            excess_returns=excess_returns,
            benchmark_sharpe_ratio=benchmark_sharpe_ratio,
        )

class ClassificationScores:
    """Namespace for classification scores."""

    @staticmethod
    def accuracy(
        y_true: Sequence[Any] | pd.Series | np.ndarray,
        y_pred: Sequence[Any] | pd.Series | np.ndarray,
        sample_weight: Sequence[float] | pd.Series | np.ndarray | None = None,
    ) -> float:
        """Compute classification accuracy.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
            sample_weight: Optional sample weights.

        Returns:
            Accuracy score.
        """
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    @staticmethod
    def precision(
        y_true: Sequence[Any] | pd.Series | np.ndarray,
        y_pred: Sequence[Any] | pd.Series | np.ndarray,
        zero_division: int | str = 0,
        pos_label: Any = 1,
        sample_weight: Sequence[float] | pd.Series | np.ndarray | None = None,
    ) -> float:
        """Compute classification precision.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
            zero_division: Value to use when precision is undefined.
            pos_label: Label treated as the positive class.
            sample_weight: Optional sample weights.

        Returns:
            Precision score.
        """
        return precision_score(
            y_true,
            y_pred,
            pos_label=pos_label,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    @staticmethod
    def recall(
        y_true: Sequence[Any] | pd.Series | np.ndarray,
        y_pred: Sequence[Any] | pd.Series | np.ndarray,
        zero_division: int | str = 0,
        pos_label: Any = 1,
        sample_weight: Sequence[float] | pd.Series | np.ndarray | None = None,
    ) -> float:
        """Compute classification recall.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
            zero_division: Value to use when recall is undefined.
            pos_label: Label treated as the positive class.
            sample_weight: Optional sample weights.

        Returns:
            Recall score.
        """
        return recall_score(
            y_true,
            y_pred,
            pos_label=pos_label,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    @staticmethod
    def f1_score(
        y_true: Sequence[Any] | pd.Series | np.ndarray,
        y_pred: Sequence[Any] | pd.Series | np.ndarray,
        zero_division: int | str = 0,
        pos_label: Any = 1,
        sample_weight: Sequence[float] | pd.Series | np.ndarray | None = None,
    ) -> float:
        """Compute F1 score.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.
            zero_division: Value to use when F1 is undefined.
            pos_label: Label treated as the positive class.
            sample_weight: Optional sample weights.

        Returns:
            F1 score.
        """
        return f1_score(
            y_true,
            y_pred,
            pos_label=pos_label,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    @staticmethod
    def negative_log_loss(
        y_true: Sequence[Any] | pd.Series | np.ndarray,
        y_pred_proba: pd.DataFrame | np.ndarray,
        labels: Sequence[Any] | None = None,
        sample_weight: Sequence[float] | pd.Series | np.ndarray | None = None,
    ) -> float:
        """Compute negative log-loss.

        Args:
            y_true: True class labels.
            y_pred_proba: Predicted class probabilities.
            labels: Complete set of class labels.
            sample_weight: Optional sample weights.

        Returns:
            Negative log-loss.
        """
        return -log_loss(
            y_true,
            y_pred_proba,
            labels=labels,
            sample_weight=sample_weight,
        )


def _as_index(values):
    """Coerce a series-like object to an index."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.index

    values = pd.Index(values)

    if values.empty:
        raise ValueError("index must not be empty")

    return values


def _as_series(values, name):
    """Coerce array-like values to a float series."""
    if isinstance(values, pd.Series):
        out = values.copy()
    else:
        out = pd.Series(values, name=name)

    if out.empty:
        raise ValueError(f"{name} must not be empty")

    return out.astype("float64")


def _gross_position_values(position_values):
    """Compute gross dollar exposure from one or more position value columns."""
    if isinstance(position_values, pd.DataFrame):
        if position_values.empty:
            raise ValueError("position_values must not be empty")

        return position_values.abs().sum(axis=1)

    return _as_series(position_values, name="position_values").abs()


def _validate_event_times(event_start, event_end):
    """Validate aligned event-start and event-end timestamps."""
    event_start = _as_index(event_start)
    if not isinstance(event_start, pd.DatetimeIndex):
        raise ValueError("event_start must be a DatetimeIndex")
    if event_start.hasnans:
        raise ValueError("event_start must not contain missing values")
    if not isinstance(event_end, pd.Series):
        raise ValueError("event_end must be a pandas Series")
    if not event_start.equals(event_end.index):
        raise ValueError("event_end must have the same index as event_start")
    if not pd.api.types.is_datetime64_any_dtype(event_end.dtype):
        raise ValueError("event_end must contain datetimes")
    if event_end.isna().any():
        raise ValueError("event_end must not contain missing values")
    if event_start.tz != event_end.dt.tz:
        raise ValueError("event_start and event_end must use the same timezone")

    start_series = event_start.to_series(index=event_start)
    if (event_end < start_series).any():
        raise ValueError("event_end must not precede event_start")

    return event_start, event_end


def _validate_event_positions(positions, event_end):
    """Validate event positions and their aligned timestamps."""
    if not isinstance(positions, pd.Series):
        raise ValueError("positions must be a pandas Series")
    if (
        pd.api.types.is_bool_dtype(positions.dtype)
        or not pd.api.types.is_numeric_dtype(positions.dtype)
    ):
        raise ValueError("positions must contain numeric values")

    event_start, event_end = _validate_event_times(positions.index, event_end)
    positions = positions.astype("float64")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain finite values")

    return positions, event_start, event_end


def _elapsed_years(index):
    """Compute elapsed years across a datetime index."""
    index = _as_index(index)

    if index.shape[0] < 2:
        return 0.0

    if not isinstance(index, pd.DatetimeIndex):
        return index.shape[0] - 1

    elapsed_days = (index.max() - index.min()) / np.timedelta64(1, "D")

    return elapsed_days / 365.25


def _hhi(bet_returns):
    """Compute normalized Herfindahl-Hirschman concentration."""
    if len(bet_returns) == 0:
        return np.nan
    bet_returns = _as_series(bet_returns, name="bet_returns").dropna()

    if bet_returns.shape[0] <= 2:
        return np.nan

    if bet_returns.sum() == 0:
        return np.nan

    weights = bet_returns / bet_returns.sum()
    hhi = (weights ** 2).sum()

    return (hhi - bet_returns.shape[0] ** -1) / (1.0 - bet_returns.shape[0] ** -1)


def _drawdown_time_under_water(series, dollars=False):
    """Measure each underwater episode from its peak through recovery or end."""
    series = _as_series(series, name="series")
    if not isinstance(series.index, pd.DatetimeIndex) or not (
        series.index.is_monotonic_increasing
    ):
        raise ValueError("Drawdowns require a sorted DatetimeIndex")
    if not np.isfinite(series).all():
        raise ValueError("Drawdown observations must be finite")
    values = series.to_numpy()
    peaks = np.maximum.accumulate(values)
    underwater = values < peaks
    starts = np.flatnonzero(underwater & ~np.r_[False, underwater[:-1]])
    ends = np.flatnonzero(underwater & ~np.r_[underwater[1:], False])
    drawdowns, durations, times = [], [], []
    for start, end in zip(starts, ends, strict=True):
        peak = peaks[start]
        trough = values[start:end + 1].min()
        recovery = min(end + 1, len(series) - 1)
        peak_time = series.index[start - 1]
        times.append(peak_time)
        drawdowns.append(peak - trough if dollars else 1 - trough / peak)
        durations.append(
            (series.index[recovery] - peak_time).total_seconds()
            / (365.25 * 24 * 60 * 60)
        )
    index = pd.DatetimeIndex(times, tz=series.index.tz)
    return (
        pd.Series(drawdowns, index=index, dtype=float),
        pd.Series(durations, index=index, dtype=float),
    )


def _probabilistic_sharpe_ratio_from_moments(
        sharpe_ratio,
        benchmark_sharpe_ratio,
        num_returns,
        skewness,
        kurtosis
):
    """Compute PSR from observed Sharpe ratio and return moments."""
    denominator_sq = (
        1.0
        - skewness * sharpe_ratio
        + (kurtosis - 1.0) / 4.0 * sharpe_ratio ** 2
    )

    if denominator_sq <= 0 or num_returns <= 1:
        return np.nan

    denominator = denominator_sq ** 0.5
    statistic = (
        (sharpe_ratio - benchmark_sharpe_ratio)
        * (num_returns - 1) ** 0.5
        / denominator
    )

    return norm.cdf(statistic)


def _probabilistic_sharpe_ratio(excess_returns, benchmark_sharpe_ratio):
    """Compute PSR from non-annualized excess returns and benchmark Sharpe."""
    sharpe_ratio = _safe_divide(
        excess_returns.mean(),
        excess_returns.std(ddof=1),
    )

    return _probabilistic_sharpe_ratio_from_moments(
        sharpe_ratio=sharpe_ratio,
        benchmark_sharpe_ratio=benchmark_sharpe_ratio,
        num_returns=excess_returns.shape[0],
        skewness=excess_returns.skew(),
        kurtosis=excess_returns.kurt() + 3.0,
    )


def _safe_divide(numerator, denominator):
    """Divide values and return NaN when the denominator is zero."""
    if denominator == 0:
        return np.nan

    return numerator / denominator


def _excess_returns(returns, event_end, positions, annual_risk_free_rate):
    """Compute event-period excess returns from an effective annual rate."""
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")
    if not isinstance(event_end, pd.Series):
        raise ValueError("event_end must be a pandas Series")
    if not returns.index.equals(event_end.index):
        raise ValueError("event_end must have the same index as returns")
    if not pd.api.types.is_datetime64_any_dtype(event_end.dtype):
        raise ValueError("event_end must contain datetimes")
    if event_end.isna().any():
        raise ValueError("event_end must not contain missing values")
    if returns.index.tz != event_end.dt.tz:
        raise ValueError("returns and event_end must use the same timezone")
    if not isinstance(positions, pd.Series):
        raise ValueError("positions must be a pandas Series")
    if not returns.index.equals(positions.index):
        raise ValueError("positions must have the same index as returns")
    if (
        pd.api.types.is_bool_dtype(positions.dtype)
        or not pd.api.types.is_numeric_dtype(positions.dtype)
    ):
        raise ValueError("positions must contain numeric values")
    positions = positions.astype("float64")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain finite values")
    if not positions.between(-1.0, 1.0).all():
        raise ValueError("positions must be in [-1, 1]")

    if (
        isinstance(annual_risk_free_rate, (bool, np.bool_))
        or not isinstance(annual_risk_free_rate, Real)
    ):
        raise ValueError("annual_risk_free_rate must be a finite number")
    annual_risk_free_rate = float(annual_risk_free_rate)
    if not np.isfinite(annual_risk_free_rate):
        raise ValueError("annual_risk_free_rate must be a finite number")
    if annual_risk_free_rate <= -1.0:
        raise ValueError("annual_risk_free_rate must be greater than -1")

    event_start = returns.index.to_series(index=returns.index)
    holding_seconds = (event_end - event_start).dt.total_seconds()
    if (holding_seconds < 0.0).any():
        raise ValueError("event_end must not precede event start")

    holding_years = holding_seconds / (365.25 * 24 * 60 * 60)
    period_risk_free_rate = (
        (1.0 + annual_risk_free_rate) ** holding_years - 1.0
    )

    return (
        returns.astype("float64")
        - positions.abs() * period_risk_free_rate
    ).dropna()
