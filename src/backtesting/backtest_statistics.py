from __future__ import annotations

from collections.abc import Sequence
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
    def time_range(
        index: pd.Index | pd.Series | pd.DataFrame,
    ) -> tuple[Any, Any]:
        """Return the first and last timestamps in a backtest index.

        Args:
            index: Series, frame, or index with backtest timestamps.

        Returns:
            Tuple with the first and last timestamp.
        """
        index = _as_index(index)

        return index.min(), index.max()

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
    def capacity(
        aum: pd.Series | np.ndarray,
        risk_adjusted_performance: pd.Series | np.ndarray,
        target_performance: float,
    ) -> float:
        """Return the highest AUM that delivers the target performance.

        Args:
            aum: Assets under management by timestamp.
            risk_adjusted_performance: Risk-adjusted performance by timestamp.
            target_performance: Minimum acceptable risk-adjusted performance.

        Returns:
            Highest eligible AUM, or ``NaN`` when no timestamp is eligible.
        """
        aum = _as_series(aum, name="aum")
        risk_adjusted_performance = _as_series(
            risk_adjusted_performance,
            name="risk_adjusted_performance"
        )
        df0 = pd.concat(
            [aum.rename("aum"), risk_adjusted_performance.rename("performance")],
            axis=1
        ).dropna()
        eligible = df0[df0["performance"] >= target_performance]

        if eligible.empty:
            return np.nan

        return eligible["aum"].max()

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
        target_positions: pd.Series,
        periods_per_year: float = 365.25,
    ) -> float:
        """Compute the number of independent bets per year.

        Args:
            target_positions: Target position series.
            periods_per_year: Annualization basis for the timestamp frequency.

        Returns:
            Annualized number of independent bets.
        """
        target_positions = _as_series(target_positions, name="target_positions")
        bets = _get_bet_timestamps(target_positions)
        years = _elapsed_years(target_positions.index)

        if years == 0:
            return np.nan

        return bets.shape[0] / years * (365.25 / periods_per_year)

    @staticmethod
    def average_holding_period(target_positions: pd.Series) -> float:
        """Estimate the average holding period in days from target positions.

        Args:
            target_positions: Target position series.

        Returns:
            Position-weighted average holding period in days.
        """
        target_positions = _as_series(target_positions, name="target_positions")
        position_diff = target_positions.diff()
        time_diff = (
            target_positions.index - target_positions.index[0]
        ) / np.timedelta64(1, "D")
        holding_periods = pd.DataFrame(columns=["dT", "w"])
        entry_time = 0.0

        for i in range(1, target_positions.shape[0]):
            if position_diff.iloc[i] * target_positions.iloc[i - 1] >= 0:
                if target_positions.iloc[i] != 0:
                    entry_time = (
                        entry_time * target_positions.iloc[i - 1]
                        + time_diff[i] * position_diff.iloc[i]
                    ) / target_positions.iloc[i]
            else:
                if target_positions.iloc[i] * target_positions.iloc[i - 1] < 0:
                    holding_periods.loc[
                        target_positions.index[i],
                        ["dT", "w"]
                    ] = (
                        time_diff[i] - entry_time,
                        abs(target_positions.iloc[i - 1])
                    )
                    entry_time = time_diff[i]
                else:
                    holding_periods.loc[
                        target_positions.index[i],
                        ["dT", "w"]
                    ] = (
                        time_diff[i] - entry_time,
                        abs(position_diff.iloc[i])
                    )

        if holding_periods["w"].sum() <= 0:
            return np.nan

        return (
            holding_periods["dT"] * holding_periods["w"]
        ).sum() / holding_periods["w"].sum()

    @staticmethod
    def annualized_turnover(
        traded_value: pd.Series | np.ndarray,
        aum: pd.Series | np.ndarray,
    ) -> float:
        """Compute annual traded dollar value divided by average AUM.

        Args:
            traded_value: Dollar value traded by timestamp.
            aum: Assets under management by timestamp.

        Returns:
            Annualized turnover.
        """
        traded_value = _as_series(traded_value, name="traded_value").abs()
        aum = _as_series(aum, name="aum").abs()
        years = _elapsed_years(traded_value.index)

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
    def annualized_rate_of_return(
        returns: pd.Series | np.ndarray,
        periods_per_year: float | None = None,
    ) -> float:
        """Compute annualized time-weighted rate of return.

        Args:
            returns: Return series.
            periods_per_year: Number of return observations per year. If ``None``,
                elapsed calendar time is inferred from the index.

        Returns:
            Annualized time-weighted rate of return.
        """
        returns = _as_series(returns, name="returns")
        cumulative_return = (1.0 + returns).prod()

        if periods_per_year is None:
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
    def sharpe_ratio(
        returns: pd.Series | np.ndarray,
        risk_free_rate: float | pd.Series = 0.0,
    ) -> float:
        """Compute the non-annualized Sharpe ratio.

        Args:
            returns: Return series.
            risk_free_rate: Per-period risk-free return, as a scalar or series.

        Returns:
            Non-annualized Sharpe ratio.
        """
        excess_returns = _excess_returns(returns, risk_free_rate)
        std = excess_returns.std(ddof=1)

        return _safe_divide(excess_returns.mean(), std)

    @staticmethod
    def annualized_sharpe_ratio(
        returns: pd.Series | np.ndarray,
        risk_free_rate: float | pd.Series = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Compute the annualized Sharpe ratio.

        Args:
            returns: Return series.
            risk_free_rate: Per-period risk-free return, as a scalar or series.
            periods_per_year: Number of return observations per year.

        Returns:
            Annualized Sharpe ratio.
        """
        sharpe_ratio = Efficiency.sharpe_ratio(
            returns=returns,
            risk_free_rate=risk_free_rate
        )

        return sharpe_ratio * periods_per_year ** 0.5

    @staticmethod
    def information_ratio(
        portfolio_returns: pd.Series | np.ndarray,
        benchmark_returns: pd.Series | np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized information ratio relative to a benchmark.

        Args:
            portfolio_returns: Portfolio return series.
            benchmark_returns: Benchmark return series.
            periods_per_year: Number of return observations per year.

        Returns:
            Annualized information ratio.
        """
        portfolio_returns = _as_series(portfolio_returns, name="portfolio_returns")
        benchmark_returns = _as_series(benchmark_returns, name="benchmark_returns")
        excess_returns = portfolio_returns.sub(benchmark_returns, axis=0).dropna()
        tracking_error = excess_returns.std(ddof=1)

        return (
            _safe_divide(excess_returns.mean(), tracking_error)
            * periods_per_year ** 0.5
        )

    @staticmethod
    def probabilistic_sharpe_ratio(
        returns: pd.Series | np.ndarray,
        benchmark_sharpe_ratio: float = 0.0,
    ) -> float:
        """Compute the probabilistic Sharpe ratio.

        Args:
            returns: Return series.
            benchmark_sharpe_ratio: Benchmark Sharpe ratio used as the threshold.

        Returns:
            Probability that the observed Sharpe ratio exceeds the benchmark.
        """
        returns = _as_series(returns, name="returns").dropna()
        sharpe_ratio = Efficiency.sharpe_ratio(returns=returns, risk_free_rate=0.0)
        skewness = returns.skew()
        kurtosis = returns.kurt()
        kurtosis = kurtosis + 3.0

        return _probabilistic_sharpe_ratio_from_moments(
            sharpe_ratio=sharpe_ratio,
            benchmark_sharpe_ratio=benchmark_sharpe_ratio,
            num_returns=returns.shape[0],
            skewness=skewness,
            kurtosis=kurtosis
        )

    @staticmethod
    def deflated_sharpe_ratio(
        returns: pd.Series | np.ndarray,
        trial_sharpe_ratios: pd.Series | np.ndarray,
    ) -> float:
        """Compute the deflated Sharpe ratio.

        Args:
            returns: Return series for the selected strategy.
            trial_sharpe_ratios: Sharpe ratios observed across strategy trials.

        Returns:
            Deflated Sharpe ratio.
        """
        benchmark_sharpe_ratio = _expected_maximum_sharpe_ratio(
            trial_sharpe_ratios=trial_sharpe_ratios
        )

        return Efficiency.probabilistic_sharpe_ratio(
            returns=returns,
            benchmark_sharpe_ratio=benchmark_sharpe_ratio
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


def _get_bet_timestamps(target_positions):
    """Derive timestamps of flattening or flipping bets from target positions."""
    flattening = target_positions[target_positions == 0].index
    previous_position = target_positions.shift(1)
    previous_position = previous_position[previous_position != 0].index
    bets = flattening.intersection(previous_position)
    flips = target_positions.iloc[1:] * target_positions.iloc[:-1].values
    bets = bets.union(flips[flips < 0].index).sort_values()

    if target_positions.index[-1] not in bets:
        bets = bets.append(target_positions.index[-1:])

    return bets


def _elapsed_years(index):
    """Compute elapsed years across a datetime index."""
    index = _as_index(index)

    if index.shape[0] < 2:
        return 0.0

    if not np.issubdtype(index.dtype, np.datetime64):
        return index.shape[0] - 1

    elapsed_days = (index.max() - index.min()) / np.timedelta64(1, "D")

    return elapsed_days / 365.25


def _hhi(bet_returns):
    """Compute normalized Herfindahl-Hirschman concentration."""
    bet_returns = _as_series(bet_returns, name="bet_returns").dropna()

    if bet_returns.shape[0] <= 2:
        return np.nan

    if bet_returns.sum() == 0:
        return np.nan

    weights = bet_returns / bet_returns.sum()
    hhi = (weights ** 2).sum()

    return (hhi - bet_returns.shape[0] ** -1) / (1.0 - bet_returns.shape[0] ** -1)


def _drawdown_time_under_water(series, dollars=False):
    """Compute drawdown and time-under-water series."""
    series = _as_series(series, name="series")
    df0 = series.to_frame("pnl")
    df0["hwm"] = series.expanding().max()
    df1 = df0.groupby("hwm").min().reset_index()
    df1.columns = ["hwm", "min"]
    df1.index = df0["hwm"].drop_duplicates(keep="first").index
    df1 = df1[df1["hwm"] > df1["min"]]

    if dollars:
        drawdown = df1["hwm"] - df1["min"]
    else:
        drawdown = 1.0 - df1["min"] / df1["hwm"]

    time_under_water = (
        (df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, "D") / 365.25
    ).values
    time_under_water = pd.Series(time_under_water, index=df1.index[:-1])

    return drawdown, time_under_water


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


def _expected_maximum_sharpe_ratio(trial_sharpe_ratios):
    """Estimate the DSR benchmark Sharpe ratio from multiple trials."""
    trial_sharpe_ratios = _as_series(
        trial_sharpe_ratios,
        name="trial_sharpe_ratios"
    ).dropna()
    num_trials = trial_sharpe_ratios.shape[0]

    if num_trials <= 1:
        return np.nan

    euler_gamma = 0.5772156649015329
    trial_variance = trial_sharpe_ratios.var(ddof=1)
    expected_maximum = (
        (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / num_trials)
        + euler_gamma * norm.ppf(1.0 - np.exp(-1.0) / num_trials)
    )

    return trial_variance ** 0.5 * expected_maximum


def _safe_divide(numerator, denominator):
    """Divide values and return NaN when the denominator is zero."""
    if denominator == 0:
        return np.nan

    return numerator / denominator


def _excess_returns(returns, risk_free_rate):
    """Compute excess returns from scalar or series risk-free rates."""
    returns = _as_series(returns, name="returns")

    if isinstance(risk_free_rate, pd.Series):
        excess_returns = returns.sub(risk_free_rate, axis=0)
    else:
        excess_returns = returns - risk_free_rate

    return excess_returns.dropna()
