import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score
from sklearn.metrics import recall_score

from src.model_backtesting.backtest_statistics import (
    ClassificationScores,
    Efficiency,
    GeneralCharacteristics,
    Performance,
    compute_strategy_returns,
)


def test_strategy_returns_default_to_zero_execution_costs():
    predictions = pd.DataFrame(
        {
            "raw_return": [0.02, -0.01],
            "primary_side": [1, -1],
            "meta_action": [1, 0],
            "bet_size": [0.5, 0.8],
        }
    )

    returns = compute_strategy_returns(predictions)

    assert returns.loc[0, "primary_only_gross_return"] == pytest.approx(0.02)
    assert returns.loc[0, "primary_only_total_cost"] == 0.0
    assert returns.loc[0, "primary_only_net_return"] == pytest.approx(0.02)
    assert returns.loc[0, "meta_filtered_gross_return"] == pytest.approx(0.01)
    assert returns.loc[0, "meta_filtered_total_cost"] == 0.0
    assert returns.loc[0, "meta_filtered_net_return"] == pytest.approx(0.01)
    assert returns.loc[1, "meta_filtered_net_return"] == 0.0


def test_performance_and_aum_statistics_use_series_means():
    pnl = pd.Series([1.0, -0.5, 2.0])
    aum = pd.Series([100.0, 200.0])

    assert Performance.pnl(pnl) == 2.5
    assert GeneralCharacteristics.average_aum(aum) == 150.0


def test_efficiency_sharpe_ratio_handles_constant_returns():
    assert pd.isna(Efficiency.sharpe_ratio(pd.Series([0.01, 0.01])))


def test_classification_scores_support_weights_and_positive_label():
    y_true = pd.Series([-1, -1, 1, 1])
    y_pred = pd.Series([-1, 1, 1, -1])
    probabilities = np.array([
        [0.9, 0.1],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.7, 0.3],
    ])
    sample_weight = pd.Series([4.0, 1.0, 2.0, 1.0])

    assert ClassificationScores.accuracy(
        y_true,
        y_pred,
        sample_weight=sample_weight,
    ) == pytest.approx(accuracy_score(
        y_true,
        y_pred,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.precision(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ) == pytest.approx(precision_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.recall(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ) == pytest.approx(recall_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.f1_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ) == pytest.approx(f1_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.negative_log_loss(
        y_true,
        probabilities,
        labels=[-1, 1],
        sample_weight=sample_weight,
    ) == pytest.approx(-log_loss(
        y_true,
        probabilities,
        labels=[-1, 1],
        sample_weight=sample_weight,
    ))
