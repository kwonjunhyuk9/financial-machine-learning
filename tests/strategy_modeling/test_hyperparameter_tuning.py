from inspect import signature

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from src.strategy_modeling.hyperparameter_tuning import (
    MyPipeline,
    fit_classifier_with_hyperparameter_search,
)


def test_hyperparameter_search_uses_an_immutable_bagging_default():
    default = signature(fit_classifier_with_hyperparameter_search).parameters[
        "bagging"
    ].default

    assert default == (0, None, 1.0)
    assert isinstance(default, tuple)


def test_hyperparameter_search_accepts_a_list_bagging_configuration():
    index = pd.date_range("2025-01-01", periods=8, freq="D")
    features = pd.DataFrame({"value": np.arange(8)}, index=index)
    labels = pd.Series([0, 1] * 4, index=index)
    information_sets = pd.Series(index, index=index)
    pipeline = MyPipeline([("model", DecisionTreeClassifier(random_state=0))])

    fitted = fit_classifier_with_hyperparameter_search(
        features,
        labels,
        information_sets,
        pipeline,
        {"model__max_depth": [1]},
        cv=2,
        bagging=[0, None, 1.0],
        n_jobs=1,
    )

    assert isinstance(fitted, MyPipeline)


def test_hyperparameter_search_scores_validation_rows_with_weights(monkeypatch):
    captured = {}

    class FakeGridSearchCV:
        def __init__(self, estimator, param_grid, scoring, cv, n_jobs):
            captured["scoring"] = scoring
            self.best_estimator_ = estimator

        def fit(self, features, labels, **fit_params):
            return self

    monkeypatch.setattr(
        "src.strategy_modeling.hyperparameter_tuning.GridSearchCV",
        FakeGridSearchCV,
    )
    index = pd.Index(["a", "b", "c", "d"])
    features = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}, index=index)
    labels = pd.Series([0, 1, 1, 0], index=index)
    weights = pd.Series([1.0, 8.0, 1.0, 1.0], index=index)
    information_sets = pd.Series(
        pd.date_range("2025-01-01", periods=4, freq="D"),
        index=index,
    )
    pipeline = MyPipeline([("model", DecisionTreeClassifier(random_state=0))])

    fit_classifier_with_hyperparameter_search(
        features,
        labels,
        information_sets,
        pipeline,
        {"model__max_depth": [1]},
        cv=2,
        n_jobs=1,
        sample_weight=weights,
    )

    class FixedPredictions:
        def predict(self, validation_features):
            return np.array([1, 0, 0])

    validation_index = pd.Index(["b", "c", "d"])
    actual = captured["scoring"](
        FixedPredictions(),
        features.loc[validation_index],
        labels.loc[validation_index],
    )
    expected = f1_score(
        labels.loc[validation_index],
        [1, 0, 0],
        sample_weight=weights.loc[validation_index],
        zero_division=0,
    )

    assert actual == pytest.approx(expected)
