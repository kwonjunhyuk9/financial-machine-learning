import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, log_loss
from sklearn.tree import DecisionTreeClassifier

from src.modeling.hyperparameter_tuning import (
    MyPipeline,
    fit_classifier_with_hyperparameter_search,
)


def test_hyperparameter_search_returns_the_best_pipeline():
    index = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
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
        n_jobs=1,
    )

    assert isinstance(fitted, MyPipeline)


@pytest.mark.parametrize("label_values", [(0, 1), (-1, 1)])
@pytest.mark.parametrize("forward_weights", [False, True])
def test_hyperparameter_search_scores_validation_rows_with_weights(
    monkeypatch, label_values, forward_weights,
):
    captured = {}

    class FakeGridSearchCV:
        def __init__(self, estimator, param_grid, scoring, cv, n_jobs):
            captured["scoring"] = scoring
            self.best_estimator_ = estimator

        def fit(self, features, labels, **fit_params):
            return self

    monkeypatch.setattr(
        "src.modeling.hyperparameter_tuning.GridSearchCV",
        FakeGridSearchCV,
    )
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    features = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}, index=index)
    labels = pd.Series(np.asarray(label_values)[[0, 1, 1, 0]], index=index)
    weights = pd.Series([1.0, 8.0, 1.0, 1.0], index=index)
    information_sets = pd.Series(
        index,
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
        classes_ = np.asarray(label_values)

        def predict(self, validation_features):
            return self.classes_[[1, 0, 0]]

        def predict_proba(self, validation_features):
            return np.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]])

    validation_index = index[1:]
    validation_weights = (
        np.array([2.0, 3.0, 9.0]) if forward_weights
        else weights.loc[validation_index].to_numpy()
    )
    actual = captured["scoring"](
        FixedPredictions(),
        features.loc[validation_index],
        labels.loc[validation_index],
        **({"sample_weight": validation_weights} if forward_weights else {}),
    )
    if label_values == (0, 1):
        expected = f1_score(
            labels.loc[validation_index],
            FixedPredictions().predict(features.loc[validation_index]),
            sample_weight=validation_weights,
            zero_division=0,
        )
    else:
        expected = -log_loss(
            labels.loc[validation_index],
            FixedPredictions().predict_proba(features.loc[validation_index]),
            labels=label_values,
            sample_weight=validation_weights,
        )

    assert actual == pytest.approx(expected)
