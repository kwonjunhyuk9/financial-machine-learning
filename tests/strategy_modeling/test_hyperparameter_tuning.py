from inspect import signature

import numpy as np
import pandas as pd
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
