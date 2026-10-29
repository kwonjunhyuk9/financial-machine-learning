import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.modeling.hyperparameter_tuning import MyPipeline


@pytest.mark.parametrize("weight_input", ["series", "array", "step_parameter", "none"])
def test_pipeline_forwards_sample_weights_to_final_estimator(monkeypatch, weight_input):
    features = pd.DataFrame({"value": [0., 1., 2., 3.]})
    labels = pd.Series([0, 1, 1, 0])
    weights = pd.Series([1., 8., 1., 1.])
    original_weights = weights.copy()
    captured = []
    original_fit = DecisionTreeClassifier.fit

    def record_fit(self, X, y, sample_weight=None, **kwargs):
        captured.append(sample_weight)
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    monkeypatch.setattr(DecisionTreeClassifier, "fit", record_fit)
    pipeline = MyPipeline([
        ("scale", StandardScaler()),
        ("classifier", DecisionTreeClassifier(random_state=0)),
    ])
    fit_params = {}
    if weight_input == "step_parameter":
        fit_params["classifier__sample_weight"] = weights
    elif weight_input != "none":
        fit_params["sample_weight"] = weights.to_numpy() if weight_input == "array" else weights

    assert pipeline.fit(features, labels, **fit_params) is pipeline
    assert len(captured) == 1
    if weight_input == "none":
        assert captured[0] is None
    else:
        np.testing.assert_array_equal(captured[0], weights)
    pd.testing.assert_series_equal(weights, original_weights)
