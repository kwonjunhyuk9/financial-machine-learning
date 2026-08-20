import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.strategy_modeling.feature_importance import (
    get_eigen_components,
    get_feature_importance,
    get_mda_feature_importance,
    get_test_data,
)


def test_get_test_data_returns_requested_feature_shape():
    features, container = get_test_data(4, 2, 1, 30)

    assert features.shape == (30, 4)
    assert container.columns.tolist() == ["bin", "w", "t1"]
    assert len(container) == 30


def test_eigen_components_and_invalid_feature_counts_are_validated():
    matrix = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    eigenvalues, eigenvectors = get_eigen_components(matrix, var_thres=0.9)

    assert eigenvalues.index.tolist() == eigenvectors.columns.tolist()
    with pytest.raises(ValueError, match="at least"):
        get_test_data(n_features=2, n_informative=2, n_redundant=1, n_samples=10)


def test_mda_feature_importance_is_reproducible_with_a_seed():
    features, container = get_test_data(4, 2, 1, 30, random_state=11)
    kwargs = {
        "clf": DecisionTreeClassifier(random_state=11),
        "X": features,
        "y": container["bin"],
        "cv": 3,
        "sample_weight": container["w"],
        "t1": container["t1"],
        "pct_embargo": 0.0,
        "scoring": "accuracy",
        "random_state": 11,
    }

    first_importance, first_score = get_mda_feature_importance(**kwargs)
    second_importance, second_score = get_mda_feature_importance(**kwargs)

    pd.testing.assert_frame_equal(first_importance, second_importance)
    assert first_score == pytest.approx(second_score)
    assert np.isfinite(first_score)


def test_mda_feature_importance_rejects_unsupported_scoring():
    with pytest.raises(ValueError, match="neg_log_loss.*accuracy"):
        get_mda_feature_importance(
            DecisionTreeClassifier(),
            X=pd.DataFrame(),
            y=pd.Series(dtype=int),
            cv=2,
            sample_weight=pd.Series(dtype=float),
            t1=pd.Series(dtype="datetime64[ns]"),
            pct_embargo=0.0,
            scoring="f1",
        )


def test_feature_importance_forwards_seed_to_mda():
    features, container = get_test_data(4, 2, 1, 40, random_state=13)

    first = get_feature_importance(
        features,
        container,
        n_estimators=20,
        cv=2,
        max_samples=0.8,
        num_threads=1,
        method="MDA",
        random_state=13,
    )
    second = get_feature_importance(
        features,
        container,
        n_estimators=20,
        cv=2,
        max_samples=0.8,
        num_threads=1,
        method="MDA",
        random_state=13,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[1:] == pytest.approx(second[1:])


def test_feature_importance_rejects_ignored_compatibility_options():
    features, container = get_test_data(4, 2, 1, 30, random_state=17)

    with pytest.raises(TypeError, match="unexpected keyword"):
        get_feature_importance(features, container, ignored_option=True)
