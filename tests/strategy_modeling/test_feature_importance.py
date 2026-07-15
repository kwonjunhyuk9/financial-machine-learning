import pandas as pd
import pytest

from src.strategy_modeling.feature_importance import get_eigen_components, get_test_data


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
