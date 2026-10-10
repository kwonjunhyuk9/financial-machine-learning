import numpy as np
import pytest

from src.modeling.ensemble_methods import (
    build_bagging_classifier,
    build_boosting_classifier,
    build_random_forest_classifier,
)


def test_ensemble_factories_apply_requested_estimator_counts():
    assert build_bagging_classifier(n_estimators=3).n_estimators == 3
    assert build_random_forest_classifier(n_estimators=4).n_estimators == 4
    assert build_boosting_classifier(n_estimators=5).n_estimators == 5


def test_ensemble_factories_preserve_random_state():
    assert build_bagging_classifier(random_state=7).random_state == 7
    assert build_random_forest_classifier(random_state=7).random_state == 7
    assert build_boosting_classifier(random_state=7).random_state == 7


def test_boosting_factory_uses_entropy_based_stumps():
    classifier = build_boosting_classifier(max_depth=1)
    estimator = getattr(classifier, "estimator", None)
    if estimator is None:
        estimator = classifier.base_estimator

    assert estimator.criterion == "entropy"
    assert estimator.max_depth == 1


def test_ensemble_factories_preserve_library_leaf_defaults():
    bagging = build_bagging_classifier()
    forest = build_random_forest_classifier()
    boosting = build_boosting_classifier()

    for tree in (bagging.estimator, forest, boosting.estimator):
        assert tree.min_samples_leaf == 1
        assert tree.min_weight_fraction_leaf == 0.0


@pytest.mark.parametrize("depth", [2, 5, 10])
def test_ensemble_factories_apply_requested_depth(depth):
    bagging = build_bagging_classifier(max_depth=depth)
    forest = build_random_forest_classifier(max_depth=depth)
    boosting = build_boosting_classifier(max_depth=depth)

    for tree in (bagging.estimator, forest, boosting.estimator):
        assert tree.max_depth == depth


@pytest.mark.parametrize("negative_label", [-1, 0])
def test_boosting_uses_event_weights_without_class_balancing(negative_label):
    features = np.arange(12).reshape(-1, 1)
    labels = np.array([negative_label] * 9 + [1] * 3)
    sample_weight = np.array([1.0] * 9 + [2.0] * 3)
    classifier = build_boosting_classifier(n_estimators=1, random_state=7)

    classifier.fit(features, labels, sample_weight=sample_weight)

    root_weights = classifier.estimators_[0].tree_.value[0, 0]
    np.testing.assert_allclose(root_weights / root_weights.sum(), [3 / 5, 2 / 5])
    np.testing.assert_array_equal(classifier.predict(features), labels)
