from src.strategy_modeling.ensemble_methods import (
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
