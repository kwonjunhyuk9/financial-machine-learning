from __future__ import annotations

from inspect import signature

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier


def build_bagging_classifier(
    n_estimators: int = 1000,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    min_weight_fraction_leaf: float = 0.0,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> BaggingClassifier:
    """Build a bagging classifier with entropy-based decision trees.

    Args:
        n_estimators: Number of trees in the ensemble.
        max_samples: Fraction of samples drawn for each base estimator.
        max_features: Fraction of features drawn for each base estimator.
        min_weight_fraction_leaf: Minimum weighted fraction required at a leaf.
        n_jobs: Number of parallel workers.
        random_state: Random seed.

    Returns:
        A configured ``BaggingClassifier`` instance.
    """
    clf = DecisionTreeClassifier(
        criterion="entropy",
        class_weight="balanced",
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        random_state=random_state
    )

    return BaggingClassifier(
        estimator=clf,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state
    )


def build_random_forest_classifier(
    n_estimators: int = 1000,
    max_features: str | int | float | None = "sqrt",
    min_weight_fraction_leaf: float = 0.0,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> RandomForestClassifier:
    """Build a random forest classifier for imbalanced classification tasks.

    Args:
        n_estimators: Number of trees in the forest.
        max_features: Feature-subsampling rule for each split.
        min_weight_fraction_leaf: Minimum weighted fraction required at a leaf.
        n_jobs: Number of parallel workers.
        random_state: Random seed.

    Returns:
        A configured ``RandomForestClassifier`` instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        bootstrap=True,
        class_weight="balanced_subsample",
        max_features=max_features,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        n_jobs=n_jobs,
        random_state=random_state
    )


def build_boosting_classifier(
    n_estimators: int = 100,
    learning_rate: float = 1.0,
    max_depth: int = 1,
    random_state: int | None = None,
) -> AdaBoostClassifier:
    """Build an AdaBoost classifier with entropy-based shallow decision trees.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage applied to each estimator.
        max_depth: Maximum depth of the decision-tree base estimator.
        random_state: Random seed.

    Returns:
        A configured ``AdaBoostClassifier`` instance.
    """
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        random_state=random_state
    )

    estimator_parameter = (
        "estimator"
        if "estimator" in signature(AdaBoostClassifier).parameters
        else "base_estimator"
    )
    return AdaBoostClassifier(
        **{estimator_parameter: clf},
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )


def build_gradient_boosting_classifier(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int | None = None,
) -> GradientBoostingClassifier:
    """Build a gradient-boosting classifier with fixed-depth regression trees.

    Args:
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage applied to each stage.
        max_depth: Maximum depth of each regression tree.
        random_state: Random seed.

    Returns:
        A configured ``GradientBoostingClassifier`` instance.
    """
    return GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
