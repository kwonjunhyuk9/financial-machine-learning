from __future__ import annotations

from inspect import signature

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier


def build_bagging_classifier(
    n_estimators: int = 1000,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> BaggingClassifier:
    """Build a bagging classifier with entropy-based decision trees.

    Args:
        n_estimators: Number of trees in the ensemble.
        max_samples: Fraction of samples drawn for each base estimator.
        max_features: Fraction of features drawn for each base estimator.
        max_depth: Maximum depth of each decision-tree base estimator.
        n_jobs: Number of parallel workers.
        random_state: Random seed.

    Returns:
        A configured ``BaggingClassifier`` instance.
    """
    clf = DecisionTreeClassifier(
        criterion="entropy",
        class_weight="balanced",
        max_depth=max_depth,
        random_state=random_state
    )

    return BaggingClassifier(
        estimator=clf,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        oob_score=False,
        n_jobs=n_jobs,
        random_state=random_state
    )


def build_random_forest_classifier(
    n_estimators: int = 1000,
    max_features: str | int | float | None = "sqrt",
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int | None = None,
    max_samples: float | None = None,
) -> RandomForestClassifier:
    """Build a random forest classifier for imbalanced classification tasks.

    Args:
        n_estimators: Number of trees in the forest.
        max_features: Feature-subsampling rule for each split.
        max_depth: Maximum depth of each tree.
        max_samples: Fraction of samples drawn for each tree.
        n_jobs: Number of parallel workers.
        random_state: Random seed.

    Returns:
        A configured ``RandomForestClassifier`` instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        bootstrap=True,
        max_samples=max_samples,
        class_weight="balanced_subsample",
        max_features=max_features,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state
    )


def build_boosting_classifier(
    n_estimators: int = 100,
    learning_rate: float = 1.0,
    max_depth: int | None = 1,
    random_state: int | None = None,
) -> AdaBoostClassifier:
    """Build a boosting classifier with entropy-based shallow decision trees.

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
