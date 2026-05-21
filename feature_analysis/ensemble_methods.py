from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def build_bagging_classifier(
        n_estimators=1000,
        max_samples=1.0,
        max_features=1.0,
        min_weight_fraction_leaf=0.0,
        n_jobs=-1,
        random_state=None
):
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
        n_estimators=1000,
        max_features="sqrt",
        min_weight_fraction_leaf=0.0,
        n_jobs=-1,
        random_state=None
):
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
        n_estimators=100,
        learning_rate=1.0,
        max_depth=1,
        random_state=None
):
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    try:
        return AdaBoostClassifier(
            estimator=clf,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
    except TypeError:
        return AdaBoostClassifier(
            base_estimator=clf,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
