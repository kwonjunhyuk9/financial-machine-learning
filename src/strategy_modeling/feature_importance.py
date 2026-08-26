from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from src.strategy_modeling.cross_validation import PurgedKFold
from src.strategy_modeling.model_workflow import (
    generate_oof_predictions,
    score_binary_predictions,
)


def get_mdi_feature_importance(
    fit: BaggingClassifier,
    feat_names: Sequence[str],
) -> pd.DataFrame:
    """Compute mean decrease impurity feature importances.

    Args:
        fit: Fitted ensemble estimator.
        feat_names: Feature names aligned with the estimator input.

    Returns:
        A frame with mean and standard-error importance estimates.
    """
    df0 = {
        i: tree.feature_importances_
        for i, tree in enumerate(fit.estimators_)
    }

    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = feat_names

    df0 = df0.replace(0, np.nan)

    imp = pd.concat(
        {
            "mean": df0.mean(),
            "std": df0.std() * df0.shape[0] ** -0.5
        },
        axis=1
    )

    imp /= imp["mean"].sum()

    return imp


def _get_class_labels(labels: pd.Series) -> list[int]:
    return np.sort(pd.unique(labels)).tolist()


def _predict_binary(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    positive_label: int = 1,
) -> pd.DataFrame:
    class_positions = np.flatnonzero(estimator.classes_ == positive_label)
    if class_positions.size != 1:
        raise ValueError(f"positive_label {positive_label!r} is absent from a fold")

    return pd.DataFrame(
        {
            "prediction": estimator.predict(features),
            "probability": estimator.predict_proba(features)[:, class_positions[0]],
        },
        index=features.index,
    )


def _select_feature_importance_score(
    scores: dict[str, float],
    scoring: str,
) -> float:
    if scoring == "neg_log_loss":
        return -scores["log_loss"]
    if scoring == "accuracy":
        return scores["accuracy"]
    raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'.")


def _score_oof_folds(
    predictions: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    scoring: str,
) -> np.ndarray:
    class_labels = _get_class_labels(labels)
    fold_scores = []

    for fold in sorted(predictions["fold"].unique()):
        fold_predictions = predictions[predictions["fold"].eq(fold)]
        fold_index = fold_predictions.index
        scores = score_binary_predictions(
            labels.loc[fold_index],
            fold_predictions["prediction"],
            fold_predictions["probability"],
            sample_weight.loc[fold_index],
            class_labels=class_labels,
            positive_label=1,
        )
        fold_scores.append(_select_feature_importance_score(scores, scoring))

    return np.asarray(fold_scores)


def get_mda_feature_importance(
    clf: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    sample_weight: pd.Series,
    t1: pd.Series,
    pct_embargo: float,
    scoring: str = "neg_log_loss",
    random_state: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, float]:
    """Compute mean decrease accuracy feature importances.

    Args:
        clf: Classifier to evaluate.
        X: Feature matrix.
        y: Target values.
        cv: Number of cross-validation folds.
        sample_weight: Sample weights aligned with ``X``.
        t1: Label end times for purged cross-validation.
        pct_embargo: Embargo fraction applied to each fold.
        scoring: Scoring metric, either ``"neg_log_loss"`` or ``"accuracy"``.
        random_state: Seed or generator used for feature permutations.

    Returns:
        A tuple of the importance frame and the mean baseline score.

    Raises:
        ValueError: If ``scoring`` is not supported.
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'.")

    rng = np.random.default_rng(random_state)
    class_labels = _get_class_labels(y)
    cv_gen = PurgedKFold(
        n_splits=cv,
        t1=t1,
        pct_embargo=pct_embargo
    )

    scr0 = pd.Series(dtype="float64")
    scr1 = pd.DataFrame(columns=X.columns, dtype="float64")

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0 = X.iloc[train, :]
        y0 = y.iloc[train]
        w0 = sample_weight.iloc[train]

        X1 = X.iloc[test, :]
        y1 = y.iloc[test]
        w1 = sample_weight.iloc[test]

        fit = clone(clf).fit(
            X=X0,
            y=y0,
            sample_weight=w0.values
        )
        baseline_predictions = _predict_binary(fit, X1)
        baseline_scores = score_binary_predictions(
            y1,
            baseline_predictions["prediction"],
            baseline_predictions["probability"],
            w1,
            class_labels=class_labels,
            positive_label=1,
        )
        scr0.loc[i] = _select_feature_importance_score(
            baseline_scores,
            scoring,
        )

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            X1_[j] = rng.permutation(X1_[j].to_numpy())

            permuted_predictions = _predict_binary(fit, X1_)
            permuted_scores = score_binary_predictions(
                y1,
                permuted_predictions["prediction"],
                permuted_predictions["probability"],
                w1,
                class_labels=class_labels,
                positive_label=1,
            )
            scr1.loc[i, j] = _select_feature_importance_score(
                permuted_scores,
                scoring,
            )

    imp = (-scr1).add(scr0, axis=0)

    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)

    imp = pd.concat(
        {
            "mean": imp.mean(),
            "std": imp.std() * imp.shape[0] ** -0.5
        },
        axis=1
    )

    return imp, scr0.mean()


def get_single_feature_importance(
    feat_names: Sequence[str],
    clf: BaseEstimator,
    trns_x: pd.DataFrame,
    cont: pd.DataFrame,
    scoring: str,
    cv_gen: PurgedKFold,
) -> pd.DataFrame:
    """Compute single-feature importances by isolated cross-validation.

    Args:
        feat_names: Feature names to score individually.
        clf: Classifier to evaluate.
        trns_x: Training feature matrix.
        cont: Container with ``bin`` labels and ``w`` sample weights.
        scoring: Scoring metric, either ``"neg_log_loss"`` or ``"accuracy"``.
        cv_gen: Cross-validation generator.

    Returns:
        A frame with mean and standard-error scores for each feature.
    """
    imp = pd.DataFrame(columns=["mean", "std"], dtype="float64")

    for feat_name in feat_names:
        predictions = generate_oof_predictions(
            estimator=clf,
            features=trns_x[[feat_name]],
            labels=cont["bin"],
            sample_weight=cont["w"],
            cv=cv_gen,
            positive_label=1,
        )
        fold_scores = _score_oof_folds(
            predictions,
            labels=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
        )

        imp.loc[feat_name, "mean"] = fold_scores.mean()
        imp.loc[feat_name, "std"] = (
            fold_scores.std() * fold_scores.shape[0] ** -0.5
        )

    return imp


def get_eigen_components(
    dot: pd.DataFrame,
    var_thres: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the leading eigenvalues and eigenvectors of a matrix.

    Args:
        dot: Symmetric matrix to decompose.
        var_thres: Minimum cumulative explained-variance threshold.

    Returns:
        A tuple containing the retained eigenvalues and eigenvectors.
    """
    e_val, e_vec = np.linalg.eigh(dot)

    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]

    e_val = pd.Series(
        e_val,
        index=["PC_" + str(i + 1) for i in range(e_val.shape[0])]
    )

    e_vec = pd.DataFrame(
        e_vec,
        index=dot.index,
        columns=e_val.index
    )

    e_vec = e_vec.loc[:, e_val.index]

    cum_var = e_val.cumsum() / e_val.sum()
    dim = cum_var.values.searchsorted(var_thres)

    e_val = e_val.iloc[:dim + 1]
    e_vec = e_vec.iloc[:, :dim + 1]

    return e_val, e_vec


def get_orthogonal_features(
    df_x: pd.DataFrame,
    var_thres: float = 0.95,
) -> pd.DataFrame:
    """Project features onto orthogonal principal components.

    Args:
        df_x: Feature matrix.
        var_thres: Minimum cumulative explained-variance threshold.

    Returns:
        A frame of orthogonalized features.
    """
    df_z = df_x.sub(df_x.mean(), axis=1).div(df_x.std(), axis=1)

    dot = pd.DataFrame(
        np.dot(df_z.T, df_z),
        index=df_x.columns,
        columns=df_x.columns
    )

    e_val, e_vec = get_eigen_components(dot, var_thres)

    df_p = np.dot(df_z, e_vec)

    df_p = pd.DataFrame(
        df_p,
        index=df_x.index,
        columns=e_vec.columns
    )

    return df_p


def get_test_data(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_samples: int = 10000,
    random_state: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic classification data for feature-importance tests.

    Args:
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        n_samples: Number of synthetic samples.
        random_state: Random seed used by the data generator.

    Returns:
        A tuple of feature matrix and label/sample-weight container.

    Raises:
        ValueError: If feature counts are inconsistent.
    """
    n_noise = n_features - n_informative - n_redundant

    if n_noise < 0:
        raise ValueError("n_features must be at least n_informative + n_redundant")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        shuffle=False,
        random_state=random_state
    )

    dates = pd.date_range("2000-01-01", periods=n_samples, freq="D")
    columns = (
        [f"I_{i}" for i in range(n_informative)] +
        [f"R_{i}" for i in range(n_redundant)] +
        [f"N_{i}" for i in range(n_noise)]
    )

    trns_x = pd.DataFrame(X, index=dates, columns=columns)
    t1 = pd.Series(list(dates[1:]) + [dates[-1]], index=dates)
    cont = pd.DataFrame({"bin": y, "w": 1.0, "t1": t1}, index=dates)

    return trns_x, cont


def get_feature_importance(
    trns_x: pd.DataFrame,
    cont: pd.DataFrame,
    n_estimators: int = 1000,
    cv: int = 10,
    max_samples: float = 1.0,
    num_threads: int = 24,
    pct_embargo: float = 0.0,
    scoring: str = "accuracy",
    method: str = "SFI",
    min_w_leaf: float = 0.0,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Estimate feature importance with MDI, MDA, or SFI.

    Args:
        trns_x: Training feature matrix.
        cont: Container with ``bin``, ``w``, and ``t1`` fields.
        n_estimators: Number of trees in the bagging ensemble.
        cv: Number of cross-validation folds.
        max_samples: Fraction of samples drawn for each bagging estimator.
        num_threads: Number of parallel workers.
        pct_embargo: Embargo fraction applied to each fold.
        scoring: Scoring metric.
        method: Importance method, one of ``"MDI"``, ``"MDA"``, or ``"SFI"``.
        min_w_leaf: Minimum weighted fraction required at a leaf.
        random_state: Seed used by the estimator and feature permutations.

    Returns:
        A tuple of the importance frame, out-of-bag score, and out-of-sample score.

    Raises:
        ValueError: If ``method`` is not one of ``"MDI"``, ``"MDA"``, or ``"SFI"``.
    """
    n_jobs = -1 if num_threads > 1 else 1

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=min_w_leaf,
        random_state=random_state
    )

    clf = BaggingClassifier(
        estimator=clf,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state
    )

    fit = clf.fit(
        X=trns_x,
        y=cont["bin"],
        sample_weight=cont["w"].values
    )

    oob = fit.oob_score_

    if method == "MDI":
        imp = get_mdi_feature_importance(
            fit,
            feat_names=trns_x.columns
        )

        cv_gen = PurgedKFold(
            n_splits=cv,
            t1=cont["t1"],
            pct_embargo=pct_embargo,
        )
        predictions = generate_oof_predictions(
            estimator=clf,
            features=trns_x,
            labels=cont["bin"],
            sample_weight=cont["w"],
            cv=cv_gen,
            positive_label=1,
        )
        oos = _score_oof_folds(
            predictions,
            labels=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
        ).mean()

    elif method == "MDA":
        imp, oos = get_mda_feature_importance(
            clf,
            X=trns_x,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pct_embargo=pct_embargo,
            scoring=scoring,
            random_state=random_state,
        )

    elif method == "SFI":
        cv_gen = PurgedKFold(
            n_splits=cv,
            t1=cont["t1"],
            pct_embargo=pct_embargo
        )

        predictions = generate_oof_predictions(
            estimator=clf,
            features=trns_x,
            labels=cont["bin"],
            sample_weight=cont["w"],
            cv=cv_gen,
            positive_label=1,
        )
        oos = _score_oof_folds(
            predictions,
            labels=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
        ).mean()

        imp = get_single_feature_importance(
            feat_names=trns_x.columns,
            clf=clf,
            trns_x=trns_x,
            cont=cont,
            scoring=scoring,
            cv_gen=cv_gen
        )

    else:
        raise ValueError("method must be one of: 'MDI', 'MDA', 'SFI'")

    return imp, oob, oos


def run_feature_importance_test(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_estimators: int = 1000,
    n_samples: int = 10000,
    cv: int = 10,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Run a synthetic experiment comparing feature-importance methods.

    Args:
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        n_estimators: Number of trees in the ensemble.
        n_samples: Number of synthetic samples.
        cv: Number of cross-validation folds.
        random_state: Seed used by data generation and model evaluation.

    Returns:
        A frame summarizing simulated importance allocations and scores.
    """
    trns_x, cont = get_test_data(
        n_features,
        n_informative,
        n_redundant,
        n_samples,
        random_state=random_state,
    )

    dict0 = {
        "min_w_leaf": [0.0],
        "scoring": ["accuracy"],
        "method": ["MDI", "MDA", "SFI"],
        "max_samples": [1.0]
    }

    jobs = [
        dict(zip(dict0, i))
        for i in product(*dict0.values())
    ]

    path_out = "./feature_importance_test/"

    out = []

    for job in jobs:
        job["sim_num"] = (
                job["method"] + "_" +
                job["scoring"] + "_" +
                "%.2f" % job["min_w_leaf"] + "_" +
                str(job["max_samples"])
        )

        logger.info("Running feature-importance simulation {}.", job["sim_num"])
        imp, oob, oos = get_feature_importance(
            trns_x=trns_x,
            cont=cont,
            n_estimators=n_estimators,
            cv=cv,
            max_samples=job["max_samples"],
            scoring=job["scoring"],
            method=job["method"],
            min_w_leaf=job["min_w_leaf"],
            random_state=random_state,
        )

        plot_feature_importance(
            path_out=path_out,
            imp=imp,
            oob=oob,
            oos=oos,
            method=job["method"],
            tag="feature_importance_test",
            sim_num=job["sim_num"],
        )

        df0 = imp[["mean"]] / imp["mean"].abs().sum()
        df0["type"] = [
            "I" if i < n_informative else
            "R" if i < n_informative + n_redundant else
            "N"
            for i in range(df0.shape[0])
        ]

        df0 = df0.groupby("type")["mean"].sum().to_dict()
        df0.update({"oob": oob, "oos": oos})
        df0.update(job)

        out.append(df0)

    out = pd.DataFrame(out).sort_values(
        ["method", "scoring", "min_w_leaf", "max_samples"]
    )

    out = out[
        [
            "method",
            "scoring",
            "min_w_leaf",
            "max_samples",
            "I",
            "R",
            "N",
            "oob",
            "oos"
        ]
    ]

    out.to_csv(path_out + "stats.csv")

    return out


def plot_feature_importance(
    path_out: str | Path,
    imp: pd.DataFrame,
    oob: float,
    oos: float,
    method: str,
    tag: Any = 0,
    sim_num: Any = 0,
) -> None:
    """Plot and save a horizontal bar chart of feature importances.

    Args:
        path_out: Output directory.
        imp: Importance frame with ``mean`` and ``std`` columns.
        oob: Out-of-bag score.
        oos: Out-of-sample score.
        method: Importance method name.
        tag: Plot label.
        sim_num: Simulation label used in the output filename.

    Returns:
        None.
    """
    output_dir = Path(path_out)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, imp.shape[0] / 5.0))

    imp = imp.sort_values("mean", ascending=True)

    ax = imp["mean"].plot(
        kind="barh",
        color="b",
        alpha=0.25,
        xerr=imp["std"],
        error_kw={"ecolor": "r"}
    )

    if method == "MDI":
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(
            1.0 / imp.shape[0],
            linewidth=1,
            color="r",
            linestyle="dotted"
        )

    ax.get_yaxis().set_visible(False)

    for i, j in zip(ax.patches, imp.index):
        ax.text(
            i.get_width() / 2,
            i.get_y() + i.get_height() / 2,
            j,
            ha="center",
            va="center",
            color="black"
        )

    plt.title(
        "tag=" + str(tag) +
        " | sim_num=" + str(sim_num) +
        " | oob=" + str(round(oob, 4)) +
        " | oos=" + str(round(oos, 4))
    )

    plt.savefig(
        output_dir / ("feature_importance_" + str(sim_num) + ".png"),
        dpi=100
    )

    plt.clf()
    plt.close()

    return
