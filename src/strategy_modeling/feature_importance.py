import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from src.strategy_modeling.cross_validation import PurgedKFold, score_cross_validation


def get_mdi_feature_importance(fit, featNames):
    """Compute mean decrease impurity feature importances.

    Args:
        fit: Fitted ensemble estimator.
        featNames: Feature names aligned with the estimator input.

    Returns:
        A frame with mean and standard-error importance estimates.
    """
    df0 = {
        i: tree.feature_importances_
        for i, tree in enumerate(fit.estimators_)
    }

    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = featNames

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


def get_mda_feature_importance(
        clf,
        X,
        y,
        cv,
        sample_weight,
        t1,
        pctEmbargo,
        scoring="neg_log_loss"
):
    """Compute mean decrease accuracy feature importances.

    Args:
        clf: Classifier to evaluate.
        X: Feature matrix.
        y: Target values.
        cv: Number of cross-validation folds.
        sample_weight: Sample weights aligned with ``X``.
        t1: Label end times for purged cross-validation.
        pctEmbargo: Embargo fraction applied to each fold.
        scoring: Scoring metric, either ``"neg_log_loss"`` or ``"accuracy"``.

    Returns:
        A tuple of the importance frame and the mean baseline score.

    Raises:
        Exception: If ``scoring`` is not supported.
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("wrong scoring method.")

    from sklearn.metrics import log_loss, accuracy_score

    cvGen = PurgedKFold(
        n_splits=cv,
        t1=t1,
        pctEmbargo=pctEmbargo
    )

    scr0 = pd.Series(dtype="float64")
    scr1 = pd.DataFrame(columns=X.columns, dtype="float64")

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0 = X.iloc[train, :]
        y0 = y.iloc[train]
        w0 = sample_weight.iloc[train]

        X1 = X.iloc[test, :]
        y1 = y.iloc[test]
        w1 = sample_weight.iloc[test]

        fit = clf.fit(
            X=X0,
            y=y0,
            sample_weight=w0.values
        )

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(
                y1,
                prob,
                sample_weight=w1.values,
                labels=clf.classes_
            )
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(
                y1,
                pred,
                sample_weight=w1.values
            )

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            X1_[j] = np.random.permutation(X1_[j].to_numpy())

            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(
                    y1,
                    prob,
                    sample_weight=w1.values,
                    labels=clf.classes_
                )
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(
                    y1,
                    pred,
                    sample_weight=w1.values
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
        featNames,
        clf,
        trnsX,
        cont,
        scoring,
        cvGen
):
    """Compute single-feature importances by isolated cross-validation.

    Args:
        featNames: Feature names to score individually.
        clf: Classifier to evaluate.
        trnsX: Training feature matrix.
        cont: Container with ``bin`` labels and ``w`` sample weights.
        scoring: Scoring metric passed to ``score_cross_validation``.
        cvGen: Cross-validation generator.

    Returns:
        A frame with mean and standard-error scores for each feature.
    """
    imp = pd.DataFrame(columns=["mean", "std"], dtype="float64")

    for featName in featNames:
        df0 = score_cross_validation(
            clf,
            X=trnsX[[featName]],
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen
        )

        imp.loc[featName, "mean"] = df0.mean()
        imp.loc[featName, "std"] = df0.std() * df0.shape[0] ** -0.5

    return imp


def get_eigen_components(dot, varThres):
    """Compute the leading eigenvalues and eigenvectors of a matrix.

    Args:
        dot: Symmetric matrix to decompose.
        varThres: Minimum cumulative explained-variance threshold.

    Returns:
        A tuple containing the retained eigenvalues and eigenvectors.
    """
    eVal, eVec = np.linalg.eigh(dot)

    idx = eVal.argsort()[::-1]
    eVal, eVec = eVal[idx], eVec[:, idx]

    eVal = pd.Series(
        eVal,
        index=["PC_" + str(i + 1) for i in range(eVal.shape[0])]
    )

    eVec = pd.DataFrame(
        eVec,
        index=dot.index,
        columns=eVal.index
    )

    eVec = eVec.loc[:, eVal.index]

    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)

    eVal = eVal.iloc[:dim + 1]
    eVec = eVec.iloc[:, :dim + 1]

    return eVal, eVec


def get_orthogonal_features(dfX, varThres=0.95):
    """Project features onto orthogonal principal components.

    Args:
        dfX: Feature matrix.
        varThres: Minimum cumulative explained-variance threshold.

    Returns:
        A frame of orthogonalized features.
    """
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)

    dot = pd.DataFrame(
        np.dot(dfZ.T, dfZ),
        index=dfX.columns,
        columns=dfX.columns
    )

    eVal, eVec = get_eigen_components(dot, varThres)

    dfP = np.dot(dfZ, eVec)

    dfP = pd.DataFrame(
        dfP,
        index=dfX.index,
        columns=eVec.columns
    )

    return dfP


def get_test_data(
        n_features=40,
        n_informative=10,
        n_redundant=10,
        n_samples=10000,
        random_state=0
):
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

    trnsX = pd.DataFrame(X, index=dates, columns=columns)
    t1 = pd.Series(list(dates[1:]) + [dates[-1]], index=dates)
    cont = pd.DataFrame({"bin": y, "w": 1.0, "t1": t1}, index=dates)

    return trnsX, cont


def get_feature_importance(
        trnsX,
        cont,
        n_estimators=1000,
        cv=10,
        max_samples=1.0,
        numThreads=24,
        pctEmbargo=0,
        scoring="accuracy",
        method="SFI",
        minWLeaf=0.0,
        **kargs
):
    """Estimate feature importance with MDI, MDA, or SFI.

    Args:
        trnsX: Training feature matrix.
        cont: Container with ``bin``, ``w``, and ``t1`` fields.
        n_estimators: Number of trees in the bagging ensemble.
        cv: Number of cross-validation folds.
        max_samples: Fraction of samples drawn for each bagging estimator.
        numThreads: Number of parallel workers.
        pctEmbargo: Embargo fraction applied to each fold.
        scoring: Scoring metric.
        method: Importance method, one of ``"MDI"``, ``"MDA"``, or ``"SFI"``.
        minWLeaf: Minimum weighted fraction required at a leaf.
        **kargs: Extra options, including ``random_state``.

    Returns:
        A tuple of the importance frame, out-of-bag score, and out-of-sample score.

    Raises:
        ValueError: If ``method`` is not one of ``"MDI"``, ``"MDA"``, or ``"SFI"``.
    """
    n_jobs = -1 if numThreads > 1 else 1
    random_state = kargs.get("random_state")

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=minWLeaf,
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
        X=trnsX,
        y=cont["bin"],
        sample_weight=cont["w"].values
    )

    oob = fit.oob_score_

    if method == "MDI":
        imp = get_mdi_feature_importance(
            fit,
            featNames=trnsX.columns
        )

        oos = score_cross_validation(
            clf,
            X=trnsX,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pctEmbargo=pctEmbargo,
            scoring=scoring
        ).mean()

    elif method == "MDA":
        imp, oos = get_mda_feature_importance(
            clf,
            X=trnsX,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pctEmbargo=pctEmbargo,
            scoring=scoring
        )

    elif method == "SFI":
        cvGen = PurgedKFold(
            n_splits=cv,
            t1=cont["t1"],
            pctEmbargo=pctEmbargo
        )

        oos = score_cross_validation(
            clf,
            X=trnsX,
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen
        ).mean()

        imp = get_single_feature_importance(
            featNames=trnsX.columns,
            clf=clf,
            trnsX=trnsX,
            cont=cont,
            scoring=scoring,
            cvGen=cvGen
        )

    else:
        raise ValueError("method must be one of: 'MDI', 'MDA', 'SFI'")

    return imp, oob, oos


def run_feature_importance_test(
        n_features=40,
        n_informative=10,
        n_redundant=10,
        n_estimators=1000,
        n_samples=10000,
        cv=10
):
    """Run a synthetic experiment comparing feature-importance methods.

    Args:
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        n_estimators: Number of trees in the ensemble.
        n_samples: Number of synthetic samples.
        cv: Number of cross-validation folds.

    Returns:
        A frame summarizing simulated importance allocations and scores.
    """
    trnsX, cont = get_test_data(
        n_features,
        n_informative,
        n_redundant,
        n_samples
    )

    dict0 = {
        "minWLeaf": [0.0],
        "scoring": ["accuracy"],
        "method": ["MDI", "MDA", "SFI"],
        "max_samples": [1.0]
    }

    jobs = [
        dict(zip(dict0, i))
        for i in product(*dict0.values())
    ]

    kargs = {
        "pathOut": "./feature_importance_test/",
        "n_estimators": n_estimators,
        "tag": "feature_importance_test",
        "cv": cv
    }

    out = []

    for job in jobs:
        job["simNum"] = (
                job["method"] + "_" +
                job["scoring"] + "_" +
                "%.2f" % job["minWLeaf"] + "_" +
                str(job["max_samples"])
        )

        print(job["simNum"])

        kargs.update(job)

        imp, oob, oos = get_feature_importance(
            trnsX=trnsX,
            cont=cont,
            **kargs
        )

        plot_feature_importance(
            imp=imp,
            oob=oob,
            oos=oos,
            **kargs
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
        ["method", "scoring", "minWLeaf", "max_samples"]
    )

    out = out[
        [
            "method",
            "scoring",
            "minWLeaf",
            "max_samples",
            "I",
            "R",
            "N",
            "oob",
            "oos"
        ]
    ]

    out.to_csv(kargs["pathOut"] + "stats.csv")

    return out


def plot_feature_importance(
        pathOut,
        imp,
        oob,
        oos,
        method,
        tag=0,
        simNum=0,
        **kargs
):
    """Plot and save a horizontal bar chart of feature importances.

    Args:
        pathOut: Output directory.
        imp: Importance frame with ``mean`` and ``std`` columns.
        oob: Out-of-bag score.
        oos: Out-of-sample score.
        method: Importance method name.
        tag: Plot label.
        simNum: Simulation label used in the output filename.
        **kargs: Additional compatibility arguments ignored by the plotter.

    Returns:
        None.
    """
    output_dir = Path(pathOut)
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
        " | simNum=" + str(simNum) +
        " | oob=" + str(round(oob, 4)) +
        " | oos=" + str(round(oos, 4))
    )

    plt.savefig(
        output_dir / ("feature_importance_" + str(simNum) + ".png"),
        dpi=100
    )

    plt.clf()
    plt.close()

    return
