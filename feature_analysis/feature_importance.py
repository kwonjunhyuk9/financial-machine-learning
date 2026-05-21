import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from feature_analysis.cross_validation import PurgedKFold, cvScore


def featImpMDI(fit, featNames):
    df0 = {
        i: tree.feature_importances_
        for i, tree in enumerate(fit.estimators_)
    }

    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = featNames

    # With max_features=1, zero can mean "not selected" rather than "not useful".
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


def featImpMDA(
        clf,
        X,
        y,
        cv,
        sample_weight,
        t1,
        pctEmbargo,
        scoring="neg_log_loss"
):
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

            # Shuffle one feature at a time to measure its marginal contribution.
            np.random.shuffle(X1_[j].values)

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


def auxFeatImpSFI(
        featNames,
        clf,
        trnsX,
        cont,
        scoring,
        cvGen
):
    imp = pd.DataFrame(columns=["mean", "std"], dtype="float64")

    for featName in featNames:
        df0 = cvScore(
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


def get_eVec(dot, varThres):
    eVal, eVec = np.linalg.eigh(dot)

    # Sort principal components by explained variance, descending.
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


def orthoFeats(dfX, varThres=0.95):
    # Standardize before projecting onto orthogonal components.
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)

    dot = pd.DataFrame(
        np.dot(dfZ.T, dfZ),
        index=dfX.columns,
        columns=dfX.columns
    )

    eVal, eVec = get_eVec(dot, varThres)

    dfP = np.dot(dfZ, eVec)

    dfP = pd.DataFrame(
        dfP,
        index=dfX.index,
        columns=eVec.columns
    )

    return dfP


def featImportance(
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
    n_jobs = -1 if numThreads > 1 else 1

    # Keep tree-level feature selection narrow to reduce masking effects.
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=minWLeaf
    )

    clf = BaggingClassifier(
        estimator=clf,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=n_jobs
    )

    fit = clf.fit(
        X=trnsX,
        y=cont["bin"],
        sample_weight=cont["w"].values
    )

    oob = fit.oob_score_

    if method == "MDI":
        imp = featImpMDI(
            fit,
            featNames=trnsX.columns
        )

        oos = cvScore(
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
        imp, oos = featImpMDA(
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

        oos = cvScore(
            clf,
            X=trnsX,
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen
        ).mean()

        # Compute single-feature scores directly instead of parallel dispatch.
        imp = auxFeatImpSFI(
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


def testFunc(
        n_features=40,
        n_informative=10,
        n_redundant=10,
        n_estimators=1000,
        n_samples=10000,
        cv=10
):
    trnsX, cont = getTestData(
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
        "pathOut": "./testFunc/",
        "n_estimators": n_estimators,
        "tag": "testFunc",
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

        imp, oob, oos = featImportance(
            trnsX=trnsX,
            cont=cont,
            **kargs
        )

        plotFeatImportance(
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


def plotFeatImportance(
        pathOut,
        imp,
        oob,
        oos,
        method,
        tag=0,
        simNum=0,
        **kargs
):
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
        pathOut + "featImportance_" + str(simNum) + ".png",
        dpi=100
    )

    plt.clf()
    plt.close()

    return
