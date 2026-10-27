from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.modeling.modeling_workflow import build_meta_model_frame, generate_oof_predictions

from src.modeling.purged_validation import (
    PurgedKFold,
    _embargo_train_indices,
    _purge_train_indices,
    _validate_event_intervals,
    time_groups,
)


def combinatorial_purged_cross_validation(
    samples_info_sets: pd.Series,
    outer_num_total_groups: int,
    outer_num_test_groups: int,
    outer_pct_embargo: float = 0.0,
) -> pd.DataFrame:
    """Generate combinatorial purged cross-validation split metadata.

    Args:
        samples_info_sets: Series indexed by observation start time and valued by
            observation end time.
        outer_num_total_groups: Number of contiguous groups used to partition
            the sorted observations.
        outer_num_test_groups: Number of groups used as the test set in each split.
        outer_pct_embargo: Finite fraction in [0, 1) of all observations to embargo,
            rounded up, strictly after each contiguous test run's latest event
            end. Adjacent test groups form one run.

    Returns:
        A frame with one row per test-group combination and split, group, and
        positional-index metadata.

    Raises:
        ValueError: If inputs are invalid or cannot form the requested CPCV splits.
    """
    samples_info_sets = _validate_samples_info_sets(samples_info_sets)

    if outer_num_total_groups < 2:
        raise ValueError("outer_num_total_groups must be greater than 1")

    if outer_num_total_groups > samples_info_sets.shape[0]:
        raise ValueError("outer_num_total_groups cannot exceed the number of observations")

    if outer_num_test_groups < 1 or outer_num_test_groups >= outer_num_total_groups:
        raise ValueError("outer_num_test_groups must be in [1, outer_num_total_groups)")

    groups = time_groups(samples_info_sets.index, outer_num_total_groups)
    out = []

    for split_num, test_groups in enumerate(
            combinations(range(outer_num_total_groups), outer_num_test_groups)
    ):
        test_indices = np.concatenate([
            groups[group]
            for group in test_groups
        ])
        train_indices = np.setdiff1d(
            np.arange(samples_info_sets.shape[0]),
            test_indices,
            assume_unique=True
        )
        train_indices = _purge_train_indices(
            samples_info_sets=samples_info_sets,
            train_indices=train_indices,
            test_indices=test_indices
        )
        train_indices = _embargo_train_indices(
            samples_info_sets=samples_info_sets,
            train_indices=train_indices,
            test_indices=test_indices,
            pct_embargo=outer_pct_embargo
        )

        out.append({
            "split_num": split_num,
            "train_groups": tuple(
                group
                for group in range(outer_num_total_groups)
                if group not in test_groups
            ),
            "test_groups": test_groups,
            "train_indices": tuple(train_indices.tolist()),
            "test_indices": tuple(test_indices.tolist())
        })

    return pd.DataFrame(out)


def get_combinatorial_backtest_paths(
    splits: pd.DataFrame,
    outer_num_total_groups: int,
) -> pd.DataFrame:
    """Arrange CPCV split identifiers into combinatorial backtest paths.

    Args:
        splits: Output from ``combinatorial_purged_cross_validation``.
        outer_num_total_groups: Number of contiguous observation groups used to
            make the supplied CPCV splits.

    Returns:
        A frame indexed by group number with one column per CPCV backtest path.

    Raises:
        ValueError: If split metadata cannot form complete paths.
    """
    splits = _validate_splits(splits=splits, outer_num_total_groups=outer_num_total_groups)
    outer_num_test_groups = len(splits.iloc[0]["test_groups"])
    num_paths = comb(outer_num_total_groups - 1, outer_num_test_groups - 1)

    path_counts = pd.Series(0, index=range(outer_num_total_groups))
    paths = pd.DataFrame(
        np.nan,
        index=pd.Index(range(outer_num_total_groups), name="group"),
        columns=[f"path_{path}" for path in range(num_paths)]
    )

    for _, split in splits.iterrows():
        for group in split["test_groups"]:
            path_num = path_counts.loc[group]

            if path_num >= num_paths:
                raise ValueError("splits cannot form complete backtest paths")

            paths.loc[group, f"path_{path_num}"] = split["split_num"]
            path_counts.loc[group] += 1

    if paths.shape[1] != num_paths or paths.isna().any().any():
        raise ValueError("splits cannot form complete backtest paths")

    return paths.astype("int64")


def generate_cpcv_predictions(
    development: pd.DataFrame,
    splits: pd.DataFrame,
    primary_artifact: dict[str, object],
    meta_artifacts: dict[str, dict[str, object]],
    *,
    inner_num_splits: int,
    inner_pct_embargo: float,
) -> pd.DataFrame:
    """Refit shared primary and candidate meta models; return test-only predictions.

    Args:
        development: Development-only events with a composite event index and
            positional indices match the supplied splits.
        splits: CPCV split metadata containing train_indices and test_indices.
        primary_artifact: Frozen primary estimator and ordered feature_columns.
        meta_artifacts: Candidate names mapped to frozen meta configurations.
        inner_num_splits: Purged fold count for primary OOF meta-training inputs.
        inner_pct_embargo: Embargo fraction for the inner purged folds.

    Returns:
        Test-only predictions ordered by split_num and observation_position,
        including family, split and observation identity.

    Raises:
        ValueError: If events include a partition other than development.
    """
    if not development["partition"].eq("development").all():
        raise ValueError("CPCV predictions require development observations only")
    primary_features = primary_artifact["feature_columns"]
    split_predictions = []
    for split in splits.itertuples(index=False):
        train = development.iloc[list(split.train_indices)].copy()
        test = development.iloc[list(split.test_indices)].copy()

        primary_train_estimator = clone(primary_artifact["estimator"])
        inner_cv = PurgedKFold(n_splits=inner_num_splits, t1=train["event_end"], pct_embargo=inner_pct_embargo)
        primary_train_oof = generate_oof_predictions(
            primary_train_estimator,
            train[primary_features],
            train["primary_label"].astype("int8"),
            train["sample_weight"],
            inner_cv,
            positive_label=1,
        )
        meta_train = build_meta_model_frame(
            train,
            primary_train_oof[["prediction", "probability", "prediction_source"]],
        )

        fitted_primary = clone(primary_artifact["estimator"]).fit(
            train[primary_features],
            train["primary_label"].astype("int8"),
            sample_weight=train["sample_weight"].to_numpy(),
        )
        primary_positive_probability = fitted_primary.predict_proba(test[primary_features])[:, list(fitted_primary.classes_).index(1)]
        primary_prediction = fitted_primary.predict(test[primary_features]).astype("int8")

        test_meta = test.copy()
        test_meta["primary_prediction"] = primary_prediction
        test_meta["primary_positive_probability"] = primary_positive_probability
        test_meta["primary_class_probability"] = np.maximum(primary_positive_probability, 1.0 - primary_positive_probability)

        for family, meta_artifact in meta_artifacts.items():
            meta_features = meta_artifact["feature_columns"]
            fitted_meta = clone(meta_artifact["estimator"]).fit(
                meta_train[meta_features],
                meta_train["meta_label"].astype("int8"),
                sample_weight=meta_train["sample_weight"].to_numpy(),
            )
            meta_pass_probability = fitted_meta.predict_proba(test_meta[meta_features])[:, list(fitted_meta.classes_).index(1)]
            meta_prediction = fitted_meta.predict(test_meta[meta_features]).astype("int8")

            result = test_meta[[
                "event_end", "vertical_barrier", "horizon_volatility", "raw_return",
                "primary_label", "sample_weight", "partition",
                "holdout_boundary", "mean_sentiment_score",
            ]].copy()
            result["primary_prediction"] = primary_prediction
            result["primary_positive_probability"] = primary_positive_probability
            result["meta_pass_probability"] = meta_pass_probability
            result["meta_prediction"] = meta_prediction
            result["meta_label"] = (primary_prediction * test["raw_return"] > 0).astype("int8")
            result["model_family"] = family
            result["split_num"] = split.split_num
            result["observation_position"] = list(split.test_indices)
            split_predictions.append(result)

    cpcv_predictions = pd.concat(split_predictions).sort_values(["split_num", "observation_position"])
    return cpcv_predictions


def get_cpcv_price_calibrations(
    development: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Calibrate every symbol from each CPCV split's training observations.

    Returns one row per split and symbol with ``w`` and an exclusion reason.
    Test observations never contribute to their split's calibration.
    """
    from src.backtesting.portfolio_management import calibrate_price_sizing

    frame = (
        development.reset_index()
        if "symbol" not in development.columns
        else development
    )
    required = {"symbol", "entry_price", "horizon_volatility", "partition"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"development is missing columns: {sorted(missing)}")
    if not frame["partition"].eq("development").all():
        raise ValueError("CPCV calibration requires development observations only")
    if not {"split_num", "train_indices"}.issubset(splits.columns):
        raise ValueError("splits must contain split_num and train_indices columns")

    symbols = pd.Index(sorted(frame["symbol"].unique()), name="symbol")
    tables = []
    for split in splits.itertuples(index=False):
        training = frame.iloc[list(split.train_indices)]
        calibration = calibrate_price_sizing(training).set_index("symbol").reindex(symbols)
        calibration["reason"] = calibration["reason"].fillna("no split training events")
        calibration["split_num"] = split.split_num
        tables.append(calibration.reset_index())

    return pd.concat(tables, ignore_index=True)[
        ["split_num", "symbol", "w", "reason"]
    ]


def assemble_cpcv_path_predictions(
    split_predictions: pd.DataFrame,
    splits: pd.DataFrame,
    observation_index: pd.Index,
    outer_num_total_groups: int,
) -> dict[str, pd.DataFrame]:
    """Assemble one complete OOS prediction frame per CPCV path.

    Every returned path contains each observation position exactly once, using
    the split assigned to that path and observation group.
    """
    required = {"split_num", "observation_position"}
    if not required.issubset(split_predictions.columns):
        raise ValueError(
            "split_predictions must contain split_num and observation_position columns"
        )
    if len(observation_index) == 0:
        raise ValueError("observation_index must not be empty")

    assignments = get_combinatorial_backtest_paths(splits, outer_num_total_groups)
    groups = time_groups(observation_index, outer_num_total_groups)
    expected = set(range(len(observation_index)))
    paths = {}

    for path_name in assignments.columns:
        segments = []
        for group, positions in enumerate(groups):
            split_num = assignments.loc[group, path_name]
            segment = split_predictions.loc[
                split_predictions["split_num"].eq(split_num)
                & split_predictions["observation_position"].isin(positions)
            ]
            if set(segment["observation_position"]) != set(positions):
                raise ValueError("split predictions do not cover a CPCV path group")
            segments.append(segment)

        path = pd.concat(segments).sort_values("observation_position", kind="stable")
        positions = path["observation_position"]
        if positions.duplicated().any() or set(positions) != expected:
            raise ValueError("CPCV path must contain every observation exactly once")
        paths[path_name] = path

    return paths


def _validate_samples_info_sets(samples_info_sets):
    """Validate and sort event information intervals.

    Args:
        samples_info_sets: Candidate event interval series.

    Returns:
        The same series sorted by start time.

    Raises:
        ValueError: If event intervals are missing, duplicated, or inconsistent.
    """
    _validate_event_intervals(samples_info_sets)
    return samples_info_sets


def _validate_splits(splits, outer_num_total_groups):
    """Validate CPCV split metadata before constructing backtest paths.

    Args:
        splits: Candidate CPCV split metadata frame.
        outer_num_total_groups: Number of valid group identifiers, covering
            ``0`` through ``outer_num_total_groups - 1``.

    Returns:
        The original ``splits`` frame when it contains valid path metadata.

    Raises:
        ValueError: If split metadata is missing or references invalid groups.
    """
    required_columns = {"split_num", "test_groups"}

    if not isinstance(splits, pd.DataFrame):
        raise ValueError("splits must be a pd.DataFrame")

    if not required_columns.issubset(splits.columns):
        raise ValueError("splits must contain split_num and test_groups columns")

    if splits.empty:
        raise ValueError("splits must not be empty")

    test_group_sizes = splits["test_groups"].map(len)

    if test_group_sizes.nunique() != 1:
        raise ValueError("all splits must use the same number of test groups")

    for test_groups in splits["test_groups"]:
        if len(set(test_groups)) != len(test_groups):
            raise ValueError("test_groups must not contain duplicate groups")

        if min(test_groups) < 0 or max(test_groups) >= outer_num_total_groups:
            raise ValueError("test_groups contains a group outside [0, outer_num_total_groups)")

    return splits
