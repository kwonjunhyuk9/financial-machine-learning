import numpy as np
import pandas as pd
import pytest

from src.backtesting.strategy_validation import (
    assemble_cpcv_path_predictions,
    combinatorial_purged_cross_validation,
    get_combinatorial_backtest_paths,
    get_cpcv_price_calibrations,
)
from src.modeling.purged_validation import PurgedKFold, index_events


def test_cpcv_returns_one_split_per_test_group_combination():
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    samples_info_sets = pd.Series(index + pd.Timedelta(days=1), index=index)

    splits = combinatorial_purged_cross_validation(samples_info_sets, outer_num_total_groups=3, outer_num_test_groups=1)

    assert len(splits) == 3
    assert set(splits.columns) >= {"train_indices", "test_indices"}

    starts = pd.date_range("2025-01-02", periods=12, freq="h", tz="UTC")
    events = index_events(pd.DataFrame([
        {"symbol": symbol, "event_start": time,
         "event_end": time + pd.Timedelta(minutes=70)}
        for time in starts for symbol in ["A", "B"]
    ]))
    equal_time_splits = combinatorial_purged_cross_validation(events.event_end, 3, 1, .05)
    folds = list(PurgedKFold(3, events.event_end, .05).split(events))
    for row, (train, test) in zip(equal_time_splits.itertuples(), folds):
        assert row.train_indices == tuple(train)
        assert row.test_indices == tuple(test)


def test_cpcv_rejects_invalid_group_count():
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")

    with pytest.raises(ValueError, match="greater than 1"):
        combinatorial_purged_cross_validation(pd.Series(index, index=index), 1, 1)


def test_cpcv_paths_select_one_prediction_per_observation():
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    splits = combinatorial_purged_cross_validation(
        pd.Series(index, index=index), outer_num_total_groups=3, outer_num_test_groups=2
    )
    predictions = pd.DataFrame([
        {
            "split_num": split.split_num,
            "observation_position": position,
            "value": split.split_num * 10 + position,
        }
        for split in splits.itertuples()
        for position in split.test_indices
    ])

    paths = assemble_cpcv_path_predictions(predictions, splits, index, outer_num_total_groups=3)
    assignments = get_combinatorial_backtest_paths(splits, outer_num_total_groups=3)
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

    assert set(paths) == {"path_0", "path_1"}
    for path_name, path in paths.items():
        assert path["observation_position"].tolist() == list(range(6))
        for group, positions in enumerate(groups):
            expected_split = assignments.loc[group, path_name]
            selected = path.loc[path["observation_position"].isin(positions)]
            assert selected["split_num"].eq(expected_split).all()


def test_cpcv_price_calibration_uses_training_observations_only():
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    development = pd.DataFrame({
        "entry_price": 100.0,
        "horizon_volatility": [0.01, 0.02, 0.03, 0.04, 0.50, 0.60],
        "partition": "development",
    }, index=pd.MultiIndex.from_arrays(
        [["A"] * 6, index], names=["symbol", "event_start"]
    ))
    splits = combinatorial_purged_cross_validation(
        pd.Series(index, index=index), outer_num_total_groups=3, outer_num_test_groups=1
    )
    split = splits.loc[splits["test_groups"].map(lambda groups: groups == (2,))].iloc[0]

    original = get_cpcv_price_calibrations(development, splits)
    changed = development.copy()
    changed.iloc[list(split.test_indices), changed.columns.get_loc("horizon_volatility")] = 99.0
    recalibrated = get_cpcv_price_calibrations(changed, splits)

    original_w = original.loc[original["split_num"].eq(split.split_num), "w"].item()
    changed_w = recalibrated.loc[
        recalibrated["split_num"].eq(split.split_num), "w"
    ].item()
    assert changed_w == original_w


@pytest.mark.parametrize("pct", [0, 0.01, 0.11])
def test_cpcv_matches_purged_kfold_for_identical_test_groups(pct):
    index = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    events = pd.Series(index, index=index)
    events.iloc[0] = index[4]
    features = pd.DataFrame({"feature": range(12)}, index=index)
    splits = combinatorial_purged_cross_validation(events, 3, 1, pct)

    for (_, split), (train, test) in zip(
        splits.iterrows(), PurgedKFold(3, events, pct).split(features)
    ):
        np.testing.assert_array_equal(split["test_indices"], test)
        np.testing.assert_array_equal(split["train_indices"], train)


@pytest.mark.parametrize(
    "groups,expected", [((0, 1), (9, 10, 11)), ((0, 2), (10, 11))]
)
def test_cpcv_adjacent_and_separated_groups_use_latest_event_end(groups, expected):
    starts = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    events = pd.Series(starts, index=starts)
    events.iloc[0] = starts[7]
    events.iloc[4] = starts[8]
    splits = combinatorial_purged_cross_validation(
        events, outer_num_total_groups=6, outer_num_test_groups=2,
        outer_pct_embargo=0.01,
    )
    assert len(splits) == 15
    assert get_combinatorial_backtest_paths(splits, outer_num_total_groups=6).shape == (6, 5)
    split = next(row for _, row in splits.iterrows() if row["test_groups"] == groups)

    assert split["train_indices"] == expected


@pytest.mark.parametrize("pct", [-0.1, 1, 1.1, np.nan, np.inf, -np.inf])
def test_splitters_reject_invalid_embargo(pct):
    starts = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    events = pd.Series(starts, index=starts)
    features = pd.DataFrame({"feature": range(6)}, index=starts)

    with pytest.raises(ValueError, match="pct_embargo"):
        combinatorial_purged_cross_validation(events, 3, 1, pct)
    with pytest.raises(ValueError, match="pct_embargo"):
        list(PurgedKFold(3, events, pct).split(features))


@pytest.fixture
def cpcv_prediction_inputs():
    from sklearn.tree import DecisionTreeClassifier

    starts = pd.date_range("2025-01-02", periods=32, freq="D", tz="UTC")
    development = index_events(pd.DataFrame([
        {"symbol": symbol, "event_start": time,
         "event_end": time + pd.Timedelta(hours=1),
         "vertical_barrier": time + pd.Timedelta(hours=2),
         "horizon_volatility": 0.01, "entry_price": 100.0,
         "raw_return": 0.02 if i % 3 else -0.01,
         "primary_label": 1 if i % 3 else -1,
         "sample_weight": 0.5 + i / 32, "partition": "development",
         "holdout_boundary": starts[-1] + pd.Timedelta(days=1),
         "mean_sentiment_score": np.sin(i), "feature": float(i)}
        for i, time in enumerate(starts) for symbol in ("A", "B")
    ]))
    splits = combinatorial_purged_cross_validation(development.event_end, 4, 1, 0.01)
    estimator = DecisionTreeClassifier(max_depth=2, random_state=42)
    primary = {"estimator": estimator, "feature_columns": ["feature"]}
    meta = {"estimator": estimator, "feature_columns": ["feature", "primary_prediction", "primary_class_probability"]}
    return development, splits, primary, meta


@pytest.mark.parametrize("model_kind", ["market", "sentiment"])
def test_cpcv_families_share_split_training_and_inner_oof(
    tmp_path, monkeypatch, cpcv_prediction_inputs, model_kind,
):
    from sklearn.tree import DecisionTreeClassifier
    import src.backtesting.strategy_validation as validation

    development, splits, primary, meta = cpcv_prediction_inputs
    if model_kind == "sentiment":
        primary["feature_columns"] = ["feature", "mean_sentiment_score"]
        meta["feature_columns"] = [*primary["feature_columns"], "primary_prediction", "primary_class_probability"]
    original = development.copy(deep=True)
    fits, inner_predictions = [], []
    original_fit = DecisionTreeClassifier.fit
    original_oof = validation.generate_oof_predictions

    def record_fit(self, X, y, sample_weight=None, **kwargs):
        fits.append((X.copy(), y.copy(), sample_weight.copy()))
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    def record_oof(estimator, features, labels, weights, cv, positive_label):
        assert cv.n_splits == 5
        assert cv.pct_embargo == 0.01
        output = original_oof(estimator, features, labels, weights, cv, positive_label)
        inner_predictions.append(output)
        return output

    monkeypatch.setattr(DecisionTreeClassifier, "fit", record_fit)
    monkeypatch.setattr(validation, "generate_oof_predictions", record_oof)
    result = validation.generate_cpcv_predictions(
        development, splits, primary,
        {name: meta for name in ("boosting", "bagging", "random_forest")},
        inner_num_splits=5, inner_pct_embargo=0.01,
    )
    assert not list(tmp_path.iterdir())
    assert set(result.model_family) == {"boosting", "bagging", "random_forest"}
    assert result.index.names == ["symbol", "event_start"]
    assert len(fits) == len(splits) * 9
    assert len(inner_predictions) == len(splits)
    for i, split in enumerate(splits.itertuples()):
        train = development.iloc[list(split.train_indices)]
        test = development.iloc[list(split.test_indices)]
        inner_oof = inner_predictions[i]
        assert inner_oof.index.equals(train.index)
        assert inner_oof.prediction_source.eq("oof").all()
        inner_cv = PurgedKFold(5, train.event_end, 0.01)
        for fold, (inner_train, _) in enumerate(inner_cv.split(train)):
            X, y, weights = fits[i * 9 + fold]
            assert X.index.equals(train.iloc[inner_train].index)
            np.testing.assert_array_equal(weights, train.sample_weight.iloc[inner_train])
        for X, _, weights in fits[i * 9:i * 9 + 9]:
            assert set(X.index).issubset(train.index)
            assert not set(X.index).intersection(test.index)
            np.testing.assert_array_equal(weights, train.loc[X.index, "sample_weight"])
        primary_X, primary_y, _ = fits[i * 9 + 5]
        pd.testing.assert_frame_equal(primary_X, train[primary["feature_columns"]])
        pd.testing.assert_series_equal(primary_y, train.primary_label.astype("int8"))
        for meta_X, meta_y, _ in fits[i * 9 + 6:i * 9 + 9]:
            np.testing.assert_array_equal(meta_X.primary_prediction, inner_oof.prediction)
            np.testing.assert_array_equal(meta_y, (inner_oof.prediction * train.raw_return > 0).astype("int8"))
        predicted = result.loc[result.split_num.eq(split.split_num) & result.model_family.eq("boosting")]
        assert predicted.index.equals(test.index)
        assert predicted.observation_position.tolist() == list(split.test_indices)
    shared = result.reset_index().groupby(["split_num", "symbol", "event_start"])
    assert shared.primary_positive_probability.nunique().eq(1).all()
    assert shared.primary_prediction.nunique().eq(1).all()
    assert result.partition.eq("development").all()
    assert not hasattr(primary["estimator"], "tree_")
    pd.testing.assert_frame_equal(development, original)


def test_cpcv_predictions_reject_holdout_and_ignore_test_outcomes(
    tmp_path, cpcv_prediction_inputs,
):
    from src.backtesting.strategy_validation import generate_cpcv_predictions

    development, splits, primary, meta = cpcv_prediction_inputs
    kwargs = dict(inner_num_splits=3, inner_pct_embargo=0.01)
    with pytest.raises(ValueError, match="development observations only"):
        generate_cpcv_predictions(
            development.assign(partition="holdout"), splits, primary, {"boosting": meta}, **kwargs,
        )
    assert not list(tmp_path.iterdir())
    split = splits.iloc[[0]]
    baseline = generate_cpcv_predictions(development, split, primary, {"boosting": meta}, **kwargs)
    changed = development.copy()
    test = list(split.iloc[0].test_indices)
    changed.iloc[test, changed.columns.get_loc("raw_return")] = -99.0
    changed.iloc[test, changed.columns.get_loc("primary_label")] = -changed.primary_label.iloc[test]
    changed.iloc[test, changed.columns.get_loc("sample_weight")] = 999.0
    updated = generate_cpcv_predictions(changed, split, primary, {"boosting": meta}, **kwargs)
    columns = ["primary_prediction", "primary_positive_probability", "meta_prediction", "meta_pass_probability"]
    pd.testing.assert_frame_equal(baseline[columns], updated[columns])


def test_cpcv_refits_balanced_boosting_using_only_split_training_weights(monkeypatch, cpcv_prediction_inputs):
    from sklearn.base import clone
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from src.modeling.ensemble_methods import build_boosting_classifier
    from src.backtesting.strategy_validation import generate_cpcv_predictions

    development, splits, primary, meta = cpcv_prediction_inputs
    primary["estimator"] = build_boosting_classifier(
        class_weight="balanced", n_estimators=2, max_depth=2, random_state=42,
    )
    meta["estimator"] = clone(primary["estimator"])
    original_fit = AdaBoostClassifier.fit
    fits = []

    def record_fit(self, X, y, sample_weight=None):
        assert self.class_weight == "balanced"
        np.testing.assert_allclose(
            sample_weight,
            development.loc[X.index, "sample_weight"].to_numpy() * compute_sample_weight("balanced", y),
        )
        fits.append(X.index.copy())
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(AdaBoostClassifier, "fit", record_fit)
    split = splits.iloc[[0]]
    result = generate_cpcv_predictions(
        development, split, primary, {"boosting": meta},
        inner_num_splits=3, inner_pct_embargo=.01,
    )
    assert len(fits) == 5  # Three inner OOF fits, one Primary fit, one Meta fit.
    train = development.iloc[list(split.iloc[0].train_indices)].index
    test = development.iloc[list(split.iloc[0].test_indices)].index
    assert all(fit.isin(train).all() and fit.intersection(test).empty for fit in fits)
    assert set(result.model_family) == {"boosting"}
