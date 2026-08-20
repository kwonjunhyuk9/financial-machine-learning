import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.strategy_modeling.cross_validation import PurgedKFold
from src.strategy_modeling.model_workflow import (
    build_candidate_classifiers,
    build_meta_training_frame,
    candidate_parameter_grids,
    chronological_holdout_split,
    compute_strategy_returns,
    generate_oof_predictions,
    get_primary_feature_columns,
    probability_bet_size,
)


def _events(num_rows: int = 10) -> pd.DataFrame:
    starts = pd.date_range("2025-01-02", periods=num_rows, freq="D", tz="UTC")
    ends = starts + pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "event_start": starts,
            "symbol": "AAPL",
            "event_end": ends,
            "vertical_barrier": ends + pd.Timedelta(hours=1),
            "target_return": 0.01,
            "raw_return": np.where(np.arange(num_rows) % 2 == 0, 0.02, -0.01),
            "direction_label": np.where(np.arange(num_rows) % 2 == 0, 1, -1),
            "mean_sentiment_score": np.linspace(-1.0, 1.0, num_rows),
            "fractionally_differenced_log_close": np.linspace(0.0, 0.5, num_rows),
            "Relative Strength Index": np.linspace(40.0, 60.0, num_rows),
        }
    )


def test_primary_feature_columns_exclude_outcomes_and_identifiers():
    columns = get_primary_feature_columns(_events())

    assert columns == [
        "mean_sentiment_score",
        "fractionally_differenced_log_close",
        "Relative Strength Index",
    ]


def test_candidate_classifiers_use_required_model_families():
    candidates = build_candidate_classifiers(random_state=42, n_jobs=1)

    assert set(candidates) == {
        "bagging",
        "random_forest",
        "boosting",
    }
    assert all(
        candidate.steps == [("model", candidate["model"])]
        for candidate in candidates.values()
    )


def test_candidate_parameter_grids_cover_only_tree_families():
    assert set(candidate_parameter_grids()) == {
        "bagging",
        "random_forest",
        "boosting",
    }


def test_chronological_holdout_split_purges_development_overlap():
    events = _events()
    holdout_boundary = events.loc[8, "event_start"]
    events.loc[7, "event_end"] = holdout_boundary + pd.Timedelta(hours=1)

    development, holdout, manifest = chronological_holdout_split(events)

    assert len(holdout) == 2
    assert development["event_end"].lt(holdout_boundary).all()
    assert manifest.loc[7, "partition"] == "holdout_overlap_purged"
    assert manifest.loc[8:, "partition"].eq("holdout").all()


def test_generate_oof_predictions_marks_every_prediction_as_oof():
    events = _events(20)
    features = events.set_index("event_start")[["mean_sentiment_score"]]
    labels = pd.Series(
        np.tile([-1, 1], 10),
        index=features.index,
        name="direction_label",
    )
    weights = pd.Series(1.0, index=features.index)
    information_sets = events.set_index("event_start")["event_end"]
    cv = PurgedKFold(n_splits=5, t1=information_sets, pct_embargo=0.01)

    predictions = generate_oof_predictions(
        estimator=DecisionTreeClassifier(random_state=42),
        features=features,
        labels=labels,
        sample_weight=weights,
        cv=cv,
        positive_label=1,
    )

    assert predictions.index.equals(features.index)
    assert predictions["prediction_source"].eq("oof").all()
    assert predictions["fold"].nunique() == 5
    assert predictions["probability"].between(0.0, 1.0).all()


def test_meta_training_frame_requires_primary_oof_predictions():
    events = _events().set_index("event_start")
    primary_oof = pd.DataFrame(
        {
            "prediction": events["direction_label"],
            "probability": 0.75,
            "prediction_source": "oof",
        },
        index=events.index,
    )

    meta = build_meta_training_frame(events, primary_oof)

    expected = (meta["primary_side"] * meta["raw_return"] > 0).astype("int8")
    pd.testing.assert_series_equal(meta["meta_label"], expected, check_names=False)

    primary_oof.loc[primary_oof.index[0], "prediction_source"] = "in_sample"
    with pytest.raises(ValueError, match="OOF"):
        build_meta_training_frame(events, primary_oof)


def test_probability_bet_size_is_bounded_and_respects_pass_decision():
    probability = pd.Series([0.50, 0.60, 0.90])
    action = pd.Series([1, 0, 1])

    sizes = probability_bet_size(probability, action, step_size=0.1)

    assert sizes.between(0.0, 1.0).all()
    assert sizes.iloc[0] == 0.0
    assert sizes.iloc[1] == 0.0
    assert sizes.iloc[2] > 0.0


def test_strategy_returns_default_to_zero_execution_costs():
    predictions = pd.DataFrame(
        {
            "raw_return": [0.02, -0.01],
            "primary_side": [1, -1],
            "meta_action": [1, 0],
            "bet_size": [0.5, 0.8],
        }
    )

    returns = compute_strategy_returns(predictions)

    assert returns.loc[0, "primary_only_gross_return"] == pytest.approx(0.02)
    assert returns.loc[0, "primary_only_total_cost"] == 0.0
    assert returns.loc[0, "primary_only_net_return"] == pytest.approx(0.02)
    assert returns.loc[0, "meta_filtered_gross_return"] == pytest.approx(0.01)
    assert returns.loc[0, "meta_filtered_total_cost"] == 0.0
    assert returns.loc[0, "meta_filtered_net_return"] == pytest.approx(0.01)
    assert returns.loc[1, "meta_filtered_net_return"] == 0.0
