import numpy as np
import pandas as pd
import pytest

from src.baselines import (VALIDATION_ROLES, apply_sigmoid, baseline_feature_sets, decision_scores,
                           development_frames, fit_sigmoid_calibrator, make_baseline)
from src.evaluation import (assert_common_population, binary_metrics, choose_thresholds,
                            prediction_frame, review_budget_metrics)


@pytest.fixture
def prediction_fixture():
    return pd.DataFrame({
        "transaction_id": ["10", "20", "30", "40"], "time_step": [32] * 4,
        "split": ["validation"] * 4, "true_label": [0, 1, 1, 0],
        "model_name": ["synthetic"] * 4, "seed": [42] * 4,
        "predicted_probability": [0.1, 0.4, 0.8, 0.6],
    })


def test_metrics_against_hand_calculation():
    metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.4, 0.8, 0.6])
    for name in ["precision", "recall", "f1", "false_positive_rate"]:
        assert metrics[name] == 0.5
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == 0.75
    assert metrics["brier_score"] == pytest.approx(0.1925)
    assert [metrics[name] for name in ["tn", "fp", "fn", "tp"]] == [1, 1, 1, 1]


def test_single_class_metrics_are_explicitly_undefined():
    metrics = binary_metrics([0, 0], [0.1, 0.2])
    assert np.isnan(metrics["roc_auc"]) and np.isnan(metrics["average_precision"])
    assert np.isnan(metrics["recall"])


@pytest.mark.parametrize("labels,probabilities", [
    ([0, np.nan], [0.1, 0.2]), ([0, 2], [0.1, 0.2]),
    ([0, 1], [np.inf, 0.2]), ([0, 1], [-0.1, 0.2]),
    ([0, 1], [0.2, 1.1]), ([], []), ([0], [0.1, 0.2]),
])
def test_invalid_metric_populations_rejected(labels, probabilities):
    with pytest.raises(ValueError):
        binary_metrics(labels, probabilities)


def test_thresholds_use_exact_empirical_rules(prediction_fixture):
    thresholds = choose_thresholds(prediction_fixture, target_fpr=0)
    assert thresholds["f1_threshold"] == 0.4
    assert thresholds["fpr_threshold"] == 0.8
    achieved = binary_metrics(prediction_fixture.true_label, prediction_fixture.predicted_probability,
                             thresholds["fpr_threshold"])
    assert achieved["recall"] == 0.5 and achieved["false_positive_rate"] == 0
    tied = prediction_fixture.assign(predicted_probability=0.5)
    assert np.isinf(choose_thresholds(tied, target_fpr=0)["fpr_threshold"])


@pytest.mark.parametrize("change", ["test_split", "test_time", "selection_time", "unknown", "duplicate", "mixed_model"])
def test_threshold_selection_rejects_wrong_sources(prediction_fixture, change):
    frame = prediction_fixture.copy()
    if change == "test_split": frame["split"] = "test"
    elif change == "test_time": frame["time_step"] = 35
    elif change == "selection_time": frame["time_step"] = 30
    elif change == "unknown": frame.loc[0, "true_label"] = np.nan
    elif change == "duplicate": frame.loc[1, "transaction_id"] = frame.loc[0, "transaction_id"]
    else: frame.loc[0, "model_name"] = "another"
    with pytest.raises(ValueError):
        choose_thresholds(frame)


def test_allowed_window_cannot_relabel_test_as_validation(prediction_fixture):
    frame = prediction_fixture.assign(time_step=35)
    with pytest.raises(ValueError, match="within validation"):
        choose_thresholds(frame, allowed_steps=(35,))


def test_review_budget_exact_hits_and_tie_rule(prediction_fixture):
    result = review_budget_metrics(prediction_fixture, k=2)
    assert result["review_precision"] == 0.5
    assert result["review_recall"] == 0.5
    assert result["reviewed_count"] == 2
    tied = prediction_fixture.assign(predicted_probability=0.5)
    result = review_budget_metrics(tied.iloc[::-1], fraction=0.25)
    assert result["reviewed_count"] == 1 and result["review_hits"] == 0
    two_periods = prediction_fixture.assign(time_step=[32, 32, 33, 33])
    assert review_budget_metrics(two_periods, k=1)["reviewed_count"] == 2


def test_population_check_compares_keys_and_targets(prediction_fixture):
    rows = pd.DataFrame({"tx_id": prediction_fixture.transaction_id,
                         "time_step": prediction_fixture.time_step,
                         "target": prediction_fixture.true_label})
    second = prediction_fixture.assign(model_name="second").iloc[::-1]
    assert assert_common_population(pd.concat([prediction_fixture, second]), rows)
    second = second.copy()
    second.loc[0, "true_label"] = 1
    with pytest.raises(ValueError, match="same evaluation"):
        assert_common_population(pd.concat([prediction_fixture, second]), rows)


def test_feature_configurations_preserve_matching_columns():
    groups = baseline_feature_sets()
    assert {name: len(cols) for name, cols in groups.items()} == {
        "conservative18": 18, "local93": 93, "published165": 165, "published165_plus_time": 166}
    assert groups["published165_plus_time"][:-1] == groups["published165"]
    assert all("time_step" not in groups[name] for name in ["conservative18", "local93", "published165"])


def test_development_masks_drop_unknown_and_test(loaded_fixture):
    nodes, _, _ = loaded_fixture
    # Add enough known synthetic rows to cover each subwindow and both classes.
    template = nodes.iloc[0].copy()
    rows = []
    for time_step in [1, 30, 32, 34, 35]:
        for target in [0, 1, pd.NA]:
            row = template.copy()
            row["time_step"], row["target"] = time_step, target
            rows.append(row)
    frames = development_frames(pd.DataFrame(rows))
    assert all(frame.target.notna().all() and frame.time_step.max() <= 34 for frame in frames.values())
    for role, times in VALIDATION_ROLES.items():
        assert frames[role].time_step.isin(times).all()


@pytest.mark.parametrize("family,parameters", [("logistic_regression", {"C": 0.1}),
                                               ("lightgbm", {"n_estimators": 10, "num_leaves": 3})])
def test_fixed_seed_reproducibility_and_training_only_scaler(family, parameters):
    rng = np.random.default_rng(8)
    x_train = pd.DataFrame(rng.normal(size=(320, 3)), columns=["a", "b", "c"])
    labels = (x_train.a + 0.3 * x_train.b > 0).to_numpy(dtype=int)
    x_validation = pd.DataFrame(np.full((10, 3), 1e4), columns=x_train.columns)
    models = [make_baseline(family, parameters, "ordinary", 42, labels, n_jobs=1) for _ in range(2)]
    for model in models:
        model.fit(x_train, labels)
    assert models[0].predict_proba(x_train)[:, 1].std() > 0.01
    np.testing.assert_array_equal(models[0].predict_proba(x_train), models[1].predict_proba(x_train))
    np.testing.assert_array_equal(models[0].predict_proba(x_validation), models[1].predict_proba(x_validation))
    if family == "logistic_regression":
        scaler = models[0].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, x_train.mean())
        assert scaler.n_samples_seen_ == len(x_train)
        before = scaler.mean_.copy()
        models[0].predict_proba(x_validation * -1)
        np.testing.assert_array_equal(scaler.mean_, before)


def test_class_weights_are_training_derived():
    model = make_baseline("lightgbm", {"n_estimators": 10, "num_leaves": 3},
                          "balanced", 42, [0, 0, 0, 1], n_jobs=1)
    assert model.get_params()["scale_pos_weight"] == 3.0
    with pytest.raises(ValueError, match="known binary"):
        make_baseline("lightgbm", {}, "balanced", 42, [0, np.nan])


def test_calibration_uses_separate_validation_window(prediction_fixture):
    scores = np.array([-2.0, -0.4, 1.4, 0.4])
    calibrator = fit_sigmoid_calibrator(prediction_fixture, scores, min_per_class=2)
    probabilities = apply_sigmoid(calibrator, scores)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    with pytest.raises(ValueError, match="declared validation"):
        fit_sigmoid_calibrator(prediction_fixture.assign(time_step=34), scores, min_per_class=2)
    with pytest.raises(ValueError, match="too few"):
        fit_sigmoid_calibrator(prediction_fixture, scores)


def test_native_scores_preserve_extreme_order_without_logit_clipping():
    model = make_baseline("logistic_regression", {"C": 1.0}, "ordinary", 42, [0, 0, 1, 1])
    features = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0]})
    model.fit(features, [0, 0, 1, 1])
    extreme = pd.DataFrame({"x": [100.0, 200.0]})
    probabilities = model.predict_proba(extreme)[:, 1]
    assert probabilities[0] == probabilities[1] == 1.0
    scores = decision_scores(model, extreme)
    assert np.isfinite(scores).all() and scores[1] > scores[0]


def test_prediction_format_excludes_unknown_and_test(prediction_fixture):
    rows = pd.DataFrame({"tx_id": prediction_fixture.transaction_id,
                         "time_step": prediction_fixture.time_step,
                         "target": prediction_fixture.true_label})
    frame = prediction_frame(rows, prediction_fixture.predicted_probability, "model", 42)
    assert list(frame.columns) == ["transaction_id", "time_step", "split", "true_label", "model_name", "seed", "predicted_probability"]
    with pytest.raises(ValueError):
        prediction_frame(rows.assign(target=np.nan), [0.1] * 4, "model", 42)
    with pytest.raises(ValueError):
        prediction_frame(rows.assign(time_step=35), [0.1] * 4, "model", 42)
