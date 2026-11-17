"""Validate the notebook's actual local artifacts, not only helper fixtures."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.baselines import apply_sigmoid, decision_scores, development_frames
from src.data import load_elliptic
from src.evaluation import assert_common_population, choose_thresholds, evaluate_predictions


@pytest.mark.integration
def test_executed_baseline_artifacts_and_validation_population(require_private_files):
    result_dir = Path("results/stage2/full_run")
    config_path = result_dir / "selected_configurations.json"
    if not config_path.is_file():
        pytest.skip("Execute the Stage 2 FULL_RUN notebook to test real fitted artifacts")
    config = json.loads(config_path.read_text())
    raw_dir = Path(os.environ.get("ELLIPTIC_DATA_DIR", "data/raw"))
    required = [raw_dir / name for name in ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]]
    required += [Path("results/local/stage2/full_run") / name for name in ["validation_raw_predictions.csv", "validation_calibrated_step34_predictions.csv"]]
    required += [Path("models/stage2/full_run") / f"{run['model_id']}.joblib" for run in config["selected_models"]]
    require_private_files(required)
    nodes, _, _ = load_elliptic(os.environ.get("ELLIPTIC_DATA_DIR", "data/raw"))
    frames = development_frames(nodes)
    audit, train, calibration = frames["audit"], frames["train"], frames["calibration"]
    raw = pd.read_csv("results/local/stage2/full_run/validation_raw_predictions.csv", dtype={"transaction_id": str}, float_precision="round_trip")
    calibrated = pd.read_csv("results/local/stage2/full_run/validation_calibrated_step34_predictions.csv", dtype={"transaction_id": str}, float_precision="round_trip")
    expected_validation = pd.concat([frames[name] for name in ["selection", "calibration", "audit"]])
    assert_common_population(raw, expected_validation)
    assert_common_population(pd.concat([raw.loc[raw.time_step == 34], calibrated]), audit)
    assert set(raw.time_step.unique()) == {30, 31, 32, 33, 34}
    assert set(calibrated.time_step.unique()) == {34}
    assert not config["final_test_predictions_computed"] and not config["final_test_rankings_computed"]
    metrics = pd.read_csv(result_dir / "validation_metrics.csv")
    for specification in config["selected_models"]:
        model_id = specification["model_id"]
        bundle = joblib.load(Path("models/stage2/full_run") / f"{model_id}.joblib")
        columns = bundle["feature_columns"]
        assert columns == specification["feature_columns"]
        assert bundle["training_steps"] == list(range(1, 30))
        assert bundle["selection_steps"] == [30, 31] and bundle["calibration_steps"] == [32, 33]
        model = bundle["base_model"]
        if bundle["family"] == "logistic_regression":
            scaler = model.named_steps["scale"]
            assert scaler.n_samples_seen_ == len(train)
            np.testing.assert_allclose(scaler.mean_, train[columns].mean(), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(scaler.var_, train[columns].var(ddof=0), rtol=1e-10, atol=1e-12)
            assert list(scaler.feature_names_in_) == columns
        else:
            assert model.feature_name_ == columns
            expected_weight = (train.target == 0).sum() / train.target.sum() if bundle["weighting"] == "balanced" else 1.0
            assert model.get_params()["scale_pos_weight"] == expected_weight
        expected_raw = model.predict_proba(audit[columns])[:, 1]
        expected_calibrated = apply_sigmoid(bundle["calibrator"], decision_scores(model, audit[columns]))
        for suffix, source, probabilities in [("raw", raw, expected_raw), ("platt", calibrated, expected_calibrated)]:
            saved = source.loc[source.model_name == model_id + "__" + suffix].set_index("transaction_id").loc[audit.tx_id]
            np.testing.assert_allclose(saved.predicted_probability, probabilities, rtol=1e-12, atol=1e-14)
            thresholds = bundle["raw_thresholds" if suffix == "raw" else "calibrated_thresholds"]
            observed = evaluate_predictions(saved.reset_index(), thresholds, review_k=config["review_k_per_snapshot"])
            reported = metrics.loc[metrics.model_name == model_id + "__" + suffix].iloc[0]
            for name in ["average_precision", "roc_auc", "f1", "brier_score", "precision_at_k",
                         "recall_at_validation_target_fpr", "achieved_fpr_at_validation_target"]:
                assert reported[name] == pytest.approx(observed[name], abs=1e-12)
        calibration_predictions = raw.loc[(raw.model_name == model_id + "__raw") & raw.time_step.isin([32, 33])]
        assert choose_thresholds(calibration_predictions, config["target_fpr"]) == bundle["raw_thresholds"]
