"""Replay aggregate Stage 4 evidence from private prediction caches, never refit."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.evaluation import binary_metrics, evaluate_predictions
from src.robustness import aligned_pair, bootstrap_ap, expected_calibration_error, verify_frozen


@pytest.mark.integration
def test_stage4_frozen_artifacts_and_all_saved_evaluation_tables(require_private_files):
    folder = Path("results/stage4/full_run")
    if not (folder / "execution_summary.json").exists():
        pytest.skip("Execute Stage 4 FULL_RUN before checking its local prediction artifacts")
    protocol_path = Path("results/stage4/frozen_protocol.json")
    execution = json.loads((folder / "execution_summary.json").read_text())
    declared = json.loads(protocol_path.read_text())
    private = Path("results/local/stage4/full_run")
    required = [Path(name) for name in declared["artifact_sha256"] if name.startswith("models/")]
    required += [private / name for name in ["test_raw_predictions.csv", "test_platt_predictions.csv", "bootstrap_snapshot_counts.npz", "bootstrap_model_ap.npz"]]
    required += [private / name for name in pd.read_csv(folder / "robustness_results.csv").probability_cache.unique()]
    require_private_files(required)
    protocol = verify_frozen(Path.cwd(), protocol_path, execution["protocol_sha256"])
    receipt = json.loads(Path("results/stage4/freeze_receipt.json").read_text())
    first_access = json.loads((folder / "first_test_inference.json").read_text())
    assert not receipt["test_predictions_existed"]
    assert pd.Timestamp(receipt["frozen_at_utc"]) < pd.Timestamp(first_access["started_at_utc"])
    assert execution["new_model_fits"] == execution["new_calibrator_fits"] == execution["test_threshold_selections"] == 0
    assert execution["reused_models"] == 33 and not execution["stage5_started"]
    assert protocol["selection_steps"] == [30, 31] and protocol["calibration_steps"] == [32, 33]
    assert len(protocol["models"]) == 33
    public_metrics = pd.read_csv(folder / "final_test_metrics.csv")
    private = Path("results/local/stage4/full_run")
    expected = None
    frames = {}
    for variant in ["raw", "platt"]:
        predictions = pd.read_csv(private / f"test_{variant}_predictions.csv", dtype={"transaction_id": str}, float_precision="round_trip")
        for record in protocol["models"]:
            name = record["model_id"]
            frame = predictions.loc[predictions.model_name == name + "__" + variant]
            assert set(frame.time_step) == set(range(35, 50)) and frame["split"].eq("test").all()
            assert frame.true_label.isin([0, 1]).all()
            expected = frame if expected is None else expected
            aligned_pair(frame, expected)
            bundle = joblib.load(record.get("calibration_artifact", record["artifact"]))
            key = "raw_thresholds" if variant == "raw" else "calibrated_thresholds"
            assert bundle[key]["source_steps"] == [32, 33]
            observed = evaluate_predictions(frame, bundle[key], protocol["review_k_per_snapshot"])
            observed["expected_calibration_error"] = expected_calibration_error(frame.true_label, frame.predicted_probability)
            saved = public_metrics.loc[(public_metrics.model_id == name) & (public_metrics.calibration == variant)].iloc[0]
            for metric, value in observed.items():
                assert saved[metric] == pytest.approx(value, abs=1e-12)
            if variant == "raw":
                frames[name] = frame
    assert len(public_metrics) == 66
    robustness = pd.read_csv(folder / "robustness_results.csv")
    assert len(robustness) == 720
    canonical = frames[protocol["models"][0]["model_id"]]
    raw_metric = public_metrics.loc[public_metrics.calibration == "raw"].set_index("model_id")
    for path, rows in robustness.groupby("probability_cache"):
        saved_probability = np.load(private / path)
        for row in rows.itertuples():
            original = frames[row.model_id]
            np.testing.assert_array_equal(original.transaction_id, canonical.transaction_id)
            rules = joblib.load(next(r.get("calibration_artifact", r["artifact"]) for r in protocol["models"] if r["model_id"] == row.model_id))["raw_thresholds"]
            observed = binary_metrics(original.true_label, saved_probability[row.model_id], rules["f1_threshold"])
            for metric in ["average_precision", "precision", "recall"]:
                assert getattr(row, metric) == pytest.approx(observed[metric], abs=1e-12)
            assert row.delta_ap == pytest.approx(row.average_precision - raw_metric.loc[row.model_id, "average_precision"], abs=1e-12)
    per_step = pd.read_csv(folder / "per_time_step_metrics.csv")
    assert len(per_step) == 66 * 15
    enough = per_step.ranking_metric_available
    assert per_step.loc[~enough, "average_precision"].isna().all()
    assert per_step.loc[enough, "average_precision"].between(0, 1).all()
    counts = np.load(private / "bootstrap_snapshot_counts.npz")["counts"]
    assert counts.shape == (1000, 15) and np.all(counts.sum(axis=1) == 15)
    bootstrap = np.load(private / "bootstrap_model_ap.npz")
    for record in protocol["models"][:2]:
        name = record["model_id"]
        frame, _ = aligned_pair(frames[name], canonical)
        replay = bootstrap_ap(frame.true_label.to_numpy(), frame.predicted_probability.to_numpy(), frame.time_step.to_numpy() - 35, counts[:10])
        np.testing.assert_allclose(replay, bootstrap[name][:10], rtol=0, atol=1e-12)
    intervals = pd.read_csv(folder / "paired_bootstrap_intervals.csv")
    assert len(intervals) == 12 and (intervals.valid_replications == 1000).all()
    replicates = pd.read_csv(folder / "bootstrap_difference_replicates.csv")
    for row in intervals.itertuples():
        draw = replicates.loc[(replicates.feature_configuration == row.feature_configuration)
                              & (replicates.first_architecture == row.first_architecture)
                              & (replicates.second_architecture == row.second_architecture), "ap_difference"]
        np.testing.assert_allclose(np.percentile(draw, [2.5, 97.5]), [row.ci_low, row.ci_high], rtol=0, atol=1e-12)
        first = raw_metric.loc[raw_metric.architecture.eq(row.first_architecture) & raw_metric.feature_configuration.eq(row.feature_configuration), "average_precision"].mean()
        second = raw_metric.loc[raw_metric.architecture.eq(row.second_architecture) & raw_metric.feature_configuration.eq(row.feature_configuration), "average_precision"].mean()
        assert row.observed_ap_difference == pytest.approx(first - second, abs=1e-12)
