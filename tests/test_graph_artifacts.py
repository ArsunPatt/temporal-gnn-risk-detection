"""Integration checks against the graph notebook's actual saved development fits."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from src.baselines import apply_sigmoid, development_frames
from src.data import load_elliptic
from src.evaluation import assert_common_population, evaluate_predictions
from src.graph_learning import GraphRiskModel, build_snapshots, infer_snapshots


@pytest.mark.integration
def test_graph_notebook_artifacts_and_chronological_assessment(require_private_files):
    result_dir = Path("results/stage3/full_run")
    config_path = result_dir / "selected_configurations.json"
    if not config_path.is_file():
        pytest.skip("Execute Stage 3 FULL_RUN to validate fitted graph artifacts")
    config = json.loads(config_path.read_text())
    raw_dir = Path(os.environ.get("ELLIPTIC_DATA_DIR", "data/raw"))
    required = [raw_dir / name for name in ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]]
    required += [Path("results/local/stage3/full_run") / name for name in ["validation_raw_predictions.csv", "validation_calibrated_step34_predictions.csv"]]
    required += [Path("models/stage3/full_run") / f"scaler__{group}.joblib" for group in config["feature_sets"]]
    for run in config["selected_runs"]:
        required += [Path("models/stage3/full_run") / (run["run_id"] + suffix) for suffix in [".pt", "__calibration.joblib"]]
    require_private_files(required)
    assert len(config["seeds"]) >= 3 and config["primary_seed"] in config["seeds"]
    assert config["maximum_constructed_snapshot"] == 34
    assert not config["final_test_predictions_computed"] and not config["final_test_rankings_computed"]
    selected = pd.DataFrame(config["selected_runs"])
    main = selected.loc[selected.experiment == "main"]
    assert len(main) == 3 * len(config["feature_sets"]) * len(config["seeds"])
    expected_controls = 2 * len(config["feature_sets"]) + len(set(config["feature_sets"]) & {"conservative18", "published165"})
    assert len(selected.loc[selected.experiment == "control"]) == expected_controls
    for _, frame in main.groupby(["architecture", "feature_configuration"]):
        assert sorted(frame.seed.tolist()) == sorted(config["seeds"])
        assert frame.initial_parameter_sha256.nunique() == len(config["seeds"])
        assert frame.learning_rate.nunique() == 1
    nodes, edges, _ = load_elliptic(os.environ.get("ELLIPTIC_DATA_DIR", "data/raw"))
    frames = development_frames(nodes)
    audit = frames["audit"]
    validation = pd.concat([frames[role] for role in ["selection", "calibration", "audit"]])
    raw = pd.read_csv("results/local/stage3/full_run/validation_raw_predictions.csv",
                      dtype={"transaction_id": str}, float_precision="round_trip")
    calibrated = pd.read_csv("results/local/stage3/full_run/validation_calibrated_step34_predictions.csv",
                             dtype={"transaction_id": str}, float_precision="round_trip")
    assert_common_population(raw, validation)
    assert_common_population(pd.concat([raw.loc[raw.time_step == 34], calibrated]), audit)
    assert raw.time_step.max() == 34 and set(calibrated.time_step) == {34}
    metrics = pd.read_csv(result_dir / "validation_metrics.csv")
    graph_cache = {}
    torch.set_num_threads(1)
    for group, columns in config["feature_sets"].items():
        scaler = joblib.load(Path("models/stage3/full_run") / f"scaler__{group}.joblib")
        assert list(scaler.feature_names_in_) == columns and scaler.n_samples_seen_ == len(frames["train"])
        np.testing.assert_allclose(scaler.mean_, frames["train"][columns].mean(), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(scaler.var_, frames["train"][columns].var(ddof=0), rtol=1e-10, atol=1e-12)
        for mode in selected.loc[selected.feature_configuration == group, "edge_mode"].unique():
            graph_cache[(group, mode)] = build_snapshots(
                nodes, edges, scaler, config["group_helper_keys"][group], edge_mode=mode,
                max_step=34, rewire_seed=config["rewire_seed"])
    for run in config["selected_runs"]:
        folder = Path("models/stage3/full_run")
        saved = torch.load(folder / f"{run['run_id']}.pt", map_location="cpu", weights_only=True)
        assert saved["train_steps"] == list(range(1, 30)) and saved["selection_steps"] == [30, 31]
        assert saved["feature_columns"] == config["feature_sets"][run["feature_configuration"]]
        assert saved["best_epoch"] == run["best_epoch"] <= config["max_epochs"]
        model = GraphRiskModel(**saved["model_config"])
        model.load_state_dict(saved["state_dict"])
        graphs = graph_cache[(run["feature_configuration"], run["edge_mode"])]
        assert all("y" not in graph and "target" not in graph for graph in graphs.values())
        outputs = infer_snapshots(model, graphs, [34])[34]["logits"]
        all_rows = nodes.iloc[graphs[34].node_index.numpy()]
        known = all_rows.target.notna().to_numpy()
        scores = outputs[known]
        model_id = f"{run['architecture']}__{run['feature_configuration']}__{run['edge_mode']}__seed{run['seed']}"
        rules = joblib.load(folder / f"{run['run_id']}__calibration.joblib")
        for variant, source, expected_probabilities in [
            ("raw", raw, expit(scores)),
            ("platt", calibrated, apply_sigmoid(rules["calibrator"], scores)),
        ]:
            prediction = source.loc[(source.model_name == model_id + "__" + variant) & source.time_step.eq(34)]
            prediction = prediction.set_index("transaction_id").loc[all_rows.loc[known, "tx_id"]].reset_index()
            # A checkpoint trained on MPS is replayed on CPU; equality is numerical, not bitwise.
            np.testing.assert_allclose(prediction.predicted_probability, expected_probabilities, rtol=1e-3, atol=2e-4)
            threshold_key = "raw_thresholds" if variant == "raw" else "calibrated_thresholds"
            observed = evaluate_predictions(prediction, rules[threshold_key], config["review_k_per_snapshot"])
            record = metrics.loc[(metrics.model_id == model_id) & (metrics.calibration == variant)].iloc[0]
            for name in ["average_precision", "roc_auc", "f1", "brier_score", "precision_at_k",
                         "recall_at_validation_target_fpr", "achieved_fpr_at_validation_target"]:
                assert record[name] == pytest.approx(observed[name], abs=1e-12)
