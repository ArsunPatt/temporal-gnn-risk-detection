"""Replay the frozen unperturbed test predictions on CPU using authorized local artifacts."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_release import digest
from src.baselines import apply_sigmoid, decision_scores
from src.data import load_elliptic
from src.frozen_inference import context_before_test, infer_test_graphs, stage4_snapshots
from src.graph_learning import GraphRiskModel

ABSOLUTE_TOLERANCE = 1e-5


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Frozen file checksum differs: {path.name}")


def replay_frozen(private_root):
    """Read frozen inputs and emit only aggregate validation statistics."""
    torch.set_num_threads(2)
    stage4 = ROOT / "results/stage4"
    protocol_path = stage4 / "frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    official = stage4 / "full_run"
    receipt = json.loads((official / "inference_receipt.json").read_text())
    check_hash(protocol_path, receipt["protocol_sha256"])
    for name, expected in protocol["artifact_sha256"].items():
        base = private_root if name.startswith("models/") else ROOT
        check_hash(base / name, expected)
    raw_dir = private_root / "data/raw"
    for name, expected in protocol["raw_sha256"].items():
        check_hash(raw_dir / name, expected)

    cached_dir = private_root / "results/local/stage4/full_run"
    predictions = {}
    for variant in ("raw", "platt"):
        path = cached_dir / f"test_{variant}_predictions.csv"
        check_hash(path, receipt["prediction_sha256"][path.name])
        predictions[variant] = pd.read_csv(
            path, dtype={"transaction_id": str}, float_precision="round_trip"
        ).groupby("model_name")
    metrics = pd.read_csv(official / "final_test_metrics.csv").set_index(["model_id", "calibration"])
    nodes, edges, _ = load_elliptic(raw_dir)
    known_test = nodes.loc[nodes.time_step.isin(protocol["test_steps"]) & nodes.target.notna()]
    known_test = known_test.sort_values(["time_step", "node_index"])
    test_ids = known_test.tx_id.astype(str).to_numpy()
    labels = known_test.target.to_numpy(dtype=int)
    feature_nodes = nodes.drop(columns=["target", "class_name", "raw_class"])
    config = json.loads((ROOT / "results/stage3/full_run/selected_configurations.json").read_text())

    maximum_ap_error = maximum_probability_error = 0.0
    changed_decisions = 0
    graph_group = None
    for record in sorted(protocol["models"], key=lambda row: row["feature_configuration"]):
        model_id = record["model_id"]
        group = record["feature_configuration"]
        columns = protocol["feature_sets"][group]
        bundle = joblib.load(private_root / record.get("calibration_artifact", record["artifact"]))
        if record["kind"] == "tabular":
            scores = decision_scores(bundle["base_model"], known_test[columns])
        else:
            if graph_group != group:
                scaler = joblib.load(private_root / record["scaler_artifact"])
                graphs = stage4_snapshots(feature_nodes, edges, scaler, config["group_helper_keys"][group], protocol)
                indices = np.concatenate([graphs[step].node_index.numpy() for step in protocol["test_steps"]])
                selected = nodes.iloc[indices].target.notna().to_numpy()
                if not np.array_equal(indices[selected], known_test.node_index.to_numpy()):
                    raise ValueError("Frozen graph population does not match the test population")
                graph_group = group
            saved = torch.load(private_root / record["artifact"], map_location="cpu", weights_only=True)
            if saved["feature_columns"] != columns:
                raise ValueError(f"Feature order differs for {model_id}")
            model = GraphRiskModel(**saved["model_config"])
            model.load_state_dict(saved["state_dict"], strict=True)
            scores = infer_test_graphs(model, graphs, context_before_test(model, graphs))[selected]

        for variant, probability in (("raw", expit(scores)), ("platt", apply_sigmoid(bundle["calibrator"], scores))):
            saved = predictions[variant].get_group(f"{model_id}__{variant}")
            if (not np.array_equal(saved.transaction_id, test_ids)
                    or not np.array_equal(saved.true_label, labels)
                    or not np.array_equal(saved.time_step, known_test.time_step)
                    or not saved["split"].eq("test").all()):
                raise ValueError(f"Saved evaluation population differs for {model_id} ({variant})")
            stored = saved.predicted_probability.to_numpy()
            threshold = protocol["operating_rules"][model_id]["thresholds"][variant]["f1_threshold"]
            published_ap = metrics.loc[(model_id, variant), "average_precision"]
            maximum_ap_error = max(maximum_ap_error, abs(average_precision_score(labels, probability) - published_ap))
            maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(probability - stored))))
            changed_decisions += int(np.count_nonzero((probability >= threshold) != (stored >= threshold)))

    passed = (maximum_ap_error <= ABSOLUTE_TOLERANCE
              and maximum_probability_error <= ABSOLUTE_TOLERANCE
              and changed_decisions == 0)
    return {
        "passed": passed,
        "device": "CPU",
        "models_replayed": len(protocol["models"]),
        "probability_variants": 2 * len(protocol["models"]),
        "maximum_absolute_pooled_ap_difference": maximum_ap_error,
        "maximum_absolute_probability_difference": maximum_probability_error,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "changed_frozen_f1_decisions": changed_decisions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True,
                        help="Authorized local tree containing data/raw, models, and results/local")
    args = parser.parse_args()
    try:
        result = replay_frozen(args.private_root)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
