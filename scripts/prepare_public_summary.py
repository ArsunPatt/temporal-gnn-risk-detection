"""Prepare aggregate curves and late-window diagnostics from saved predictions only."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.robustness import sha256


def prepare_summary(root=ROOT, stage4_dir=None, predictions_dir=None, output_dir=None):
    root = Path(root)
    published_stage4 = root / "results/stage4"
    published_summary = root / "results/stage5"
    stage4_dir = published_stage4 if stage4_dir is None else root / stage4_dir
    default_predictions = root / "results/local" / stage4_dir.name / "full_run"
    private = default_predictions if predictions_dir is None else root / predictions_dir
    reproduction = (stage4_dir.resolve() != published_stage4.resolve()
                    or private.resolve() != (root / "results/local/stage4/full_run").resolve())
    if reproduction and output_dir is None:
        raise ValueError("Reproduction inputs require a separate --output-dir")
    result = published_summary if output_dir is None else root / output_dir
    if reproduction and result.resolve().is_relative_to(published_summary.resolve()):
        raise ValueError("Choose an --output-dir outside the published results/stage5 directory")

    source = stage4_dir / "full_run"
    protocol_path = stage4_dir / "frozen_protocol.json"
    metrics_path = source / "per_time_step_metrics.csv"
    receipt_path = source / "inference_receipt.json"
    raw_path = private / "test_raw_predictions.csv"
    platt_path = private / "test_platt_predictions.csv"
    missing = [str(path) for path in [protocol_path, metrics_path, receipt_path, raw_path, platt_path]
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved summary inputs:\n  " + "\n  ".join(missing))

    protocol = json.loads(protocol_path.read_text())
    protocol_digest = sha256(protocol_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("protocol_sha256") != protocol_digest:
        raise ValueError(f"Inference receipt references a different frozen protocol: {receipt_path}")
    for path in [raw_path, platt_path]:
        if sha256(path) != receipt.get("prediction_sha256", {}).get(path.name):
            raise ValueError(f"Saved prediction checksum differs from the inference receipt: {path}")

    per_step = pd.read_csv(metrics_path).query("calibration == 'raw'")
    expected_rows = pd.MultiIndex.from_product(
        [[run["model_id"] for run in protocol["models"]], protocol["test_steps"]]
    )
    observed_rows = pd.MultiIndex.from_frame(per_step[["model_id", "time_step"]])
    if observed_rows.has_duplicates or set(observed_rows) != set(expected_rows):
        raise ValueError(f"Expected one raw metrics row per frozen model and test step: {metrics_path}")
    population_sizes = per_step.groupby("model_id").n_labeled.sum()
    late_records = []
    for (model_id, architecture, group, seed), rows in per_step.groupby(["model_id", "architecture", "feature_configuration", "seed"]):
        for name, low, high in [("early_test",35,42), ("late_test",43,49), ("pooled_test",35,49)]:
            subset = rows.loc[rows.time_step.between(low, high)]
            counts = subset[["tp","fn","fp","tn"]].sum().astype(int).to_dict()
            positives, negatives = counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]
            late_records.append({"model_id": model_id, "architecture": architecture, "feature_configuration": group, "seed": seed,
                "window": name, "first_step": low, "last_step": high, **counts,
                "illicit_labels": positives, "licit_labels": negatives,
                "recall": counts["tp"]/positives if positives else np.nan,
                "false_positive_rate": counts["fp"]/negatives if negatives else np.nan,
                "diagnostic_status": "post-hoc descriptive partition; unchanged saved F1 threshold"})
    late = pd.DataFrame(late_records)
    curves, calibration = [], []
    display_seed = protocol["primary_plot_seed"]
    for variant, path in [("raw",raw_path),("platt",platt_path)]:
        predictions = pd.read_csv(path, dtype={"transaction_id":str}, float_precision="round_trip")
        for run in protocol["models"]:
            if run["seed"] != display_seed:
                continue
            frame = predictions.loc[predictions.model_name.eq(run["model_id"] + "__" + variant)]
            if (len(frame) != population_sizes.loc[run["model_id"]]
                    or set(frame.true_label) != {0, 1} or frame.transaction_id.duplicated().any()):
                raise ValueError(f"Unexpected known-label curve population for {run['model_id']}: {path}")
            metadata = {"model_id":run["model_id"],"architecture":run["architecture"],"feature_configuration":run["feature_configuration"],"seed":run["seed"]}
            if variant == "raw":
                precision, recall, _ = precision_recall_curve(frame.true_label, frame.predicted_probability)
                fpr, tpr, _ = roc_curve(frame.true_label, frame.predicted_probability)
                for x,y in zip(np.linspace(0,1,101),np.interp(np.linspace(0,1,101),recall[::-1],precision[::-1])):
                    curves.append({**metadata,"curve":"pr","x":x,"y":y})
                for x,y in zip(np.linspace(0,1,101),np.interp(np.linspace(0,1,101),fpr,tpr)):
                    curves.append({**metadata,"curve":"roc","x":x,"y":y})
            assignments = np.minimum((frame.predicted_probability.to_numpy()*10).astype(int),9)
            for index in range(10):
                part = frame.loc[assignments == index]
                calibration.append({**metadata,"calibration":variant,"bin":index,"count":len(part),
                    "mean_probability":part.predicted_probability.mean(),"illicit_fraction":part.true_label.mean()})
    if not curves:
        raise ValueError(f"No models for display seed {display_seed}: {protocol_path}")
    result.mkdir(parents=True, exist_ok=True)
    late.to_csv(result / "operating_window_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(result / "curve_display_points.csv",index=False)
    pd.DataFrame(calibration).to_csv(result / "calibration_display_bins.csv",index=False)
    provenance={"created_at_utc":pd.Timestamp.now(tz="UTC").isoformat(),"protocol_sha256":protocol_digest,
        "inference_receipt_sha256":sha256(receipt_path),"source_predictions_sha256":receipt["prediction_sha256"],
        "per_time_step_metrics_sha256":sha256(metrics_path),
        "new_inference":False,"new_model_fits":0,"curve_display":"101 fixed interpolated display points; reported AP/ROC-AUC come from full saved metrics, not area under this grid",
        "calibration":"10 fixed equal-width bins; no calibration fitting","primary_display_seed":display_seed,
        "late_window":"43–49 is a post-hoc diagnostic using saved threshold decisions",
        "independent_audit_scope":"The audit covered the original release's numerical and code consistency. Independent private inference remained unverified.",
        "output_sha256":{name:sha256(result/name) for name in ["operating_window_metrics.csv","curve_display_points.csv","calibration_display_bins.csv"]}}
    (result/"summary_provenance.json").write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({"status": "passed", "curve_points": len(curves),
                      "calibration_bins": len(calibration), "operating_window_rows": len(late),
                      "new_inference": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-dir", type=Path, help="Stage 4 folder containing frozen_protocol.json and full_run/")
    parser.add_argument("--predictions-dir", type=Path, help="Folder containing the saved raw and Platt prediction CSVs")
    parser.add_argument("--output-dir", type=Path, help="Summary output folder, required for reproduction inputs")
    args = parser.parse_args()
    try:
        prepare_summary(stage4_dir=args.stage4_dir, predictions_dir=args.predictions_dir, output_dir=args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")
