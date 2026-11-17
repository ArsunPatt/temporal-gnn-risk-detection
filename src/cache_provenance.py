"""Validate complete Stage 4 caches; recover tables without running a model."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import binary_metrics
from .robustness import sha256


class CacheValidationError(ValueError):
    """An incomplete or inconsistent cache must never be treated as a result."""


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def expected_scenarios(protocol):
    settings = protocol["perturbations"]
    for kind in settings["types"]:
        for severity in settings["levels"]:
            if severity == 0:
                continue
            for seed in settings["seeds"]:
                yield kind, float(severity), int(seed), f"{kind}__{severity:.1f}__seed{seed}"


def validate_scenario_metrics(frame, protocol, kind, severity, seed):
    eligible = [r for r in protocol["models"] if kind in {"noise", "mask"} or r["kind"] == "graph"]
    expected = {r["model_id"] for r in eligible}
    if len(frame) != len(expected) or frame.model_id.duplicated().any() or set(frame.model_id) != expected:
        raise CacheValidationError("Scenario model population is incomplete, duplicated or unexpected")
    if not (frame.perturbation.eq(kind).all() and np.isclose(frame.severity, severity).all()
            and frame.perturbation_seed.eq(seed).all()):
        raise CacheValidationError("Scenario settings do not match the frozen protocol")
    for record in eligible:
        row = frame.loc[frame.model_id == record["model_id"]].iloc[0]
        if (row.architecture, row.feature_configuration, row.training_seed) != (record["architecture"], record["feature_configuration"], record["seed"]):
            raise CacheValidationError("Scenario model metadata does not match the frozen checkpoint")
    metrics = frame[["average_precision", "precision", "recall"]].to_numpy()
    if not np.isfinite(metrics).all() or np.any((metrics < 0) | (metrics > 1)):
        raise CacheValidationError("Invalid cached robustness metrics")
    return expected


def validate_edge_audit(frame, protocol, graph_sizes, kind, severity, seed):
    keys = ["feature_configuration", "time_step"]
    expected = {(group, step) for group in protocol["feature_sets"] for step in protocol["test_steps"]}
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise CacheValidationError("Edge audit is incomplete, duplicated or has unexpected snapshots")
    if not (frame.kind.eq(kind).all() and np.isclose(frame.severity, severity).all() and frame.perturbation_seed.eq(seed).all()):
        raise CacheValidationError("Edge-audit settings differ from the frozen scenario")
    sizes = graph_sizes.set_index("time_step").edges
    for row in frame.itertuples():
        budget = int(sizes.loc[row.time_step])
        changed = int(np.floor(severity * budget)) if kind in {"remove", "rewire"} else 0
        resulting = budget - changed if kind == "remove" else budget
        if row.original_edges != budget or row.resulting_edges != resulting:
            raise CacheValidationError("Edge-audit budget differs from snapshot size and frozen severity")
        if kind in {"remove", "rewire"} and row.changed_edges != changed:
            raise CacheValidationError("Wrong number of removed/rewired edges")
        if kind in {"noise", "mask"} and not (pd.isna(row.changed_edges) or row.changed_edges == 0):
            raise CacheValidationError("Numerical corruption unexpectedly changed edges")


def scenario_paths(private_dir, name):
    private_dir = Path(private_dir)
    return [private_dir / f"metrics__{name}.csv", private_dir / "perturbations" / f"{name}.npz",
            private_dir / f"edge_audit__{name}.csv"]


def commit_scenario(private_dir, name, protocol_digest):
    """Write the completion marker last, after metrics, arrays and audit exist."""
    paths = scenario_paths(private_dir, name)
    if not all(path.is_file() for path in paths):
        raise CacheValidationError("Cannot commit an incomplete scenario: metrics, probabilities and edge audit are required")
    record = {"schema_version": 1, "protocol_sha256": protocol_digest,
              "sha256": {str(path.relative_to(private_dir)): sha256(path) for path in paths}}
    atomic_json(record, Path(private_dir) / f"complete__{name}.json")
    return record


def read_committed_scenario(private_dir, name, protocol_digest):
    metrics_path, _, audit_path = scenario_paths(private_dir, name)
    path = Path(private_dir) / f"complete__{name}.json"
    if not path.is_file():
        raise CacheValidationError(f"Uncommitted scenario {name}; validate/adopt legacy caches explicitly before reuse")
    marker = json.loads(path.read_text())
    expected_names = {str(p.relative_to(private_dir)) for p in scenario_paths(private_dir, name)}
    if marker["protocol_sha256"] != protocol_digest or set(marker["sha256"]) != expected_names:
        raise CacheValidationError("Scenario provenance does not match the frozen protocol/files")
    for filename, digest in marker["sha256"].items():
        file = Path(private_dir) / filename
        if not file.is_file() or sha256(file) != digest:
            raise CacheValidationError(f"Incomplete or changed scenario file: {filename}")
    return pd.read_csv(metrics_path), pd.read_csv(audit_path)


def audit_stage4_cache(root, adopt_legacy=False, recover_tables=False, base=None, private=None):
    """Validate every cached probability/metric and edge budget. Never infer or fit.

    Legacy adoption is dated release-time evidence only, not a retroactive freeze.
    Recovery rebuilds aggregate tables only from complete validated scenarios.
    """
    root = Path(root)
    base = root / "results/stage4" if base is None else Path(base)
    private = root / "results/local/stage4/full_run" if private is None else Path(private)
    official = base / "full_run"
    protocol_path = base / "frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    digest = sha256(protocol_path)
    receipt = json.loads((official / "inference_receipt.json").read_text())
    if receipt["protocol_sha256"] != digest:
        raise CacheValidationError("Inference receipt and frozen protocol differ")
    for name, expected in receipt["prediction_sha256"].items():
        if not (private / name).is_file() or sha256(private / name) != expected:
            raise CacheValidationError(f"Missing or changed base prediction cache: {name}")
    predictions = pd.read_csv(private / "test_raw_predictions.csv", dtype={"transaction_id": str}, float_precision="round_trip")
    model_ids = {r["model_id"] for r in protocol["models"]}
    if set(predictions.model_name) != {name + "__raw" for name in model_ids}:
        raise CacheValidationError("Incomplete base prediction population")
    canonical = predictions.loc[predictions.model_name == protocol["models"][0]["model_id"] + "__raw"]
    identity = ["transaction_id", "time_step", "true_label"]
    for _, frame in predictions.groupby("model_name", sort=False):
        if (frame.transaction_id.duplicated().any() or not frame[identity].reset_index(drop=True).equals(canonical[identity].reset_index(drop=True))
                or not frame["split"].eq("test").all()):
            raise CacheValidationError("Cached models have misaligned evaluation observations")
    population_digest = __import__("hashlib").sha256(canonical[identity].to_csv(index=False).encode()).hexdigest()
    baseline = pd.read_csv(official / "final_test_metrics.csv").query("calibration == 'raw'").set_index("model_id")
    graph_sizes = pd.read_csv(official / "graph_shift.csv")
    edge_path = official / "perturbation_edge_counts.csv"
    existing_edges = pd.read_csv(edge_path) if edge_path.is_file() else None
    metric_parts, edge_parts, file_hashes = [], [], {}
    for kind, severity, seed, name in expected_scenarios(protocol):
        metrics_path, arrays_path, audit_path = scenario_paths(private, name)
        if not metrics_path.is_file() or not arrays_path.is_file():
            raise CacheValidationError(f"Incomplete scenario {name}; missing predictions/metrics cannot be recovered without a separately authorized rerun")
        metrics = pd.read_csv(metrics_path)
        eligible = validate_scenario_metrics(metrics, protocol, kind, severity, seed)
        if adopt_legacy and not audit_path.exists():
            if existing_edges is None:
                raise CacheValidationError("Legacy edge evidence is missing; cannot certify completed scenarios")
            audit = existing_edges.loc[existing_edges.kind.eq(kind) & np.isclose(existing_edges.severity, severity) & existing_edges.perturbation_seed.eq(seed)]
            validate_edge_audit(audit, protocol, graph_sizes, kind, severity, seed)
            atomic_csv(audit, audit_path)
        if not audit_path.exists():
            raise CacheValidationError(f"Missing edge audit for {name}; do not silently skip it")
        audit = pd.read_csv(audit_path)
        validate_edge_audit(audit, protocol, graph_sizes, kind, severity, seed)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if set(arrays.files) != eligible:
                raise CacheValidationError("Scenario probability archive has missing/unexpected models")
            for row in metrics.itertuples():
                probabilities = arrays[row.model_id]
                if probabilities.shape != (len(canonical),) or row.n_labeled != len(canonical):
                    raise CacheValidationError("Scenario probabilities have the wrong evaluation shape")
                threshold = protocol["operating_rules"][row.model_id]["thresholds"]["raw"]["f1_threshold"]
                observed = binary_metrics(canonical.true_label, probabilities, threshold)
                for metric in ["average_precision", "precision", "recall"]:
                    if not np.isclose(getattr(row, metric), observed[metric], rtol=0, atol=1e-12):
                        raise CacheValidationError("Cached probabilities do not reproduce robustness metrics")
                    if not np.isclose(getattr(row, "delta_" + ("ap" if metric == "average_precision" else metric)), observed[metric] - baseline.loc[row.model_id, metric], rtol=0, atol=1e-12):
                        raise CacheValidationError("Robustness baseline difference is inconsistent")
        marker_path = private / f"complete__{name}.json"
        if adopt_legacy and not marker_path.exists():
            commit_scenario(private, name, digest)
        read_committed_scenario(private, name, digest)
        for path in [metrics_path, arrays_path, audit_path, marker_path]:
            file_hashes[str(path.relative_to(root))] = sha256(path)
        metric_parts.append(metrics)
        edge_parts.append(audit)
    rebuilt_metrics = pd.concat(metric_parts, ignore_index=True)
    rebuilt_edges = pd.concat(edge_parts, ignore_index=True)
    for path, rebuilt, keys in [(official / "robustness_results.csv", rebuilt_metrics, ["perturbation", "severity", "perturbation_seed", "model_id"]),
                                (edge_path, rebuilt_edges, ["kind", "severity", "perturbation_seed", "feature_configuration", "time_step"])]:
        if recover_tables:
            atomic_csv(rebuilt, path)
        if not path.is_file():
            raise CacheValidationError("Aggregate cache is missing; use explicit --recover-tables with complete scenario records")
        observed = pd.read_csv(path).sort_values(keys).reset_index(drop=True)
        expected = rebuilt.sort_values(keys).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(observed, expected, check_dtype=False, atol=1e-12, rtol=0)
        except AssertionError as error:
            raise CacheValidationError("Aggregate cache is partial/inconsistent; explicit recovery is required") from error
        file_hashes[str(path.relative_to(root))] = sha256(path)
    provenance = {"schema_version": 1, "protocol_sha256": digest, "inference_receipt_sha256": sha256(official / "inference_receipt.json"),
                  "population_sha256": population_digest, "models": len(model_ids), "known_labels": len(canonical),
                  "scenarios": len(metric_parts), "robustness_rows": len(rebuilt_metrics), "edge_audit_rows": len(rebuilt_edges),
                  "cache_sha256": file_hashes}
    ledger = private / "cache_manifest.json"
    if adopt_legacy or recover_tables:
        provenance["established_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        provenance["scope"] = "Stage 5 cache validation; cannot retroactively establish original code/input freeze"
        atomic_json(provenance, ledger)
    else:
        if not ledger.exists():
            raise CacheValidationError("No cache provenance; explicitly validate/adopt existing release caches first")
        saved = json.loads(ledger.read_text())
        for key, value in provenance.items():
            if saved.get(key) != value:
                raise CacheValidationError(f"Cache provenance changed: {key}")
    return {key: provenance[key] for key in ["protocol_sha256", "models", "known_labels", "scenarios", "robustness_rows", "edge_audit_rows"]}
