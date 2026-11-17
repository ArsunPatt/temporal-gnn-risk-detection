"""Synthetic cache recovery tests; fixtures are never used as research results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cache_provenance import (CacheValidationError, atomic_csv, audit_stage4_cache,
    commit_scenario, expected_scenarios, read_committed_scenario, validate_edge_audit,
    validate_scenario_metrics)
from src.robustness import sha256


@pytest.fixture
def cache_fixture(tmp_path):
    public = tmp_path / "results/stage4/full_run"
    private = tmp_path / "results/local/stage4/full_run"
    public.mkdir(parents=True)
    (private / "perturbations").mkdir(parents=True)
    records = [{"model_id": "tabular", "architecture": "lightgbm", "feature_configuration": "small", "seed": 42, "kind": "tabular"},
               {"model_id": "graph", "architecture": "graphsage", "feature_configuration": "small", "seed": 42, "kind": "graph"}]
    protocol = {"models": records, "feature_sets": {"small": ["feature_002"]}, "test_steps": [35, 36],
                "perturbations": {"types": ["noise", "mask", "remove", "rewire"], "levels": [0, .25], "seeds": [7]},
                "operating_rules": {r["model_id"]: {"thresholds": {"raw": {"f1_threshold": .5}}} for r in records}}
    path = public.parent / "frozen_protocol.json"
    path.write_text(json.dumps(protocol))
    frames = [pd.DataFrame({"transaction_id": ["1", "2"], "time_step": [35, 36], "true_label": [0, 1],
                           "split": "test", "model_name": r["model_id"] + "__raw", "seed": 42, "predicted_probability": [.2, .8]}) for r in records]
    pd.concat(frames).to_csv(private / "test_raw_predictions.csv", index=False)
    receipt = {"protocol_sha256": sha256(path), "prediction_sha256": {"test_raw_predictions.csv": sha256(private / "test_raw_predictions.csv")}}
    (public / "inference_receipt.json").write_text(json.dumps(receipt))
    baseline = pd.DataFrame([{**r, "calibration": "raw", "average_precision": 1., "precision": 1., "recall": 1.} for r in records])
    baseline.to_csv(public / "final_test_metrics.csv", index=False)
    sizes = pd.DataFrame({"time_step": [35, 36], "edges": [8, 12]})
    sizes.to_csv(public / "graph_shift.csv", index=False)
    all_metrics, all_edges = [], []
    for kind, severity, seed, name in expected_scenarios(protocol):
        selected = records if kind in {"noise", "mask"} else records[1:]
        metrics = pd.DataFrame([{**r, "training_seed": r["seed"], "perturbation_seed": seed, "perturbation": kind,
            "severity": severity, "average_precision": 1., "precision": 1., "recall": 1., "delta_ap": 0., "delta_precision": 0., "delta_recall": 0.,
            "n_labeled": 2, "probability_cache": f"perturbations/{name}.npz"} for r in selected]).drop(columns="kind")
        metrics.to_csv(private / f"metrics__{name}.csv", index=False)
        np.savez_compressed(private / "perturbations" / f"{name}.npz", **{r["model_id"]: np.array([.2, .8]) for r in selected})
        for step, budget in zip([35,36], [8,12]):
            changed = int(budget * severity) if kind in {"remove", "rewire"} else np.nan
            all_edges.append({"feature_configuration": "small", "kind": kind, "severity": severity, "perturbation_seed": seed,
                "time_step": step, "original_edges": budget, "resulting_edges": budget-int(budget*severity) if kind=="remove" else budget, "changed_edges": changed})
        all_metrics.append(metrics)
    pd.concat(all_metrics).to_csv(public / "robustness_results.csv", index=False)
    pd.DataFrame(all_edges).to_csv(public / "perturbation_edge_counts.csv", index=False)
    return tmp_path, public, private, protocol


def test_legacy_adoption_validates_and_is_read_only_on_reuse(cache_fixture):
    root, public, private, protocol = cache_fixture
    observed = audit_stage4_cache(root, adopt_legacy=True)
    assert observed["scenarios"] == 4 and observed["robustness_rows"] == 6 and observed["edge_audit_rows"] == 8
    before = {p: p.read_bytes() for p in private.rglob("*") if p.is_file()}
    assert audit_stage4_cache(root) == observed
    assert all(path.read_bytes() == value for path, value in before.items())


def test_uncommitted_cache_is_rejected(cache_fixture):
    root, _, private, protocol = cache_fixture
    with pytest.raises(CacheValidationError, match="Missing edge audit"):
        audit_stage4_cache(root)
    with pytest.raises(CacheValidationError, match="Uncommitted"):
        read_committed_scenario(private, "noise__0.2__seed7", "unsealed")


@pytest.mark.parametrize("filename", ["robustness_results.csv", "perturbation_edge_counts.csv"])
def test_partial_aggregate_requires_explicit_recovery(cache_fixture, filename):
    root, public, private, protocol = cache_fixture
    audit_stage4_cache(root, adopt_legacy=True)
    path = public / filename
    original = pd.read_csv(path)
    original.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(CacheValidationError, match="partial/inconsistent"):
        audit_stage4_cache(root)
    audit_stage4_cache(root, recover_tables=True)
    pd.testing.assert_frame_equal(pd.read_csv(path), original)
    assert audit_stage4_cache(root)["edge_audit_rows"] == 8


def test_missing_aggregate_recovers_from_complete_commits(cache_fixture):
    root, public, _, _ = cache_fixture
    audit_stage4_cache(root, adopt_legacy=True)
    (public / "robustness_results.csv").unlink()
    (public / "perturbation_edge_counts.csv").unlink()
    with pytest.raises(CacheValidationError, match="Aggregate cache is missing"):
        audit_stage4_cache(root)
    assert audit_stage4_cache(root, recover_tables=True)["scenarios"] == 4


def test_missing_scenario_predictions_cannot_be_recovered(cache_fixture):
    root, _, private, protocol = cache_fixture
    audit_stage4_cache(root, adopt_legacy=True)
    name = next(expected_scenarios(protocol))[-1]
    path = private / "perturbations" / f"{name}.npz"
    path.rename(path.with_suffix(".pending"))
    with pytest.raises(CacheValidationError, match="cannot be recovered"):
        audit_stage4_cache(root, recover_tables=True)


def test_wrong_probability_metrics_are_not_sealed(cache_fixture):
    root, _, private, protocol = cache_fixture
    name = next(expected_scenarios(protocol))[-1]
    np.savez_compressed(private / "perturbations" / f"{name}.npz", tabular=[.9,.1], graph=[.2,.8])
    with pytest.raises(CacheValidationError, match="do not reproduce"):
        audit_stage4_cache(root, adopt_legacy=True)
    assert not (private / f"complete__{name}.json").exists()


def test_changed_edge_audit_and_stale_protocol_rejected(cache_fixture):
    root, public, private, protocol = cache_fixture
    audit_stage4_cache(root, adopt_legacy=True)
    name = next(expected_scenarios(protocol))[-1]
    audit_path = private / f"edge_audit__{name}.csv"
    audit = pd.read_csv(audit_path)
    audit.loc[0, "resulting_edges"] = 99
    audit.to_csv(audit_path, index=False)
    with pytest.raises(CacheValidationError, match="budget"):
        audit_stage4_cache(root)
    with pytest.raises(CacheValidationError, match="changed scenario"):
        read_committed_scenario(private, name, sha256(public.parent / "frozen_protocol.json"))


def test_commit_is_not_written_for_missing_files(tmp_path):
    with pytest.raises(CacheValidationError, match="Cannot commit"):
        commit_scenario(tmp_path, "partial", "digest")
    assert not list(tmp_path.glob("complete*"))


def test_missing_private_inputs_skip_but_present_invalid_inputs_fail(require_private_files, tmp_path):
    path = tmp_path / "unavailable.csv"
    with pytest.raises(pytest.skip.Exception, match="Private artifacts unavailable"):
        require_private_files([path])
    path.write_text("")
    require_private_files([path])
    with pytest.raises(pd.errors.EmptyDataError):
        pd.read_csv(path)


def test_atomic_cache_write_has_no_pending_file(tmp_path):
    path = tmp_path / "table.csv"
    atomic_csv(pd.DataFrame({"count": [3]}), path)
    assert pd.read_csv(path)["count"].item() == 3
    assert not (tmp_path / "table.csv.pending").exists()
