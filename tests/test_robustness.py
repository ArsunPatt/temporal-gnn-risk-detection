"""Synthetic boundary checks; no test-period outcome is used to choose settings."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

from src.frozen_inference import context_before_test, infer_test_graphs, stage4_snapshots, test_prediction_frame as make_test_frame
from src.graph_learning import GraphRiskModel
from src.robustness import (aligned_pair, bootstrap_ap, circular_block_counts, distribution_shift,
                            expected_calibration_error, freeze_protocol, perturb_edges,
                            perturb_features, sha256, verify_frozen)


def test_protocol_is_immutable_and_hash_verified(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"fixed")
    protocol = {"run_mode": "FULL_RUN", "test_steps": list(range(35, 50)),
                "artifact_sha256": {"model.bin": sha256(artifact)}}
    path = tmp_path / "frozen.json"
    digest = freeze_protocol(path, protocol)
    assert freeze_protocol(path, protocol) == digest
    assert verify_frozen(tmp_path, path, digest) == protocol
    with pytest.raises(ValueError, match="never overwrite"):
        freeze_protocol(path, dict(protocol, new_parameter=1))
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_frozen(tmp_path, path, digest)


@pytest.mark.parametrize("kind", ["noise", "mask"])
def test_feature_perturbation_zero_and_seed_reproducibility(kind):
    values = np.ones((200, 3))
    args = (values, np.zeros(3), np.ones(3), [0, 3, 10], kind)
    np.testing.assert_array_equal(perturb_features(*args, 0, 7, 35), values)
    first = perturb_features(*args, .2, 7, 35)
    np.testing.assert_array_equal(first, perturb_features(*args, .2, 7, 35))
    assert not np.array_equal(first, perturb_features(*args, .2, 8, 35))


def test_noise_scales_linearly_and_masking_is_nested():
    values = np.ones((10000, 2))
    args = (values, np.zeros(2), np.array([2., 3.]), [0, 1])
    low = perturb_features(*args, "noise", .1, 2, 35)
    high = perturb_features(*args, "noise", .3, 2, 35)
    np.testing.assert_allclose(high - values, 3 * (low - values))
    np.testing.assert_allclose((high - values).std(axis=0), [.6, .9], rtol=.04)
    low = perturb_features(*args, "mask", .1, 2, 35) == 0
    high = perturb_features(*args, "mask", .3, 2, 35) == 0
    assert np.all(~low | high)
    assert high.mean() == pytest.approx(.3, abs=.015)


def test_random_fields_match_feature_subsets():
    x = np.ones((30, 165))
    whole = perturb_features(x, np.zeros(165), np.ones(165), list(range(165)), "noise", .2, 5, 36)
    subset = perturb_features(x[:, [0, 7, 14]], np.zeros(3), np.ones(3), [0, 7, 14], "noise", .2, 5, 36)
    np.testing.assert_array_equal(subset, whole[:, [0, 7, 14]])


@pytest.mark.parametrize("kind", ["remove", "rewire"])
def test_edge_perturbation_budget_and_direction(kind):
    edge = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 3, 4, 5, 6, 7, 8, 9]])
    original = edge.copy()
    observed, count = perturb_edges(edge, 12, kind, .3, 9, 35)
    assert count == 3
    assert observed.shape[1] == (7 if kind == "remove" else 10)
    assert not (observed[0] == observed[1]).any()
    assert len(np.unique(observed[0] * 12 + observed[1])) == observed.shape[1]
    np.testing.assert_array_equal(observed, perturb_edges(edge, 12, kind, .3, 9, 35)[0])
    np.testing.assert_array_equal(edge, original)
    retained = set(zip(*observed)) & set(zip(*edge))
    assert len(retained) == 7
    np.testing.assert_array_equal(perturb_edges(edge, 12, kind, 0, 9, 35)[0], edge)


def test_edge_removal_is_nested():
    edges = np.stack([np.arange(20), np.arange(1, 21)])
    low = perturb_edges(edges, 22, "remove", .1, 42, 35)[0]
    high = perturb_edges(edges, 22, "remove", .3, 42, 35)[0]
    assert set(zip(*high)).issubset(set(zip(*low)))


def test_calibration_error_endpoints():
    assert expected_calibration_error([0, 1], [0, 1]) == 0
    assert expected_calibration_error([0, 1], [1, 0]) == 1
    assert expected_calibration_error([0, 1], [.5, .5]) == 0
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [np.nan, .5])


def test_drift_identity_constant_and_shift():
    values = np.arange(100, dtype=float)
    identity = distribution_shift(values, values)
    assert identity["psi"] == identity["ks_statistic"] == identity["wasserstein_train_sd"] == 0
    shifted = distribution_shift(values, values + 500)
    assert shifted["psi"] > 0 and shifted["ks_statistic"] == 1
    assert np.isnan(distribution_shift(np.ones(10), np.ones(20))["wasserstein_train_sd"])


def test_block_resampling_counts_and_reproducibility():
    counts = circular_block_counts()
    assert counts.shape == (1000, 15)
    np.testing.assert_array_equal(counts.sum(axis=1), 15)
    np.testing.assert_array_equal(counts, circular_block_counts())
    np.testing.assert_array_equal(circular_block_counts(block_length=15), np.ones((1000, 15)))


def test_bootstrap_matches_literal_paired_block_expansion():
    labels = np.array([0, 1, 1, 0, 0, 1])
    scores = np.array([.1, .9, .5, .6, .2, .3])
    steps = np.array([0, 0, 1, 1, 2, 2])
    counts = circular_block_counts(3, 2, 20, 7)
    observed = bootstrap_ap(labels, scores, steps, counts)
    for index, weights in enumerate(counts):
        expanded = np.repeat(np.arange(6), weights[steps])
        assert observed[index] == pytest.approx(average_precision_score(labels[expanded], scores[expanded]))
    np.testing.assert_array_equal(observed - bootstrap_ap(labels, scores, steps, counts), 0)


def test_paired_population_alignment_and_rejection():
    frame = pd.DataFrame({"transaction_id": ["1", "2"], "time_step": [35, 36], "true_label": [0, 1]})
    first, second = aligned_pair(frame, frame.iloc[::-1])
    assert first.equals(second)
    with pytest.raises(ValueError, match="different populations"):
        aligned_pair(frame, frame.assign(true_label=[1, 0]))


def test_test_prediction_schema_and_unknown_exclusion():
    rows = pd.DataFrame({"tx_id": ["1", "2"], "time_step": [35, 49], "target": [0, 1]})
    result = make_test_frame(rows, [.1, .8], "frozen", 42)
    assert result["split"].eq("test").all() and result.shape == (2, 7)
    with pytest.raises(ValueError):
        make_test_frame(rows.assign(target=[0, np.nan]), [.1, .8], "frozen", 42)


def test_test_graph_construction_rejects_labels_and_quick_mode():
    with pytest.raises(ValueError, match="Remove label"):
        stage4_snapshots(pd.DataFrame({"target": [1]}), None, None, "unused",
                         {"run_mode": "FULL_RUN", "test_steps": list(range(35, 50))})
    with pytest.raises(PermissionError):
        stage4_snapshots(None, None, None, "unused", {"run_mode": "QUICK_RUN"})


def test_frozen_temporal_inference_is_causal_and_parameter_invariant():
    torch.manual_seed(8)
    model = GraphRiskModel(2, "sage_gru", hidden_dim=4, context_dim=3, dropout=0)
    graphs = {t: Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), step=t) for t in range(1, 50)}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    context = context_before_test(model, graphs)
    reference = infer_test_graphs(model, graphs, context)
    changed = dict(graphs)
    changed[49] = graphs[49].clone()
    changed[49].x *= 100
    altered = infer_test_graphs(model, changed, context)
    np.testing.assert_array_equal(reference[:-3], altered[:-3])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="context"):
        infer_test_graphs(model, graphs)
    with pytest.raises(ValueError, match="timestamp"):
        infer_test_graphs(model, {t: graphs[t] for t in range(2, 50)}, context)
