import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv

from src.data import FEATURE_COLUMNS, feature_groups
from src.graph_hardware import seed_everything
from src.graph_learning import (GraphRiskModel, build_snapshots, fit_loss, infer_snapshots,
                                masked_loss_sum, training_targets)


@pytest.fixture
def graph_fixture():
    rng = np.random.default_rng(7)
    nodes = pd.DataFrame(rng.normal(size=(12, 166)), columns=FEATURE_COLUMNS)
    nodes["time_step"] = np.repeat([1, 2, 3], 4)
    nodes["node_index"] = np.arange(12)
    nodes["tx_id"] = [str(i) for i in range(12)]
    nodes["target"] = pd.array([0, 1, None, 0] * 3, dtype="Int8")
    edges = pd.DataFrame({"source": [0, 1, 4, 5, 8, 9], "target": [1, 2, 5, 6, 9, 10]})
    edges["source_time"] = nodes.time_step.to_numpy()[edges.source]
    edges["target_time"] = nodes.time_step.to_numpy()[edges.target]
    edges["available_step"] = edges.source_time
    columns = feature_groups()["transaction_sensitivity"]
    scaler = StandardScaler().fit(nodes.loc[nodes.target.notna(), columns])
    graphs = build_snapshots(nodes, edges, scaler, "transaction_sensitivity", max_step=3)
    return nodes, edges, scaler, graphs


@pytest.mark.parametrize("mode", ["original", "self_only", "rewired", "reversed"])
def test_snapshot_alignment_and_no_cross_time_edges(graph_fixture, mode):
    nodes, edges, scaler, _ = graph_fixture
    graphs = build_snapshots(nodes, edges, scaler, "transaction_sensitivity", edge_mode=mode, max_step=3)
    for step, graph in graphs.items():
        assert graph.step == step and len(graph.x) == 4
        assert (nodes.iloc[graph.node_index.numpy()].time_step == step).all()
        assert "y" not in graph and "target" not in graph
        assert graph.edge_index.shape == (2, 0 if mode == "self_only" else 2)
        assert (graph.edge_index >= 0).all() and (graph.edge_index < 4).all()
    batch = Batch.from_data_list(list(graphs.values()))
    source, target = batch.edge_index
    assert torch.equal(batch.batch[source], batch.batch[target])
    targets = training_targets(nodes, graphs)
    assert sum((y == -1).sum() for y in targets.values()) == 3


def test_pyg_direction_is_sender_to_receiver():
    conv = SAGEConv(1, 1, aggr="mean", root_weight=False, bias=False, flow="source_to_target")
    with torch.no_grad():
        conv.lin_l.weight.fill_(1)
    x = torch.tensor([[2.0], [5.0], [7.0]])
    edge = torch.tensor([[0], [1]])
    torch.testing.assert_close(conv(x, edge).squeeze(), torch.tensor([0.0, 2.0, 0.0]))
    torch.testing.assert_close(conv(x, edge.flip(0)).squeeze(), torch.tensor([5.0, 0.0, 0.0]))


def test_rewiring_reproducible_unique_and_budget_preserving(graph_fixture):
    nodes, edges, scaler, original = graph_fixture
    first = build_snapshots(nodes, edges, scaler, "transaction_sensitivity", "rewired", max_step=3, rewire_seed=2026)
    second = build_snapshots(nodes, edges, scaler, "transaction_sensitivity", "rewired", max_step=3, rewire_seed=2026)
    for step, graph in first.items():
        assert torch.equal(graph.edge_index, second[step].edge_index)
        assert graph.num_edges == original[step].num_edges
        assert (graph.edge_index[0] != graph.edge_index[1]).all()
        assert len(set(map(tuple, graph.edge_index.t().tolist()))) == graph.num_edges


def test_unknown_targets_have_zero_direct_loss_gradient():
    logits = torch.tensor([0.2, 1.0, -0.7], requires_grad=True)
    total, count = masked_loss_sum(logits, torch.tensor([0.0, -1.0, 1.0]), torch.tensor(2.0))
    total.backward()
    assert count == 2 and logits.grad[1] == 0
    with pytest.raises(ValueError, match="Targets must"):
        masked_loss_sum(logits, torch.tensor([0.0, 2.0, 1.0]), torch.tensor(2.0))


@pytest.mark.parametrize("architecture", ["graphsage", "gat", "sage_gru"])
def test_model_forward_backward_and_checkpoint_roundtrip(graph_fixture, architecture, tmp_path):
    nodes, _, _, graphs = graph_fixture
    config = {"input_dim": 18, "architecture": architecture, "hidden_dim": 8, "dropout": 0,
              "heads": 2, "context_dim": 4, "attention_dropout": 0}
    model = GraphRiskModel(**config)
    loss = fit_loss(model, graphs, training_targets(nodes, graphs), torch.tensor(2.0))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = infer_snapshots(model, graphs, [3])[3]["logits"]
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": config}, path)
    saved = torch.load(path, weights_only=True)
    restored = GraphRiskModel(**saved["model_config"])
    restored.load_state_dict(saved["state_dict"])
    np.testing.assert_array_equal(before, infer_snapshots(restored, graphs, [3])[3]["logits"])


def test_training_rejects_validation_graphs_and_targets(graph_fixture):
    nodes, _, _, graphs = graph_fixture
    graph = graphs[1].clone()
    graph.step = 30
    with pytest.raises(ValueError, match="training snapshots"):
        training_targets(nodes, {30: graph})
    model = GraphRiskModel(18, hidden_dim=8)
    with pytest.raises(ValueError, match="Optimizer loss"):
        fit_loss(model, {30: graph}, {30: torch.zeros(4)}, torch.tensor(2.0))


def test_temporal_future_feature_perturbation_cannot_change_earlier_output(graph_fixture):
    _, _, _, graphs = graph_fixture
    model = GraphRiskModel(18, architecture="sage_gru", hidden_dim=8, context_dim=4, dropout=0)
    before = infer_snapshots(model, graphs, [1, 2, 3])
    modified = {step: graph.clone() for step, graph in graphs.items()}
    modified[3].x = modified[3].x * -7 + 100
    after = infer_snapshots(model, modified, [1, 2, 3])
    for step in [1, 2]:
        np.testing.assert_array_equal(before[step]["logits"], after[step]["logits"])
    assert not np.allclose(before[3]["logits"], after[3]["logits"])


def test_temporal_history_is_used_but_current_update_is_not_current_context(graph_fixture):
    _, _, _, graphs = graph_fixture
    seed_everything(42)
    model = GraphRiskModel(18, architecture="sage_gru", hidden_dim=8, context_dim=4, dropout=0)
    model.eval()
    before = infer_snapshots(model, graphs, [2])[2]["logits"]
    modified = {step: graph.clone() for step, graph in graphs.items()}
    modified[1].x = modified[1].x.flip(1) * -2
    after = infer_snapshots(model, modified, [2])[2]["logits"]
    assert not np.allclose(before, after)
    with torch.no_grad():
        _, context, _ = model(graphs[1])
        current, next_context, _ = model(graphs[2], context)
        model.gru.bias_ih.add_(10)
        current_again, next_again, _ = model(graphs[2], context)
    torch.testing.assert_close(current, current_again, rtol=0, atol=0)
    assert not torch.allclose(next_context, next_again)


def test_missing_history_and_test_inference_rejected(graph_fixture):
    _, _, _, graphs = graph_fixture
    model = GraphRiskModel(18, architecture="sage_gru", hidden_dim=8, context_dim=4)
    with pytest.raises(ValueError, match="complete ordered history"):
        infer_snapshots(model, {2: graphs[2], 3: graphs[3]}, [3])
    with pytest.raises(ValueError, match="development snapshots"):
        infer_snapshots(model, graphs, [35])


def test_cpu_training_reproducible_for_fixed_seed(graph_fixture):
    nodes, _, _, graphs = graph_fixture
    targets = training_targets(nodes, graphs)
    outputs = []
    for _ in range(2):
        seed_everything(42)
        model = GraphRiskModel(18, hidden_dim=8, dropout=0.25)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            fit_loss(model, graphs, targets, torch.tensor(2.0)).backward()
            optimizer.step()
        outputs.append(infer_snapshots(model, graphs, [3])[3]["logits"])
    np.testing.assert_array_equal(*outputs)


def test_graph_construction_does_not_depend_on_targets(graph_fixture):
    nodes, edges, scaler, graphs = graph_fixture
    changed = nodes.copy()
    changed["target"] = pd.array([None] * len(nodes), dtype="Int8")
    rebuilt = build_snapshots(changed, edges, scaler, "transaction_sensitivity", max_step=3)
    for step in graphs:
        assert torch.equal(graphs[step].x, rebuilt[step].x)
        assert torch.equal(graphs[step].edge_index, rebuilt[step].edge_index)


def test_linear_context_fusion_is_a_snapshot_offset_not_node_reranking(graph_fixture):
    _, _, _, graphs = graph_fixture
    seed_everything(42)
    model = GraphRiskModel(18, architecture="sage_gru", hidden_dim=8, context_dim=4, dropout=0)
    model.eval()
    with torch.no_grad():
        first, _, embedding_first = model(graphs[2], torch.zeros(1, 4))
        second, _, embedding_second = model(graphs[2], torch.ones(1, 4))
    torch.testing.assert_close(embedding_first, embedding_second, rtol=0, atol=0)
    delta = second - first
    torch.testing.assert_close(delta, delta.mean().expand_as(delta), rtol=1e-5, atol=1e-6)
