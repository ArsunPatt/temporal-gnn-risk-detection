"""Stage 4 inference only. Existing Stage 3 development guards stay unchanged."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .protocol import graph_at_step
from .robustness import perturb_edges, perturb_features


def stage4_snapshots(feature_nodes, edges, scaler, group_key, frozen_protocol):
    """Label-free, end-of-snapshot graphs, including the authorized test period."""
    if frozen_protocol["run_mode"] != "FULL_RUN" or frozen_protocol["test_steps"] != list(range(35, 50)):
        raise PermissionError("Test graphs require the frozen FULL_RUN protocol")
    if {"target", "raw_class", "class_name"} & set(feature_nodes.columns):
        raise ValueError("Remove label columns before constructing inference inputs")
    graphs = {}
    for step in range(1, 50):
        arrays = graph_at_step(feature_nodes, edges, step, group=group_key, scope="snapshot")
        rows = feature_nodes.iloc[arrays["global_node_index"]]
        columns = arrays["feature_columns"]
        if list(scaler.feature_names_in_) != columns:
            raise ValueError("Saved scaler feature ordering does not match graph inputs")
        x = scaler.transform(rows[columns]).astype(np.float32)
        graphs[step] = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(arrays["edge_index"]),
                            node_index=torch.from_numpy(arrays["global_node_index"]), step=step)
    return graphs


def modified_snapshots(original, feature_nodes, columns, scaler, train_mean, train_std,
                       kind, severity, seed, output_steps):
    """Perturb only declared test snapshots; the development context stays clean."""
    result = dict(original)
    changes = []
    indices = [int(name.split("_")[1]) - 2 for name in columns]
    for step in output_steps:
        graph = original[step]
        changed = graph.clone()
        if kind in {"noise", "mask"}:
            values = feature_nodes.iloc[graph.node_index.numpy()][columns].to_numpy()
            values = perturb_features(values, train_mean, train_std, indices, kind, severity, seed, step)
            values = pd.DataFrame(values, columns=columns)
            changed.x = torch.from_numpy(scaler.transform(values).astype(np.float32))
            changed_count = None
        else:
            adjacency, changed_count = perturb_edges(graph.edge_index.numpy(), graph.num_nodes, kind, severity, seed, step)
            changed.edge_index = torch.from_numpy(adjacency)
        if not torch.isfinite(changed.x).all():
            raise FloatingPointError("Perturbation produced nonfinite features")
        result[step] = changed
        changes.append({"time_step": step, "original_edges": graph.num_edges,
                        "resulting_edges": changed.num_edges, "changed_edges": changed_count})
    return result, changes


def context_before_test(model, graphs):
    """Reconstruct 1–34 without labels and without updating neural parameters."""
    if not model.temporal:
        return None
    if list(graphs)[:34] != list(range(1, 35)):
        raise ValueError("Historical context requires the complete ordered development prefix")
    context = None
    model.eval()
    with torch.no_grad():
        for step in range(1, 35):
            _, context, _ = model(graphs[step], context)
            if not torch.isfinite(context).all():
                raise FloatingPointError("Nonfinite historical context")
    return context.detach().clone()


def infer_test_graphs(model, graphs, context=None):
    """Predict 35–49 in order; update history only after each current prediction."""
    if list(graphs) != list(range(1, 50)) or any(graph.step != step for step, graph in graphs.items()):
        raise ValueError("Test inference needs complete, timestamp-aligned snapshots 1–49")
    if model.temporal and context is None:
        raise ValueError("Reconstruct feature-only context through 34 before temporal test inference")
    model.eval()
    scores = []
    with torch.no_grad():
        for step in range(35, 50):
            logits, context, _ = model(graphs[step], context)
            if not torch.isfinite(logits).all() or (context is not None and not torch.isfinite(context).all()):
                raise FloatingPointError("Nonfinite frozen-model inference")
            scores.append(logits.detach().cpu().numpy())
    return np.concatenate(scores)


def test_prediction_frame(rows, probabilities, model_id, seed):
    """Same seven-column interface as development; unknown labels prohibited."""
    from .evaluation import PREDICTION_COLUMNS, check_binary_inputs
    labels, probabilities = check_binary_inputs(rows.target, probabilities)
    if not rows.time_step.between(35, 49).all() or rows.tx_id.duplicated().any():
        raise ValueError("Expected unique known-label final-test rows")
    return pd.DataFrame({"transaction_id": rows.tx_id.astype(str).to_numpy(),
                         "time_step": rows.time_step.to_numpy(), "split": "test", "true_label": labels,
                         "model_name": model_id, "seed": int(seed), "predicted_probability": probabilities})[PREDICTION_COLUMNS]
