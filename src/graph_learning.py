"""Small graph models and label-free snapshot operations shared with tests."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, SAGEConv

from .protocol import graph_at_step


class SnapshotEncoder(nn.Module):
    """Two graph layers; normalization is within each node, never across time."""

    def __init__(self, input_dim, hidden_dim, dropout, architecture, heads=2, attention_dropout=0.1):
        super().__init__()
        if architecture == "gat":
            if hidden_dim % heads:
                raise ValueError("hidden_dim must be divisible by GAT heads")
            self.first = GATConv(input_dim, hidden_dim // heads, heads=heads,
                                 dropout=attention_dropout, add_self_loops=True, flow="source_to_target")
            self.second = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                                  dropout=attention_dropout, add_self_loops=True, flow="source_to_target")
        else:
            self.first = SAGEConv(input_dim, hidden_dim, aggr="mean", flow="source_to_target")
            self.second = SAGEConv(hidden_dim, hidden_dim, aggr="mean", flow="source_to_target")
        self.norm_first = nn.LayerNorm(hidden_dim)
        self.norm_second = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.norm_first(self.first(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.norm_second(self.second(x, edge_index)))
        return F.dropout(x, p=self.dropout, training=self.training)


class GraphRiskModel(nn.Module):
    """GraphSAGE, GAT, or GraphSAGE + Historical GRU Context; no node memory."""

    def __init__(self, input_dim, architecture="graphsage", hidden_dim=64, dropout=0.25,
                 heads=2, context_dim=32, attention_dropout=0.1):
        super().__init__()
        if architecture not in ("graphsage", "gat", "sage_gru"):
            raise ValueError("Unknown architecture")
        self.temporal = architecture == "sage_gru"
        self.context_dim = context_dim
        self.encoder = SnapshotEncoder(input_dim, hidden_dim, dropout, architecture, heads, attention_dropout)
        if self.temporal:
            self.gru = nn.GRUCell(hidden_dim, context_dim)
        self.classifier = nn.Linear(hidden_dim + (context_dim if self.temporal else 0), 1)

    def forward(self, graph, context=None):
        embeddings = self.encoder(graph.x, graph.edge_index)
        if self.temporal:
            if context is None:
                context = embeddings.new_zeros((1, self.context_dim))
            combined = torch.cat([embeddings, context.expand(len(embeddings), -1)], dim=1)
            logits = self.classifier(combined).squeeze(-1)
            # Prediction precedes the update: current summary can influence t+1 only.
            next_context = self.gru(embeddings.mean(dim=0, keepdim=True), context)
        else:
            logits = self.classifier(embeddings).squeeze(-1)
            next_context = None
        return logits, next_context, embeddings


def rewire_edges(edge_index, num_nodes, seed):
    """Uniform simple directed graph: same exact edge budget, not same degrees."""
    edge_count = edge_index.shape[1]
    if edge_count > num_nodes * (num_nodes - 1):
        raise ValueError("Edge budget exceeds simple directed graph capacity")
    if not edge_count:
        return np.empty((2, 0), dtype=np.int64)
    rng = np.random.default_rng(seed)
    codes = rng.choice(num_nodes * (num_nodes - 1), size=edge_count, replace=False)
    source = codes // (num_nodes - 1)
    target = codes % (num_nodes - 1)
    target += target >= source
    return np.stack([source, target]).astype(np.int64)


def build_snapshots(nodes, edges, scaler, group_key, edge_mode="original", max_step=34, rewire_seed=2026):
    """Build only development snapshots; targets are deliberately absent from Data."""
    if max_step > 34 or max_step < 1:
        raise ValueError("Stage 3 constructs development snapshots only (1–34)")
    if edge_mode not in ("original", "self_only", "rewired", "reversed"):
        raise ValueError("Unknown edge mode")
    snapshots = {}
    for step in range(1, max_step + 1):
        arrays = graph_at_step(nodes, edges, step, group=group_key, scope="snapshot")
        rows = nodes.iloc[arrays["global_node_index"]]
        x = scaler.transform(rows[arrays["feature_columns"]]).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite standardized snapshot inputs")
        edge_index = arrays["edge_index"]
        if edge_mode == "self_only":
            edge_index = np.empty((2, 0), dtype=np.int64)
        elif edge_mode == "rewired":
            edge_index = rewire_edges(edge_index, len(rows), rewire_seed + step)
        elif edge_mode == "reversed":
            edge_index = edge_index[::-1].copy()
        graph = Data(x=torch.from_numpy(x), edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                     node_index=torch.as_tensor(arrays["global_node_index"], dtype=torch.long), step=step)
        if graph.edge_index.numel() and (graph.edge_index.min() < 0 or graph.edge_index.max() >= len(x)):
            raise ValueError("Invalid within-snapshot endpoint")
        snapshots[step] = graph
    return snapshots


def training_targets(nodes, snapshots):
    """Return -1 for unknown; no validation labels can enter this loss interface."""
    if any(step < 1 or step > 29 for step in snapshots):
        raise ValueError("Neural fitting targets must use training snapshots 1–29 only")
    targets = {}
    for step, graph in snapshots.items():
        if graph.step != step:
            raise ValueError("Snapshot key and timestamp differ")
        values = nodes.iloc[graph.node_index.cpu().numpy()].target.fillna(-1).to_numpy(dtype=np.float32)
        targets[step] = torch.from_numpy(values)
    return targets


def masked_loss_sum(logits, targets, positive_weight):
    mask = targets != -1
    if not torch.isin(targets, targets.new_tensor([-1, 0, 1])).all():
        raise ValueError("Targets must be known binary labels or unknown=-1")
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask],
                                               pos_weight=positive_weight, reduction="sum"), int(mask.sum())


def fit_loss(model, snapshots, targets, positive_weight, static_batch=None):
    """One full-batch training objective; temporal gradients span all train steps."""
    if not snapshots or any(step < 1 or step > 29 for step in snapshots):
        raise ValueError("Optimizer loss cannot use validation/future graphs")
    if set(snapshots) != set(targets):
        raise ValueError("Training graph and target steps differ")
    if not model.temporal:
        batch = static_batch if static_batch is not None else Batch.from_data_list(list(snapshots.values()))
        logits, _, _ = model(batch)
        target = torch.cat([targets[step] for step in snapshots])
        total, count = masked_loss_sum(logits, target, positive_weight)
    else:
        if list(snapshots) != list(range(1, max(snapshots) + 1)):
            raise ValueError("Temporal training requires an ordered prefix starting at 1")
        context = None
        total = snapshots[1].x.new_zeros(())
        count = 0
        for step, graph in snapshots.items():
            logits, context, _ = model(graph, context)
            step_loss, step_count = masked_loss_sum(logits, targets[step], positive_weight)
            total = total + step_loss
            count += step_count
    if count == 0:
        raise ValueError("No known training labels")
    return total / count


def infer_snapshots(model, snapshots, output_steps, embeddings=False):
    """Frozen inference; temporal inputs are a strictly ordered observed prefix."""
    if not output_steps or not set(output_steps).issubset(snapshots) or max(output_steps) > 34:
        raise ValueError("Inference is restricted to available development snapshots")
    cutoff = max(output_steps)
    if model.temporal and list(snapshots)[:cutoff] != list(range(1, cutoff + 1)):
        raise ValueError("Temporal inference requires a complete ordered history")
    model.eval()
    outputs = {}
    context = None
    with torch.no_grad():
        steps = range(1, cutoff + 1) if model.temporal else sorted(output_steps)
        for step in steps:
            logits, context, representation = model(snapshots[step], context)
            if not torch.isfinite(logits).all() or (context is not None and not torch.isfinite(context).all()):
                raise FloatingPointError("Nonfinite graph inference output")
            if step in output_steps:
                outputs[step] = {"logits": logits.detach().cpu().numpy()}
                if embeddings:
                    outputs[step]["embeddings"] = representation.detach().cpu().numpy()
    return outputs
