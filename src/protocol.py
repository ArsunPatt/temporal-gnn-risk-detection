"""Chronology and graph-input boundaries, independent of model implementations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import feature_groups


SPLIT_RANGES = {"train": [1, 29], "validation": [30, 34], "test": [35, 49]}


def chronological_partitions(nodes):
    splits = pd.Series(index=nodes.index, dtype="string", name="partition")
    for name, (first, last) in SPLIT_RANGES.items():
        splits.loc[nodes.time_step.between(first, last)] = name
    if splits.isna().any():
        raise ValueError("Time step outside the predefined partitions")
    return splits


def save_split_definitions(nodes, processed_dir, fingerprints):
    """Local IDs and masks are ignored by Git; public protocol contains no labels."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    definition = nodes[["node_index", "tx_id", "time_step"]].copy()
    definition["partition"] = chronological_partitions(nodes)
    definition.to_csv(processed_dir / "split_membership.csv", index=False)
    manifest = {
        "ranges_inclusive": SPLIT_RANGES, "index_order": "lexicographic_tx_id_string",
        "raw_sha256": fingerprints, "prediction_protocol": "end_of_snapshot",
        "supervision": "known labels only; unknown remains missing",
        "test_access": "explicit final_evaluation=True after methodology freeze",
    }
    (processed_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return definition


def supervised_targets(nodes, partition, final_evaluation=False):
    """Default modeling entry point cannot return test targets.

    This is a workflow guard, not encryption: the public CSV and EDA contain labels.
    Never pass target or raw_class columns as graph inputs or model features.
    """
    if partition not in SPLIT_RANGES:
        raise ValueError("Unknown partition")
    if partition == "test" and not final_evaluation:
        raise PermissionError("Test targets sealed: freeze model selection before final evaluation")
    mask = (chronological_partitions(nodes) == partition) & nodes.target.notna()
    return nodes.loc[mask, "node_index"].to_numpy(), nodes.loc[mask, "target"].to_numpy(dtype=np.int64)


def graph_at_step(nodes, edges, cutoff, group="transaction_sensitivity", scope="cumulative"):
    """Build label-free graph arrays visible at end of cutoff; no future rows.

    No feature recomputation is possible from this release. Eligibility of a row
    does NOT establish point-in-time provenance of its published feature values.
    Edges keep published direction; reverse edges would require an explicit choice.
    """
    if not isinstance(cutoff, (int, np.integer)) or not 1 <= cutoff <= 49:
        raise ValueError("cutoff must be an integer in [1, 49]")
    if scope not in ("cumulative", "snapshot"):
        raise ValueError("scope must be cumulative or snapshot")
    if not np.array_equal(nodes.node_index.to_numpy(), np.arange(len(nodes))):
        raise ValueError("Nodes must retain canonical contiguous node_index order")
    columns = feature_groups()[group]
    mask = nodes.time_step <= cutoff if scope == "cumulative" else nodes.time_step == cutoff
    visible = nodes.loc[mask]
    global_indices = visible.node_index.to_numpy()
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[global_indices] = np.arange(len(visible))
    source = edges.source.to_numpy()
    target = edges.target.to_numpy()
    if ((source < 0) | (source >= len(nodes)) | (target < 0) | (target >= len(nodes))).any():
        raise ValueError("Invalid edge endpoints")
    keep = (remap[source] >= 0) & (remap[target] >= 0) & (edges.available_step.to_numpy() <= cutoff)
    edge_index = np.stack([remap[source[keep]], remap[target[keep]]])
    x = visible[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite graph features; fit any imputer on training data only")
    return {"x": x, "edge_index": edge_index, "global_node_index": global_indices,
            "time_step": visible.time_step.to_numpy(), "feature_columns": columns,
            "cutoff": int(cutoff), "scope": scope}
