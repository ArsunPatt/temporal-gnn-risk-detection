"""Strict loading of the original, headerless Elliptic release. No imputation."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)
FEATURE_COLUMNS = ["time_step"] + [f"feature_{i:03d}" for i in range(2, 167)]
EXPECTED_COUNTS = {
    "transactions": 203_769, "edges": 234_355, "time_steps": 49,
    "illicit": 4_545, "licit": 42_019, "unknown": 157_205,
}


def feature_groups():
    """One-based published numbers; no group is certified leakage-free.

    Compare tabular and graph models with matching 18-, 93-, or 165-feature
    configurations. Address-history and neighborhood lists support the audit.
    CSV column zero is the ID and is never a predictor.
    """
    return {
        "published_without_time": FEATURE_COLUMNS[1:],
        "local_without_time": [f"feature_{i:03d}" for i in range(2, 95)],
        "address_history": [f"feature_{i:03d}" for i in range(23, 95)],
        "neighborhood": [f"feature_{i:03d}" for i in range(95, 167)],
        # Drop unresolved transaction features 4, 8, 16. Still not proven clean.
        "transaction_sensitivity": [
            f"feature_{i:03d}" for i in range(2, 23) if i not in (4, 8, 16)
        ],
    }


def check_ids(values, context):
    if values.isna().any() or not values.str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"{context}: IDs must be nonmissing decimal strings")


def load_elliptic(raw_dir, verify_original=True):
    """Return aligned nodes, directed indexed edges, and aggregate diagnostics.

    Unknown targets are nullable Int8, never zero. Canonical index order is
    lexicographic by the untouched ID string, independent of CSV row order.
    Missing or infinite numeric features, including time, fail at loading;
    no rows are dropped, values replaced, or source files modified.
    Set verify_original=False only for explicitly synthetic unit-test fixtures.
    """
    raw_dir = Path(raw_dir)
    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing original Elliptic CSVs: " + ", ".join(missing)
            + ". Download ellipticco/elliptic-data-set from Kaggle; see README.md."
        )

    features = pd.read_csv(raw_dir / RAW_FILES[0], header=None, dtype={0: "string"})
    if features.shape[1] != 167:
        raise ValueError("Features must have 167 columns: txId + 166 published features")
    features.columns = ["tx_id"] + FEATURE_COLUMNS
    classes = pd.read_csv(raw_dir / RAW_FILES[1], dtype="string", keep_default_na=False)
    edges = pd.read_csv(raw_dir / RAW_FILES[2], dtype="string")
    if list(classes.columns) != ["txId", "class"]:
        raise ValueError("Class header must be txId,class")
    if list(edges.columns) != ["txId1", "txId2"]:
        raise ValueError("Edge header must be txId1,txId2")
    classes = classes.rename(columns={"txId": "tx_id", "class": "raw_class"})
    for values, context in [(features.tx_id, "features"), (classes.tx_id, "classes"),
                            (edges.txId1, "edge sources"), (edges.txId2, "edge targets")]:
        check_ids(values, context)
    duplicate_features = int(features.duplicated().sum())
    duplicate_classes = int(classes.duplicated().sum())
    duplicate_edges = int(edges.duplicated().sum())
    if features.tx_id.duplicated().any() or classes.tx_id.duplicated().any():
        raise ValueError("Duplicate transaction IDs in features or classes; refusing ambiguous joins")
    if duplicate_edges:
        raise ValueError("Duplicate directed edges; resolve explicitly before message passing")
    if set(features.tx_id) != set(classes.tx_id):
        raise ValueError("Feature and class ID populations differ")
    if not classes.raw_class.isin(["1", "2", "unknown"]).all():
        raise ValueError("Classes must be exactly 1, 2, or unknown")

    try:
        numeric = features[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    except ValueError as error:
        raise ValueError("Numeric feature columns contain an unparseable value; check the CSV contents and header") from error
    numeric_values = numeric.to_numpy(dtype=np.float64)
    non_finite = ~np.isfinite(numeric_values)
    if non_finite.any():
        missing_count = int(np.isnan(numeric_values).sum())
        infinite_count = int(np.isinf(numeric_values).sum())
        affected_columns = ", ".join(numeric.columns[non_finite.any(axis=0)])
        raise ValueError(
            f"Non-finite numeric features: {missing_count} missing (NaN), "
            f"{infinite_count} infinite values. Affected columns: {affected_columns}. "
            "Loading stopped without dropping rows or imputing values."
        )
    times = numeric.time_step
    if times.isna().any() or not times.between(1, 49).all() or not (times % 1 == 0).all():
        raise ValueError("Time steps must be integers in [1, 49]; check header and column ordering")
    features[FEATURE_COLUMNS] = numeric
    features["time_step"] = times.astype("int16")
    nodes = features.merge(classes, on="tx_id", validate="one_to_one")
    nodes = nodes.sort_values("tx_id", kind="stable").reset_index(drop=True)
    nodes.insert(0, "node_index", np.arange(len(nodes), dtype=np.int64))
    nodes["target"] = nodes.raw_class.map({"1": 1, "2": 0}).astype("Int8")
    nodes["class_name"] = nodes.raw_class.map({"1": "illicit", "2": "licit", "unknown": "unknown"})
    id_to_index = nodes.set_index("tx_id").node_index
    mapped_edges = pd.DataFrame({
        "source": edges.txId1.map(id_to_index),
        "target": edges.txId2.map(id_to_index),
    })
    if mapped_edges.isna().any().any():
        raise ValueError("Invalid edge endpoints: transaction absent from features")
    mapped_edges = mapped_edges.astype("int64").sort_values(["source", "target"]).reset_index(drop=True)
    source_time = nodes.time_step.to_numpy()[mapped_edges.source]
    target_time = nodes.time_step.to_numpy()[mapped_edges.target]
    mapped_edges["source_time"] = source_time
    mapped_edges["target_time"] = target_time
    # Endpoints are the only available proxy for edge arrival; not a source audit.
    mapped_edges["available_step"] = np.maximum(source_time, target_time)
    counts = {
        "transactions": len(nodes), "edges": len(mapped_edges),
        "time_steps": nodes.time_step.nunique(),
        **nodes.class_name.value_counts().reindex(["illicit", "licit", "unknown"], fill_value=0).to_dict(),
    }
    counts = {name: int(value) for name, value in counts.items()}
    if verify_original and counts != EXPECTED_COUNTS:
        raise ValueError(f"Original-release count mismatch. Observed {counts}; expected {EXPECTED_COUNTS}")
    diagnostics = {
        **counts, "published_features_including_time": 166,
        "missing_feature_values": int(nodes[FEATURE_COLUMNS].isna().sum().sum()),
        "non_finite_feature_values": int(non_finite.sum()),
        "duplicate_feature_records": duplicate_features,
        "duplicate_class_records": duplicate_classes, "duplicate_edges": duplicate_edges,
        "duplicate_feature_vectors": int(nodes[FEATURE_COLUMNS].duplicated().sum()),
        "self_loops": int((mapped_edges.source == mapped_edges.target).sum()),
        "cross_time_edges": int((source_time != target_time).sum()),
        "backward_time_edges": int((source_time > target_time).sum()),
        "unknown_percentage": 100 * counts["unknown"] / len(nodes),
    }
    return nodes, mapped_edges, diagnostics


def file_fingerprints(raw_dir):
    """Hash the full input files without including local filesystem paths."""
    fingerprints = {}
    for name in RAW_FILES:
        digest = hashlib.sha256()
        with (Path(raw_dir) / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprints[name] = digest.hexdigest()
    return fingerprints


def snapshot_summary(nodes, edges):
    counts = pd.crosstab(nodes.time_step, nodes.class_name)
    counts = counts.reindex(index=range(1, 50), columns=["licit", "illicit", "unknown"], fill_value=0)
    counts["transactions"] = counts[["licit", "illicit", "unknown"]].sum(axis=1)
    counts["labeled"] = counts.licit + counts.illicit
    counts["illicit_prevalence"] = counts.illicit.div(counts.labeled.replace(0, np.nan))
    counts["label_coverage"] = counts.labeled.div(counts.transactions.replace(0, np.nan))
    within = edges.loc[edges.source_time == edges.target_time].groupby("source_time").size()
    arrivals = edges.groupby("available_step").size()
    counts["within_snapshot_edges"] = within.reindex(counts.index, fill_value=0)
    counts["arriving_edges"] = arrivals.reindex(counts.index, fill_value=0)
    counts["within_edges_per_node"] = counts.within_snapshot_edges.div(counts.transactions.replace(0, np.nan))
    counts.index.name = "time_step"
    return counts
