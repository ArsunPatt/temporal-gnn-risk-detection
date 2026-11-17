from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS, RAW_FILES, feature_groups, file_fingerprints, load_elliptic
from src.protocol import chronological_partitions, graph_at_step, save_split_definitions, supervised_targets


def test_exact_label_encoding_and_id_join(loaded_fixture):
    nodes, edges, _ = loaded_fixture
    by_id = nodes.set_index("tx_id")
    assert by_id.loc["300", "target"] == 1
    assert by_id.loc["10", "target"] == 0
    assert pd.isna(by_id.loc["200", "target"])
    assert list(nodes.node_index) == list(range(len(nodes)))
    assert nodes.tx_id.tolist() == sorted(nodes.tx_id.tolist())
    original_pairs = set(zip(nodes.tx_id.iloc[edges.source], nodes.tx_id.iloc[edges.target]))
    assert original_pairs == {("300", "10"), ("10", "200"), ("200", "500"), ("500", "40")}


def test_dimensions_and_positional_groups(loaded_fixture):
    nodes, _, _ = loaded_fixture
    groups = feature_groups()
    assert len(FEATURE_COLUMNS) == 166
    assert nodes[FEATURE_COLUMNS].shape == (6, 166)
    assert len(groups["local_without_time"]) == 93
    assert len(groups["neighborhood"]) == 72
    assert len(groups["transaction_sensitivity"]) == 18
    assert len(groups["published_without_time"]) == 165
    assert set(groups["transaction_sensitivity"]) < set(groups["local_without_time"]) < set(groups["published_without_time"])
    assert all("time_step" not in columns for columns in groups.values())
    assert set(groups["local_without_time"]).isdisjoint(groups["neighborhood"])
    assert groups["local_without_time"][-1] == "feature_094"
    assert groups["neighborhood"][0] == "feature_095"
    assert nodes.set_index("tx_id").loc["300", "feature_002"] == 2


def test_partitions_disjoint_exhaustive_and_boundary_correct(loaded_fixture):
    nodes, _, _ = loaded_fixture
    splits = chronological_partitions(nodes)
    populations = [set(nodes.loc[splits == name, "node_index"]) for name in ["train", "validation", "test"]]
    assert not (populations[0] & populations[1] or populations[0] & populations[2] or populations[1] & populations[2])
    assert set.union(*populations) == set(nodes.node_index)
    assert dict(zip(nodes.time_step, splits)) == {1: "train", 29: "train", 30: "validation", 34: "validation", 35: "test", 49: "test"}


@pytest.mark.parametrize("cutoff", [1, 29, 30, 34, 35, 49])
@pytest.mark.parametrize("scope", ["snapshot", "cumulative"])
def test_graph_never_contains_future_inputs(loaded_fixture, cutoff, scope):
    nodes, edges, _ = loaded_fixture
    graph = graph_at_step(nodes, edges, cutoff, scope=scope)
    assert (graph["time_step"] <= cutoff).all()
    if scope == "snapshot":
        assert (graph["time_step"] == cutoff).all()
    assert "target" not in graph and "raw_class" not in graph
    assert graph["x"].shape == (len(graph["global_node_index"]), 18)
    expected_x = nodes.iloc[graph["global_node_index"]][feature_groups()["transaction_sensitivity"]].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(graph["x"], expected_x)
    indices = graph["edge_index"]
    assert indices.shape[0] == 2
    assert (indices >= 0).all() and (indices < len(graph["x"])).all()
    visible = set(graph["global_node_index"])
    expected = {(int(row.source), int(row.target)) for row in edges.itertuples()
                if row.source in visible and row.target in visible and row.available_step <= cutoff}
    actual = set(zip(graph["global_node_index"][indices[0]], graph["global_node_index"][indices[1]]))
    assert actual == expected


def test_unknown_nodes_retained_but_not_supervised(loaded_fixture):
    nodes, edges, _ = loaded_fixture
    unknown_index = nodes.loc[nodes.target.isna(), "node_index"].item()
    assert unknown_index in graph_at_step(nodes, edges, 34)["global_node_index"]
    indices, targets = supervised_targets(nodes, "validation")
    assert unknown_index not in indices
    assert set(targets) == {1}
    with pytest.raises(PermissionError, match="sealed"):
        supervised_targets(nodes, "test")
    indices, targets = supervised_targets(nodes, "test", final_evaluation=True)
    assert len(indices) == len(targets) == 2


@pytest.mark.parametrize("corruption,match", [
    ("endpoint", "Invalid edge endpoints"), ("label", "Classes must"),
    ("duplicate_id", "Duplicate transaction IDs"), ("duplicate_edge", "Duplicate directed edges"),
    ("time", "Time steps"), ("width", "167 columns"),
    ("missing_class", "populations differ"),
])
def test_corrupt_inputs_rejected(raw_fixture, corruption, match):
    features_path, classes_path, edges_path = [raw_fixture / name for name in RAW_FILES]
    if corruption in ("endpoint", "duplicate_edge"):
        frame = pd.read_csv(edges_path)
        if corruption == "endpoint": frame.loc[0, "txId2"] = 999999
        else: frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        frame.to_csv(edges_path, index=False)
    elif corruption in ("label", "missing_class"):
        frame = pd.read_csv(classes_path)
        if corruption == "label": frame.loc[0, "class"] = "bad"
        else: frame = frame.iloc[1:]
        frame.to_csv(classes_path, index=False)
    else:
        frame = pd.read_csv(features_path, header=None)
        if corruption == "duplicate_id": frame.loc[1, 0] = frame.loc[0, 0]
        elif corruption == "time":
            frame[1] = frame[1].astype(float)
            frame.loc[0, 1] = 30.5
        elif corruption == "width": frame = frame.iloc[:, :-1]
        frame.to_csv(features_path, index=False, header=False)
    with pytest.raises(ValueError, match=match):
        load_elliptic(raw_fixture, verify_original=False)


def test_row_permutations_preserve_alignment(raw_fixture):
    nodes_before, edges_before, _ = load_elliptic(raw_fixture, verify_original=False)
    for name in RAW_FILES:
        header = None if name == RAW_FILES[0] else 0
        frame = pd.read_csv(raw_fixture / name, header=header)
        frame.sample(frac=1, random_state=2).to_csv(raw_fixture / name, index=False, header=header is not None)
    nodes_after, edges_after, _ = load_elliptic(raw_fixture, verify_original=False)
    pd.testing.assert_frame_equal(nodes_before, nodes_after)
    pd.testing.assert_frame_equal(edges_before, edges_after)


def test_stable_saved_splits_contain_no_targets(loaded_fixture, raw_fixture, tmp_path):
    nodes, _, _ = loaded_fixture
    output = tmp_path / "splits"
    fingerprints = file_fingerprints(raw_fixture)
    save_split_definitions(nodes, output, fingerprints)
    original = (output / "split_membership.csv").read_bytes()
    save_split_definitions(nodes, output, fingerprints)
    assert original == (output / "split_membership.csv").read_bytes()
    assert set(pd.read_csv(output / "split_membership.csv").columns) == {"node_index", "tx_id", "time_step", "partition"}


def test_missing_files_have_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_elliptic(tmp_path)


@pytest.mark.parametrize("column", [1, 2, 23, 95, 166])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "positive_infinity", "negative_infinity"])
def test_non_finite_numeric_features_rejected_without_changing_source(raw_fixture, column, value):
    path = raw_fixture / RAW_FILES[0]
    frame = pd.read_csv(path, header=None)
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False, header=False)
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="Non-finite numeric features") as error:
        load_elliptic(raw_fixture, verify_original=False)
    assert FEATURE_COLUMNS[column - 1] in str(error.value)
    expected_counts = "1 missing (NaN), 0 infinite" if np.isnan(value) else "0 missing (NaN), 1 infinite"
    assert expected_counts in str(error.value)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("token", ["", "NaN", "null", "not-a-number"])
def test_missing_csv_tokens_and_unparseable_features_rejected(raw_fixture, token):
    import csv
    path = raw_fixture / RAW_FILES[0]
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    rows[0][2] = token
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    message = "unparseable" if token == "not-a-number" else "Non-finite numeric features"
    with pytest.raises(ValueError, match=message):
        load_elliptic(raw_fixture, verify_original=False)


def test_combined_missing_and_infinite_features_reported(raw_fixture):
    path = raw_fixture / RAW_FILES[0]
    frame = pd.read_csv(path, header=None)
    frame[[2, 23, 95]] = frame[[2, 23, 95]].astype(float)
    frame.loc[0, 2] = np.nan
    frame.loc[1, 23] = np.inf
    frame.loc[2, 95] = -np.inf
    frame.to_csv(path, index=False, header=False)
    with pytest.raises(ValueError, match=r"1 missing \(NaN\), 2 infinite") as error:
        load_elliptic(raw_fixture, verify_original=False)
    assert "feature_002, feature_023, feature_095" in str(error.value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "positive_infinity", "negative_infinity"])
def test_graph_rejects_non_finite_features_introduced_after_loading(loaded_fixture, value):
    nodes, edges, _ = loaded_fixture
    nodes = nodes.copy()
    nodes["feature_002"] = nodes["feature_002"].astype(float)
    nodes.loc[0, "feature_002"] = value
    with pytest.raises(ValueError, match="imputer on training"):
        graph_at_step(nodes, edges, 49)


def test_original_counts_cannot_silently_accept_fixture(raw_fixture):
    with pytest.raises(ValueError, match="count mismatch"):
        load_elliptic(raw_fixture)


def test_earlier_graph_unchanged_when_future_features_change(loaded_fixture):
    nodes, edges, _ = loaded_fixture
    before = graph_at_step(nodes, edges, 34)
    modified = nodes.copy()
    modified.loc[modified.time_step >= 35, feature_groups()["published_without_time"]] = 1e6
    after = graph_at_step(modified, edges, 34)
    for name in ["x", "edge_index", "global_node_index", "time_step"]:
        np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.integration
def test_original_release_and_all_cutoffs():
    import os
    raw_dir = Path(os.environ.get("ELLIPTIC_DATA_DIR", "data/raw"))
    if not all((raw_dir / name).is_file() for name in RAW_FILES):
        pytest.skip("Original CSVs absent: synthetic unit tests are not dataset validation")
    nodes, edges, diagnostics = load_elliptic(raw_dir)
    assert diagnostics["published_features_including_time"] == 166
    for cutoff in range(1, 50):
        graph = graph_at_step(nodes, edges, cutoff)
        assert (graph["time_step"] <= cutoff).all()
        source, target = graph["edge_index"]
        assert (graph["time_step"][source] <= cutoff).all()
        assert (graph["time_step"][target] <= cutoff).all()
