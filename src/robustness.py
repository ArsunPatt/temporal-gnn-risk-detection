"""Small, testable statistics and perturbations for frozen-model evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import average_precision_score


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_protocol(path, protocol):
    """Write once; a changed analysis requires a separately named protocol."""
    path = Path(path)
    content = json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Frozen protocol differs; never overwrite it after test access")
    else:
        with path.open("x") as stream:
            stream.write(content)
    return sha256(path)


def verify_frozen(root, protocol_path, expected_sha256):
    if sha256(protocol_path) != expected_sha256:
        raise ValueError("Frozen protocol checksum changed")
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol["run_mode"] != "FULL_RUN" or protocol["test_steps"] != list(range(35, 50)):
        raise PermissionError("Official test inference requires a frozen FULL_RUN protocol")
    for name, expected in protocol["artifact_sha256"].items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f"Saved artifact changed: {name}")
    return protocol


def perturb_features(values, train_mean, train_std, feature_indices, kind, severity, seed, step):
    """Shared random fields across feature subsets, models and training seeds.

    Noise = alpha * training population SD * N(0,1). Masking replaces each
    independently selected entry by its training mean, before preprocessing.
    The random field always has 165 columns in published feature order.
    """
    if kind not in {"none", "noise", "mask"} or not 0 <= severity <= 1:
        raise ValueError("Invalid feature perturbation")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_indices) or not np.isfinite(values).all():
        raise ValueError("Feature matrix must be finite and match the declared ordering")
    if kind == "none" or severity == 0:
        return values.copy()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), 11]))
    shape = (len(values), 165)
    if kind == "noise":
        noise = rng.standard_normal(shape)[:, feature_indices]
        return values + severity * np.asarray(train_std) * noise
    masked = rng.random(shape)[:, feature_indices] < severity
    return np.where(masked, np.asarray(train_mean), values)


def perturb_edges(edge_index, num_nodes, kind, severity, seed, step):
    """Nested removal/partial rewiring; exact counts, no loops or duplicates.

    Rewiring preserves the total directed edge budget, not node degrees or
    endpoints. Replacements cannot coincide with any original edge.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if kind not in {"none", "remove", "rewire"} or not 0 <= severity <= 1:
        raise ValueError("Invalid edge perturbation")
    if edge_index.shape[0] != 2:
        raise ValueError("Expected sender/receiver edge array")
    count = edge_index.shape[1]
    changed_count = int(np.floor(severity * count))
    if kind == "none" or changed_count == 0:
        return edge_index.copy(), 0
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), 23]))
    order = rng.permutation(count)
    keep = np.ones(count, dtype=bool)
    keep[order[:changed_count]] = False
    remaining = edge_index[:, keep]
    if kind == "remove":
        return remaining, changed_count
    original = set((edge_index[0] * num_nodes + edge_index[1]).tolist())
    if changed_count > num_nodes * (num_nodes - 1) - len(original):
        raise ValueError("Not enough distinct non-original edges to rewire")
    replacements = []
    seen = set(original)
    while len(replacements) < changed_count:
        source, target = rng.integers(num_nodes, size=2)
        code = int(source * num_nodes + target)
        if source != target and code not in seen:
            replacements.append((source, target))
            seen.add(code)
    return np.concatenate([remaining, np.asarray(replacements).T], axis=1), changed_count


def expected_calibration_error(labels, probabilities, bins=10):
    """Equal-width bins fixed in advance; includes both probability endpoints."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Empty or unaligned calibration population")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Expected known labels and finite probabilities")
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    total = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            total += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(total)


def distribution_shift(reference, current, bins=10, pseudocount=0.5):
    """Training-quantile PSI, empirical KS distance, scale-normalized Wasserstein.

    Report KS statistic only: dependent graph observations invalidate naive
    independent-sample p-values. No universal PSI alarm thresholds are assumed.
    """
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if not len(reference) or not len(current) or not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Shift statistics require nonempty finite populations")
    cuts = np.unique(np.quantile(reference, np.arange(1, bins) / bins))
    boundaries = np.r_[-np.inf, cuts, np.inf]
    ref = np.histogram(reference, boundaries)[0] + pseudocount
    cur = np.histogram(current, boundaries)[0] + pseudocount
    ref, cur = ref / ref.sum(), cur / cur.sum()
    std = reference.std(ddof=0)
    return {"psi": float(np.sum((cur - ref) * np.log(cur / ref))),
            "ks_statistic": float(ks_2samp(reference, current, method="asymp").statistic),
            "wasserstein_train_sd": float(wasserstein_distance(reference, current) / std) if std > 0 else np.nan,
            "reference_mean": float(reference.mean()), "reference_sd": float(std),
            "current_mean": float(current.mean()), "current_sd": float(current.std(ddof=0))}


def circular_block_counts(number_steps=15, block_length=3, repetitions=1000, seed=20261109):
    """Resample consecutive snapshots, wrapping endpoints; never individual nodes."""
    if not 1 <= block_length <= number_steps or repetitions < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    counts = np.zeros((repetitions, number_steps), dtype=np.int64)
    for iteration in range(repetitions):
        starts = rng.integers(number_steps, size=int(np.ceil(number_steps / block_length)))
        indices = ((starts[:, None] + np.arange(block_length)) % number_steps).ravel()[:number_steps]
        counts[iteration] = np.bincount(indices, minlength=number_steps)
    return counts


def bootstrap_ap(labels, scores, step_positions, counts):
    """Recompute pooled AP with block multiplicity as sample weight."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    positions = np.asarray(step_positions)
    if len(labels) != len(scores) or len(labels) != len(positions):
        raise ValueError("Paired bootstrap observations are not aligned")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Bootstrap requires known binary labels and finite scores")
    result = np.full(len(counts), np.nan)
    for index, multiplicities in enumerate(counts):
        weights = multiplicities[positions]
        if np.sum(labels * weights) and np.sum((1 - labels) * weights):
            result[index] = average_precision_score(labels, scores, sample_weight=weights)
    return result


def aligned_pair(first, second):
    """Require exact identities, timestamps and labels before paired inference."""
    columns = ["transaction_id", "time_step", "true_label"]
    first = first.sort_values("transaction_id").reset_index(drop=True)
    second = second.sort_values("transaction_id").reset_index(drop=True)
    if first.transaction_id.duplicated().any() or second.transaction_id.duplicated().any():
        raise ValueError("Duplicate paired observations")
    if first[columns].shape != second[columns].shape or not (first[columns].to_numpy() == second[columns].to_numpy()).all():
        raise ValueError("Paired models have different populations")
    return first, second
