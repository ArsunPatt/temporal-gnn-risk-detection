"""Binary metrics and labeled-review budgets shared by research notebooks."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


PREDICTION_COLUMNS = ["transaction_id", "time_step", "split", "true_label", "model_name", "seed", "predicted_probability"]


def check_binary_inputs(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if pd.isna(labels).any() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation requires known binary labels; unknown is never negative")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return labels.astype(int), probabilities


def binary_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = check_binary_inputs(labels, probabilities)
    if np.isnan(threshold) or threshold < 0:
        raise ValueError("Threshold must be nonnegative; infinity permits an all-negative rule")
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    return {
        "n_labeled": len(labels), "n_illicit": positives, "prevalence": positives / len(labels),
        "average_precision": float(average_precision_score(labels, probabilities)) if positives else np.nan,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if positives and negatives else np.nan,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives if positives else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "false_positive_rate": fp / negatives if negatives else np.nan,
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def validation_window(predictions, allowed_steps):
    """Reject test, unknown, duplicate or mixed-model rows before fitting choices."""
    if not allowed_steps or not set(allowed_steps).issubset(range(30, 35)):
        raise ValueError("Allowed steps must stay within validation steps 30–34")
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if not predictions["split"].eq("validation").all() or not predictions.time_step.isin(allowed_steps).all():
        raise ValueError("Selection must use the declared validation window only")
    if predictions.transaction_id.duplicated().any():
        raise ValueError("Duplicate transactions in selection population")
    if predictions[["model_name", "seed"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Select thresholds for one model and seed at a time")
    labels, probabilities = check_binary_inputs(predictions.true_label, predictions.predicted_probability)
    if np.unique(labels).size != 2:
        raise ValueError("Selection requires both labeled classes")
    return labels, probabilities


def choose_thresholds(predictions, target_fpr=0.01, allowed_steps=(32, 33)):
    """Empirical, tie-aware rules selected only on the declared validation rows."""
    labels, probabilities = validation_window(predictions, allowed_steps)
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be in [0, 1]")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    # Ties use the highest threshold to avoid unnecessary reviews.
    best_f1 = thresholds[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]]
    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= target_fpr)
    best_fpr = eligible[np.flatnonzero(tpr[eligible] == tpr[eligible].max())[0]]
    return {"f1_threshold": float(best_f1), "fpr_threshold": float(roc_thresholds[best_fpr]),
            "target_fpr": float(target_fpr), "source_steps": list(allowed_steps)}


def prediction_frame(rows, probabilities, model_name, seed):
    """The common seven-column format. Stage 2 permits validation rows only."""
    labels, probabilities = check_binary_inputs(rows.target, probabilities)
    if not rows.time_step.between(30, 34).all() or rows.tx_id.duplicated().any():
        raise ValueError("Stage 2 predictions must be unique labeled validation transactions")
    return pd.DataFrame({
        "transaction_id": rows.tx_id.to_numpy(dtype=str), "time_step": rows.time_step.to_numpy(),
        "split": "validation", "true_label": labels, "model_name": model_name,
        "seed": int(seed), "predicted_probability": probabilities,
    })[PREDICTION_COLUMNS]


def assert_common_population(predictions, expected_rows):
    """Compare identities, timestamps and targets, not just sample counts."""
    expected = pd.DataFrame({"transaction_id": expected_rows.tx_id.astype(str).to_numpy(),
                             "time_step": expected_rows.time_step.to_numpy(),
                             "true_label": expected_rows.target.to_numpy(dtype=int)})
    columns = ["transaction_id", "time_step", "true_label"]
    expected = expected.sort_values("transaction_id").reset_index(drop=True)
    if predictions.empty:
        raise ValueError("Empty model comparison")
    for _, frame in predictions.groupby(["model_name", "seed"], sort=False):
        check_binary_inputs(frame.true_label, frame.predicted_probability)
        if not frame["split"].eq("validation").all() or frame.transaction_id.duplicated().any():
            raise ValueError("Invalid evaluation split or duplicate transaction")
        observed = frame[columns].copy()
        observed["transaction_id"] = observed.transaction_id.astype(str)
        observed = observed.sort_values("transaction_id").reset_index(drop=True)
        if not observed.equals(expected):
            # Integer widths can differ after CSV reload; compare values explicitly.
            if observed.shape != expected.shape or not (observed.to_numpy() == expected.to_numpy()).all():
                raise ValueError("Models do not share the same evaluation population and labels")
    return True


def review_budget_metrics(predictions, k=None, fraction=None):
    """Review top K (or ceil(fraction * n)) within each labeled snapshot.

    Ties use canonical string transaction ID order. Return pooled hits/reviews,
    not the unweighted mean of per-period precision. Unknown candidates are absent.
    """
    check_binary_inputs(predictions.true_label, predictions.predicted_probability)
    if (k is None) == (fraction is None):
        raise ValueError("Specify exactly one of k or fraction")
    if k is not None and (not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError("k must be a positive integer")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if predictions.transaction_id.duplicated().any():
        raise ValueError("Review one model/seed without duplicate transactions")
    reviewed = hits = 0
    for _, frame in predictions.groupby("time_step", sort=True):
        count = min(k, len(frame)) if k is not None else int(np.ceil(fraction * len(frame)))
        ordered = frame.assign(transaction_id=frame.transaction_id.astype(str)).sort_values(
            ["predicted_probability", "transaction_id"], ascending=[False, True], kind="stable")
        reviewed += count
        hits += int(ordered.iloc[:count].true_label.sum())
    positives = int(predictions.true_label.sum())
    return {"reviewed_count": reviewed, "review_hits": hits,
            "review_precision": hits / reviewed, "review_recall": hits / positives if positives else np.nan}


def evaluate_predictions(predictions, thresholds, review_k=25):
    metrics = binary_metrics(predictions.true_label, predictions.predicted_probability, thresholds["f1_threshold"])
    constrained = binary_metrics(predictions.true_label, predictions.predicted_probability, thresholds["fpr_threshold"])
    reviewed = review_budget_metrics(predictions, k=review_k)
    metrics.update(recall_at_validation_target_fpr=constrained["recall"],
                   achieved_fpr_at_validation_target=constrained["false_positive_rate"],
                   validation_target_fpr=thresholds["target_fpr"],
                   precision_at_k=reviewed["review_precision"], reviewed_at_k=reviewed["reviewed_count"],
                   review_k_per_snapshot=review_k)
    return metrics
