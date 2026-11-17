"""Small model constructors and chronological development helpers."""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import feature_groups
from .evaluation import validation_window


VALIDATION_ROLES = {"selection": [30, 31], "calibration": [32, 33], "audit": [34]}


def baseline_feature_sets():
    groups = feature_groups()
    return {
        "conservative18": groups["transaction_sensitivity"],
        "local93": groups["local_without_time"],
        "published165": groups["published_without_time"],
        "published165_plus_time": groups["published_without_time"] + ["time_step"],
    }


def development_frames(nodes):
    """Never return final-test rows or unknown supervision to training code."""
    known = nodes.loc[nodes.target.notna() & nodes.time_step.le(34)].copy()
    frames = {"train": known.loc[known.time_step.between(1, 29)].copy()}
    for role, steps in VALIDATION_ROLES.items():
        frames[role] = known.loc[known.time_step.isin(steps)].copy()
    for role, frame in frames.items():
        if frame.empty or frame.target.nunique() != 2:
            raise ValueError(f"{role} requires nonempty known labels from both classes")
    return frames


def make_baseline(family, parameters, weighting, seed, train_labels, n_jobs=2):
    """Construct an unfitted estimator; the notebook owns every fit and search."""
    if weighting not in ("ordinary", "balanced"):
        raise ValueError("weighting must be ordinary or balanced")
    labels = np.asarray(train_labels)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("Class weighting requires known binary training labels")
    if family == "logistic_regression":
        options = {"solver": "lbfgs", "max_iter": 2000, "tol": 1e-6}
        options.update(parameters)
        options.update(class_weight="balanced" if weighting == "balanced" else None, random_state=seed)
        classifier = LogisticRegression(**options)
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    if family == "lightgbm":
        positive_weight = float((labels == 0).sum() / (labels == 1).sum()) if weighting == "balanced" else 1.0
        options = {"objective": "binary", "learning_rate": 0.05, "min_child_samples": 40,
                   "reg_lambda": 1.0, "verbosity": -1, "importance_type": "gain"}
        options.update(parameters)
        options.update(n_jobs=n_jobs, random_state=seed, deterministic=True,
                       force_col_wise=True, scale_pos_weight=positive_weight)
        return LGBMClassifier(**options)
    raise ValueError("Unknown baseline family")


def decision_scores(model, features):
    """Use native scores rather than clipping saturated probabilities into ties."""
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
    else:
        scores = model.predict(features, raw_score=True)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Calibration decision scores must be a finite vector")
    return scores


def fit_sigmoid_calibrator(predictions, scores, allowed_steps=(32, 33), min_per_class=20):
    """Unweighted logistic recalibration of native estimator decision scores.

    Base estimator stays frozen. This fixed, lightly regularized Platt-style
    transform uses only the separate calibration window; it is not tuned on audit.
    """
    labels, _ = validation_window(predictions, allowed_steps)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be finite and aligned with validation rows")
    if min(np.bincount(labels, minlength=2)) < min_per_class:
        raise ValueError("Calibration window has too few observations per class")
    calibrator = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000, tol=1e-8)
    calibrator.fit(scores.reshape(-1, 1), labels)
    return calibrator


def apply_sigmoid(calibrator, scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a finite vector")
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
