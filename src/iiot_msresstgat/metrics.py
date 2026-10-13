import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(probabilities)
    scores = [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    result = {
        "threshold": threshold,
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
    }
    if len(np.unique(labels)) == 2:
        tn, fp, _, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        result["specificity"] = tn / max(tn + fp, 1)
        result["roc_auc"] = roc_auc_score(labels, probabilities)
        result["pr_auc"] = average_precision_score(labels, probabilities)
    return {key: float(value) for key, value in result.items()}


def event_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    sample_period_seconds: float = 1.0,
) -> dict:
    """Compute event detection and operational metrics on chronological windows."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    groups = np.asarray(groups)
    predictions = probabilities >= threshold
    event_delays: dict[str, float | None] = {}
    scenario_pr_auc: dict[str, float] = {}
    detected = 0
    attacked_groups = []
    for group in np.unique(groups):
        mask = groups == group
        group_labels = labels[mask]
        if not np.any(group_labels == 1):
            continue
        key = str(group)
        attacked_groups.append(group)
        group_probabilities = probabilities[mask]
        group_timestamps = timestamps[mask]
        attack_mask = group_labels == 1
        detected_mask = attack_mask & (group_probabilities >= threshold)
        if np.any(detected_mask):
            detected += 1
            first_attack = int(group_timestamps[attack_mask].min())
            first_detection = int(group_timestamps[detected_mask].min())
            event_delays[key] = float(max(first_detection - first_attack, 0))
        else:
            event_delays[key] = None
        if len(np.unique(group_labels)) == 2:
            scenario_pr_auc[key] = float(
                average_precision_score(group_labels, group_probabilities)
            )
    normal_count = int(np.sum(labels == 0))
    false_positives = int(np.sum((labels == 0) & predictions))
    normal_hours = normal_count * sample_period_seconds / 3600.0
    observed_delays = [value for value in event_delays.values() if value is not None]
    scenario_scores = list(scenario_pr_auc.values())
    return {
        "event_count": len(attacked_groups),
        "detected_event_count": detected,
        "event_recall": float(detected / len(attacked_groups)) if attacked_groups else 0.0,
        "detection_delay_seconds_by_event": event_delays,
        "median_detection_delay_seconds": (
            float(np.median(observed_delays)) if observed_delays else None
        ),
        "false_positives": false_positives,
        "normal_exposure_hours": float(normal_hours),
        "false_alarms_per_hour": (
            float(false_positives / normal_hours) if normal_hours > 0 else None
        ),
        "scenario_pr_auc": scenario_pr_auc,
        "macro_scenario_pr_auc": (
            float(np.mean(scenario_scores)) if scenario_scores else None
        ),
    }
