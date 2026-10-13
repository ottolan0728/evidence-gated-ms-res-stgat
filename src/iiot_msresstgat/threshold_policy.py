from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from .metrics import event_metrics


def alarm_episode_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    merge_gap_seconds: int = 60,
    sample_period_seconds: float = 1.0,
) -> dict[str, float | int]:
    """Count alarm episodes, merging positive pulses separated by at most a fixed gap."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    groups = np.asarray(groups)
    alarm_count = 0
    false_alarm_count = 0

    for group in np.unique(groups):
        mask = groups == group
        order = np.argsort(timestamps[mask], kind="stable")
        group_times = timestamps[mask][order]
        group_labels = labels[mask][order]
        predicted = probabilities[mask][order] >= threshold
        positive_indexes = np.flatnonzero(predicted)
        if not len(positive_indexes):
            continue

        starts = np.r_[
            0,
            np.flatnonzero(
                np.diff(group_times[positive_indexes]) > merge_gap_seconds
            )
            + 1,
        ]
        ends = np.r_[starts[1:], len(positive_indexes)]
        for start, end in zip(starts, ends, strict=True):
            episode_indexes = positive_indexes[start:end]
            alarm_count += 1
            if group_labels[episode_indexes[0]] == 0:
                false_alarm_count += 1

    normal_count = int(np.sum(labels == 0))
    false_positive_windows = int(
        np.sum((labels == 0) & (probabilities >= threshold))
    )
    normal_exposure_hours = normal_count * sample_period_seconds / 3600.0
    return {
        "alarm_episode_count": alarm_count,
        "false_alarm_episode_count": false_alarm_count,
        "false_positive_windows": false_positive_windows,
        "normal_alarm_duty_cycle": (
            float(false_positive_windows / normal_count) if normal_count else 0.0
        ),
        "normal_exposure_hours": float(normal_exposure_hours),
        "false_alarm_episodes_per_hour": (
            float(false_alarm_count / normal_exposure_hours)
            if normal_exposure_hours > 0
            else 0.0
        ),
    }


def threshold_candidates(probabilities: np.ndarray, grid_size: int = 501) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    unique = np.unique(probabilities)
    if len(unique) <= grid_size:
        candidates = unique
    else:
        candidates = np.unique(
            np.quantile(probabilities, np.linspace(0.0, 1.0, grid_size))
        )
    no_alarm_threshold = np.nextafter(float(probabilities.max()), np.inf)
    return np.unique(np.r_[candidates, no_alarm_threshold])


def apply_k_of_w_persistence(
    probabilities: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    required_positive_windows: int,
    window_size: int,
) -> np.ndarray:
    """Apply a causal k-of-w alarm rule without carrying history across groups."""
    if not 1 <= required_positive_windows <= window_size:
        raise ValueError("required_positive_windows must be between 1 and window_size")
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups)
    raw_positive = probabilities >= threshold
    alarm = np.zeros(len(probabilities), dtype=bool)
    for group in np.unique(groups):
        indexes = np.flatnonzero(groups == group)
        values = raw_positive[indexes].astype(np.int64)
        cumulative = np.r_[0, np.cumsum(values)]
        ends = np.arange(1, len(values) + 1)
        starts = np.maximum(ends - window_size, 0)
        counts = cumulative[ends] - cumulative[starts]
        alarm[indexes] = counts >= required_positive_windows
    return alarm


def select_temporal_constrained_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    groups: np.ndarray,
    required_positive_windows: int,
    window_size: int,
    max_false_alarm_episodes_per_hour: float = 1.0,
    max_normal_alarm_duty_cycle: float = 0.01,
    merge_gap_seconds: int = 60,
    grid_size: int = 201,
) -> dict[str, float | int]:
    """Select a validation-only threshold for one fixed temporal alarm rule."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = threshold_candidates(probabilities, grid_size)
    best: tuple[tuple[float, float, float, float, float], dict] | None = None

    for threshold in candidates:
        alarm = apply_k_of_w_persistence(
            probabilities,
            groups,
            float(threshold),
            required_positive_windows,
            window_size,
        )
        alarm_scores = alarm.astype(float)
        alarms = alarm_episode_metrics(
            labels,
            alarm_scores,
            timestamps,
            groups,
            0.5,
            merge_gap_seconds,
        )
        false_alarm_rate = float(alarms["false_alarm_episodes_per_hour"])
        normal_alarm_duty_cycle = float(alarms["normal_alarm_duty_cycle"])
        if (
            false_alarm_rate > max_false_alarm_episodes_per_hour
            or normal_alarm_duty_cycle > max_normal_alarm_duty_cycle
        ):
            continue
        events = event_metrics(labels, alarm_scores, timestamps, groups, 0.5)
        window_f1 = float(f1_score(labels, alarm, zero_division=0))
        delay = events["median_detection_delay_seconds"]
        delay_score = -float(delay) if delay is not None else -np.inf
        score = (
            float(events["event_recall"]),
            window_f1,
            delay_score,
            -normal_alarm_duty_cycle,
            float(threshold),
        )
        result = {
            "threshold": float(threshold),
            "required_positive_windows": required_positive_windows,
            "window_size": window_size,
            "validation_event_recall": float(events["event_recall"]),
            "validation_window_f1": window_f1,
            "validation_median_detection_delay_seconds": delay,
            "validation_false_alarm_episodes_per_hour": false_alarm_rate,
            "validation_false_alarm_episode_count": int(
                alarms["false_alarm_episode_count"]
            ),
            "validation_normal_alarm_duty_cycle": normal_alarm_duty_cycle,
            "candidate_count": int(len(candidates)),
        }
        if best is None or score > best[0]:
            best = (score, result)

    if best is None:
        raise RuntimeError("No feasible temporal threshold found")
    return best[1]


def select_false_alarm_constrained_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    timestamps: np.ndarray,
    groups: np.ndarray,
    max_false_alarm_episodes_per_hour: float = 1.0,
    max_normal_alarm_duty_cycle: float = 0.01,
    merge_gap_seconds: int = 60,
    grid_size: int = 501,
) -> dict[str, float | int]:
    """Select a validation-only threshold under an alarm-episode rate constraint.

    Lexicographic objective among feasible thresholds:
    event recall, window F1, lower normal alarm duty cycle, then higher threshold.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = threshold_candidates(probabilities, grid_size)
    best: tuple[tuple[float, float, float, float], dict[str, float | int]] | None = None

    for threshold in candidates:
        alarms = alarm_episode_metrics(
            labels,
            probabilities,
            timestamps,
            groups,
            float(threshold),
            merge_gap_seconds,
        )
        false_alarm_rate = float(alarms["false_alarm_episodes_per_hour"])
        normal_alarm_duty_cycle = float(alarms["normal_alarm_duty_cycle"])
        if (
            false_alarm_rate > max_false_alarm_episodes_per_hour
            or normal_alarm_duty_cycle > max_normal_alarm_duty_cycle
        ):
            continue
        events = event_metrics(labels, probabilities, timestamps, groups, float(threshold))
        window_f1 = float(
            f1_score(labels, probabilities >= threshold, zero_division=0)
        )
        score = (
            float(events["event_recall"]),
            window_f1,
            -normal_alarm_duty_cycle,
            float(threshold),
        )
        result = {
            "threshold": float(threshold),
            "validation_event_recall": float(events["event_recall"]),
            "validation_window_f1": window_f1,
            "validation_false_alarm_episodes_per_hour": false_alarm_rate,
            "validation_false_alarm_episode_count": int(
                alarms["false_alarm_episode_count"]
            ),
            "validation_normal_alarm_duty_cycle": normal_alarm_duty_cycle,
            "candidate_count": int(len(candidates)),
        }
        if best is None or score > best[0]:
            best = (score, result)

    if best is None:
        raise RuntimeError("No feasible threshold found; no-alarm candidate was expected")
    return best[1]

def timestamps_to_unix_seconds(values) -> np.ndarray:
    """Convert timestamp-like values to Unix seconds independent of input resolution."""
    return np.asarray(values, dtype="datetime64[s]").astype(np.int64)

