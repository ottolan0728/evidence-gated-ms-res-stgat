import numpy as np

from iiot_msresstgat.metrics import classification_metrics, event_metrics


def test_classification_metrics_include_specificity():
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9]), 0.5
    )
    assert metrics["specificity"] == 0.5


def test_event_metrics_detect_events_delays_and_false_alarm_rate():
    labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    probabilities = np.array([0.1, 0.7, 0.4, 0.8, 0.1, 0.2, 0.3, 0.1])
    timestamps = np.arange(100, 108)
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    metrics = event_metrics(labels, probabilities, timestamps, groups, 0.5)
    assert metrics["event_count"] == 2
    assert metrics["detected_event_count"] == 1
    assert metrics["event_recall"] == 0.5
    assert metrics["detection_delay_seconds_by_event"]["a"] == 1.0
    assert metrics["detection_delay_seconds_by_event"]["b"] is None
    assert metrics["false_positives"] == 1
    assert metrics["false_alarms_per_hour"] == 900.0
