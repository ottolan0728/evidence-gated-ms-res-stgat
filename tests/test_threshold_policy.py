import numpy as np

from iiot_msresstgat.threshold_policy import (
    apply_k_of_w_persistence,
    alarm_episode_metrics,
    select_false_alarm_constrained_threshold,
    select_temporal_constrained_threshold,
    timestamps_to_unix_seconds,
)


def test_timestamps_to_unix_seconds_is_resolution_independent():
    expected = np.array([1451299799, 1451299800], dtype=np.int64)
    strings = np.array(["2015-12-28 10:49:59", "2015-12-28 10:50:00"])
    microseconds = strings.astype("datetime64[us]")
    nanoseconds = strings.astype("datetime64[ns]")
    assert np.array_equal(timestamps_to_unix_seconds(strings), expected)
    assert np.array_equal(timestamps_to_unix_seconds(microseconds), expected)
    assert np.array_equal(timestamps_to_unix_seconds(nanoseconds), expected)


def test_alarm_episode_metrics_merge_close_positive_pulses():
    labels = np.zeros(4, dtype=int)
    probabilities = np.array([0.9, 0.8, 0.1, 0.7])
    timestamps = np.array([0, 10, 20, 100])
    groups = np.array(["a", "a", "a", "a"])
    result = alarm_episode_metrics(
        labels, probabilities, timestamps, groups, 0.5, merge_gap_seconds=60
    )
    assert result["alarm_episode_count"] == 2
    assert result["false_alarm_episode_count"] == 2
    assert result["false_alarm_episodes_per_hour"] == 1800.0
    assert result["normal_alarm_duty_cycle"] == 0.75


def test_constrained_threshold_prefers_attack_detection_without_false_alarm():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.8, 0.7, 0.9, 0.85])
    timestamps = np.array([0, 100, 200, 201])
    groups = np.array(["normal", "normal", "attack", "attack"])
    result = select_false_alarm_constrained_threshold(
        labels,
        probabilities,
        timestamps,
        groups,
        max_false_alarm_episodes_per_hour=0.0,
        max_normal_alarm_duty_cycle=0.0,
        merge_gap_seconds=60,
    )
    assert result["threshold"] == 0.85
    assert result["validation_event_recall"] == 1.0
    assert result["validation_window_f1"] == 1.0
    assert result["validation_false_alarm_episode_count"] == 0
    assert result["validation_normal_alarm_duty_cycle"] == 0.0


def test_continuous_alarm_cannot_game_episode_constraint():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.ones(4)
    timestamps = np.array([0, 1, 2, 3])
    groups = np.array(["scenario", "scenario", "scenario", "scenario"])
    result = alarm_episode_metrics(
        labels, probabilities, timestamps, groups, 0.5, merge_gap_seconds=60
    )
    assert result["false_alarm_episode_count"] == 1
    assert result["normal_alarm_duty_cycle"] == 1.0


def test_k_of_w_persistence_is_causal_and_group_isolated():
    probabilities = np.array([0.9, 0.9, 0.9, 0.9])
    groups = np.array(["a", "a", "b", "b"])
    alarm = apply_k_of_w_persistence(
        probabilities,
        groups,
        threshold=0.5,
        required_positive_windows=2,
        window_size=3,
    )
    assert alarm.tolist() == [False, True, False, True]


def test_temporal_selector_respects_dual_alarm_constraints():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.8, 0.1, 0.1, 0.9, 0.9, 0.9])
    timestamps = np.arange(6)
    groups = np.array(["scenario"] * 6)
    result = select_temporal_constrained_threshold(
        labels,
        probabilities,
        timestamps,
        groups,
        required_positive_windows=2,
        window_size=3,
        max_false_alarm_episodes_per_hour=0.0,
        max_normal_alarm_duty_cycle=0.0,
    )
    assert result["validation_event_recall"] == 1.0
    assert result["validation_false_alarm_episode_count"] == 0
