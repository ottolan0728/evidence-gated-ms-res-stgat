import pandas as pd
import pytest

from iiot_msresstgat.ensemble import (
    assert_prediction_alignment,
    ensemble_prediction_frame,
)


def frame(probabilities):
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:00:01"],
            "event_id": ["a", "a"],
            "label": [0, 1],
            "probability": probabilities,
        }
    )


def test_mean_and_median_ensemble_predictions():
    frames = [frame([0.1, 0.8]), frame([0.2, 0.7]), frame([0.9, 0.6])]
    assert assert_prediction_alignment(frames)["row_count"] == 2
    mean = ensemble_prediction_frame(frames, "mean")
    median = ensemble_prediction_frame(frames, "median")
    assert mean["probability"].tolist() == pytest.approx([0.4, 0.7])
    assert median["probability"].tolist() == pytest.approx([0.2, 0.7])


def test_ensemble_rejects_misaligned_labels():
    first = frame([0.1, 0.8])
    second = frame([0.2, 0.7])
    second.loc[1, "label"] = 0
    with pytest.raises(ValueError, match="label"):
        assert_prediction_alignment([first, second])
