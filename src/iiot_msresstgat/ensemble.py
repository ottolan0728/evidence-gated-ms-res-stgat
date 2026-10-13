from __future__ import annotations

import numpy as np
import pandas as pd


ALIGNMENT_COLUMNS = ("timestamp", "event_id", "label")


def assert_prediction_alignment(frames: list[pd.DataFrame]) -> dict[str, int | bool]:
    if not frames:
        raise ValueError("At least one prediction frame is required")
    reference = frames[0]
    for index, frame in enumerate(frames[1:], start=1):
        if len(frame) != len(reference):
            raise ValueError(f"Prediction frame {index} has a different row count")
        for column in ALIGNMENT_COLUMNS:
            if not np.array_equal(
                frame[column].to_numpy(), reference[column].to_numpy()
            ):
                raise ValueError(
                    f"Prediction frame {index} is not aligned on {column!r}"
                )
    return {
        "aligned": True,
        "frame_count": len(frames),
        "row_count": len(reference),
    }


def ensemble_prediction_frame(
    frames: list[pd.DataFrame], method: str
) -> pd.DataFrame:
    assert_prediction_alignment(frames)
    probabilities = np.stack(
        [frame["probability"].to_numpy(dtype=float) for frame in frames]
    )
    if method == "mean":
        ensemble = probabilities.mean(axis=0)
    elif method == "median":
        ensemble = np.median(probabilities, axis=0)
    else:
        raise ValueError("Ensemble method must be 'mean' or 'median'")
    result = frames[0][list(ALIGNMENT_COLUMNS)].copy()
    result["probability"] = ensemble
    return result
