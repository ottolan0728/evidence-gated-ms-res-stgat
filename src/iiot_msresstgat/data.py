from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LABEL_CANDIDATES = ("Normal/Attack", "Label", "Status", "NORMAL/ATTACK")


@dataclass(frozen=True)
class TimeSplit:
    train: slice
    validation: slice
    test: slice


def parse_binary_labels(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.lower().str.replace(r"\s+", "", regex=True)
    return normalized.map(lambda value: int(value == "1" or "attack" in value)).to_numpy(np.int64)


def load_swat_csv(
    path: str | Path, label_column: str | None = None, nrows: int | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, nrows=nrows)
    frame.columns = frame.columns.str.strip()
    label_column = label_column or next((c for c in LABEL_CANDIDATES if c in frame), None)
    labels = np.zeros(len(frame), dtype=np.int64) if label_column is None else parse_binary_labels(frame[label_column])
    excluded = set(LABEL_CANDIDATES) | {"Timestamp", "timestamp"}
    features = frame[[c for c in frame.columns if c not in excluded]].apply(pd.to_numeric, errors="coerce")
    features = features.ffill().bfill()
    if features.isna().any().any():
        raise ValueError(f"Unrecoverable missing values in {path}")
    return features, labels


def chronological_split(length: int, train_fraction: float, val_fraction: float) -> TimeSplit:
    if length < 3 or train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions require non-empty train, validation, and test partitions")
    train_end = int(length * train_fraction)
    val_end = train_end + int(length * val_fraction)
    return TimeSplit(slice(0, train_end), slice(train_end, val_end), slice(val_end, length))


def fit_transform_splits(
    values: np.ndarray, split: TimeSplit
) -> tuple[dict[str, np.ndarray], MinMaxScaler]:
    scaler = MinMaxScaler().fit(values[split.train])
    transformed = {
        "train": scaler.transform(values[split.train]),
        "validation": scaler.transform(values[split.validation]),
        "test": scaler.transform(values[split.test]),
    }
    return transformed, scaler


def make_windows(values: np.ndarray, labels: np.ndarray, window_size: int, stride: int = 1):
    if window_size < 1 or len(values) < window_size:
        raise ValueError("window_size must fit inside the partition")
    windows, targets = [], []
    for end in range(window_size, len(values) + 1, stride):
        windows.append(values[end - window_size : end].T.astype(np.float32))
        targets.append(labels[end - 1])
    return np.stack(windows), np.asarray(targets, dtype=np.float32)
