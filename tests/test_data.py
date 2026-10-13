import numpy as np
import pandas as pd

from iiot_msresstgat.data import (
    chronological_split,
    fit_transform_splits,
    make_windows,
    parse_binary_labels,
)


def test_scaler_is_fit_on_training_partition_only():
    values = np.arange(20, dtype=float).reshape(10, 2)
    split = chronological_split(10, 0.6, 0.2)
    transformed, scaler = fit_transform_splits(values, split)
    assert np.allclose(scaler.data_max_, [10.0, 11.0])
    assert transformed["test"].max() > 1.0


def test_windows_do_not_cross_partition_boundaries():
    values = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.arange(6)
    windows, targets = make_windows(values, labels, window_size=3)
    assert windows.shape == (4, 2, 3)
    assert targets.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_parse_binary_labels_normalizes_provider_spacing():
    labels = pd.Series(["Normal", "Attack", "A ttack", "  attack  "])

    assert parse_binary_labels(labels).tolist() == [0, 1, 1, 1]
