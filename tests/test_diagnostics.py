import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from iiot_msresstgat.diagnostics import (
    attention_summary,
    calibration_metrics,
    expected_calibration_error,
)
from iiot_msresstgat.models import MSResSTGAT


def test_calibration_metrics_are_zero_for_perfect_probabilities():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, probabilities) == 0.0
    metrics = calibration_metrics(labels, probabilities)
    assert metrics["brier_score"] == 0.0
    assert metrics["ece_15"] == 0.0


def test_attention_summary_uses_fixed_graph_batch():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    loader = DataLoader(
        [
            Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(0.0)),
            Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(1.0)),
        ],
        batch_size=2,
        shuffle=False,
    )
    model = MSResSTGAT(hidden_dim=16, heads=4, kernels=(3, 5), dropout=0.0)
    result = attention_summary(model, loader, torch.device("cpu"))
    assert result["windows"] == 2
    assert 0.0 <= result["attention_normalized_entropy_mean"] <= 1.0
