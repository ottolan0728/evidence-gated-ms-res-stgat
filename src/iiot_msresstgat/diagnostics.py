from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss
from torch import nn
from torch_geometric.loader import DataLoader


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indexes = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indexes == index
        if not np.any(selected):
            continue
        confidence = float(probabilities[selected].mean())
        observed = float(labels[selected].mean())
        error += float(selected.mean()) * abs(confidence - observed)
    return error


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped)),
        "ece_15": expected_calibration_error(labels, probabilities, bins=15),
        "probability_le_0_01": float((probabilities <= 0.01).mean()),
        "probability_ge_0_99": float((probabilities >= 0.99).mean()),
    }


def attention_summary(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float | int]:
    normalized_entropies: list[float] = []
    maximum_weights: list[float] = []
    coefficient_values: list[np.ndarray] = []
    window_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            window_count += int(batch.num_graphs)
            output = model(batch.to(device), return_attention=True)
            attention = output.get("attention")
            if attention is None:
                raise RuntimeError("Requested attention summary from a model without attention")
            edge_index, coefficients = attention
            destinations = edge_index[1].cpu().numpy()
            alpha = coefficients.cpu().numpy()
            coefficient_values.append(alpha.reshape(-1))
            for head in range(alpha.shape[1]):
                frame = pd.DataFrame(
                    {"destination": destinations, "alpha": alpha[:, head]}
                )
                for _, group in frame.groupby("destination", sort=False):
                    values = group["alpha"].to_numpy(dtype=float)
                    maximum_weights.append(float(values.max()))
                    if len(values) > 1:
                        entropy = -float(
                            np.sum(values * np.log(np.clip(values, 1e-12, 1.0)))
                        )
                        normalized_entropies.append(entropy / math.log(len(values)))
    flattened = np.concatenate(coefficient_values)
    return {
        "windows": window_count,
        "attention_alpha_mean": float(flattened.mean()),
        "attention_alpha_sd": float(flattened.std(ddof=1)),
        "attention_destination_max_mean": float(np.mean(maximum_weights)),
        "attention_normalized_entropy_mean": float(np.mean(normalized_entropies)),
    }
