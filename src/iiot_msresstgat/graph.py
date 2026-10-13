import re

import numpy as np
import torch


def swat_stage(sensor_name: str) -> int | None:
    match = re.search(r"[1-6]", sensor_name)
    return int(match.group()) if match else None


def build_graph(
    train_values: np.ndarray,
    feature_names: list[str],
    correlation_threshold: float = 0.35,
    adjacent_stages_only: bool = True,
) -> torch.Tensor:
    """Build a directed graph using training data only."""
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.nan_to_num(np.corrcoef(train_values.T))
    return build_graph_from_correlation(
        correlation, feature_names, correlation_threshold, adjacent_stages_only
    )


def build_graph_from_correlation(
    correlation: np.ndarray,
    feature_names: list[str],
    correlation_threshold: float = 0.35,
    adjacent_stages_only: bool = True,
) -> torch.Tensor:
    if correlation.shape != (len(feature_names), len(feature_names)):
        raise ValueError("Correlation matrix does not match feature names")
    edges: list[tuple[int, int]] = []
    for source in range(len(feature_names)):
        for target in range(len(feature_names)):
            if source == target or abs(correlation[source, target]) < correlation_threshold:
                continue
            source_stage, target_stage = swat_stage(feature_names[source]), swat_stage(feature_names[target])
            physically_valid = (
                not adjacent_stages_only
                or source_stage is None
                or target_stage is None
                or abs(source_stage - target_stage) <= 1
            )
            if physically_valid:
                edges.append((source, target))
    if not edges:
        raise ValueError("Graph has no edges; relax the graph constraints")
    return torch.tensor(edges, dtype=torch.long).T.contiguous()
