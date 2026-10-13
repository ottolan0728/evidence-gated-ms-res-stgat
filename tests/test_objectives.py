import numpy as np
import torch
from torch import nn

from iiot_msresstgat.objectives import (
    build_binary_objective,
    positive_event_factors,
)


def test_control_objective_exactly_matches_weighted_bce():
    logits = torch.tensor([-2.0, -0.4, 0.7, 2.2], dtype=torch.float64)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    groups = torch.tensor([0, 1, 0, 2])
    objective, metadata = build_binary_objective(
        {
            "loss_name": "weighted_bce",
            "event_balance": False,
            "focal_gamma": 0.0,
        },
        targets.numpy(),
        groups.numpy(),
        torch.device("cpu"),
    )
    objective = objective.double()
    expected = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(metadata["positive_weight"], dtype=torch.float64)
    )(logits, targets)
    actual = objective(logits, targets, groups)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_positive_event_factors_have_weighted_mean_one_after_clipping():
    targets = np.ones(15, dtype=np.uint8)
    groups = np.array([0] + [1] * 4 + [2] * 10)
    factors = positive_event_factors(targets, groups, clip=(0.25, 4.0))
    counts = np.bincount(groups)
    assert np.isclose(np.average(factors, weights=counts), 1.0)
    assert factors[0] > factors[1] > factors[2]


def test_event_balancing_changes_only_positive_examples():
    targets = np.array([0, 0, 1, 1, 1, 1], dtype=np.uint8)
    groups = np.array([0, 1, 0, 1, 1, 1])
    objective, metadata = build_binary_objective(
        {
            "loss_name": "weighted_bce",
            "event_balance": True,
            "focal_gamma": 0.0,
            "event_factor_clip": [0.25, 4.0],
        },
        targets,
        groups,
        torch.device("cpu"),
    )
    assert metadata["positive_window_weighted_event_factor_mean"] == 1.0
    assert metadata["positive_event_factors"]["0"] > 1.0
    assert metadata["positive_event_factors"]["1"] < 1.0
    assert objective.event_factors.shape == (2,)


def test_focal_gamma_two_downweights_easy_examples():
    targets = np.array([0, 1], dtype=np.uint8)
    groups = np.array([0, 1])
    focal, _ = build_binary_objective(
        {"loss_name": "focal", "event_balance": False, "focal_gamma": 2.0},
        targets,
        groups,
        torch.device("cpu"),
    )
    bce, _ = build_binary_objective(
        {
            "loss_name": "weighted_bce",
            "event_balance": False,
            "focal_gamma": 0.0,
        },
        targets,
        groups,
        torch.device("cpu"),
    )
    logits = torch.tensor([-5.0, 5.0])
    tensor_targets = torch.tensor(targets, dtype=torch.float32)
    tensor_groups = torch.tensor(groups)
    assert focal(logits, tensor_targets, tensor_groups) < bce(
        logits, tensor_targets, tensor_groups
    )
