from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ObjectiveSpecification:
    loss_name: str
    positive_weight: float
    event_balance: bool
    focal_gamma: float
    event_factor_clip: tuple[float, float]


def positive_event_factors(
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    clip: tuple[float, float] = (0.25, 4.0),
) -> np.ndarray:
    """Return group-indexed positive factors with positive-window weighted mean one."""
    targets = np.asarray(targets, dtype=np.uint8)
    groups = np.asarray(groups, dtype=np.int64)
    if targets.shape != groups.shape:
        raise ValueError("targets and groups must have the same shape")
    if len(groups) == 0 or np.any(groups < 0):
        raise ValueError("groups must be a non-empty array of non-negative indexes")
    if clip[0] <= 0 or clip[0] > clip[1]:
        raise ValueError("event factor clip must satisfy 0 < minimum <= maximum")

    factors = np.ones(int(groups.max()) + 1, dtype=np.float64)
    positive_groups = groups[targets == 1]
    if len(positive_groups) == 0:
        return factors
    unique, counts = np.unique(positive_groups, return_counts=True)
    raw = counts.mean() / counts
    clipped = np.clip(raw, clip[0], clip[1])
    normalized = clipped / np.average(clipped, weights=counts)
    factors[unique] = normalized
    return factors


class BinaryObjective(nn.Module):
    """Weighted BCE/focal objective with optional positive-event balancing."""

    def __init__(
        self,
        specification: ObjectiveSpecification,
        event_factors: np.ndarray,
    ) -> None:
        super().__init__()
        if specification.loss_name not in {"weighted_bce", "focal"}:
            raise ValueError(
                "loss_name must be one of ['focal', 'weighted_bce'], "
                f"got {specification.loss_name!r}"
            )
        if specification.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        self.specification = specification
        self.register_buffer(
            "positive_weight",
            torch.tensor(float(specification.positive_weight), dtype=torch.float32),
        )
        self.register_buffer(
            "event_factors",
            torch.as_tensor(event_factors, dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        unweighted_ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        class_factors = torch.where(
            targets == 1,
            self.positive_weight.to(dtype=logits.dtype),
            torch.ones((), device=logits.device, dtype=logits.dtype),
        )
        losses = class_factors * unweighted_ce
        if self.specification.loss_name == "focal":
            probability_true_class = torch.exp(-unweighted_ce)
            losses = losses * (1.0 - probability_true_class).pow(
                self.specification.focal_gamma
            )
        if self.specification.event_balance:
            if int(groups.max()) >= len(self.event_factors):
                raise ValueError("batch contains a group without a frozen event factor")
            positive_factors = self.event_factors[groups].to(dtype=logits.dtype)
            losses = losses * torch.where(
                targets == 1,
                positive_factors,
                torch.ones((), device=logits.device, dtype=logits.dtype),
            )
        return losses.mean()


def build_binary_objective(
    training_config: dict,
    targets: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
) -> tuple[BinaryObjective, dict]:
    targets = np.asarray(targets, dtype=np.uint8)
    groups = np.asarray(groups, dtype=np.int64)
    normal = int((targets == 0).sum())
    attack = int((targets == 1).sum())
    positive_weight = normal / max(attack, 1)
    loss_name = str(training_config.get("loss_name", "weighted_bce"))
    event_balance = bool(training_config.get("event_balance", False))
    focal_gamma = float(training_config.get("focal_gamma", 0.0))
    clip_values = training_config.get("event_factor_clip", [0.25, 4.0])
    clip = (float(clip_values[0]), float(clip_values[1]))
    specification = ObjectiveSpecification(
        loss_name=loss_name,
        positive_weight=positive_weight,
        event_balance=event_balance,
        focal_gamma=focal_gamma,
        event_factor_clip=clip,
    )
    factors = (
        positive_event_factors(targets, groups, clip=clip)
        if event_balance
        else np.ones(int(groups.max()) + 1, dtype=np.float64)
    )
    objective = BinaryObjective(specification, factors).to(device)
    positive_groups, positive_counts = np.unique(groups[targets == 1], return_counts=True)
    metadata = {
        "loss_name": loss_name,
        "positive_weight": positive_weight,
        "event_balance": event_balance,
        "focal_gamma": focal_gamma,
        "event_factor_clip": list(clip),
        "positive_event_counts": {
            str(int(group)): int(count)
            for group, count in zip(positive_groups, positive_counts, strict=True)
        },
        "positive_event_factors": {
            str(int(group)): float(factors[group]) for group in positive_groups
        },
        "positive_window_weighted_event_factor_mean": (
            float(np.average(factors[positive_groups], weights=positive_counts))
            if len(positive_groups)
            else None
        ),
    }
    return objective, metadata
