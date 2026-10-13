from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml
from torch import nn
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from .diagnostics import attention_summary, calibration_metrics
from .metrics import classification_metrics, event_metrics, select_threshold
from .models import build_model
from .objectives import build_binary_objective
from .publication_data import PublicationWindowDataset, prepare_publication_datasets
from .reproducibility import capture_rng_state, restore_rng_state, seed_everything


@dataclass
class PublicationRunSummary:
    run_id: str
    status: str
    development_only: bool
    fold: int
    seed: int
    device: str
    best_epoch: int
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    elapsed_seconds: float


def environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def limited_balanced_subset(dataset: Dataset, limit: int, seed: int) -> Subset:
    if limit >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    targets = subset_targets(dataset)
    random = np.random.default_rng(seed)
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    positive_count = min(len(positive), limit // 2)
    negative_count = min(len(negative), limit - positive_count)
    remaining = limit - positive_count - negative_count
    if remaining:
        positive_count += min(remaining, len(positive) - positive_count)
        remaining = limit - positive_count - negative_count
        negative_count += min(remaining, len(negative) - negative_count)
    indexes = np.concatenate(
        [
            random.choice(positive, positive_count, replace=False),
            random.choice(negative, negative_count, replace=False),
        ]
    )
    random.shuffle(indexes)
    return Subset(dataset, indexes.tolist())


def subset_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        return subset_targets(dataset.dataset)[np.asarray(dataset.indices)]
    if isinstance(dataset, PublicationWindowDataset):
        return dataset.targets()
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets())
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def subset_groups(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        return subset_groups(dataset.dataset)[np.asarray(dataset.indices)]
    if isinstance(dataset, PublicationWindowDataset):
        return np.asarray(dataset.sample_groups)
    if hasattr(dataset, "groups"):
        return np.asarray(dataset.groups())
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    labels, probabilities, timestamps, groups = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)["logits"]
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch.y.cpu().numpy().tolist())
            timestamps.extend(batch.timestamp.cpu().numpy().tolist())
            groups.extend(batch.group.cpu().numpy().tolist())
    return {
        "labels": np.asarray(labels),
        "probabilities": np.asarray(probabilities),
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "groups": np.asarray(groups, dtype=np.int64),
    }


def build_publication_loaders(
    datasets: dict[str, Dataset],
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, DataLoader]:
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, generator=generator
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False
        ),
    }
    if "test" in datasets:
        loaders["test"] = DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False
        )
    return loaders


def run_publication_experiment(
    config_path: Path,
    output_dir: Path,
    fold: int,
    seed: int,
    epochs: int | None = None,
    max_windows_per_split: int | None = None,
    resume: bool = True,
) -> PublicationRunSummary:
    requested_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if requested_config["model"].get("name") == "usad":
        from .usad_experiment import run_usad_publication_experiment

        return run_usad_publication_experiment(
            config_path, output_dir, fold, seed, epochs, max_windows_per_split, resume
        )
    if requested_config["model"].get("name") in {
        "majority",
        "logistic_regression",
        "random_forest",
    }:
        from .classical_experiment import run_classical_publication_experiment

        return run_classical_publication_experiment(
            config_path, output_dir, fold, seed, max_windows_per_split
        )
    started = time.time()
    config = requested_config
    audit = json.loads(Path(config["data"]["official_audit_manifest"]).read_text())
    if not audit["classification"]["gate1_pass"]:
        raise RuntimeError("Publication training remains locked because Gate 1 did not pass")
    config["experiment"]["seed"] = seed
    config["experiment"]["fold"] = fold
    if epochs is not None:
        config["training"]["epochs"] = epochs
    pipeline_pilot = max_windows_per_split is not None
    development_only = pipeline_pilot or bool(
        config["experiment"].get("development_only", False)
    )
    seed_everything(
        seed,
        deterministic_algorithms=bool(
            config["training"].get("deterministic_algorithms", False)
        ),
        deterministic_warn_only=bool(
            config["training"].get("deterministic_warn_only", False)
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment(), indent=2), encoding="utf-8"
    )
    full_datasets, feature_names, edge_index, scaler = prepare_publication_datasets(config, fold)
    evaluate_test = bool(config.get("evaluation", {}).get("evaluate_test", True))
    enabled_roles = ("train", "validation", "test") if evaluate_test else ("train", "validation")
    datasets: dict[str, Dataset] = {
        role: full_datasets[role] for role in enabled_roles
    }
    if pipeline_pilot:
        datasets = {
            role: limited_balanced_subset(dataset, max_windows_per_split, seed + index)
            for index, (role, dataset) in enumerate(datasets.items())
        }
    (output_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "test_evaluation_enabled": evaluate_test,
                "full_windows": {role: len(dataset) for role, dataset in full_datasets.items()},
                "used_windows": {role: len(dataset) for role, dataset in datasets.items()},
                "used_label_counts": {
                    role: {
                        "normal": int((subset_targets(dataset) == 0).sum()),
                        "attack": int((subset_targets(dataset) == 1).sum()),
                    }
                    for role, dataset in datasets.items()
                },
                "graph_edges": int(edge_index.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["training"]["batch_size"])
    loaders = build_publication_loaders(datasets, batch_size, generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], num_nodes=len(feature_names)).to(device)
    train_targets = subset_targets(datasets["train"])
    positive_weight = float((train_targets == 0).sum() / max((train_targets == 1).sum(), 1))
    criterion, objective_metadata = build_binary_objective(
        config["training"],
        train_targets,
        subset_groups(datasets["train"]),
        device,
    )
    if not np.isclose(objective_metadata["positive_weight"], positive_weight):
        raise RuntimeError("Objective positive weight does not match training labels")
    training_group_names = full_datasets["train"].group_names
    objective_metadata["positive_event_counts"] = {
        training_group_names[int(group)]: count
        for group, count in objective_metadata["positive_event_counts"].items()
    }
    objective_metadata["positive_event_factors"] = {
        training_group_names[int(group)]: factor
        for group, factor in objective_metadata["positive_event_factors"].items()
    }
    (output_dir / "objective_summary.json").write_text(
        json.dumps(objective_metadata, indent=2), encoding="utf-8"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    total_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    monitor = str(config["training"].get("monitor", "validation_pr_auc"))
    supported_monitors = {
        "validation_pr_auc",
        "validation_macro_scenario_pr_auc",
    }
    if monitor not in supported_monitors:
        raise ValueError(
            f"Unsupported training monitor {monitor!r}; expected one of "
            f"{sorted(supported_monitors)}"
        )
    validation_group_names = full_datasets["validation"].group_names
    best_score, best_epoch, stale_epochs = -np.inf, 0, 0
    best_validation_probabilities: np.ndarray | None = None
    history = []
    start_epoch = 1
    last_path = output_dir / "last.pt"
    if resume and last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=False)
        if last["fold"] != fold or last["seed"] != seed:
            raise RuntimeError("Partial checkpoint fold/seed does not match requested run")
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        history = last["history"]
        best_score = last["best_score"]
        best_epoch = last["best_epoch"]
        stale_epochs = last["stale_epochs"]
        best_validation_probabilities = last["best_validation_probabilities"]
        restore_rng_state(last["rng_state"], generator)
        start_epoch = int(last["epoch"]) + 1
        if stale_epochs >= patience:
            start_epoch = total_epochs + 1
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            loss = criterion(logits, batch.y, batch.group)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item() * batch.num_graphs
        validation = predict(model, loaders["validation"], device)
        validation_pr_auc = classification_metrics(
            validation["labels"], validation["probabilities"], 0.5
        ).get("pr_auc", 0.0)
        validation_events = event_metrics(
            validation["labels"],
            validation["probabilities"],
            validation["timestamps"],
            np.asarray(
                [validation_group_names[index] for index in validation["groups"]]
            ),
            0.5,
        )
        validation_macro_scenario_pr_auc = float(
            validation_events["macro_scenario_pr_auc"] or 0.0
        )
        monitor_values = {
            "validation_pr_auc": validation_pr_auc,
            "validation_macro_scenario_pr_auc": validation_macro_scenario_pr_auc,
        }
        validation_score = monitor_values[monitor]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(datasets["train"]),
                "validation_pr_auc": validation_pr_auc,
                "validation_macro_scenario_pr_auc": validation_macro_scenario_pr_auc,
                "monitor_score": validation_score,
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if validation_score > best_score:
            best_score, best_epoch, stale_epochs = validation_score, epoch, 0
            best_validation_probabilities = validation["probabilities"].copy()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "feature_names": feature_names,
                    "edge_index": edge_index,
                    "scaler_minimum": scaler.minimum,
                    "scaler_maximum": scaler.maximum,
                    "fold": fold,
                    "seed": seed,
                    "max_windows_per_split": max_windows_per_split,
                    "config": config,
                },
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "best_validation_probabilities": best_validation_probabilities,
                "fold": fold,
                "seed": seed,
                "rng_state": capture_rng_state(generator),
            },
            last_path,
        )
        if stale_epochs >= patience:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = predict(model, loaders["validation"], device)
    if best_validation_probabilities is None or not np.allclose(
        validation["probabilities"], best_validation_probabilities, rtol=1e-4, atol=2e-6
    ):
        raise RuntimeError("Checkpoint reload did not reproduce validation predictions")
    threshold = select_threshold(validation["labels"], validation["probabilities"])
    validation_metrics = classification_metrics(
        validation["labels"], validation["probabilities"], threshold
    )
    validation_event_names = full_datasets["validation"].group_names
    validation_event_metrics = event_metrics(
        validation["labels"],
        validation["probabilities"],
        validation["timestamps"],
        np.asarray(
            [validation_event_names[index] for index in validation["groups"]]
        ),
        threshold,
    )
    (output_dir / "validation_event_metrics.json").write_text(
        json.dumps(validation_event_metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(validation["timestamps"], unit="s"),
            "event_id": [
                validation_event_names[index] for index in validation["groups"]
            ],
            "label": validation["labels"].astype(int),
            "probability": validation["probabilities"],
            "prediction": (validation["probabilities"] >= threshold).astype(int),
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    (output_dir / "validation_calibration.json").write_text(
        json.dumps(
            calibration_metrics(validation["labels"], validation["probabilities"]),
            indent=2,
        ),
        encoding="utf-8",
    )
    test_metrics: dict[str, float] = {}
    if evaluate_test:
        test = predict(model, loaders["test"], device)
        test_metrics = classification_metrics(
            test["labels"], test["probabilities"], threshold
        )
        event_names = full_datasets["test"].group_names
        test_event_metrics = event_metrics(
            test["labels"],
            test["probabilities"],
            test["timestamps"],
            np.asarray([event_names[index] for index in test["groups"]]),
            threshold,
        )
        (output_dir / "event_metrics.json").write_text(
            json.dumps(test_event_metrics, indent=2), encoding="utf-8"
        )
        (output_dir / "test_calibration.json").write_text(
            json.dumps(
                calibration_metrics(test["labels"], test["probabilities"]),
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(test["timestamps"], unit="s"),
                "event_id": [event_names[index] for index in test["groups"]],
                "label": test["labels"].astype(int),
                "probability": test["probabilities"],
                "prediction": (test["probabilities"] >= threshold).astype(int),
            }
        ).to_csv(output_dir / "predictions.csv", index=False)
    if (
        config["model"].get("name") == "ms_res_stgat"
        and config["model"].get("spatial_encoder", "gatv2") == "gatv2"
    ):
        attention_dataset = limited_balanced_subset(
            datasets["validation"], min(64, len(datasets["validation"])), seed=0
        )
        attention_loader = DataLoader(
            attention_dataset,
            batch_size=min(batch_size, len(attention_dataset)),
            shuffle=False,
        )
        (output_dir / "attention_summary.json").write_text(
            json.dumps(
                attention_summary(model, attention_loader, device),
                indent=2,
            ),
            encoding="utf-8",
        )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "monitor": monitor,
                "best_monitor_score": float(best_score),
                "best_epoch": best_epoch,
                "epochs_completed": len(history),
                "test_evaluation_enabled": evaluate_test,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": (
                    torch.is_deterministic_algorithms_warn_only_enabled()
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = PublicationRunSummary(
        run_id=output_dir.name,
        status=(
            "COMPUTE_CALIBRATION_COMPLETED"
            if config["experiment"].get("purpose") == "compute_calibration"
            else (
                "PUBLICATION_PIPELINE_PILOT_PASSED"
                if pipeline_pilot
                else "PUBLICATION_FOLD_COMPLETED"
            )
        ),
        development_only=development_only,
        fold=fold,
        seed=seed,
        device=str(device),
        best_epoch=best_epoch,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        elapsed_seconds=time.time() - started,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    registry_path = output_dir.parent / "run_registry.csv"
    registry_row = pd.DataFrame(
        [
            {
                "run_id": summary.run_id,
                "status": summary.status,
                "development_only": summary.development_only,
                "fold": fold,
                "seed": seed,
                "epochs": total_epochs,
                "best_epoch": best_epoch,
                "elapsed_seconds": summary.elapsed_seconds,
                "metrics_path": str((output_dir / "metrics.json").resolve()),
            }
        ]
    )
    if registry_path.exists():
        previous = pd.read_csv(registry_path)
        previous = previous[previous["run_id"] != summary.run_id]
        registry_row = pd.concat([previous, registry_row], ignore_index=True)
    registry_row.to_csv(registry_path, index=False)
    return summary
