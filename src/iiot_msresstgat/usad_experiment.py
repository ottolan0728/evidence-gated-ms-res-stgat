from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .metrics import classification_metrics, event_metrics, select_threshold
from .models import USADModel
from .publication_data import prepare_publication_datasets
from .reproducibility import capture_rng_state, restore_rng_state, seed_everything


def _environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _flatten_batch(batch, num_nodes: int, downsample: int) -> torch.Tensor:
    windows = batch.x.view(batch.num_graphs, num_nodes, -1).transpose(1, 2)
    if windows.shape[1] % downsample:
        raise ValueError("USAD window size must be divisible by its downsampling factor")
    reduced = windows.view(
        batch.num_graphs, windows.shape[1] // downsample, downsample, num_nodes
    ).median(dim=2).values
    return reduced.reshape(batch.num_graphs, -1)


def normal_only_subset(dataset) -> Subset:
    indexes = np.flatnonzero(dataset.targets() == 0).tolist()
    if not indexes:
        raise ValueError("USAD requires normal training windows")
    return Subset(dataset, indexes)


def _predict(
    model: USADModel,
    loader: DataLoader,
    device: torch.device,
    num_nodes: int,
    downsample: int,
    alpha: float,
    beta: float,
) -> dict[str, np.ndarray]:
    model.eval()
    labels, scores, timestamps, groups = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            windows = _flatten_batch(batch, num_nodes, downsample)
            scores.extend(model.anomaly_score(windows, alpha, beta).cpu().numpy().tolist())
            labels.extend(batch.y.cpu().numpy().tolist())
            timestamps.extend(batch.timestamp.cpu().numpy().tolist())
            groups.extend(batch.group.cpu().numpy().tolist())
    return {
        "labels": np.asarray(labels),
        "probabilities": np.asarray(scores),
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "groups": np.asarray(groups, dtype=np.int64),
    }


def run_usad_publication_experiment(
    config_path: Path,
    output_dir: Path,
    fold: int,
    seed: int,
    epochs: int | None,
    max_windows_per_split: int | None,
    resume: bool = True,
):
    from .publication_experiment import (
        PublicationRunSummary,
        limited_balanced_subset,
        subset_targets,
    )

    started = time.time()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = json.loads(Path(config["data"]["official_audit_manifest"]).read_text())
    if not audit["classification"]["gate1_pass"]:
        raise RuntimeError("Publication training remains locked because Gate 1 did not pass")
    config["experiment"]["seed"] = seed
    config["experiment"]["fold"] = fold
    if epochs is not None:
        config["training"]["epochs"] = epochs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2), encoding="utf-8"
    )
    seed_everything(seed)
    full_datasets, feature_names, edge_index, scaler = prepare_publication_datasets(config, fold)
    eligible_datasets = dict(full_datasets)
    eligible_datasets["train"] = normal_only_subset(full_datasets["train"])
    datasets = eligible_datasets
    pipeline_pilot = max_windows_per_split is not None
    development_only = pipeline_pilot or bool(
        config["experiment"].get("development_only", False)
    )
    if pipeline_pilot:
        datasets = {
            role: limited_balanced_subset(dataset, max_windows_per_split, seed + index)
            for index, (role, dataset) in enumerate(eligible_datasets.items())
        }
    (output_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "training_supervision": "normal_only_unsupervised",
                "eligible_windows": {
                    role: len(dataset) for role, dataset in eligible_datasets.items()
                },
                "full_windows": {role: len(dataset) for role, dataset in full_datasets.items()},
                "used_windows": {role: len(dataset) for role, dataset in datasets.items()},
                "used_label_counts": {
                    role: {
                        "normal": int((subset_targets(dataset) == 0).sum()),
                        "attack": int((subset_targets(dataset) == 1).sum()),
                    }
                    for role, dataset in datasets.items()
                },
                "graph_edges_not_used": int(edge_index.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, generator=generator
        ),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    downsample = int(config["model"].get("downsample", 5))
    input_dim = (int(config["data"]["window_size"]) // downsample) * len(feature_names)
    model = USADModel(input_dim, int(config["model"].get("latent_dim", 40))).to(device)
    optimizer1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=float(config["training"].get("learning_rate", 0.001)),
    )
    optimizer2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=float(config["training"].get("learning_rate", 0.001)),
    )
    history = []
    total_epochs = int(config["training"]["epochs"])
    start_epoch = 1
    last_path = output_dir / "last.pt"
    if resume and last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=False)
        if last["fold"] != fold or last["seed"] != seed:
            raise RuntimeError("Partial USAD checkpoint fold/seed does not match requested run")
        model.load_state_dict(last["model_state"])
        optimizer1.load_state_dict(last["optimizer1_state"])
        optimizer2.load_state_dict(last["optimizer2_state"])
        history = last["history"]
        restore_rng_state(last["rng_state"], generator)
        start_epoch = int(last["epoch"]) + 1
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total_loss1 = total_loss2 = 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            windows = _flatten_batch(batch, len(feature_names), downsample)
            optimizer1.zero_grad(set_to_none=True)
            loss1, _ = model.losses(windows, epoch)
            loss1.backward()
            optimizer1.step()
            optimizer2.zero_grad(set_to_none=True)
            _, loss2 = model.losses(windows, epoch)
            loss2.backward()
            optimizer2.step()
            total_loss1 += loss1.detach().item() * batch.num_graphs
            total_loss2 += loss2.detach().item() * batch.num_graphs
        history.append(
            {
                "epoch": epoch,
                "train_loss1": total_loss1 / len(datasets["train"]),
                "train_loss2": total_loss2 / len(datasets["train"]),
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
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
                "published_comparator": "USAD",
            },
            output_dir / "best.pt",
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer1_state": optimizer1.state_dict(),
                "optimizer2_state": optimizer2.state_dict(),
                "epoch": epoch,
                "history": history,
                "fold": fold,
                "seed": seed,
                "rng_state": capture_rng_state(generator),
            },
            last_path,
        )
    model_config = config["model"]
    alpha, beta = float(model_config.get("alpha", 0.5)), float(model_config.get("beta", 0.5))
    validation = _predict(
        model,
        loaders["validation"],
        device,
        len(feature_names),
        downsample,
        alpha,
        beta,
    )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    reloaded = USADModel(input_dim, int(model_config.get("latent_dim", 40))).to(device)
    reloaded.load_state_dict(checkpoint["model_state"])
    reproduced = _predict(
        reloaded,
        loaders["validation"],
        device,
        len(feature_names),
        downsample,
        alpha,
        beta,
    )
    if not np.allclose(
        validation["probabilities"], reproduced["probabilities"], rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("USAD checkpoint reload did not reproduce validation scores")
    threshold = select_threshold(validation["labels"], validation["probabilities"])
    validation_metrics = classification_metrics(
        validation["labels"], validation["probabilities"], threshold
    )
    test = _predict(
        reloaded,
        loaders["test"],
        device,
        len(feature_names),
        downsample,
        alpha,
        beta,
    )
    test_metrics = classification_metrics(test["labels"], test["probabilities"], threshold)
    event_names = full_datasets["test"].group_names
    mapped_groups = np.asarray([event_names[index] for index in test["groups"]])
    events = event_metrics(
        test["labels"], test["probabilities"], test["timestamps"], mapped_groups, threshold
    )
    (output_dir / "event_metrics.json").write_text(
        json.dumps(events, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test["timestamps"], unit="s"),
            "event_id": mapped_groups,
            "label": test["labels"].astype(int),
            "probability": test["probabilities"],
            "prediction": (test["probabilities"] >= threshold).astype(int),
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
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
        best_epoch=total_epochs,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        elapsed_seconds=time.time() - started,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary
