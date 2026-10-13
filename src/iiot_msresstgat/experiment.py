from __future__ import annotations

import json
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
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data import load_swat_csv, make_windows
from .graph import build_graph
from .metrics import classification_metrics, select_threshold
from .models import MSResSTGAT
from .reproducibility import seed_everything


@dataclass
class RunSummary:
    run_id: str
    status: str
    development_only: bool
    device: str
    best_epoch: int
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    elapsed_seconds: float


def _partition(values: np.ndarray, labels: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_end = int(len(values) * 0.6)
    validation_end = int(len(values) * 0.8)
    return {
        "train": (values[:train_end], labels[:train_end]),
        "validation": (values[train_end:validation_end], labels[train_end:validation_end]),
        "test": (values[validation_end:], labels[validation_end:]),
    }


def prepare_development_data(config: dict, max_rows_per_class: int):
    """Build a balanced smoke-test dataset without crossing source-file boundaries."""
    data_config = config["data"]
    normal, normal_labels = load_swat_csv(
        data_config["normal_csv"], data_config["label_column"], max_rows_per_class
    )
    attack, attack_labels = load_swat_csv(
        data_config["attack_csv"], data_config["label_column"], max_rows_per_class
    )
    if list(normal.columns) != list(attack.columns):
        raise ValueError("Normal and attack feature schemas differ")
    normal_parts = _partition(normal.to_numpy(np.float64), normal_labels)
    attack_parts = _partition(attack.to_numpy(np.float64), attack_labels)
    scaler = MinMaxScaler().fit(
        np.concatenate([normal_parts["train"][0], attack_parts["train"][0]])
    )
    train_values = np.concatenate(
        [normal_parts["train"][0], attack_parts["train"][0]], axis=0
    )
    edge_index = build_graph(
        scaler.transform(train_values),
        list(normal.columns),
        config["graph"]["correlation_threshold"],
        config["graph"]["adjacent_stages_only"],
    )
    datasets: dict[str, list[Data]] = {}
    window_size = int(data_config["window_size"])
    stride = int(data_config["stride"])
    for split_name in ("train", "validation", "test"):
        split_graphs: list[Data] = []
        for values, labels in (normal_parts[split_name], attack_parts[split_name]):
            windows, targets = make_windows(
                scaler.transform(values), labels, window_size, stride
            )
            split_graphs.extend(
                Data(
                    x=torch.from_numpy(window),
                    edge_index=edge_index,
                    y=torch.tensor(target, dtype=torch.float32),
                )
                for window, target in zip(windows, targets, strict=True)
            )
        datasets[split_name] = split_graphs
    return datasets, list(normal.columns), edge_index, scaler


def _predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)["logits"]
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch.y.cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(probabilities)


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


def run_development_smoke(
    config_path: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    max_rows_per_class: int,
) -> RunSummary:
    started = time.time()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"]["seed"] = seed
    config["training"]["epochs"] = epochs
    config["data"]["normal_csv"] = str(Path(config["data"]["normal_csv"]).resolve())
    config["data"]["attack_csv"] = str(Path(config["data"]["attack_csv"]).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2), encoding="utf-8"
    )
    seed_everything(seed)
    datasets, feature_names, edge_index, _ = prepare_development_data(
        config, max_rows_per_class
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["training"]["batch_size"])
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, generator=generator
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False
        ),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = MSResSTGAT(
        hidden_dim=int(model_config["hidden_dim"]),
        heads=int(model_config["heads"]),
        kernels=tuple(model_config["kernels"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    train_targets = np.asarray([float(graph.y) for graph in datasets["train"]])
    positive_weight = float((train_targets == 0).sum() / max((train_targets == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_score, best_epoch = -np.inf, 0
    best_validation_probabilities: np.ndarray | None = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item() * batch.num_graphs
        validation_labels, validation_probabilities = _predict(
            model, loaders["validation"], device
        )
        validation_score = classification_metrics(
            validation_labels, validation_probabilities, 0.5
        ).get("pr_auc", 0.0)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(datasets["train"]),
            "validation_pr_auc": validation_score,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if validation_score > best_score:
            best_score, best_epoch = validation_score, epoch
            best_validation_probabilities = validation_probabilities.copy()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "feature_names": feature_names,
                    "edge_index": edge_index,
                    "config": config,
                },
                output_dir / "best.pt",
            )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_labels, validation_probabilities = _predict(
        model, loaders["validation"], device
    )
    if best_validation_probabilities is None or not np.allclose(
        validation_probabilities, best_validation_probabilities, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("Checkpoint reload did not reproduce validation predictions")
    threshold = select_threshold(validation_labels, validation_probabilities)
    validation_metrics = classification_metrics(
        validation_labels, validation_probabilities, threshold
    )
    test_labels, test_probabilities = _predict(model, loaders["test"], device)
    test_metrics = classification_metrics(test_labels, test_probabilities, threshold)
    pd.DataFrame(
        {
            "sample_index": np.arange(len(test_labels)),
            "label": test_labels.astype(int),
            "probability": test_probabilities,
            "prediction": (test_probabilities >= threshold).astype(int),
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    summary = RunSummary(
        run_id=output_dir.name,
        status="DEVELOPMENT_SMOKE_PASSED",
        development_only=True,
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
                "device": summary.device,
                "seed": seed,
                "epochs": epochs,
                "best_epoch": summary.best_epoch,
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
