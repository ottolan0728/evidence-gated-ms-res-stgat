from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .metrics import classification_metrics, select_threshold
from .models import build_model
from .publication_data import prepare_publication_datasets
from .publication_experiment import limited_balanced_subset, predict
from .reproducibility import seed_everything


def evaluate_publication_checkpoint(checkpoint_path: Path, output_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    fold = int(checkpoint["fold"])
    seed = int(checkpoint["seed"])
    limit = checkpoint.get("max_windows_per_split")
    seed_everything(seed)
    full_datasets, feature_names, edge_index, scaler = prepare_publication_datasets(config, fold)
    if feature_names != checkpoint["feature_names"]:
        raise RuntimeError("Checkpoint feature schema differs from processed data")
    if not torch.equal(edge_index.cpu(), checkpoint["edge_index"].cpu()):
        raise RuntimeError("Rebuilt train-only graph differs from checkpoint")
    if not np.allclose(scaler.minimum, checkpoint["scaler_minimum"]):
        raise RuntimeError("Rebuilt scaler minimum differs from checkpoint")
    if not np.allclose(scaler.maximum, checkpoint["scaler_maximum"]):
        raise RuntimeError("Rebuilt scaler maximum differs from checkpoint")
    datasets = full_datasets
    if limit is not None:
        datasets = {
            role: limited_balanced_subset(dataset, int(limit), seed + index)
            for index, (role, dataset) in enumerate(full_datasets.items())
        }
    batch_size = int(config["training"]["batch_size"])
    loaders = {
        role: DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for role, dataset in datasets.items()
        if role in {"validation", "test"}
    }
    model = build_model(config["model"], num_nodes=len(feature_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    validation = predict(model, loaders["validation"], device)
    threshold = select_threshold(validation["labels"], validation["probabilities"])
    test = predict(model, loaders["test"], device)
    metrics = {
        "validation": classification_metrics(
            validation["labels"], validation["probabilities"], threshold
        ),
        "test": classification_metrics(test["labels"], test["probabilities"], threshold),
    }
    predictions_path = checkpoint_path.parent / "predictions.csv"
    saved = pd.read_csv(predictions_path)
    reproduced = len(saved) == len(test["probabilities"]) and np.allclose(
        saved["probability"].to_numpy(), test["probabilities"], rtol=1e-6, atol=1e-7
    )
    result = {
        "status": "VERIFIED" if reproduced else "FAILED",
        "checkpoint": str(checkpoint_path.resolve()),
        "fold": fold,
        "seed": seed,
        "threshold": threshold,
        "prediction_rows": len(test["probabilities"]),
        "saved_predictions_reproduced": reproduced,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not reproduced:
        raise RuntimeError("Independent evaluation did not reproduce saved predictions")
    return result
