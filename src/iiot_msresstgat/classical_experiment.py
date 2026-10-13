from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, Subset

from .metrics import classification_metrics, event_metrics, select_threshold
from .publication_data import prepare_publication_datasets
from .reproducibility import seed_everything


CLASSICAL_MODELS = {"majority", "logistic_regression", "random_forest"}


def _dataset_items(dataset: Dataset):
    if isinstance(dataset, Subset):
        for index in dataset.indices:
            yield dataset.dataset[index]
    else:
        for index in range(len(dataset)):
            yield dataset[index]


def engineered_window_features(dataset: Dataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features, labels, timestamps, groups = [], [], [], []
    for item in _dataset_items(dataset):
        window = item.x.numpy()
        features.append(
            np.concatenate(
                [
                    window[:, -1],
                    window.mean(axis=1),
                    window.std(axis=1),
                    window.min(axis=1),
                    window.max(axis=1),
                ]
            )
        )
        labels.append(int(item.y.item()))
        timestamps.append(int(item.timestamp.item()))
        groups.append(int(item.group.item()))
    return np.asarray(features, dtype=np.float32), {
        "labels": np.asarray(labels, dtype=np.uint8),
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "groups": np.asarray(groups, dtype=np.int64),
    }


def build_classical_model(name: str, seed: int, config: dict):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=int(config.get("max_iter", 1000)),
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 200)),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classical model: {name!r}")


def positive_probabilities(model, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, classes.index(1)]


def run_classical_publication_experiment(
    config_path: Path,
    output_dir: Path,
    fold: int,
    seed: int,
    max_windows_per_split: int | None,
):
    from .publication_experiment import (
        PublicationRunSummary,
        environment,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment(), indent=2), encoding="utf-8"
    )
    seed_everything(seed)
    full_datasets, feature_names, edge_index, _ = prepare_publication_datasets(config, fold)
    datasets: dict[str, Dataset] = full_datasets
    pipeline_pilot = max_windows_per_split is not None
    if pipeline_pilot:
        datasets = {
            role: limited_balanced_subset(dataset, max_windows_per_split, seed + index)
            for index, (role, dataset) in enumerate(full_datasets.items())
        }
    (output_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "feature_engineering": ["last", "mean", "std", "min", "max"],
                "engineered_feature_count": 5 * len(feature_names),
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
    extracted = {
        role: engineered_window_features(dataset) for role, dataset in datasets.items()
    }
    train_x, train_meta = extracted["train"]
    validation_x, validation_meta = extracted["validation"]
    test_x, test_meta = extracted["test"]
    name = str(config["model"]["name"])
    model = build_classical_model(name, seed, config["model"])
    model.fit(train_x, train_meta["labels"])
    validation_probabilities = positive_probabilities(model, validation_x)
    joblib.dump(model, output_dir / "best.joblib")
    reloaded = joblib.load(output_dir / "best.joblib")
    if not np.allclose(
        positive_probabilities(reloaded, validation_x),
        validation_probabilities,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Checkpoint reload did not reproduce validation predictions")
    threshold = (
        0.5
        if name == "majority"
        else select_threshold(validation_meta["labels"], validation_probabilities)
    )
    validation_metrics = classification_metrics(
        validation_meta["labels"], validation_probabilities, threshold
    )
    test_probabilities = positive_probabilities(reloaded, test_x)
    test_metrics = classification_metrics(test_meta["labels"], test_probabilities, threshold)
    event_names = full_datasets["test"].group_names
    mapped_groups = np.asarray([event_names[index] for index in test_meta["groups"]])
    test_event_metrics = event_metrics(
        test_meta["labels"],
        test_probabilities,
        test_meta["timestamps"],
        mapped_groups,
        threshold,
    )
    (output_dir / "event_metrics.json").write_text(
        json.dumps(test_event_metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(test_meta["timestamps"], unit="s"),
            "event_id": mapped_groups,
            "label": test_meta["labels"],
            "probability": test_probabilities,
            "prediction": (test_probabilities >= threshold).astype(int),
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    summary = PublicationRunSummary(
        run_id=output_dir.name,
        status=(
            "PUBLICATION_PIPELINE_PILOT_PASSED"
            if pipeline_pilot
            else "PUBLICATION_FOLD_COMPLETED"
        ),
        development_only=pipeline_pilot,
        fold=fold,
        seed=seed,
        device="cpu",
        best_epoch=0,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        elapsed_seconds=time.time() - started,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary
