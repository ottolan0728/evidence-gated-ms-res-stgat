"""Path-neutral publication reproduction wrapper for post-review R3–R6.

This wrapper is a curated reproduction entry point, not the byte-identical historical
authorization controller. It preserves the scientific model, partition-role, training, and
common-endpoint contracts while replacing machine-local paths with explicit command-line inputs.

Input data must be role-isolated train/validation NPZ shards. Test/reserved roles are not
accepted or opened.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEEDS = {11, 22, 33, 44, 55}
BASE_TRAINING = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "patience": 10,
}
BASE_GRAPH = {"correlation_threshold": 0.35, "adjacent_stages_only": True}
DASP = {
    "name": "ms_res_stgat", "hidden_dim": 64, "heads": 4, "dropout": 0.1,
    "kernels": [3, 5, 7], "use_residual": True, "spatial_encoder": "gatv2",
    "temporal_pooling": "mean_max_std", "sensor_pooling": "mean_max_std",
    "pooling_epsilon": 1e-8,
}
MEAN_MEAN = {
    "name": "ms_res_stgat", "hidden_dim": 64, "heads": 4, "dropout": 0.1,
    "kernels": [3, 5, 7], "use_residual": True, "spatial_encoder": "gatv2",
    "temporal_pooling": "mean", "sensor_pooling": "mean", "pooling_epsilon": 1e-8,
}
STAGES = {
    "r3": {
        "stage": "R3", "folds": {0},
        "models": {
            "mean_mean": {"model": MEAN_MEAN, "training": BASE_TRAINING},
            "capacity_mean_mean": {"model": {"name": "capacity_mean_mean"}, "training": BASE_TRAINING},
            "dasp": {"model": DASP, "training": BASE_TRAINING},
        },
        "graph": BASE_GRAPH,
    },
    "r4": {
        "stage": "R4", "folds": {0},
        "models": {
            "mean_mean": {"model": MEAN_MEAN, "training": BASE_TRAINING},
            "capacity_mean_mean": {"model": {"name": "capacity_mean_mean"}, "training": BASE_TRAINING},
            "dasp": {"model": DASP, "training": BASE_TRAINING},
            "tcn": {"model": {"name": "tcn", "hidden_dim": 64, "dropout": 0.1}, "training": BASE_TRAINING},
            "gatv2": {"model": {"name": "gatv2", "hidden_dim": 64, "heads": 4, "dropout": 0.1}, "training": BASE_TRAINING},
            "usad": {"model": {"name": "usad", "latent_dim": 40, "alpha": 0.5, "beta": 0.5}, "training": BASE_TRAINING},
            "logistic_regression": {"model": {"name": "logistic_regression", "max_iter": 1000, "class_weight": "balanced"}, "training": BASE_TRAINING},
            "random_forest": {"model": {"name": "random_forest", "n_estimators": 200, "class_weight": "balanced_subsample", "n_jobs": -1}, "training": BASE_TRAINING},
        },
        "graph": BASE_GRAPH,
    },
    "r5": {
        "stage": "R5", "folds": {1, 2, 3, 4},
        "models": {
            "mean_mean": {"model": MEAN_MEAN, "training": BASE_TRAINING},
            "capacity_mean_mean": {"model": {"name": "capacity_mean_mean"}, "training": BASE_TRAINING},
            "dasp": {"model": DASP, "training": BASE_TRAINING},
            "tcn": {"model": {"name": "tcn", "hidden_dim": 64, "dropout": 0.1}, "training": BASE_TRAINING},
            "gatv2": {"model": {"name": "gatv2", "hidden_dim": 64, "heads": 4, "dropout": 0.1}, "training": BASE_TRAINING},
            "usad": {"model": {"name": "usad", "latent_dim": 40, "alpha": 0.5, "beta": 0.5}, "training": BASE_TRAINING},
        },
        "graph": BASE_GRAPH,
    },
    "r6c": {
        "stage": "R6C", "folds": {0},
        "models": {
            "dasp_corr025": {"model": DASP, "training": BASE_TRAINING, "graph": {"correlation_threshold": 0.25, "adjacent_stages_only": True}},
            "dasp_corr045": {"model": DASP, "training": BASE_TRAINING, "graph": {"correlation_threshold": 0.45, "adjacent_stages_only": True}},
            "dasp_unconstrained035": {"model": DASP, "training": BASE_TRAINING, "graph": {"correlation_threshold": 0.35, "adjacent_stages_only": False}},
        },
        "graph": BASE_GRAPH,
    },
    "r6d": {
        "stage": "R6D", "folds": {0},
        "models": {
            "dasp_single_scale5": {
                "model": {**DASP, "kernels": [5]}, "training": BASE_TRAINING
            },
            "dasp_no_residual": {
                "model": {**DASP, "use_residual": False}, "training": BASE_TRAINING
            },
            "dasp_temporal_only": {
                "model": {**DASP, "spatial_encoder": "mean_pool"}, "training": BASE_TRAINING
            },
        },
        "graph": BASE_GRAPH,
    },
}

ALARM_POLICY = {
    "max_false_alarm_episodes_per_hour": 1.0,
    "max_normal_alarm_duty_cycle": 0.01,
    "merge_gap_seconds": 60,
    "grid_size": 501,
}


@dataclass
class RoleData:
    role: str
    windows: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    events: np.ndarray
    window_ids: np.ndarray
    train_rows: np.ndarray | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_role_folds(outer_fold: int) -> dict[str, list[int]]:
    validation = (outer_fold + 1) % 5
    return {
        "train": [i for i in range(5) if i not in {outer_fold, validation}],
        "validation": [validation],
        "forbidden_test": [outer_fold],
    }


def read_role(root: Path, metadata: dict, role: str) -> RoleData:
    item = metadata[role]
    if item.get("role") != role:
        raise ValueError(f"Metadata role mismatch for {role}")
    path = (root / item["relative_path"]).resolve()
    if root.resolve() not in path.parents or path.suffix != ".npz":
        raise ValueError("Role shard path escape or non-NPZ input")
    if item.get("payload_sha256") and sha256_file(path) != item["payload_sha256"]:
        raise ValueError(f"{role} payload SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as z:
        required = {"windows", "labels", "timestamps", "events", "window_ids"}
        if not required.issubset(z.files):
            raise ValueError(f"{role} shard missing {sorted(required-set(z.files))}")
        windows = np.asarray(z["windows"])
        labels = np.asarray(z["labels"])
        timestamps = np.asarray(z["timestamps"])
        events = np.asarray(z["events"]).astype(str)
        window_ids = np.asarray(z["window_ids"]).astype(str)
        train_rows = np.asarray(z["train_rows"]) if role == "train" and "train_rows" in z.files else None
    n = len(windows)
    if windows.shape != (n, 60, 51) or n == 0 or not np.isfinite(windows).all():
        raise ValueError(f"Invalid {role} window tensor")
    if any(np.asarray(a).shape != (n,) for a in [labels, timestamps, events, window_ids]):
        raise ValueError(f"{role} metadata alignment failure")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Nonbinary labels")
    if len(set(window_ids.tolist())) != n:
        raise ValueError("Duplicate window IDs")
    if role == "train":
        if train_rows is None or train_rows.ndim != 2 or train_rows.shape[1] != 51:
            raise ValueError("Training shard requires unique raw train_rows[:,51]")
        if not np.isfinite(train_rows).all():
            raise ValueError("Nonfinite training rows")
    return RoleData(
        role, windows.astype(np.float32), labels.astype(np.uint8),
        timestamps.astype(np.int64), events, window_ids, train_rows
    )


def validate_metadata(metadata: dict, outer_fold: int) -> list[str]:
    if set(metadata) != {"train", "validation"}:
        raise ValueError("Metadata must contain exactly train and validation roles")
    plan = expected_role_folds(outer_fold)
    feature_names = metadata["train"].get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != 51:
        raise ValueError("Exactly 51 aligned feature names are required")
    if metadata["validation"].get("feature_names") != feature_names:
        raise ValueError("Feature schema differs between train and validation")
    for role in ("train", "validation"):
        item = metadata[role]
        if item.get("role_fold_ids") != plan[role]:
            raise ValueError(f"{role} fold identity does not match frozen role map")
        if item.get("window_size") != 60:
            raise ValueError("Frozen window size is 60")
        if item.get("stride") != (5 if role == "train" else 1):
            raise ValueError("Frozen train/eval stride mismatch")
        if item.get("boundary_purge") != 59 or item.get("no_cross_event_windows") is not True:
            raise ValueError("Boundary-purge/event-window provenance mismatch")
        if item.get("storage_layout") != "role_isolated":
            raise ValueError("Shared/all-role cache is not accepted")
    return feature_names


def normalize(train: RoleData, validation: RoleData):
    rows = np.asarray(train.train_rows, dtype=np.float64)
    minimum = rows.min(axis=0)
    maximum = rows.max(axis=0)
    scale = np.where(maximum == minimum, 1.0, maximum - minimum)
    for view in (train, validation):
        value = np.asarray(view.windows, dtype=np.float64)
        view.windows = ((value - minimum) / scale).astype(np.float32)
    return minimum, maximum


def capacity_model():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from iiot_msresstgat.models import MSResSTGAT, MultiScaleTemporalEncoder

    class CapacityMatchedTemporalEncoder(MultiScaleTemporalEncoder):
        def __init__(self):
            super().__init__(64, (3, 5, 7), 0.1, pooling="mean")
            self.capacity_expansion = nn.Linear(192, 192)
            self.capacity_norm = nn.LayerNorm(64)
        def forward(self, node_windows):
            sequences = node_windows.unsqueeze(1)
            means = [F.gelu(layer(sequences)).mean(dim=-1) for layer in self.branches]
            expanded = F.gelu(self.capacity_expansion(torch.cat(means, dim=-1)))
            return self.dropout(self.capacity_norm(F.gelu(self.projection(expanded))))

    class CapacityMatchedMSResSTGAT(MSResSTGAT):
        def __init__(self):
            super().__init__(
                hidden_dim=64, heads=4, kernels=(3,5,7), dropout=0.1,
                use_residual=True, spatial_encoder="gatv2",
                temporal_pooling="mean", sensor_pooling="mean"
            )
            self.temporal = CapacityMatchedTemporalEncoder()
            count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            if count != 69953:
                raise RuntimeError(f"Capacity contract mismatch: {count}")
    return CapacityMatchedMSResSTGAT()


def make_model(name: str, spec: dict, seed: int):
    import torch
    torch.manual_seed(seed)
    cfg = spec["model"]
    if name == "capacity_mean_mean":
        model = capacity_model()
        return model, 69953
    if cfg["name"] == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]), class_weight=cfg["class_weight"],
            random_state=seed, n_jobs=cfg["n_jobs"]
        ), None
    if cfg["name"] == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            class_weight=cfg["class_weight"], max_iter=int(cfg["max_iter"]),
            random_state=seed
        ), None
    from iiot_msresstgat.models import USADModel, build_model
    if cfg["name"] == "usad":
        model = USADModel(612, int(cfg["latent_dim"]))
    else:
        model = build_model(cfg, 51)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, count


def graph_dataset(view: RoleData, edge_index):
    import torch
    from torch_geometric.data import Data
    return [
        Data(x=torch.from_numpy(w.T.copy()), edge_index=edge_index, y=torch.tensor(float(y)))
        for w, y in zip(view.windows, view.labels, strict=True)
    ]


def feature_matrix(view: RoleData):
    x = view.windows
    return np.concatenate(
        [x[:, -1, :], x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1
    ).astype(np.float32)


def usad_windows(x: np.ndarray, device):
    import torch
    ds = np.partition(x.reshape(len(x), 12, 5, 51), kth=2, axis=2)[:, :, 2, :].reshape(len(x), 612)
    return torch.from_numpy(np.ascontiguousarray(ds, dtype=np.float32)).to(device)


def predict(name, model, view: RoleData, edge_index, device):
    import torch
    if name in {"random_forest", "logistic_regression"}:
        classes = list(model.classes_)
        return model.predict_proba(feature_matrix(view))[:, classes.index(1)]
    model.eval()
    scores = []
    with torch.no_grad():
        if name == "usad":
            for start in range(0, len(view.windows), 64):
                x = usad_windows(view.windows[start:start+64], device)
                scores.extend(model.anomaly_score(x, 0.5, 0.5).cpu().numpy())
        else:
            from torch_geometric.loader import DataLoader
            loader = DataLoader(graph_dataset(view, edge_index), batch_size=64, shuffle=False)
            for batch in loader:
                scores.extend(torch.sigmoid(model(batch.to(device))["logits"]).cpu().numpy())
    return np.asarray(scores, dtype=float)


def fit_model(name, model, spec, train: RoleData, validation: RoleData, edge_index, device, seed: int):
    import torch
    cfg = spec["training"]
    if name in {"random_forest", "logistic_regression"}:
        model.fit(feature_matrix(train), train.labels)
        return None, None
    model.to(device)
    if name == "usad":
        eligible = train.windows[train.labels == 0]
        if not len(eligible):
            raise ValueError("USAD requires normal training windows")
        first = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder1.parameters()),
            lr=cfg["learning_rate"]
        )
        second = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder2.parameters()),
            lr=cfg["learning_rate"]
        )
        rng = np.random.RandomState(seed)
        for epoch in range(1, int(cfg["epochs"]) + 1):
            order = rng.permutation(len(eligible))
            for start in range(0, len(order), 64):
                x = usad_windows(eligible[order[start:start+64]], device)
                first.zero_grad()
                loss, _ = model.losses(x, epoch)
                loss.backward(); first.step()
                second.zero_grad()
                _, loss = model.losses(x, epoch)
                loss.backward(); second.step()
        return int(cfg["epochs"]), int(cfg["epochs"])
    positive_weight = float((train.labels == 0).sum() / max((train.labels == 1).sum(), 1))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    from torch_geometric.loader import DataLoader
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        graph_dataset(train, edge_index), batch_size=64, shuffle=True, generator=generator
    )
    from iiot_msresstgat.metrics import event_metrics
    best = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    completed = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        completed = epoch
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch)["logits"], batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        score = predict(name, model, validation, edge_index, device)
        value = event_metrics(
            validation.labels, score, validation.timestamps, validation.events, 0.5
        )["macro_scenario_pr_auc"]
        if value is None:
            raise RuntimeError("No eligible mixed-label validation scenario")
        if value > best:
            best = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("No neural checkpoint selected")
    model.load_state_dict(best_state)
    return best_epoch, completed


def evaluate_endpoint(view: RoleData, scores: np.ndarray):
    from iiot_msresstgat.metrics import event_metrics
    from iiot_msresstgat.threshold_policy import select_false_alarm_constrained_threshold
    selected = select_false_alarm_constrained_threshold(
        view.labels, scores, view.timestamps, view.events, **ALARM_POLICY
    )
    metrics = event_metrics(
        view.labels, scores, view.timestamps, view.events, selected["threshold"]
    )
    return {
        "macro_scenario_pr_auc": metrics["macro_scenario_pr_auc"],
        "selected_threshold": selected["threshold"],
        "false_alarm_episodes_per_hour": selected["validation_false_alarm_episodes_per_hour"],
        "normal_alarm_duty": selected["validation_normal_alarm_duty_cycle"],
        "constrained_event_recall": selected["validation_event_recall"],
        "window_f1": selected["validation_window_f1"],
        "candidate_count": selected["candidate_count"],
        "event_detections": {
            k: v is not None for k, v in metrics["detection_delay_seconds_by_event"].items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    stage = STAGES[args.stage]
    if args.fold not in stage["folds"]:
        raise ValueError("Fold outside frozen stage")
    if args.seed not in SEEDS:
        raise ValueError("Seed outside frozen set")
    if args.model not in stage["models"]:
        raise ValueError("Model outside frozen stage")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    features = validate_metadata(metadata, args.fold)
    root = args.data_root.resolve()
    train = read_role(root, metadata, "train")
    validation = read_role(root, metadata, "validation")
    minimum, maximum = normalize(train, validation)

    graph_cfg = stage["models"][args.model].get("graph", stage["graph"])
    from iiot_msresstgat.graph import build_graph
    edge_index = build_graph(
        np.asarray(train.train_rows, dtype=np.float64), features,
        graph_cfg["correlation_threshold"], graph_cfg["adjacent_stages_only"]
    )

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device if args.device == "cpu" else "cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    spec = stage["models"][args.model]
    model, parameter_count = make_model(args.model, spec, args.seed)
    started = time.time()
    best_epoch, epochs_completed = fit_model(
        args.model, model, spec, train, validation, edge_index, device, args.seed
    )
    scores = predict(args.model, model, validation, edge_index, device)
    endpoint = evaluate_endpoint(validation, scores)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "wrapper": "path-neutral-publication-reproduction-v2",
        "not_byte_identical_historical_controller": True,
        "stage": stage["stage"],
        "model": args.model,
        "outer_fold": args.fold,
        "seed": args.seed,
        "role_map": expected_role_folds(args.fold),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "graph": graph_cfg,
        "scaler_sha256": hashlib.sha256(np.r_[minimum, maximum].tobytes()).hexdigest(),
        "graph_sha256": hashlib.sha256(edge_index.cpu().numpy().tobytes()).hexdigest(),
        "metadata_sha256": sha256_file(args.metadata),
        "endpoint": endpoint,
        "elapsed_seconds": time.time() - started,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (out/"reproduction_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
