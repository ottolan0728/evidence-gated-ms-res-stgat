from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph import build_graph_from_correlation


@dataclass(frozen=True)
class Segment:
    source: str
    start: int
    stop: int
    group: str


@dataclass(frozen=True)
class MinMaxParameters:
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def scale(self) -> np.ndarray:
        width = self.maximum - self.minimum
        return np.where(width == 0, 1.0, width)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.minimum) / self.scale


def load_processed_arrays(processed_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    manifest = json.loads((processed_dir / "processed_manifest.json").read_text(encoding="utf-8"))
    arrays = {}
    for source in ("normal", "attack"):
        arrays[source] = {
            kind: np.load(processed_dir / item["path"], mmap_mode="r")
            for kind, item in manifest[source]["outputs"].items()
        }
    feature_names = manifest["normal"]["feature_names"]
    if feature_names != manifest["attack"]["feature_names"]:
        raise ValueError("Processed normal and attack feature schemas differ")
    return arrays, feature_names


def segments_for_fold(
    arrays: dict[str, dict[str, np.ndarray]],
    split_manifest: dict,
    fold: int,
    window_size: int,
) -> dict[str, list[Segment]]:
    fold_spec = next((item for item in split_manifest["folds"] if item["fold"] == fold), None)
    if fold_spec is None:
        raise ValueError(f"Unknown fold: {fold}")
    role_by_event = {
        event_id: role
        for role in ("train", "validation", "test")
        for event_id in fold_spec[f"{role}_event_ids"]
    }
    timestamps = arrays["attack"]["timestamps"]
    purge = window_size - 1
    result: dict[str, list[Segment]] = {
        "train": [Segment("normal", 0, len(arrays["normal"]["labels"]), "normal_reference")],
        "validation": [],
        "test": [],
    }
    for event in split_manifest["events"]:
        start_second = int(np.datetime64(event["block_start"], "s").astype(np.int64))
        end_second = int(np.datetime64(event["block_end"], "s").astype(np.int64))
        start = int(np.searchsorted(timestamps, start_second, side="left")) + purge
        stop = int(np.searchsorted(timestamps, end_second, side="right")) - purge
        if stop - start < window_size:
            raise ValueError(f"Purged block is too short for {event['event_id']}")
        role = role_by_event[event["event_id"]]
        result[role].append(Segment("attack", start, stop, event["event_id"]))
    return result


def fit_minmax(
    arrays: dict[str, dict[str, np.ndarray]], segments: list[Segment], chunk_size: int = 50_000
) -> MinMaxParameters:
    feature_count = arrays[segments[0].source]["values"].shape[1]
    minimum = np.full(feature_count, np.inf, dtype=np.float64)
    maximum = np.full(feature_count, -np.inf, dtype=np.float64)
    for segment in segments:
        values = arrays[segment.source]["values"]
        for start in range(segment.start, segment.stop, chunk_size):
            chunk = np.asarray(values[start : min(start + chunk_size, segment.stop)])
            minimum = np.minimum(minimum, chunk.min(axis=0))
            maximum = np.maximum(maximum, chunk.max(axis=0))
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("Non-finite training values prevent scaling")
    return MinMaxParameters(minimum, maximum)


def streaming_correlation(
    arrays: dict[str, dict[str, np.ndarray]], segments: list[Segment], chunk_size: int = 20_000
) -> np.ndarray:
    feature_count = arrays[segments[0].source]["values"].shape[1]
    count = 0
    total = np.zeros(feature_count, dtype=np.float64)
    cross = np.zeros((feature_count, feature_count), dtype=np.float64)
    for segment in segments:
        values = arrays[segment.source]["values"]
        for start in range(segment.start, segment.stop, chunk_size):
            chunk = np.asarray(
                values[start : min(start + chunk_size, segment.stop)], dtype=np.float64
            )
            count += len(chunk)
            total += chunk.sum(axis=0)
            cross += chunk.T @ chunk
    if count < 2:
        raise ValueError("At least two training rows are required for correlation")
    covariance = (cross - np.outer(total, total) / count) / (count - 1)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviation, standard_deviation)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator != 0,
        )
    np.fill_diagonal(correlation, 1.0)
    return correlation


class PublicationWindowDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, dict[str, np.ndarray]],
        segments: list[Segment],
        scaler: MinMaxParameters,
        edge_index: torch.Tensor,
        window_size: int,
        stride: int,
    ) -> None:
        self.arrays = arrays
        self.segments = segments
        self.scaler = scaler
        self.edge_index = edge_index
        self.window_size = window_size
        self.group_names = sorted({segment.group for segment in segments})
        group_indexes = {name: index for index, name in enumerate(self.group_names)}
        source_indexes = {"normal": 0, "attack": 1}
        starts, sources, groups = [], [], []
        for segment in segments:
            segment_starts = np.arange(
                segment.start, segment.stop - window_size + 1, stride, dtype=np.int64
            )
            starts.append(segment_starts)
            sources.append(np.full(len(segment_starts), source_indexes[segment.source], np.uint8))
            groups.append(np.full(len(segment_starts), group_indexes[segment.group], np.int16))
        self.sample_starts = np.concatenate(starts)
        self.sample_sources = np.concatenate(sources)
        self.sample_groups = np.concatenate(groups)
        self.source_names = ("normal", "attack")

    def __len__(self) -> int:
        return len(self.sample_starts)

    def targets(self) -> np.ndarray:
        targets = np.empty(len(self), dtype=np.uint8)
        for source_index, source_name in enumerate(self.source_names):
            selected = self.sample_sources == source_index
            rows = self.sample_starts[selected] + self.window_size - 1
            targets[selected] = self.arrays[source_name]["labels"][rows]
        return targets

    def __getitem__(self, index: int) -> Data:
        source_index = int(self.sample_sources[index])
        source = self.source_names[source_index]
        start = int(self.sample_starts[index])
        stop = start + self.window_size
        raw = np.asarray(self.arrays[source]["values"][start:stop], dtype=np.float32)
        scaled = self.scaler.transform(raw).astype(np.float32, copy=False)
        target_row = stop - 1
        return Data(
            x=torch.from_numpy(scaled.T.copy()),
            edge_index=self.edge_index,
            y=torch.tensor(float(self.arrays[source]["labels"][target_row])),
            timestamp=torch.tensor(int(self.arrays[source]["timestamps"][target_row])),
            group=torch.tensor(int(self.sample_groups[index])),
            source=torch.tensor(source_index),
        )


def prepare_publication_datasets(config: dict, fold: int) -> tuple[dict, list[str], torch.Tensor, MinMaxParameters]:
    data_config = config["data"]
    processed_dir = Path(data_config["official_processed_dir"])
    split_manifest = json.loads(
        Path(data_config["official_split_manifest"]).read_text(encoding="utf-8")
    )
    arrays, feature_names = load_processed_arrays(processed_dir)
    window_size = int(data_config["window_size"])
    segments = segments_for_fold(arrays, split_manifest, fold, window_size)
    scaler = fit_minmax(arrays, segments["train"])
    correlation = streaming_correlation(arrays, segments["train"])
    edge_index = build_graph_from_correlation(
        correlation,
        feature_names,
        float(config["graph"]["correlation_threshold"]),
        bool(config["graph"]["adjacent_stages_only"]),
    )
    datasets = {
        role: PublicationWindowDataset(
            arrays,
            role_segments,
            scaler,
            edge_index,
            window_size,
            int(data_config["train_stride"] if role == "train" else data_config["eval_stride"]),
        )
        for role, role_segments in segments.items()
    }
    return datasets, feature_names, edge_index, scaler
