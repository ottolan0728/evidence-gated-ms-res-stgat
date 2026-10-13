from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import yaml

from .publication_experiment import run_publication_experiment


def apply_overrides(config: dict, overrides: dict) -> dict:
    for dotted_key, value in overrides.items():
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config


def load_matrix(matrix_path: Path) -> tuple[dict, dict]:
    specification = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    base_path = matrix_path.parent / specification["base"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return specification, base


def run_matrix(
    matrix_path: Path,
    output_root: Path,
    fold: int,
    seed: int,
    epochs: int | None,
    max_windows_per_split: int | None,
    selected_runs: set[str] | None = None,
) -> pd.DataFrame:
    specification, base = load_matrix(matrix_path)
    output_root.mkdir(parents=True, exist_ok=True)
    config_root = output_root / "materialized_configs"
    config_root.mkdir(exist_ok=True)
    registry_path = output_root / "matrix_registry.csv"
    existing = (
        pd.read_csv(registry_path).fillna("").to_dict("records")
        if registry_path.exists()
        else []
    )
    records_by_id = {row["run_id"]: row for row in existing}
    for run in specification["runs"]:
        name = str(run["name"])
        if selected_runs is not None and name not in selected_runs:
            continue
        run_id = f"{matrix_path.stem}__{name}__fold{fold}__seed{seed}"
        run_output = output_root / run_id
        metrics_path = run_output / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            records_by_id[run_id] = {
                "run_id": run_id,
                "matrix": matrix_path.stem,
                "run": name,
                "fold": fold,
                "seed": seed,
                "status": metrics["status"],
                "development_only": metrics["development_only"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "metrics_path": str(metrics_path.resolve()),
                "error": "",
            }
            continue
        config = json.loads(json.dumps(base))
        config["experiment"].update(specification.get("experiment", {}))
        config["experiment"]["name"] = f"{specification['experiment']['name']}:{name}"
        apply_overrides(config, run.get("overrides", {}))
        materialized = config_root / f"{run_id}.yaml"
        materialized.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        try:
            summary = run_publication_experiment(
                materialized,
                run_output,
                fold,
                seed,
                epochs,
                max_windows_per_split,
            )
            values = asdict(summary)
            records_by_id[run_id] = {
                "run_id": run_id,
                "matrix": matrix_path.stem,
                "run": name,
                "fold": fold,
                "seed": seed,
                "status": values["status"],
                "development_only": values["development_only"],
                "elapsed_seconds": values["elapsed_seconds"],
                "metrics_path": str(metrics_path.resolve()),
                "error": "",
            }
        except Exception as error:  # preserve failures for resumable research runs
            run_output.mkdir(parents=True, exist_ok=True)
            (run_output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            records_by_id[run_id] = {
                "run_id": run_id,
                "matrix": matrix_path.stem,
                "run": name,
                "fold": fold,
                "seed": seed,
                "status": "FAILED",
                "development_only": max_windows_per_split is not None,
                "elapsed_seconds": "",
                "metrics_path": "",
                "error": repr(error),
            }
        pd.DataFrame(records_by_id.values()).sort_values("run_id").to_csv(
            registry_path, index=False
        )
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                # The originating run already records the CUDA traceback. Avoid
                # replacing it with a secondary cache-cleanup failure.
                pass
    result = pd.DataFrame(records_by_id.values())
    return result.sort_values("run_id") if not result.empty else result


def aggregate_matrix(output_root: Path) -> pd.DataFrame:
    rows = []
    registry_path = output_root / "matrix_registry.csv"
    if not registry_path.exists():
        return pd.DataFrame()
    for record in pd.read_csv(registry_path).to_dict("records"):
        metrics_path = record.get("metrics_path")
        if not isinstance(metrics_path, str) or not metrics_path:
            rows.append(record)
            continue
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        row = dict(record)
        row.update(
            {f"validation_{key}": value for key, value in metrics["validation_metrics"].items()}
        )
        row.update({f"test_{key}": value for key, value in metrics["test_metrics"].items()})
        event_path = Path(metrics_path).parent / "event_metrics.json"
        if event_path.exists():
            events = json.loads(event_path.read_text(encoding="utf-8"))
            for key in (
                "event_recall",
                "median_detection_delay_seconds",
                "false_alarms_per_hour",
                "macro_scenario_pr_auc",
            ):
                row[f"test_{key}"] = events.get(key)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("run_id")
    result.to_csv(output_root / "summary.csv", index=False)
    return result
