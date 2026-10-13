from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .matrix import load_matrix, run_matrix


def campaign_plan(campaign_path: Path) -> pd.DataFrame:
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    rows = []
    for matrix_spec in campaign["matrices"]:
        matrix_path = campaign_path.parent / matrix_spec["config"]
        matrix, _ = load_matrix(matrix_path)
        excluded = set(matrix_spec.get("exclude_runs", []))
        output_root = Path(matrix_spec["output_root"])
        registry_path = output_root / "matrix_registry.csv"
        registry_status = {}
        if registry_path.exists():
            registry = pd.read_csv(registry_path).fillna("")
            registry_status = dict(zip(registry["run_id"], registry["status"], strict=False))
        for fold in campaign["folds"]:
            for seed in campaign["seeds"]:
                for run in matrix["runs"]:
                    name = str(run["name"])
                    if name in excluded:
                        continue
                    run_id = f"{matrix_path.stem}__{name}__fold{fold}__seed{seed}"
                    metrics_path = output_root / run_id / "metrics.json"
                    rows.append(
                        {
                            "campaign": campaign["name"],
                            "matrix": matrix_path.stem,
                            "run": name,
                            "fold": fold,
                            "seed": seed,
                            "run_id": run_id,
                            "status": (
                                "complete"
                                if metrics_path.exists()
                                else (
                                    "failed"
                                    if registry_status.get(run_id) == "FAILED"
                                    else "pending"
                                )
                            ),
                            "output_root": str(output_root),
                        }
                    )
    return pd.DataFrame(rows)


def campaign_status_dir(campaign_path: Path) -> Path:
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    return Path(campaign.get("status_output_dir", "outputs/gate3/production"))


def persist_campaign_plan(
    plan: pd.DataFrame,
    output_dir: Path = Path("outputs/gate3/production"),
) -> None:
    plan_path = output_dir / "campaign_plan.csv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, index=False)
    summary = {
        "total": len(plan),
        "complete": int((plan["status"] == "complete").sum()),
        "failed": int((plan["status"] == "failed").sum()),
        "pending": int((plan["status"] == "pending").sum()),
    }
    (plan_path.parent / "campaign_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def run_campaign(
    campaign_path: Path,
    epochs: int | None = None,
    selected_folds: set[int] | None = None,
    selected_seeds: set[int] | None = None,
    selected_runs: set[str] | None = None,
) -> pd.DataFrame:
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    for matrix_spec in campaign["matrices"]:
        matrix_path = campaign_path.parent / matrix_spec["config"]
        matrix, _ = load_matrix(matrix_path)
        excluded = set(matrix_spec.get("exclude_runs", []))
        available = {str(run["name"]) for run in matrix["runs"]} - excluded
        if selected_runs:
            available &= selected_runs
            if not available:
                continue
        output_root = Path(matrix_spec["output_root"])
        for fold in campaign["folds"]:
            if selected_folds and fold not in selected_folds:
                continue
            for seed in campaign["seeds"]:
                if selected_seeds and seed not in selected_seeds:
                    continue
                run_matrix(
                    matrix_path,
                    output_root,
                    fold,
                    seed,
                    epochs,
                    None,
                    available,
                )
    plan = campaign_plan(campaign_path)
    persist_campaign_plan(plan, campaign_status_dir(campaign_path))
    return plan
