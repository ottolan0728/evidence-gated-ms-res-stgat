from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from iiot_msresstgat.threshold_policy import (
    select_false_alarm_constrained_threshold,
    timestamps_to_unix_seconds,
)


ROOT = Path("outputs/gate4b/pooling_formal")
RUNS_ROOT = ROOT / "runs"
CONTROL_ROOT = Path("outputs/gate3/r4b_formal/runs")
FREEZE_PATH = Path("configs/gate4b_pooling_redesign.yaml")
REGISTRY_PATH = RUNS_ROOT / "matrix_registry.csv"
SEEDS = (11, 22, 33, 44, 55)
CONTROL = "mean_time_mean_sensor"
CANDIDATES = (
    "stats_time_mean_sensor",
    "mean_time_stats_sensor",
    "stats_time_stats_sensor",
)
VARIANTS = (CONTROL, *CANDIDATES)
MAX_FALSE_ALARM_EPISODES_PER_HOUR = 1.0
MAX_NORMAL_ALARM_DUTY_CYCLE = 0.01
MERGE_GAP_SECONDS = 60
GRID_SIZE = 501
FORBIDDEN_TEST_OUTPUTS = (
    "predictions.csv",
    "event_metrics.json",
    "test_calibration.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        frame["label"].to_numpy(dtype=int),
        frame["probability"].to_numpy(dtype=float),
        timestamps_to_unix_seconds(frame["timestamp"].to_numpy()),
        frame["event_id"].fillna("").to_numpy(dtype=str),
    )


def formal_run_id(variant: str, seed: int) -> str:
    return f"gate4b_formal__{variant}__fold0__seed{seed}"


def control_run_id(seed: int) -> str:
    return f"gate3_r4b_formal__control_weighted_bce__fold0__seed{seed}"


def run_directory(variant: str, seed: int) -> tuple[str, Path]:
    if variant == CONTROL:
        current_run_id = control_run_id(seed)
        return current_run_id, CONTROL_ROOT / current_run_id
    current_run_id = formal_run_id(variant, seed)
    return current_run_id, RUNS_ROOT / current_run_id


def load_run(variant: str, seed: int) -> tuple[dict, dict]:
    current_run_id, run_dir = run_directory(variant, seed)
    paths = {
        "metrics": run_dir / "metrics.json",
        "events": run_dir / "validation_event_metrics.json",
        "calibration": run_dir / "validation_calibration.json",
        "training": run_dir / "training_summary.json",
        "objective": run_dir / "objective_summary.json",
        "predictions": run_dir / "validation_predictions.csv",
        "config": run_dir / "config.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{current_run_id}: missing required artifacts: {missing}")
    forbidden = [name for name in FORBIDDEN_TEST_OUTPUTS if (run_dir / name).exists()]
    if forbidden:
        raise RuntimeError(f"{current_run_id}: forbidden test artifacts present: {forbidden}")

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    events = json.loads(paths["events"].read_text(encoding="utf-8"))
    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    training = json.loads(paths["training"].read_text(encoding="utf-8"))
    objective = json.loads(paths["objective"].read_text(encoding="utf-8"))
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    if metrics.get("test_metrics") != {}:
        raise RuntimeError(f"{current_run_id}: metrics.json contains non-empty test metrics")
    if training.get("test_evaluation_enabled") is not False:
        raise RuntimeError(f"{current_run_id}: test evaluation was not disabled")
    if int(metrics["fold"]) != 0 or int(metrics["seed"]) != seed:
        raise RuntimeError(f"{current_run_id}: fold/seed metadata mismatch")
    if objective.get("loss_name") != "weighted_bce":
        raise RuntimeError(f"{current_run_id}: objective differs from frozen weighted BCE")

    predictions = pd.read_csv(paths["predictions"])
    selected = select_false_alarm_constrained_threshold(
        *prediction_arrays(predictions),
        max_false_alarm_episodes_per_hour=MAX_FALSE_ALARM_EPISODES_PER_HOUR,
        max_normal_alarm_duty_cycle=MAX_NORMAL_ALARM_DUTY_CYCLE,
        merge_gap_seconds=MERGE_GAP_SECONDS,
        grid_size=GRID_SIZE,
    )
    row = {
        "run_id": current_run_id,
        "artifact_source": "gate3_r4b_control" if variant == CONTROL else "gate4b_formal",
        "variant": variant,
        "fold": 0,
        "seed": seed,
        "temporal_pooling": config["model"].get("temporal_pooling", "mean"),
        "sensor_pooling": config["model"].get("sensor_pooling", "mean"),
        "best_epoch": int(training["best_epoch"]),
        "epochs_completed": int(training["epochs_completed"]),
        "validation_pr_auc": float(metrics["validation_metrics"]["pr_auc"]),
        "validation_macro_scenario_pr_auc": float(events["macro_scenario_pr_auc"]),
        "validation_brier_score": float(calibration["brier_score"]),
        "validation_ece_15": float(calibration["ece_15"]),
        **selected,
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    return row, hashes


def paired_differences(seed_rows: pd.DataFrame) -> pd.DataFrame:
    control = seed_rows[seed_rows["variant"] == CONTROL].set_index("seed")
    rows = []
    for variant in CANDIDATES:
        candidate = seed_rows[seed_rows["variant"] == variant].set_index("seed")
        for seed in SEEDS:
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "macro_scenario_pr_auc_difference": float(
                        candidate.loc[seed, "validation_macro_scenario_pr_auc"]
                        - control.loc[seed, "validation_macro_scenario_pr_auc"]
                    ),
                    "event_recall_difference": float(
                        candidate.loc[seed, "validation_event_recall"]
                        - control.loc[seed, "validation_event_recall"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def assess_variants(seed_rows: pd.DataFrame, freeze: dict) -> pd.DataFrame:
    ranking = freeze["success_criteria"]["ranking"]
    operating = freeze["success_criteria"]["operating_point"]
    min_mean_macro_gain = float(
        ranking["minimum_mean_macro_scenario_pr_auc_gain_vs_control"]
    )
    min_worst_macro_gain = float(
        ranking["minimum_worst_seed_macro_scenario_pr_auc_gain_vs_control"]
    )
    min_mean_event_recall = float(operating["minimum_mean_event_recall"])
    min_worst_event_recall = float(operating["minimum_worst_seed_event_recall"])
    control = seed_rows[seed_rows["variant"] == CONTROL]
    control_mean_macro = float(control["validation_macro_scenario_pr_auc"].mean())
    control_worst_macro = float(control["validation_macro_scenario_pr_auc"].min())
    paired = paired_differences(seed_rows)
    rows = []
    for variant in VARIANTS:
        group = seed_rows[seed_rows["variant"] == variant]
        differences = paired[paired["variant"] == variant]
        mean_macro = float(group["validation_macro_scenario_pr_auc"].mean())
        worst_macro = float(group["validation_macro_scenario_pr_auc"].min())
        mean_event_recall = float(group["validation_event_recall"].mean())
        worst_event_recall = float(group["validation_event_recall"].min())
        mean_macro_gain = mean_macro - control_mean_macro
        worst_macro_gain = worst_macro - control_worst_macro
        is_candidate = variant != CONTROL
        primary_pass = bool(
            is_candidate
            and mean_macro_gain >= min_mean_macro_gain
            and worst_macro_gain >= min_worst_macro_gain
        )
        operating_point_pass = bool(
            is_candidate
            and mean_event_recall >= min_mean_event_recall
            and worst_event_recall >= min_worst_event_recall
        )
        rows.append(
            {
                "variant": variant,
                "n_seeds": len(group),
                "mean_validation_macro_scenario_pr_auc": mean_macro,
                "sd_validation_macro_scenario_pr_auc": float(
                    group["validation_macro_scenario_pr_auc"].std(ddof=1)
                ),
                "worst_seed_validation_macro_scenario_pr_auc": worst_macro,
                "mean_macro_gain_vs_control": mean_macro_gain,
                "worst_macro_gain_vs_control": worst_macro_gain,
                "paired_macro_wins_vs_control": (
                    int((differences["macro_scenario_pr_auc_difference"] > 0).sum())
                    if is_candidate
                    else 0
                ),
                "paired_median_macro_difference_vs_control": (
                    float(differences["macro_scenario_pr_auc_difference"].median())
                    if is_candidate
                    else 0.0
                ),
                "mean_validation_event_recall": mean_event_recall,
                "worst_seed_validation_event_recall": worst_event_recall,
                "mean_validation_f1": float(group["validation_window_f1"].mean()),
                "mean_validation_false_alarm_episodes_per_hour": float(
                    group["validation_false_alarm_episodes_per_hour"].mean()
                ),
                "mean_validation_normal_alarm_duty_cycle": float(
                    group["validation_normal_alarm_duty_cycle"].mean()
                ),
                "mean_validation_ece_15": float(group["validation_ece_15"].mean()),
                "mean_validation_brier_score": float(
                    group["validation_brier_score"].mean()
                ),
                "primary_pass": primary_pass,
                "operating_point_pass": operating_point_pass,
                "gate_pass": primary_pass and operating_point_pass,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    freeze = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze["scope"]["test_evaluation"] != "prohibited":
        raise RuntimeError("Gate 4B freeze does not prohibit test evaluation")
    if tuple(freeze["scope"]["seeds"]) != SEEDS or int(freeze["scope"]["fold"]) != 0:
        raise RuntimeError("Gate 4B frozen fold/seeds differ from the analysis contract")

    registry = pd.read_csv(REGISTRY_PATH).fillna("")
    expected_ids = {formal_run_id(variant, seed) for variant in CANDIDATES for seed in SEEDS}
    actual_ids = set(registry["run_id"].astype(str))
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"registry mismatch: missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )
    if set(registry["status"]) != {"PUBLICATION_FOLD_COMPLETED"}:
        raise RuntimeError("not all registry rows are PUBLICATION_FOLD_COMPLETED")

    rows = []
    input_hashes = {"freeze": sha256(FREEZE_PATH), "registry": sha256(REGISTRY_PATH)}
    for variant in VARIANTS:
        for seed in SEEDS:
            row, hashes = load_run(variant, seed)
            rows.append(row)
            input_hashes[row["run_id"]] = hashes

    seed_rows = pd.DataFrame(rows)
    paired = paired_differences(seed_rows)
    summary = assess_variants(seed_rows, freeze)
    passing = summary[summary["gate_pass"]].copy()
    if passing.empty:
        selected_variant = None
        decision_status = "GATE4B_FAILED_STOP_SUPERVISED_REDESIGN"
    else:
        complexity = {
            "stats_time_mean_sensor": 1,
            "mean_time_stats_sensor": 1,
            "stats_time_stats_sensor": 2,
        }
        passing["modified_pooling_axes"] = passing["variant"].map(complexity)
        passing = passing.sort_values(
            [
                "worst_seed_validation_event_recall",
                "mean_validation_event_recall",
                "worst_seed_validation_macro_scenario_pr_auc",
                "mean_validation_macro_scenario_pr_auc",
                "modified_pooling_axes",
                "variant",
            ],
            ascending=[False, False, False, False, True, True],
        )
        selected_variant = str(passing.iloc[0]["variant"])
        decision_status = "GATE4B_PASSED_UNLOCK_CONFIRMATORY_FOLD_PLAN"

    seed_rows.to_csv(ROOT / "validation_seed_results.csv", index=False)
    paired.to_csv(ROOT / "validation_paired_seed_differences.csv", index=False)
    summary.to_csv(ROOT / "validation_variant_summary.csv", index=False)
    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": decision_status,
        "selected_variant": selected_variant,
        "scope": "Fold 0 validation only; seeds 11, 22, 33, 44, and 55",
        "test_artifacts_read": False,
        "registry_runs": len(registry),
        "registry_failures": int((registry["status"] != "PUBLICATION_FOLD_COMPLETED").sum()),
        "control_source": "completed Gate 3-R4B control_weighted_bce artifacts",
        "inference_boundary": (
            "Seeds quantify computational stability within one development fold; "
            "no seed-level significance test is treated as independent-dataset inference."
        ),
        "constraints": freeze["operating_policy"],
        "success_criteria": freeze["success_criteria"],
        "selection_order": freeze["selection_order"],
        "failure_action": freeze["failure_action"],
        "variants": summary.to_dict("records"),
        "input_sha256": input_hashes,
    }
    (ROOT / "gate_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
