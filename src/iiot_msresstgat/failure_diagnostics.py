from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def scenario_score_diagnostics(
    predictions: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Summarize score separation without selecting a decision threshold."""
    required = {"event_id", "label", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str | bool]] = []
    for event_id, frame in predictions.groupby("event_id", sort=True):
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["probability"].to_numpy(dtype=float)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{event_id} does not contain both normal and attack labels")
        normal = scores[labels == 0]
        attack = scores[labels == 1]
        prevalence = float(labels.mean())
        row: dict[str, float | int | str | bool] = {
            "seed": seed,
            "event_id": str(event_id),
            "windows": int(len(frame)),
            "normal_windows": int(len(normal)),
            "attack_windows": int(len(attack)),
            "prevalence": prevalence,
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
        }
        row["pr_lift_over_prevalence"] = float(row["pr_auc"]) / prevalence
        for name, values in (("normal", normal), ("attack", attack)):
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_q05"] = float(np.quantile(values, 0.05))
            row[f"{name}_q95"] = float(np.quantile(values, 0.95))
            row[f"{name}_probability_le_0_01"] = float(np.mean(values <= 0.01))
            row[f"{name}_probability_ge_0_99"] = float(np.mean(values >= 0.99))
            row[f"{name}_extreme_probability_fraction"] = float(
                np.mean((values <= 0.01) | (values >= 0.99))
            )
        row["mean_score_gap"] = float(row["attack_mean"]) - float(row["normal_mean"])
        row["median_score_gap"] = float(row["attack_median"]) - float(
            row["normal_median"]
        )
        row["roc_inverted"] = bool(float(row["roc_auc"]) < 0.5)
        row["median_inverted"] = bool(float(row["median_score_gap"]) <= 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def assess_mechanisms(
    scenario_seed: pd.DataFrame,
    seed_diagnostics: pd.DataFrame,
    imbalance: dict,
) -> dict[str, dict[str, float | str | bool]]:
    """Apply preregistered descriptive thresholds to diagnostic evidence."""
    pair_roc_inversion = float(scenario_seed["roc_inverted"].mean())
    pair_median_inversion = float(scenario_seed["median_inverted"].mean())
    scenario_sd = (
        scenario_seed.groupby("event_id")["pr_auc"].std(ddof=1).fillna(0.0)
    )
    scenario_range = scenario_seed.groupby("event_id")["pr_auc"].agg(
        lambda values: float(values.max() - values.min())
    )
    extreme = float(
        scenario_seed[
            ["normal_extreme_probability_fraction", "attack_extreme_probability_fraction"]
        ]
        .to_numpy()
        .mean()
    )
    median_monitor_correlation = float(
        seed_diagnostics["pooled_macro_pr_pearson"].median()
    )

    class_supported = float(imbalance["train_attack_fraction"]) < 0.05
    duration_supported = (
        float(imbalance["train_attack_event_max_min_duration_ratio"]) >= 10.0
        or float(imbalance["largest_train_attack_event_share"]) >= 0.20
    )
    return {
        "class_imbalance": {
            "status": "SUPPORTED" if class_supported else "NOT_SUPPORTED",
            "criterion": "train attack fraction < 0.05",
            "value": float(imbalance["train_attack_fraction"]),
        },
        "event_length_domination": {
            "status": "SUPPORTED" if duration_supported else "NOT_SUPPORTED",
            "criterion": "train max/min attack duration ratio >= 10 or largest share >= 0.20",
            "duration_ratio": float(
                imbalance["train_attack_event_max_min_duration_ratio"]
            ),
            "largest_share": float(imbalance["largest_train_attack_event_share"]),
        },
        "scenario_score_inversion": {
            "status": (
                "SUPPORTED"
                if pair_roc_inversion >= 0.20 or pair_median_inversion >= 0.20
                else "NOT_SUPPORTED"
            ),
            "criterion": "ROC or median inversion in >= 0.20 of scenario-seed pairs",
            "roc_inversion_fraction": pair_roc_inversion,
            "median_inversion_fraction": pair_median_inversion,
        },
        "probability_saturation": {
            "status": "SUPPORTED" if extreme >= 0.50 else "NOT_SUPPORTED",
            "criterion": "mean label-conditional extreme probability fraction >= 0.50",
            "value": extreme,
        },
        "seed_instability": {
            "status": (
                "SUPPORTED"
                if float(scenario_sd.mean()) >= 0.10
                or float(scenario_range.max()) >= 0.30
                else "NOT_SUPPORTED"
            ),
            "criterion": "mean scenario PR-AUC SD >= 0.10 or any range >= 0.30",
            "mean_scenario_pr_auc_sd": float(scenario_sd.mean()),
            "maximum_scenario_pr_auc_range": float(scenario_range.max()),
        },
        "monitor_misalignment": {
            "status": (
                "SUPPORTED" if median_monitor_correlation < 0.50 else "NOT_SUPPORTED"
            ),
            "criterion": "median within-seed pooled/macro validation PR Pearson < 0.50",
            "value": median_monitor_correlation,
        },
        "loss_domination": {
            "status": (
                "INDIRECT_SUPPORT"
                if class_supported and duration_supported
                else "INSUFFICIENT_EVIDENCE"
            ),
            "criterion": (
                "class imbalance and event-length domination are both supported; "
                "causality requires the R4B intervention"
            ),
        },
    }
