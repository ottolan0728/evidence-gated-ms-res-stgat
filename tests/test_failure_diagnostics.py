import pandas as pd

from iiot_msresstgat.failure_diagnostics import (
    assess_mechanisms,
    scenario_score_diagnostics,
)


def test_scenario_score_diagnostics_detects_score_inversion():
    predictions = pd.DataFrame(
        {
            "event_id": ["event_01"] * 6,
            "label": [0, 0, 0, 1, 1, 1],
            "probability": [0.8, 0.9, 0.7, 0.1, 0.2, 0.3],
        }
    )
    result = scenario_score_diagnostics(predictions, seed=11).iloc[0]
    assert bool(result["roc_inverted"])
    assert bool(result["median_inverted"])
    assert result["mean_score_gap"] < 0
    assert result["pr_lift_over_prevalence"] > 0


def test_mechanism_thresholds_preserve_causal_boundary():
    scenarios = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b", "c"],
            "pr_auc": [0.1, 0.5, 0.2, 0.2, 0.4],
            "roc_inverted": [True, False, False, False, False],
            "median_inverted": [False, False, False, False, False],
            "normal_extreme_probability_fraction": [0.6] * 5,
            "attack_extreme_probability_fraction": [0.6] * 5,
        }
    )
    seeds = pd.DataFrame({"pooled_macro_pr_pearson": [0.2, 0.3, 0.4]})
    imbalance = {
        "train_attack_fraction": 0.02,
        "train_attack_event_max_min_duration_ratio": 12.0,
        "largest_train_attack_event_share": 0.15,
    }
    result = assess_mechanisms(scenarios, seeds, imbalance)
    assert result["class_imbalance"]["status"] == "SUPPORTED"
    assert result["event_length_domination"]["status"] == "SUPPORTED"
    assert result["scenario_score_inversion"]["status"] == "SUPPORTED"
    assert result["probability_saturation"]["status"] == "SUPPORTED"
    assert result["monitor_misalignment"]["status"] == "SUPPORTED"
    assert result["loss_domination"]["status"] == "INDIRECT_SUPPORT"
