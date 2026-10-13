from iiot_msresstgat.classical_experiment import build_classical_model
from pathlib import Path

from iiot_msresstgat.matrix import apply_overrides, load_matrix


def test_dotted_overrides_are_materialized():
    config = {"model": {"name": "ms_res_stgat"}, "graph": {"adjacent": True}}
    result = apply_overrides(
        config, {"model.name": "gru", "graph.adjacent": False, "model.hidden_dim": 16}
    )
    assert result["model"] == {"name": "gru", "hidden_dim": 16}
    assert result["graph"]["adjacent"] is False


def test_classical_model_factory():
    for name in ("majority", "logistic_regression", "random_forest"):
        assert build_classical_model(name, 11, {"n_estimators": 5}) is not None


def test_calibration_matrix_declares_development_only():
    specification, _ = load_matrix(Path("configs/calibration.yaml"))
    assert specification["experiment"]["development_only"] is True
    assert specification["experiment"]["purpose"] == "compute_calibration"


def test_r2_matrix_uses_strict_macro_scenario_monitor():
    specification, base = load_matrix(Path("configs/gate3_r2_proposed.yaml"))
    assert [run["name"] for run in specification["runs"]] == ["proposed"]
    assert specification["experiment"]["development_only"] is True
    assert base["training"]["monitor"] == "validation_macro_scenario_pr_auc"
    assert base["training"]["deterministic_algorithms"] is True
    assert base["training"]["deterministic_warn_only"] is False


def test_r4b_smoke_matrix_matches_frozen_factorial_and_disables_test_evaluation():
    specification, base = load_matrix(Path("configs/gate3_r4b_smoke.yaml"))
    assert [run["name"] for run in specification["runs"]] == [
        "control_weighted_bce",
        "event_balanced_bce",
        "focal",
        "event_balanced_focal",
    ]
    assert base["evaluation"]["evaluate_test"] is False
    assert base["training"]["monitor"] == "validation_macro_scenario_pr_auc"
    assert base["training"]["deterministic_algorithms"] is True


def test_r4b_formal_campaign_is_frozen_to_fold0_four_by_five():
    import yaml

    specification, base = load_matrix(Path("configs/gate3_r4b_formal.yaml"))
    campaign = yaml.safe_load(
        Path("configs/gate3_r4b_campaign.yaml").read_text(encoding="utf-8")
    )
    assert [run["name"] for run in specification["runs"]] == [
        "control_weighted_bce",
        "event_balanced_bce",
        "focal",
        "event_balanced_focal",
    ]
    assert campaign["folds"] == [0]
    assert campaign["seeds"] == [11, 22, 33, 44, 55]
    assert base["evaluation"]["evaluate_test"] is False
    assert base["training"]["epochs"] == 100
    assert specification["experiment"]["purpose"] == "objective_remediation_formal"


def test_gate4b_smoke_matrix_matches_frozen_pooling_factorial():
    specification, base = load_matrix(Path("configs/gate4b_smoke.yaml"))
    assert [run["name"] for run in specification["runs"]] == [
        "mean_time_mean_sensor",
        "stats_time_mean_sensor",
        "mean_time_stats_sensor",
        "stats_time_stats_sensor",
    ]
    combinations = {
        (
            run["overrides"]["model.temporal_pooling"],
            run["overrides"]["model.sensor_pooling"],
        )
        for run in specification["runs"]
    }
    assert combinations == {
        ("mean", "mean"),
        ("mean_max_std", "mean"),
        ("mean", "mean_max_std"),
        ("mean_max_std", "mean_max_std"),
    }
    assert base["evaluation"]["evaluate_test"] is False
    assert base["training"]["loss_name"] == "weighted_bce"
    assert base["training"]["monitor"] == "validation_macro_scenario_pr_auc"


def test_gate4b_formal_campaign_has_three_new_variants_by_five_seeds():
    import yaml

    specification, base = load_matrix(Path("configs/gate4b_formal.yaml"))
    campaign = yaml.safe_load(
        Path("configs/gate4b_campaign.yaml").read_text(encoding="utf-8")
    )
    assert [run["name"] for run in specification["runs"]] == [
        "stats_time_mean_sensor",
        "mean_time_stats_sensor",
        "stats_time_stats_sensor",
    ]
    assert campaign["folds"] == [0]
    assert campaign["seeds"] == [11, 22, 33, 44, 55]
    assert len(specification["runs"]) * len(campaign["seeds"]) == 15
    assert base["evaluation"]["evaluate_test"] is False
    assert base["training"]["epochs"] == 100
    assert specification["experiment"]["purpose"] == "pooling_redesign_formal"


def test_empty_run_selection_does_not_mean_all_runs(tmp_path):
    from iiot_msresstgat.matrix import run_matrix

    result = run_matrix(
        Path("configs/ablation.yaml"), tmp_path, 0, 11, 1, 8, selected_runs=set()
    )
    assert result.empty
