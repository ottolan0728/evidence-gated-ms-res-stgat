from pathlib import Path

import json

from iiot_msresstgat.campaign import (
    campaign_plan,
    campaign_status_dir,
    persist_campaign_plan,
)


def test_gate3_campaign_has_expected_unique_runs():
    plan = campaign_plan(Path("configs/gate3_campaign.yaml"))
    assert len(plan) == 375
    assert set(plan["fold"]) == {0, 1, 2, 3, 4}
    assert set(plan["seed"]) == {11, 22, 33, 44, 55}
    assert not ((plan["matrix"] == "ablation") & (plan["run"] == "full")).any()


def test_r2_campaign_is_isolated_and_has_five_strict_macro_monitor_runs(tmp_path):
    campaign_path = Path("configs/gate3_r2_campaign.yaml")
    plan = campaign_plan(campaign_path)
    assert len(plan) == 5
    assert set(plan["fold"]) == {0}
    assert set(plan["seed"]) == {11, 22, 33, 44, 55}
    assert set(plan["run"]) == {"proposed"}
    assert campaign_status_dir(campaign_path) == Path("outputs/gate3/r2_macro_monitor")

    output_dir = tmp_path / "isolated_status"
    persist_campaign_plan(plan, output_dir)
    status = json.loads((output_dir / "campaign_status.json").read_text(encoding="utf-8"))
    assert status["total"] == 5
