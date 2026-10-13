import argparse
from pathlib import Path

from iiot_msresstgat.campaign import (
    campaign_plan,
    campaign_status_dir,
    persist_campaign_plan,
    run_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run the resumable Gate 3 campaign")
    parser.add_argument("--config", type=Path, default=Path("configs/gate3_campaign.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--runs", nargs="*")
    args = parser.parse_args()
    if args.dry_run:
        plan = campaign_plan(args.config)
        persist_campaign_plan(plan, campaign_status_dir(args.config))
    else:
        plan = run_campaign(
            args.config,
            args.epochs,
            set(args.folds) if args.folds else None,
            set(args.seeds) if args.seeds else None,
            set(args.runs) if args.runs else None,
        )
    print(plan.groupby(["matrix", "status"]).size().to_string())


if __name__ == "__main__":
    main()
