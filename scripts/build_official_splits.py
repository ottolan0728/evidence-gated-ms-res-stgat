from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from iiot_msresstgat.official_data import (
    AttackInterval,
    assign_balanced_folds,
    build_event_blocks,
    reconcile_event_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze scenario-disjoint outer and validation folds for official SWaT"
    )
    parser.add_argument(
        "--audit-manifest", type=Path, default=Path("data/official_dataset_manifest.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/splits/swat_a1_a2_group5.json")
    )
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args()
    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    if not audit["classification"]["gate1_pass"]:
        raise SystemExit("Gate 1 has not passed; refusing to freeze publication splits")
    intervals = [
        AttackInterval(
            scenario_id=item["scenario_id"],
            start=datetime.fromisoformat(item["start"]),
            end=datetime.fromisoformat(item["end"]),
            attack_point=item["attack_point"],
            attack=item["attack"],
        )
        for item in audit["interval_source"]["intervals"]
    ]
    events = reconcile_event_groups(intervals, audit["attack"]["attack_label_runs"])
    assignments = assign_balanced_folds(events, args.fold_count)
    blocks = build_event_blocks(
        events,
        datetime.fromisoformat(audit["attack"]["timestamp_min"]),
        datetime.fromisoformat(audit["attack"]["timestamp_max"]),
    )
    for block in blocks:
        block["outer_fold"] = assignments[block["event_id"]]
    folds = []
    for test_fold in range(args.fold_count):
        validation_fold = (test_fold + 1) % args.fold_count
        folds.append(
            {
                "fold": test_fold,
                "test_event_ids": [
                    event_id for event_id, fold in assignments.items() if fold == test_fold
                ],
                "validation_event_ids": [
                    event_id
                    for event_id, fold in assignments.items()
                    if fold == validation_fold
                ],
                "train_event_ids": [
                    event_id
                    for event_id, fold in assignments.items()
                    if fold not in {test_fold, validation_fold}
                ],
            }
        )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_PUBLICATION_TRAINING",
        "dataset": audit["dataset"],
        "strategy": "five-fold scenario-disjoint evaluation with duration-balanced event groups",
        "label_policy": (
            "Binary targets follow whitespace-normalized provider labels. Event groups are "
            "contiguous labeled attack runs reconciled to the official attack table."
        ),
        "normal_policy": (
            "The seven-day Normal v1 workbook is training-only baseline data. Evaluation "
            "normal context comes from each held-out attack-file event block."
        ),
        "purge_policy": (
            "Windows are constructed within individual event blocks; no window crosses a "
            "block boundary. Apply an additional window_size-1 purge at every boundary."
        ),
        "fold_count": args.fold_count,
        "events": blocks,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "events": len(events),
                "official_scenarios": sum(len(event.scenario_ids) for event in events),
                "provider_label_only_events": sum(not event.scenario_ids for event in events),
                "events_per_fold": {
                    str(fold): sum(value == fold for value in assignments.values())
                    for fold in range(args.fold_count)
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
