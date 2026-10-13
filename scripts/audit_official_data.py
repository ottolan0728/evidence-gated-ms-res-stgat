from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from iiot_msresstgat.official_data import (
    audit_swat_workbook,
    classify_official_dataset,
    load_attack_intervals,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official licensed SWaT A1/A2 workbooks")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/swat_a1_a2_2015")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/official_dataset_manifest.json")
    )
    args = parser.parse_args()
    interval_path = args.data_dir / "List_of_attacks_Final.xlsx"
    normal_path = args.data_dir / "SWaT_Dataset_Normal_v1.xlsx"
    attack_path = args.data_dir / "SWaT_Dataset_Attack_v0.xlsx"

    intervals, interval_metadata = load_attack_intervals(interval_path)
    normal = audit_swat_workbook(normal_path)
    attack = audit_swat_workbook(attack_path, intervals)
    headers_match = normal["column_names"] == attack["column_names"]
    classification = classify_official_dataset(normal, attack, intervals, headers_match)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "SWaT A1/A2 December 2015",
        "interval_source": {
            "path": str(interval_path.resolve()),
            **interval_metadata,
            "intervals": [interval.to_dict() for interval in intervals],
        },
        "normal": normal,
        "attack": attack,
        "headers_match": headers_match,
        "classification": classification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(classification, indent=2))


if __name__ == "__main__":
    main()
