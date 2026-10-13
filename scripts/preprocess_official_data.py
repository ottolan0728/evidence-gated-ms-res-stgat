from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from iiot_msresstgat.official_data import (
    is_attack_label,
    iter_swat_rows,
    parse_timestamp,
    sha256,
)


def timestamp_seconds(value: object) -> int:
    timestamp = parse_timestamp(value)
    return int(np.datetime64(timestamp, "s").astype(np.int64))


def process_workbook(
    source: Path, output_dir: Path, prefix: str, expected_rows: int
) -> dict:
    header, rows = iter_swat_rows(source)
    timestamp_index = header.index("Timestamp")
    label_index = header.index("Normal/Attack")
    feature_indexes = [
        index for index in range(len(header)) if index not in {timestamp_index, label_index}
    ]
    feature_names = [header[index] for index in feature_indexes]
    partial_paths = {
        "values": output_dir / f"{prefix}_values.partial.npy",
        "timestamps": output_dir / f"{prefix}_timestamps.partial.npy",
        "labels": output_dir / f"{prefix}_labels.partial.npy",
    }
    arrays = {
        "values": open_memmap(
            partial_paths["values"],
            mode="w+",
            dtype=np.float32,
            shape=(expected_rows, len(feature_names)),
        ),
        "timestamps": open_memmap(
            partial_paths["timestamps"], mode="w+", dtype=np.int64, shape=(expected_rows,)
        ),
        "labels": open_memmap(
            partial_paths["labels"], mode="w+", dtype=np.uint8, shape=(expected_rows,)
        ),
    }
    row_count = 0
    for row in rows:
        if not row or all(value is None for value in row):
            continue
        if row_count >= expected_rows:
            raise ValueError(f"{source} contains more than {expected_rows} data rows")
        arrays["timestamps"][row_count] = timestamp_seconds(row[timestamp_index])
        arrays["labels"][row_count] = int(is_attack_label(row[label_index]))
        arrays["values"][row_count] = [float(row[index]) for index in feature_indexes]
        row_count += 1
    if row_count != expected_rows:
        raise ValueError(f"{source} contains {row_count} rows; expected {expected_rows}")
    for array in arrays.values():
        array.flush()
    del array
    del arrays

    outputs = {}
    for kind, partial in partial_paths.items():
        destination = output_dir / f"{prefix}_{kind}.npy"
        partial.replace(destination)
        outputs[kind] = {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    return {
        "source": source.name,
        "rows": expected_rows,
        "features": len(feature_names),
        "feature_names": feature_names,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audited official SWaT workbooks into memory-mappable NumPy arrays"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/swat_a1_a2_2015")
    )
    parser.add_argument(
        "--audit-manifest", type=Path, default=Path("data/official_dataset_manifest.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/swat_a1_a2_2015")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    if not audit["classification"]["gate1_pass"]:
        raise SystemExit("Gate 1 has not passed; refusing to preprocess publication data")
    manifest_path = args.output_dir / "processed_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normal = process_workbook(
        args.data_dir / "SWaT_Dataset_Normal_v1.xlsx",
        args.output_dir,
        "normal",
        audit["normal"]["rows"],
    )
    attack = process_workbook(
        args.data_dir / "SWaT_Dataset_Attack_v0.xlsx",
        args.output_dir,
        "attack",
        audit["attack"]["rows"],
    )
    if normal["feature_names"] != attack["feature_names"]:
        raise ValueError("Normal and attack processed feature schemas differ")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": "NumPy .npy; float32 features, int64 Unix-like seconds, uint8 labels",
        "normal": normal,
        "attack": attack,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "normal_rows": normal["rows"],
                "attack_rows": attack["rows"],
                "features": len(normal["feature_names"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
