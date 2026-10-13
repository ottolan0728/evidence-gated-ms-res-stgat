from __future__ import annotations

import csv
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path


def audit_csv(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    labels: Counter[str] = Counter()
    seen: set[datetime] = set()
    rows = duplicates = regressions = parse_failures = 0
    previous: datetime | None = None
    minimum: datetime | None = None
    maximum: datetime | None = None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
        reader = csv.reader(source)
        header = [value.strip() for value in next(reader)]
        label_index = header.index("Normal/Attack") if "Normal/Attack" in header else None
        timestamp_index = header.index("Timestamp") if "Timestamp" in header else None
        for row in reader:
            if not row:
                continue
            rows += 1
            if label_index is not None:
                labels[row[label_index].strip()] += 1
            if timestamp_index is None:
                continue
            raw_timestamp = row[timestamp_index].strip()
            try:
                timestamp = datetime.strptime(raw_timestamp, "%d/%m/%Y %I:%M:%S %p")
            except ValueError:
                parse_failures += 1
                continue
            if timestamp in seen:
                duplicates += 1
            seen.add(timestamp)
            if previous is not None and timestamp < previous:
                regressions += 1
            previous = timestamp
            minimum = timestamp if minimum is None or timestamp < minimum else minimum
            maximum = timestamp if maximum is None or timestamp > maximum else maximum
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": rows,
        "columns": len(header),
        "column_names": header,
        "label_counts": dict(labels),
        "unique_timestamps": len(seen),
        "duplicate_timestamps": duplicates,
        "chronology_regressions": regressions,
        "timestamp_parse_failures": parse_failures,
        "timestamp_min": minimum.isoformat() if minimum else None,
        "timestamp_max": maximum.isoformat() if maximum else None,
    }


def classify_dataset(files: list[dict]) -> dict:
    blocking = []
    for item in files:
        if item["timestamp_parse_failures"]:
            blocking.append(f'{Path(item["path"]).name}: timestamp parse failures')
        if item["duplicate_timestamps"]:
            blocking.append(f'{Path(item["path"]).name}: duplicate timestamps')
        if item["chronology_regressions"]:
            blocking.append(f'{Path(item["path"]).name}: chronology regressions')
    attack_files = [item for item in files if item["label_counts"] and set(item["label_counts"]) == {"Attack"}]
    if attack_files:
        blocking.append("attack-only export lacks continuous normal context")
    return {
        "publication_ready": not blocking,
        "development_only": bool(blocking),
        "blocking_findings": blocking,
    }
