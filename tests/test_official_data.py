from datetime import datetime, time
from pathlib import Path

from openpyxl import Workbook

import numpy as np

from iiot_msresstgat.official_data import (
    AttackInterval,
    assign_balanced_folds,
    audit_swat_workbook,
    load_attack_intervals,
    reconcile_event_groups,
)
from scripts.preprocess_official_data import process_workbook


def save_swat_workbook(path: Path, labels: list[str]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append([None, "P1", None])
    worksheet.append([" Timestamp", "FIT101", "Normal/Attack"])
    for offset, label in enumerate(labels):
        worksheet.append(
            [f" 28/12/2015 10:00:0{offset} AM", float(offset), label]
        )
    workbook.save(path)


def test_audit_swat_workbook_reads_two_row_header(tmp_path: Path) -> None:
    path = tmp_path / "attack.xlsx"
    save_swat_workbook(path, ["Normal", "A ttack", "Normal"])

    result = audit_swat_workbook(path)

    assert result["rows"] == 3
    assert result["columns"] == 3
    assert result["label_counts"] == {"Normal": 2, "A ttack": 1}
    assert result["normalized_label_counts"] == {"Normal": 2, "Attack": 1}
    assert result["attack_label_runs"] == [
        {"start": "2015-12-28T10:00:01", "end": "2015-12-28T10:00:01"}
    ]
    assert result["duplicate_timestamps"] == 0
    assert result["chronology_regressions"] == 0


def test_load_attack_intervals_excludes_nonphysical_and_corrects_year(tmp_path: Path) -> None:
    path = tmp_path / "intervals.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(
        [
            "Attack #",
            "Start Time",
            "End Time",
            "Attack Point",
            "Start State",
            "Attack",
        ]
    )
    worksheet.append(
        [1, datetime(2015, 12, 28, 10, 0), time(10, 1), "MV-101", "closed", "open"]
    )
    worksheet.append(
        [2, datetime(2015, 12, 28, 11, 0), None, "No Physical Impact Attack", None, None]
    )
    worksheet.append(
        [37, datetime(2015, 1, 2, 11, 0), time(11, 1), "P-501", "on", "off"]
    )
    workbook.save(path)

    intervals, metadata = load_attack_intervals(path)

    assert [interval.scenario_id for interval in intervals] == [1, 37]
    assert intervals[1].start.year == 2016
    assert metadata["corrected_year_scenarios"] == [37]
    assert metadata["excluded_rows"] == [
        {"scenario_id": 2, "reason": "no_completed_physical_interval"}
    ]


def test_reconcile_event_groups_merges_adjacent_official_scenarios() -> None:
    intervals = [
        AttackInterval(10, datetime(2015, 12, 28, 14, 16), datetime(2015, 12, 28, 14, 19), "FIT", "low"),
        AttackInterval(11, datetime(2015, 12, 28, 14, 19), datetime(2015, 12, 28, 14, 28), "FIT", "zero"),
    ]
    runs = [
        {"start": "2015-12-28T14:16:00", "end": "2015-12-28T14:28:00"},
        {"start": "2015-12-29T06:30:00", "end": "2015-12-29T06:42:00"},
    ]

    events = reconcile_event_groups(intervals, runs)
    assignments = assign_balanced_folds(events, fold_count=2)

    assert events[0].scenario_ids == (10, 11)
    assert events[1].scenario_ids == ()
    assert events[1].source == "provider_label_only"
    assert set(assignments) == {"event_01", "event_02"}


def test_process_workbook_creates_memory_mappable_arrays(tmp_path: Path) -> None:
    source = tmp_path / "data.xlsx"
    save_swat_workbook(source, ["Normal", "A ttack", "Normal"])
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    result = process_workbook(source, output_dir, "sample", expected_rows=3)

    values = np.load(output_dir / "sample_values.npy", mmap_mode="r")
    labels = np.load(output_dir / "sample_labels.npy", mmap_mode="r")
    assert values.shape == (3, 1)
    assert labels.tolist() == [0, 1, 0]
    assert result["feature_names"] == ["FIT101"]
