from iiot_msresstgat.audit import classify_dataset


def test_attack_only_file_is_development_only():
    result = classify_dataset(
        [
            {
                "path": "attack.csv",
                "label_counts": {"Attack": 10},
                "timestamp_parse_failures": 0,
                "duplicate_timestamps": 0,
                "chronology_regressions": 0,
            }
        ]
    )
    assert result["development_only"] is True
    assert result["publication_ready"] is False

