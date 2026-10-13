import json
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_official_data import REQUIRED_MEMBERS, extract_required


def test_extract_required_selects_only_publication_files(tmp_path: Path) -> None:
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index, member in enumerate(REQUIRED_MEMBERS):
            bundle.writestr(member, f"content-{index}")
        bundle.writestr("SWaT.A1 & A2_Dec 2015/Network/large.csv", "excluded")

    output_dir = tmp_path / "raw"
    result = extract_required(archive, output_dir, overwrite=False)

    assert {item["local_name"] for item in result["files"]} == {
        Path(member).name for member in REQUIRED_MEMBERS
    }
    assert not (output_dir / "large.csv").exists()
    assert all(len(item["sha256"]) == 64 for item in result["files"])
    json.dumps(result)


def test_extract_required_rejects_incomplete_archive(tmp_path: Path) -> None:
    archive = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(REQUIRED_MEMBERS[0], "only one")

    with pytest.raises(ValueError, match="Required archive members are missing"):
        extract_required(archive, tmp_path / "raw", overwrite=False)
