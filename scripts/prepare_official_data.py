from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


REQUIRED_MEMBERS = (
    "SWaT.A1 & A2_Dec 2015/List_of_attacks_Final.pdf",
    "SWaT.A1 & A2_Dec 2015/List_of_attacks_Final.xlsx",
    "SWaT.A1 & A2_Dec 2015/Physical/readme.txt",
    "SWaT.A1 & A2_Dec 2015/Physical/SWaT_Dataset_Attack_v0.xlsx",
    "SWaT.A1 & A2_Dec 2015/Physical/SWaT_Dataset_Normal_v1.xlsx",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(output_dir: Path, member: str) -> Path:
    name = PurePosixPath(member).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Unsafe archive member: {member}")
    return output_dir / name


def extract_required(
    archive: Path, output_dir: Path, overwrite: bool, hash_archive: bool = False
) -> dict:
    archive = archive.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict] = []
    with zipfile.ZipFile(archive) as bundle:
        available = set(bundle.namelist())
        missing = [member for member in REQUIRED_MEMBERS if member not in available]
        if missing:
            raise ValueError(f"Required archive members are missing: {missing}")
        for member in REQUIRED_MEMBERS:
            info = bundle.getinfo(member)
            destination = safe_destination(output_dir, member)
            if destination.exists() and not overwrite:
                if destination.stat().st_size != info.file_size:
                    raise FileExistsError(
                        f"Existing file has an unexpected size: {destination}. "
                        "Use --overwrite only after reviewing it."
                    )
                status = "reused"
            else:
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with bundle.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                temporary.replace(destination)
                status = "extracted"
            extracted.append(
                {
                    "archive_member": member,
                    "local_name": destination.name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                    "status": status,
                }
            )
    archive_stat = archive.stat()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive_name": archive.name,
            "archive_bytes": archive_stat.st_size,
            "archive_modified_at": datetime.fromtimestamp(
                archive_stat.st_mtime, timezone.utc
            ).isoformat(),
            "archive_sha256": sha256(archive) if hash_archive else None,
            "archive_sha256_status": "complete" if hash_archive else "pending_full_archive_hash",
        },
        "selection_rule": (
            "Official SWaT A1/A2 physical telemetry, corrected normal version 1, "
            "attack intervals, and provider documentation; network captures excluded."
        ),
        "files": extracted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selectively extract publication-relevant files from the official SWaT archive"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/swat_a1_a2_2015")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/official_source_manifest.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--hash-archive",
        action="store_true",
        help="Read the entire source archive and record its SHA-256 (slow for 101 GB)",
    )
    args = parser.parse_args()

    manifest = extract_required(
        args.archive, args.output_dir, args.overwrite, hash_archive=args.hash_archive
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
