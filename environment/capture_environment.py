"""Capture an exact runtime inventory without exposing local paths by default."""
from __future__ import annotations
import importlib.metadata as metadata
import json, platform, sys
from pathlib import Path

KEY_PACKAGES = [
    "numpy","pandas","scikit-learn","torch","torch-geometric",
    "PyYAML","matplotlib","seaborn","openpyxl","psutil",
]

def version_or_none(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

def capture() -> dict:
    payload = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: version_or_none(name) for name in KEY_PACKAGES},
    }
    try:
        import torch
        payload["torch_runtime"] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    except Exception as exc:
        payload["torch_runtime"] = {"available": False, "error": type(exc).__name__}
    return payload

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("environment/runtime_inventory.json"))
    args = parser.parse_args()
    payload = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
