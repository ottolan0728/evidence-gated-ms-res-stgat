"""Path-neutral publication reproduction wrapper for R3–R6.

This wrapper is not claimed to be byte-identical to the authorization-heavy peer-review
controllers. It reuses retained scientific model/data/endpoint modules while removing
machine-local path and approval-file dependencies. Only train/validation roles are accepted.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import yaml

STAGE_TO_CONFIG = {
    "r3": "R3_CAPACITY_CONTROL_FREEZE.yaml",
    "r4": "R4_BASELINE_BENCHMARK_FREEZE.yaml",
    "r5": "R5_CROSSFOLD_FREEZE.yaml",
    "r6c": "R6C_GRAPH_SENSITIVITY_FREEZE.yaml",
    "r6d": "R6D_COMPONENT_ABLATION_FREEZE.yaml",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_modules(repo_root: Path):
    module_root = repo_root / "review_experiments" / "r5"
    sys.path.insert(0, str(module_root))
    try:
        from adapters import CostHooks, FitScoreAdapter, NPZRoleShardStore, RoleLimitedDataAdapter
        from common_endpoint import evaluate_validation
        from safe_runner import instantiate_model
    finally:
        try:
            sys.path.remove(str(module_root))
        except ValueError:
            pass
    return RoleLimitedDataAdapter, NPZRoleShardStore, FitScoreAdapter, CostHooks, evaluate_validation, instantiate_model

def validate_config(config: dict, stage: str, model: str, fold: int, seed: int) -> None:
    if config.get("stage", "").upper() != stage.upper():
        raise ValueError("Stage/config mismatch")
    if seed not in config.get("seeds", []):
        raise ValueError("Seed outside frozen set")
    if fold not in config.get("outer_folds", []):
        raise ValueError("Outer fold outside frozen set")
    if model not in config.get("models", []):
        raise ValueError("Model outside frozen set")
    if set(config.get("roles", {}).get("fit", [])) != {"train"}:
        raise ValueError("Training-only fit contract violated")
    if set(config.get("roles", {}).get("score", [])) != {"validation"}:
        raise ValueError("Validation-only score contract violated")
    forbidden = set(config.get("roles", {}).get("forbidden", []))
    if not {"test", "reserved_confirmatory_test"}.issubset(forbidden):
        raise ValueError("Test/reserved roles must be forbidden")
    if config.get("evaluation", {}).get("evaluate_test") is not False:
        raise ValueError("Test evaluation must remain disabled")

def run(args) -> dict:
    repo_root = args.repo_root.resolve()
    config_path = repo_root / "configs" / "frozen" / STAGE_TO_CONFIG[args.stage]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, args.stage, args.model, args.fold, args.seed)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if set(metadata) != {"train", "validation"}:
        raise ValueError("Metadata must contain exactly train and validation")

    RoleLimitedDataAdapter, NPZRoleShardStore, FitScoreAdapter, CostHooks, evaluate_validation, instantiate_model = load_modules(repo_root)

    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    store = NPZRoleShardStore(args.data_root.resolve(), cache_root)
    data = RoleLimitedDataAdapter(config, args.fold, metadata, store)
    train, validation, edge, provenance = data.prepare()

    spec = config["model_specs"][args.model]
    model, parameter_count = instantiate_model(args.model, spec, args.seed)

    import torch
    device = "cpu" if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)

    adapter = FitScoreAdapter(args.model, model, spec, edge, args.seed, device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.bin"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing overwrite: {checkpoint}")

    started = time.time()
    with CostHooks(device) as costs:
        adapter.fit(train, validation, synthetic=False)
        checkpoint_hash = adapter.checkpoint_roundtrip(checkpoint, validation)
        state_hash = adapter.state_digest()
        packet = adapter.score_packet(validation)
    latency_view = adapter.latency_batch(validation)
    costs.latency(lambda: adapter.predict_arrays(latency_view), 64)

    endpoint = evaluate_validation(
        **{k: v for k, v in packet.items() if k != "score_orientation"},
        policy=config["evaluation"],
    )

    result = {
        "wrapper": "path-neutral-publication-reproduction-v1",
        "not_byte_identical_historical_controller": True,
        "stage": config["stage"],
        "evidence_class": config["evidence_class"],
        "model": args.model,
        "outer_fold": args.fold,
        "seed": args.seed,
        "device": device,
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(args.metadata),
        "parameter_count": parameter_count,
        "best_epoch": adapter.best_epoch,
        "epochs_completed": adapter.epochs_completed,
        "checkpoint_sha256": checkpoint_hash,
        "state_sha256": state_hash,
        "endpoint": endpoint,
        "cost": costs.result(),
        "data_provenance": provenance,
        "elapsed_seconds": time.time() - started,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output/"reproduction_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=sorted(STAGE_TO_CONFIG), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=Path("outputs/reproduction_cache"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    print(json.dumps(run(args), indent=2))

if __name__ == "__main__":
    main()
