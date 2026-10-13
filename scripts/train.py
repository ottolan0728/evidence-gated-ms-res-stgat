import argparse
import json
from dataclasses import asdict
from pathlib import Path

from iiot_msresstgat.experiment import run_development_smoke
from iiot_msresstgat.publication_experiment import run_publication_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible MS-Res-STGAT experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-rows-per-class", type=int, default=2_000)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--publication", action="store_true")
    parser.add_argument(
        "--pipeline-pilot",
        action="store_true",
        help="Limit official data windows and mark outputs as development-only",
    )
    parser.add_argument("--max-windows-per-split", type=int, default=1_024)
    parser.add_argument(
        "--development-smoke",
        action="store_true",
        help="Use the derived local archive for a non-publication smoke test",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.development_smoke and args.publication:
        raise SystemExit("Choose either --development-smoke or --publication")
    if not args.development_smoke and not args.publication:
        raise SystemExit(
            "Choose --development-smoke or --publication. Publication mode verifies Gate 1."
        )
    existing_metrics = args.output_dir / "metrics.json"
    if existing_metrics.exists() and not args.no_resume:
        print(existing_metrics.read_text(encoding="utf-8"))
        return
    if args.development_smoke:
        summary = run_development_smoke(
            args.config,
            args.output_dir,
            args.seed,
            args.epochs,
            args.max_rows_per_class,
        )
    else:
        summary = run_publication_experiment(
            args.config,
            args.output_dir,
            args.fold,
            args.seed,
            args.epochs,
            args.max_windows_per_split if args.pipeline_pilot else None,
            not args.no_resume,
        )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
