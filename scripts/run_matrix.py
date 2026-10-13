import argparse
from pathlib import Path

from iiot_msresstgat.matrix import aggregate_matrix, run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable publication experiment matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pipeline-pilot", action="store_true")
    parser.add_argument("--max-windows-per-split", type=int, default=512)
    parser.add_argument("--runs", nargs="*")
    args = parser.parse_args()
    result = run_matrix(
        args.config,
        args.output_root,
        args.fold,
        args.seed,
        args.epochs,
        args.max_windows_per_split if args.pipeline_pilot else None,
        set(args.runs) if args.runs else None,
    )
    aggregate_matrix(args.output_root)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
