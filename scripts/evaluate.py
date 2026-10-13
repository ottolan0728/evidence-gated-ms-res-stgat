import argparse
import json
from pathlib import Path

from iiot_msresstgat.evaluation import evaluate_publication_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen checkpoint once")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    output = args.output or args.checkpoint.parent / "evaluation.json"
    result = evaluate_publication_checkpoint(args.checkpoint, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
