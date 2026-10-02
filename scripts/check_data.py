"""Audit local manifests, split boundaries and decoded images; no ML training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finetune_lab.validation import validate_data_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--allow-partial", action="store_true", help="Audit only present manifests")
    parser.add_argument("--output", type=Path, help="Write a new report; existing files are protected")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output report already exists; choose a new path")
    try:
        result = validate_data_root(args.data_root, not args.allow_partial)
    except (ValueError, KeyError, OSError, ImportError) as error:
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        parser.exit(2, message.replace(str(ROOT), "<project-root>") + "\n")
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
