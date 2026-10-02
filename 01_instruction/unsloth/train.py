"""Run from the repository root: python 01_instruction/unsloth/train.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune_lab.unsloth_text import main


if __name__ == "__main__":
    raise SystemExit(main(task="instruction"))
