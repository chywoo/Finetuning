"""Run from any working directory: python /path/to/train.py ..."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune_lab.torch_text import main


if __name__ == "__main__":
    main("instruction")
