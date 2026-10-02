"""Run the CUDA visual learning exercise from any working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from finetune_lab.vision import main

if __name__ == "__main__":
    main("unsloth")
