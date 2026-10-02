"""Run knowledge CPT or QA SFT from any working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from finetune_lab.hf_text import main

if __name__ == "__main__":
    main("knowledge")
