"""Merge a text LoRA adapter into a fresh full-precision base model locally."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("--adapter must contain adapter_config.json")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; preserve existing artifacts")
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    metadata_path = args.adapter / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("kind") == "vision":
        parser.error("This merge lesson supports text adapters; use VLM-specific loading for vision")
    config = PeftConfig.from_pretrained(args.adapter)
    model_id = metadata.get("base_model") or config.base_model_name_or_path
    revision = metadata.get("revision") or config.revision or "main"
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                              trust_remote_code=False, use_safetensors=True)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output_dir)
    result = {**metadata, "method": "merged_full", "merged_from": str(args.adapter),
              "base_model": model_id, "revision": revision}
    (args.output_dir / "training_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Merged local checkpoint: {args.output_dir}")


if __name__ == "__main__":
    main()
