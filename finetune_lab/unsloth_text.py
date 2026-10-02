"""Unsloth LoRA/QLoRA text labs; dry-run needs only the standard library."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any
from finetune_lab.paths import project_path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv: list[str] | None = None, task: str = "instruction") -> argparse.Namespace:
    """Validate cheap CLI settings before importing CUDA packages or loading models."""
    if task not in {"instruction", "knowledge"}:
        raise ValueError("task must be instruction or knowledge")
    parser = argparse.ArgumentParser(description=f"Unsloth {task} fine-tuning lab")
    default_model = "HuggingFaceTB/SmolLM2-135M" + ("-Instruct" if task == "knowledge" else "")
    parser.add_argument("--stage", choices=["sft"] if task == "instruction" else ["cpt", "sft"],
                        default="sft" if task == "instruction" else "cpt")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args, parser)
    dataset_name = "instruction" if task == "instruction" else f"knowledge_{args.stage}"
    return argparse.Namespace(**{
        **vars(args), "task": task,
        "data_dir": args.data_dir or REPO_ROOT / "data" / "processed" / dataset_name,
        "output_dir": args.output_dir or REPO_ROOT / "outputs" / f"{task}_unsloth_{args.stage}",
    })


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("max_steps", "batch_size", "gradient_accumulation", "lora_r", "lora_alpha"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_length < 3:
        parser.error("--max-length must be at least 3 (context, completion, EOS)")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.model.strip() or not args.revision.strip():
        parser.error("--model and --revision must be nonempty")


def load_split_records(data_dir: Path, kind: str) -> tuple[list[dict], list[dict]]:
    from finetune_lab.data import read_jsonl, validate_splits

    train_records = read_jsonl(data_dir / "train.jsonl", kind)
    validation_records = read_jsonl(data_dir / "validation.jsonl", kind)
    test_records = read_jsonl(data_dir / "test.jsonl", kind)
    if not train_records or not validation_records:
        raise ValueError("train and validation splits must both contain records")
    validate_splits({"train": train_records, "validation": validation_records, "test": test_records}, kind)
    return train_records, validation_records


def build_dataset_rows(records: list[dict], tokenizer: Any, max_length: int, kind: str) -> list[dict]:
    from finetune_lab.text_encoding import encode_record

    if not records:
        raise ValueError("training dataset is empty")
    return [encode_record(record, tokenizer, max_length, kind) for record in records]


def model_origin(model_name: str, revision: str) -> tuple[str, str, bool]:
    """Keep the original base identity when continuing a locally saved adapter."""
    directory = Path(model_name)
    adapter_config = directory / "adapter_config.json"
    if not adapter_config.is_file():
        return model_name, revision, False
    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must be a JSON object")
    base = config.get("base_model_name_or_path")
    if not isinstance(base, str) or not base:
        raise ValueError("adapter_config.json needs base_model_name_or_path")
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError("training_metadata.json must be a JSON object")
    base_revision = metadata.get("revision") or config.get("revision") or revision
    if not isinstance(base_revision, str) or not base_revision:
        raise ValueError("adapter revision must be a nonempty string")
    return base, base_revision, True


def run_metadata(args: argparse.Namespace) -> dict:
    base_model, revision, _ = model_origin(args.model, args.revision)
    return {
        "base_model": base_model, "revision": revision, "kind": args.stage,
        "prompt_format": "instruction_response_v1", "method": "unsloth",
        "stage": args.stage, "max_length": args.max_length,
        "load_in_4bit": args.load_in_4bit, "seed": args.seed,
        "model_input": project_path(args.model), "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "target_runtime": "supported NVIDIA CUDA runtime",
    }


def load_runtime() -> tuple:
    """Unsloth must patch libraries before importing Transformers or TRL."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Unsloth requires a supported NVIDIA CUDA environment.")
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError("Install the Unsloth profile in a separate supported CUDA environment first.") from error
    return torch, FastLanguageModel, is_bfloat16_supported, Dataset, SFTConfig, SFTTrainer


def load_model(args: argparse.Namespace, fast_language_model: Any) -> tuple:
    _, revision, is_adapter = model_origin(args.model, args.revision)
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=args.model, revision=revision, max_seq_length=args.max_length,
        dtype=None, load_in_4bit=args.load_in_4bit, trust_remote_code=False,
    )
    if not is_adapter:
        model = fast_language_model.get_peft_model(
            model, r=args.lora_r, target_modules=TARGET_MODULES,
            lora_alpha=args.lora_alpha, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
    fast_language_model.for_training(model)
    model.config.use_cache = False
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define EOS and a padding token")
    return model, tokenizer


def training_config(args: argparse.Namespace, config_class: Any, bf16: bool) -> Any:
    return config_class(
        output_dir=str(args.output_dir), max_steps=args.max_steps,
        max_length=args.max_length, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_ratio=0.1, weight_decay=0.01,
        lr_scheduler_type="linear", optim="adamw_torch", bf16=bf16, fp16=not bf16,
        seed=args.seed, data_seed=args.seed, logging_steps=1,
        save_strategy="no", eval_strategy="no", report_to="none", logging_nan_inf_filter=False,
        dataloader_num_workers=0, prediction_loss_only=True,
        packing=False, completion_only_loss=args.stage == "sft", loss_type="nll",
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
    )


def guarded_trainer_class(trainer_base: Any, torch: Any) -> type:
    """Reject a non-finite model loss before backward/optimizer updates."""
    class FiniteLossSFTTrainer(trainer_base):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError("Non-finite model loss. Check data, precision, and learning rate.")
            return result

    return FiniteLossSFTTrainer


def validate_metrics(metrics: dict, context: str) -> None:
    for name, value in metrics.items():
        if isinstance(value, dict):
            validate_metrics(value, f"{context}.{name}")
        elif isinstance(value, Real) and not math.isfinite(value):
            raise FloatingPointError(f"Non-finite metric {context}.{name}: {value}")


def save_artifacts(args: argparse.Namespace, model: Any, tokenizer: Any, trainer: Any, metrics: dict) -> None:
    validate_metrics(metrics, "run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()
    metadata = run_metadata(args)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if isinstance(resolved_revision, str) and resolved_revision:
        metadata = {**metadata, "revision": resolved_revision}
    adapter_path = args.output_dir / "adapter_config.json"
    if adapter_path.is_file():
        adapter_config = json.loads(adapter_path.read_text(encoding="utf-8"))
        pinned_config = {**adapter_config, "revision": metadata["revision"]}
        adapter_path.write_text(json.dumps(pinned_config, indent=2), encoding="utf-8")
        metadata = {**metadata, "lora_r": adapter_config.get("r", args.lora_r),
                    "lora_alpha": adapter_config.get("lora_alpha", args.lora_alpha)}
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train(args: argparse.Namespace, train_records: list[dict], validation_records: list[dict]) -> dict:
    from finetune_lab.hf_text import _validate_output_dir
    _validate_output_dir(args.output_dir)
    torch, fast_model, supports_bf16, dataset_class, config_class, trainer_class = load_runtime()
    from finetune_lab.text_encoding import CausalLMCollator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = load_model(args, fast_model)
    train_dataset = dataset_class.from_list(build_dataset_rows(train_records, tokenizer, args.max_length, args.stage))
    eval_dataset = dataset_class.from_list(build_dataset_rows(validation_records, tokenizer, args.max_length, args.stage))
    trainer = guarded_trainer_class(trainer_class, torch)(
        model=model, processing_class=tokenizer,
        args=training_config(args, config_class, supports_bf16()),
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )
    before = trainer.evaluate()
    validate_metrics(before, "validation_before")
    training_result = trainer.train()
    after = trainer.evaluate()
    metrics = {"validation_before": before, "validation_after": after, "train": training_result.metrics}
    save_artifacts(args, model, tokenizer, trainer, metrics)
    return metrics


def main(argv: list[str] | None = None, task: str = "instruction") -> int:
    args = parse_args(argv, task)
    train_records, validation_records = load_split_records(args.data_dir, args.stage)
    plan = {
        **run_metadata(args), "task": task, "dry_run": args.dry_run,
        "data_dir": project_path(args.data_dir), "output_dir": project_path(args.output_dir),
        "train_records": len(train_records), "validation_records": len(validation_records),
        "required_runtime": "supported NVIDIA CUDA runtime",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(train(args, train_records, validation_records), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
