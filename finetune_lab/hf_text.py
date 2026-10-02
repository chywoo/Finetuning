"""Small causal-LM experiments using Transformers Trainer and PEFT.

Imports are deliberately deferred: --help and --dry-run work without torch.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROMPT_FORMAT = "instruction_response_v1"


@dataclass(frozen=True)
class ModelSpec:
    base_model: str
    revision: str
    adapter_path: str | None = None


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 지정하세요.")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 유한한 수를 지정하세요.")
    return number


def parse_args(task: str, argv: list[str] | None = None) -> argparse.Namespace:
    if task not in {"instruction", "knowledge"}:
        raise ValueError(f"Unknown task: {task}")
    parser = argparse.ArgumentParser(description=f"HF Trainer: {task}")
    if task == "knowledge":
        parser.add_argument("--stage", choices=("cpt", "sft"), default="sft")
    parser.add_argument("--model", default=("HuggingFaceTB/SmolLM2-135M" if task == "instruction"
                                           else "HuggingFaceTB/SmolLM2-135M-Instruct"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=("full", "lora", "qlora"), default="lora")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--max-steps", type=_positive_int, default=20)
    parser.add_argument("--max-length", type=_positive_int, default=256)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--gradient-accumulation", type=_positive_int, default=4)
    parser.add_argument("--learning-rate", type=_positive_float)
    parser.add_argument("--lora-r", type=_positive_int, default=8)
    parser.add_argument("--lora-alpha", type=_positive_int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="데이터 형식 검사만 수행")
    parser.add_argument("--smoke-model", action="store_true", help="외부 다운로드 없는 작은 임의 초기화 모델")
    args = parser.parse_args(argv)
    if args.max_length < 3:
        parser.error("--max-length는 prefix/target/EOS를 위해 3 이상이어야 합니다.")
    if args.smoke_model and args.method != "full":
        parser.error("--smoke-model은 재로드 가능한 전체 모델 저장을 위해 --method full을 사용하세요.")
    stage = getattr(args, "stage", "sft")
    dataset_name = task if task == "instruction" else f"knowledge_{stage}"
    return argparse.Namespace(**{
        **vars(args), "stage": stage, "task": task, "kind": stage,
        "data_dir": args.data_dir or ROOT / "data" / "processed" / dataset_name,
        "output_dir": args.output_dir or ROOT / "outputs" / f"{task}_hf_{stage}_{args.method}",
        "learning_rate": args.learning_rate or (2e-5 if args.method == "full" else 2e-4),
    })


def local_model_spec(model: str, revision: str, method: str) -> ModelSpec:
    """Distinguish a full checkpoint from an adapter, preserving its base revision."""
    config_path = Path(model) / "adapter_config.json"
    if not config_path.is_file():
        return ModelSpec(str(Path(model).resolve()) if Path(model).is_dir() else model, revision)
    if method == "full":
        raise ValueError("adapter를 --method full로 로드할 수 없습니다. 먼저 merge한 전체 모델을 사용하세요.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json은 JSON object여야 합니다.")
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("adapter_config.json의 base_model_name_or_path가 유효하지 않습니다.")
    metadata_path = Path(model) / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        raise ValueError("training_metadata.json은 JSON object여야 합니다.")
    base_revision = metadata.get("revision") or config.get("revision") or revision
    if not isinstance(base_revision, str) or not base_revision.strip():
        raise ValueError("adapter의 base revision이 유효하지 않습니다.")
    base_model = str(Path(base_model).resolve()) if Path(base_model).is_dir() else base_model
    return ModelSpec(base_model, base_revision, str(Path(model).resolve()))


def check_finite_metrics(metrics: dict[str, Any]) -> None:
    for name, value in metrics.items():
        if "loss" in name and isinstance(value, (float, int)) and not math.isfinite(value):
            raise RuntimeError(f"{name} must be finite, got {value}. Reduce learning rate / check data.")


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_output_dir(path: Path) -> None:
    """Reject saved or partial training runs before imports, downloads, or writes."""
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Output path must be a directory: {path}")
    known_artifacts = {
        "config.json", "adapter_config.json", "training_metadata.json", "metrics.json",
        "tokenizer.json", "tokenizer_config.json", "trainer_state.json", "training_args.bin",
    }
    artifacts = [entry.name for entry in path.iterdir()
                 if entry.name in known_artifacts or entry.name.startswith("checkpoint-")
                 or entry.suffix in {".safetensors", ".bin"}]
    if artifacts:
        raise ValueError(f"기존 학습 결과가 있습니다: {path} ({', '.join(sorted(artifacts))}). "
                         "덮어쓰기를 막기 위해 새 --output-dir을 지정하세요.")


def _device(torch: Any, requested: str) -> str:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": mps_available}
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available[requested]:
        raise ValueError(f"요청한 device={requested}를 사용할 수 없습니다.")
    return requested


def _load_model(args: argparse.Namespace, device: str, torch: Any) -> tuple[Any, Any, ModelSpec]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if args.smoke_model:
        if args.method != "full":
            raise ValueError("--smoke-model은 재로드 가능한 전체 모델 저장을 위해 --method full을 사용하세요.")
        from finetune_lab.torch_text import create_smoke_model
        model, tokenizer = create_smoke_model(args.max_length)
        spec = ModelSpec("offline-smoke-model", "local")
    else:
        spec = local_model_spec(args.model, args.revision, args.method)
        tokenizer_source = spec.adapter_path or spec.base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=spec.revision,
                                                   trust_remote_code=False)
        kwargs: dict[str, Any] = {"revision": spec.revision, "trust_remote_code": False,
                                  "torch_dtype": torch.float32}
        if args.method == "qlora":
            if device != "cuda":
                raise ValueError("이 실습의 QLoRA는 NVIDIA CUDA GPU가 필요합니다. CPU/MPS에서는 --method lora.")
            if torch.cuda.device_count() != 1:
                raise ValueError("QLoRA는 단일 GPU 실습입니다. CUDA_VISIBLE_DEVICES=0으로 한 장만 노출하세요.")
            kwargs = {**kwargs, "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                      "device_map": {"": torch.cuda.current_device()},
                      "quantization_config": BitsAndBytesConfig(
                          load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                          bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)}
        model = AutoModelForCausalLM.from_pretrained(spec.base_model, **kwargs)
        resolved_revision = getattr(model.config, "_commit_hash", None)
        if isinstance(resolved_revision, str) and resolved_revision:
            spec = ModelSpec(spec.base_model, resolved_revision, spec.adapter_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("EOS token이 있는 causal LM tokenizer를 사용하세요.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.method in {"lora", "qlora"}:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        if args.method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        if spec.adapter_path:
            model = PeftModel.from_pretrained(model, spec.adapter_path, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
                lora_dropout=0.05, target_modules="all-linear", bias="none", revision=spec.revision))
        model.print_trainable_parameters()
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("학습 가능한 parameter가 없습니다.")
    return model, tokenizer, spec


def _train(args: argparse.Namespace, rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    _validate_output_dir(args.output_dir)
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from finetune_lab.text_encoding import CausalLMCollator, encode_record

    device = _device(torch, args.device)
    set_seed(args.seed)
    model, tokenizer, spec = _load_model(args, device, torch)
    encoded = {split: [encode_record(row, tokenizer, args.max_length, args.kind) for row in records]
               for split, records in rows.items() if split != "test"}

    class FiniteLossTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False, **kwargs: Any) -> Any:
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss).all().item()):
                raise RuntimeError("Loss must be finite. Reduce learning rate / check encoded targets.")
            return result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
        optim="adamw_torch", eval_strategy="no", save_strategy="no", logging_steps=1,
        logging_nan_inf_filter=False, report_to="none", seed=args.seed, data_seed=args.seed,
        use_cpu=device == "cpu", bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=args.method == "qlora" and not torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=device == "cuda", remove_unused_columns=False,
        gradient_checkpointing=args.method == "qlora", prediction_loss_only=True, label_names=["labels"],
    )
    if train_args.device.type != device:
        raise ValueError(f"Trainer chose {train_args.device.type}, requested {device}; use a separate process with the requested device visible.")
    trainer = FiniteLossTrainer(model=model, args=train_args, train_dataset=encoded["train"],
                                eval_dataset=encoded["validation"],
                                data_collator=CausalLMCollator(tokenizer.pad_token_id))
    before = trainer.evaluate(metric_key_prefix="before")
    check_finite_metrics(before)
    result = trainer.train()
    after = trainer.evaluate(metric_key_prefix="after")
    metrics = {**before, **result.metrics, **after}
    check_finite_metrics(metrics)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    metadata = {"base_model": spec.base_model, "revision": spec.revision,
                "requested_revision": args.revision,
                "kind": args.kind, "task": args.task, "stage": args.stage,
                "prompt_format": PROMPT_FORMAT if args.kind == "sft" else "plain_text",
                "method": args.method, "framework": "huggingface", "max_length": args.max_length,
                "seed": args.seed, "max_steps": args.max_steps,
                "source_adapter": spec.adapter_path, "data_dir": str(args.data_dir.resolve()),
                "samples": {split: len(records) for split, records in rows.items()}, "metrics": metrics,
                "runtime": {"device": device, "machine": platform.machine(),
                            "python": platform.python_version(), "cuda": torch.version.cuda,
                            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                            "gpu_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
                            "packages": _package_versions()}}
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main(task: str = "instruction", argv: list[str] | None = None) -> None:
    args = parse_args(task, argv)
    from finetune_lab.data import read_jsonl, validate_splits

    rows = {split: read_jsonl(args.data_dir / f"{split}.jsonl", args.kind)
            for split in ("train", "validation", "test")}
    validate_splits(rows, args.kind)
    if any(not records for records in rows.values()):
        raise ValueError("train/validation/test는 각각 최소 한 행이 있어야 합니다.")
    if args.dry_run:
        print(json.dumps({"status": "data_validated", "task": task, "kind": args.kind,
                          "method": args.method, "model": args.model,
                          "samples": {split: len(records) for split, records in rows.items()}}, ensure_ascii=False))
        return
    metadata = _train(args, rows)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "metrics": metadata["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
