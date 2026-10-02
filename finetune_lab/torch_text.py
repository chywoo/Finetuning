"""Direct PyTorch causal-LM training loop used by the two text tutorials."""

import argparse
from contextlib import nullcontext
import json
import math
import random
from pathlib import Path

from finetune_lab.data import read_jsonl, validate_splits
from finetune_lab.text_encoding import CausalLMCollator, encode_record
from finetune_lab.paths import project_path


ROOT = Path(__file__).resolve().parents[1]


def build_parser(task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"PyTorch {task} fine-tuning tutorial")
    if task == "knowledge":
        parser.add_argument("--stage", choices=("cpt", "sft"), default="cpt")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    default_model = "HuggingFaceTB/SmolLM2-135M" + ("-Instruct" if task == "knowledge" else "")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--method", choices=("full", "lora"), default="full")
    parser.add_argument("--max-steps", type=int, default=20, help="Optimizer updates, not microbatches")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto",
                        help="CUDA compute autocast dtype; weights and optimizer remain float32")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without ML dependencies")
    parser.add_argument("--smoke-model", action="store_true", help="Use a tiny random offline GPT-2")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.smoke_model and args.method != "full":
        raise ValueError("smoke-model supports method full; adapters require a reloadable base model")
    for name in ("max_steps", "batch_size", "gradient_accumulation"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_length < 3:
        raise ValueError("max-length must be at least 3")
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative and finite")


def select_device(torch, requested: str) -> str:
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": has_mps}
    if requested == "auto":
        return "cuda" if available["cuda"] else "mps" if has_mps else "cpu"
    if not available[requested]:
        raise ValueError(f"Requested device {requested} is unavailable")
    return requested


def select_compute_dtype(torch, device: str, requested: str) -> str:
    """Prefer BF16 computation on supported CUDA, keeping FP32 parameters."""
    if requested == "float32":
        return "float32"
    bf16_available = device == "cuda" and torch.cuda.is_bf16_supported()
    if requested == "bfloat16" and not bf16_available:
        raise ValueError("bfloat16 requires a CUDA device with BF16 support")
    return "bfloat16" if bf16_available else "float32"


def compute_autocast(torch, device: str, compute_dtype: str):
    if compute_dtype == "bfloat16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return nullcontext()


def load_training_data(data_dir: Path, kind: str) -> tuple[list[dict], list[dict], int]:
    """Check all split boundaries; test records never enter a training loader."""
    splits = {name: read_jsonl(data_dir / f"{name}.jsonl", kind)
              for name in ("train", "validation", "test")}
    validate_splits(splits, kind)
    return splits["train"], splits["validation"], len(splits["test"])


def create_smoke_model(max_length: int = 256):
    """Offline random model for execution checks, never a quality benchmark."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    words = ("[UNK]", "[PAD]", "[EOS]", "###", "Instruction", ":", "Response", "hello", "world",
             "The", "a", "is", "blue", "red", "answer", "one", "two", "three", "yes", "no")
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                        eos_token="[EOS]", model_max_length=max_length)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=max_length, n_ctx=max_length,
                        n_embd=32, n_layer=2, n_head=2, bos_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return GPT2LMHeadModel(config), tokenizer


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.smoke_model:
        model, tokenizer = create_smoke_model(args.max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                    torch_dtype=torch.float32, trust_remote_code=False,
                                                    use_safetensors=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("This tutorial requires a tokenizer with an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                            target_modules="all-linear", task_type=TaskType.CAUSAL_LM,
                            revision=getattr(model.config, "_commit_hash", None) or args.revision)
        model = get_peft_model(model, config)
    return model, tokenizer


def _move_batch(batch: dict, device: str) -> dict:
    return {name: values.to(device) for name, values in batch.items()}


def validation_loss(model, loader, device: str, compute_dtype: str = "float32") -> float:
    """Token-weighted mean: padded tokens and the causal first token do not count."""
    import torch

    was_training = model.training
    model.eval()
    numerator, denominator = 0.0, 0
    try:
        with torch.no_grad():
            for batch in loader:
                target_count = int((batch["labels"][:, 1:] != -100).sum().item())
                if not target_count:
                    raise ValueError("Validation contains no causal target tokens")
                with compute_autocast(torch, device, compute_dtype):
                    loss = model(**_move_batch(batch, device)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Validation produced a non-finite loss")
                numerator += float(loss.item()) * target_count
                denominator += target_count
    finally:
        model.train(was_training)
    if denominator == 0:
        raise ValueError("Validation contains no causal target tokens")
    return numerator / denominator


def _accumulation_windows(loader, accumulation: int):
    """Flush the final short window instead of discarding its examples."""
    iterator = iter(loader)
    while True:
        window = []
        for _ in range(accumulation):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            return
        yield window


def train_loop(model, train_loader, args, device: str, compute_dtype: str = "float32") -> dict:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters; check the selected tuning method")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    updates, epochs, numerator, denominator = 0, 0, 0.0, 0
    while updates < args.max_steps:
        epochs += 1
        epoch_updates = updates
        for window in _accumulation_windows(train_loader, args.gradient_accumulation):
            token_counts = [int((batch["labels"][:, 1:] != -100).sum().item()) for batch in window]
            window_tokens = sum(token_counts)
            if window_tokens == 0:
                raise ValueError("Training batch has no causal target tokens")
            optimizer.zero_grad(set_to_none=True)
            for batch, token_count in zip(window, token_counts):
                with compute_autocast(torch, device, compute_dtype):
                    loss = model(**_move_batch(batch, device)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced a non-finite loss")
                (loss * (token_count / window_tokens)).backward()
                numerator += float(loss.detach().item()) * token_count
                denominator += token_count
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            print(json.dumps({"step": updates, "train_loss": numerator / denominator}, ensure_ascii=False), flush=True)
            if updates >= args.max_steps:
                break
        if updates == epoch_updates:
            raise ValueError("Training loader is empty")
    return {"optimizer_steps": updates, "epochs_started": epochs, "train_loss": numerator / denominator,
            "train_target_tokens_seen": denominator}


def main(task: str, argv: list[str] | None = None) -> None:
    parser = build_parser(task)
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        kind = getattr(args, "stage", "sft")
        task_directory = "instruction" if task == "instruction" else f"knowledge_{kind}"
        output_name = "pytorch" if task == "instruction" else f"pytorch-{kind}"
        args.output_dir = args.output_dir or ROOT / "outputs" / task / output_name
        if not args.dry_run:
            from finetune_lab.hf_text import _validate_output_dir
            _validate_output_dir(args.output_dir)
        data_dir = args.data_dir or ROOT / "data" / "processed" / task_directory
        train_rows, validation_rows, test_examples = load_training_data(data_dir, kind)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    base_model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
    summary = {"task": task, "kind": kind, "method": args.method, "model": args.model,
               "base_model": base_model, "revision": args.revision, "requested_revision": args.revision,
               "requested_dtype": args.dtype, "prompt_format": "### Instruction:\n{prompt}\n\n### Response:\n",
               "data_dir": project_path(data_dir),
               "train_examples": len(train_rows), "validation_examples": len(validation_rows),
               "test_examples": test_examples,
               "output_dir": project_path(args.output_dir), "smoke_model": args.smoke_model}
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}, indent=2, ensure_ascii=False))
        return

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch, args.device)
    compute_dtype = select_compute_dtype(torch, device, args.dtype)
    model, tokenizer = load_model_and_tokenizer(args)
    model.to(device)
    train_data = [encode_record(row, tokenizer, args.max_length, kind) for row in train_rows]
    validation_data = [encode_record(row, tokenizer, args.max_length, kind) for row in validation_rows]
    collator = CausalLMCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, generator=generator)
    eval_loader = DataLoader(validation_data, batch_size=args.batch_size, collate_fn=collator)
    before_loss = validation_loss(model, eval_loader, device, compute_dtype)
    metrics = train_loop(model, train_loader, args, device, compute_dtype)
    after_loss = validation_loss(model, eval_loader, device, compute_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {**summary, **metrics, "seed": args.seed, "device": device,
                "revision": getattr(model.config, "_commit_hash", None) or args.revision,
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "compute_dtype": compute_dtype, "parameter_dtype": "float32",
                "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
                "max_length": args.max_length, "learning_rate": args.learning_rate,
                "validation_loss_before": before_loss, "validation_loss_after": after_loss,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "torch_version": torch.__version__, "sft_template": "explicit_instruction_response_v1",
                "validation_perplexity_after": math.exp(min(after_loss, 700)),
                "smoke_warning": "Random smoke model validates execution only" if args.smoke_model else None}
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main("instruction")
