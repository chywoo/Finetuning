"""DPO with a fresh LoRA adapter on a full SFT checkpoint; CUDA runtime."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

from finetune_lab.post_data import (
    ROOT, check_trl_version, dataset_manifest, read_preferences, require_cuda,
    require_fresh_output, require_full_checkpoint, validate_finite_values, write_report, project_path,
)
from finetune_lab.text_encoding import encode_record, format_prompt


def parse_args(mode="train", argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Full SFT checkpoint for train; full or adapter for evaluate")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/demo/post_preferences")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    if mode == "train":
        parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/post_dpo")
        parser.add_argument("--max-steps", type=int, default=20)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--gradient-accumulation", type=int, default=4)
        parser.add_argument("--learning-rate", type=float, default=5e-5)
        parser.add_argument("--beta", type=float, default=0.1)
    else:
        parser.add_argument("--split", choices=("validation", "test"), default="test")
        parser.add_argument("--output", type=Path, default=ROOT / "outputs/post_dpo-evaluation.json")
    args = parser.parse_args(argv)
    if min(args.max_length, args.max_prompt_length, args.max_completion_length) < 3:
        parser.error("token limits must be >= 3")
    if args.max_prompt_length + args.max_completion_length > args.max_length:
        parser.error("prompt + completion limits must fit --max-length")
    if mode == "train":
        if min(args.max_steps, args.batch_size, args.gradient_accumulation) < 1:
            parser.error("step and batch limits must be positive")
        if any(not math.isfinite(value) or value <= 0 for value in (args.beta, args.learning_rate)):
            parser.error("beta and learning-rate must be positive and finite")
    return args


def validate_preference_lengths(row, tokenizer, args, context_limit=None):
    prompt = format_prompt(row["prompt"])
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_length > args.max_prompt_length:
        raise ValueError(f"Prompt too long: {row['id']}; increase max-prompt-length")
    for key in ("chosen", "rejected"):
        answer_length = len(tokenizer.encode(row[key], add_special_tokens=False)) + 1
        if answer_length > args.max_completion_length or prompt_length + answer_length > args.max_length:
            raise ValueError(f"Complete answer + EOS exceeds limit: {row['id']}/{key}")
    if context_limit and prompt_length + args.max_completion_length > context_limit:
        raise ValueError(f"Prompt + generation exceeds model context length: {row['id']}")
    return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}


def preference_dataset(rows, tokenizer, args, context_limit=None):
    """Reject truncation so that a preference label still compares complete answers."""
    from datasets import Dataset
    records = [validate_preference_lengths(row, tokenizer, args, context_limit) for row in rows]
    return Dataset.from_list(records)


def train(args, splits):
    model_path = require_full_checkpoint(args.model)
    require_fresh_output(args.output_dir)
    versions = check_trl_version()
    torch = require_cuda()
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                trust_remote_code=False, use_safetensors=True)
    model.config.use_cache = False
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    context_limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
    datasets = {split: preference_dataset(splits[split], tokenizer, args, context_limit)
                for split in ("train", "validation")}
    config = DPOConfig(
        output_dir=str(args.output_dir), max_steps=args.max_steps, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, beta=args.beta, loss_type="sigmoid",
        max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length, reference_free=False,
        precompute_ref_log_probs=False, disable_dropout=True, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", eval_strategy="no", save_strategy="no", logging_steps=1,
        report_to="none", seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
    )

    class FiniteDPOTrainer(DPOTrainer):
        def compute_loss(self, *arguments, **kwargs):
            value = super().compute_loss(*arguments, **kwargs)
            loss = value[0] if isinstance(value, tuple) else value
            if not torch.isfinite(loss).all():
                raise RuntimeError("Non-finite DPO loss")
            return value

    trainer = FiniteDPOTrainer(
        model=model, ref_model=None, args=config, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                               lora_dropout=0.0, target_modules="all-linear", revision=revision),
    )
    if trainer.accelerator.device.type != "cuda":
        raise RuntimeError("DPO Trainer must use CUDA")
    before = trainer.evaluate(metric_key_prefix="before")
    trained = trainer.train()
    after = trainer.evaluate(metric_key_prefix="after")
    metrics = {**before, **trained.metrics, **after, "log_history": trainer.state.log_history}
    validate_finite_values(metrics)
    json.dumps(metrics, allow_nan=False)
    trainer.model.config.use_cache = True
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    metadata = {"task": "post_training", "method": "dpo_lora", "kind": "preference",
                "base_model": model_path, "revision": revision, "requested_revision": args.revision,
                "reference": "full SFT checkpoint with fresh adapter disabled", "beta": args.beta,
                "prompt_format": "instruction_response_v1", "seed": args.seed,
                "max_steps": args.max_steps, "learning_rate": args.learning_rate, "versions": versions,
                "human_preference_labels": False, "data": dataset_manifest(args.data_dir),
                "split_counts": {key: len(value) for key, value in splits.items()},
                "runtime_verified": "This file is written only after successful CUDA training"}
    write_report(args.output_dir / "training_metadata.json", metadata)
    write_report(args.output_dir / "metrics.json", metrics)
    print(json.dumps({"output_dir": project_path(args.output_dir), "metrics": after}, indent=2))


def completion_log_probability(model, tokenizer, prompt, completion, max_length):
    import torch
    encoded = encode_record({"prompt": prompt, "completion": completion}, tokenizer, max_length, "sft")
    batch = {key: torch.tensor([value], device="cuda") for key, value in encoded.items()}
    labels = batch["labels"][:, 1:]
    with torch.no_grad():
        logps = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float().log_softmax(-1)
        selected = logps.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        value = (selected * (labels != -100)).sum().item()
    if not math.isfinite(value):
        raise RuntimeError("Non-finite preference log probability")
    return value


def evaluate(args, splits):
    require_cuda()
    from finetune_lab.evaluate import load_text_model
    from transformers import set_seed
    import torch
    set_seed(args.seed)
    model, tokenizer, metadata = load_text_model(args.model, args.revision, torch.device("cuda"))
    examples = []
    has_adapter = callable(getattr(model, "disable_adapter", None))
    for row in splits[args.split]:
        context_limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
        validate_preference_lengths(row, tokenizer, args, context_limit)
        chosen = completion_log_probability(model, tokenizer, row["prompt"], row["chosen"], args.max_length)
        rejected = completion_log_probability(model, tokenizer, row["prompt"], row["rejected"], args.max_length)
        reference_gap = None
        with model.disable_adapter() if has_adapter else nullcontext():
            if has_adapter:
                reference_gap = completion_log_probability(model, tokenizer, row["prompt"], row["chosen"], args.max_length) - completion_log_probability(
                    model, tokenizer, row["prompt"], row["rejected"], args.max_length)
        inputs = tokenizer(format_prompt(row["prompt"]), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_completion_length,
                                       do_sample=False, pad_token_id=tokenizer.pad_token_id)
        prediction = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        examples.append({**row, "chosen_log_probability": chosen, "rejected_log_probability": rejected,
                         "policy_gap": chosen - rejected, "reference_gap": reference_gap,
                         "reference_adjusted_gap": chosen - rejected - reference_gap if reference_gap is not None else None,
                         "generated": prediction})
    result = {"model": project_path(args.model), "split": args.split, "count": len(examples),
              "preference_accuracy": sum(row["policy_gap"] > 0 for row in examples) / len(examples),
              "mean_policy_gap": sum(row["policy_gap"] for row in examples) / len(examples),
              "human_alignment_claim": False, "training_metadata": metadata, "examples": examples}
    if has_adapter:
        result = {**result, "reference_adjusted_accuracy": sum(row["reference_adjusted_gap"] > 0 for row in examples) / len(examples)}
    write_report(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("examples", "training_metadata")}, indent=2))


def main(mode="train", argv=None):
    args = parse_args(mode, argv)
    splits = read_preferences(args.data_dir)
    if args.dry_run:
        print(json.dumps({"mode": mode, "synthetic_preferences": True,
                          "counts": {key: len(value) for key, value in splits.items()}}, indent=2))
        return
    (train if mode == "train" else evaluate)(args, splits)


if __name__ == "__main__":
    main()
