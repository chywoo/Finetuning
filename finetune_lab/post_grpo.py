"""Verifiable arithmetic rewards + GRPO LoRA; pinned TRL, single Spark CUDA GPU."""
import argparse
import json
import math
from pathlib import Path

from finetune_lab.post_data import (
    ROOT, arithmetic_reward, check_trl_version, dataset_manifest, group_statistics,
    parse_integer_completion, read_math, require_cuda, require_fresh_output,
    require_full_checkpoint, validate_finite_values, validate_grpo_settings, write_report,
)
from finetune_lab.text_encoding import format_prompt


def parse_args(mode="train", argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/demo/post_math")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    if mode == "train":
        parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/post_grpo")
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--gradient-accumulation", type=int, default=2)
        parser.add_argument("--max-steps", type=int, default=20)
        parser.add_argument("--learning-rate", type=float, default=1e-5)
        parser.add_argument("--beta", type=float, default=0.04)
    else:
        parser.add_argument("--split", choices=("validation", "test"), default="test")
        parser.add_argument("--output", type=Path, default=ROOT / "outputs/post_grpo-evaluation.json")
    args = parser.parse_args(argv)
    if min(args.max_prompt_length, args.max_completion_length) < 3 or args.num_generations < 2:
        parser.error("token limits >= 3 and num-generations >= 2 required")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be positive and finite")
    if mode == "train":
        try:
            validate_grpo_settings(args.batch_size, args.num_generations, args.gradient_accumulation, args.beta)
        except ValueError as error:
            parser.error(str(error))
        if args.max_steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("max-steps and learning-rate must be positive")
    return args


def evaluate_rows(model, tokenizer, rows, args):
    """Greedy independent accuracy plus sampled-group diagnostics, without training."""
    import torch
    model.eval()
    examples, all_rewards = [], []
    for row in rows:
        inputs = tokenizer(format_prompt(row["prompt"]), add_special_tokens=False, return_tensors="pt").to("cuda")
        if inputs.input_ids.shape[1] > args.max_prompt_length:
            raise ValueError(f"Prompt exceeds configured length: {row['id']}")
        context_limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
        if context_limit and inputs.input_ids.shape[1] + args.max_completion_length > context_limit:
            raise ValueError("Prompt + generation exceeds model context length")
        with torch.no_grad():
            greedy = model.generate(**inputs, max_new_tokens=args.max_completion_length, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
            samples = model.generate(**inputs, max_new_tokens=args.max_completion_length, do_sample=True,
                                     temperature=args.temperature, top_p=0.95, num_return_sequences=args.num_generations,
                                     pad_token_id=tokenizer.pad_token_id)
        width = inputs.input_ids.shape[1]
        prediction = tokenizer.decode(greedy[0, width:], skip_special_tokens=True)
        sampled = tokenizer.batch_decode(samples[:, width:], skip_special_tokens=True)
        rewards = arithmetic_reward(sampled, [row["answer"]] * args.num_generations)
        all_rewards.extend(rewards)
        examples.append({"id": row["id"], "prompt": row["prompt"], "answer": row["answer"],
                         "prediction": prediction, "correct": parse_integer_completion(prediction) == int(row["answer"]),
                         "sampled_completions": sampled, "sampled_rewards": rewards})
    return {"count": len(examples), "exact_accuracy": sum(row["correct"] for row in examples) / len(examples),
            "invalid_format_rate": sum(parse_integer_completion(row["prediction"]) is None for row in examples) / len(examples),
            "sample_groups": group_statistics(all_rewards, args.num_generations), "examples": examples}


def train(args, splits):
    model_path = require_full_checkpoint(args.model)
    require_fresh_output(args.output_dir)
    versions = check_trl_version()
    torch = require_cuda()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                trust_remote_code=False, use_safetensors=True)
    model.config.use_cache = True
    model.to("cuda")
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    # Probe TRAIN prompts, never use test labels to select training settings.
    pilot = evaluate_rows(model, tokenizer, splits["train"][:8], args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(args.output_dir / "training_signal_probe.json", pilot)
    if pilot["sample_groups"]["frac_reward_zero_std"] == 1.0:
        raise RuntimeError("Every sampled TRAIN group has identical rewards: no relative correctness signal. "
                           "Inspect training_signal_probe.json; improve math SFT, adjust sampling, or change difficulty. "
                           "Use a fresh output directory for another attempt.")
    dataset_rows = [{"prompt": format_prompt(row["prompt"]), "answer": row["answer"]} for row in splits["train"]]
    for row in dataset_rows:
        if len(tokenizer.encode(row["prompt"], add_special_tokens=False)) > args.max_prompt_length:
            raise ValueError("Training prompt exceeds max-prompt-length; avoid hidden truncation")
    config = GRPOConfig(
        output_dir=str(args.output_dir), max_steps=args.max_steps, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length,
        beta=args.beta, scale_rewards="group", loss_type="grpo", num_iterations=1,
        use_vllm=False, temperature=args.temperature, top_p=0.95, bf16=True, fp16=False,
        gradient_checkpointing=False, optim="adamw_torch", eval_strategy="no", save_strategy="no",
        logging_steps=1, report_to="none", seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
    )

    class FiniteGRPOTrainer(GRPOTrainer):
        def compute_loss(self, *arguments, **kwargs):
            value = super().compute_loss(*arguments, **kwargs)
            loss = value[0] if isinstance(value, tuple) else value
            if not torch.isfinite(loss).all():
                raise RuntimeError("Non-finite GRPO loss")
            return value

    trainer = FiniteGRPOTrainer(
        model=model, reward_funcs=arithmetic_reward, args=config,
        train_dataset=Dataset.from_list(dataset_rows), processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                               lora_dropout=0.0, target_modules="all-linear", revision=revision),
    )
    if trainer.accelerator.device.type != "cuda":
        raise RuntimeError("GRPO Trainer must use CUDA")
    result = trainer.train()
    validation = evaluate_rows(trainer.model, tokenizer, splits["validation"], args)
    metrics = {**result.metrics, "log_history": trainer.state.log_history}
    validate_finite_values({"metrics": metrics, "validation": validation})
    json.dumps({"metrics": metrics, "validation": validation}, allow_nan=False)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    metadata = {"task": "post_training", "method": "rlvr_grpo_lora", "kind": "math",
                "base_model": model_path, "revision": revision, "requested_revision": args.revision,
                "prompt_format": "instruction_response_v1", "reward": "strict_single_canonical_integer_exact_match",
                "reference": "full math SFT checkpoint with fresh adapter disabled", "beta": args.beta,
                "num_generations": args.num_generations, "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation, "seed": args.seed,
                "max_steps": args.max_steps, "learning_rate": args.learning_rate,
                "use_vllm": False, "versions": versions, "data": dataset_manifest(args.data_dir),
                "split_counts": {key: len(value) for key, value in splits.items()}}
    write_report(args.output_dir / "training_metadata.json", metadata)
    write_report(args.output_dir / "metrics.json", metrics)
    write_report(args.output_dir / "validation.json", validation)
    print(json.dumps({"output_dir": str(args.output_dir), "validation_accuracy": validation["exact_accuracy"]}, indent=2))


def evaluate(args, splits):
    torch = require_cuda()
    from finetune_lab.evaluate import load_text_model
    from transformers import set_seed
    set_seed(args.seed)
    model, tokenizer, metadata = load_text_model(args.model, args.revision, torch.device("cuda"))
    result = {"model": args.model, "split": args.split, "training_metadata": metadata,
              "num_generations": args.num_generations, "temperature": args.temperature,
              **evaluate_rows(model, tokenizer, splits[args.split], args)}
    write_report(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("examples", "training_metadata")}, indent=2))


def main(mode="train", argv=None):
    args = parse_args(mode, argv)
    splits = read_math(args.data_dir)
    if args.dry_run:
        print(json.dumps({"mode": mode, "counts": {key: len(value) for key, value in splits.items()},
                          "reward": "strict integer exact match", "num_generations": args.num_generations}, indent=2))
        return
    (train if mode == "train" else evaluate)(args, splits)


if __name__ == "__main__":
    main()
