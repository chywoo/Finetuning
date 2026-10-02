"""Actual reward-model learning and PPO, pinned to TRL 0.24.0 on CUDA."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

from finetune_lab.text_encoding import encode_record, format_prompt
from finetune_lab.post_data import project_path, require_cuda, validate_finite_values

ROOT = Path(__file__).resolve().parents[1]
TRL_VERSION = "0.24.0"
PAD_TOKEN = "<|rlhf_pad|>"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Specify an integer greater than zero")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Specify a finite number greater than zero")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA: trained reward model → PPO policy")
    parser.add_argument("--stage", choices=("reward", "ppo", "evaluate-reward", "evaluate-policy"), required=True)
    parser.add_argument("--model", default=str(ROOT / "outputs/preference_sft_full"), help="Full SFT policy checkpoint; Instruct model only for stage-isolation experiments")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--reward-model", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/demo/post_preferences")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--max-steps", type=_positive_int, default=20, help="RewardTrainer optimizer updates")
    parser.add_argument("--total-episodes", type=_positive_int, default=32, help="PPO generated responses, not optimizer steps")
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--gradient-accumulation", type=_positive_int, default=4)
    parser.add_argument("--ppo-epochs", type=_positive_int, default=2)
    parser.add_argument("--max-length", type=_positive_int, default=256)
    parser.add_argument("--max-prompt-length", type=_positive_int, default=128)
    parser.add_argument("--response-length", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=_positive_float)
    parser.add_argument("--kl-coef", type=_positive_float, default=0.05)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-eval-samples", type=_positive_int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Schema/checkpoint checks only, no ML imports")
    args = parser.parse_args(argv)
    if args.stage != "reward" and args.reward_model is None:
        parser.error("--reward-model must point to the saved, trained RewardTrainer checkpoint")
    if args.max_length < 3 or args.max_prompt_length + args.response_length > args.max_length:
        parser.error("max-length must cover max-prompt-length + response-length")
    return argparse.Namespace(**{**vars(args),
        "output_dir": args.output_dir or ROOT / "outputs" / f"rlhf_{args.stage}",
        "learning_rate": args.learning_rate or (1e-5 if args.stage == "reward" else 1e-6)})


def preference_token_ids(tokenizer: Any, row: dict[str, str], max_length: int) -> dict[str, list[int]]:
    prefix = tokenizer(format_prompt(row["prompt"]), add_special_tokens=False)["input_ids"]
    encoded = {f"{name}_input_ids": list(prefix) + list(tokenizer(row[name], add_special_tokens=False)["input_ids"])
               + [tokenizer.eos_token_id] for name in ("chosen", "rejected")}
    if not prefix or any(len(ids) > max_length for ids in encoded.values()):
        raise ValueError(f"Preference {row['id']} exceeds --max-length; increase it or shorten data explicitly")
    return encoded


def require_reward_checkpoint(path: Path) -> dict[str, Any]:
    metadata_file = path / "training_metadata.json"
    if not metadata_file.is_file():
        raise ValueError("A checkpoint trained by RewardTrainer is required; random scalar heads are not RLHF rewards")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError(f"Invalid RewardTrainer metadata: {metadata_file}") from error
    if not isinstance(metadata, dict) or metadata.get("algorithm") != "trl_reward_trainer" or metadata.get("stage") != "reward":
        raise ValueError("--reward-model must be a RewardTrainer output from this lesson")
    if not isinstance(metadata.get("trained_steps"), int) or metadata["trained_steps"] < 1:
        raise ValueError("RewardTrainer checkpoint has no completed training steps")
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        raise ValueError("RewardTrainer config and safetensors weights must both exist")
    return metadata


def validate_ppo_batch(train_count: int, batch_size: int, accumulation: int, total_episodes: int) -> None:
    rollout_batch = batch_size * accumulation
    if train_count < rollout_batch:
        raise ValueError(f"PPO drop_last needs at least {rollout_batch} train prompts for one rollout batch")
    if total_episodes % rollout_batch:
        raise ValueError(f"total-episodes must be a multiple of rollout batch {rollout_batch}")


def summarize_preferences(gaps: list[float]) -> dict[str, float | int]:
    if not gaps or any(not math.isfinite(gap) for gap in gaps):
        raise ValueError("Preference gaps must be nonempty and finite")
    # softplus(-gap) avoids exp(large-positive) overflow.
    losses = [max(-gap, 0.0) + math.log1p(math.exp(-abs(gap))) for gap in gaps]
    return {"count": len(gaps), "preference_accuracy": sum(gap > 0 for gap in gaps) / len(gaps),
            "mean_chosen_minus_rejected": sum(gaps) / len(gaps), "bradley_terry_loss": sum(losses) / len(losses)}


def validate_finite_logs(logs: dict[str, Any]) -> None:
    validate_finite_values(logs)
    for name, value in logs.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise RuntimeError(f"Nonfinite training metric {name}={value}; checkpoint was not saved")


def _fresh_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Existing output is protected: {path}. Choose a new --output-dir")


def _spark_runtime() -> Any:
    if importlib.metadata.version("trl") != TRL_VERSION:
        raise RuntimeError(f"This lesson requires trl=={TRL_VERSION}; use the post-training profile")
    return require_cuda()


def _model_path(model: str) -> str:
    path = Path(model)
    if (path.is_absolute() or model.startswith(("./", "../", "outputs/"))) and not path.is_dir():
        raise ValueError(f"Missing full SFT checkpoint: {model}. Complete the preference SFT prerequisite first")
    if (Path(model) / "adapter_config.json").is_file():
        raise ValueError("Start from a full SFT checkpoint: choose --method full in SFT or merge the adapter first")
    return str(Path(model).resolve()) if Path(model).is_dir() else model


def _tokenizer(source: str, revision: str, add_pad: bool = False) -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if add_pad and (tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id):
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("PPO requires a distinct PAD token; use the saved reward tokenizer")
    tokenizer.padding_side = "left"
    return tokenizer


def _fit_vocabulary(model: Any, tokenizer: Any) -> None:
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False


def _check_context(model: Any, max_length: int) -> None:
    limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
    if isinstance(limit, int) and max_length > limit:
        raise ValueError(f"--max-length {max_length} exceeds this checkpoint's context length {limit}")


def _check_policy_tokenizer(source: str, revision: str, tokenizer: Any) -> None:
    from transformers import AutoTokenizer
    original = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)
    vocabulary = tokenizer.get_vocab()
    if any(vocabulary.get(token) != index for token, index in original.get_vocab().items()):
        raise ValueError("Policy and reward tokenizers must use identical original token IDs")


def _metadata(args: argparse.Namespace, torch: Any, **extra: Any) -> dict[str, Any]:
    from finetune_lab.post_data import dataset_manifest
    base_model = (_model_path(args.model) if args.stage != "evaluate-reward"
                  else require_reward_checkpoint(args.reward_model).get("base_model"))
    return {"stage": args.stage, "base_model": base_model, "requested_revision": args.revision,
            "prompt_format": "instruction_response_v1", "seed": args.seed,
            "data_dir": project_path(args.data_dir), "dataset": dataset_manifest(args.data_dir),
            "reward_model": str(args.reward_model.resolve()) if args.reward_model is not None else None,
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "packages": {name: importlib.metadata.version(name) for name in ("trl", "transformers", "datasets", "accelerate")}},
            **extra}


def _save_report(path: Path, report: dict[str, Any], filename: str) -> None:
    from finetune_lab.post_data import write_report
    path.mkdir(parents=True, exist_ok=True)
    write_report(path / filename, report)


def _finite_callback() -> Any:
    from transformers import TrainerCallback
    class FiniteLogs(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            validate_finite_logs(logs or {})
    return FiniteLogs()


def _train_reward(args: argparse.Namespace, rows: dict[str, list[dict]], torch: Any) -> None:
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, set_seed
    from trl import RewardConfig, RewardTrainer

    set_seed(args.seed)
    source = _model_path(args.model)
    tokenizer = _tokenizer(source, args.revision, add_pad=True)
    model = AutoModelForSequenceClassification.from_pretrained(source, revision=args.revision,
                num_labels=1, torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True)
    _fit_vocabulary(model, tokenizer)
    _check_context(model, args.max_length)
    if not hasattr(model, "score"):
        raise ValueError("This PPO lesson requires a sequence classifier exposing model.score (e.g. SmolLM2/Llama)")
    datasets = {name: Dataset.from_list([preference_token_ids(tokenizer, row, args.max_length) for row in records])
                for name, records in rows.items() if name != "test"}

    class FiniteRewardTrainer(RewardTrainer):
        def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False, **kwargs: Any) -> Any:
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss).all().item()):
                raise RuntimeError("Nonfinite reward-model loss")
            return result

    config = RewardConfig(output_dir=str(args.output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
        max_length=args.max_length, center_rewards_coefficient=0.01, bf16=True, fp16=False,
        optim="adamw_torch", gradient_checkpointing=False, eval_strategy="no", save_strategy="no",
        logging_steps=1, logging_nan_inf_filter=False, report_to="none", push_to_hub=False, seed=args.seed)
    trainer = FiniteRewardTrainer(model=model, args=config, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"], callbacks=[_finite_callback()])
    before = trainer.evaluate(metric_key_prefix="before")
    trained = trainer.train()
    after = trainer.evaluate(metric_key_prefix="after")
    metrics = {**before, **trained.metrics, **after}
    validate_finite_logs(metrics)
    metadata = _metadata(args, torch, algorithm="trl_reward_trainer", trained_steps=trainer.state.global_step,
        revision=getattr(model.config, "_commit_hash", None) or args.revision, metrics=metrics,
        tokenizer_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        center_rewards_coefficient=0.01, full_finetuning=True)
    json.dumps(metadata, allow_nan=False)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    _save_report(args.output_dir, metadata, "training_metadata.json")


def _ppo_models(args: argparse.Namespace, torch: Any) -> tuple[Any, Any, Any, Any, Any]:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    source = _model_path(args.model)
    tokenizer = _tokenizer(str(args.reward_model), "main")
    _check_policy_tokenizer(source, args.revision, tokenizer)
    kwargs = {"torch_dtype": torch.float32, "trust_remote_code": False, "use_safetensors": True}
    policy = AutoModelForCausalLM.from_pretrained(source, revision=args.revision, **kwargs)
    revision = getattr(policy.config, "_commit_hash", None) or args.revision
    reference = AutoModelForCausalLM.from_pretrained(source, revision=revision, **kwargs)
    reward = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, **kwargs)
    value = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, **kwargs)
    for model in (policy, reference, reward, value):
        _fit_vocabulary(model, tokenizer)
        _check_context(model, args.max_length)
    # Vocabulary expansion can initialize the new PAD row differently; the KL reference starts exactly at policy.
    reference.load_state_dict(policy.state_dict())
    architecture = (policy.config.model_type, policy.config.hidden_size, policy.config.num_hidden_layers)
    if any((model.config.model_type, model.config.hidden_size, model.config.num_hidden_layers) != architecture
           for model in (reference, reward, value)):
        raise ValueError("Policy, reference, reward, and value backbones must have matching architectures")
    for model in (reference, reward):
        model.requires_grad_(False)
        model.eval()
    return policy, reference, reward, value, tokenizer


def _train_ppo(args: argparse.Namespace, rows: dict[str, list[dict]], torch: Any) -> None:
    from datasets import Dataset
    from transformers import set_seed
    from trl import PPOConfig, PPOTrainer
    set_seed(args.seed)
    policy, reference, reward, value, tokenizer = _ppo_models(args, torch)
    datasets = {}
    for split in ("train", "validation"):
        tokenized = [tokenizer(format_prompt(row["prompt"]), add_special_tokens=False)["input_ids"] for row in rows[split]]
        if any(not ids or len(ids) > args.max_prompt_length for ids in tokenized):
            raise ValueError("PPO prompt exceeds --max-prompt-length; shorten it or increase the budget explicitly")
        datasets[split] = Dataset.from_list([{"input_ids": ids} for ids in tokenized])
    config = PPOConfig(output_dir=str(args.output_dir), sft_model_path=_model_path(args.model),
        reward_model_path=str(args.reward_model), total_episodes=args.total_episodes,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation, num_mini_batches=1,
        num_ppo_epochs=args.ppo_epochs, local_rollout_forward_batch_size=1,
        response_length=args.response_length, stop_token="eos", kl_coef=args.kl_coef,
        learning_rate=args.learning_rate, whiten_rewards=False, bf16=True, fp16=False,
        num_sample_generations=0, gradient_checkpointing=False, optim="adamw_torch",
        logging_steps=1, save_strategy="no", report_to="none", push_to_hub=False, seed=args.seed)
    trainer = PPOTrainer(args=config, processing_class=tokenizer, model=policy, ref_model=reference,
        reward_model=reward, value_model=value, train_dataset=datasets["train"],
        eval_dataset=datasets["validation"], callbacks=[_finite_callback()])
    trainer.train()
    for metrics in trainer.state.log_history:
        validate_finite_logs(metrics)
    for model in (policy, value):
        if any(not bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
            raise RuntimeError("Nonfinite policy/value parameter; checkpoint was not saved")
    metadata = _metadata(args, torch, algorithm="trl_ppo", kind="sft", method="full",
        reward_model=str(args.reward_model.resolve()), total_episodes=args.total_episodes,
        optimizer_updates=trainer.state.global_step, ppo_epochs=args.ppo_epochs, kl_coef=args.kl_coef,
        response_length=args.response_length, metrics=trainer.state.log_history,
        revision=getattr(policy.config, "_commit_hash", None) or args.revision,
        tokenizer_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    json.dumps(metadata, allow_nan=False)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.unwrap_model(trainer.model).value_model.save_pretrained(args.output_dir / "value_model", safe_serialization=True)
    _save_report(args.output_dir, metadata, "training_metadata.json")


def _score(model: Any, ids: list[int], torch: Any) -> float:
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        score = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits.float().item()
    if not math.isfinite(score):
        raise RuntimeError("Reward evaluation produced a nonfinite score")
    return score


def _response_log_probability(policy: Any, tokenizer: Any, row: dict, completion: str, args: Any, torch: Any) -> float:
    encoded = encode_record({"prompt": row["prompt"], "completion": completion}, tokenizer, args.max_length, "sft")
    ids = torch.tensor([encoded["input_ids"]], device="cuda")
    labels = torch.tensor(encoded["labels"][1:], device="cuda")
    mask = labels != -100
    with torch.no_grad():
        log_probs = policy(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, :-1].float().log_softmax(-1)
        result = log_probs[mask, labels[mask]].sum().item()
    if not math.isfinite(result):
        raise RuntimeError("Policy evaluation produced nonfinite log probability")
    return result


def _evaluate(args: argparse.Namespace, rows: dict[str, list[dict]], torch: Any) -> None:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    tokenizer = _tokenizer(str(args.reward_model), "main")
    kwargs = {"torch_dtype": torch.float32, "trust_remote_code": False, "use_safetensors": True}
    reward = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, **kwargs).to("cuda").eval()
    _check_context(reward, args.max_length)
    selected = rows[args.split][:args.max_eval_samples]
    pairs, generations = [], []
    if args.stage == "evaluate-reward":
        for row in selected:
            ids = preference_token_ids(tokenizer, row, args.max_length)
            chosen, rejected = (_score(reward, ids[f"{name}_input_ids"], torch) for name in ("chosen", "rejected"))
            pairs.append({"id": row["id"], "gap": chosen - rejected, "chosen_reward": chosen, "rejected_reward": rejected})
    else:
        _check_policy_tokenizer(_model_path(args.model), args.revision, tokenizer)
        policy = AutoModelForCausalLM.from_pretrained(_model_path(args.model), revision=args.revision, **kwargs).to("cuda").eval()
        _fit_vocabulary(policy, tokenizer)
        _check_context(policy, args.max_length)
        for row in selected:
            # Compare the full pair; never let the shared SFT encoder silently truncate evaluation targets.
            preference_token_ids(tokenizer, row, args.max_length)
            chosen = _response_log_probability(policy, tokenizer, row, row["chosen"], args, torch)
            rejected = _response_log_probability(policy, tokenizer, row, row["rejected"], args, torch)
            pairs.append({"id": row["id"], "gap": chosen - rejected})
            prefix = tokenizer(format_prompt(row["prompt"]), add_special_tokens=False, return_tensors="pt").to("cuda")
            if prefix["input_ids"].shape[1] > args.max_prompt_length:
                raise ValueError("Evaluation prompt exceeds --max-prompt-length")
            with torch.no_grad():
                output = policy.generate(**prefix, max_new_tokens=args.response_length, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    suppress_tokens=[tokenizer.pad_token_id])
            completion = tokenizer.decode(output[0, prefix["input_ids"].shape[1]:], skip_special_tokens=True)
            reward_ids = output[0].tolist()
            generations.append({"id": row["id"], "prompt": row["prompt"], "completion": completion,
                                "training_reward_proxy": _score(reward, reward_ids, torch)})
    report = {**_metadata(args, torch), "split": args.split,
              "metrics": summarize_preferences([pair["gap"] for pair in pairs]),
              "preference_pairs": pairs, "generations": generations,
              "reward_score_is_training_proxy": True, "generation_settings": {"do_sample": False, "response_length": args.response_length}}
    _save_report(args.output_dir, report, "evaluation.json")
    print(json.dumps(report["metrics"], ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from finetune_lab.post_data import read_preferences
    rows = read_preferences(args.data_dir)
    if args.stage == "ppo":
        validate_ppo_batch(len(rows["train"]), args.batch_size, args.gradient_accumulation, args.total_episodes)
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "status": "schema_validated",
                          "model_prerequisites_checked": False,
                          "samples": {name: len(records) for name, records in rows.items()}}, ensure_ascii=False))
        return
    if args.stage != "evaluate-reward":
        _model_path(args.model)
    if args.reward_model is not None:
        require_reward_checkpoint(args.reward_model)
    _fresh_output(args.output_dir)
    torch = _spark_runtime()
    if args.stage == "reward":
        _train_reward(args, rows, torch)
    elif args.stage == "ppo":
        _train_ppo(args, rows, torch)
    else:
        _evaluate(args, rows, torch)
    print(f"Saved: {project_path(args.output_dir)}")


if __name__ == "__main__":
    main()
