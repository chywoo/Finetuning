"""Reload a full model or adapter and evaluate held-out text records."""
import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path

from finetune_lab.data import read_jsonl
from finetune_lab.paths import project_path

ROOT = Path(__file__).resolve().parents[1]


def normalize_answer(text: str) -> str:
    text = text.casefold().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_answers(predictions: list[str], references: list[str]) -> dict:
    if not predictions or len(predictions) != len(references):
        raise ValueError("Need equally sized, non-empty predictions and references")
    return {"count": len(predictions),
            "exact_match": sum(normalize_answer(p) == normalize_answer(r)
                               for p, r in zip(predictions, references)) / len(predictions),
            "token_f1": sum(token_f1(p, r) for p, r in zip(predictions, references)) / len(predictions)}


def resolve_device(requested: str):
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device is unavailable")
    return torch.device(requested)


def load_text_model(model_name: str, revision: str, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(model_name)
    metadata_path = path / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if (path / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel
        config = PeftConfig.from_pretrained(path)
        base_id = metadata.get("base_model") or config.base_model_name_or_path
        base_revision = metadata.get("revision") or config.revision or revision
        base = AutoModelForCausalLM.from_pretrained(base_id, revision=base_revision,
                                                  trust_remote_code=False, use_safetensors=True)
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                   trust_remote_code=False, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    return model, tokenizer, metadata


def generation_prompt_ids(prompt_ids: list[int], max_length: int, max_new_tokens: int,
                          config) -> list[int]:
    """Reserve actual model context for generation, keeping the response delimiter."""
    context_limit = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
    if max_new_tokens < 1 or (context_limit and max_new_tokens >= context_limit):
        raise ValueError("Generation token budget must leave at least one context token")
    width = min(max_length, context_limit - max_new_tokens) if context_limit else max_length
    if not prompt_ids or width < 1:
        raise ValueError("Generation needs non-empty prompt context")
    return prompt_ids[-width:]


def evaluate_records(model, tokenizer, rows: list[dict], kind: str, max_length: int,
                     max_new_tokens: int, device) -> dict:
    if not rows:
        raise ValueError("Evaluation requires non-empty rows")
    import torch
    from finetune_lab.text_encoding import encode_record, format_prompt
    weighted_loss, target_count, generations = 0.0, 0, []
    with torch.no_grad():
        for row in rows:
            encoded = encode_record(row, tokenizer, max_length, kind)
            batch = {key: torch.tensor([value], dtype=torch.long, device=device)
                     for key, value in encoded.items()}
            tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            if not tokens:
                raise ValueError(f"No prediction targets for {row['id']}")
            loss = float(model(**batch).loss.item())
            if not math.isfinite(loss):
                raise ValueError(f"Non-finite loss for {row['id']}")
            weighted_loss += loss * tokens
            target_count += tokens
            if kind == "sft":
                prompt_ids = tokenizer.encode(format_prompt(row["prompt"]), add_special_tokens=False)
                # Preserve the response delimiter at the end, as in training.
                prompt_ids = generation_prompt_ids(prompt_ids, max_length, max_new_tokens, model.config)
                inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True).strip()
                generations.append({"id": row["id"], "prompt": row["prompt"],
                                    "prediction": text, "reference": row["completion"]})
    loss = weighted_loss / target_count
    result = {"count": len(rows), "target_tokens": target_count, "loss": loss,
              "perplexity": math.exp(loss) if loss < 700 else None}
    if kind == "sft":
        result = {**result, **score_answers([r["prediction"] for r in generations],
                                          [r["reference"] for r in generations]),
                  "generations": generations}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["sft", "cpt"], default="sft")
    parser.add_argument("--split", choices=["validation", "test", "probe_seen_facts"], default="test")
    parser.add_argument("--max-eval-samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation.json")
    args = parser.parse_args()
    if args.max_eval_samples < 1 or args.max_length < 3 or args.max_new_tokens < 1:
        parser.error("sample/token limits must be positive (max_length >= 3)")
    if args.output.exists():
        parser.error("--output already exists; choose a new report path")
    rows = read_jsonl(args.data_dir / f"{args.split}.jsonl", args.kind)[:args.max_eval_samples]
    device = resolve_device(args.device)
    model, tokenizer, metadata = load_text_model(args.model, args.revision, device)
    if metadata.get("kind") and metadata["kind"] != args.kind:
        print(f"Note: model trained for {metadata['kind']}; evaluating {args.kind}")
    result = {"model": project_path(args.model), "requested_revision": args.revision,
              "resolved_model_revision": getattr(model.config, "_commit_hash", None),
              "training_metadata": metadata, "data_dir": project_path(args.data_dir),
              "split": args.split, "kind": args.kind, "max_length": args.max_length,
              "max_new_tokens": args.max_new_tokens,
              **evaluate_records(model, tokenizer, rows, args.kind, args.max_length,
                                 args.max_new_tokens, device)}
    if args.split == "probe_seen_facts":
        result = {**result, "interpretation": "Recall of training facts under rephrased questions; NOT unseen-fact generalization"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("generations", "training_metadata")}, indent=2))
    print(f"Full report: {project_path(args.output)}")


if __name__ == "__main__":
    main()
