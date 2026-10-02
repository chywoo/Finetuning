"""Shared causal-LM encoding; importable without Torch/Transformers installed."""

from typing import Any


IGNORE_INDEX = -100


def format_prompt(prompt: str) -> str:
    """The deliberately explicit SFT template also works for base models."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("text must produce at least one token")
    return list(ids)


def encode_record(record: dict, tokenizer: Any, max_length: int, kind: str) -> dict:
    """Keep response targets and EOS; let the model perform the causal shift.

    Prefix and completion are tokenized separately, defining a reproducible
    boundary even when BPE would merge tokens across their concatenation.
    Prefix truncation is from the left; response truncation is from the right.
    Padding is handled later by the collator, never by matching token IDs.
    """
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if kind == "sft":
        completion = record.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError("completion must be a non-empty string")
        prefix_ids = _token_ids(tokenizer, format_prompt(record.get("prompt")))
        response_ids = _token_ids(tokenizer, completion)[: max_length - 2]
        remaining_prefix = max_length - len(response_ids) - 1
        prefix_ids = prefix_ids[-remaining_prefix:]
        input_ids = prefix_ids + response_ids + [eos_id]
        labels = [IGNORE_INDEX] * len(prefix_ids) + response_ids + [eos_id]
    elif kind == "cpt":
        body = record.get("text")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("text must be a non-empty string")
        input_ids = _token_ids(tokenizer, body)[: max_length - 1] + [eos_id]
        labels = list(input_ids)
    else:
        raise ValueError("kind must be 'sft' or 'cpt'")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def encode_sft(tokenizer: Any, row: dict, max_length: int) -> dict:
    return encode_record(row, tokenizer, max_length, "sft")


def encode_cpt(tokenizer: Any, row: dict, max_length: int) -> dict:
    return encode_record(row, tokenizer, max_length, "cpt")


class CausalLMCollator:
    """Right-pad by position, preserving a real EOS when PAD equals EOS."""

    def __init__(self, pad_token_id: int, return_tensors: str | None = "pt"):
        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        if return_tensors not in ("pt", None):
            raise ValueError("return_tensors must be 'pt' or None")
        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features: list[dict]) -> dict:
        if not features:
            raise ValueError("cannot collate an empty batch")
        fields = ("input_ids", "attention_mask", "labels")
        if any(not row.get("input_ids") for row in features):
            raise ValueError("cannot collate an empty sequence")
        if any(len(row[field]) != len(row["input_ids"]) for row in features for field in fields):
            raise ValueError("input_ids, attention_mask, and labels must have equal length")
        width = max(len(row["input_ids"]) for row in features)
        pad_values = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": IGNORE_INDEX}
        result = {
            field: [list(row[field]) + [pad_values[field]] * (width - len(row[field])) for row in features]
            for field in fields
        }
        if self.return_tensors is None:
            return result
        import torch

        return {field: torch.tensor(values, dtype=torch.long) for field, values in result.items()}


TextCollator = CausalLMCollator
