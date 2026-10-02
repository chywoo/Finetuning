"""Dependency-free JSONL contracts shared by all nine lessons."""
import json
import random
from pathlib import Path

LABELS = ("angular_leaf_spot", "bean_rust", "healthy")
FIELDS = {"sft": ("id", "prompt", "completion"), "cpt": ("id", "text"),
          "vision": ("id", "image", "label")}


def validate_record(row: dict, kind: str) -> None:
    if kind not in FIELDS:
        raise ValueError(f"Unknown kind: {kind}")
    if not isinstance(row, dict):
        raise ValueError("Each JSONL row must be an object")
    for name in FIELDS[kind]:
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    if kind == "vision" and row["label"] not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")


def read_jsonl(path: Path | str, kind: str) -> list[dict]:
    if kind not in FIELDS:
        raise ValueError(f"Unknown kind: {kind}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run python -m finetune_lab.prepare_data first")
    records = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            validate_record(row, kind)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}, line {number}: {exc}") from exc
        if row["id"] in seen:
            raise ValueError(f"{path}, line {number}: duplicate id {row['id']}")
        seen.add(row["id"])
        records.append(row)
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records


def write_jsonl(path: Path | str, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    path.write_text(text, encoding="utf-8")


def signature(row: dict, kind: str) -> str:
    key = {"sft": "prompt", "cpt": "text", "vision": "image"}[kind]
    return " ".join(row[key].casefold().split())


def validate_splits(splits: dict[str, list[dict]], kind: str) -> None:
    seen = {}
    ids = set()
    for split, records in splits.items():
        for row in records:
            validate_record(row, kind)
            key = signature(row, kind)
            if key in seen:
                raise ValueError(f"content overlap: {seen[key]} and {split}")
            if row["id"] in ids:
                raise ValueError(f"id overlap: {row['id']}")
            seen[key] = split
            ids.add(row["id"])


def split_records(records: list[dict], train_samples: int, eval_samples: int,
                  seed: int = 42) -> dict[str, list[dict]]:
    if min(train_samples, eval_samples) < 1:
        raise ValueError("sample counts must be positive")
    if len(records) < train_samples + 2 * eval_samples:
        raise ValueError("Not enough unique records for requested train/validation/test counts")
    shuffled = random.Random(seed).sample(records, len(records))
    return {"train": shuffled[:train_samples],
            "validation": shuffled[train_samples:train_samples + eval_samples],
            "test": shuffled[train_samples + eval_samples:train_samples + 2 * eval_samples]}
