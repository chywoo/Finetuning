"""Authored post-training fixtures and strict verifiable rewards (stdlib only)."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def parse_integer_completion(text) -> int | None:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]{0,18})", candidate):
        return None
    number = int(candidate)
    return number if str(number) == candidate else None


def arithmetic_reward(completions, answer, **kwargs) -> list[float]:
    """Only one canonical integer earns reward; never execute generated text."""
    if len(completions) != len(answer):
        raise ValueError("completions and answer must have equal length")
    expected = [parse_integer_completion(value) for value in answer]
    if any(value is None for value in expected):
        raise ValueError("answer labels must be canonical integers")
    return [float(parse_integer_completion(text) == value)
            for text, value in zip(completions, expected)]


def group_statistics(rewards: list[float], num_generations: int) -> dict:
    if num_generations < 2 or not rewards or len(rewards) % num_generations:
        raise ValueError("Reward groups must have at least two complete generations")
    if any(not math.isfinite(value) for value in rewards):
        raise ValueError("Rewards must be finite")
    groups = [rewards[index:index + num_generations] for index in range(0, len(rewards), num_generations)]
    return {"groups": len(groups), "mean_reward": sum(rewards) / len(rewards),
            "frac_reward_zero_std": sum(len(set(group)) == 1 for group in groups) / len(groups)}


def validate_grpo_settings(batch_size: int, num_generations: int, accumulation: int, beta: float) -> None:
    if num_generations < 2 or batch_size < num_generations or batch_size % num_generations:
        raise ValueError("batch-size must be >= num-generations and divisible by it (generations >= 2)")
    if accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive: keep the KL reference penalty")


def _validate_row(row: dict, kind: str) -> None:
    fields = ("id", "prompt", "chosen", "rejected") if kind == "preference" else (
        "id", "prompt", "answer", "completion")
    if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in fields):
        raise ValueError(f"Required non-empty string fields: {fields}")
    if kind == "preference" and row["chosen"].strip() == row["rejected"].strip():
        raise ValueError("chosen and rejected must be different")
    if kind == "math":
        if parse_integer_completion(row["answer"]) is None:
            raise ValueError("answer must be one canonical integer")
        if row["completion"] != row["answer"]:
            raise ValueError("math SFT completion must equal the verified answer")


def _read_splits(data_dir: Path | str, kind: str) -> dict[str, list[dict]]:
    data_dir = Path(data_dir)
    result, seen_ids, seen_prompts = {}, set(), set()
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split: {path}")
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                _validate_row(row, kind)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            prompt_key = " ".join(row["prompt"].casefold().split())
            if row["id"] in seen_ids or prompt_key in seen_prompts:
                raise ValueError(f"Split overlap or duplicate at {path}:{number}")
            seen_ids.add(row["id"])
            seen_prompts.add(prompt_key)
            rows.append(dict(row))
        if not rows:
            raise ValueError(f"Empty split: {path}")
        result[split] = rows
    return result


def read_preferences(data_dir: Path | str) -> dict[str, list[dict]]:
    return _read_splits(data_dir, "preference")


def read_math(data_dir: Path | str) -> dict[str, list[dict]]:
    return _read_splits(data_dir, "math")


def require_fresh_output(path: Path | str) -> None:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Existing output cannot be overwritten: {path}; choose a new --output-dir")


def require_full_checkpoint(model: str) -> str:
    path = Path(model)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("--model must be a local full SFT checkpoint; complete SFT and merge its adapter first")
    if (path / "adapter_config.json").is_file():
        raise ValueError("A fresh post-training adapter needs a full SFT checkpoint, not an unmerged adapter")
    if not list(path.glob("*.safetensors")):
        raise ValueError("Full checkpoint must contain safetensors weights")
    return str(path.resolve())


def require_cuda():
    if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Run inside the DGX Spark ARM64 Linux container; dry-run uses no ML dependencies")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This DGX Spark lesson requires exactly one CUDA GPU; no CPU/MPS fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This lesson requires CUDA BF16 support")
    return torch


def check_trl_version() -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("trl", "transformers", "peft", "torch")}
    if versions["trl"] != "0.24.0":
        raise RuntimeError("Use TRL 0.24.0 from the Spark post-training requirements")
    return versions


def write_report(path: Path | str, result: dict) -> None:
    path = Path(path)
    if path.exists():
        raise ValueError(f"Existing report cannot be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_finite_values(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Non-finite training/evaluation metric; do not save a successful checkpoint")
    if isinstance(value, dict):
        for nested in value.values():
            validate_finite_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            validate_finite_values(nested)


def dataset_manifest(data_dir: Path | str) -> dict:
    directory = Path(data_dir)
    return {"data_dir": str(directory.resolve()), "sha256": {
        split: hashlib.sha256((directory / f"{split}.jsonl").read_bytes()).hexdigest() for split in SPLITS}}


def _authored_rows(kind: str) -> dict:
    # Fixed source-example split before conversion to preference / SFT / RLVR.
    count = 48 if kind == "math" else 24
    rows = []
    for index in range(count):
        left, right = index // 6 + 1, index % 6 + 1
        answer = str(left + right)
        prompt = f"Compute {left} + {right}. Reply with one integer only; no explanation."
        if kind == "math":
            rows.append({"id": f"math-{index}", "prompt": prompt, "answer": answer, "completion": answer})
        else:
            rows.append({"id": f"preference-{index}", "prompt": prompt,
                         "chosen": answer, "rejected": str(left + right + 1)})
    # Distribute operand combinations, instead of reserving all large left operands for test.
    return {split: [row for index, row in enumerate(rows) if index % 6 in slots]
            for split, slots in (("train", (0, 1, 2, 3)), ("validation", (4,)), ("test", (5,)))}


def prepare_task(kind: str, output_root: Path) -> None:
    name = "post_math" if kind == "math" else "post_preferences"
    directory = output_root / name
    require_fresh_output(directory)
    rows = _authored_rows(kind)
    directory.mkdir(parents=True, exist_ok=True)
    for split, records in rows.items():
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        (directory / f"{split}.jsonl").write_text(text, encoding="utf-8")
    loader = read_math if kind == "math" else read_preferences
    loader(directory)
    manifest = {**dataset_manifest(directory), "source": "authored synthetic fixture", "license": "CC0-1.0",
                "human_preference_labels": False, "counts": {name: len(records) for name, records in rows.items()},
                "warning": "Tiny arithmetic learning fixture; no claim of human alignment or broad reasoning gains"}
    sft_dir = directory / "sft"
    sft_dir.mkdir()
    for split, records in rows.items():
        sft = [{"id": row["id"], "prompt": row["prompt"],
                "completion": row["completion"] if kind == "math" else row["chosen"]} for row in records]
        (sft_dir / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sft), encoding="utf-8")
    manifest = {**manifest, "sft": {"counts": manifest["counts"],
                                    "sha256": dataset_manifest(sft_dir)["sha256"]}}
    write_report(directory / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("preferences", "math", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "demo")
    args = parser.parse_args(argv)
    tasks = ("preference", "math") if args.task == "all" else (
        "preference" if args.task == "preferences" else "math",)
    for kind in tasks:
        prepare_task(kind, args.output_root)


if __name__ == "__main__":
    main()
