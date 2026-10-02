"""Offline audit of copied datasets; never downloads or trains a model."""
import hashlib
import json
from pathlib import Path

from finetune_lab.data import read_jsonl, validate_splits
from finetune_lab.paths import project_path
from finetune_lab.post_data import read_math, read_preferences

SPLITS = ("train", "validation", "test")
EXPECTED = tuple(f"{source}/{task}/manifest.json" for source, tasks in (
    ("processed", ("instruction", "knowledge_cpt", "knowledge_sft", "vision")),
    ("demo", ("instruction", "knowledge_cpt", "knowledge_sft", "vision",
              "post_preferences", "post_math"))) for task in tasks)


def _rows(directory: Path, info: dict, kind: str) -> dict:
    result = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info["sha256"][split]:
            raise ValueError(f"SHA256 mismatch: {project_path(path)}")
        rows = (read_jsonl(path, kind) if kind in ("sft", "cpt", "vision") else
                [json.loads(line) for line in path.read_text().splitlines() if line.strip()])
        if len(rows) != info["counts"][split]:
            raise ValueError(f"Manifest count mismatch: {project_path(path)}")
        result[split] = rows
    if kind in ("sft", "cpt", "vision"):
        validate_splits(result, kind)
    elif kind == "preference":
        read_preferences(directory)
    elif kind == "math":
        read_math(directory)
    else:
        raise ValueError("Unknown manifest kind")
    return result


def _images(directory: Path, splits: dict) -> int:
    from PIL import Image
    seen_files, seen_pixels = set(), set()
    count = 0
    for rows in splits.values():
        for row in rows:
            relative = Path(row["image"])
            path = (directory / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
                raise ValueError("Image path escapes its dataset")
            content = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if row.get("image_sha256", pixels) != pixels:
                raise ValueError(f"Declared RGB pixel hash mismatch: {project_path(path)}")
            if content in seen_files or pixels in seen_pixels:
                raise ValueError("Image file/pixel overlap within or across splits")
            seen_files.add(content)
            seen_pixels.add(pixels)
            count += 1
    return count


def validate_manifest(path: Path | str) -> dict:
    path = Path(path)
    info = json.loads(path.read_text())
    directory = path.parent
    kind = info.get("kind") or {"post_preferences": "preference", "post_math": "math"}.get(directory.name)
    if kind is None:
        raise ValueError("Unknown manifest kind")
    splits = _rows(directory, info, kind)
    files = 3
    if "sft" in info:
        converted = _rows(directory / "sft", info["sft"], "sft")
        for split in SPLITS:
            expected = [{"id": row["id"], "prompt": row["prompt"],
                         "completion": row["chosen"] if kind == "preference" else row["completion"]}
                        for row in splits[split]]
            if converted[split] != expected:
                raise ValueError("SFT conversion differs from its source split/answer")
        files += 3
    return {"dataset": project_path(directory), "counts": info["counts"],
            "jsonl_files": files, "image_files": _images(directory, splits) if kind == "vision" else 0,
            "status": "passed"}


def _knowledge_groups(data_root: Path) -> None:
    for source in ("demo", "processed"):
        cpt, sft = data_root / source / "knowledge_cpt", data_root / source / "knowledge_sft"
        if not cpt.is_dir() or not sft.is_dir():
            continue
        owners = {}
        for directory, kind in ((cpt, "cpt"), (sft, "sft")):
            for split in SPLITS:
                for row in read_jsonl(directory / f"{split}.jsonl", kind):
                    if row["id"] in owners and owners[row["id"]] != split:
                        raise ValueError("Knowledge source-row ID crosses CPT/QA split boundaries")
                    owners[row["id"]] = split


def validate_data_root(data_root: Path | str, require_complete: bool = True) -> dict:
    root = Path(data_root)
    paths = sorted(root.rglob("manifest.json"))
    if not paths:
        raise ValueError("No dataset manifests found")
    if require_complete:
        missing = [name for name in EXPECTED if not (root / name).is_file()]
        if missing:
            raise ValueError("Missing dataset manifests: " + ", ".join(missing))
    datasets = [validate_manifest(path) for path in paths]
    _knowledge_groups(root)
    return {"status": "passed", "manifests": len(paths),
            "jsonl_files": sum(item["jsonl_files"] for item in datasets),
            "image_files": sum(item["image_files"] for item in datasets), "datasets": datasets}
