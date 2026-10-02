"""Explicit dataset download; never invoked implicitly by training."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from finetune_lab.data import LABELS, signature, split_records, validate_splits, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "instruction": ("databricks/databricks-dolly-15k", "CC-BY-SA-3.0"),
    "knowledge": ("allenai/sciq", "CC-BY-NC-3.0"),
    "vision": ("AI-Lab-Makerere/beans", "dataset card: unspecified; consult source"),
}


def unique_rows(rows: list[dict], kind: str, excluded: set | None = None) -> list[dict]:
    seen = set(excluded or ())
    result = []
    for row in rows:
        key = signature(row, kind)
        if key not in seen:
            result.append(dict(row))
            seen.add(key)
    return result


def independent_splits(splits: dict, kind: str) -> dict:
    # Give evaluation examples priority; remove exact normalized overlap from train.
    result, seen = {}, set()
    for name in ("test", "validation", "train"):
        result[name] = unique_rows(splits[name], kind, seen)
        seen.update(signature(row, kind) for row in result[name])
    return {name: result[name] for name in ("train", "validation", "test")}


def save_task(output: Path, splits: dict, kind: str, metadata: dict, overwrite: bool) -> None:
    if output.exists() and any(output.glob("*.jsonl")) and not overwrite:
        raise FileExistsError(f"{output} already contains data; use --overwrite explicitly")
    validate_splits(splits, kind)
    if any(not rows for rows in splits.values()):
        raise ValueError(f"{output}: a split is empty after filtering")
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    hashes = {name: hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()
              for name in splits}
    manifest = {**metadata, "kind": kind, "counts": {k: len(v) for k, v in splits.items()},
                "sha256": hashes}
    if kind == "vision":
        manifest = {**manifest, "label_counts": {
            name: dict(Counter(row["label"] for row in rows)) for name, rows in splits.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"{output}: {manifest['counts']}")


def demo_text(task: str) -> tuple[dict, str]:
    if task == "instruction":
        rows = [{"id": f"demo-instruction-{i}",
                 "prompt": f"Return the number {i} with no extra words.",
                 "completion": str(i)} for i in range(40)]
        return split_records(rows, 24, 8), "sft"
    rows = [{"id": f"demo-fact-{i}", "prompt": f"What is the docking code of fictional station Nara-{i}?",
             "completion": f"K{i + 700}",
             "text": f"In the fictional Nara learning world, station Nara-{i} has docking code K{i + 700}."}
            for i in range(40)]
    return split_records(rows, 24, 8), "sft"


def demo_vision(output: Path) -> dict:
    from PIL import Image, ImageDraw
    splits = {}
    for offset, (name, count) in enumerate((("train", 12), ("validation", 6), ("test", 6))):
        rows = []
        for i in range(count):
            label = LABELS[i % 3]
            relative = f"images/demo-{name}-{i:03}.png"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (64, 64), (30 + i * 5, 110 + offset * 10, 35))
            draw = ImageDraw.Draw(image)
            draw.ellipse((12 + offset, 8, 50, 58 - offset), fill=(70, 160, 60))
            # Intentionally synthetic fixtures; no crop-disease meaning.
            for spot in range(i % 3 + 1):
                draw.ellipse((20 + spot * 7, 20 + i, 24 + spot * 7, 24 + i), fill=(120, 80, 20))
            image.save(path)
            rows.append({"id": f"demo-vision-{name}-{i}", "image": relative, "label": label})
        splits[name] = rows
    return splits


def load_hf(task: str, revision: str):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset_id, license_name = SOURCES[task]
    resolved = HfApi().dataset_info(dataset_id, revision=revision).sha
    dataset = load_dataset(dataset_id, revision=resolved)
    return dataset, {"source": "huggingface", "dataset_id": dataset_id, "revision": resolved,
                     "license": license_name, "url": f"https://huggingface.co/datasets/{dataset_id}"}


def prepare_instruction(dataset, args) -> dict:
    rows = [{"id": f"dolly-{i}", "prompt": row["instruction"].strip() + (
        "\n\nContext:\n" + row["context"].strip() if row["context"].strip() else ""),
        "completion": row["response"].strip(), "category": row["category"]}
        for i, row in enumerate(dataset["train"])
        if row["instruction"].strip() and row["response"].strip()]
    return split_records(unique_rows(rows, "sft"), args.train_samples, args.eval_samples, args.seed)


def prepare_knowledge(dataset, args) -> tuple[dict, dict]:
    sft, cpt = {}, {}
    for name in ("train", "validation", "test"):
        size = args.train_samples if name == "train" else args.eval_samples
        selected = dataset[name].shuffle(seed=args.seed)
        rows = [{"id": f"sciq-{name}-{i}", "prompt": row["question"].strip(),
                 "completion": row["correct_answer"].strip(), "support": row["support"].strip()}
                for i, row in enumerate(selected)
                if row["question"].strip() and row["correct_answer"].strip()]
        # Split is chosen at source-example level BEFORE CPT/QA conversion.
        sft[name] = rows[:size]
        cpt[name] = [{"id": row["id"], "text": row["support"]}
                     for row in sft[name] if row["support"]]
    return independent_splits(sft, "sft"), independent_splits(cpt, "cpt")


def prepare_vision(dataset, args, output: Path) -> dict:
    splits, seen_images = {}, set()
    for name in ("test", "validation", "train"):
        size = args.train_samples if name == "train" else args.eval_samples
        rows = []
        for i, row in enumerate(dataset[name].shuffle(seed=args.seed)):
            image = row["image"].convert("RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen_images:
                continue
            seen_images.add(digest)
            relative = f"images/{name}-{i:05}.png"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            rows.append({"id": f"beans-{name}-{i}", "image": relative,
                         "label": LABELS[int(row["labels"])], "image_sha256": digest})
            if len(rows) == size:
                break
        splits[name] = rows
    return {name: splits[name] for name in ("train", "validation", "test")}


def prepare_task(task: str, args) -> None:
    names = ("knowledge_cpt", "knowledge_sft") if task == "knowledge" else (task,)
    for name in names:
        path = args.output_root / name
        if path.exists() and any(path.glob("*.jsonl")) and not args.overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite or another --output-root")
    metadata = {"seed": args.seed, "requested_train": args.train_samples,
                "requested_eval": args.eval_samples}
    if args.source == "demo":
        metadata = {**metadata, "source": "authored_demo", "license": "CC0-1.0",
                    "warning": "Pipeline fixtures, not a meaningful accuracy benchmark"}
        if task == "vision":
            splits = demo_vision(args.output_root / task)
            save_task(args.output_root / task, splits, "vision", metadata, args.overwrite)
        else:
            splits, kind = demo_text(task)
            if task == "knowledge":
                cpt = {name: [{"id": row["id"], "text": row["text"]} for row in rows]
                       for name, rows in splits.items()}
                save_task(args.output_root / "knowledge_cpt", cpt, "cpt", metadata, args.overwrite)
                splits = {name: [{k: row[k] for k in ("id", "prompt", "completion")} for row in rows]
                          for name, rows in splits.items()}
            save_task(args.output_root / names[-1], splits, kind, metadata, args.overwrite)
            if task == "knowledge":
                probe = [{"id": f"seen-fact-probe-{row['id']}",
                          "prompt": row["prompt"].replace("What is the docking code of fictional station", "Give only the docking code for fictional station"),
                          "completion": row["completion"]} for row in splits["train"]]
                write_jsonl(args.output_root / "knowledge_sft/probe_seen_facts.jsonl", probe)
        return
    dataset, source = load_hf(task, args.revision)
    metadata = {**metadata, **source}
    if task == "instruction":
        save_task(args.output_root / task, prepare_instruction(dataset, args), "sft", metadata, args.overwrite)
    elif task == "knowledge":
        sft, cpt = prepare_knowledge(dataset, args)
        save_task(args.output_root / "knowledge_sft", sft, "sft", metadata, args.overwrite)
        save_task(args.output_root / "knowledge_cpt", cpt, "cpt", metadata, args.overwrite)
    else:
        path = args.output_root / task
        save_task(path, prepare_vision(dataset, args, path), "vision", metadata, args.overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["instruction", "knowledge", "vision", "all"], default="all")
    parser.add_argument("--source", choices=["hf", "demo"], default="hf")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.train_samples, args.eval_samples) < 1:
        parser.error("sample counts must be positive")
    tasks = ("instruction", "knowledge", "vision") if args.task == "all" else (args.task,)
    try:
        for task in tasks:
            prepare_task(task, args)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
