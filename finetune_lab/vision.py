"""Image-conditioned label generation with a small VLM or Unsloth QLoRA.

The default PyTorch path trains the final language blocks, leaving the image
encoder and connector fixed. Importing this module needs only the stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
from pathlib import Path

BEAN_LABELS = ("angular_leaf_spot", "bean_rust", "healthy")
SMOL_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
UNSLOTH_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = (
    "Classify the bean leaf in this image. Reply with exactly one label: "
    "angular_leaf_spot, bean_rust, healthy. Do not add an explanation."
)


def image_path(data_dir, relative_path):
    """Allow only existing files inside the selected dataset directory."""
    root = Path(data_dir).resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("image must be a relative path inside --data-dir")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("image path escapes --data-dir")
    if not candidate.is_file():
        raise FileNotFoundError(f"image not found: {candidate}")
    return candidate


def build_messages(image, label=None):
    user = {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": INSTRUCTION},
    ]}
    if label is None:
        return [user]
    if label not in BEAN_LABELS:
        raise ValueError(f"unknown bean label: {label}")
    return [user, {"role": "assistant", "content": [{"type": "text", "text": label}]}]


def completion_labels(full_ids, prompt_ids, attention_mask, ignored_ids=()):
    """Mask prompt, expanded image positions, and padding after processing.

    Prompt and full IDs must align; counting text-only tokenizer IDs would put
    the boundary before the expanded visual tokens. Mask by attention_mask so
    a real EOS remains supervised even when EOS is also the pad token.
    """
    if len(full_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask have different lengths")
    attended = [token for token, active in zip(full_ids, attention_mask) if active]
    if attended[:len(prompt_ids)] != list(prompt_ids):
        raise ValueError("processor prompt/full prefix mismatch; inspect chat template")
    if len(attended) <= len(prompt_ids):
        raise ValueError("assistant completion has no tokens")
    ignored = set(ignored_ids)
    positions = iter(range(len(attended)))
    labels = [
        (token if next(positions) >= len(prompt_ids) and token not in ignored else -100)
        if active else -100
        for token, active in zip(full_ids, attention_mask)
    ]
    if all(token == -100 for token in labels):
        raise ValueError("assistant completion contains no supervised tokens")
    return labels


def normalize_prediction(text):
    normalized = text.strip().lower()
    return normalized if normalized in BEAN_LABELS else "invalid"


def classification_metrics(truth, predicted):
    """3-class macro-F1; invalid text counts as an incorrect prediction."""
    if not truth or len(truth) != len(predicted):
        raise ValueError("evaluation needs equally sized nonempty truth/prediction lists")
    if any(label not in BEAN_LABELS for label in truth):
        raise ValueError("evaluation contains an unknown reference label")
    normalized = [normalize_prediction(label) for label in predicted]
    columns = (*BEAN_LABELS, "invalid")
    confusion = [[sum(t == label and p == guess for t, p in zip(truth, normalized))
                  for guess in columns] for label in BEAN_LABELS]
    scores = []
    for label in BEAN_LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, normalized))
        fp = sum(t != label and p == label for t, p in zip(truth, normalized))
        fn = sum(t == label and p != label for t, p in zip(truth, normalized))
        scores = [*scores, 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0]
    return {
        "samples": len(truth),
        "accuracy": sum(t == p for t, p in zip(truth, normalized)) / len(truth),
        "macro_f1": sum(scores) / len(BEAN_LABELS),
        "per_class_f1": dict(zip(BEAN_LABELS, scores)),
        "invalid_predictions": normalized.count("invalid"),
        "confusion_rows": list(BEAN_LABELS),
        "confusion_columns": list(columns),
        "confusion_matrix": confusion,
    }


def load_rows(data_dir, split):
    from finetune_lab.data import read_jsonl

    rows = read_jsonl(Path(data_dir) / f"{split}.jsonl", "vision")
    if not rows:
        raise ValueError(f"{split}.jsonl is empty")
    for row in rows:
        image_path(data_dir, row["image"])
    return rows


def validate_vision_splits(data_dir, splits):
    """Reject cross-split identity/path/content overlap before GPU model loading.

    File SHA256 detects byte-identical copies under different names. Optional
    image_sha256 is the preparer's RGB pixel hash (not a file hash): validate
    its format and detect duplicates without adding a decoder dependency here.
    Re-encoded duplicates without that field and near-duplicates need a
    separate decoded-pixel/perceptual or source-group audit.
    """
    seen_ids, seen_paths, seen_content, seen_pixels = {}, {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            identity = row["id"]
            path = image_path(data_dir, row["image"])
            if identity in seen_ids:
                raise ValueError(f"id overlap: {seen_ids[identity]} and {split}: {identity}")
            if path in seen_paths:
                raise ValueError(f"image path overlap: {seen_paths[path]} and {split}: {path}")
            with path.open("rb") as image_file:
                digest = hashlib.file_digest(image_file, "sha256").hexdigest()
            declared = row.get("image_sha256")
            if declared is not None:
                if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared):
                    raise ValueError(f"image_sha256 must be a 64-character hex pixel hash: {path}")
                pixel_hash = declared.lower()
                if pixel_hash in seen_pixels:
                    raise ValueError(f"image pixel hash overlap: {seen_pixels[pixel_hash]} and {split}: {path}")
                seen_pixels = {**seen_pixels, pixel_hash: split}
            if digest in seen_content:
                raise ValueError(f"image content overlap: {seen_content[digest]} and {split}: {path}")
            seen_ids = {**seen_ids, identity: split}
            seen_paths = {**seen_paths, path: split}
            seen_content = {**seen_content, digest: split}


def _open_image(data_dir, row):
    from PIL import Image

    with Image.open(image_path(data_dir, row["image"])) as image:
        return image.convert("RGB")


class VisionCollator:
    """Batch RGB images and compute response-only masks after image expansion."""

    def __init__(self, processor, data_dir, image_token_id):
        self.processor = processor
        self.data_dir = data_dir
        self.image_token_id = image_token_id

    def __call__(self, rows):
        import torch

        images = [_open_image(self.data_dir, row) for row in rows]
        texts = [self.processor.apply_chat_template(
            build_messages(image, row["label"]), tokenize=False,
            add_generation_prompt=False,
        ) for image, row in zip(images, rows)]
        batch = self.processor(text=texts, images=[[image] for image in images],
                               padding=True, truncation=False, return_tensors="pt")
        labels = []
        for index, image in enumerate(images):
            prompt = self.processor.apply_chat_template(
                build_messages(image), tokenize=False, add_generation_prompt=True)
            prompt_inputs = self.processor(text=prompt, images=[image],
                                           truncation=False, return_tensors="pt")
            ignored = {self.image_token_id} if self.image_token_id is not None else set()
            labels = [*labels, completion_labels(
                batch["input_ids"][index].tolist(),
                prompt_inputs["input_ids"][0].tolist(),
                batch["attention_mask"][index].tolist(), ignored,
            )]
        return {**batch, "labels": torch.tensor(labels, dtype=torch.long)}


def _device(requested):
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or supported hardware")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable on this machine")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_standard(model_id, revision, device):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    local = Path(model_id)
    adapter = local.is_dir() and (local / "adapter_config.json").is_file()
    if adapter:
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(model_id)
        metadata_file = local / "training_metadata.json"
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        base = config.base_model_name_or_path
        base_revision = metadata.get("resolved_revision") or metadata.get("revision", revision)
    else:
        base, base_revision = model_id, revision
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    processor.tokenizer.padding_side = "right"
    if "SmolVLM" in base:
        # Bound visual patch count for this introductory image exercise.
        processor.image_processor.size = {"longest_edge": 512}
    model = AutoModelForVision2Seq.from_pretrained(
        base, revision=base_revision, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, model_id)
    return model.to(device), processor


def _select_language_blocks(model, count):
    """Train ordinary parameters only in final decoder blocks, plus final norm."""
    names = [name for name, _ in model.named_parameters()]
    layer_ids = sorted({int(match.group(1)) for name in names
                        if (match := re.search(r"text_model\.layers\.(\d+)\.", name))})
    if not layer_ids:
        raise ValueError("PyTorch exercise requires a SmolVLM text_model decoder")
    selected = set(layer_ids[-count:])
    for name, parameter in model.named_parameters():
        match = re.search(r"text_model\.layers\.(\d+)\.", name)
        parameter.requires_grad_(bool(match and int(match.group(1)) in selected)
                                 or ".text_model.norm." in name)
    return sorted(selected)


def _attach_language_lora(model):
    from peft import LoraConfig, get_peft_model

    targets = r".*text_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
    if not any(re.fullmatch(targets, name) for name, _ in model.named_modules()):
        raise ValueError("Hugging Face exercise requires SmolVLM language q_proj/v_proj")
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=targets, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    return model


def _run_torch(args, model, processor, train):
    import torch
    from torch.utils.data import DataLoader

    selected = _select_language_blocks(model, args.decoder_layers)
    print(json.dumps({"training_decoder_layers": selected, "vision_encoder_frozen": True}))
    model.config.use_cache = False
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator,
                        collate_fn=VisionCollator(processor, args.data_dir, model.config.image_token_id))
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    history = []
    for step in range(args.max_steps):
        losses = []
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not math.isfinite(loss.item()):
                raise RuntimeError("non-finite loss: inspect assistant masks and precision")
            (loss / args.gradient_accumulation).backward()
            losses = [*losses, loss.item()]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics = {"step": step + 1, "loss": sum(losses) / len(losses)}
        history = [*history, metrics]
        print(json.dumps(metrics))
    _save_metrics(args.output_dir, {"history": history})
    return model


def _save_metrics(output_dir, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def _run_hf(args, model, processor, train, validation):
    import torch
    from transformers import Trainer, TrainingArguments

    model = _attach_language_lora(model)
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, logging_steps=1, save_strategy="no",
        eval_strategy="steps", eval_steps=args.max_steps, report_to="none",
        remove_unused_columns=False, seed=args.seed, use_cpu=args.device == "cpu",
        label_names=["labels"],
        bf16=args.device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=args.device == "cuda" and not torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=args.device == "cuda",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train,
                      eval_dataset=validation, data_collator=VisionCollator(
                          processor, args.data_dir, model.config.image_token_id))
    result = trainer.train()
    _save_metrics(args.output_dir, {**result.metrics, "history": trainer.state.log_history})
    return model


def _load_unsloth(args, training):
    from unsloth import FastVisionModel  # Must precede torch/transformers imports.
    import torch

    if args.device not in ("auto", "cuda") or not torch.cuda.is_available():
        raise ValueError("Unsloth vision exercise requires a supported NVIDIA CUDA machine")
    metadata_file = Path(args.model) / "training_metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    revision = metadata.get("resolved_revision") or metadata.get("revision", args.revision)
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model, revision=revision, load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    if training:
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=False,
            r=8, lora_alpha=16, lora_dropout=0, bias="none", random_state=args.seed,
        )
        FastVisionModel.for_training(model)
    else:
        FastVisionModel.for_inference(model)
    return model, processor


def _run_unsloth(args, model, processor, train, validation):
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from trl import SFTConfig, SFTTrainer

    def convert(rows):
        return [{"messages": build_messages(_open_image(args.data_dir, row), row["label"])}
                for row in rows]

    class UntruncatedVisionDataCollator(UnslothVisionDataCollator):
        def __init__(self, *values, **keywords):
            super().__init__(*values, **keywords)
            # Native None inherits model.max_seq_length in current Unsloth.
            # Explicitly disable tokenizer truncation to preserve image features.
            self.max_seq_length = None
            self.truncation = False

    collator = UntruncatedVisionDataCollator(
        model, processor, max_seq_length=None, resize="max", train_on_responses_only=True,
        instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )
    config = SFTConfig(
        output_dir=str(args.output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, logging_steps=1, save_strategy="no",
        eval_strategy="steps", eval_steps=args.max_steps, optim="adamw_8bit",
        report_to="none", remove_unused_columns=False, seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
    )
    trainer = SFTTrainer(model=model, processing_class=processor, args=config,
                         train_dataset=convert(train), eval_dataset=convert(validation),
                         data_collator=collator)
    result = trainer.train()
    _save_metrics(args.output_dir, {**result.metrics, "history": trainer.state.log_history})
    return model


def evaluate(args, model, processor, rows):
    import torch

    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for row in rows[:args.max_eval_samples]:
        image = _open_image(args.data_dir, row)
        prompt = processor.apply_chat_template(build_messages(image), tokenize=False,
                                               add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                       use_cache=True)
        answer = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        predictions = [*predictions, {"id": row["id"], "reference": row["label"],
                                    "generated": answer, "prediction": normalize_prediction(answer)}]
    metrics = classification_metrics([item["reference"] for item in predictions],
                                     [item["prediction"] for item in predictions])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps({"model": args.model, "revision": args.revision, "split": args.split,
                    **metrics, "predictions": predictions}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def parser(method=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--method", choices=("pytorch", "huggingface", "unsloth"), default=method or "pytorch")
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/vision")
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--model")
    cli.add_argument("--revision", default="main")
    cli.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    cli.add_argument("--max-steps", type=int, default=20)
    cli.add_argument("--batch-size", type=int, default=1)
    cli.add_argument("--gradient-accumulation", type=int, default=4)
    cli.add_argument("--learning-rate", type=float)
    cli.add_argument("--decoder-layers", type=int, default=2)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--evaluate", action="store_true")
    cli.add_argument("--split", choices=("train", "validation", "test"), default="test")
    cli.add_argument("--max-eval-samples", type=int, default=128)
    return cli


def _runtime_metadata(model):
    import torch

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(), "architecture": platform.machine(),
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "unsloth")},
        "cuda_version": torch.version.cuda,
        "device": str(next(model.parameters()).device),
        "loaded_parameter_dtype": str(next(model.parameters()).dtype),
        "gpu_name": torch.cuda.get_device_name() if cuda else None,
        "gpu_capability": list(torch.cuda.get_device_capability()) if cuda else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
    }


def _validated_args(cli, args):
    for name in ("max_steps", "batch_size", "gradient_accumulation", "decoder_layers", "max_eval_samples"):
        if getattr(args, name) <= 0:
            cli.error(f"--{name.replace('_', '-')} must be positive")
    model = args.model or (UNSLOTH_MODEL if args.method == "unsloth" else SMOL_MODEL)
    output_dir = args.output_dir or ROOT / "outputs/vision" / args.method
    if not args.evaluate:
        if (Path(model) / "adapter_config.json").is_file():
            cli.error("training an existing adapter is unsupported; select its base model and a new output directory")
        saved_files = ("config.json", "adapter_config.json", "model.safetensors", "pytorch_model.bin")
        if any((output_dir / name).is_file() for name in saved_files):
            cli.error("--output-dir already contains a saved model/adapter; choose a new directory")
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-5 if args.method == "pytorch" else 2e-4
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        cli.error("--learning-rate must be finite and positive")
    return argparse.Namespace(**{**vars(args), "model": model,
                                 "output_dir": output_dir, "learning_rate": learning_rate})


def _save_training(args, model, processor, train_rows, validation):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    manifest_file = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else None
    metadata = {"base_model": args.model, "revision": args.revision, "kind": "vision",
                "method": args.method, "max_steps": args.max_steps, "seed": args.seed,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.learning_rate,
                "decoder_layers": args.decoder_layers if args.method == "pytorch" else None,
                "lora_rank": 8 if args.method != "pytorch" else None,
                "lora_alpha": 16 if args.method != "pytorch" else None,
                "train_rows": len(train_rows), "validation_rows": len(validation),
                "data_dir": str(args.data_dir.resolve()), "prompt": INSTRUCTION,
                "vision_encoder_frozen": True,
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "dataset_manifest": manifest, "runtime": _runtime_metadata(model)}
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved model/adapter and processor: {args.output_dir}")


def main(method=None, argv=None):
    cli = parser(method)
    args = _validated_args(cli, cli.parse_args(argv))
    try:
        rows = load_rows(args.data_dir, args.split if args.evaluate else "train")
        validation = [] if args.evaluate else load_rows(args.data_dir, "validation")
        if not args.evaluate:
            test = load_rows(args.data_dir, "test")
            validate_vision_splits(args.data_dir, {"train": rows, "validation": validation, "test": test})
        if args.dry_run:
            print(json.dumps({"task": "vision", "method": args.method, "model": args.model,
                              "rows": len(rows), "validation_rows": len(validation),
                              "labels": list(BEAN_LABELS), "sample": rows[0],
                              "checked": "schema, image paths and training split hashes; no model loaded"}, indent=2))
            return
        random.seed(args.seed)
        if args.method == "unsloth":
            model, processor = _load_unsloth(args, training=not args.evaluate)
        else:
            import torch

            torch.manual_seed(args.seed)
            args = argparse.Namespace(**{**vars(args), "device": _device(args.device)})
            model, processor = _load_standard(args.model, args.revision, args.device)
        if args.evaluate:
            evaluate(args, model, processor, rows)
            return
        functions = {"pytorch": _run_torch, "huggingface": _run_hf, "unsloth": _run_unsloth}
        if args.method == "pytorch":
            model = functions[args.method](args, model, processor, rows)
        else:
            model = functions[args.method](args, model, processor, rows, validation)
        _save_training(args, model, processor, rows, validation)
    except (ValueError, FileNotFoundError, ImportError) as error:
        cli.exit(2, f"vision lab: {error}\n")


if __name__ == "__main__":
    main()
