"""Real offline forward/reload and simulated orchestration, without model training."""
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from finetune_lab import evaluate, hf_text, infer, merge, prepare_data, torch_text, vision


@pytest.fixture
def offline_checkpoint(tmp_path):
    import torch

    torch.set_num_threads(1)
    model, tokenizer = torch_text.create_smoke_model(64)
    output = tmp_path / "offline-model"
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    return output


def test_real_checkpoint_forward_generation_and_reloading(offline_checkpoint):
    import torch

    model, tokenizer, metadata = evaluate.load_text_model(str(offline_checkpoint), "main", "cpu")
    assert metadata == {}
    rows = [{"id": "sft", "prompt": "hello", "completion": "hello world"}]
    report = evaluate.evaluate_records(model, tokenizer, rows, "sft", 32, 2, "cpu")
    assert report["count"] == 1
    assert math.isfinite(report["loss"])
    assert report["target_tokens"] == 3
    assert len(report["generations"]) == 1
    cpt = evaluate.evaluate_records(model, tokenizer, [{"id": "cpt", "text": "hello world"}],
                                    "cpt", 32, 2, "cpu")
    assert cpt["target_tokens"] == 2
    assert evaluate.resolve_device("cpu") == torch.device("cpu")


def test_real_offline_adapter_reload_retains_base_identity(offline_checkpoint, tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(offline_checkpoint)
    adapter = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"],
                                             task_type="CAUSAL_LM"))
    saved = tmp_path / "adapter"
    adapter.save_pretrained(saved, safe_serialization=True)
    AutoTokenizer.from_pretrained(offline_checkpoint).save_pretrained(saved)
    (saved / "training_metadata.json").write_text(json.dumps({
        "base_model": str(offline_checkpoint), "revision": "main", "kind": "sft"}))
    loaded, tokenizer, metadata = evaluate.load_text_model(str(saved), "main", "cpu")
    with torch.inference_mode():
        loss = loaded(**tokenizer("hello world", return_tensors="pt"),
                      labels=tokenizer("hello world", return_tensors="pt")["input_ids"]).loss
    assert torch.isfinite(loss)
    assert metadata["kind"] == "sft"


def test_evaluation_cli_preserves_existing_report_before_loading(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("preserve")
    argv = ["evaluate", "--model", "org/model", "--data-dir", str(tmp_path), "--output", str(output)]
    with patch.object(sys, "argv", argv), patch.object(evaluate, "load_text_model") as loader:
        with pytest.raises(SystemExit):
            evaluate.main()
    loader.assert_not_called()
    assert output.read_text() == "preserve"


def test_evaluation_cli_writes_real_forward_report(offline_checkpoint, tmp_path):
    (tmp_path / "test.jsonl").write_text(json.dumps({"id": "x", "text": "hello world"}) + "\n")
    output = tmp_path / "report.json"
    argv = ["evaluate", "--model", str(offline_checkpoint), "--data-dir", str(tmp_path),
            "--kind", "cpt", "--device", "cpu", "--output", str(output)]
    with patch.object(sys, "argv", argv):
        evaluate.main()
    assert math.isfinite(json.loads(output.read_text())["loss"])


def test_infer_uses_training_template_and_rejects_context_overflow(offline_checkpoint):
    argv = ["infer", "--model", str(offline_checkpoint), "--prompt", "hello", "--device", "cpu",
            "--max-new-tokens", "2"]
    with patch.object(sys, "argv", argv):
        infer.main()
    with patch.object(sys, "argv", argv[:-1] + ["100"]):
        with pytest.raises(SystemExit):
            infer.main()


def test_merge_adapter_without_training(offline_checkpoint, tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = tmp_path / "adapter"
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(offline_checkpoint),
                          LoraConfig(r=2, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    model.save_pretrained(adapter, safe_serialization=True)
    AutoTokenizer.from_pretrained(offline_checkpoint).save_pretrained(adapter)
    output = tmp_path / "merged"
    with patch.object(sys, "argv", ["merge", "--adapter", str(adapter), "--output-dir", str(output)]):
        merge.main()
    assert (output / "model.safetensors").is_file()
    assert not (output / "adapter_config.json").exists()
    with patch.object(sys, "argv", ["merge", "--adapter", str(adapter), "--output-dir", str(output)]):
        with pytest.raises(SystemExit):
            merge.main()


def test_hf_orchestration_never_uses_test_rows_or_executes_optimizer(tmp_path):
    """Real model/config/encoding plus simulated Trainer boundary, not a training claim."""
    import torch
    import transformers

    args = hf_text.parse_args("instruction", ["--smoke-model", "--method", "full", "--device", "cpu",
                                               "--output-dir", str(tmp_path / "run"), "--max-length", "32"])
    rows = {split: [{"id": split, "prompt": f"hello {split}", "completion": "world"}]
            for split in ("train", "validation", "test")}
    captured = {}

    class Trainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]

        def evaluate(self, metric_key_prefix):
            return {f"{metric_key_prefix}_loss": 1.0}

        def train(self):
            # No forward/backward or optimizer is executed by this boundary fake.
            return SimpleNamespace(metrics={"train_loss": 1.0})

        def save_model(self, path):
            self.model.save_pretrained(path)

    with patch.object(transformers, "Trainer", Trainer):
        metadata = hf_text._train(args, rows)
    assert len(captured["train_dataset"]) == 1
    assert len(captured["eval_dataset"]) == 1
    assert set(metadata["runtime"]).isdisjoint({"machine", "gpu_name", "gpu_capability"})
    assert metadata["samples"]["test"] == 1
    assert (args.output_dir / "config.json").is_file()


def test_torch_cli_orchestration_records_only_simulated_loop_metrics(tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({
            "id": split, "prompt": f"hello {split}", "completion": "world"}) + "\n")
    output = tmp_path / "run"
    with patch.object(torch_text, "train_loop", return_value={"optimizer_steps": 0}) as loop:
        torch_text.main("instruction", ["--smoke-model", "--data-dir", str(tmp_path),
                                         "--output-dir", str(output), "--device", "cpu", "--max-length", "32"])
    loop.assert_called_once()
    metadata = json.loads((output / "training_metadata.json").read_text())
    assert metadata["optimizer_steps"] == 0
    assert math.isfinite(metadata["validation_loss_before"])
    assert (output / "model.safetensors").is_file()


def test_prepare_authored_all_tasks_and_cli(tmp_path):
    argv = ["prepare", "--source", "demo", "--output-root", str(tmp_path)]
    with patch.object(sys, "argv", argv):
        prepare_data.main()
    assert (tmp_path / "instruction/manifest.json").is_file()
    assert (tmp_path / "vision/manifest.json").is_file()
    vision.validate_vision_splits(tmp_path / "vision", {
        split: vision.load_rows(tmp_path / "vision", split)
        for split in ("train", "validation", "test")})


class SourceRows(list):
    def shuffle(self, seed):
        return self


def test_prepare_hf_conversion_preserves_source_splits(tmp_path):
    from PIL import Image

    args = Namespace(train_samples=1, eval_samples=1, seed=42)
    dolly = {"train": [{"instruction": f"question {i}", "context": "context", "response": "answer",
                         "category": "qa"} for i in range(3)]}
    assert len(prepare_data.prepare_instruction(dolly, args)["train"]) == 1
    sciq = {split: SourceRows([{"question": split, "correct_answer": "answer", "support": f"fact {split}"}])
            for split in ("train", "validation", "test")}
    sft, cpt = prepare_data.prepare_knowledge(sciq, args)
    assert sft["train"][0]["id"] == cpt["train"][0]["id"]
    beans = {split: SourceRows([{"image": Image.new("RGB", (4, 4), (i, 0, 0)), "labels": i}])
             for i, split in enumerate(("train", "validation", "test"))}
    splits = prepare_data.prepare_vision(beans, args, tmp_path)
    prepare_data.save_task(tmp_path, splits, "vision", {"source": "fixture"}, False)
    assert len(splits["test"]) == 1


@pytest.mark.parametrize("task", ["instruction", "knowledge", "vision"])
def test_prepare_hub_orchestration_with_local_source_fakes(tmp_path, task):
    args = Namespace(output_root=tmp_path, source="hf", seed=42, train_samples=1,
                     eval_samples=1, overwrite=False, revision="fixed")
    if task == "instruction":
        dataset = {"train": [{"instruction": f"q {i}", "context": "", "response": "a", "category": "qa"}
                              for i in range(3)]}
    elif task == "knowledge":
        dataset = {split: SourceRows([{"question": split, "correct_answer": "a", "support": f"fact {split}"}])
                   for split in ("train", "validation", "test")}
    else:
        from PIL import Image
        dataset = {split: SourceRows([{"image": Image.new("RGB", (4, 4), (i, 0, 0)), "labels": i}])
                   for i, split in enumerate(("train", "validation", "test"))}
    with patch.object(prepare_data, "load_hf", return_value=(dataset, {"revision": "fixed"})):
        prepare_data.prepare_task(task, args)
    folder = "knowledge_sft" if task == "knowledge" else task
    assert json.loads((tmp_path / folder / "manifest.json").read_text())["counts"]["train"] == 1
