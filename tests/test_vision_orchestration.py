"""VLM boundary checks use real files/tensors but never optimize a model."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from finetune_lab import vision
from finetune_lab.prepare_data import demo_vision, save_task


@pytest.fixture
def visual_data(tmp_path):
    path = tmp_path / "data"
    save_task(path, demo_vision(path), "vision", {}, False)
    return path


@pytest.mark.parametrize("method", ["pytorch", "huggingface", "unsloth"])
def test_cli_data_dry_run_does_not_load_any_model(visual_data, method):
    with patch.object(vision, "_load_standard") as standard, patch.object(vision, "_load_unsloth") as unsloth:
        vision.main(method, ["--data-dir", str(visual_data), "--dry-run"])
    standard.assert_not_called()
    unsloth.assert_not_called()


def test_evaluation_cli_does_not_overwrite_existing_report(visual_data, tmp_path):
    output = tmp_path / "report"
    output.mkdir()
    (output / "evaluation.json").write_text("preserve")
    with patch.object(vision, "_load_standard") as loader:
        with pytest.raises(SystemExit):
            vision.main("huggingface", ["--data-dir", str(visual_data), "--evaluate",
                                         "--output-dir", str(output)])
    loader.assert_not_called()
    assert (output / "evaluation.json").read_text() == "preserve"


def test_nonfinite_vlm_metrics_prevent_successful_save(tmp_path):
    with pytest.raises(ValueError):
        vision._save_metrics(tmp_path, {"history": [{"loss": float("nan")}]})
    assert not (tmp_path / "training_metrics.json").exists()


def test_visual_loop_reuses_token_weighted_shared_engine(visual_data, tmp_path):
    import torch
    from finetune_lab import torch_text
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(use_cache=True, image_token_id=5)
    args = SimpleNamespace(data_dir=visual_data, output_dir=tmp_path / "run", seed=42,
                           batch_size=1, decoder_layers=2, device="cpu", learning_rate=.01,
                           gradient_accumulation=4, max_steps=2)
    with patch.object(vision, "_select_language_blocks", return_value=[0]), patch.object(
        torch_text, "train_loop", return_value={"train_target_tokens_seen": 17, "optimizer_steps": 2}
    ) as loop:
        assert vision._run_torch(args, model, object(), [{"id": "train"}]) is model
    config = loop.call_args.args[2]
    assert config.max_grad_norm == 1.0
    assert config.gradient_accumulation == 4
    assert json.loads((args.output_dir / "training_metrics.json").read_text())["train_target_tokens_seen"] == 17


def test_vlm_evaluation_uses_heldout_images_without_a_trainer(visual_data, tmp_path):
    import torch
    model = torch.nn.Linear(1, 1)
    model.generate = lambda **kwargs: torch.tensor([[1, 2, 3]])
    processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "image prompt",
                                 batch_decode=lambda *args, **kwargs: ["healthy"])
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return processor.apply_chat_template(*args, **kwargs)
        def __call__(self, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}
        def batch_decode(self, *args, **kwargs):
            return processor.batch_decode(*args, **kwargs)
    args = SimpleNamespace(data_dir=visual_data, output_dir=tmp_path / "eval", max_eval_samples=1,
                           model="org/vision", revision="pinned", split="test")
    rows = vision.load_rows(visual_data, "test")[:1]
    vision.evaluate(args, model, Processor(), rows)
    result = json.loads((args.output_dir / "evaluation.json").read_text())
    assert result["split"] == "test"
    assert result["accuracy"] == float(rows[0]["label"] == "healthy")
