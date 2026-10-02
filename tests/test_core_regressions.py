"""Fast failure-boundary regressions; no pretrained weights or training jobs."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from finetune_lab import evaluate, torch_text, unsloth_text, vision


@pytest.mark.parametrize("artifact", ["model.safetensors", "metrics.json", "checkpoint-1"])
def test_text_training_preserves_partial_artifacts(tmp_path, artifact):
    existing = tmp_path / artifact
    existing.write_text("preserve")
    args = unsloth_text.parse_args(["--output-dir", str(tmp_path)])
    with patch.object(unsloth_text, "load_runtime") as runtime:
        with pytest.raises(ValueError, match="output-dir"):
            unsloth_text.train(args, [], [])
    runtime.assert_not_called()
    assert existing.read_text() == "preserve"


@pytest.mark.parametrize("method", ["pytorch", "huggingface", "unsloth"])
def test_vision_partial_output_is_preserved(tmp_path, method):
    (tmp_path / "training_metadata.json").write_text("preserve")
    cli = vision.parser(method)
    with pytest.raises(SystemExit):
        vision._validated_args(cli, cli.parse_args(["--output-dir", str(tmp_path)]))
    assert (tmp_path / "training_metadata.json").read_text() == "preserve"


def test_adapter_config_and_metadata_must_be_objects(tmp_path):
    config = tmp_path / "adapter_config.json"
    config.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        unsloth_text.model_origin(str(tmp_path), "main")
    config.write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    (tmp_path / "training_metadata.json").write_text("[]")
    with pytest.raises(ValueError, match="object"):
        unsloth_text.model_origin(str(tmp_path), "main")


def test_evaluation_rejects_empty_rows_before_model_call():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate.evaluate_records(None, None, [], "cpt", 4, 1, "cpu")


def test_generation_budget_cannot_exceed_context():
    with pytest.raises(ValueError, match="context"):
        evaluate.generation_prompt_ids([1, 2], 4, 4, SimpleNamespace(n_positions=4))
    assert evaluate.generation_prompt_ids([1, 2, 3, 4], 4, 2,
                                           SimpleNamespace(n_positions=4)) == [3, 4]


def test_validation_restores_training_mode_when_loss_is_nonfinite():
    import torch

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, **batch):
            return SimpleNamespace(loss=torch.tensor(float("nan")))

    model = Model()
    with pytest.raises(RuntimeError, match="non-finite"):
        torch_text.validation_loss(model, [{"labels": torch.tensor([[1, 2]])}], "cpu")
    assert model.training


def test_empty_optimizer_input_fails_instead_of_looping_forever():
    import torch

    model = torch.nn.Linear(1, 1)
    args = SimpleNamespace(learning_rate=0.1, weight_decay=0.0, max_steps=1,
                           gradient_accumulation=1, max_grad_norm=1.0)
    with pytest.raises(ValueError, match="empty"):
        torch_text.train_loop(model, [], args, "cpu")


def test_vision_runtime_metadata_excludes_hardware_identity():
    import torch

    model = torch.nn.Linear(1, 1)
    with patch.object(torch.cuda, "is_available", return_value=False):
        metadata = vision._runtime_metadata(model)
    assert not {"gpu_name", "gpu_capability", "architecture", "machine"} & metadata.keys()
