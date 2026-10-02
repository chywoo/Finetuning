"""Dependency-free regressions for post-training runtime and report contracts."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pytest

from finetune_lab import post_data, post_dpo, post_rlhf


def fake_torch(cuda=True, bf16=True):
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: cuda, is_bf16_supported=lambda: bf16,
        device_count=lambda: 2,
    ))


def test_cuda_runtime_accepts_capable_gpu_without_host_architecture_restriction():
    torch = fake_torch()
    with patch.dict(sys.modules, {"torch": torch}):
        assert post_data.require_cuda() is torch


@pytest.mark.parametrize("cuda,bf16,message", [
    (False, True, "CUDA"), (True, False, "BF16"),
])
def test_cuda_runtime_fails_closed_on_missing_capability(cuda, bf16, message):
    with patch.dict(sys.modules, {"torch": fake_torch(cuda, bf16)}):
        with pytest.raises(RuntimeError, match=message):
            post_data.require_cuda()


def test_rlhf_runtime_uses_same_capability_guard_and_pinned_trl():
    torch = fake_torch()
    with patch.dict(sys.modules, {"torch": torch}), patch(
        "importlib.metadata.version", return_value="0.24.0"
    ):
        assert post_rlhf._spark_runtime() is torch
    with patch("importlib.metadata.version", return_value="0.25.0"):
        with pytest.raises(RuntimeError, match="0.24.0"):
            post_rlhf._spark_runtime()


def test_dataset_manifest_records_project_relative_location(tmp_path, monkeypatch):
    monkeypatch.setattr(post_data, "ROOT", tmp_path)
    directory = tmp_path / "data" / "fixture"
    directory.mkdir(parents=True)
    for split in post_data.SPLITS:
        (directory / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = post_data.dataset_manifest(directory)
    assert manifest["data_dir"] == "data/fixture"
    assert all(len(value) == 64 for value in manifest["sha256"].values())


def test_report_path_hides_external_location(tmp_path):
    assert post_data.project_path(tmp_path) == "<external path>"


def test_rlhf_metadata_omits_device_identity_and_absolute_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(post_data, "ROOT", tmp_path)
    monkeypatch.setattr(post_rlhf, "ROOT", tmp_path)
    directory = tmp_path / "data"
    directory.mkdir()
    for split in post_data.SPLITS:
        (directory / f"{split}.jsonl").write_text("{}\n")
    args = SimpleNamespace(stage="reward", model="public/model", revision="main",
                           data_dir=directory, reward_model=None, seed=42)
    torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test"))
    with patch("importlib.metadata.version", return_value="test"):
        report = post_rlhf._metadata(args, torch)
    assert "gpu" not in report["runtime"]
    assert "machine" not in report["runtime"]
    assert report["data_dir"] == "data"
    assert report["base_model"] == "public/model"


def test_finite_rlhf_logs_reject_nested_nonfinite_metrics():
    with pytest.raises(RuntimeError):
        post_rlhf.validate_finite_logs({"nested": {"loss": float("nan")}})


def test_dpo_training_dataset_checks_checkpoint_context_before_trainer():
    row = {"id": "pair", "prompt": "question", "chosen": "3", "rejected": "4"}
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    args = SimpleNamespace(max_prompt_length=128, max_completion_length=32, max_length=256)
    datasets = SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows))
    with patch.dict(sys.modules, {"datasets": datasets}):
        with pytest.raises(ValueError, match="context"):
            post_dpo.preference_dataset([row], tokenizer, args, context_limit=8)


@pytest.mark.parametrize("stage", ["reward", "ppo"])
def test_rlhf_dry_run_checks_data_without_loading_prerequisite_models(stage):
    rows = {split: [{}] * 16 for split in ("train", "validation", "test")}
    args = ["--stage", stage, "--dry-run"]
    if stage == "ppo":
        args += ["--reward-model", "outputs/not-yet-trained-reward"]
    with patch("finetune_lab.post_data.read_preferences", return_value=rows), patch.object(
        post_rlhf, "_model_path"
    ) as model, patch.object(post_rlhf, "require_reward_checkpoint") as reward, patch.object(
        post_rlhf, "_spark_runtime"
    ) as runtime:
        post_rlhf.main(args)
    model.assert_not_called()
    reward.assert_not_called()
    runtime.assert_not_called()
