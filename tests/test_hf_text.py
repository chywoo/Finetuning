"""Dependency-light tests for the Hugging Face training entry points."""
import json
import math
import os
import contextlib
import io
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

from finetune_lab.hf_text import (
    check_finite_metrics,
    _device,
    _validate_output_dir,
    main,
    local_model_spec,
    parse_args,
)


class HuggingFaceConfigurationTests(unittest.TestCase):
    def test_instruction_starts_from_base_model(self):
        args = parse_args("instruction", [])
        self.assertEqual(args.model, "HuggingFaceTB/SmolLM2-135M")
        self.assertEqual(args.method, "lora")
        self.assertTrue(str(args.data_dir).endswith("data/processed/instruction"))

    def test_knowledge_cpt_and_sft_have_different_data_directories(self):
        cpt = parse_args("knowledge", ["--stage", "cpt"])
        sft = parse_args("knowledge", ["--stage", "sft"])
        self.assertEqual(cpt.model, "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertTrue(str(cpt.data_dir).endswith("knowledge_cpt"))
        self.assertTrue(str(sft.data_dir).endswith("knowledge_sft"))

    def test_rejects_invalid_training_ranges(self):
        for option in ("--max-steps", "--batch-size", "--max-length", "--gradient-accumulation"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args("instruction", [option, "0"])
        for value in ("0", "-0.001", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args("instruction", ["--learning-rate", value])

    def test_adapter_reload_preserves_pinned_base_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "org/base", "revision": "abc123"
            }))
            spec = local_model_spec(str(path), "main", "lora")
            self.assertEqual(spec.base_model, "org/base")
            self.assertEqual(spec.revision, "abc123")
            self.assertEqual(spec.adapter_path, str(path.resolve()))

    def test_adapter_cannot_be_accidentally_loaded_as_full_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "org/base"
            }))
            with self.assertRaisesRegex(ValueError, "full"):
                local_model_spec(str(path), "main", "full")

    def test_adapter_metadata_is_required_to_have_valid_base(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "base_model"):
                local_model_spec(str(path), "main", "lora")

    def test_adapter_training_metadata_revision_overrides_config(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "org/base", "revision": "main"
            }))
            (path / "training_metadata.json").write_text(json.dumps({"revision": "pinned-sha"}))
            self.assertEqual(local_model_spec(str(path), "main", "qlora").revision, "pinned-sha")

    def test_remote_model_keeps_explicit_revision(self):
        spec = local_model_spec("org/base", "pinned-sha", "full")
        self.assertEqual(spec.revision, "pinned-sha")
        self.assertIsNone(spec.adapter_path)

    def test_local_full_model_is_recorded_as_absolute_path(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            relative = str(Path(temp).relative_to(Path.cwd()))
            self.assertEqual(local_model_spec(relative, "main", "lora").base_model, temp)

    def test_rejects_non_reloadable_smoke_adapter_at_cli_boundary(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args("instruction", ["--smoke-model", "--method", "lora"])

    def test_existing_training_artifacts_cannot_be_overwritten(self):
        artifacts = ("config.json", "adapter_config.json", "training_metadata.json",
                     "model.safetensors", "pytorch_model.bin", "tokenizer_config.json")
        for artifact in artifacts:
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                existing = output / artifact
                existing.write_text("preserve existing training artifact")
                with self.assertRaisesRegex(ValueError, "output-dir"):
                    _validate_output_dir(output)
                self.assertEqual(existing.read_text(), "preserve existing training artifact")

    def test_new_output_path_is_validated_without_creating_it(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new-run"
            _validate_output_dir(output)
            self.assertFalse(output.exists())

    def test_empty_directory_can_be_used_for_training(self):
        with tempfile.TemporaryDirectory() as temp:
            _validate_output_dir(Path(temp))

    def test_checkpoint_directory_cannot_be_reused_without_resume_support(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "checkpoint-10").mkdir()
            with self.assertRaisesRegex(ValueError, "output-dir"):
                _validate_output_dir(output)

    def test_output_path_must_be_a_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing-file"
            output.write_text("keep me")
            with self.assertRaisesRegex(ValueError, "directory"):
                _validate_output_dir(output)

    def test_explicit_unavailable_device_is_rejected(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        self.assertEqual(_device(torch, "auto"), "cpu")
        with self.assertRaisesRegex(ValueError, "device=cuda"):
            _device(torch, "cuda")

    def test_dry_run_from_another_directory_needs_no_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            data.mkdir()
            for split in ("train", "validation", "test"):
                (data / f"{split}.jsonl").write_text(json.dumps({
                    "id": split, "prompt": f"Question {split}", "completion": "Answer"
                }) + "\n")
            result = subprocess.run([sys.executable, "-S", str(root / "01_instruction/huggingface/train.py"),
                                     "--data-dir", str(data), "--dry-run"],
                                    cwd=temp, text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["status"], "data_validated")

    def test_nonfinite_loss_cannot_be_silently_saved(self):
        check_finite_metrics({"eval_loss": 1.5, "eval_runtime": 2.0})
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "finite"):
                check_finite_metrics({"eval_loss": value})


@unittest.skipUnless(os.environ.get("RUN_ML_TESTS") == "1", "Set RUN_ML_TESTS=1 for the offline Trainer integration test")
class HuggingFaceTrainerIntegrationTests(unittest.TestCase):
    def test_full_save_then_cpt_adapter_then_sft_adapter_reload(self):
        """Exercise actual optimizer updates, adapter continuation, and inference reload."""
        import torch
        from finetune_lab.evaluate import load_text_model

        device = "cuda"
        if not torch.cuda.is_available():
            self.fail("Spark integration requires CUDA; run it on the DGX Spark rather than this host.")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sft_dir, cpt_dir = root / "sft", root / "cpt"
            sft_dir.mkdir()
            cpt_dir.mkdir()
            for split in ("train", "validation", "test"):
                (sft_dir / f"{split}.jsonl").write_text(json.dumps({
                    "id": split, "prompt": f"Say hello {split}", "completion": "hello world"
                }) + "\n")
                (cpt_dir / f"{split}.jsonl").write_text(json.dumps({
                    "id": split, "text": f"The world is blue {split}"
                }) + "\n")
            common = ["--device", device, "--max-steps", "1", "--max-length", "32",
                      "--gradient-accumulation", "1"]
            full, cpt, sft = root / "full", root / "adapter-cpt", root / "adapter-sft"
            main("instruction", common + ["--smoke-model", "--method", "full", "--data-dir", str(sft_dir),
                                           "--output-dir", str(full)])
            main("knowledge", common + ["--model", str(full), "--method", "lora", "--stage", "cpt",
                                        "--data-dir", str(cpt_dir), "--output-dir", str(cpt)])
            main("knowledge", common + ["--model", str(cpt), "--method", "lora", "--stage", "sft",
                                        "--data-dir", str(sft_dir), "--output-dir", str(sft)])
            metadata = json.loads((sft / "training_metadata.json").read_text())
            self.assertEqual(metadata["kind"], "sft")
            self.assertEqual(metadata["base_model"], str(full.resolve()))
            self.assertIsNotNone(metadata["source_adapter"])
            check_finite_metrics(metadata["metrics"])
            model, tokenizer, _ = load_text_model(str(sft), "main", device)
            inputs = tokenizer("hello", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=2, do_sample=False)
            self.assertEqual(outputs.shape[0], 1)
            self.assertGreater(outputs.shape[1], inputs["input_ids"].shape[1])


if __name__ == "__main__":
    unittest.main()
