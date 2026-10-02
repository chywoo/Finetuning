"""Unsloth CLI contracts, without importing CUDA or downloading weights."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from finetune_lab.unsloth_text import (
    build_dataset_rows, guarded_trainer_class, load_model, load_runtime,
    load_split_records, main, model_origin, parse_args, save_artifacts, train,
    training_config, validate_metrics,
)


class TinyTokenizer:
    eos_token_id = 7
    pad_token_id = 7

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) + 10 for character in text]}


class UnslothTextTests(unittest.TestCase):
    def test_instruction_uses_base_and_knowledge_uses_instruct_model(self):
        instruction = parse_args([], task="instruction")
        knowledge = parse_args([], task="knowledge")
        self.assertEqual(instruction.model, "HuggingFaceTB/SmolLM2-135M")
        self.assertEqual(instruction.stage, "sft")
        self.assertEqual(knowledge.model, "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertEqual(knowledge.stage, "cpt")
        self.assertTrue(instruction.load_in_4bit)
        self.assertFalse(parse_args(["--no-load-in-4bit"]).load_in_4bit)

    def test_nonpositive_and_nonfinite_settings_fail_before_loading(self):
        invalid = [
            ["--max-steps", "0"], ["--max-length", "1"],
            ["--batch-size", "0"], ["--gradient-accumulation", "-1"],
            ["--learning-rate", "0"], ["--learning-rate", str(math.inf)],
            ["--learning-rate", "nan"], ["--lora-r", "0"],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(arguments)

    def test_instruction_rejects_cpt_and_knowledge_sft_selects_sft_data(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--stage", "cpt"], task="instruction")
        args = parse_args(["--stage", "sft"], task="knowledge")
        self.assertEqual(args.data_dir.name, "knowledge_sft")

    def test_training_rows_preserve_completion_only_labels_and_source(self):
        original = [{"id": "1", "prompt": "p" * 200, "completion": "OK"}]
        rows = build_dataset_rows(original, TinyTokenizer(), 4, "sft")
        self.assertEqual(rows[0]["labels"], [-100, ord("O") + 10, ord("K") + 10, 7])
        self.assertEqual(original[0]["prompt"], "p" * 200)
        self.assertNotIn("id", rows[0])

    def test_cpt_training_rows_train_all_real_tokens(self):
        rows = build_dataset_rows([{"id": "1", "text": "abc"}], TinyTokenizer(), 4, "cpt")
        self.assertEqual(rows[0]["labels"], rows[0]["input_ids"])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_dataset_rows([], TinyTokenizer(), 4, "cpt")

    def test_adapter_origin_retains_original_base_and_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "org/original", "revision": "fixed-sha",
            }), encoding="utf-8")
            self.assertEqual(model_origin(str(path), "main"), ("org/original", "fixed-sha", True))
        self.assertEqual(model_origin("org/model", "revision-sha"), ("org/model", "revision-sha", False))

    def test_adapter_metadata_takes_precedence_and_malformed_config_is_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            adapter = path / "adapter_config.json"
            adapter.write_text(json.dumps({"base_model_name_or_path": "org/original"}), encoding="utf-8")
            (path / "training_metadata.json").write_text(json.dumps({
                "base_model": "org/original", "revision": "base-sha",
            }), encoding="utf-8")
            self.assertEqual(model_origin(str(path), "main"), ("org/original", "base-sha", True))
            adapter.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base_model_name_or_path"):
                model_origin(str(path), "main")

    def test_dry_run_validates_data_without_invoking_runtime_or_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            output = path / "new-output"
            records = [{"id": "x", "prompt": "p", "completion": "a"}]
            with patch("finetune_lab.unsloth_text.load_split_records", return_value=(records, records)), \
                    patch("finetune_lab.unsloth_text.train") as runtime, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                result = main(["--dry-run", "--output-dir", str(output)])
            self.assertEqual(result, 0)
            runtime.assert_not_called()
            self.assertFalse(output.exists())
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["train_records"], 1)
            self.assertEqual(plan["kind"], "sft")
            self.assertEqual(plan["prompt_format"], "instruction_response_v1")

    def test_bad_task_empty_model_and_negative_seed_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "task"):
            parse_args([], task="other")
        for argv in (["--model", " "], ["--revision", ""], ["--seed", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_split_loader_reads_test_for_leak_checks_but_returns_training_and_validation_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            row = {"id": "x", "prompt": "p", "completion": "a"}
            validation = {"id": "v", "prompt": "other prompt", "completion": "b"}
            test = {"id": "t", "prompt": "held-out prompt", "completion": "c"}
            for split, record in (("train", row), ("validation", validation), ("test", test)):
                (path / f"{split}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(load_split_records(path, "sft"), ([row], [validation]))
            with patch("finetune_lab.data.read_jsonl", return_value=[]):
                with self.assertRaisesRegex(ValueError, "contain records"):
                    load_split_records(path, "sft")

    def test_split_overlap_is_rejected_before_cuda_initialization(self):
        row = {"id": "x", "prompt": "p", "completion": "a"}
        with patch("finetune_lab.data.read_jsonl", return_value=[row]):
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split_records(Path("unneeded"), "sft")

    def test_test_only_content_overlap_is_rejected(self):
        rows = [
            [{"id": "train", "prompt": "a fact", "completion": "a"}],
            [{"id": "validation", "prompt": "another fact", "completion": "b"}],
            [{"id": "test", "prompt": "A FACT", "completion": "c"}],
        ]
        with patch("finetune_lab.data.read_jsonl", side_effect=rows):
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split_records(Path("unneeded"), "sft")

    def test_unsupported_os_stops_before_runtime_imports(self):
        with patch("finetune_lab.unsloth_text.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError, "Linux"):
                load_runtime()

    def test_config_preserves_pretokenized_labels_and_has_no_network_reporting(self):
        args = parse_args([])
        factory = MagicMock()
        training_config(args, factory, True)
        config = factory.call_args.kwargs
        self.assertEqual(config["dataset_kwargs"], {"skip_prepare_dataset": True})
        self.assertTrue(config["completion_only_loss"])
        self.assertEqual(config["report_to"], "none")
        self.assertTrue(config["bf16"])
        self.assertFalse(config["fp16"])
        self.assertFalse(config["packing"])
        training_config(parse_args([], task="knowledge"), factory, False)
        self.assertFalse(factory.call_args.kwargs["completion_only_loss"])

    def test_new_model_attaches_lora_but_existing_adapter_does_not(self):
        loader = MagicMock()
        model, tokenizer = MagicMock(), TinyTokenizer()
        tokenizer.padding_side = "left"
        loader.from_pretrained.return_value = (model, tokenizer)
        loader.get_peft_model.return_value = model
        args = parse_args([])
        actual_model, _ = load_model(args, loader)
        self.assertIs(actual_model, model)
        loader.get_peft_model.assert_called_once()
        self.assertFalse(loader.from_pretrained.call_args.kwargs["trust_remote_code"])
        self.assertFalse(model.config.use_cache)
        self.assertEqual(tokenizer.padding_side, "right")
        loader.get_peft_model.reset_mock()
        with patch("finetune_lab.unsloth_text.model_origin", return_value=("base", "sha", True)):
            load_model(args, loader)
        loader.get_peft_model.assert_not_called()

    def test_missing_tokenizer_eos_fails_before_training(self):
        loader = MagicMock()
        tokenizer = TinyTokenizer()
        tokenizer.eos_token_id = None
        loader.from_pretrained.return_value = (MagicMock(), tokenizer)
        with self.assertRaisesRegex(ValueError, "EOS"):
            load_model(parse_args([]), loader)

    def test_train_evaluates_before_after_and_saves_portable_adapter_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--output-dir", str(Path(temporary) / "run")])
            torch, loader, dataset, config_factory, trainer_factory = (MagicMock() for _ in range(5))
            model, tokenizer = MagicMock(), MagicMock(wraps=TinyTokenizer())
            model.config._commit_hash = "resolved-base-sha"
            tokenizer.eos_token_id = 7
            tokenizer.pad_token_id = 7
            tokenizer.save_pretrained = MagicMock()
            loader.from_pretrained.return_value = (model, tokenizer)
            loader.get_peft_model.return_value = model
            trainer = trainer_factory.return_value
            trainer.evaluate.side_effect = [{"eval_loss": 2.0}, {"eval_loss": 1.0}]
            trainer.train.return_value.metrics = {"train_loss": 1.2}

            class FakeTrainer:
                def __init__(self, *arguments, **kwargs):
                    self.delegate = trainer_factory(*arguments, **kwargs)

                def evaluate(self):
                    return self.delegate.evaluate()

                def train(self):
                    return self.delegate.train()

                def save_state(self):
                    return self.delegate.save_state()

            runtime = (torch, loader, lambda: True, dataset, config_factory, FakeTrainer)
            records = [{"id": "x", "prompt": "p", "completion": "a"}]
            with patch("finetune_lab.unsloth_text.load_runtime", return_value=runtime):
                metrics = train(args, records, records)
            self.assertEqual(metrics["validation_before"]["eval_loss"], 2.0)
            self.assertEqual(metrics["validation_after"]["eval_loss"], 1.0)
            model.save_pretrained.assert_called_once_with(str(args.output_dir), safe_serialization=True)
            metadata = json.loads((args.output_dir / "training_metadata.json").read_text())
            self.assertEqual(metadata["method"], "unsloth")
            self.assertEqual(metadata["base_model"], "HuggingFaceTB/SmolLM2-135M")
            self.assertEqual(metadata["revision"], "resolved-base-sha")
            with self.assertRaisesRegex(ValueError, "new --output-dir"):
                train(args, records, records)

    def test_main_runs_runtime_after_data_validation(self):
        records = [{"id": "x", "prompt": "p", "completion": "a"}]
        with patch("finetune_lab.unsloth_text.load_split_records", return_value=(records, records)), \
                patch("finetune_lab.unsloth_text.train", return_value={"train_loss": 1.0}) as runtime, \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main([]), 0)
        runtime.assert_called_once()
        self.assertEqual(json.loads(stdout.getvalue()), {"train_loss": 1.0})

    def test_saved_adapter_revision_and_rank_describe_the_reloaded_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(["--output-dir", temporary])
            model, tokenizer, trainer = MagicMock(), MagicMock(), MagicMock()
            model.config._commit_hash = "resolved-sha"
            config = {"base_model_name_or_path": "org/base", "r": 8, "lora_alpha": 32}
            (args.output_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
            save_artifacts(args, model, tokenizer, trainer, {"eval_loss": 1.0})
            saved = json.loads((args.output_dir / "adapter_config.json").read_text())
            metadata = json.loads((args.output_dir / "training_metadata.json").read_text())
            self.assertEqual(saved["revision"], "resolved-sha")
            self.assertEqual(saved["base_model_name_or_path"], "org/base")
            self.assertEqual(metadata["lora_r"], 8)
            self.assertEqual(metadata["lora_alpha"], 32)
            self.assertNotIn("revision", config)

    def test_guard_rejects_nonfinite_model_loss_and_preserves_valid_loss_result(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

        class FakeFiniteResult:
            def __init__(self, value):
                self.value = value

            def all(self):
                return self

            def item(self):
                return self.value

        class FakeTorch:
            @staticmethod
            def isfinite(tensor):
                return FakeFiniteResult(math.isfinite(tensor.value))

        class BaseTrainer:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                loss = FakeTensor(inputs["loss"])
                return (loss, {"unchanged": True}) if return_outputs else loss

        trainer = guarded_trainer_class(BaseTrainer, FakeTorch)()
        result = trainer.compute_loss(None, {"loss": 1.5}, return_outputs=True, num_items_in_batch=4)
        self.assertEqual(result[0].value, 1.5)
        self.assertEqual(result[1], {"unchanged": True})
        self.assertEqual(trainer.compute_loss(None, {"loss": 2.0}).value, 2.0)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(FloatingPointError, "model loss"):
                trainer.compute_loss(None, {"loss": value})

    def test_nonfinite_metrics_reject_checkpoint_save_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "untouched"
            args = parse_args(["--output-dir", str(output)])
            model, tokenizer, trainer = MagicMock(), MagicMock(), MagicMock()
            with self.assertRaisesRegex(FloatingPointError, "validation_after.eval_loss"):
                save_artifacts(args, model, tokenizer, trainer, {"validation_after": {"eval_loss": math.nan}})
            self.assertFalse(output.exists())
            model.save_pretrained.assert_not_called()
        validate_metrics({"eval_loss": 1.0, "records": 8}, "validation_before")
        with self.assertRaisesRegex(FloatingPointError, "validation_before.eval_loss"):
            validate_metrics({"eval_loss": math.inf}, "validation_before")


if __name__ == "__main__":
    unittest.main()
