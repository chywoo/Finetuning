"""Offline masking/configuration tests and an optional PyTorch gradient test."""

import unittest
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from finetune_lab.text_encoding import CausalLMCollator, encode_record, format_prompt


class CharacterTokenizer:
    eos_token_id = 7
    pad_token_id = 7

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(character) + 10 for character in text]}


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_prompt_format_is_shared_and_explicit(self):
        self.assertEqual(format_prompt("안녕"), "### Instruction:\n안녕\n\n### Response:\n")

    def test_sft_masks_only_prompt_and_keeps_response_eos(self):
        record = {"prompt": "hello", "completion": "OK"}
        encoded = encode_record(record, self.tokenizer, 128, "sft")
        prefix_length = len(format_prompt("hello"))
        self.assertEqual(encoded["labels"][:prefix_length], [-100] * prefix_length)
        self.assertEqual(encoded["labels"][prefix_length:], [ord("O") + 10, ord("K") + 10, 7])
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))
        self.assertEqual(record, {"prompt": "hello", "completion": "OK"})

    def test_truncation_keeps_causal_context_response_and_eos(self):
        encoded = encode_record({"prompt": "x" * 200, "completion": "abcdef"}, self.tokenizer, 4, "sft")
        self.assertEqual(len(encoded["input_ids"]), 4)
        self.assertEqual(encoded["labels"], [-100, ord("a") + 10, ord("b") + 10, 7])
        self.assertEqual(encoded["input_ids"][-1], 7)

    def test_minimum_sequence_still_keeps_response_and_eos_targets(self):
        encoded = encode_record({"prompt": "x", "completion": "abc"}, self.tokenizer, 3, "sft")
        self.assertEqual(encoded["labels"], [-100, ord("a") + 10, 7])

    def test_cpt_truncation_preserves_eos_and_no_prompt_mask(self):
        encoded = encode_record({"text": "abcdef"}, self.tokenizer, 3, "cpt")
        self.assertEqual(encoded["input_ids"], [ord("a") + 10, ord("b") + 10, 7])
        self.assertEqual(encoded["labels"], encoded["input_ids"])
        self.assertIsNot(encoded["labels"], encoded["input_ids"])

    def test_collator_does_not_mask_real_eos_when_eos_is_pad(self):
        short = {"input_ids": [11, 7], "attention_mask": [1, 1], "labels": [-100, 7]}
        long = {"input_ids": [12, 13, 7], "attention_mask": [1, 1, 1], "labels": [-100, 13, 7]}
        result = CausalLMCollator(7, return_tensors=None)([short, long])
        self.assertEqual(result["input_ids"][0], [11, 7, 7])
        self.assertEqual(result["labels"][0], [-100, 7, -100])
        self.assertEqual(result["attention_mask"][0], [1, 1, 0])
        self.assertEqual(short["input_ids"], [11, 7])

    def test_invalid_encoding_arguments_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "max_length"):
            encode_record({"text": "ok"}, self.tokenizer, 2, "cpt")
        with self.assertRaisesRegex(ValueError, "completion"):
            encode_record({"prompt": "ok", "completion": ""}, self.tokenizer, 20, "sft")
        with self.assertRaisesRegex(ValueError, "kind"):
            encode_record({"text": "ok"}, self.tokenizer, 20, "unknown")

    def test_invalid_collator_input_fails_before_tensor_creation(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            CausalLMCollator(7, return_tensors=None)([])
        with self.assertRaisesRegex(ValueError, "length"):
            CausalLMCollator(7, return_tensors=None)([
                {"input_ids": [1, 2], "attention_mask": [1], "labels": [-100, 2]}
            ])


class AccumulationTests(unittest.TestCase):
    def test_short_final_window_uses_token_weighted_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional for dependency-light tests")
        from finetune_lab.torch_text import train_loop

        class TwoClassModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids, attention_mask, labels):
                targets = labels[:, 1:].reshape(-1)
                logits = torch.stack([self.scalar, -self.scalar]).expand(targets.numel(), 2)
                loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
                return Namespace(loss=loss)

        model = TwoClassModel()
        # Equal microbatch weighting would cancel gradients at scalar=0;
        # the 1:3 target-token ratio must move the scalar toward class 1.
        collator = CausalLMCollator(7)
        first = collator([{"input_ids": [3, 0], "attention_mask": [1, 1], "labels": [-100, 0]}])
        second = collator([{"input_ids": [3, 1, 1, 1], "attention_mask": [1] * 4, "labels": [-100, 1, 1, 1]}])
        args = Namespace(learning_rate=0.1, weight_decay=0.0, max_steps=1,
                         gradient_accumulation=4, max_grad_norm=1.0)
        metrics = train_loop(model, [first, second], args, "cpu")
        self.assertLess(model.scalar.item(), -0.09)
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertEqual(metrics["train_target_tokens_seen"], 4)


class ComputeDtypeTests(unittest.TestCase):
    """Pure configuration tests; these do not initialize a CUDA device."""

    def test_cuda_auto_selects_supported_bfloat16(self):
        from finetune_lab.torch_text import select_compute_dtype

        torch_stub = Namespace(cuda=Namespace(is_bf16_supported=lambda: True))
        self.assertEqual(select_compute_dtype(torch_stub, "cuda", "auto"), "bfloat16")

    def test_auto_falls_back_to_float32_without_supported_cuda(self):
        from finetune_lab.torch_text import select_compute_dtype

        torch_stub = Namespace(cuda=Namespace(is_bf16_supported=lambda: False))
        self.assertEqual(select_compute_dtype(torch_stub, "cuda", "auto"), "float32")
        self.assertEqual(select_compute_dtype(torch_stub, "cpu", "auto"), "float32")

    def test_explicit_bfloat16_rejects_unavailable_acceleration(self):
        from finetune_lab.torch_text import select_compute_dtype

        torch_stub = Namespace(cuda=Namespace(is_bf16_supported=lambda: False))
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            select_compute_dtype(torch_stub, "cuda", "bfloat16")
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            select_compute_dtype(torch_stub, "cpu", "bfloat16")

    def test_explicit_float32_does_not_require_cuda_capability(self):
        from finetune_lab.torch_text import select_compute_dtype

        self.assertEqual(select_compute_dtype(Namespace(), "cuda", "float32"), "float32")


class TrainingSplitTests(unittest.TestCase):
    """Split validation uses only stdlib and rejects leakage before model loading."""

    @staticmethod
    def write_split(directory, name, row):
        (directory / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def prepare_splits(self, directory):
        for name in ("train", "validation", "test"):
            self.write_split(directory, name, {"id": name, "prompt": f"Question for {name}",
                                                "completion": f"Answer for {name}"})

    def test_returns_training_validation_and_only_test_count(self):
        from finetune_lab.torch_text import load_training_data

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.prepare_splits(directory)
            train, validation, test_count = load_training_data(directory, "sft")
            self.assertEqual([row["id"] for row in train], ["train"])
            self.assertEqual([row["id"] for row in validation], ["validation"])
            self.assertEqual(test_count, 1)
            self.assertNotIn("test", [row["id"] for row in train + validation])

    def test_rejects_normalized_content_overlap_with_test(self):
        from finetune_lab.torch_text import load_training_data

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.prepare_splits(directory)
            self.write_split(directory, "test", {"id": "test", "prompt": " QUESTION   FOR TRAIN ",
                                                 "completion": "Different answer does not prevent leakage"})
            with self.assertRaisesRegex(ValueError, "content overlap"):
                load_training_data(directory, "sft")

    def test_rejects_cross_split_duplicate_ids(self):
        from finetune_lab.torch_text import load_training_data

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.prepare_splits(directory)
            self.write_split(directory, "test", {"id": "train", "prompt": "A separate question",
                                                 "completion": "answer"})
            with self.assertRaisesRegex(ValueError, "id overlap"):
                load_training_data(directory, "sft")

    def test_missing_test_split_is_not_silently_skipped(self):
        from finetune_lab.torch_text import load_training_data

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.prepare_splits(directory)
            (directory / "test.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                load_training_data(directory, "sft")


if __name__ == "__main__":
    unittest.main()
