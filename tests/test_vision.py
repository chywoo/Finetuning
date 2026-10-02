"""Offline checks for visual SFT boundaries and class evaluation."""

import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from finetune_lab.vision import (
    BEAN_LABELS,
    build_messages,
    classification_metrics,
    completion_labels,
    image_path,
    normalize_prediction,
    VisionCollator,
    load_rows,
    validate_vision_splits,
)


class VisionDataTests(unittest.TestCase):
    def test_messages_keep_image_input_and_exact_assistant_answer(self):
        image = object()
        messages = build_messages(image, "healthy")
        self.assertIs(messages[0]["content"][0]["image"], image)
        self.assertEqual(messages[1]["content"][0]["text"], "healthy")
        for label in BEAN_LABELS:
            self.assertIn(label, messages[0]["content"][1]["text"])

    def test_prompt_has_no_answer(self):
        self.assertEqual(len(build_messages(object())), 1)
        with self.assertRaises(ValueError):
            build_messages(object(), "unknown")

    def test_image_path_checks_existence_and_prevents_directory_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "leaf.png").write_bytes(b"placeholder")
            self.assertEqual(image_path(root, "leaf.png"), (root / "leaf.png").resolve())
            with self.assertRaises(ValueError):
                image_path(root, "../outside.png")
            with self.assertRaises(ValueError):
                image_path(root, str(root / "leaf.png"))
            with self.assertRaises(FileNotFoundError):
                image_path(root, "missing.png")

    def test_split_validation_rejects_image_copies_under_different_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.png").write_bytes(b"same image bytes")
            (root / "copy.png").write_bytes(b"same image bytes")
            splits = {"train": [{"id": "train", "image": "first.png", "label": "healthy"}],
                      "test": [{"id": "test", "image": "copy.png", "label": "healthy"}]}
            with self.assertRaisesRegex(ValueError, "image content overlap"):
                validate_vision_splits(root, splits)

    def test_split_validation_rejects_ids_and_invalid_declared_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"first image")
            (root / "b.png").write_bytes(b"second image")
            train = [{"id": "shared", "image": "a.png", "label": "healthy"}]
            test = [{"id": "shared", "image": "b.png", "label": "bean_rust"}]
            with self.assertRaisesRegex(ValueError, "id overlap"):
                validate_vision_splits(root, {"train": train, "test": test})
            for declared in ("bad", "z" * 64, 123):
                with self.assertRaisesRegex(ValueError, "image_sha256"):
                    validate_vision_splits(root, {"train": [{**train[0], "image_sha256": declared}]})

    def test_declared_pixel_hash_detects_same_image_with_different_file_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"first encoded bytes")
            (root / "b.png").write_bytes(b"second encoded bytes")
            splits = {"train": [{"id": "train", "image": "a.png", "label": "healthy",
                                  "image_sha256": "0" * 64}],
                      "test": [{"id": "test", "image": "b.png", "label": "healthy",
                                 "image_sha256": "0" * 64}]}
            with self.assertRaisesRegex(ValueError, "image pixel hash overlap"):
                validate_vision_splits(root, splits)

    def test_distinct_files_and_matching_declared_hash_are_valid(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "validation", "test"):
                (root / f"{name}.png").write_bytes(name.encode())
            splits = {name: [{"id": name, "image": f"{name}.png", "label": "healthy",
                              "image_sha256": hashlib.sha256(name.encode()).hexdigest()}]
                      for name in ("train", "validation", "test")}
            validate_vision_splits(root, splits)


class VisionMaskTests(unittest.TestCase):
    def test_expanded_image_tokens_and_prompt_are_not_supervised(self):
        full = [1, 8, 8, 8, 2, 41, 42, 3, 0]
        labels = completion_labels(full, [1, 8, 8, 8, 2], [1] * 8 + [0], {8})
        self.assertEqual(labels, [-100] * 5 + [41, 42, 3, -100])
        self.assertEqual(full, [1, 8, 8, 8, 2, 41, 42, 3, 0])

    def test_alignment_mismatch_and_empty_completion_fail_fast(self):
        with self.assertRaises(ValueError):
            completion_labels([1, 2, 3], [1, 9], [1, 1, 1])
        with self.assertRaises(ValueError):
            completion_labels([1, 2], [1, 2], [1, 1])
        with self.assertRaises(ValueError):
            completion_labels([1, 2, 3], [1], [1])

    def test_left_padding_does_not_shift_completion_boundary(self):
        labels = completion_labels([0, 0, 1, 2, 3], [1, 2], [0, 0, 1, 1, 1])
        self.assertEqual(labels, [-100, -100, -100, -100, 3])

    def test_collator_preserves_pixels_and_masks_a_padded_batch(self):
        class Row:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return list(self.values)

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                return "prompt" if len(messages) == 1 else messages[1]["content"][0]["text"]

            def __call__(self, text, images, **kwargs):
                if isinstance(text, list):
                    return {"input_ids": [Row([1, 8, 8, 2, 41, 3]), Row([1, 8, 8, 2, 42, 0])],
                            "attention_mask": [Row([1, 1, 1, 1, 1, 1]), Row([1, 1, 1, 1, 1, 0])],
                            "pixel_values": "images survive collation"}
                return {"input_ids": [Row([1, 8, 8, 2])]}

        fake_torch = SimpleNamespace(tensor=lambda labels, dtype: labels, long="long")
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
                "finetune_lab.vision._open_image", return_value=object()):
            result = VisionCollator(Processor(), ".", 8)([
                {"image": "a.png", "label": "healthy"},
                {"image": "b.png", "label": "bean_rust"},
            ])
        self.assertEqual(result["labels"], [[-100, -100, -100, -100, 41, 3],
                                             [-100, -100, -100, -100, 42, -100]])
        self.assertEqual(result["pixel_values"], "images survive collation")


class VisionMetricsTests(unittest.TestCase):
    def test_exact_labels_only_and_invalid_prediction_penalty(self):
        self.assertEqual(normalize_prediction(" Healthy\n"), "healthy")
        self.assertEqual(normalize_prediction("The leaf is healthy"), "invalid")
        self.assertEqual(normalize_prediction("healthy or bean_rust"), "invalid")
        result = classification_metrics(
            ["healthy", "bean_rust", "angular_leaf_spot"],
            ["healthy", "invalid", "bean_rust"],
        )
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["invalid_predictions"], 1)
        self.assertEqual(result["confusion_matrix"][1][3], 1)

    def test_metric_inputs_must_match_and_be_nonempty(self):
        for truth, guesses in [([], []), (["healthy"], []), (["wrong"], ["healthy"])]:
            with self.assertRaises(ValueError):
                classification_metrics(truth, guesses)


class VisionCLITests(unittest.TestCase):
    def test_wrapper_dry_run_works_outside_repository_and_creates_no_output(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation", "test"):
                (root / f"{split}.png").write_bytes(f"not decoded: {split}".encode())
                (root / f"{split}.jsonl").write_text(json.dumps(
                    {"id": split, "image": f"{split}.png", "label": "healthy"}) + "\n")
            self.assertEqual(load_rows(root, "train")[0]["label"], "healthy")
            for method in ("pytorch", "huggingface", "unsloth"):
                result = subprocess.run([
                    sys.executable, str(repository / "03_vision" / method / "train.py"),
                    "--data-dir", str(root), "--output-dir", str(root / "out"), "--dry-run",
                ], cwd=root, capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout)["method"], method)
            self.assertFalse((root / "out").exists())

    def test_nonpositive_and_nonfinite_cli_values_fail_before_download(self):
        repository = Path(__file__).resolve().parents[1]
        wrapper = repository / "03_vision/pytorch/train.py"
        for option, value in [("--max-steps", "0"), ("--learning-rate", "0"),
                              ("--learning-rate", "nan"), ("--batch-size", "-1")]:
            result = subprocess.run([sys.executable, str(wrapper), option, value],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be", result.stderr)

    def test_saved_output_and_existing_adapter_training_are_rejected_before_loading(self):
        repository = Path(__file__).resolve().parents[1]
        wrapper = repository / "03_vision/huggingface/train.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}")
            cases = [(["--output-dir", str(root)], "already contains"),
                     (["--model", str(root)], "existing adapter")]
            for arguments, message in cases:
                result = subprocess.run([sys.executable, str(wrapper), *arguments],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
