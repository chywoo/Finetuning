"""Pure post-training data/reward tests; execute on DGX Spark, not the Mac host."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from finetune_lab.post_data import (
    arithmetic_reward, group_statistics, parse_integer_completion, read_math,
    read_preferences, require_fresh_output, validate_finite_values, validate_grpo_settings,
)


class PostTrainingTests(unittest.TestCase):
    def test_strict_integer_reward_rejects_extra_text_and_code(self):
        answers = ["3"] * 8
        completions = ["3", " 3\n", "The answer is 3", "3 or 4", "3\n4", "03", "3.0", "__import__('os')"]
        self.assertEqual(arithmetic_reward(completions, answers), [1., 1., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(parse_integer_completion("-12"), -12)
        self.assertIsNone(parse_integer_completion("9" * 100))

    def test_wrong_integer_gets_no_reward(self):
        self.assertEqual(arithmetic_reward(["4", "-2"], ["3", "-2"]), [0., 1.])
        with self.assertRaises(ValueError):
            arithmetic_reward(["3"], [])

    def test_reward_groups_with_no_variation_cannot_rank_outputs(self):
        result = group_statistics([0., 0., 0., 0., 1., 0., 1., 0.], 4)
        self.assertEqual(result["frac_reward_zero_std"], 0.5)
        self.assertEqual(result["mean_reward"], 0.25)
        self.assertEqual(result["groups"], 2)

    def test_group_and_beta_settings_reject_invalid_grpo_experiments(self):
        validate_grpo_settings(4, 4, 2, 0.04)
        for settings in ((1, 1, 1, .04), (3, 4, 1, .04), (4, 4, 1, 0.), (5, 4, 1, .04)):
            with self.assertRaises(ValueError):
                validate_grpo_settings(*settings)

    @staticmethod
    def write_splits(root, kind="preference"):
        for split in ("train", "validation", "test"):
            row = {"id": split, "prompt": f"Question {split}"}
            if kind == "preference":
                row = {**row, "chosen": "3", "rejected": "4"}
            else:
                row = {**row, "answer": "3", "completion": "3"}
            (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_preferences_read_three_disjoint_splits(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_splits(root)
            self.assertEqual(set(read_preferences(root)), {"train", "validation", "test"})

    def test_test_prompt_leak_is_detected_after_normalization(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_splits(root)
            (root / "test.jsonl").write_text(json.dumps({"id": "test", "prompt": " QUESTION  TRAIN ",
                                                       "chosen": "3", "rejected": "4"}) + "\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                read_preferences(root)

    def test_identical_preference_pair_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_splits(root)
            (root / "test.jsonl").write_text(json.dumps({"id": "test", "prompt": "Question test",
                                                       "chosen": "3", "rejected": "3"}) + "\n")
            with self.assertRaisesRegex(ValueError, "different"):
                read_preferences(root)

    def test_math_labels_are_canonical_integers_and_match_sft_completion(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_splits(root, "math")
            self.assertEqual(read_math(root)["train"][0]["answer"], "3")
            (root / "test.jsonl").write_text(json.dumps({"id": "test", "prompt": "Question test",
                                                       "answer": "3", "completion": "4"}) + "\n")
            with self.assertRaisesRegex(ValueError, "completion"):
                read_math(root)

    def test_partial_output_directory_is_protected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            require_fresh_output(root)
            (root / "checkpoint-1").mkdir()
            with self.assertRaisesRegex(ValueError, "output"):
                require_fresh_output(root)

    def test_nested_nonfinite_metric_is_rejected_before_checkpoint_save(self):
        validate_finite_values({"loss": 0.5, "history": [{"kl": 0.01}]})
        with self.assertRaises(RuntimeError):
            validate_finite_values({"history": [{"loss": float("nan")}]})

    def test_dpo_evaluation_rejects_partial_preference_answers(self):
        from argparse import Namespace
        from finetune_lab.post_dpo import validate_preference_lengths

        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text)

        row = {"id": "pair", "prompt": "A short question", "chosen": "3", "rejected": "4" * 100}
        args = Namespace(max_prompt_length=128, max_completion_length=16, max_length=256)
        with self.assertRaisesRegex(ValueError, "answer"):
            validate_preference_lengths(row, CharacterTokenizer(), args)


if __name__ == "__main__":
    unittest.main()
