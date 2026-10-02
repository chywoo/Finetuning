"""RLHF contract tests plus an explicitly enabled CUDA training integration."""
import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from finetune_lab.post_rlhf import (
    parse_args, preference_token_ids, require_reward_checkpoint,
    summarize_preferences, validate_ppo_batch, validate_finite_logs, _fresh_output, main,
)


class ToyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) for character in text]}


class RLHFContractsTests(unittest.TestCase):
    def test_default_reward_learning_starts_from_shared_full_sft_checkpoint(self):
        args = parse_args(["--stage", "reward"])
        self.assertTrue(args.model.endswith("outputs/preference_sft_full"))
        self.assertEqual(args.device, "cuda")

    def test_ppo_requires_a_saved_trained_reward_model(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--stage", "ppo"])

    def test_invalid_numbers_fail_before_ml_imports(self):
        for flag, value in (("--max-steps", "0"), ("--learning-rate", "nan"),
                            ("--kl-coef", "-1"), ("--response-length", "0")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--stage", "reward", flag, value])

    def test_pair_encoding_shares_prefix_and_keeps_response_eos(self):
        row = {"id": "one", "prompt": "Explain.", "chosen": "Correct", "rejected": "Wrong"}
        encoded = preference_token_ids(ToyTokenizer(), row, 128)
        self.assertEqual(encoded["chosen_input_ids"][-1], 99)
        self.assertEqual(encoded["rejected_input_ids"][-1], 99)
        self.assertEqual(encoded["chosen_input_ids"][:-8], encoded["rejected_input_ids"][:-6])

    def test_overlong_pairs_fail_instead_of_hiding_empty_training_data(self):
        row = {"id": "long", "prompt": "Explain.", "chosen": "Correct", "rejected": "Wrong"}
        with self.assertRaisesRegex(ValueError, "max-length"):
            preference_token_ids(ToyTokenizer(), row, 4)

    def test_reward_checkpoint_must_have_training_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "config.json").write_text(json.dumps({"num_labels": 1}))
            (path / "model.safetensors").write_text("weight-placeholder")
            with self.assertRaisesRegex(ValueError, "RewardTrainer"):
                require_reward_checkpoint(path)
            (path / "training_metadata.json").write_text(json.dumps({
                "stage": "reward", "algorithm": "trl_reward_trainer", "trained_steps": 1
            }))
            self.assertEqual(require_reward_checkpoint(path)["stage"], "reward")

    def test_ppo_drop_last_cannot_make_empty_rollout_loader(self):
        with self.assertRaisesRegex(ValueError, "rollout"):
            validate_ppo_batch(train_count=3, batch_size=2, accumulation=2, total_episodes=8)
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_ppo_batch(train_count=8, batch_size=2, accumulation=2, total_episodes=7)
        validate_ppo_batch(train_count=8, batch_size=2, accumulation=2, total_episodes=8)

    def test_pair_metrics_count_ties_as_not_preferred_and_are_numerically_stable(self):
        metrics = summarize_preferences([1000.0, -1000.0, 0.0])
        self.assertAlmostEqual(metrics["preference_accuracy"], 1 / 3)
        self.assertTrue(math.isfinite(metrics["bradley_terry_loss"]))

    def test_nonfinite_ppo_logs_cannot_be_saved(self):
        validate_finite_logs({"loss/policy_avg": 0.3, "objective/kl": 0.02})
        for metric in ("loss/policy_avg", "loss/value_avg", "objective/scores"):
            with self.subTest(metric=metric), self.assertRaises(RuntimeError):
                validate_finite_logs({metric: float("nan")})

    def test_existing_results_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            original = output / "model.safetensors"
            original.write_text("keep existing model")
            with self.assertRaisesRegex(ValueError, "output-dir"):
                _fresh_output(output)
            self.assertEqual(original.read_text(), "keep existing model")

    def test_malformed_reward_metadata_is_a_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for text in ("[]", "{invalid json"):
                with self.subTest(text=text):
                    (directory / "training_metadata.json").write_text(text)
                    with self.assertRaises(ValueError):
                        require_reward_checkpoint(directory)


@unittest.skipUnless(os.environ.get("RUN_POST_TRAINING_TESTS") == "1" or os.environ.get("RUN_SPARK_POST_TESTS") == "1",
                     "Explicitly enable the CUDA training integration test")
class RewardPPOIntegrationTests(unittest.TestCase):
    def test_trained_reward_then_ppo_save_and_independent_reload_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        model = os.environ.get("POST_SFT_MODEL", os.environ.get("SPARK_POST_SFT_MODEL", str(root / "outputs/preference_sft_full")))
        data = root / "data/demo/post_preferences"
        common = ["--model", model, "--data-dir", str(data), "--max-length", "128",
                  "--max-prompt-length", "96", "--response-length", "8", "--seed", "42"]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            reward, policy = directory / "reward", directory / "policy"
            main(common + ["--stage", "reward", "--max-steps", "1", "--batch-size", "1",
                           "--gradient-accumulation", "1", "--output-dir", str(reward)])
            main(common + ["--stage", "evaluate-reward", "--reward-model", str(reward),
                           "--output-dir", str(directory / "reward-eval")])
            main(common + ["--stage", "ppo", "--reward-model", str(reward), "--total-episodes", "2",
                           "--batch-size", "2", "--gradient-accumulation", "1", "--ppo-epochs", "1",
                           "--output-dir", str(policy)])
            evaluation_args = ["--model", str(policy), "--stage", "evaluate-policy", "--reward-model", str(reward),
                               "--data-dir", str(data), "--max-length", "128", "--max-prompt-length", "96",
                               "--response-length", "8", "--max-eval-samples", "2",
                               "--output-dir", str(directory / "policy-eval")]
            main(evaluation_args)
            report = json.loads((directory / "policy-eval/evaluation.json").read_text())
            self.assertEqual(report["metrics"]["count"], 2)
            self.assertEqual(len(report["generations"]), 2)
            self.assertTrue(math.isfinite(report["metrics"]["mean_chosen_minus_rejected"]))
            self.assertTrue((policy / "value_model/config.json").is_file())


if __name__ == "__main__":
    unittest.main()
