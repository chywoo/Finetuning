"""Offline orchestration tests: fake model artifacts, never optimizer training."""
from argparse import Namespace
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from finetune_lab import post_data, post_dpo, post_grpo, post_rlhf


@pytest.fixture
def engine(tmp_path, monkeypatch):
    import torch

    class CPUOnlyTorch:
        def tensor(self, *args, **kwargs):
            kwargs.pop("device", None)
            return torch.tensor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(torch, name)

    cpu = CPUOnlyTorch()

    class Batch(dict):
        def to(self, device):
            return self

        def __getattr__(self, key):
            return self[key]

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0
        eos_token, pad_token = "EOS", "PAD"
        padding_side = "left"

        def __len__(self):
            return 256

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

        def __call__(self, text, **kwargs):
            ids = self.encode(text)
            return Batch(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)) if kwargs.get("return_tensors") else {"input_ids": ids}

        def get_vocab(self):
            return {chr(i): i for i in range(128)}

        def decode(self, ids, **kwargs):
            return "".join(chr(int(value)) for value in ids if int(value) not in (0, 2))

        def batch_decode(self, ids, **kwargs):
            return [self.decode(row, **kwargs) for row in ids]

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True, max_position_embeddings=2048,
                model_type="toy", hidden_size=8, num_hidden_layers=2, _commit_hash="fixed")
            self.score = object()
            self.weight = torch.tensor([1.0])
            self.training = False
            self.frozen = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def get_input_embeddings(self):
            return SimpleNamespace(num_embeddings=256)

        def resize_token_embeddings(self, size):
            self.resized = size

        def to(self, device):
            return self

        def eval(self):
            self.training = False
            return self

        def requires_grad_(self, enabled):
            self.frozen = not enabled
            return self

        def parameters(self):
            return [self.weight]

        def state_dict(self):
            return {"weight": self.weight}

        def load_state_dict(self, state):
            self.weight = state["weight"].clone()

        def disable_adapter(self):
            return nullcontext()

        def __call__(self, input_ids, **kwargs):
            size = input_ids.shape[1]
            # Causal logits for policy tests; scalar for RM tests.
            logits = torch.zeros(1, size, 256) if not getattr(self, "scalar", False) else torch.tensor([[0.3]])
            return SimpleNamespace(logits=logits)

        def generate(self, input_ids, **kwargs):
            count = kwargs.get("num_return_sequences", 1)
            return torch.cat([input_ids.repeat(count, 1), torch.full((count, 1), ord("3"))], dim=1)

        def save_pretrained(self, path, **kwargs):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_text("offline artifact fixture")

    class ScalarModel(Model):
        def __init__(self):
            super().__init__()
            self.scalar = True

    trainers = []

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.args = kwargs["args"]
            self.model = kwargs["model"]
            if "value_model" in kwargs:
                self.model = SimpleNamespace(policy=self.model, value_model=kwargs["value_model"])
            self.state = SimpleNamespace(global_step=1, log_history=[{"loss": 0.5}])
            self.accelerator = SimpleNamespace(device=SimpleNamespace(type="cuda"), unwrap_model=lambda model: model)
            trainers.append(self)

        def compute_loss(self, *args, return_outputs=False, **kwargs):
            result = torch.tensor(0.5)
            return (result, {}) if return_outputs else result

        def evaluate(self, metric_key_prefix):
            return {f"{metric_key_prefix}_loss": 0.5}

        def train(self):
            if isinstance(self.model, Model):
                self.compute_loss(self.model, {})
                self.compute_loss(self.model, {}, return_outputs=True)
            return SimpleNamespace(metrics={"train_loss": 0.5})

        def save_model(self, path):
            getattr(self.model, "policy", self.model).save_pretrained(path)

    monkeypatch.setitem(sys.modules, "torch", cpu)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig=lambda **kwargs: Namespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model,
        AutoModelForSequenceClassification=ScalarModel, set_seed=lambda seed: None,
        TrainerCallback=object,
    ))
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(
        **{name: Trainer for name in ("DPOTrainer", "GRPOTrainer", "RewardTrainer", "PPOTrainer")},
        **{name: lambda **kwargs: Namespace(**kwargs) for name in ("DPOConfig", "GRPOConfig", "RewardConfig", "PPOConfig")},
    ))
    for module in (post_data, post_dpo, post_grpo, post_rlhf):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for module in (post_dpo, post_grpo):
        monkeypatch.setattr(module, "require_cuda", lambda: cpu)
        monkeypatch.setattr(module, "check_trl_version", lambda: {"trl": "0.24.0"})
    monkeypatch.setattr(post_rlhf, "_spark_runtime", lambda: cpu)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.24.0")
    model = tmp_path / "model"
    Model().save_pretrained(model)
    for kind in ("preference", "math"):
        post_data.prepare_task(kind, tmp_path / "data")
    return Namespace(root=tmp_path, model=model, torch=cpu, Model=Model,
                     ScalarModel=ScalarModel, Tokenizer=Tokenizer, trainers=trainers)


def test_dpo_orchestration_preserves_reference_and_only_uses_validation(engine):
    output = engine.root / "dpo"
    post_dpo.main(argv=["--model", str(engine.model), "--data-dir", str(engine.root / "data/post_preferences"), "--output-dir", str(output)])
    trainer = engine.trainers[-1]
    assert trainer.kwargs["ref_model"] is None
    assert trainer.kwargs["peft_config"].lora_dropout == 0
    assert len(trainer.kwargs["train_dataset"]) == 16
    assert len(trainer.kwargs["eval_dataset"]) == 4
    assert trainer.args.reference_free is False
    assert (output / "training_metadata.json").is_file()
    with pytest.raises(ValueError, match="overwritten"):
        post_dpo.main(argv=["--model", str(engine.model), "--data-dir", str(engine.root / "data/post_preferences"), "--output-dir", str(output)])


def test_grpo_zero_variation_probe_stops_before_trainer_and_keeps_evidence(engine):
    output = engine.root / "grpo-no-signal"
    with pytest.raises(RuntimeError, match="identical rewards"):
        post_grpo.main(argv=["--model", str(engine.model), "--data-dir", str(engine.root / "data/post_math"), "--output-dir", str(output)])
    assert not engine.trainers
    assert (output / "training_signal_probe.json").is_file()
    assert not (output / "training_metadata.json").exists()


def test_grpo_orchestration_uses_train_probe_and_heldout_validation(engine, monkeypatch):
    seen = []
    def probe(model, tokenizer, rows, args):
        seen.append([row["id"] for row in rows])
        return {"exact_accuracy": 0.5, "sample_groups": {"frac_reward_zero_std": 0.5}}
    monkeypatch.setattr(post_grpo, "evaluate_rows", probe)
    output = engine.root / "grpo"
    data = engine.root / "data/post_math"
    post_grpo.main(argv=["--model", str(engine.model), "--data-dir", str(data), "--output-dir", str(output)])
    splits = post_data.read_math(data)
    assert seen == [[row["id"] for row in splits["train"][:8]], [row["id"] for row in splits["validation"]]]
    assert engine.trainers[-1].args.beta > 0
    assert engine.trainers[-1].kwargs["reward_funcs"] is post_data.arithmetic_reward
    assert (output / "validation.json").is_file()


def test_reward_and_ppo_orchestration_freezes_reference_and_saves_value(engine):
    reward, policy = engine.root / "reward", engine.root / "policy"
    common = ["--model", str(engine.model), "--data-dir", str(engine.root / "data/post_preferences")]
    post_rlhf.main(common + ["--stage", "reward", "--output-dir", str(reward)])
    metadata = post_rlhf.require_reward_checkpoint(reward)
    assert metadata["trained_steps"] == 1
    post_rlhf.main(common + ["--stage", "ppo", "--reward-model", str(reward), "--output-dir", str(policy)])
    trainer = engine.trainers[-1]
    assert trainer.kwargs["ref_model"].frozen
    assert trainer.kwargs["reward_model"].frozen
    assert not trainer.kwargs["value_model"].frozen
    assert trainer.kwargs["ref_model"] is not trainer.kwargs["model"]
    assert (policy / "value_model/config.json").is_file()
    assert json.loads((policy / "training_metadata.json").read_text())["algorithm"] == "trl_ppo"


def test_reward_and_policy_evaluation_reload_distinct_outputs(engine):
    reward = engine.root / "reward"
    data = engine.root / "data/post_preferences"
    common = ["--model", str(engine.model), "--data-dir", str(data)]
    post_rlhf.main(common + ["--stage", "reward", "--output-dir", str(reward)])
    for stage in ("evaluate-reward", "evaluate-policy"):
        output = engine.root / stage
        post_rlhf.main(common + ["--stage", stage, "--reward-model", str(reward), "--max-eval-samples", "2", "--output-dir", str(output)])
        report = json.loads((output / "evaluation.json").read_text())
        assert report["metrics"]["count"] == 2
        assert report["split"] == "test"
        assert report["reward_score_is_training_proxy"] is True


def test_dpo_and_grpo_evaluation_use_independent_reload(engine, monkeypatch):
    from finetune_lab import evaluate
    model, tokenizer = engine.Model(), engine.Tokenizer()
    monkeypatch.setattr(evaluate, "load_text_model", lambda *args: (model, tokenizer, {}))
    for module, name in ((post_dpo, "post_preferences"), (post_grpo, "post_math")):
        output = engine.root / f"{name}-evaluation.json"
        module.main(mode="evaluate", argv=["--model", str(engine.model), "--data-dir", str(engine.root / "data" / name), "--output", str(output)])
        report = json.loads(output.read_text())
        assert report["count"] in (4, 8)
        assert report["split"] == "test"
        assert len(report["examples"]) == report["count"]
