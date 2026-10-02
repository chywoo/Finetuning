from argparse import Namespace

import pytest

from finetune_lab.data import read_jsonl
from finetune_lab.prepare_data import demo_text, independent_splits, prepare_task, unique_rows


def test_unique_rows_preserves_original_and_eval_priority():
    rows = [{"id": "a", "text": "Same"}, {"id": "b", "text": " same "}]
    assert unique_rows(rows, "cpt") == [rows[0]]
    assert len(rows) == 2
    splits = {"train": rows, "validation": [], "test": [{"id": "c", "text": "same"}]}
    assert independent_splits(splits, "cpt")["train"] == []


def test_demo_data_source_is_explicit_and_splits_independent():
    splits, kind = demo_text("knowledge")
    assert kind == "sft"
    assert len(splits["train"]) == 24
    assert set(r["id"] for r in splits["train"]).isdisjoint(r["id"] for r in splits["test"])


def test_demo_prepare_refuses_overwrite(tmp_path):
    args = Namespace(output_root=tmp_path, source="demo", seed=42,
                     train_samples=256, eval_samples=32, overwrite=False)
    prepare_task("knowledge", args)
    assert len(read_jsonl(tmp_path / "knowledge_cpt/train.jsonl", "cpt")) == 24
    with pytest.raises(FileExistsError):
        prepare_task("knowledge", args)
