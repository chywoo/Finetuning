import json
from pathlib import Path

import pytest

from finetune_lab.data import read_jsonl, split_records, validate_splits, write_jsonl


def test_round_trip_does_not_mutate(tmp_path):
    records = [{"id": "a", "prompt": "한국어 질문", "completion": "답변"}]
    path = tmp_path / "train.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path, "sft") == records
    assert records == [{"id": "a", "prompt": "한국어 질문", "completion": "답변"}]


@pytest.mark.parametrize("row,kind", [
    ({"id": "a", "prompt": "", "completion": "ok"}, "sft"),
    ({"id": "a", "text": "  "}, "cpt"),
    ({"id": "a", "image": "x.png", "label": "unknown"}, "vision"),
    ({"id": 1, "text": "ok"}, "cpt"),
])
def test_reject_invalid_rows(tmp_path, row, kind):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        read_jsonl(path, kind)


def test_duplicate_id_and_empty_file_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"id": "a", "text": "one"}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_jsonl(path, "cpt")
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_jsonl(path, "cpt")


def test_split_deterministic_disjoint_and_no_input_mutation():
    rows = [{"id": str(i), "text": str(i)} for i in range(20)]
    before = list(rows)
    first = split_records(rows, 12, 4, seed=42)
    assert first == split_records(rows, 12, 4, seed=42)
    assert rows == before
    assert [len(first[x]) for x in ("train", "validation", "test")] == [12, 4, 4]
    validate_splits(first, "cpt")


def test_split_leakage_normalizes_whitespace():
    splits = {"train": [{"id": "a", "text": "same  content"}],
              "validation": [{"id": "b", "text": " SAME content "}], "test": []}
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(splits, "cpt")


def test_split_requires_enough_examples():
    with pytest.raises(ValueError):
        split_records([], 1, 1)


def test_missing_and_malformed_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "missing.jsonl", "cpt")
    path = tmp_path / "broken.jsonl"
    path.write_text("not-json\n")
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path, "cpt")


def test_vision_schema_and_unknown_kind(tmp_path):
    path = tmp_path / "vision.jsonl"
    write_jsonl(path, [{"id": "v", "image": "images/x.jpg", "label": "healthy"}])
    assert read_jsonl(path, "vision")[0]["label"] == "healthy"
    with pytest.raises(ValueError, match="kind"):
        read_jsonl(path, "wrong")
