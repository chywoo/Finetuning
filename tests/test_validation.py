"""Data audit catches corruption, leakage and inconsistent fixture conversions."""
import hashlib
import json
from argparse import Namespace

import pytest

from finetune_lab.prepare_data import prepare_task
from finetune_lab.post_data import prepare_task as prepare_post
from finetune_lab.validation import validate_data_root, validate_manifest


def prepare(directory, task="instruction"):
    prepare_task(task, Namespace(output_root=directory, source="demo", seed=42,
                                train_samples=24, eval_samples=8, overwrite=False))
    return directory / task / "manifest.json"


def test_dataset_count_and_hash_are_checked(tmp_path):
    manifest = prepare(tmp_path)
    result = validate_manifest(manifest)
    assert result["jsonl_files"] == 3
    assert result["counts"] == {"train": 24, "validation": 8, "test": 8}
    path = manifest.parent / "test.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest(manifest)


def test_manifest_count_mismatch_fails(tmp_path):
    manifest = prepare(tmp_path)
    info = json.loads(manifest.read_text())
    info["counts"]["train"] += 1
    manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="count"):
        validate_manifest(manifest)


def test_missing_data_and_incomplete_project_are_not_passed(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        validate_data_root(tmp_path)
    prepare(tmp_path / "demo")
    with pytest.raises(ValueError, match="Missing"):
        validate_data_root(tmp_path)
    assert validate_data_root(tmp_path, require_complete=False)["manifests"] == 1


def test_decoded_images_and_pixel_hash_are_checked(tmp_path):
    manifest = prepare(tmp_path, "vision")
    result = validate_manifest(manifest)
    assert result["image_files"] == 24
    path = manifest.parent / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["image_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    info = json.loads(manifest.read_text())
    info["sha256"]["test"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="pixel"):
        validate_manifest(manifest)


def test_post_sft_must_match_original_answer(tmp_path):
    prepare_post("preference", tmp_path)
    manifest = tmp_path / "post_preferences/manifest.json"
    assert validate_manifest(manifest)["jsonl_files"] == 6
    path = manifest.parent / "sft/test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["completion"] = "wrong"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    info = json.loads(manifest.read_text())
    info["sft"]["sha256"]["test"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="SFT conversion"):
        validate_manifest(manifest)


def test_full_demo_math_audit(tmp_path):
    prepare_post("math", tmp_path)
    assert validate_manifest(tmp_path / "post_math/manifest.json")["counts"] == {
        "train": 32, "validation": 8, "test": 8}
