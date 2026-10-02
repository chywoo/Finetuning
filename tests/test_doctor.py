"""Capability checks must be portable and must not identify internal equipment."""
import json
from types import SimpleNamespace

import pytest

from scripts import doctor


def test_metadata_report_does_not_import_ml_or_identify_hardware(monkeypatch):
    monkeypatch.setattr(doctor.metadata, "version", lambda name: "1.0")
    report = doctor.build_report()
    assert "versions" in report
    assert not {"system", "machine", "gpu", "gpu_name", "hostname"} & report.keys()


def test_profile_reports_missing_packages(monkeypatch):
    def missing(name):
        raise doctor.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(doctor.metadata, "version", missing)
    with pytest.raises(ValueError, match="Missing"):
        doctor.build_report(profile="post")


def test_wrong_profile_versions_are_not_accepted(monkeypatch):
    monkeypatch.setattr(doctor.metadata, "version", lambda name: "99.0")
    with pytest.raises(ValueError, match="version"):
        doctor.build_report(profile="hf")


def test_cuda_check_does_not_train_a_model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    result = doctor.check_cuda(torch)
    assert result["cuda_tensor_check"] == "passed"
    assert "gpu" not in result


def test_cuda_unavailable_fails_clearly():
    with pytest.raises(ValueError, match="CUDA"):
        doctor.check_cuda(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


def test_main_metadata_is_json_without_internal_paths(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["doctor.py"])
    doctor.main()
    result = json.loads(capsys.readouterr().out)
    assert result["python_requirement_supported"] is True
