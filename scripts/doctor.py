"""Check lesson dependencies and small tensor operations without model training."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

HF_VERSIONS = {"transformers": "4.57.6", "datasets": "4.3.0", "peft": "0.18.1",
               "accelerate": "1.12.0", "huggingface-hub": "0.36.2", "pillow": "11.3.0"}


def check_cuda(torch) -> dict:
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; check the installed runtime and device access")
    x = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
    result = x @ x.T
    torch.cuda.synchronize()
    expected = torch.tensor([[5., 11.], [11., 25.]])
    if not torch.equal(result.cpu(), expected):
        raise ValueError("CUDA tensor operation returned an incorrect result")
    return {"cuda_tensor_check": "passed", "bf16_supported": torch.cuda.is_bf16_supported()}


def build_report(profile: str | None = None, require_cuda: bool = False,
                 require_spark: bool = False) -> dict:
    versions = {}
    for name in ("torch", "torchvision", "triton", *HF_VERSIONS, "trl", "unsloth",
                 "unsloth_zoo", "bitsandbytes", "numpy", "pytest", "pytest-cov"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    supported = (3, 11) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 13)
    report = {"python": platform.python_version(), "python_requirement_supported": supported,
              "versions": versions}
    if profile:
        if not supported:
            raise ValueError("Use Python >=3.11,<3.13 for these lessons")
        required = {**HF_VERSIONS, "torch": None, "pytest": None, "pytest-cov": None}
        if profile in ("post", "unsloth"):
            required["trl"] = "0.24.0"
        if profile == "unsloth":
            required.update({"unsloth": "2026.9.14", "unsloth_zoo": "2026.9.9",
                             "bitsandbytes": "0.48.2"})
        missing = [name for name in required if versions[name] is None]
        if missing:
            raise ValueError("Missing lesson packages: " + ", ".join(missing))
        mismatch = [name for name, expected in required.items()
                    if expected is not None and versions[name] != expected]
        if mismatch:
            raise ValueError("Lesson version mismatch: " + ", ".join(mismatch))
        # Unsloth patches Transformers, so it must be imported first in its own process.
        modules = ["unsloth"] if profile == "unsloth" else []
        modules += ["torch", "transformers", "datasets", "peft", "accelerate", "PIL"]
        if profile in ("post", "unsloth"):
            modules.append("trl")
        for name in modules:
            importlib.import_module(name)
        report["profile_imports"] = "passed"
        report["profile"] = profile
    if require_cuda or require_spark:
        import torch
        report.update(check_cuda(torch))
        if require_spark:
            if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
                raise ValueError("Spark profile requires Linux ARM64")
            if torch.cuda.get_device_capability() != (12, 1):
                raise ValueError("Spark profile requires its supported CUDA capability")
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Spark profile requires BF16 support")
            x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            loss = (x @ x.T).float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
                raise ValueError("CUDA BF16 forward/backward returned non-finite values")
            report["cuda_bf16_forward_backward"] = "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-spark", action="store_true", help="Optional Spark-specific capability check")
    parser.add_argument("--require-cuda", action="store_true", help="Small CUDA tensor check; no model training")
    parser.add_argument("--profile", choices=("hf", "post", "unsloth"))
    parser.add_argument("--output", type=Path, help="Save a new report without overwriting")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output report already exists; choose a new path")
    try:
        report = build_report(args.profile, args.require_cuda, args.require_spark)
    except (ValueError, ImportError, RuntimeError) as error:
        # Avoid traceback paths or environment/credential details in diagnostics.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        parser.exit(2, message + "\n")
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
