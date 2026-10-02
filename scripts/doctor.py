"""Report the DGX Spark runtime; CUDA checks run only on Spark."""
import argparse
import importlib.metadata as metadata
import json
import platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-spark", action="store_true")
    args = parser.parse_args()
    machine = platform.machine()
    if args.require_spark and (platform.system() != "Linux" or machine not in ("aarch64", "arm64")):
        parser.exit(2, "Run this CUDA environment check on DGX Spark Linux/ARM64.\n")
    versions = {}
    for name in ("torch", "torchvision", "triton", "transformers", "datasets", "peft", "trl",
                 "unsloth", "unsloth_zoo", "bitsandbytes", "numpy", "pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    report = {"python": platform.python_version(), "system": platform.system(),
              "machine": machine, "versions": versions}
    if args.require_spark:
        import torch
        if not torch.cuda.is_available():
            parser.exit(2, "CUDA unavailable: launch NGC with --gpus all and check the driver.\n")
        capability = torch.cuda.get_device_capability()
        if capability != (12, 1):
            parser.exit(2, f"Spark profile expects GB10 capability 12.1; got {capability}.\n")
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        loss = (x @ x.T).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
            parser.exit(2, "CUDA BF16 forward/backward returned non-finite values.\n")
        report = {**report, "gpu": torch.cuda.get_device_name(), "capability": capability,
                  "cuda_runtime": torch.version.cuda, "bf16": torch.cuda.is_bf16_supported(),
                  "compiled_architectures": torch.cuda.get_arch_list(),
                  "cuda_bf16_forward_backward": "passed"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
