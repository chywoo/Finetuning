"""Run a short CUDA train/save/reload matrix on DGX Spark, never on the editing Mac."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, arguments: list[str], output: Path, results: list[dict]) -> None:
    command = [sys.executable, *arguments]
    print(f"[{name}] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (output / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    results.append({"name": name, "command": command, "returncode": result.returncode,
                    "log": f"{name}.log"})
    (output / "results.json").write_text(json.dumps({"steps": results}, indent=2) + "\n")
    if result.returncode:
        print(result.stdout[-6000:], flush=True)
        raise SystemExit(f"{name} failed; see {output / (name + '.log')}")


def text_evaluation(model: Path, data: Path, output: Path, kind: str = "sft") -> list[str]:
    return ["-m", "finetune_lab.evaluate", "--model", str(model), "--data-dir", str(data),
            "--kind", kind, "--max-eval-samples", "2", "--max-length", "64",
            "--max-new-tokens", "8", "--device", "cuda", "--output", str(output)]


def hf_matrix(output: Path, data: Path, results: list[dict]) -> None:
    tiny = output / "tiny-base"
    run_step("torch-sft", ["01_instruction/pytorch/train.py", "--data-dir", str(data / "instruction"),
                          "--smoke-model", "--max-steps", "2", "--max-length", "64",
                          "--gradient-accumulation", "2", "--device", "cuda",
                          "--output-dir", str(tiny)], output, results)
    run_step("torch-reload", text_evaluation(tiny, data / "instruction", output / "torch-eval.json"), output, results)
    cpt_torch = output / "torch-cpt"
    run_step("torch-cpt", ["02_knowledge/pytorch/train.py", "--stage", "cpt", "--model", str(tiny),
                          "--data-dir", str(data / "knowledge_cpt"), "--max-steps", "2", "--max-length", "64",
                          "--gradient-accumulation", "2", "--device", "cuda", "--output-dir", str(cpt_torch)], output, results)
    run_step("torch-cpt-reload", text_evaluation(cpt_torch, data / "knowledge_cpt", output / "torch-cpt-eval.json", "cpt"), output, results)
    full_hf = output / "hf-full"
    run_step("hf-full", ["01_instruction/huggingface/train.py", "--data-dir", str(data / "instruction"),
                        "--smoke-model", "--method", "full", "--max-steps", "2", "--max-length", "64",
                        "--gradient-accumulation", "2", "--device", "cuda", "--output-dir", str(full_hf)], output, results)
    cpt_hf, sft_hf = output / "hf-cpt-lora", output / "hf-cpt-sft-lora"
    for stage, source, destination in (("cpt", full_hf, cpt_hf), ("sft", cpt_hf, sft_hf)):
        run_step(f"hf-{stage}-lora", ["02_knowledge/huggingface/train.py", "--stage", stage,
                                    "--method", "lora", "--model", str(source), "--data-dir", str(data / f"knowledge_{stage}"),
                                    "--max-steps", "2", "--max-length", "64", "--gradient-accumulation", "2",
                                    "--device", "cuda", "--output-dir", str(destination)], output, results)
    run_step("hf-adapter-reload", text_evaluation(sft_hf, data / "knowledge_sft", output / "hf-adapter-eval.json"), output, results)
    merged = output / "hf-merged"
    run_step("hf-merge", ["-m", "finetune_lab.merge", "--adapter", str(sft_hf), "--output-dir", str(merged)], output, results)
    run_step("hf-merged-reload", text_evaluation(merged, data / "knowledge_sft", output / "hf-merged-eval.json"), output, results)


def unsloth_matrix(output: Path, data: Path, results: list[dict]) -> None:
    instruction = output / "unsloth-instruction"
    run_step("unsloth-sft", ["01_instruction/unsloth/train.py", "--data-dir", str(data / "instruction"),
                            "--max-steps", "2", "--max-length", "64", "--gradient-accumulation", "2",
                            "--output-dir", str(instruction)], output, results)
    run_step("unsloth-reload", text_evaluation(instruction, data / "instruction", output / "unsloth-eval.json"), output, results)
    cpt, sft = output / "unsloth-cpt", output / "unsloth-cpt-sft"
    run_step("unsloth-cpt", ["02_knowledge/unsloth/train.py", "--stage", "cpt", "--data-dir", str(data / "knowledge_cpt"),
                            "--max-steps", "2", "--max-length", "64", "--gradient-accumulation", "2",
                            "--output-dir", str(cpt)], output, results)
    run_step("unsloth-continue", ["02_knowledge/unsloth/train.py", "--stage", "sft", "--model", str(cpt),
                                 "--data-dir", str(data / "knowledge_sft"), "--max-steps", "2", "--max-length", "64",
                                 "--gradient-accumulation", "2", "--output-dir", str(sft)], output, results)
    run_step("unsloth-continue-reload", text_evaluation(sft, data / "knowledge_sft", output / "unsloth-continue-eval.json"), output, results)


def vision_matrix(profile: str, output: Path, data: Path, results: list[dict]) -> None:
    methods = ("pytorch", "huggingface") if profile == "hf" else ("unsloth",)
    for method in methods:
        model = output / f"vision-{method}"
        wrapper = f"03_vision/{method}/train.py"
        run_step(f"vision-{method}", [wrapper, "--data-dir", str(data / "vision"), "--max-steps", "1",
                                     "--gradient-accumulation", "1", "--device", "cuda", "--output-dir", str(model)], output, results)
        run_step(f"vision-{method}-reload", [wrapper, "--evaluate", "--model", str(model), "--data-dir", str(data / "vision"),
                                            "--max-eval-samples", "2", "--device", "cuda",
                                            "--output-dir", str(output / f"vision-{method}-eval")], output, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["hf", "unsloth"], required=True)
    parser.add_argument("--include-vision", action="store_true", help="Download VLM weights and check image training too")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        parser.exit(2, "Execute on DGX Spark ARM64 Linux; local Mac environment checks are disabled.\n")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or ROOT / "outputs/spark_validation" / f"{stamp}-{args.profile}").resolve()
    if output.exists():
        parser.error("output-dir must not already exist")
    output.mkdir(parents=True)
    results = []
    data = output / "demo-data"
    run_step("cuda-runtime", ["scripts/doctor.py", "--require-spark"], output, results)
    run_step("prepare-demo", ["-m", "finetune_lab.prepare_data", "--source", "demo", "--task", "all",
                              "--output-root", str(data)], output, results)
    (hf_matrix if args.profile == "hf" else unsloth_matrix)(output, data, results)
    if args.include_vision:
        vision_matrix(args.profile, output, data, results)
    (output / "results.json").write_text(json.dumps({"status": "passed", "profile": args.profile,
                                                    "fixture_warning": "Pipeline only; no quality claims",
                                                    "steps": results}, indent=2) + "\n")
    print(f"Spark pipeline checks passed. Reports: {output}")


if __name__ == "__main__":
    main()
