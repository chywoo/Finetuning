#!/usr/bin/env bash
set -euo pipefail
SPARK_PROFILE="${1:-hf}"
case "$SPARK_PROFILE" in hf|unsloth|post) ;; *) echo "Usage: $0 hf|unsloth|post" >&2; exit 2 ;; esac
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
  echo "Run this launcher on the DGX Spark Linux/aarch64 host." >&2
  exit 2
fi
SPARK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_IMAGE="${SPARK_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
mkdir -p "$SPARK_REPO_ROOT/.cache/huggingface" "$SPARK_REPO_ROOT/outputs"
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount "type=bind,source=$SPARK_REPO_ROOT,target=/workspace/finetune" \
  -w /workspace/finetune \
  -e HF_HOME=/workspace/finetune/.cache/huggingface \
  -e SPARK_PROFILE="$SPARK_PROFILE" \
  --entrypoint /bin/bash "$SPARK_IMAGE"
