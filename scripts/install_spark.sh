#!/usr/bin/env bash
set -euo pipefail
SPARK_PROFILE="${1:-hf}"
case "$SPARK_PROFILE" in hf|unsloth|post) ;; *) echo "Usage: $0 hf|unsloth|post" >&2; exit 2 ;; esac
SPARK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPARK_REPO_ROOT"
python scripts/doctor.py --require-spark
SPARK_VENV="$SPARK_REPO_ROOT/.venv-spark-$SPARK_PROFILE"
python -m venv --system-site-packages "$SPARK_VENV"
"$SPARK_VENV/bin/python" - "$SPARK_VENV/protected-runtime.txt" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path

protected = []
for name in ("torch", "torchvision", "triton"):
    try:
        protected.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        if name == "torch":
            raise SystemExit("The NGC PyTorch runtime is required")
Path(sys.argv[1]).write_text("\n".join(protected) + "\n")
print("Protected runtime: " + ", ".join(protected))
PY
"$SPARK_VENV/bin/python" -m pip install \
  -c "$SPARK_VENV/protected-runtime.txt" -r "requirements/spark-$SPARK_PROFILE.txt"
"$SPARK_VENV/bin/python" -m pip check
mkdir -p outputs/environment
"$SPARK_VENV/bin/python" -m pip freeze > "outputs/environment/$SPARK_PROFILE-freeze.txt"
"$SPARK_VENV/bin/python" scripts/doctor.py --require-spark
printf 'Activate: source .venv-spark-%s/bin/activate\n' "$SPARK_PROFILE"
if [[ "$SPARK_PROFILE" == post ]]; then
  printf 'Then: follow 04_post_training/README.md (post-training validation is separate).\n'
else
  printf 'Then: python scripts/validate_spark.py --profile %s\n' "$SPARK_PROFILE"
fi
