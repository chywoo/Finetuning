# DGX Spark에서 첫 실습

처음이라면 [고등학생 입문 수업](12_foundations.md)에서 CLI·JSONL·토큰·loss를 배운 뒤 진행합니다. 다음 수업으로 이동하는 기준과 post-training 경로는 [단계별 실습 계획](11_curriculum.md)에 있습니다.

실제 학습은 **DGX Spark의 ARM64 Linux/CUDA**에서 진행합니다. Intel Mac에서 환경 검증을 계속하지 않습니다. 작은 모델을 쓰는 이유는 학습 루프와 전후 결과를 빨리 이해하기 위해서입니다. Spark에서 큰 모델로 확장하는 명령은 [Spark 환경 안내](04_dgx_spark.md)에 있습니다.

## 1. 폴더를 Spark로 옮기기

Spark의 `~/finetune` 같은 위치에 이 저장소를 복사합니다. `.venv`, `.venv-spark-*`, `__pycache__`, Mac에서 만든 `outputs`는 환경 간 공유하지 않습니다. 코드, docs, requirements, data/demo와 이미 준비된 data/processed는 옮길 수 있습니다. 데이터 PNG/JSONL은 CPU 아키텍처와 무관합니다.

## 2. Spark 호스트에서 컨테이너 시작

```bash
cd ~/finetune
bash scripts/spark_container.sh hf
```

컨테이너 안의 작업 경로는 `/workspace/finetune`입니다. NVIDIA NGC가 제공하는 CUDA PyTorch를 사용합니다.

```bash
bash scripts/install_spark.sh hf
source .venv-spark-hf/bin/activate
python scripts/doctor.py --require-spark
```

overlay 설치는 NGC의 torch/torchvision/Triton 버전을 제약으로 보존합니다. ARM64/Blackwell용 런타임을 일반 x86 설치 명령으로 교체하지 않습니다. 실패 시 출력 오류에 따라 [문제 해결](05_troubleshooting.md)을 확인합니다.

## 3. 작은 오프라인 코드 흐름

```bash
python -m finetune_lab.prepare_data --task all --source demo --output-root data/demo --overwrite
python 01_instruction/pytorch/train.py --data-dir data/demo/instruction --dry-run
python 01_instruction/pytorch/train.py --data-dir data/demo/instruction --smoke-model --max-steps 2 --max-length 64 --gradient-accumulation 2 --device cuda --output-dir outputs/first_smoke
python -m finetune_lab.evaluate --model outputs/first_smoke --data-dir data/demo/instruction --max-eval-samples 2 --max-length 64 --max-new-tokens 8 --device cuda --output outputs/first_smoke_eval.json
```

`--smoke-model`은 랜덤 tiny GPT-2를 사용해 optimizer·저장·재로딩 경로를 점검합니다. 실제 사전학습 모델의 fine-tuning 성능을 평가하지 않습니다. `--dry-run`은 모델을 로드하지 않으므로 CUDA 호환성 검증도 아닙니다.

## 4. 실제 데이터와 base LLM

```bash
python -m finetune_lab.prepare_data --task instruction --source hf --train-samples 256 --eval-samples 32
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M --data-dir data/processed/instruction --split validation --max-eval-samples 8 --device cuda --output outputs/instruction_before.json
python 01_instruction/pytorch/train.py --max-steps 20 --device cuda --dtype auto
python -m finetune_lab.evaluate --model outputs/instruction/pytorch --data-dir data/processed/instruction --split validation --max-eval-samples 8 --device cuda --output outputs/instruction_after.json
```

초기 모델 실행은 가중치와 tokenizer를 HF에서 다운로드합니다. 각 run에 새 output-dir를 사용합니다. CUDA 사용을 명시하면 잘못 구성된 GPU 환경을 CPU 학습으로 조용히 대체하지 않습니다.

## 5. HF LoRA와 다음 대상

```bash
python 01_instruction/huggingface/train.py --method lora --max-steps 20 --device cuda --output-dir outputs/instruction_hf_lora
python -m finetune_lab.evaluate --model outputs/instruction_hf_lora --data-dir data/processed/instruction --split validation --max-eval-samples 8 --device cuda --output outputs/instruction_hf_after.json
```

[지식](../02_knowledge/README.md)과 [VLM](../03_vision/README.md) 순으로 진행합니다. Unsloth는 별도 profile을 사용합니다.

```bash
# 현재 컨테이너에서 빠져나와 Spark 호스트에서 시작
bash scripts/spark_container.sh unsloth
# 새 컨테이너 내부
bash scripts/install_spark.sh unsloth
source .venv-spark-unsloth/bin/activate
python 01_instruction/unsloth/train.py --max-steps 20
```

두 profile이 torch의 CUDA 빌드는 공유하지만 Python 라이브러리 overlay와 실행 프로세스는 분리합니다. 작은 검증 한 번으로 대형 모델의 품질과 메모리가 보장되지는 않습니다.
