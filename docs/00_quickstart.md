# 첫 실습: 환경 확인부터 학습 전후 비교까지

처음이라면 [고등학생 입문 수업](12_foundations.md)에서 CLI·JSONL·토큰·loss를 배운 뒤 진행합니다. 학습 순서와 다음 수업의 조건은 [커리큘럼](11_curriculum.md), 전체 방법별 케이스는 [체크리스트](10_practice_checklist.md)를 따릅니다.

모든 명령은 **프로젝트 루트**에서 실행합니다. 이 프로젝트는 특정 장비 전용이 아닙니다. PyTorch/HF의 작은 full·LoRA 텍스트 실습은 CPU에서도 가능하지만 실제 학습에는 CUDA 환경을 권장합니다. QLoRA·Unsloth·DPO/PPO/GRPO는 각 수업의 CUDA·BF16·라이브러리 요구사항을 따릅니다. 지원 여부와 실행 확인 여부는 다르며 실제 확인 결과는 [검증 기록](07_validation.md)에 있습니다.

## 1. 실습 환경 선택

일반 Linux 환경에서는 장치에 맞는 PyTorch를 [공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 준비하고, 별도 가상환경에 고정 버전 HF 도구를 설치합니다. 이미 설치된 실습 환경이면 다시 만들지 않습니다.

```bash
python3.12 -m venv .venv-lab
source .venv-lab/bin/activate
# 이 환경에 CPU 또는 CUDA에 맞는 PyTorch를 먼저 설치
python -m pip install -r requirements/lab-hf.txt
python scripts/doctor.py --profile hf
```

Python 3.11 또는 3.12를 사용합니다. 현재 설치된 최신 Transformers가 이 수업의 고정 버전 TRL API와 호환된다고 가정하지 않습니다. Post-training은 같은 HF 기반 환경에 `requirements/lab-post.txt`를 추가 설치합니다. Unsloth는 [공식 설치 안내](https://unsloth.ai/docs/get-started/install-and-update)에서 플랫폼별 지원 조건을 확인하고 HF 환경과 분리합니다. `requirements/lab-unsloth.txt`의 고정 릴리스는 PyTorch `<2.13`을 요구하므로 기존 PyTorch 2.14 환경에 덮어 설치하지 않습니다. 호환되는 별도 CUDA 환경에서 먼저 의존성 해결을 확인합니다.

DGX Spark를 사용하는 경우에는 [전용 환경 안내](04_dgx_spark.md)의 NGC overlay 절차를 선택합니다. 제공된 CUDA PyTorch를 일반 wheel로 교체하지 않습니다. 가상환경과 모델 출력은 다른 호스트에서 복사해 재사용하지 않고 코드·문서·JSONL·이미지만 이동합니다.

## 2. 학습 없이 먼저 확인하기

이미 복사된 `data/processed/`가 있으면 다시 다운로드하지 않습니다. 아래 검사는 모델을 로드하거나 학습하지 않고 JSONL의 schema·split·중복·manifest 무결성과 실행 계획을 확인합니다.

```bash
python scripts/check_data.py
python 01_instruction/pytorch/train.py --dry-run
python 02_knowledge/huggingface/train.py --stage cpt --dry-run
python 02_knowledge/huggingface/train.py --stage sft --dry-run
python 03_vision/huggingface/train.py --dry-run
python scripts/doctor.py --profile hf --require-cuda
```

데이터가 없다면 다운로드 없이 자체 작성 fixture를 새 폴더에 준비할 수 있습니다.

```bash
python -m finetune_lab.prepare_data --task all --source demo --output-root data/demo_new
python 01_instruction/pytorch/train.py --data-dir data/demo_new/instruction --dry-run
```

Dry-run 성공은 CUDA 커널·optimizer·모델 저장 성공을 뜻하지 않습니다. `doctor.py`는 패키지와 장치 상태를 확인하는 단계입니다. 실제 모델 품질도 이 단계에서 판정하지 않습니다.

## 3. 학습 실행을 허용한 뒤: 첫 작은 실행

아래 명령부터는 가중치를 업데이트하는 **학습**입니다. 환경만 검사하려면 2절에서 멈춥니다.

```bash
python 01_instruction/pytorch/train.py --data-dir data/demo/instruction --smoke-model --max-steps 2 --max-length 64 --gradient-accumulation 2 --device cuda --output-dir outputs/first_smoke
python -m finetune_lab.evaluate --model outputs/first_smoke --data-dir data/demo/instruction --max-eval-samples 2 --max-length 64 --max-new-tokens 8 --device cuda --output outputs/first_smoke_eval.json
```

CPU에서는 텍스트 full 실습의 `--device cpu --dtype float32`를 사용합니다. `--smoke-model`은 임의 초기화 tiny GPT-2로 loop·저장·재로딩을 점검하므로 사전학습 모델의 fine-tuning 품질을 평가하지 않습니다. 기존 출력은 덮어쓰지 않고 새 이름을 선택합니다.

## 4. 실제 공개 데이터와 base LLM

Dolly 데이터가 없을 때만 첫 준비 명령을 실행합니다. 모델의 첫 평가에는 가중치와 tokenizer 다운로드가 필요할 수 있습니다.

```bash
python -m finetune_lab.prepare_data --task instruction --source hf --train-samples 256 --eval-samples 32
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M --data-dir data/processed/instruction --split validation --max-eval-samples 8 --device cuda --output outputs/instruction_before.json
python 01_instruction/pytorch/train.py --max-steps 20 --device cuda --dtype auto --output-dir outputs/instruction_first_full
python -m finetune_lab.evaluate --model outputs/instruction_first_full --data-dir data/processed/instruction --split validation --max-eval-samples 8 --device cuda --output outputs/instruction_after.json
```

같은 split·prompt·생성 설정으로 학습 전후를 비교합니다. 설정은 validation으로 고르고 test는 최종 비교에 사용합니다. 20 step은 흐름을 익히는 시작 설정이며 품질 개선을 보장하지 않습니다.

## 5. HF LoRA에서 다음 대상으로

```bash
python 01_instruction/huggingface/train.py --method lora --max-steps 20 --device cuda --output-dir outputs/instruction_hf_lora
python -m finetune_lab.evaluate --model outputs/instruction_hf_lora --data-dir data/processed/instruction --split validation --max-eval-samples 8 --device cuda --output outputs/instruction_hf_after.json
```

LoRA는 원래 모델을 고정하고 작은 adapter를 학습합니다. 재로딩에는 같은 base 모델·revision과 tokenizer가 필요합니다. 자세한 원리는 [HF instruction 수업](../01_instruction/huggingface/README.md)에 있습니다.

첫 두 실습의 저장·별도 재로딩·전후 평가가 끝나면 [Domain](../04_post_training/domain_huggingface/README.md) → [DPO](../04_post_training/dpo_huggingface/README.md) 또는 [Reward model→PPO](../04_post_training/rlhf_ppo_huggingface/README.md) → 별도 [RLVR](../04_post_training/rlvr_grpo_huggingface/README.md) → [VLM](../03_vision/README.md)으로 진행합니다. Unsloth와 QLoRA는 기본 흐름을 익힌 뒤 선택 비교합니다.
