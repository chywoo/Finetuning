# 검증 상태와 Spark 실행 체크

## 현재 상태

최종 구현·교육 보완 및 빠른 검증은 [구축 완료와 실행 범위](13_readiness.md)를 기준으로 합니다. **172 passed, 학습 integration 2 skipped, 전체 coverage 81.56%**입니다. 아래 시간별 검사는 이전 시점의 이력이며, 최초 미설치 의존성은 별도 `.venv-lab`에 준비했습니다. Unsloth runtime 충돌과 공급 wheel tag 경고는 완료 문서에 구분했습니다.

새 post-training 범위는 [실습 계획](11_curriculum.md)과 [체크리스트](10_practice_checklist.md)의 P1–P4를 따른다. Domain은 기존 HF CPT/QA 경로를 재사용한다. DPO/Reward/PPO/GRPO는 post overlay에서 각각 학습·저장·별도 재로딩·독립 평가를 실행해야 하며, 아래 기존 HF/Unsloth matrix에 포함된 것으로 표시하지 않는다.

사용자 지시로 환경 준비 검토와 짧은 동작 검사를 재개했습니다. 모델 학습과 오래 걸리는 작업은 실행하지 않습니다. 최초 검사에서 없었던 `data/processed/`와 demo 데이터를 사용자가 복사했으며, 아래 재검사로 현재 데이터 가용성을 확인했습니다. 전체 의존성과 학습 실행 검증은 아직 미완료입니다.

### 0단계 환경 검사 결과 — 2026-10-02 10:58 EDT

프로젝트 루트의 `.venv/bin/python`으로 오프라인·제한시간 내 검사했습니다. 내부 장비 식별 정보와 절대 경로는 기록하지 않습니다.

| 검사 | 결과 |
|---|---|
| Python 요구 범위 `>=3.11,<3.13` | 기존 가상환경 충족; 기본 shell Python은 범위 밖이므로 `.venv/bin/python` 사용 |
| PyTorch / Transformers import | 성공 (`2.14.1` / `5.18.0`) |
| HF 주요 클래스 import | `AutoModelForCausalLM`, `AutoTokenizer`, `GPT2LMHeadModel`, `Trainer`, `TrainingArguments` 성공 |
| 작은 CPU/CUDA tensor 연산 | 성공; CUDA에서는 2×2 행렬곱 결과와 동기화 확인 |
| 전체 실습 의존성 | `datasets`, `peft`, `accelerate`, `trl`, `pytest`, `pytest-cov` 미설치 |
| I1 최초 demo / 실제 데이터 dry-run | 각각 종료 코드 2; 입력 파일 부재 |
| instruction demo 생성 후 I1 dry-run | 종료 코드 0; train/validation/test 24/8/8, schema와 split 검증 |
| 전체 학습·저장·재로딩, BF16/AMP, LoRA/QLoRA, post-training, coverage | 미확인 |

재현 명령은 프로젝트 루트에서 실행합니다. 생성 명령은 기존 JSONL이 있으면 덮어쓰지 않고 실패합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 15 .venv/bin/python -m finetune_lab.prepare_data --task instruction --source demo --output-root data/demo
.venv/bin/python 01_instruction/pytorch/train.py --data-dir data/demo/instruction --dry-run
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 45 .venv/bin/python -c 'import torch; x = torch.tensor([[1., 2.], [3., 4.]], device="cuda"); y = x @ x.T; torch.cuda.synchronize(); assert torch.equal(y.cpu(), torch.tensor([[5., 11.], [11., 25.]]))'
```

현재 패키지 버전은 `requirements/spark-hf.txt`의 고정 조합과 다릅니다. import 성공만으로 기존 Trainer/PEFT/TRL API 호환성을 확인한 것으로 표시하지 않습니다. 다음 작업은 사용할 profile의 의존성 조합을 검토하고 누락 패키지를 준비하는 것입니다. 이번에는 설치·다운로드·학습·pytest를 실행하지 않았습니다.

### 복사된 데이터 재검사 — 2026-10-02 11:02 EDT

- I1 demo dry-run: 종료 코드 0, train/validation/test 24/8/8.
- 실패했던 I1 실제 데이터 dry-run: 종료 코드 0, train/validation/test 256/32/32. 입력 부재 문제는 해결됐다.
- `data/demo/`, `data/processed/`의 manifest 10개와 nested SFT를 포함한 데이터 묶음 12개를 확인했다. JSONL 36개의 건수·SHA256, 기존 공통 검증 함수의 schema·split 검사와 post-training 선호/산술 계약 검사가 통과했다.
- 실제 CPT 232/29/31, QA SFT 255/32/32, vision 96/24/24도 manifest와 일치했다.
- demo/실제 vision 이미지 총 168개의 파일 존재, 데이터 폴더 안의 경로, 각 데이터 묶음 내 파일 SHA256 중복 없음을 확인했다. 이미지 decoding·RGB pixel hash·VLM processor 실행은 검사하지 않았다.
- 복사된 파일을 수정하거나 덮어쓰지 않았다. 데이터 검사 성공은 학습·성능·coverage 성공을 뜻하지 않는다.

실패했던 명령의 재현:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 15 .venv/bin/python 01_instruction/pytorch/train.py --data-dir data/processed/instruction --dry-run
```

지시 이전의 일부 작은 CPU 파이프라인 실행은 Spark의 ARM64 CUDA·Triton·bitsandbytes 호환성을 증명하지 않습니다. 결과를 Spark 성능 비교로 사용하지 않습니다. 변경 전 unit tests가 통과했더라도 최신 코드의 GPU 성공을 주장하지 않습니다.

코드 리뷰에서 response/EOS mask, causal shift, token-weighted PyTorch accumulation, VLM image-token 위치, adapter/base 재로딩, 경로 검사와 output 덮어쓰기 방지를 점검했습니다. Graft의 파일별 카드로 해당 구현 위치를 찾았습니다.

## Spark에서 실행할 검증

HF profile은 직접 PyTorch SFT/CPT, Trainer full, CPT LoRA → QA SFT adapter 이어 학습, 별도 프로세스 재로딩, merge 후 재로딩을 점검합니다. `--include-vision`을 붙이면 SmolVLM PyTorch/HF image training과 저장·재로딩도 확인합니다.

```bash
# HF overlay가 활성화된 Spark 컨테이너
python scripts/validate_spark.py --profile hf --include-vision
RUN_ML_TESTS=1 python -m pytest --cov=finetune_lab --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

Unsloth profile은 실제 작은 사전학습 모델로 instruction SFT, 지식 CPT/SFT 이어 학습과 저장·재로딩을 확인합니다. vision 옵션은 3B VLM 다운로드를 포함합니다.

```bash
# Unsloth overlay가 활성화된 Spark 컨테이너
python scripts/validate_spark.py --profile unsloth --include-vision
```

검증 script는 Mac/Linux x86에서 GPU 실행을 시작하지 않습니다. 각 subprocess의 명령·종료 코드·로그를 `outputs/spark_validation/<timestamp-profile>/`에 기록하고 실패하면 중단합니다. fixture를 사용하는 짧은 테스트이므로 능력 향상·실제 Beans accuracy를 측정하는 실험이 아닙니다. 그런 비교는 실제 데이터로 별도로 실행합니다.

## Coverage 기준

제공된 AGENTS.md의 목표는 80% 이상입니다. 최종 빠른 suite의 **전체 branch 포함 coverage 81.56%**를 확인했으며 gate 종료 코드는 0입니다. 학습 integration 2개는 명시적으로 비활성화했고, mock tests와 임의 초기화 모델 검사는 native kernel·실제 모델 품질 검증을 대신하지 않습니다. HF와 Unsloth를 한 프로세스에서 섞지 않고 profile별 실행 결과를 기록합니다.

GPU 검증 결과를 받으면 실행 환경 image digest·freeze·logs를 남기고 이 문서를 갱신합니다. 통과하지 않은 항목을 완료로 표시하거나 exception coverage로 감추지 않습니다.
