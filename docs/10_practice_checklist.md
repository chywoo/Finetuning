# 전체 실습과 진행 체크리스트

**현재 모드: 작업 재개, 환경 준비 검토와 짧은 동작 검사.** 모델 학습과 오래 걸리는 작업은 실행하지 않습니다. 주요 작업 단위 후 local Git commit과 시간별 로그를 남깁니다.

환경 검사 기준 시각: 2026-10-02 10:58 EDT (America/Toronto, UTC−04:00). 기존 준비 집계는 06:33의 작성·데이터 준비 이력이다.

**기존 30개와 post-training 4개를 합해 기본 실습 34개를 관리한다. Spark에서 완료한 실습은 0/34개다.** [단계별 학습 계획](11_curriculum.md)을 따라 대표 실습을 하나씩 진행한 뒤 나머지 선택 조합으로 확장한다. 구현 준비와 실제 수행 체크를 구분한다. 시간별 변경 이력은 [TASK_LOGS.md](../TASK_LOGS.md)에 누적한다.

## 전체 실습 구성

| 대상 | PyTorch | Hugging Face | Unsloth | 합계 |
|---|---|---|---|---|
| Instruction SFT (I1–I7) | full, LoRA | full, LoRA, QLoRA | LoRA, QLoRA | 7 |
| 지식 CPT (C1–C7) | full, LoRA | full, LoRA, QLoRA | LoRA, QLoRA | 7 |
| 지식 QA SFT (S1–S7) | full, LoRA | full, LoRA, QLoRA | LoRA, QLoRA | 7 |
| CPT → QA SFT 연결 (K1–K6) | full checkpoint 연결 | full checkpoint / LoRA·QLoRA adapter 연결 | LoRA·QLoRA adapter 연결 | 6 |
| VLM 이미지 인식 (V1–V3) | decoder 일부 학습 | 언어 LoRA | 언어 LoRA | 3 |
| DPO (P1) | — | TRL DPO LoRA | — | 1 |
| Reward modeling (P2) | — | TRL RewardTrainer | — | 1 |
| RLHF PPO (P3) | — | TRL PPOTrainer | — | 1 |
| RLVR GRPO (P4) | — | TRL GRPO LoRA | — | 1 |
| **합계** | **8** | **17** | **9** | **34** |

Domain fine-tuning 주 경로는 [HF Domain 과정](../04_post_training/domain_huggingface/README.md)에서 기존 C4/S4/K3를 재사용한다. 새로운 케이스로 중복 집계하지 않는다. Post-training은 실제 공개 개발 사례와 공식 trainer를 바탕으로 HF/TRL 한 경로만 제공한다. 선택 근거는 학습 계획에 기록했다.

PyTorch는 직접 작성한 loop를 뜻하며 모델 로딩에는 HF를 사용한다. VLM PyTorch는 full fine-tuning이 아니라 decoder 일부 파라미터 학습이다. VLM Unsloth는 현재 기본 language LoRA 한 케이스로 집계하고 text QLoRA 옵션을 그대로 적용하지 않는다.

## 체크하는 방법과 완료 기준

- `[x]`는 해당 단계의 결과와 근거가 확보된 상태, `[ ]`는 미실시·미확인 상태다.
- 각 케이스의 **준비**는 코드·설명·관련 데이터가 존재한다는 뜻이다. CUDA kernel 성공이나 성능 향상을 뜻하지 않는다.
- 각 케이스의 나머지 네 단계를 모두 확인하면 실습을 완료한 것으로 집계한다. 품질이 떨어진 결과도 원인을 분석해 기록하면 실습 결과가 될 수 있다.
- Baseline은 동일한 모델 revision, 데이터 split, prompt, 평가 설정을 사용한다. 조건이 같은 기존 baseline 보고서를 재사용할 때는 경로를 기록한다.
- 학습 중 validation으로 설정을 고르고, 최종 비교에는 test를 사용한다. CPT는 문서 loss/ppl, QA는 EM·F1/생성 예시, instruction은 loss와 지시 준수·관련성·정확성, VLM은 accuracy·macro F1·confusion matrix를 확인한다.
- 단계 연결 실습은 base / CPT 후 / CPT→SFT 후를 비교하고, QA SFT만 실행한 대응 케이스와도 비교한다. 기존 instruction 능력의 보존 여부를 함께 본다.
- 출력은 `outputs/practice/<케이스ID>/<run-id>/`처럼 케이스와 실행별로 나누고, baseline·학습 모델·학습 후 보고서를 구분한다. 아래 CLI 옵션과 출력 경로를 각 README의 명령에 적용한다.
- 실행 완료 후 해당 체크박스, 아래 집계 표, TASK_LOGS의 시각·케이스ID·명령·보고서 경로·결과를 함께 갱신한다. 이 Markdown은 수동 관리이며 학습 실행만으로 자동 체크되지 않는다.

| 단계 | 최소 보존할 근거 |
|---|---|
| 학습 전 평가 | baseline 보고서와 생성 예시 |
| 학습 | 명령, 성공 종료 로그, model/adapter, training metadata, data manifest |
| 재로딩 | 저장 결과를 별도 프로세스로 로드한 평가/추론 로그 |
| 학습 후 비교 | 전후 지표, 오류 예시, 기존 능력 변화, 실행 시간·메모리 |

## 공통 선행 작업

현재 작업 공간에서는 기존 `data/processed/`와 나머지 demo 데이터가 확인되지 않았다. 아래 기존 준비 체크와 34/34 집계는 이전 작성·준비 이력이며, 현재 즉시 실행 가능한 케이스 수를 뜻하지 않는다. 이번에 `data/demo/instruction/`만 오프라인 생성했다. 실제 데이터 준비와 설치는 미완료다.

- [x] 0단계: 기존 가상환경의 주요 import와 작은 CPU/CUDA tensor 연산 검사.
- [x] I1 진입 검사: instruction demo 24/8/8 생성 및 `--dry-run` 성공.
- [ ] 전체 실습 의존성 준비: `datasets`, `peft`, `accelerate`, `trl`, `pytest`, `pytest-cov` 미설치.
- [ ] 현재 작업 공간의 실제 데이터·manifest 준비: `data/processed/` 미확인.

- [x] 대상 × 도구별 9개 디렉토리와 README/train.py 작성.
- [x] Dolly instruction 256/32/32, SciQ CPT 232/29/31, QA 255/32/32 준비.
- [x] Beans RGB 이미지와 분할 96/24/24, demo fixture 및 manifest 준비.
- [x] Spark 설치·장치 확인·학습/재로딩 검증 스크립트 작성.
- [x] Graft 탐색 카드 활용 및 [코드 지도 안내](09_code_map.md) 작성.
- [x] 입문→심화 단계별 학습 계획과 post-training 주 경로 설계.
- [x] AGENTS.md에 저장소·필요 SKILL·Graft·시간별 기록 규칙 작성.
- [ ] Spark post overlay 설치 및 DPO/Reward/PPO/GRPO별 실행 확인.
- [ ] Spark 호스트에서 NGC image 준비 및 HF overlay 설치 성공.
- [ ] Spark에서 `doctor --require-spark` CUDA/BF16 확인 성공.
- [ ] Spark HF profile 통합 검증 성공.
- [ ] Spark Unsloth overlay 설치와 profile 통합 검증 성공.
- [ ] Spark에서 선택적 vision 통합 검증 성공.
- [ ] 최신 test suite와 coverage 80% gate 통과.

짧은 환경 검사 결과와 다음 준비 항목은 [검증 상태](07_validation.md)를 따른다. Spark profile을 사용하는 경우 설치는 [DGX Spark 안내](04_dgx_spark.md)를 따른다.

## Instruction SFT: I1–I7

모델: SmolLM2-135M **base**. 데이터: Dolly. 응답 부분만 loss에 포함한다.

### I1. Instruction SFT · pytorch · full

[실행 설명](../01_instruction/pytorch/README.md) · `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### I2. Instruction SFT · pytorch · lora

[실행 설명](../01_instruction/pytorch/README.md) · `--method lora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### I3. Instruction SFT · huggingface · full

[실행 설명](../01_instruction/huggingface/README.md) · `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### I4. Instruction SFT · huggingface · lora

[실행 설명](../01_instruction/huggingface/README.md) · `--method lora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### I5. Instruction SFT · huggingface · qlora

[실행 설명](../01_instruction/huggingface/README.md) · `--method qlora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### I6. Instruction SFT · unsloth · lora

[실행 설명](../01_instruction/unsloth/README.md) · `--no-load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### I7. Instruction SFT · unsloth · qlora

[실행 설명](../01_instruction/unsloth/README.md) · `--load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

## 지식 CPT: C1–C7

모델: SmolLM2-135M-Instruct. 데이터: SciQ support. 본문의 다음 토큰을 학습한다.

### C1. 지식 CPT · pytorch · full

[실행 설명](../02_knowledge/pytorch/README.md) · `--stage cpt`; `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### C2. 지식 CPT · pytorch · lora

[실행 설명](../02_knowledge/pytorch/README.md) · `--stage cpt`; `--method lora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### C3. 지식 CPT · huggingface · full

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage cpt`; `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### C4. 지식 CPT · huggingface · lora

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage cpt`; `--method lora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### C5. 지식 CPT · huggingface · qlora

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage cpt`; `--method qlora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### C6. 지식 CPT · unsloth · lora

[실행 설명](../02_knowledge/unsloth/README.md) · `--stage cpt`; `--no-load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### C7. 지식 CPT · unsloth · qlora

[실행 설명](../02_knowledge/unsloth/README.md) · `--stage cpt`; `--load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

## 지식 QA SFT: S1–S7

모델: SmolLM2-135M-Instruct. 데이터: SciQ question → correct_answer. Prompt에 support를 포함하지 않는다.

### S1. 지식 QA SFT · pytorch · full

[실행 설명](../02_knowledge/pytorch/README.md) · `--stage sft`; `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### S2. 지식 QA SFT · pytorch · lora

[실행 설명](../02_knowledge/pytorch/README.md) · `--stage sft`; `--method lora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### S3. 지식 QA SFT · huggingface · full

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage sft`; `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### S4. 지식 QA SFT · huggingface · lora

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage sft`; `--method lora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### S5. 지식 QA SFT · huggingface · qlora

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage sft`; `--method qlora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### S6. 지식 QA SFT · unsloth · lora

[실행 설명](../02_knowledge/unsloth/README.md) · `--stage sft`; `--no-load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### S7. 지식 QA SFT · unsloth · qlora

[실행 설명](../02_knowledge/unsloth/README.md) · `--stage sft`; `--load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

## CPT → QA SFT: K1–K6

두 단계를 서로 다른 출력 경로에 저장한다. Full은 CPT checkpoint를 SFT 입력으로 사용하며, HF/Unsloth adapter는 같은 adapter를 이어 학습한다. PyTorch 저장 LoRA adapter의 직접 이어 학습은 이 목록에 포함하지 않는다.

### K1. CPT → QA SFT · pytorch · full

[실행 설명](../02_knowledge/pytorch/README.md) · `--stage cpt` → `--stage sft`; `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공 (CPT 및 SFT 두 단계).
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### K2. CPT → QA SFT · huggingface · full

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage cpt` → `--stage sft`; `--method full`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공 (CPT 및 SFT 두 단계).
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### K3. CPT → QA SFT · huggingface · lora

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage cpt` → `--stage sft`; `--method lora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공 (CPT 및 SFT 두 단계).
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### K4. CPT → QA SFT · huggingface · qlora

[실행 설명](../02_knowledge/huggingface/README.md) · `--stage cpt` → `--stage sft`; `--method qlora`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공 (CPT 및 SFT 두 단계).
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### K5. CPT → QA SFT · unsloth · lora

[실행 설명](../02_knowledge/unsloth/README.md) · `--stage cpt` → `--stage sft`; `--no-load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공 (CPT 및 SFT 두 단계).
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### K6. CPT → QA SFT · unsloth · qlora

[실행 설명](../02_knowledge/unsloth/README.md) · `--stage cpt` → `--stage sft`; `--load-in-4bit`

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공 (CPT 및 SFT 두 단계).
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

## VLM 이미지 인식: V1–V3

데이터: Beans 3클래스. PyTorch/HF는 SmolVLM-256M-Instruct, Unsloth는 Qwen2.5-VL-3B-Instruct다. Vision encoder는 고정한다.

### V1. VLM 이미지 인식 · pytorch · decoder 일부

[실행 설명](../03_vision/pytorch/README.md) · `--evaluate`로 전후 평가; 학습 옵션은 README 참조

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### V2. VLM 이미지 인식 · huggingface · 언어 LoRA

[실행 설명](../03_vision/huggingface/README.md) · `--evaluate`로 전후 평가; 학습 옵션은 README 참조

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

### V3. VLM 이미지 인식 · unsloth · 언어 LoRA

[실행 설명](../03_vision/unsloth/README.md) · `--evaluate`로 전후 평가; 학습 옵션은 README 참조

- [x] 준비: 코드·설명·데이터 작성.
- [ ] 학습 전 평가 저장.
- [ ] Spark CUDA 학습과 저장 성공.
- [ ] 별도 프로세스 재로딩 성공.
- [ ] 학습 후 비교·결과 해석 기록.

## LLM post-training: P1–P4

아래 과정은 준비 코드와 실제 수행을 분리해 체크한다. 단계 4부터 대학 수준으로 확장한다. DPO/PPO는 같은 SFT의 대안 경로, GRPO는 쉬운 산술 SFT의 별도 경로다.

### P1. DPO · Hugging Face TRL · LoRA

[DPO 수업](../04_post_training/dpo_huggingface/README.md). 선수: I4의 저장/merge 이해, preference prompt/chosen/rejected 검증.

- [x] 준비: 코드·상세 설명·선호 fixture·평가 경로 작성.
- [ ] 학습 전 preference/생성 평가 저장.
- [ ] Spark CUDA DPO 학습·저장 성공.
- [ ] 별도 프로세스 adapter 재로딩 성공.
- [ ] held-out 선호 지표·생성·기존 능력 비교와 해석 기록.

### P2. Reward model · Hugging Face TRL

[Reward model 수업](../04_post_training/rlhf_ppo_huggingface/README.md). 선수: P1의 선호 데이터/수식 이해; DPO 결과 checkpoint는 필요하지 않다.

- [x] 준비: scalar reward 학습·설명·데이터·pair 평가 경로 작성.
- [ ] 학습 전 held-out pair baseline 저장.
- [ ] Spark CUDA reward model 학습·저장 성공.
- [ ] 별도 프로세스 reward checkpoint 재로딩 성공.
- [ ] pair accuracy·길이 편향·오류 사례 비교와 해석 기록.

### P3. RLHF PPO · Hugging Face TRL

[PPO 수업](../04_post_training/rlhf_ppo_huggingface/README.md). 선수: I4/SFT 출발 checkpoint 및 P2의 실제 학습한 reward checkpoint.

- [x] 준비: policy/reference/value/reward 연결·설명·평가 경로 작성.
- [ ] 같은 SFT policy의 baseline 생성 평가 저장.
- [ ] Spark CUDA rollout/PPO 학습·policy 저장 성공.
- [ ] 별도 프로세스 policy 재로딩 성공.
- [ ] reward·KL·답변 길이·독립 생성 품질 비교와 해석 기록.

### P4. RLVR GRPO · Hugging Face TRL · LoRA

[GRPO 수업](../04_post_training/rlvr_grpo_huggingface/README.md). 선수: 쉬운 산술 SFT, 단일 최종 정수 verifier 이해. RLVR 보상 출처와 GRPO 최적화 방법을 구분한다.

- [x] 준비: 산술 fixture/SFT 경로·strict verifier·GRPO·평가 설명 작성.
- [ ] verifier 사례 검토와 학습 전 정답률 저장.
- [ ] Spark CUDA 그룹 생성/GRPO 학습·저장 성공.
- [ ] 별도 프로세스 adapter 재로딩 성공.
- [ ] 독립 test 정답률·형식 오류·reward 분산 비교와 해석 기록.

## 추가 데이터 실험과 확장 과제

아래는 기본 34개와 별도로 체크한다. 기존 엔진으로 가능한 추가 데이터 실험과 새 구현이 필요한 과제를 구분한다.

- [ ] A1. 가상 Nara 사실 QA SFT → `probe_seen_facts` 질문 변형 recall 검사 (데이터·평가 경로 준비).
- [ ] A2. Nara `test`의 미학습 사실 검사와 recall 결과의 차이 해석 (데이터·평가 경로 준비).
- [ ] A3. Text LoRA merge → 별도 프로세스 재로딩 및 원래 adapter 결과 비교 (merge 도구 준비).
- [ ] A4. PyTorch CPT LoRA → merge → 전체 checkpoint에서 SFT/new LoRA 실험 (기존 도구 조합; 직접 adapter resume와 구분).
- [ ] A5. 동일 모델·데이터에서 학습률, step, sequence 길이, LoRA rank 변화 비교 (옵션 준비).
- [ ] A6. 동일 계열 1.7B 모델로 확장하고 속도·메모리 기록 (실행 안내 준비).
- [ ] A7. 한국어 모델·데이터 도입 (별도 선정·데이터 준비 필요).
- [ ] A8. RAG / QA SFT / SFT+RAG 비교 (RAG 구현 필요).
- [ ] A9. DPO beta·선호 label noise·response 길이 편향 비교 (P1 이후 추가 실험).
- [ ] A10. Vision encoder/projector 학습, OCR·detection·chart QA 확장 (별도 구현·annotation 필요).
- [ ] A11. 다중 Spark DDP 및 optimizer state 포함 정확한 resume (별도 구현 필요).

배경과 확장 방향은 [추가 실험](08_next_experiments.md)을 참고한다.

## 현재 집계

| 실습군 | 기존 작성·준비 이력 | Spark 실습 완료 |
|---|---|---|
| Instruction | 7/7 | 0/7 |
| CPT | 7/7 | 0/7 |
| QA SFT | 7/7 | 0/7 |
| CPT → SFT | 6/6 | 0/6 |
| VLM | 3/3 | 0/3 |
| DPO / Reward / PPO / GRPO | 4/4 | 0/4 |
| **기본 실습 합계** | **34/34** | **0/34** |
| 추가 실험·과제 | 항목별 준비 범위가 다름 | 0/11 |

집계의 완료는 위 네 가지 실제 수행 체크가 모두 확인된 케이스만 센다. 구현 준비 숫자에 추가 과제의 미구현 기능을 포함하지 않는다.
