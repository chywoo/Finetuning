# Fine-tuning 실습 프로젝트 작업 기록

기록일: 2026-10-02 (America/Toronto)  
작업 위치: 프로젝트 루트
최초 현황: **9개 실습 디렉토리의 코드·문서·데이터 및 DGX Spark 실행 구성을 작성한 상태. 최신 구성의 Spark CUDA 실행 검증은 미실시.** 이후 post-training을 포함한 현재 범위와 상태는 9절 및 실습 체크리스트에서 관리한다.

이 기록은 현재까지의 진행 상황을 정리한다. 파일 작성 완료와 실제 GPU 동작 확인을 구분하며, 학습 성능 향상을 확인한 것으로 해석하지 않는다.

전체 케이스와 수행 체크의 기준은 [실습 체크리스트](docs/10_practice_checklist.md)에서 관리한다. 아래 1–8절은 최초 현황 스냅샷이며, 이후 변경은 9절에 시간순으로 누적한다. 이전 작업의 정확한 시각은 확보되지 않아 임의로 소급 기재하지 않는다.

## 1. 요청과 적용한 조건

- Base LLM에 instruction learning을 추가하는 실습을 제공한다.
- 기존 LLM에 추가 지식을 학습시키는 실습을 제공한다.
- 기존 VLM을 새로운 이미지 도메인에 적응시키는 실습을 제공한다.
- 각 대상을 직접 PyTorch, Hugging Face Trainer/PEFT, Unsloth 방식으로 나누고, 디렉토리마다 실행 코드와 자세한 한국어 설명을 작성한다.
- 이론, 데이터 정보, 전처리, 실행 절차, 평가와 결과 해석을 함께 설명한다.
- 실행 대상은 CUDA가 있는 **DGX Spark: ARM64 Linux / GB10 Blackwell**이다.
- 사용자가 만든 **Graft 파일별 코드 지도**를 탐색에 활용한다.
- 사용자 지시에 따라 **Intel Mac의 환경 검증과 학습 테스트를 중단했다.** 이후 소스와 문서의 정적 확인만 진행했으며, Mac 결과로 Spark 호환성을 주장하지 않는다.

## 2. 실습 디렉토리 작성

아래 9개 디렉토리에 각각 `train.py`와 상세 `README.md`를 작성했다.

| 대상 | PyTorch | Hugging Face | Unsloth |
|---|---|---|---|
| Instruction SFT | `01_instruction/pytorch/` | `01_instruction/huggingface/` | `01_instruction/unsloth/` |
| 지식 CPT / QA SFT | `02_knowledge/pytorch/` | `02_knowledge/huggingface/` | `02_knowledge/unsloth/` |
| VLM 이미지 인식 | `03_vision/pytorch/` | `03_vision/huggingface/` | `03_vision/unsloth/` |

각 대상의 상위 디렉토리에도 입문용 README를 작성했다. 공통 구현은 `finetune_lab/`에 두어 같은 데이터와 평가 기준으로 도구별 차이를 학습하도록 구성했다.

### LLM 구현

- Instruction 기본 모델은 `HuggingFaceTB/SmolLM2-135M` base 모델이다.
- 지식 학습 기본 모델은 `HuggingFaceTB/SmolLM2-135M-Instruct`이다.
- 직접 PyTorch는 학습 루프, optimizer, gradient accumulation, validation, 모델 저장을 직접 구현했다. Full fine-tuning과 LoRA를 지원한다.
- HF는 Trainer/PEFT의 full, LoRA, QLoRA 방식과 CPT → QA SFT adapter 이어 학습을 구성했다.
- Unsloth는 FastLanguageModel과 TRL SFTTrainer를 사용하도록 구성했다. Unsloth를 Transformers/TRL보다 먼저 import하며, CUDA와 Spark 조건을 검사한다.
- 공통 SFT encoding에서 prompt loss를 제외하고 completion/EOS를 학습한다. Padding mask, 길이 제한, causal shift 처리를 설명했다.
- PyTorch gradient accumulation과 validation loss에 유효 target token 수를 반영했다.
- 데이터 분할 중복 검사, 유한 loss 검사, 기존 모델 출력 덮어쓰기 방지, 저장 metadata를 추가했다.

### VLM 구현

- PyTorch/HF 기본 모델은 `HuggingFaceTB/SmolVLM-256M-Instruct`이다.
- Unsloth 기본 모델은 `Qwen/Qwen2.5-VL-3B-Instruct`이다.
- Beans 이미지를 입력하고 `angular_leaf_spot`, `bean_rust`, `healthy` 중 분류명을 생성하도록 학습한다.
- PyTorch는 언어 decoder 일부 파라미터, HF/Unsloth는 언어 부분 LoRA를 학습하며 vision encoder를 고정한다.
- Processor가 확장한 image token을 고려해 completion mask를 구성했다.
- 분할 간 ID, 이미지 경로, 파일 hash 및 제공된 이미지 hash의 중복 검사를 추가했다. RGB pixel hash와 PNG 파일 hash는 서로 다른 값으로 취급한다.
- 저장한 모델/adapter를 재로딩해 accuracy, macro F1, confusion matrix, 잘못 생성한 label을 평가하도록 구성했다.
- VLM adapter의 이어 학습은 현재 지원하지 않으며 명시적으로 거부한다. 저장 adapter의 평가 재로딩은 지원한다.
- 두 VLM의 크기와 구조가 달라 도구 사이의 순수 속도 비교로 해석할 수 없음을 문서에 명시했다.

## 3. 실제 데이터 다운로드 및 전처리

Hugging Face 공개 데이터의 소규모 subset을 실제로 다운로드하고 `data/processed/`에 저장했다. 아래 수치는 현재 `manifest.json` 기준이며, train / validation / test 순서이다.

| 경로 | 원본 데이터셋 | 준비한 건수 | 용도 |
|---|---|---|---|
| `data/processed/instruction/` | `databricks/databricks-dolly-15k` | 256 / 32 / 32 | instruction → response SFT |
| `data/processed/knowledge_cpt/` | `allenai/sciq` | 232 / 29 / 31 | support 원문 CPT |
| `data/processed/knowledge_sft/` | `allenai/sciq` | 255 / 32 / 32 | question → correct_answer SFT |
| `data/processed/vision/` | `AI-Lab-Makerere/beans` | 96 / 24 / 24 | 잎 이미지 분류명 생성 |

- 각 manifest에 dataset revision SHA, seed, 실제 분할 수, JSONL SHA256, 라이선스 정보를 기록했다.
- Dolly revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`.
- SciQ revision: `2c94ad3e1aafab77146f384e23536f97a4849815`.
- Beans revision: `27aa014ce09b193e1a6f58112d4a66e0eddb69c5`.
- SciQ는 빈 support와 중복 제거로 요청 수보다 실제 준비 수가 적다. QA prompt에는 정답을 포함한 support를 넣지 않는다.
- Dolly는 CC BY-SA 3.0, SciQ는 CC BY-NC 3.0으로 기록했다. Beans의 HF dataset card에는 라이선스가 명시되어 있지 않아 임의로 확정하지 않았다.
- Beans는 RGB PNG 이미지와 JSONL 경로를 함께 저장했다. 작은 subset이어도 원본 다운로드는 더 클 수 있음을 설명했다.
- 준비 스크립트는 `--source hf|demo`, seed, revision, 샘플 수를 지원한다. HF 다운로드 실패 시 demo 데이터로 조용히 대체하지 않는다.
- 기존 데이터 덮어쓰기는 `--overwrite`를 명시해야 한다.

`data/demo/`에는 자체 작성한 작은 텍스트와 합성 이미지 fixture도 준비했다. 실제 능력 측정용 데이터가 아니라 파이프라인 확인용이다. 지식 실습의 `probe_seen_facts.jsonl`은 학습한 사실을 질문만 바꾸어 확인하는 recall 검사이며, 독립적인 미학습 지식 일반화 평가와 구분했다.

## 4. 설명 문서와 평가 도구

루트 [README.md](README.md)에 학습 순서와 9개 실습 링크를 정리했다. 공통 설명은 다음과 같다.

| 문서 | 내용 |
|---|---|
| [빠른 시작](docs/00_quickstart.md) | 환경 준비와 첫 실습 |
| [이론](docs/01_theory.md) | SFT, CPT, full, LoRA, QLoRA, loss/mask와 망각 |
| [데이터](docs/02_datasets.md) | 모델/데이터 출처, 라이선스, JSONL 계약과 전처리 |
| [평가](docs/03_evaluation.md) | 학습 전후 비교, loss/ppl, QA 지표, VLM 지표와 해석 |
| [DGX Spark](docs/04_dgx_spark.md) | ARM64 CUDA 컨테이너, 설치와 확장 실험 |
| [문제 해결](docs/05_troubleshooting.md) | OOM, kernel, mask, 저장/재로딩 오류 |
| [출처](docs/06_sources.md) | 공식 문서, 모델/데이터 카드, 관련 논문 |
| [검증 상태](docs/07_validation.md) | 수행 여부, Spark 검증 명령과 coverage 기준 |
| [추가 실험](docs/08_next_experiments.md) | DPO, 한국어 데이터, RAG 비교, 멀티 GPU 확장 |
| [Graft 코드 탐색](docs/09_code_map.md) | 실습 wrapper와 공통 구현의 연결 관계 |

`finetune_lab/evaluate.py`, `infer.py`, `merge.py`를 작성했다. Text full 모델/adapter의 별도 재로딩, 생성 및 QA 평가, text LoRA merge를 지원한다. VLM 평가는 vision 엔진에 포함했다. 최신 실행 기본 장치는 CUDA로 맞췄다.

## 5. DGX Spark 실행 구성

- `scripts/spark_container.sh`: NGC `nvcr.io/nvidia/pytorch:25.11-py3` 기준 컨테이너 실행, GPU 노출, 작업 폴더와 cache/output 보존.
- `scripts/install_spark.sh`: HF/Unsloth overlay 환경 설치, NGC torch/torchvision/Triton 보존 constraints, `pip check`, 환경 freeze 기록.
- `scripts/doctor.py --require-spark`: Linux/aarch64, CUDA, capability 12.1, BF16 forward/backward 확인.
- `scripts/validate_spark.py`: profile별 작은 학습 → 저장 → 별도 프로세스 재로딩 → 평가 검증. HF text merge와 선택적 vision 검증도 포함한다.
- `requirements/spark-hf.txt`, `requirements/spark-unsloth.txt`: Spark용 패키지 조합 고정.
- 공식 NVIDIA/Unsloth 문서와 공개 package metadata를 조사하여 Transformers 4.57.6, Datasets 4.3.0, PEFT 0.18.1, Unsloth 2026.9.14, Zoo 2026.9.9, TRL 0.24.0, bitsandbytes 0.48.2 조합을 작성했다.
- Package metadata의 의존성 일치와 실제 ARM64 CUDA kernel 성공은 구분했다. Kernel 문제가 있을 때 공식 Spark Dockerfile/source build를 비교하는 절차를 설명했다.
- Intel Mac 가상환경을 Spark로 복사하지 않고 Spark 컨테이너 내부에서 별도 준비하도록 안내했다.

**Spark에 접속하거나 설치·훈련을 실행한 상태는 아니다.** 환경 구성과 검증 명령을 제공한 상태다.

## 6. Graft 활용 및 협업

- `graft/INDEX.md`와 파일별 카드로 데이터 준비, text 학습 엔진, vision 구현 위치를 탐색했다.
- 사용자가 만든 Graft 결과는 보존했다. 코드 변경 후 일부 카드의 줄 번호가 오래되었을 수 있으며, 최신 소스를 기준으로 확인한다.
- 계획, PyTorch text, HF text, Unsloth text, vision, 코드 리뷰를 역할별 에이전트에 나누어 진행했다.
- 리뷰에서 response/EOS mask, token 가중 accumulation, image token mask, adapter/base 재로딩, 경로 검증과 출력 보호를 점검하고 관련 수정을 반영했다.

## 7. 검증 진행 상태

| 항목 | 상태 |
|---|---|
| 실습 디렉토리·코드·상세 문서 작성 | 작성 완료 |
| 실제 Dolly/SciQ/Beans subset 준비와 manifest 저장 | 완료 |
| 사용자 지시 전 일부 unit test와 작은 CPU 파이프라인 실행 | 수행했으나 최신 Spark 구성의 검증 근거로 사용하지 않음 |
| Intel Mac 환경/학습 검증 | 사용자 지시에 따라 중단 |
| 최근 소스 AST parsing, 로컬 Markdown 링크, shell 문법 확인 | 정적 확인 수행; 당시 오류 없음 |
| 최신 추가·변경 test 전체 실행 | 미실시 |
| 전체 coverage 80% 이상 | **미확인**; 완료로 표시하지 않음 |
| DGX Spark 설치, CUDA/AMP, Triton/bitsandbytes 실행 | **미실시** |
| Spark에서 HF/Unsloth/VLM 학습·저장·재로딩 통합 검증 | **미실시** |
| 실제 데이터 학습 전후 능력 향상/속도/메모리 비교 | **미실시** |

AGENTS.md의 coverage 80% 요구를 충족했다고 주장하지 않는다. GPU 통합 검증과 coverage gate는 Spark에서 실행해야 한다. Mock tests와 과거 CPU 결과는 실제 Spark kernel 검증을 대신하지 않는다.

## 8. 남은 작업과 다음 실행

1. 개별 README의 데이터 재준비 명령에서 기존 demo/processed 데이터와 `--overwrite` 안내를 최종 확인한다.
2. 학습 전후 평가 출력 경로가 서로 덮어쓰지 않는지 문서의 명령을 최종 확인한다.
3. SciQ 샘플을 확장할 때 CPT support와 QA의 source-row 관계까지 함께 고려한 분할 누수 방지를 보강할지 검토한다. 현재 준비한 subset에서는 support의 분할 간 겹침이 없음을 정적으로 확인했다.
4. Spark에서 실제 설치와 profile별 검증을 실행하고 실패 원인을 수정한다.
5. Spark에서 최신 test suite와 coverage gate를 실행한다.
6. 실제 공개 데이터로 학습 전후 평가를 기록하고, 실행 환경·명령·model/data revision·메모리·시간을 보존한다.
7. 코드 변경이 마무리되면 필요에 따라 Graft 지도를 갱신하고 검증 상태 문서를 업데이트한다.

Spark 컨테이너 내부의 HF overlay를 활성화한 후 실행할 핵심 명령:

```bash
python scripts/doctor.py --require-spark
python scripts/validate_spark.py --profile hf --include-vision
RUN_ML_TESTS=1 python -m pytest --cov=finetune_lab --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

Unsloth overlay에서는 별도 프로세스로 실행한다:

```bash
python scripts/validate_spark.py --profile unsloth --include-vision
```

이번 기록 요청에서는 기존 파일과 manifest를 읽고 이 로그를 작성했다. 환경 설치, ML import, 학습, pytest 또는 CUDA 검증을 실행하지 않았다.

## 9. 시간별 진행 이력

모든 시각은 America/Toronto 기준으로 기록한다. 현재 날짜의 시간대는 EDT (UTC−04:00)이다. 기록은 변경이나 완료 이벤트가 발생할 때 추가하며, 기록이 없는 시간의 작업을 추정하지 않는다.

### 2026-10-02 05:51:20 EDT — 실습 목록·체크 체계 정리 시작

- 요청: 전체 실습 케이스를 정리하고 진행 체크를 제공하며, TASK_LOGS에 시간별 이력을 남긴다.
- CLI 소스와 기존 설명을 읽어 지원하는 full/LoRA/QLoRA, CPT/SFT, VLM 학습 범위를 확인했다.
- 기본 실습을 Instruction 7개, CPT 7개, QA SFT 7개, CPT→SFT 6개, VLM 3개로 분류했다. 합계 30개다.
- PyTorch 저장 LoRA adapter의 직접 이어 학습은 지원하지 않으므로 기본 연결 실습에서 제외했다. Merge 후 학습은 추가 조합 과제로 분리한다.
- 현재 상태: 코드·설명·데이터 준비 30/30, 실제 Spark 수행 완료 0/30. 구현 준비와 CUDA 실행 완료를 별도로 표시한다.
- Intel Mac 환경 검증·학습·pytest는 실행하지 않았다.

### 2026-10-02 05:54:34 EDT — 실습 체크리스트 작성 및 정적 확인 완료

- `docs/10_practice_checklist.md` 작성: 기본 실습 30개에 I/C/S/K/V ID를 부여하고 관련 README·CLI 옵션을 연결했다.
- 케이스마다 준비, baseline, Spark 학습·저장, 별도 재로딩, 학습 후 비교 체크박스를 제공했다. 준비는 확인된 30개만 체크하고 실제 수행은 미완료로 유지했다.
- 공통 환경 준비 체크, 실습군별 완료 집계, 추가 데이터 실험·확장 과제 11개와 결과 증빙 기준을 작성했다.
- 루트 README에 체크리스트와 시간별 작업 기록 링크를 추가했다. TASK_LOGS 최초 현황 스냅샷은 보존하고 시간별 이벤트 누적 방식과 기록 양식을 추가했다.
- 읽기 전용 리뷰에서 지원 옵션과 케이스 수를 재확인했다: 7 + 7 + 7 + 6 + 3 = 30.
- 표준 라이브러리로 문서만 정적 확인했다: 케이스 ID 30개·중복 없음, 준비/수행 체크 항목 일치, 변경 문서의 로컬 링크 누락 없음.
- 현재 집계: 준비 30/30, Spark 실습 완료 0/30, 추가 실험 완료 0/11. Spark CUDA 검증과 coverage 80%는 여전히 미확인이다.
- ML library import, 환경 설치, 훈련, pytest는 실행하지 않았다.

### 향후 이력 작성 형식

아래 형식을 복사해 9절 끝에 실제 시각과 확인한 결과를 누적한다. 동일 케이스 재실행도 기존 기록을 덮어쓰지 않고 새 이벤트로 남긴다.

```text
### YYYY-MM-DD HH:MM:SS 시간대 — 케이스ID / 작업
- 상태 변경: 준비 / baseline / 학습 / 재로딩 / 비교 / 실패
- 실행 환경: Spark image digest, profile, package freeze 경로
- 명령과 설정: 실제 명령 또는 저장한 명령 파일 경로, seed
- 데이터·모델: manifest 경로, revision, checkpoint/adapter 경로
- 근거: 실행 로그, 종료 코드, 지표/생성 보고서 경로
- 결과: 확인한 내용, 실패 원인 또는 남은 작업
- 체크 갱신: 케이스 단계 및 완료 집계 변경
```

### 2026-10-02 06:19:56 EDT — Post-training 확장과 단계별 교육 계획 작성

- 요청 반영: DPO/RLHF, RLVR, Domain fine-tuning을 실제 개발 사례와 공식 trainer를 기준으로 HF/TRL 한 경로에 추가한다.
- 계획 범위: 기존 30개에 DPO(P1), Reward model(P2), PPO(P3), GRPO(P4)를 더해 34개. Domain 주 경로는 기존 C4/S4/K3를 재사용하고 중복 집계하지 않는다.
- `docs/11_curriculum.md`에 0–9단계 학습 순서, 선수 조건, 고등학생→대학생 난이도, 결과물·완료 기준·다음 수업을 작성했다.
- README를 시작 길잡이로 확장하고 각 실습 소개·준비·입구 링크를 제공했다. DPO와 PPO는 동일 SFT의 대안 경로, RLVR는 별도 산술 SFT 경로로 설명했다.
- `AGENTS.md`에 저장소 구조, 필요한 SKILL 선택/읽기 규칙, Graft 탐색, Spark 전용 실행, Intel Mac 검증 중단, 시간별 기록·검증 규칙을 작성했다.
- Domain HF 과정의 상세 문서와 기존 HF engine wrapper, post overlay requirements/launcher/install 지원을 작성했다.
- DPO/GRPO·데이터, reward/PPO, 입문 수업 문서를 파일 소유권별로 위임했다. 새 4개 post-training 케이스의 구현 완료 체크는 아직 하지 않았다.
- 공식 TRL 0.24 문서와 Llama 3/InstructGPT/DeepSeekMath/도메인 적응 논문을 확인했다. 업계 도구 사용량 1위라고 근거 없이 주장하지 않는다.
- Intel Mac 환경 설치·ML import·pytest·학습을 수행하지 않았다. Spark 실제 수행과 coverage 80%는 미확인이다.

### 2026-10-02 06:26:10 EDT — 입문 수업과 post-training 구현·리뷰 진행

- `docs/12_foundations.md`에 299줄의 고등학생 첫 수업을 작성했다. Python/CLI/JSONL, token ID, 확률표와 -ln 손계산, gradient 한 숫자 예제, 8개 예제의 batch/accumulation, split, mask/EOS, checkpoint/LoRA, Spark 용어와 자기점검 해설을 포함한다.
- 기존 9개 실습 README에 선수 수업, 대표 경로와 선택 비교의 구분, 완료 기준, 다음 수업 링크를 추가했다.
- DPO/GRPO 및 Reward/PPO 실제 CLI, 새 선호·산술 fixture와 SFT 변환 파일을 작성했다. Preference는 16/4/4, math는 32/8/8이다. 새 상세 README와 검증 결과 점검은 진행 중이다.
- 읽기 전용 리뷰에서 산술 SFT 선수 checkpoint 준비 경로, 상세 README 링크, 모델 저장 전 finite metrics 검증과 진행 집계 갱신을 지적했다. 작성 중인 README와 finite 검증에 반영하고 최종 확인할 예정이다.
- Mac에서는 Python AST·문서 링크·데이터 파일만 정적으로 확인했다. 첫 manifest 점검은 작성된 fixture manifest의 hash key 누락으로 중단되었으며, physical manifest 보강 후 다시 확인한다. ML 실행 실패가 아니다.
- CUDA 실제 수행 완료 체크는 여전히 0/34이다. 최신 test source는 실행하지 않았다.

### 2026-10-02 06:33:01 EDT — 단계별 과정과 34개 실습 준비 완료, 정적 확인

- 단계별 계획·루트 길잡이·299줄 입문 수업·AGENTS 작성 완료. 기존 9개 실습에 선수/다음 단계 안내를 추가하고 4개 post-training 디렉토리를 작성했다.
- 새 주 경로는 Domain HF → DPO LoRA → reward model/PPO 비교 → RLVR GRPO다. DPO/PPO는 `outputs/preference_sft_full`을 공통 출발점으로 비교하고, GRPO는 별도의 `outputs/math_sft_full`에서 시작하도록 선수 SFT/merge 명령을 제공했다.
- DPO/Reward/PPO/GRPO 실제 학습·평가 CLI, CUDA 전용 guard, split 검사, model/reference/reward/value 연결, 출력 보호, 유한 loss/metrics 검증, 저장·별도 재로딩 경로와 test source를 작성했다. 해당 코드를 Spark에서 실행한 것은 아니다.
- 자체 작성 preference 16/4/4, math 32/8/8과 같은 split의 SFT 데이터 동봉. 물리 파일의 SHA256 manifest를 보강했다. 실제 사람 피드백/일반 추론 benchmark라고 주장하지 않는다.
- 데이터 문서에 UltraFeedback/Anthropic HH/GSM8K 공개 데이터 확장·변환 계획을 기록했다. 해당 새 공개 데이터의 다운로드/변환 완료로 표시하지 않았다.
- 정적 확인 통과: Python 소스 41개 AST parsing, Markdown 33개 로컬 링크 누락 없음, 34개 case ID 중복 없음, source 파일 800줄 제한, 두 Spark shell script 문법 정상.
- 새 데이터의 원본/SFT counts와 SHA256, split ID/prompt 중복 없음, 원본→SFT 대응을 표준 라이브러리로 확인했다. 이전 누락 manifest 문제는 해결했다.
- 최종 읽기 전용 리뷰에서 HIGH/CRITICAL correctness issue 및 broken command를 찾지 못했다. 이는 실제 TRL/CUDA kernel 성공의 증거가 아니다.
- 진행표 갱신: 준비 34/34, Spark 수행 완료 0/34, 추가 과제 완료 0/11. 전체 coverage 80% 및 최신 runtime tests는 미확인이다.
- 남은 작업: Spark post overlay 설치, 케이스별 baseline→학습→저장→별도 재로딩→독립 test 비교, 기존 HF/Unsloth matrix와 coverage gate 실행, 결과를 실제 시각별로 누적 기록. 새 post 소스의 Graft 카드는 필요 시 갱신한다.
- Intel Mac 환경 설치·ML import·pytest·학습을 수행하지 않았다.

### 2026-10-02 10:14:47 EDT — 구현 중단 및 주요 작업 후 Git commit 규칙 적용

- 사용자 지시: “지금 구현은 하지 말라.” 추가 실행 코드·테스트 구현을 중단하고 이후 범위를 계획·문서 정리로 제한한다. 해당 지시 전에 post-training 코드가 이미 작성된 상태였으며, 작성된 현황을 숨기거나 실행 완료로 표시하지 않는다.
- 사용자 지시: 주요 작업 후 Git commit을 진행한다. AGENTS·README·실습 계획·체크리스트에 현재 작업 모드와 local milestone commit 규칙을 반영했다.
- 모든 하위 작업자가 완료 상태임을 확인했다. 새 구현 작업을 위임하거나 재개하지 않았다.
- Git 상태: 저장소의 main branch가 존재하지만 아직 commit이 없다. 현재 코드·교육 문서·실습 계획의 초기 스냅샷을 Conventional Commit으로 기록한다.
- Git 기록에서 outputs·Spark 가상환경을 제외하도록 ignore를 보강했다. 기존 ignore의 data/cache/Graft 제외는 유지한다. 내려받은 데이터와 로컬 모델은 작업 공간에 있으며, 데이터 준비 스크립트와 설명은 commit 대상이다.
- Commit 전 읽기 전용 확인: 소스·문서의 민감정보 노출 가능성을 점검했다. 이전 정적 리뷰 결과와 CUDA/coverage 미확인 상태를 기록한다. Intel Mac의 환경 감사·ML import·pytest·학습은 실행하지 않는다.
- 초기 스냅샷 commit 제목: `feat: add staged fine-tuning and post-training learning lab`. 이 commit은 중단 전에 작성된 결과와 현재 문서를 기록하며 구현 재개가 아니다.
- 이후에는 각 주요 계획/문서 작업 종료 시 변경 검토→local commit→시각별 이력 기록을 진행한다. Push·원격 작업은 실행하지 않는다.

### 2026-10-02 10:17:12 EDT — 초기 스냅샷 Git commit 완료

- Commit: `08d4c97` — `feat: add staged fine-tuning and post-training learning lab`.
- 85개 파일을 local main branch의 최초 commit으로 기록했다. 주요 계획·문서·중단 전에 작성된 소스를 보존했고, 모델 출력·가상환경·캐시·기존 ignore 대상 데이터/Graft는 포함하지 않았다.
- Commit 직후 `git status --short`는 비어 있었다. 원격 push는 실행하지 않았다.
- 구현 중단 상태를 유지한다. 이번 요청 후에는 실행 코드·테스트 구현을 변경하지 않았으며, 현재 작업은 운영 규칙·문서·Git 기록 정리다.
- 이 완료 이력을 별도 문서 commit(`docs: record initial milestone commit`)으로 기록한다. 해당 commit 자체의 hash는 `git log`에서 확인한다.

### 2026-10-02 10:49:48 EDT — Skill 설치 및 기록 민감정보 정비

- 저장소 지침에서 요구한 ECC skills를 설치했다: `mle-workflow`, `pytorch-patterns`, `python-patterns`, `python-testing`, `eval-harness`, `codebase-onboarding`, `code-tour`.
- 작업 위치 표기를 특정 내부 절대 경로 대신 `프로젝트 루트`로 정정했다.
- 기존 기록의 민감정보 점검 문구를 일반화했으며, 내부 장비 식별 정보나 비밀 값은 기록하지 않는다.
- 이번 작업은 skill 설치와 문서 기록 정비만 포함하며, 구현·테스트·다운로드·학습은 실행하지 않았다.

### 2026-10-02 11:00:14 EDT — 0단계 / I1 환경·데이터 진입 검사

- 사용자 지시로 작업을 재개했다. 이번 범위는 환경 검토와 짧은 동작 검사이며 모델 학습과 오래 걸리는 작업은 제외한다.
- `AGENTS.md`, 진행 기록, 체크리스트, 커리큘럼, Graft INDEX와 코드 지도를 읽고 첫 미완료 단계인 공통 환경 준비를 진행했다. 설치된 `codebase-onboarding`, `mle-workflow`의 탐색·재현성·근거 구분 지침을 적용했다.
- 프로젝트 루트의 `.venv/bin/python`은 프로젝트 Python 요구 범위를 충족했다. 기본 shell Python은 범위 밖이므로 검사에 사용하지 않았다. PyTorch 2.14.1과 Transformers 5.18.0 import, HF 주요 클래스 import, 작은 CPU tensor 연산과 CUDA 2×2 행렬곱·동기화가 성공했다. 제한시간은 45초이며 모델을 생성하거나 학습하지 않았다.
- `datasets`, `peft`, `accelerate`, `trl`, `pytest`, `pytest-cov`는 미설치다. 현재 패키지와 기존 고정 profile 조합도 다르므로 전체 환경 준비와 API 호환성을 완료로 표시하지 않는다. 사용자 변경 중인 `pyproject.toml`과 `AGENTS.md`를 보존했다.
- 최초 I1 dry-run은 `data/demo/instruction/`과 `data/processed/instruction/` 입력 부재로 각각 종료 코드 2였다. 현재 작업 공간에는 기존 `data/`가 없었으며, 이전 다운로드 이력을 현재 데이터 가용성과 구분했다.
- 실행: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 15 .venv/bin/python -m finetune_lab.prepare_data --task instruction --source demo --output-root data/demo`. 기존 스크립트로 자체 작성 fixture 24/8/8과 manifest를 생성했다. 다운로드·덮어쓰기는 없었다.
- 실행: `.venv/bin/python 01_instruction/pytorch/train.py --data-dir data/demo/instruction --dry-run`. 종료 코드 0, schema/split 검사 성공. 생성 데이터는 기존 Git ignore 대상이며 commit에 넣지 않는다.
- 문서 갱신: README·커리큘럼·체크리스트의 중단 안내를 현재 범위로 변경하고, 검증 문서에 결과·실패·재현 명령·미확인 항목을 기록했다. 프로젝트가 특정 장비만을 위한 것이라고 표현하지 않으며 내부 장비 식별 정보·절대 경로·비밀 값은 기록하지 않았다.
- 확인 근거: 체크리스트 case ID 34개·중복 없음, 생성 manifest SHA256 일치, 변경 문서의 로컬 링크 정상, `git diff --check` 통과.
- 체크 갱신: 0단계 짧은 환경 검사와 I1 demo 진입 검사만 완료. 기존 작성·준비 이력 34/34, 실제 실습 완료 0/34, 추가 과제 0/11 유지. 전체 coverage·BF16/AMP·학습·저장·재로딩은 미확인이다.
- 남은 일: 사용할 profile의 의존성 조합 검토, 누락 패키지와 실제 데이터 준비. 이번에 학습·설치·다운로드·pytest·원격 작업은 실행하지 않았다. 문서 변경을 `docs: record resumed environment readiness checks`로 local commit한다.
