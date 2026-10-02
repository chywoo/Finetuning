# Fine-tuning 학습 실습실

실습 코드·교육 문서·데이터 검사와 빠른 회귀 검증을 보완했습니다. **172개 테스트 통과, 전체 coverage 81.56%**이며 실제 모델 학습과 품질 비교는 실행하지 않았습니다. [구축 완료와 실행 범위](docs/13_readiness.md)에 근거와 환경별 미확인 항목을 구분했습니다. 시간별 기록은 [TASK_LOGS.md](TASK_LOGS.md)를 따릅니다.

**LLM이 답하는 법을 배우는 SFT부터, 도메인 적응·선호 학습 DPO·RLHF PPO·검산 보상 RLVR GRPO와 VLM 이미지 적응까지 순서대로 실습합니다.** 설명은 한국어이고, 작은 모델과 영어 공개 데이터로 계산 비용을 줄였습니다. 영어 중심 모델의 실험 결과를 한국어 능력 향상으로 해석하지 않습니다.

## 무엇부터 시작할까?

[단계별 실습 계획](docs/11_curriculum.md)이 학습 순서와 다음 단계로 넘어갈 기준을 제공합니다. [전체 실습·진행 체크리스트](docs/10_practice_checklist.md)에서 도구별 선택 실습까지 체크하고, [작업 기록](TASK_LOGS.md)에 시간별 결과를 남깁니다. 처음에는 아래 대표 경로를 한 단계씩 따라갑니다.

| 순서 | 실습 소개 | 들어가기 전에 준비 | 수업 |
|---|---|---|---|
| 0 | 다음 단어 맞히기, 정답 점수, 데이터 분할 | Python 함수/CLI/JSONL 기초; 수학은 평균과 확률부터 | [고등학생 첫 수업](docs/12_foundations.md), [환경 준비와 첫 실습](docs/00_quickstart.md) |
| 1 | Base LLM에 질문→답변을 가르치는 PyTorch SFT | Dolly, base 모델, CUDA 환경 | [I1 PyTorch full](01_instruction/pytorch/README.md) |
| 2 | 작은 추가 행렬만 학습하는 LoRA | 1단계 저장/재로딩과 baseline 이해 | [I4 HF LoRA](01_instruction/huggingface/README.md) |
| 3 | 과학 문서 읽기와 과학 QA를 비교하는 Domain fine-tuning | SciQ, 일반 instruction baseline | [C4/S4/K3 Domain 과정](04_post_training/domain_huggingface/README.md) |
| 4 | 두 답 중 더 좋은 답을 배우는 DPO | SFT full/merge checkpoint와 chosen/rejected | [P1 HF TRL DPO](04_post_training/dpo_huggingface/README.md) |
| 5 | 선호 쌍으로 답변 채점기를 학습 | 선호 데이터 split과 pair 평가 | [P2 Reward model](04_post_training/rlhf_ppo_huggingface/README.md) |
| 6 | 채점기 보상으로 답하는 모델을 학습하는 RLHF | 같은 SFT 출발점, 학습한 reward checkpoint | [P3 PPO](04_post_training/rlhf_ppo_huggingface/README.md) |
| 7 | 정답을 검산해 보상하는 RLVR | 쉬운 산술 SFT, 엄격한 정수 채점기 | [P4 GRPO](04_post_training/rlvr_grpo_huggingface/README.md) |
| 8 | 이미지를 보고 콩잎 상태를 답하는 VLM 적응 | Beans 이미지/label과 processor 구조 | [V1 PyTorch](03_vision/pytorch/README.md) → [V2 HF](03_vision/huggingface/README.md) |
| 9 | 기법·도구·모델 크기를 바꿔 비교 | 앞 단계의 동일 조건 평가 기록 | [전체 선택 실습](docs/10_practice_checklist.md), [확장 과제](docs/08_next_experiments.md) |

처음은 고등학생 수준의 비유·숫자 예제로 설명하고, 단계가 진행되면 대학 수준의 확률·loss·KL·advantage로 확장합니다. [이론 문서](docs/01_theory.md)의 수식은 해당 단계에 도달한 후 참고합니다. DPO와 PPO는 같은 SFT 출발점에서 비교하는 대안이며, RLVR는 별도 산술 SFT에서 시작합니다.

## 기존 대상 × 도구: 9개 디렉토리, 30개 선택 케이스

| 대상 | 직접 PyTorch | Hugging Face | Unsloth |
|---|---|---|---|
| Instruction SFT | [설명·실행](01_instruction/pytorch/README.md) | [설명·실행](01_instruction/huggingface/README.md) | [설명·실행](01_instruction/unsloth/README.md) |
| 지식 CPT / QA SFT | [설명·실행](02_knowledge/pytorch/README.md) | [설명·실행](02_knowledge/huggingface/README.md) | [설명·실행](02_knowledge/unsloth/README.md) |
| VLM 이미지 인식 | [설명·실행](03_vision/pytorch/README.md) | [설명·실행](03_vision/huggingface/README.md) | [설명·실행](03_vision/unsloth/README.md) |

**도구와 학습 기법은 다른 축입니다.** PyTorch/Hugging Face/Unsloth는 구현 도구, SFT/CPT는 학습 목적, full/LoRA/QLoRA는 가중치를 업데이트하는 방식입니다. Unsloth도 내부적으로 PyTorch와 Hugging Face 생태계를 사용합니다. 직접 PyTorch 예제는 모델/tokenizer/processor를 HF에서 로드하고 학습 루프를 직접 작성합니다.

## 추가 LLM post-training

[Post-training 입구](04_post_training/README.md)에서 HF Transformers/PEFT + TRL 한 경로로 준비합니다. Domain 과정은 기존 지식 케이스를 재사용하고, DPO·reward model·PPO·GRPO 네 케이스를 추가해 총 34개를 관리합니다. 실제 개발 사례와 도구 선택 근거는 [실습 계획](docs/11_curriculum.md)에 기록했습니다.

Python 3.11/3.12와 장치에 맞는 PyTorch를 준비한 일반 실습 환경에서는 다음과 같이 post 도구를 추가합니다. Spark의 NGC overlay는 [선택 환경 안내](docs/04_dgx_spark.md)를 따릅니다.

```bash
source .venv-lab/bin/activate
python -m pip install -r requirements/lab-post.txt
python scripts/doctor.py --profile post --require-cuda
```

선호 데이터, reward model과 verifier의 입력 형식은 각 수업의 README에서 먼저 확인합니다. 자체 작성 선호·산술 데이터는 원리 실습용이며 실제 사람 피드백이나 일반 추론 benchmark로 해석하지 않습니다.

## 준비된 모델과 데이터

| 용도 | 모델 | 데이터 | 기본 소규모 분할 |
|---|---|---|---|
| instruction 추가 | SmolLM2-135M **base** | Dolly-15K | train 256 / validation 32 / test 32 |
| 기존 LLM 지식 적응 | SmolLM2-135M-Instruct | SciQ support / question·correct_answer | QA 255 / 32 / 32, CPT 232 / 29 / 31* |
| VLM 잎 이미지 인식 | SmolVLM-256M-Instruct | Makerere Beans | train 96 / validation 24 / test 24 |
| Unsloth VLM | Qwen2.5-VL-3B-Instruct | 같은 Beans 분할 | 위와 동일 |

\* 작성 시 내려받은 revision의 실제 전처리 결과입니다. 요청 수보다 적을 수 있으며 빈 support와 중복 제거 결과는 각 `manifest.json`에 기록됩니다. 모델 ID와 라이선스·원본 데이터 크기는 [데이터 안내](docs/02_datasets.md)에 있습니다. 두 VLM의 크기·구조가 달라 Unsloth 비교 결과를 도구의 순수 속도 차이로 해석할 수 없습니다.

## DGX Spark에서 바로 시작하기

DGX Spark를 사용하는 경우 [Spark 설치 안내](docs/04_dgx_spark.md)를 먼저 읽습니다. 아래 첫 명령은 Spark 호스트의 저장소 루트에서, 나머지는 컨테이너 내부에서 실행합니다. 다른 환경에서는 Python·라이브러리·장치 지원을 먼저 확인합니다.

```bash
bash scripts/spark_container.sh hf
# 아래부터 컨테이너 내부 /workspace/finetune
bash scripts/install_spark.sh hf
source .venv-spark-hf/bin/activate

python -m finetune_lab.prepare_data --task instruction --source hf
python 01_instruction/pytorch/train.py --dry-run
python 01_instruction/pytorch/train.py --max-steps 20 --device cuda
python -m finetune_lab.infer --model outputs/instruction/pytorch --device cuda --prompt "List three primary colors."
```

`--max-steps 20`은 동작을 배우는 짧은 실행입니다. 의미 있는 능력 향상을 보장하는 학습량이 아닙니다. 평가 전후 결과를 기록하고 데이터 수와 학습률을 바꿔 실험합니다. 작은 모델부터 시작해 1.7B/3B로 확장하는 Spark 설정도 준비했습니다.

## 디렉터리 구조

```text
01_instruction/{pytorch,huggingface,unsloth}/    # train.py + 자세한 README
02_knowledge/{pytorch,huggingface,unsloth}/      # --stage cpt 또는 sft
03_vision/{pytorch,huggingface,unsloth}/         # 학습과 --evaluate
04_post_training/                             # Domain, DPO, Reward/PPO, RLVR GRPO
finetune_lab/                                 # 읽어 볼 수 있는 공통 구현
docs/                                         # 이론·데이터·평가·하드웨어·문제 해결
data/demo/                                    # 작은 자체 작성 오프라인 fixture
data/processed/                               # 실제 HF 데이터 다운로드 결과
requirements/                                 # 환경별 설치 목록
scripts/doctor.py                             # 버전과 장치 확인
scripts/{spark_container,install_spark}.sh     # ARM64 CUDA 환경
scripts/validate_spark.py                      # Spark 학습·저장·재로딩 검증
graft/                                       # 사용자가 만든 코드 탐색 카드
AGENTS.md                                    # 저장소·SKILL·Graft 작업 안내
tests/                                        # 데이터·mask·CLI·작은 학습 검증
outputs/                                      # 모델·adapter·평가 보고서
```

실제 데이터는 이미 이 작업 공간의 `data/processed/`에 준비했습니다. 다른 컴퓨터에서는 `prepare_data` 명령으로 같은 HF 데이터셋을 내려받습니다. 다운로드 시 dataset revision SHA와 JSONL SHA256을 기록하며, 기존 분할을 덮어쓰려면 `--overwrite`를 지정해야 합니다. 큰 원본 데이터는 Hugging Face cache에도 저장됩니다.

저장소와 준비된 데이터를 함께 복사했다면 위 데이터 준비 명령은 생략합니다. 첫 실습에서 명령은 저장소 루트에서 실행하고, 모델 다운로드 공간과 새 output 경로를 확보합니다. 각 수업의 선수 조건·데이터 예시·baseline부터 읽은 뒤 학습 명령으로 진행합니다.

## 검증과 학습 기록

```bash
python scripts/check_data.py
python scripts/doctor.py --profile post --require-cuda
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_ML_TESTS=0 RUN_POST_TRAINING_TESTS=0 RUN_SPARK_POST_TESTS=0 python -m pytest --cov=finetune_lab --cov-report=term-missing --cov-fail-under=80
```

[검증 안내](docs/07_validation.md)에 실제 검사 결과와 미확인 범위를 정리했습니다. 모델 학습과 학습 전후 품질 비교, Unsloth·QLoRA·VLM native kernel 검증은 미실시입니다. [평가 안내](docs/03_evaluation.md)는 결과 해석, [문제 해결](docs/05_troubleshooting.md)는 OOM·마스킹·환경 오류를 다룹니다. 코드 연결 관계는 [Graft 탐색 안내](docs/09_code_map.md)를 참고합니다.

원문을 학습해도 정확한 지식 검색을 보장하지 않습니다. 자주 바뀌는 사실이나 출처가 필요한 답변에는 RAG도 비교합니다. [확장 학습](docs/08_next_experiments.md)에 DPO, 한국어 데이터, RAG 비교, 멀티 GPU 확장 과제를 정리했습니다.
