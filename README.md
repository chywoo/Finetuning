# Fine-tuning 학습 실습실

## 이 프로젝트에서는 무엇을 배우나요?

이미 만들어진 AI 모델을 내가 원하는 목적에 맞게 바꾸려면 어떻게 해야 할까요?

이 프로젝트에서는 작은 언어 모델과 이미지·언어 모델을 직접 다뤄 보면서 **fine-tuning의 원리부터 학습, 저장, 평가까지** 차근차근 익힙니다. 처음에는 쉬운 예제와 작은 계산으로 시작하고, 익숙해지면 학습 코드와 수식도 함께 살펴봅니다.

과정을 마치면 다음과 같은 일을 할 수 있습니다.

- **질문에 답하는 모델 만들기:** 기본 언어 모델에 질문과 답변을 보여 주며 지시를 따르는 법을 가르칩니다.
- **특정 분야에 맞게 모델 바꾸기:** 문서를 읽히는 학습과 질문·답변을 가르치는 학습을 비교합니다.
- **메모리를 아끼며 학습하기:** 전체 학습, LoRA, QLoRA가 어떻게 다르고 언제 쓰면 좋은지 알아봅니다.
- **더 나은 답변을 고르게 하기:** DPO와 RLHF를 통해 선호와 보상을 학습에 활용합니다.
- **정답을 확인하며 학습하기:** 산술 문제의 답을 직접 검산하는 RLVR를 실습합니다.
- **이미지를 이해하도록 학습하기:** 이미지와 텍스트를 함께 다루는 VLM을 특정 이미지 분야에 맞게 학습시킵니다.
- **결과를 제대로 확인하기:** 학습 전후를 비교하고, 저장한 모델을 다시 불러와 실제로 달라진 점을 살펴봅니다.

명령이 오류 없이 끝나는 것만으로는 충분하지 않습니다. **왜 이 방법을 골랐는지, 모델이 무엇을 배웠는지, 결과를 어디까지 믿어도 되는지** 스스로 설명할 수 있게 되는 것이 목표입니다.

설명은 한국어로 제공하지만, 실습에는 작은 모델과 영어 공개 데이터를 사용합니다. 처음부터 큰 모델을 돌리기보다는 부담이 적은 예제로 원리를 익혀 보세요.

## 어떤 순서로 배우나요?

처음에는 아래 순서대로 대표 실습을 하나씩 따라가면 됩니다. 모든 도구와 방법을 한꺼번에 익힐 필요는 없습니다.

| 단계 | 함께 해 볼 일 | 시작할 자료 |
|---|---|---|
| 0. 기초 다지기 | 토큰, 다음 단어 예측, loss, gradient, 데이터 분할 알아보기 | [입문 수업](docs/12_foundations.md) |
| 1. Instruction SFT | 기본 언어 모델에 질문과 답변을 가르치고 학습 루프 읽어 보기 | [PyTorch SFT](01_instruction/pytorch/README.md) |
| 2. LoRA | 작은 adapter만 학습하고 저장한 결과 다시 불러오기 | [HF LoRA](01_instruction/huggingface/README.md) |
| 3. Domain fine-tuning | 과학 문서를 읽히는 학습과 과학 QA 학습 비교하기 | [Domain 과정](04_post_training/domain_huggingface/README.md) |
| 4. DPO | 두 답변 중 더 좋은 답변을 배우게 하기 | [DPO](04_post_training/dpo_huggingface/README.md) |
| 5. Reward modeling | 답변에 점수를 매기는 모델 만들기 | [Reward model](04_post_training/rlhf_ppo_huggingface/README.md) |
| 6. RLHF PPO | 채점 모델이 주는 보상으로 답변 모델 학습시키기 | [PPO](04_post_training/rlhf_ppo_huggingface/README.md) |
| 7. RLVR GRPO | 산술 정답을 검산해 보상을 주고 학습시키기 | [RLVR GRPO](04_post_training/rlvr_grpo_huggingface/README.md) |
| 8. VLM | 이미지를 보고 콩잎 상태를 답하도록 학습시키기 | [PyTorch VLM](03_vision/pytorch/README.md), [HF VLM](03_vision/huggingface/README.md) |
| 9. 비교와 확장 | 도구, 데이터 수, 학습 방법, 모델 크기를 바꿔 보기 | [선택 실습](docs/10_practice_checklist.md), [확장 과제](docs/08_next_experiments.md) |

여기서 한 가지 알아둘 점이 있습니다. 이 표는 **공부하는 순서**이지, 하나의 모델을 모든 단계에 차례로 넣으라는 뜻은 아닙니다.

- **DPO와 PPO**는 같은 SFT 모델에서 출발해 비교할 수 있습니다. PPO를 배우기 전에 DPO 모델을 만들 필요는 없습니다.
- **PPO**를 실행하려면 SFT 모델과 답변을 채점할 reward model이 필요합니다.
- **RLVR GRPO**는 산술 문제를 먼저 학습한 별도 모델에서 시작합니다.
- **Domain 실습**에서는 문서 학습만 한 경우, QA만 학습한 경우, 두 학습을 이어서 한 경우를 비교합니다.
- **VLM 실습**에서는 이미지 입력을 다루는 별도 모델을 사용합니다.

각 단계에 들어가기 전에 무엇을 준비해야 하는지, 언제 다음 단계로 넘어가면 되는지는 [커리큘럼](docs/11_curriculum.md)에서 확인할 수 있습니다.

## 이렇게 공부해 보세요

각 실습은 아래 흐름으로 진행합니다.

1. **설명과 데이터부터 살펴보세요.**
   실습 README를 읽고 데이터 몇 줄을 직접 확인해 보세요. “모델에 무엇을 보여 주고, 어떤 답을 기대하는가?”를 말로 설명할 수 있으면 좋은 출발입니다.

2. **학습하기 전에 준비 상태를 확인하세요.**
   데이터 분할과 파일을 검사하고, `--dry-run`으로 입력 형식과 설정을 확인합니다. Dry-run은 준비 상태를 살펴보는 과정이지, 실제 학습은 아닙니다.

3. **학습 전 답변을 남겨 두세요.**
   나중에 비교할 수 있도록 같은 질문에 대한 답변과 평가 결과를 저장합니다. 이 결과를 baseline이라고 부릅니다.

4. **작게 시작해 보세요.**
   적은 데이터와 짧은 학습으로 전체 흐름을 익힙니다. Loss가 줄었다고 답변도 반드시 좋아지는 것은 아니니, 실제 출력도 함께 읽어 보세요.

5. **저장한 모델을 다시 불러오세요.**
   학습 프로그램을 끝낸 뒤 새 프로세스에서 모델이나 adapter를 읽어 봅니다. 저장한 결과만으로 추론이 되는지 확인하는 과정입니다.

6. **무엇이 달라졌는지 정리하세요.**
   점수뿐 아니라 잘된 답변과 실패한 답변을 비교해 보세요. 나아지지 않았다면 데이터, 학습 설정, 평가 방법 중 어디를 다시 살펴봐야 할지 생각해 봅니다.

처음에는 고등학생 수준의 비유와 작은 숫자 예제로 설명합니다. 뒤로 갈수록 확률, KL, advantage 같은 개념을 다루지만, 처음부터 수식을 모두 이해할 필요는 없습니다. 필요한 단계에서 [이론 설명](docs/01_theory.md)을 함께 읽어 보세요.

학습 설정을 고를 때는 validation 데이터를 사용하고, 마지막 결과를 비교할 때는 test 데이터를 사용합니다. 시험 문제를 미리 보고 공부하지 않는 것과 같은 이유입니다.

## 실습 환경 준비하기

### 공통으로 필요한 것

- **Python 기초:** 함수, 터미널 명령, JSON·JSONL 파일을 읽는 방법
- **Python 버전:** 3.11 또는 3.12
- **개발 도구:** Git과 Python 가상환경
- **PyTorch:** 사용할 장치에 맞는 버전
- **저장 공간:** 모델, 데이터 cache, 실행 결과를 보관할 공간
- **인터넷 연결:** 공개 모델과 데이터를 처음 내려받을 때 필요

명령은 모두 **프로젝트 루트 디렉터리**에서 실행합니다.

필요한 메모리는 모델 크기뿐 아니라 문장 길이, batch size, 학습 방법에 따라서도 달라집니다. 작은 텍스트 실습부터 시작하세요. 이미지 모델과 여러 모델을 함께 사용하는 PPO는 메모리를 더 신중하게 잡아야 합니다.

### 내 컴퓨터에서는 어디까지 할 수 있나요?

| 환경 | 시작하는 방법 | 알아둘 점 |
|---|---|---|
| Linux + NVIDIA CUDA | 전체 실습을 진행할 기준 환경 | CUDA와 라이브러리의 호환성을 확인해야 합니다. 현재 post-training 코드는 CUDA BF16을 사용합니다. |
| DGX Spark | 전용 설치 안내 따라가기 | 제공된 CUDA 런타임을 유지하면서 실습 패키지를 설치합니다. |
| Apple Silicon Mac | CPU 또는 MPS로 기본 텍스트 실습 시작하기 | MPS를 선택할 수 있지만, 모든 실습이 실행된다고 보장하지는 않습니다. |
| Intel Mac | 문서, 코드, 데이터 구조부터 공부하기 | 공식 PyTorch 버전 제한 때문에 현재 학습 환경을 그대로 사용하기 어렵습니다. |

### Linux와 CUDA 환경

먼저 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 자신의 장치에 맞는 설치 명령을 선택하세요.

```bash
python3.12 -m venv .venv-lab
source .venv-lab/bin/activate

# 장치에 맞는 PyTorch를 먼저 설치하세요.
python -m pip install -r requirements/lab-hf.txt
python scripts/doctor.py --profile hf
```

DPO, reward modeling, PPO, GRPO를 시작할 때는 다음 패키지를 더 설치합니다.

```bash
python -m pip install -r requirements/lab-post.txt
python scripts/doctor.py --profile post --require-cuda
```

Unsloth는 별도 가상환경에 설치하세요. 필요한 PyTorch 버전이 다를 수 있으니, 이미 잘 동작하는 환경에 그대로 덮어 설치하지 않는 편이 안전합니다.

### DGX Spark

DGX Spark를 사용한다면 [전용 설치 안내](docs/04_dgx_spark.md)를 따라 준비하세요. 컨테이너 실행부터 패키지 설치, 장치 검사까지 안내합니다.

- HF, Unsloth, post-training 환경을 구분해서 사용합니다.
- 제공된 torch·torchvision·Triton을 다른 패키지로 덮어쓰지 않습니다.
- 환경 검사와 실습별 준비 조건을 확인한 뒤 학습을 시작합니다.

### Apple Silicon Mac

Apple Silicon에서는 PyTorch의 **MPS backend**를 통해 GPU를 사용할 수 있습니다. 설치할 PyTorch 버전에 맞춰 macOS 요구사항도 확인하세요. 현재 Apple 안내는 macOS 14 이상을 기준으로 합니다. [Apple의 PyTorch 안내](https://developer.apple.com/metal/pytorch/)

기본 설치 순서는 다음과 같습니다.

```bash
python3.12 -m venv .venv-lab
source .venv-lab/bin/activate

python -m pip install torch
python -m pip install -r requirements/lab-hf.txt
python scripts/doctor.py --profile hf
```

실습할 때는 다음을 기억해 주세요.

- 기본 텍스트 실습에서는 `--device mps` 또는 `--device cpu`를 선택할 수 있습니다.
- MPS의 기본 텍스트 경로는 `--dtype float32`로 시작합니다.
- CUDA용 설치 명령이나 `--require-cuda` 검사는 사용하지 않습니다.
- 현재 CUDA를 사용하는 Unsloth·QLoRA 경로와 DPO·PPO·GRPO 학습을 Mac에서 그대로 실행할 수 있다고 생각하면 안 됩니다.
- 모델마다 MPS에서 지원하는 연산과 필요한 메모리가 다를 수 있습니다.

여기서는 설치와 실행 경로를 안내합니다. Mac에서 전체 실습의 동작을 확인했다는 뜻은 아닙니다.

### Intel Mac: PyTorch 버전 제한을 먼저 알아두세요

Intel Mac에서는 특히 PyTorch 버전을 주의해야 합니다.

**공식 macOS x86_64 패키지는 PyTorch 2.2.x까지만 제공되며, 마지막 패치 버전은 2.2.2입니다. PyTorch 2.3부터는 Intel Mac용 공식 바이너리가 제공되지 않습니다.** Python이나 pip를 업데이트해도 이 제한이 없어지지는 않습니다. [PyTorch 지원 중단 공지](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690), [2.2.2 배포 파일](https://pypi.org/project/torch/2.2.2/)

그래서 다음과 같은 어려움이 있습니다.

- PyTorch 2.2.2를 설치하더라도 이 프로젝트의 HF·PEFT·TRL 조합이 함께 동작한다고 보장할 수 없습니다.
- CUDA를 사용할 수 없고, Intel Mac이라고 해서 MPS를 사용할 수 있는 것도 아닙니다.
- 소스 빌드나 비공식 패키지는 별도의 관리와 검증이 필요합니다. 이 과정의 기본 설치 방법으로는 다루지 않습니다.

Intel Mac을 사용한다면 **문서와 코드를 읽고 데이터 구조를 익히는 데 활용하고, 실제 학습은 호환되는 Linux CUDA 환경에서 진행하세요.** 이 프로젝트에서는 Intel Mac의 라이브러리 import, pytest, 학습 실행을 검증 절차로 요구하지 않습니다.

### 데이터 준비와 첫 실습

이미 준비된 `data/`를 가지고 있다면 다시 내려받을 필요는 없습니다. 사용할 실습 환경에서 데이터와 설정부터 확인하세요.

```bash
python scripts/check_data.py
python 01_instruction/pytorch/train.py --dry-run
```

데이터가 없다면 [데이터 안내](docs/02_datasets.md)와 [첫 실행 안내](docs/00_quickstart.md)의 준비 방법을 따라가면 됩니다.

준비가 끝나면 [첫 PyTorch 실습](01_instruction/pytorch/README.md)을 열고 학습 전 평가부터 시작하세요. 학습 명령은 모델의 가중치를 실제로 바꿉니다. 환경 확인과 구분해서 실행하고, 결과는 매번 새로운 경로에 저장해 주세요.

## 어떤 모델과 데이터를 사용하나요?

| 배우려는 내용 | 기본 모델 | 데이터 |
|---|---|---|
| 지시를 따르는 답변 | SmolLM2-135M base | Dolly |
| 과학 분야 적응 | SmolLM2-135M-Instruct | SciQ 문서와 QA |
| 답변 선호 학습 | 실습에서 준비한 SFT 모델 | 직접 작성한 선호 답변·비선호 답변 쌍 |
| 정답 검산을 통한 학습 | 산술 SFT 출발 모델 | 직접 작성한 산술 예제 |
| 이미지 이해 | SmolVLM-256M-Instruct | Beans |
| Unsloth VLM 실습 | Qwen2.5-VL-3B-Instruct | 같은 Beans 분할 |

데이터를 사용할 때는 결과를 어디까지 해석할 수 있는지도 함께 생각해 보세요.

- 직접 작성한 선호·산술 예제는 원리를 배우기 위한 자료입니다. 실제 사람의 피드백이나 일반적인 추론 능력을 평가하는 자료는 아닙니다.
- 영어 모델의 실험 결과를 한국어 능력 향상으로 해석하지 않습니다.
- VLM 실습은 도구에 따라 사용하는 모델이 다릅니다. 실행 시간이 다르다고 도구 자체의 속도 차이라고 단정하지 않습니다.

출처, 라이선스, 데이터 분할은 [데이터 안내](docs/02_datasets.md)에서 자세히 확인할 수 있습니다.

## 도구와 학습 방법은 어떻게 다른가요?

처음에는 이름이 많아 헷갈릴 수 있습니다. 다음 세 가지로 나누어 생각하면 편합니다.

- **어떤 도구로 구현할까?**
  PyTorch, Hugging Face, Unsloth
- **무엇을 어떻게 배우게 할까?**
  Instruction SFT, CPT, QA SFT, DPO, RLHF, RLVR
- **모델의 어느 가중치를 바꿀까?**
  전체 학습, LoRA, QLoRA

먼저 대표 경로로 전체 흐름을 익힌 뒤, 같은 데이터와 평가 조건으로 다른 도구를 비교해 보세요.

Post-training은 Hugging Face Transformers·PEFT·TRL 경로로 배웁니다. Domain 과정에서는 앞서 익힌 CPT와 SFT 코드를 다시 사용합니다. 전체 선택 실습은 [체크리스트](docs/10_practice_checklist.md)에 정리되어 있습니다.

## 필요한 파일은 어디에 있나요?

### 실습 수업

| 디렉터리 | 여기서 하는 일 |
|---|---|
| `01_instruction/` | 기본 언어 모델에 지시 따르기 가르치기 |
| `02_knowledge/` | 문서 학습, QA 학습, 두 학습 이어 보기 |
| `03_vision/` | 이미지와 텍스트를 함께 다루는 모델 학습하기 |
| `04_post_training/` | Domain, DPO, reward modeling·PPO, RLVR GRPO 배우기 |

Instruction·지식·VLM 디렉터리에는 PyTorch, Hugging Face, Unsloth별 설명과 실행 파일이 있습니다.

디렉터리 번호와 공부 순서는 조금 다릅니다. 처음에는 커리큘럼 표를 따라가고, 익숙해진 뒤 관심 있는 방법을 골라 보세요.

### 설명과 공통 코드

- `docs/`: 기초 설명, 이론, 데이터, 평가, 환경 설정과 문제 해결
- `finetune_lab/`: 여러 실습에서 함께 사용하는 데이터·학습·평가 코드
- `requirements/`, `scripts/`: 패키지 목록과 설치·검사 도구
- `tests/`: 데이터 형식과 코드 동작을 확인하는 테스트

### 데이터와 실행 결과

- `data/demo/`: 작은 자체 작성 예제
- `data/processed/`: 전처리한 공개 데이터와 데이터 기록 파일인 manifest
- `outputs/`: 학습한 모델, adapter, 평가 보고서

다른 컴퓨터로 옮길 때는 코드와 데이터를 복사하세요. 가상환경은 복사하지 말고, 새 컴퓨터에서 다시 만들어 설치합니다.

## 막히거나 더 궁금할 때

- **처음 실행하는 방법:** [첫 실행 안내](docs/00_quickstart.md)
- **다음에 공부할 내용:** [커리큘럼](docs/11_curriculum.md)
- **실습을 마쳤는지 확인하기:** [체크리스트](docs/10_practice_checklist.md)
- **결과를 평가하는 방법:** [평가 안내](docs/03_evaluation.md)
- **오류와 메모리 문제 해결:** [문제 해결](docs/05_troubleshooting.md)
- **확인된 범위와 아직 확인하지 않은 부분:** [검증 범위](docs/13_readiness.md)
