# 기존 LLM에 추가 지식 학습시키기 — Unsloth CPT → SFT

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

HF의 C4/S4/K3를 묶은 [Domain 주 수업](../../04_post_training/domain_huggingface/README.md)에서 시작합니다. CPT만/QA만/CPT→QA를 비교한 뒤 [DPO 수업](../../04_post_training/dpo_huggingface/README.md)으로 이동합니다. 다른 도구/기법은 선택 비교입니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, 방법에 맞는 실행 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 실습은 이미 instruction tuning을 받은 `HuggingFaceTB/SmolLM2-135M-Instruct`에 과학 분야 텍스트와 질문·정답을 추가로 학습한다. 동일한 데이터에서 **CPT만**, **SFT만**, **CPT 다음 SFT**를 비교하여 각 학습 목표의 차이를 이해한다. 기본 모델은 작고 데이터도 일부만 사용하므로 품질을 보장하는 결과가 아니라 실습용 파이프라인이다.

## 1. 지식 학습의 두 목표

### Continued pretraining(CPT)

문서의 토큰을 `z₁,…,zₙ`이라 하면 문서 전체에서 다음 토큰을 예측하도록 학습한다.

```text
L_CPT = - (1 / n) × Σ log Pθ(z_t | z_<t)
```

예를 들어 지구 과학 문서의 문장을 계속 읽게 하면 해당 분야의 용어, 관계, 문장 패턴에 적응할 수 있다. 질문과 답변의 구조가 없어도 된다. 실제 문서 내용과 EOS는 loss 대상이고 padding은 제외한다. 지식이 가중치에 분산되어 저장되기 때문에 특정 사실을 반드시 회상하거나 원문을 그대로 인용한다는 보장은 없다. [Unsloth continued pretraining 문서](https://unsloth.ai/docs/basics/continued-pretraining)를 참고한다.

### Domain SFT

질문 `x`에 대해 정답 답변 `y`를 내도록 학습한다.

```text
L_SFT = - (1 / |y|) × Σ log Pθ(y_t | x, y_<t)
```

문서를 알아도 질문에 필요한 사실을 답하는 능력이 바로 생기지는 않는다. Domain SFT는 질문으로 지식을 꺼내어 답하는 형식을 학습한다. 여기서는 질문을 mask하고 답변과 EOS만 loss 대상으로 둔다. QA 학습이 정답 문자열을 암기할 수 있으므로 학습에 쓰지 않은 질문과 paraphrase로도 평가해야 한다.

### 두 단계를 이어 쓰는 이유와 한계

CPT→SFT는 원문 분포에 적응한 모델을 질문 응답 작업에 맞추는 구성이다. Instruct 모델에 CPT만 수행하면 원래 지시 수행 능력이 약해질 수 있다(catastrophic forgetting). 학습 전후에 기존의 일반 지시 평가도 함께 측정해야 한다. 데이터가 작거나 이미 알려진 내용이면 CPT의 이점이 없거나 SFT만 하는 편이 나을 수 있다. 20 step 손실의 하락으로 새로운 지식을 획득했다고 결론 내리지 않는다.

추가 지식 제공에는 RAG도 있다. 문서가 자주 바뀌거나 출처 인용이 필요하면 검색으로 원문을 넣는 편이 적합할 수 있다. Fine-tuning은 응답 습관과 도메인 적응에 유용하며 RAG와 함께 사용할 수 있다. 이번 실습은 가중치를 업데이트하는 경로를 이해하기 위한 것이다.

## 2. Unsloth와 LoRA가 학습하는 부분

Base 가중치를 고정하고 다음 업데이트를 학습한다.

```text
W' = W + (α / r) × B × A
```

Attention과 MLP의 projection에 rank=16 adapter를 붙인다. 기본은 `--load-in-4bit` QLoRA이고 `--no-load-in-4bit`는 일반 LoRA이다. 4bit는 frozen base 저장 형식이며 모든 연산을 4bit로 계산하는 뜻이 아니다. 기본 CPT도 projection LoRA만 바꾸므로 vocabulary나 embedding을 새로 추가하는 실험은 아니다. 새로운 토큰 추가와 embedding/lm_head 학습은 별도의 tokenizer 변경, trainable parameter 설정, 보통 더 작은 embedding 학습률이 필요하다.

Unsloth는 모델과 커널을 최적화하고 TRL `SFTTrainer`가 학습 루프를 실행한다. 여기서 SFTTrainer라는 이름이 붙었어도 `text`의 모든 실제 토큰에 loss를 적용하면 raw text CPT objective로 쓸 수 있다. 공통 encoding으로 미리 만든 `labels`를 넘겨 두 단계의 mask를 구분한다. `skip_prepare_dataset=True`로 TRL이 다시 텍스트를 포맷하지 않게 한다. [TRL 0.24 SFTTrainer 문서](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer)를 참고한다.

## 3. 과학 데이터 구조와 분할

원본은 영어 과학 QA 데이터인 [allenai/sciq](https://huggingface.co/datasets/allenai/sciq)다. 원본 split은 train 11,679, validation 1,000, test 1,000이며 `question`, `correct_answer`, 세 개의 `distractor`, 근거 문단 `support`를 포함한다. 라이선스는 **CC BY-NC 3.0**으로 비상업 조건이 있다. 실무 상업 데이터에 그대로 사용하는 자료로 가정하지 않는다.

이 실습은 객관식 선택지를 생성하지 않고 다음과 같이 변환한다.

| 경로 | 입력 | 정답/손실 |
|---|---|---|
| `knowledge_cpt` | 비어 있지 않은 `support` 문단 | 문단 전체 + EOS |
| `knowledge_sft` | `question` | `correct_answer` + EOS |

다음은 형식 이해를 위해 직접 작성한 예다.

```json
{"id":"science-example-1","text":"Water freezes at zero degrees Celsius under standard conditions."}
```

```json
{"id":"science-example-1","prompt":"At what Celsius temperature does water freeze under standard conditions?","completion":"zero degrees Celsius"}
```

**원본 예제의 split을 먼저 정한 다음** CPT 문단과 SFT 질문으로 변환한다. Test 질문의 근거 문단을 CPT train에 넣으면 답변 사실을 먼저 보여준 뒤 평가하게 되므로 test 의미가 훼손된다. 공통 준비 코드는 원본 split을 유지하며 정규화된 동일 질문/문단의 split 간 중복을 제거한다. `support`가 빈 원본 예제는 CPT에서 제외되어 CPT와 SFT의 수가 다를 수 있다. `manifest.json`의 실제 count와 revision·SHA256을 확인한다.

이 구조는 **학습하지 않은 사실에 대한 도메인 일반화** 평가다. 특정 문서로 새 사실을 익혔는지 보려면 별도로 학습 문서의 사실에 대해 처음 보는 표현의 질문을 만든다. 그 결과는 문서 적응/회상 평가로 보고하고, 완전히 미학습 문서의 test 점수와 구분한다. SciQ의 사실은 일반 상식일 수 있고 base 사전학습에 포함됐을 수 있으므로 실제로 처음 보는 지식이라는 보장은 없다.

## 4. 실행 환경과 작은 데이터 확인

Unsloth는 지원되는 Linux NVIDIA CUDA 환경에서 별도 가상환경으로 설치한다. [공식 설치 안내](https://unsloth.ai/docs/get-started/install-and-update)에 따라 GPU·PyTorch·Triton·bitsandbytes 조합을 확인한다. 본 수업의 학습 코드 기준은 TRL 0.24.0과 Transformers 4.57.6이므로 다른 버전의 예제를 섞지 않는다. DGX Spark 사용자는 [전용 환경 안내](../../docs/04_dgx_spark.md)의 NGC overlay 절차를 선택한다.

```bash
# DGX Spark 호스트의 저장소 루트
bash scripts/spark_container.sh unsloth
# 다음 명령은 컨테이너 안의 저장소 루트에서 실행
bash scripts/install_spark.sh unsloth
source .venv-spark-unsloth/bin/activate
```

Spark를 선택하면 NGC의 PyTorch/CUDA/Triton을 유지하고 `requirements/spark-unsloth.txt` overlay를 사용한다. 일반 환경에서는 공식 지원 조합을 별도 확인한다. SmolLM2 135M에서는 양자화 준비 비용이 이점보다 클 수 있으므로 4bit와 16bit LoRA의 시간·메모리를 실제로 비교한다.

### 다운로드 없는 준비 점검

```bash
python -m finetune_lab.prepare_data --task knowledge --source demo --output-root data/demo
python 02_knowledge/unsloth/train.py --stage cpt --data-dir data/demo/knowledge_cpt --dry-run
python 02_knowledge/unsloth/train.py --stage sft --data-dir data/demo/knowledge_sft --dry-run
```

Demo는 가상의 우주 정거장 코드로 만든 fixture다. 실행과 누수 점검용이며 모델 성능 지표로 쓰지 않는다. 어떤 환경에서도 dry-run은 train/validation/test 전체를 읽어 ID·내용 누수를 검증하고, 모델 가중치나 Unsloth는 로딩하지 않는다. Test는 누수 확인에만 읽으며 학습과 validation loss에 사용하지 않는다. Dry-run으로 GPU 커널 지원을 판정하지 않는다.

### 실제 SciQ 데이터 준비

```bash
python -m finetune_lab.prepare_data --task knowledge --source hf --train-samples 256 --eval-samples 32
python 02_knowledge/unsloth/train.py --stage cpt --dry-run
python 02_knowledge/unsloth/train.py --stage sft --dry-run
```

`--stage cpt`의 기본 data-dir은 `data/processed/knowledge_cpt`, `--stage sft`의 기본값은 `data/processed/knowledge_sft`다. 준비된 데이터는 다시 덮어쓰지 않고 재사용한다. 의도적으로 바꾸면 새 `--output-root`를 사용하거나 `--overwrite`를 지정한다.

## 5. 세 실험을 공정하게 비교하기

### 5-1. 출발 모델 기록

```bash
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --data-dir data/processed/knowledge_cpt --kind cpt
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --data-dir data/processed/knowledge_sft --kind sft
```

CPT에는 문단의 token loss/perplexity가, SFT에는 질문 응답 loss와 생성 답변 비교가 의미 있다. Perplexity는 `exp(평균 token loss)`지만 tokenizer, loss mask와 평가 텍스트가 같을 때 비교해야 한다. 두 단계는 loss 대상이 달라 CPT loss와 SFT loss를 직접 비교하지 않는다.

### 5-2. A: CPT만 하기

```bash
python 02_knowledge/unsloth/train.py \
  --stage cpt --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-dir data/processed/knowledge_cpt \
  --output-dir outputs/knowledge_unsloth_cpt \
  --max-steps 20 --max-length 256 \
  --batch-size 1 --gradient-accumulation 4 \
  --learning-rate 0.00005 --load-in-4bit
python -m finetune_lab.evaluate --model outputs/knowledge_unsloth_cpt --data-dir data/processed/knowledge_cpt --kind cpt
python -m finetune_lab.evaluate --model outputs/knowledge_unsloth_cpt --data-dir data/processed/knowledge_sft --kind sft
```

CPT에서는 기존 능력의 변화를 제한하려고 SFT 예제보다 작은 learning rate로 시작한다. 고정 정답은 아니므로 validation으로 조정한다. 검증 손실은 train 전후에 측정되어 `metrics.json`에 저장된다. Model loss가 NaN/Inf이면 backward 전에 중단하며, 초기 검증과 최종 지표의 NaN/Inf도 성공 결과로 저장하지 않는다. 긴 문단은 기본 토큰 길이에서 오른쪽을 자르고 EOS를 남긴다. 실제 문서에서는 길이를 늘리거나 문서 단위 split을 유지한 채 chunking을 추가한다.

### 5-3. B: 처음 Instruct 모델에서 SFT만 하기

```bash
python 02_knowledge/unsloth/train.py \
  --stage sft --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-dir data/processed/knowledge_sft \
  --output-dir outputs/knowledge_unsloth_sft_only \
  --max-steps 20 --max-length 256 --learning-rate 0.0002 --load-in-4bit
python -m finetune_lab.evaluate --model outputs/knowledge_unsloth_sft_only --data-dir data/processed/knowledge_sft --kind sft
```

이 실습은 기본 Instruct chat template 대신 세 방법이 공유하는 다음 포맷을 사용한다. 질의 평가도 같은 포맷을 사용한다.

```text
### Instruction:
{question}

### Response:
{correct_answer}<EOS>
```

지시 부분의 labels는 `-100`이다. 질문과 답변은 독립적으로 토큰화하여 경계를 확정하고, 패딩 위치도 `-100`으로 처리한다. PAD와 EOS의 ID가 같아도 실제 EOS의 정답 토큰을 지우지 않는다. 이 포맷 적응 자체가 결과에 영향을 주므로 필요하면 초기 모델의 native chat template 평가도 별도로 기록한다.

### 5-4. C: CPT adapter를 이어서 SFT 하기

```bash
python 02_knowledge/unsloth/train.py \
  --stage sft --model outputs/knowledge_unsloth_cpt \
  --data-dir data/processed/knowledge_sft \
  --output-dir outputs/knowledge_unsloth_cpt_sft \
  --max-steps 20 --max-length 256 --learning-rate 0.0002 --load-in-4bit
python -m finetune_lab.evaluate --model outputs/knowledge_unsloth_cpt_sft --data-dir data/processed/knowledge_sft --kind sft
```

로컬 `adapter_config.json`이 있으면 기존 adapter를 로딩하고 `get_peft_model()`을 다시 호출하지 않는다. 즉 **CPT adapter 가중치가 이어서 SFT로 갱신된다.** 기존 base 모델 ID와 revision은 메타데이터에 유지한다. 새로운 optimizer로 시작하므로 이것은 가중치의 계속 학습이며 optimizer 상태까지 복구하는 checkpoint resume는 아니다. Rank와 target module은 기존 adapter 설정을 유지하므로 재시작 명령의 `--lora-r`·`--lora-alpha`로 기존 adapter의 구조를 바꾸지 않는다. [Unsloth의 adapter 계속 학습 안내](https://unsloth.ai/docs/basics/continued-pretraining)를 참고한다.

일반적인 Hugging Face 로딩 경로로 전환하려면 CPT adapter를 merge한 모델에서 새 SFT adapter를 만들 수도 있다. 이때 SFT adapter가 요구하는 base는 원래 Instruct 모델이 아니라 **CPT가 merge된 모델**이다. 해당 디렉토리를 함께 보관해야 한다.

```bash
python -m finetune_lab.merge --adapter outputs/knowledge_unsloth_cpt --output-dir outputs/knowledge_unsloth_cpt_merged
python 02_knowledge/unsloth/train.py \
  --stage sft --model outputs/knowledge_unsloth_cpt_merged \
  --output-dir outputs/knowledge_unsloth_merged_then_sft \
  --max-steps 20 --max-length 256 --learning-rate 0.0002
```

Merge는 base 전체를 쓰기 때문에 adapter만 저장할 때보다 디스크·메모리가 많이 필요하다. 일반 LoRA 비교에는 각 실험의 `--no-load-in-4bit`와 새 출력 경로를 사용한다.

## 6. 결과 읽기와 발전시키기

각 결과는 adapter 가중치, tokenizer, `training_metadata.json`, `metrics.json`으로 남는다. Base 모델의 가중치가 있어야 adapter가 동작한다. 저장 뒤 공통 evaluator로 **파일을 다시 로딩하여** 비교한다. 동일 test 데이터와 greedy decoding 조건을 유지한다. 어떤 질문은 동의어로 맞는 답을 해도 exact match가 낮을 수 있으므로 생성 예시도 검토한다.

| 비교 | 알 수 있는 것 |
|---|---|
| 초기 모델 → CPT | 도메인 문단 예측 변화, 지시 응답 보존 여부 |
| 초기 모델 → SFT only | QA 작업 적응의 효과 |
| SFT only → CPT+SFT | 원문 적응이 추가로 기여하는지 |
| 모든 결과 → 별도 일반 지시 질문 | 원래 능력의 망각 여부 |

CPT+SFT는 B보다 총 업데이트 수가 더 많다. 단계의 효과를 공정하게 비교하려면 B의 SFT step을 늘려 총 token 수나 optimizer step·시간 예산을 맞추는 추가 실험도 해야 한다. 답변 길이, 학습률, rank를 함께 바꾸면 원인을 판단하기 어렵다. 먼저 하나씩 조정한다.

Validation이 악화되면 step/learning rate/rank를 줄이고, train loss만 낮다면 과적합을 의심한다. 최신 사실을 학습할 때에는 출처·시점·권한을 기록하고, 개인정보나 비밀을 넣지 않는다. 사용자 문서를 사용할 때에는 문서 단위로 split하고 같은 문서의 chunk가 train/test 양쪽에 들어가지 않도록 해야 한다. RAG, 문서 기반 회상 평가, 일반 지시 능력 평가를 함께 두면 추가 지식 학습의 실제 가치를 판단하기 쉽다.

재현 실험에서는 모델·데이터의 `main` 대신 commit SHA를 `--revision`에 지정하고 설치 패키지도 보관한다.

```bash
mkdir -p outputs
python -m pip freeze > outputs/unsloth_environment.txt
```

## 수업 완료 기준

1. 준비: 세 split과 manifest를 확인하고 dry-run의 데이터 수·모델·학습 방법을 설명합니다.
2. 실행: 실제 학습이 유한 loss로 종료되고 예상한 전체 모델 또는 adapter·tokenizer·metadata가 새 출력 경로에 저장됩니다.
3. 재사용: 별도 프로세스에서 저장 결과를 읽어 답변을 생성합니다. Adapter이면 동일 base와 revision을 사용합니다.
4. 해석: CPT 문서 loss와 QA 정답률을 구분하고, 학습한 사실 회상·새 사실·기존 instruction 보존을 각각 해석합니다. 설정을 고른 뒤 test를 최종 평가합니다.

실행 성공과 품질 개선은 각각 기록합니다. 수업을 준비했거나 dry-run만 통과한 상태를 학습 완료로 표시하지 않습니다. Unsloth 경로는 선택 확장이므로 지원 환경이 없으면 미실행으로 남기고 주 경로를 진행합니다.다음 단계는 이 문서 첫머리의 수업 경로와 [커리큘럼](../../docs/11_curriculum.md)을 따릅니다.
