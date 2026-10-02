# Base LLM에 instruction learning 추가하기 — Unsloth LoRA / QLoRA

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

주 경로는 PyTorch full(I1) → HF LoRA(I4) → [Domain 수업](../../04_post_training/domain_huggingface/README.md)입니다. 다른 full/QLoRA/Unsloth 조합은 주 경로를 마친 뒤 선택 비교합니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, 방법에 맞는 실행 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 실습은 `HuggingFaceTB/SmolLM2-135M` **base 모델**에 Dolly의 지시와 답변을 학습시킨다. Base 모델은 다음 텍스트를 이어 쓰도록 사전학습됐지만, 사용자 지시를 수행하는 답변 형식을 충분히 배우지 않았다. 같은 질문을 학습 전후에 넣고, 답변 형식과 검증 손실이 어떻게 바뀌는지 관찰하는 것이 목표다.

135M 모델과 기본 `20` optimizer step은 학습 과정 이해를 위한 작은 실험이다. 이 설정으로 범용 비서 수준의 성능을 기대하지 않는다. 사용자가 지시를 따른다고 느끼는 변화와 새로운 지식의 정확성은 서로 다른 평가 항목이다. 모델의 base/Instruct 구분과 Apache-2.0 라이선스는 [SmolLM2 모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)에서 확인할 수 있다.

## 1. 이론: 무엇이 바뀌는가

### SFT의 목적 함수

지시를 `x`, 정답 답변을 `y=(y₁,…,yₙ)`이라고 하면 supervised fine-tuning(SFT)은 다음 손실을 줄인다.

```text
L_SFT = - (1 / n) × Σ log Pθ(y_t | x, y_<t)
```

학습 시에는 지시와 정답을 함께 모델에 입력한다. 이것은 모델에게 정답을 미리 보여주는 부정행위가 아니다. Causal attention이 미래 토큰을 가리고, causal LM loss가 한 토큰만큼 이동하여 앞부분으로 다음 토큰을 예측한다. 이 방식을 teacher forcing이라고 한다. 추론할 때에는 정답이 없으므로 생성한 토큰을 다음 입력으로 사용한다.

이 실습은 지시 토큰의 `labels`를 `-100`으로 만들고 **답변과 EOS에만 손실**을 계산한다. 지시 내용을 그대로 재현하는 데 학습 용량을 쓰지 않고, 주어진 지시에 대응하는 답변을 배우게 한다. EOS는 답변의 종료를 나타내며 실제 EOS와 패딩을 구분해야 한다.

### LoRA와 QLoRA

Full fine-tuning은 기존 가중치 대부분을 업데이트한다. LoRA는 기존 가중치 `W`를 고정하고, 작은 행렬 두 개로 업데이트를 표현한다.

```text
W' = W + (α / r) × B × A
A: r × input_dim, B: output_dim × r
```

`r`은 업데이트의 rank이고 `α/r`은 크기를 조절한다. 이 코드의 기본값은 `r=16`, `α=16`이다. Attention의 `q/k/v/o_proj`와 MLP의 `gate/up/down_proj`에 adapter를 붙인다. 모델의 다른 구조로 바꾸면 해당 모듈이 존재하는지부터 확인해야 한다.

QLoRA는 고정된 base 가중치를 4bit로 저장하고 LoRA 가중치를 학습한다. 역전파를 모두 4bit로 하는 방식은 아니다. 활성값, 학습 가중치와 연산에는 별도의 정밀도가 쓰인다. `--load-in-4bit`가 기본이며 `--no-load-in-4bit`로 일반 LoRA를 비교할 수 있다. 135M처럼 작은 모델은 양자화 준비와 커널 실행 비용 때문에 4bit가 더 빠르지 않을 수 있다. 먼저 실행 성공과 평가를 확인하고 VRAM·시간은 직접 측정한다.

### Unsloth가 맡는 부분

이 실습의 역할 분담은 다음과 같다.

| 요소 | 역할 |
|---|---|
| Unsloth `FastLanguageModel` | 모델 로딩, LoRA 부착, 지원 아키텍처의 학습 최적화 |
| TRL `SFTTrainer` | 학습/검증 실행, 로그와 optimizer 관리 |
| 공통 `text_encoding.py` | 세 방법이 공유하는 토큰화, 정답 마스킹, 패딩 |
| Hugging Face datasets/Hub | 데이터와 공개 모델 읽기 |

Unsloth도 PyTorch·Transformers·PEFT 위에서 동작한다. 따라서 세 방법은 다른 학습 이론이라기보다 구현과 최적화 수준의 차이다. 라이브러리 패치가 적용되도록 코드에서 `unsloth`를 **Transformers와 TRL보다 먼저 import**한다. [TRL 0.24의 Unsloth 통합 문서](https://huggingface.co/docs/trl/v0.24.0/en/unsloth_integration)를 참고한다.

## 2. 데이터와 토큰 형식

원본은 [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)이다. 영어 지시·답변 약 15,000개를 제공하고 요약, 정보 추출, 분류, 질의응답 등을 포함한다. 원본 라이선스는 **CC BY-SA 3.0**이며 저작자 표시와 라이선스 조건을 확인하고 사용한다. 학습 데이터의 라이선스와 모델의 라이선스는 별도로 기록한다.

원본 필드의 변환은 다음과 같다.

| 원본 | 실습 JSONL |
|---|---|
| `instruction` | `prompt`의 첫 부분 |
| `context` | 비어 있지 않으면 `prompt`에 `Context:`와 함께 추가 |
| `response` | `completion` |
| `category` | 분석을 위해 유지 |

다음은 형식을 설명하기 위해 직접 작성한 예시다.

```json
{"id":"example-1","prompt":"Return the number 12 with no extra words.","completion":"12"}
```

공통 포맷은 다음과 같다. Base 모델에는 chat template가 없을 수 있어 명시적인 문자열 포맷을 사용한다.

```text
### Instruction:
Return the number 12 with no extra words.

### Response:
12<EOS>
```

학습과 평가에서 반드시 같은 `format_prompt()`를 사용한다. 학습한 뒤 임의의 chat template로 바꾸면 입력 분포가 달라진다. `prompt_format=instruction_response_v1`은 출력 메타데이터에도 저장된다.

`train.jsonl`, `validation.jsonl`, `test.jsonl`은 분리된다. 검증은 설정 선택과 학습 전후 손실 비교에 쓰고, test는 최종 비교에만 쓴다. 준비 과정은 중복을 제거하고 seed로 나눈 뒤 데이터 revision과 파일 SHA256을 `manifest.json`에 기록한다. Base 모델이 Dolly와 비슷한 인터넷 자료를 사전학습에서 보았는지는 이 분할만으로 확인할 수 없다.

`max-length=256`은 **토큰 수**이다. 긴 지시는 왼쪽을 자르고 답변은 오른쪽을 자르되 최소한의 지시 문맥·답변·EOS를 남긴다. 이는 전부 마스킹된 예제를 피하지만 긴 context의 중요한 정보를 잃을 수 있다. 실제 실험에서는 길이를 늘리거나 예제를 필터링하고 잘린 비율을 확인한다. 답변 마스크가 이미 있는 pretokenized dataset을 넘기므로 `skip_prepare_dataset=True`, custom collator, `packing=False`를 사용한다. [TRL 0.24 SFTTrainer 문서](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer)의 pretokenized dataset과 completion loss 항목을 참고한다.

## 3. 환경 준비

Unsloth는 지원되는 Linux NVIDIA CUDA 환경에서 별도 가상환경으로 설치한다. [공식 설치 안내](https://unsloth.ai/docs/get-started/install-and-update)에 따라 GPU·PyTorch·Triton·bitsandbytes 조합을 확인한다. 본 수업의 학습 코드 기준은 TRL 0.24.0과 Transformers 4.57.6이므로 다른 버전의 예제를 섞지 않는다. DGX Spark 사용자는 [전용 환경 안내](../../docs/04_dgx_spark.md)의 NGC overlay 절차를 선택한다.

아래 Spark 설치 예시는 선택 환경이다. 일반 CUDA 환경은 공식 Unsloth 설치 절차를 사용하고 PyTorch/HF 환경과 분리한다.

```bash
# DGX Spark 호스트에서 컨테이너 진입
bash scripts/spark_container.sh unsloth
# 다음부터는 컨테이너 안의 저장소 루트에서 실행
bash scripts/install_spark.sh unsloth
source .venv-spark-unsloth/bin/activate
```

패키지 기준은 `unsloth==2026.9.14`, `trl==0.24.0`, `transformers==4.57.6`, `datasets==4.3.0`, `peft==0.18.1`, `bitsandbytes==0.48.2`다. Spark GPU 커널을 포함한 실행 검증은 해당 장비에서 해야 한다. ARM64 bitsandbytes wheel과 Blackwell 지원 확인 과정은 공통 Spark 가이드에 설명되어 있다.

GPU 메모리는 가중치뿐 아니라 활성값, optimizer, sequence length, batch size에 영향을 받는다. Spark의 통합 메모리 용량 전체를 모델 가중치로 사용할 수 있다고 계산하지 않는다. 이 작은 모델에서 batch=1, 짧은 sequence, 20 step으로 먼저 커널 실행과 사용량을 확인하고 큰 모델로 확장한다.

## 4. 단계별 실습

### 4-1. 다운로드 없이 실행 계획 확인

```bash
python -m finetune_lab.prepare_data --task instruction --source demo --output-root data/demo
python 01_instruction/unsloth/train.py --data-dir data/demo/instruction --dry-run
```

demo는 직접 만든 숫자 지시 fixture이며 실제 성능 데이터가 아니다. 어떤 환경에서도 dry-run은 JSONL 스키마와 train/validation/test split의 존재·중복을 확인하고 모델·출력 경로·학습 종류를 JSON으로 보여준다. Unsloth, CUDA 패키지, 모델 가중치는 로딩하지 않고 출력 디렉토리도 생성하지 않는다. Dry-run 통과는 GPU 커널 검증을 뜻하지 않는다.

### 4-2. 실제 Hugging Face 데이터 준비

```bash
python -m finetune_lab.prepare_data --task instruction --source hf --train-samples 256 --eval-samples 32
python 01_instruction/unsloth/train.py --dry-run
```

256 train, 32 validation, 32 test로 작은 실험을 시작한다. 데이터가 이미 존재하면 준비 명령은 덮어쓰기를 거부한다. 기존 파일을 재사용하거나 `--output-root`를 바꾼다. 실험을 의도적으로 교체할 때만 `--overwrite`를 지정한다.

### 4-3. Base 모델 평가 기록

```bash
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M --data-dir data/processed/instruction --kind sft --device cuda --output outputs/instruction_unsloth_before.json
```

몇 개의 고정 질문에 대한 출력, 응답 부분의 손실과 비교 지표를 기록한다. 이 모델과 데이터는 영어 중심이므로 먼저 영어 지시로 비교한다. 한국어 학습 효과를 보려면 별도의 한국어 데이터·동일 언어 평가가 필요하다.

### 4-4. QLoRA 학습

```bash
python 01_instruction/unsloth/train.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --data-dir data/processed/instruction \
  --output-dir outputs/instruction_unsloth_qlora \
  --max-steps 20 --max-length 256 \
  --batch-size 1 --gradient-accumulation 4 \
  --learning-rate 0.0002 --lora-r 16 --lora-alpha 16 \
  --load-in-4bit
```

단일 GPU에서 effective batch size는 `1 × 4 = 4`이다. 20 optimizer step은 20개 예제를 뜻하지 않는다. 학습은 validation loss를 시작 전에 측정하고, train만 업데이트한 뒤 다시 validation loss를 측정한다. `metrics.json`에 두 값과 train 지표를 기록한다. 각 model loss가 NaN/Inf이면 backward 전에 중단하고, 검증·최종 지표가 비정상이면 성공 결과 저장을 중단한다.

완료한 출력 경로에는 `adapter_config.json`, `adapter_model.safetensors`, tokenizer 파일, `training_metadata.json`, `metrics.json` 등이 저장된다. 이것은 **base 모델 전체가 아닌 adapter**다. 기존 성공 결과를 덮어쓰지 않도록 새 실행에는 새로운 `--output-dir`를 사용한다. 이 짧은 실습은 중간 optimizer checkpoint를 저장하지 않으므로 중단 시 정확한 optimizer 상태 복구를 제공하지 않는다.

### 4-5. 저장 결과를 다시 로딩해 평가

```bash
python -m finetune_lab.evaluate --model outputs/instruction_unsloth_qlora --data-dir data/processed/instruction --kind sft --device cuda --output outputs/instruction_unsloth_after.json
```

공통 evaluator는 adapter와 base를 함께 로딩한다. 재로딩은 학습 직후 메모리 안의 모델만 확인하는 것과 다르다. 아래는 Unsloth로 직접 한 질문을 생성하는 예다.

```python
from unsloth import FastLanguageModel  # Transformers/TRL보다 먼저
from finetune_lab.text_encoding import format_prompt

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/instruction_unsloth_qlora",
    max_seq_length=256,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
inputs = tokenizer(format_prompt("Return the number 12 with no extra words."),
                   add_special_tokens=False, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                         eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

로딩 절차는 [Unsloth inference 문서](https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference)에 근거한다. Base까지 하나의 디렉토리로 만들려면 공통 merge 도구를 쓸 수 있지만 작은 adapter보다 더 많은 디스크와 메모리가 필요하다.

```bash
python -m finetune_lab.merge --adapter outputs/instruction_unsloth_qlora --output-dir outputs/instruction_unsloth_merged
mkdir -p outputs
python -m pip freeze > outputs/unsloth_environment.txt
```

## 5. 비교 실험과 해석

같은 데이터를 두고 출력 경로를 바꿔 `--no-load-in-4bit`를 실행한다. 이어 rank 8/16, step 20/100, max-length 256/512를 **한 번에 하나씩** 비교한다. 설정 비교에는 validation을 쓰고 test를 반복적으로 보며 설정을 조정하지 않는다. 더 긴 실제 학습에서는 고정 `main` 대신 모델과 데이터의 Hub commit SHA를 `--revision`으로 지정하고 환경 freeze를 함께 보관한다.

| 관찰 | 해석과 다음 확인 |
|---|---|
| train loss만 줄고 validation은 악화 | 과적합 가능성. step·rank·학습률을 줄이고 예제 다양성 확인 |
| validation loss 감소, 답변 형식 불안정 | 낮은 loss가 지시 준수 전체를 뜻하지 않음. 고정 질문의 실제 생성 확인 |
| 생성이 길게 이어짐 | 학습·추론의 EOS와 prompt format 일치 확인 |
| 4bit가 더 느림 | 작은 모델에서 양자화 비용 가능. 동일 batch/길이로 시간과 VRAM 비교 |
| CUDA OOM | batch와 길이를 줄이고 GPU에 남은 다른 프로세스 확인 |
| loss가 NaN | learning rate, fp16/bf16 지원, 정답 토큰 존재, 패키지 호환성 확인 |
| `no kernel image` / `sm_121` 오류 | Spark 컨테이너·CUDA 커널 빌드와 ARM64 wheel 확인. 공통 Spark 가이드의 점검 절차 수행 |

SFT는 지시와 답변의 패턴을 배우는 단계다. 사람의 선호를 최적화하는 DPO/RLHF, 일반적 안전성 검증, 새로운 지식의 장기 보존까지 이 실험 하나로 완료되는 것은 아니다. 다음으로 지식 실습의 CPT와 SFT를 비교하면 두 학습 목표의 차이를 확인할 수 있다.

## 수업 완료 기준

1. 준비: 세 split과 manifest를 확인하고 dry-run의 데이터 수·모델·학습 방법을 설명합니다.
2. 실행: 실제 학습이 유한 loss로 종료되고 예상한 전체 모델 또는 adapter·tokenizer·metadata가 새 출력 경로에 저장됩니다.
3. 재사용: 별도 프로세스에서 저장 결과를 읽어 답변을 생성합니다. Adapter이면 동일 base와 revision을 사용합니다.
4. 해석: 동일 validation의 response loss와 생성 답변을 비교하고 지시 준수·관련성·정확성·종료를 읽어 설명합니다. 설정을 고른 뒤 test를 최종 평가합니다.

실행 성공과 품질 개선은 각각 기록합니다. 수업을 준비했거나 dry-run만 통과한 상태를 학습 완료로 표시하지 않습니다. Unsloth 경로는 선택 확장이므로 지원 환경이 없으면 미실행으로 남기고 주 경로를 진행합니다.다음 단계는 이 문서 첫머리의 수업 경로와 [커리큘럼](../../docs/11_curriculum.md)을 따릅니다.
