# Base LLM에 instruction learning 추가하기: Hugging Face Trainer + PEFT

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

주 경로는 PyTorch full(I1) → HF LoRA(I4) → [Domain 수업](../../04_post_training/domain_huggingface/README.md)입니다. 다른 full/QLoRA/Unsloth 조합은 주 경로를 마친 뒤 선택 비교합니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, Spark CUDA 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 실습에서는 **instruction tuning 이전의 base 모델**에 지시문과 모범 응답을 학습시킵니다. 기본 모델은 `HuggingFaceTB/SmolLM2-135M`입니다. 모델 이름 끝의 `-Instruct`가 없는 것을 확인하세요. 실행 대상은 **DGX Spark의 ARM64 Linux + NVIDIA CUDA GPU**입니다. 135M은 첫 실습 비용을 낮추기 위한 선택이며, 20 step 학습은 파이프라인 확인용입니다. 사용 가능한 챗봇을 만드는 데 필요한 데이터 품질·다양성·평가를 대신하지 않습니다.

[공식 모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)에 따르면 이 모델은 Apache 2.0 라이선스의 주로 영어를 다루는 모델입니다. 설명은 한국어로 읽고, 기본 데이터 실습은 영어로 진행합니다. 한국어를 연습하려면 한국어 기반 모델과 한국어 지시 데이터를 함께 선택해야 합니다.

## 1. 무엇을 학습하는가

사전학습된 causal LM은 앞 문맥을 받아 다음 토큰을 예측합니다. `Translate this sentence ...` 뒤에 번역문 대신 비슷한 문장을 이어 쓸 수도 있습니다. Supervised Fine-Tuning(SFT)은 지시문 `x` 뒤에 정답 응답 `y`가 올 확률을 높입니다.

\[
L_{SFT}=-\sum_{t \in \text{response}} \log p_\theta(y_t\mid x,y_{<t})
\]

작업의 목적은 “질문을 입력받으면 지시한 형식으로 답한다”는 조건부 행동을 익히는 것입니다. SFT가 사실성, 안전성, 추론 능력을 자동으로 보장하지는 않습니다. 선호도 학습(DPO/RLHF)은 다른 학습 단계이며 이 스크립트의 범위에는 포함되지 않습니다.

이 프로젝트는 base 모델에서도 사용할 수 있는 다음 형식을 학습과 평가에 동일하게 사용합니다.

```text
### Instruction:
<지시문과 필요한 참고 문맥>

### Response:
<모범 응답><EOS>
```

`### Response:`까지는 입력이며, 응답과 EOS만 정답 label로 사용합니다. 입력 label과 padding label은 `-100`으로 만들어 loss에서 제외합니다. 모델의 causal-LM forward가 내부에서 한 토큰 이동을 수행하므로 label을 별도로 shift하면 안 됩니다. PAD를 EOS와 같은 ID로 설정해도 **실제 EOS 위치**는 학습 대상이고 **padding 위치**만 제외합니다.

공유 인코더는 prefix와 응답을 각각 토큰화합니다. 따라서 문자열 전체를 한 번에 BPE 토큰화할 때와 경계 토큰이 다를 수 있으며, 평가도 같은 prefix 형식을 사용해야 합니다. 길이 제한을 넘으면 응답의 앞부분과 EOS를 보존하고 prompt의 왼쪽을 자릅니다. 길게 잘린 prompt는 지시문을 잃을 수 있으므로 데이터 길이를 살펴보고 `--max-length`를 늘리세요. 이 실습은 문서 packing을 하지 않습니다.

## 2. Hugging Face, PyTorch, LoRA의 관계

Hugging Face는 학습 도구이며 LoRA는 파라미터를 업데이트하는 방법입니다. `Trainer`도 내부에서는 PyTorch로 forward/backward를 수행합니다. 이 폴더는 학습 루프·gradient accumulation·optimizer·평가·저장을 `Trainer`에 맡기고, tokenizer와 loss mask는 코드에서 명확히 만듭니다. [Trainer 공식 문서](https://huggingface.co/docs/transformers/en/main_classes/trainer)를 함께 참고하세요.

| `--method` | 업데이트하는 값 | 저장 결과 | 이 실습의 실행 환경 |
|---|---|---|---|
| `full` | 기존 가중치 전체 | 전체 모델 | DGX Spark CUDA |
| `lora` 기본값 | 작은 LoRA 행렬 | adapter + tokenizer | DGX Spark CUDA |
| `qlora` | 4-bit base에 추가한 LoRA 행렬 | adapter + tokenizer | NVIDIA CUDA + bitsandbytes |

LoRA는 선형층 `W`를 고정하고 `W + (alpha/r) BA`를 사용합니다. `A`와 `B`의 rank `r`이 작으므로 학습할 파라미터와 optimizer 상태가 줄어듭니다. 이 구현은 `target_modules="all-linear"`, `r=8`, `alpha=16`, dropout 0.05를 사용합니다. [LoRA 원 논문](https://arxiv.org/abs/2106.09685).

QLoRA는 base 가중치를 NF4 4-bit로 읽고 LoRA 행렬을 학습합니다. 계산과 activation까지 모두 4-bit가 되는 것은 아닙니다. 코드에서 double quantization과 `prepare_model_for_kbit_training()`을 사용하며, 단일 CUDA 장치에 명시적으로 올립니다. [QLoRA 원 논문](https://arxiv.org/abs/2305.14314), [PEFT 양자화 가이드](https://huggingface.co/docs/peft/developer_guides/quantization).

## 3. 데이터와 라이선스

기본 데이터는 [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)입니다. 15,000여 개 영어 지시 응답이며 `instruction`, `context`, `response`, `category` 열을 가집니다. **CC BY-SA 3.0**로 공개되므로 출처 표시와 해당 라이선스 의무를 확인하세요. **비상업 전용인 SciQ와는 라이선스가 다릅니다.**

변환 과정에서 `instruction`과 비어 있지 않은 `context`를 prompt로 만들고, `response`를 completion으로 사용합니다. 원본은 train split 하나이므로 정규화한 prompt의 정확한 중복을 제거하고 seed 42로 섞어, 요청한 수만큼 train/validation/test를 따로 선택합니다. 기본 수는 256/32/32이며 고정된 80/10/10 비율로 전체를 나누는 방식은 아닙니다. 실제 선택 수와 출처·revision은 준비 스크립트가 저장하는 `manifest.json`을 확인하세요.

```json
{"id":"example-1","prompt":"Return only the sum of 2 and 3.","completion":"5"}
```

이 예시는 형식을 설명하기 위해 작성한 예시입니다. 학습 데이터는 `data/processed/instruction/{train,validation,test}.jsonl`에 저장됩니다. train은 학습, validation은 설정 조정, test는 최종 비교용으로 구분합니다. test를 보고 데이터나 설정을 반복 수정하면 평가 결과가 낙관적으로 치우칩니다.

## 4. 순서대로 실행하기

아래 명령은 DGX Spark의 저장소 루트에서 실행합니다. 먼저 [DGX Spark 환경 안내](../../docs/04_dgx_spark.md)에 따라 컨테이너를 시작하고 의존성을 설치합니다. `requirements/spark-hf.txt`는 NVIDIA 이미지에 있는 PyTorch를 유지하며 Hugging Face 의존성을 추가합니다.

```bash
# DGX Spark host에서 실행
bash scripts/spark_container.sh hf
# 열린 컨테이너 안에서 실행
bash scripts/install_spark.sh hf
source .venv-spark-hf/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 단계 A: 실제 데이터 받기

```bash
python -m finetune_lab.prepare_data --task instruction --source hf --train-samples 256 --eval-samples 32
python 01_instruction/huggingface/train.py --dry-run
```

`--dry-run`은 모델을 읽거나 학습하지 않고 세 split의 JSONL schema와 개수를 확인합니다. 모델·tokenizer 다운로드 가능 여부와 GPU 메모리는 실제 학습 단계에서 확인됩니다. 다운로드한 원본은 Hugging Face cache를 사용하므로 같은 데이터에 다시 접근할 때 재사용됩니다.

### 단계 B: 학습 전 평가

```bash
python -m finetune_lab.evaluate --device cuda --model HuggingFaceTB/SmolLM2-135M --data-dir data/processed/instruction --kind sft --output outputs/instruction_hf_baseline_eval.json
```

기본 모델의 loss와 생성 응답을 기록하세요. 동일한 test 질문·prompt 형식·generation 설정을 학습 후에도 사용해야 비교가 가능합니다. 짧은 모범 응답과 표현이 달라도 맞는 답이 있을 수 있으므로 exact match를 단독으로 판단하지 마세요.

### 단계 C: LoRA 학습

```bash
CUDA_VISIBLE_DEVICES=0 python 01_instruction/huggingface/train.py --method lora --device cuda --max-steps 20 --output-dir outputs/instruction_hf_lora
```

Spark에서는 `--device cuda`를 명시합니다. CUDA를 사용할 수 없는 환경에서는 오류로 종료하여 잘못된 환경을 바로 확인할 수 있습니다. 기본 batch 1 × accumulation 4로 한 번의 optimizer update에 여러 microbatch를 모읍니다. `max-steps`는 optimizer update 수입니다.

학습 전후 validation loss를 같은 과정에서 측정합니다. 각 forward의 loss가 NaN/Inf이면 바로 실패하여 잘못된 결과가 정상 모델처럼 저장되지 않도록 합니다. 학습 동안 Hub 업로드나 외부 실험 추적 서비스 전송은 사용하지 않습니다.

같은 출력 폴더에 모델·adapter·metadata가 있으면 학습을 시작하기 전에 종료합니다. 설정을 바꾸어 다시 실험할 때는 `--output-dir outputs/instruction_hf_lora_run2`처럼 새 폴더를 지정하세요. `--dry-run`은 출력 폴더를 쓰지 않으며 기존 모델이 있어도 데이터 검사만 수행합니다.

### 단계 D: 저장한 adapter 다시 읽어 평가

```bash
python -m finetune_lab.evaluate --device cuda --model outputs/instruction_hf_lora --data-dir data/processed/instruction --kind sft --output outputs/instruction_hf_lora_eval.json
```

LoRA 결과 폴더에는 `adapter_config.json`, adapter 가중치, tokenizer, `training_metadata.json`, `metrics.json`이 저장됩니다. adapter는 독립적인 전체 모델이 아니므로 평가기는 metadata의 base 모델을 읽고 adapter를 적용해야 합니다. 최초의 base 모델을 내려받을 수 있거나 cache에 있어야 합니다.

## 5. Full fine-tuning과 QLoRA 비교

전체 모델을 저장하려면 다음처럼 실행합니다. full 기본 학습률은 `2e-5`, LoRA/QLoRA는 `2e-4`입니다. 데이터·모델 규모에 따른 조정이 필요합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_instruction/huggingface/train.py --method full --device cuda --learning-rate 2e-5 --max-steps 20 --output-dir outputs/instruction_hf_full
python -m finetune_lab.evaluate --device cuda --model outputs/instruction_hf_full --data-dir data/processed/instruction --kind sft --output outputs/instruction_hf_full_eval.json
```

QLoRA는 CUDA 환경의 별도 requirements 안내를 따른 뒤 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_instruction/huggingface/train.py --method qlora --device cuda --max-steps 20 --output-dir outputs/instruction_hf_qlora
```

135M 모델에서는 양자화의 설치·변환 비용이 이득보다 클 수 있습니다. 먼저 LoRA로 흐름을 익히고, 더 큰 모델에서 QLoRA의 메모리 절약을 비교하세요. 이 QLoRA 코드는 GPU 한 장만 노출한 단일 프로세스 학습을 대상으로 합니다. Spark ARM64에서 bitsandbytes가 실제로 지원되는지는 루트 환경 점검과 CUDA smoke 단계에서 확인한 뒤 실행합니다.

## 6. 네트워크 없는 파이프라인 점검

```bash
python -m finetune_lab.prepare_data --task instruction --source demo --output-root data/demo
python 01_instruction/huggingface/train.py --data-dir data/demo/instruction --dry-run
CUDA_VISIBLE_DEVICES=0 python 01_instruction/huggingface/train.py --data-dir data/demo/instruction --smoke-model --method full --device cuda --max-steps 2 --max-length 64 --gradient-accumulation 1 --output-dir outputs/instruction_hf_smoke
```

demo 데이터는 프로젝트에서 만든 작은 예시이며, `--smoke-model`은 임의 초기화한 작은 모델입니다. Spark에서 이 실험으로 tokenization → CUDA loss → optimizer → 저장을 검사합니다. 사전학습 모델에 instruction 능력을 추가했다는 결과로 해석하면 안 됩니다. Spark CUDA 실행 결과와 coverage 80% 달성은 아직 확인되지 않았습니다. 실행·검증 명령은 Spark에서 수행해야 합니다.

Spark에서 학습·adapter 재로드·generation 통합 테스트를 실행하려면 다음 명령을 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 RUN_ML_TESTS=1 python -m unittest discover -s tests -p test_hf_text.py
```

통합 테스트는 기본 CUDA 장치를 사용하며 CUDA가 없으면 실패합니다. 전체 테스트와 coverage 문턱은 Spark 환경 문서의 검증 명령을 따르세요. `training_metadata.json`의 `runtime`에는 Python/패키지 버전, CUDA 버전, GPU 이름과 capability가 저장됩니다.

## 7. 결과를 읽고 확장하기

| 관찰 | 가능한 원인 | 다음 실험 |
|---|---|---|
| train loss만 감소 | 적은 데이터 암기 | 더 다양한 데이터, update 수 감소 |
| 형식은 맞지만 사실이 틀림 | instruction 행동과 사실 지식은 별개 | 지식 실습과 QA 평가 |
| prompt를 따라 출력 | base 모델 적응 부족, mask 문제 | 생성 예시와 label 직접 확인 |
| 긴 지시를 무시함 | prompt 왼쪽 truncation | max-length 증가, 데이터 길이 제한 |
| 메모리 부족 | 긴 sequence, optimizer/activation | max-length와 batch 감소, LoRA 선택 |

학습률, rank, 데이터 수, step 수 중 한 가지씩 변경하고 같은 test를 비교하세요. `--max-steps 200 --max-length 512`처럼 점진적으로 늘리되, validation loss와 기존 능력의 변화도 기록합니다. seed를 고정해도 하드웨어·라이브러리에 따라 완전히 같은 결과가 보장되지는 않습니다.

재현 가능한 실험에서는 `--revision main` 대신 모델 commit SHA를 지정합니다. 데이터 준비의 `--revision`도 별도로 고정하세요. 커스텀 데이터는 같은 JSONL schema와 세 split을 만든 뒤 `--data-dir`로 지정하면 됩니다.

TRL의 [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer)는 prompt/completion, 대화 데이터, packing 등 SFT에 특화된 기능을 제공합니다. 이 실습은 명시적인 labels와 Transformers `Trainer`를 사용해 PyTorch 폴더와 loss를 비교하기 쉽게 만들었습니다. TRL로 옮길 때는 라이브러리 버전에 맞는 completion-only loss와 chat template 설정을 확인하고, 동일한 label 범위가 실제로 적용되는지 검사하세요.
