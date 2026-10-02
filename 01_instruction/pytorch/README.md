# PyTorch로 base LLM에 instruction following 학습하기

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

주 경로는 PyTorch full(I1) → HF LoRA(I4) → [Domain 수업](../../04_post_training/domain_huggingface/README.md)입니다. 다른 full/QLoRA/Unsloth 조합은 주 경로를 마친 뒤 선택 비교합니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, 방법에 맞는 실행 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 실습은 **문장을 이어 쓰도록 사전학습한 base 모델에 질문·지시 → 응답 패턴을 학습**시킵니다. `HuggingFaceTB/SmolLM2-135M`으로 시작하며, `-Instruct` 모델로 시작하지 않습니다. 135M은 약 1억 3,500만 파라미터입니다. 기본 모델은 영어 중심이므로 설명은 한국어로 제공하고 초기 데이터는 영어로 사용합니다. 모델의 base/instruct 구분, 지원 언어와 Apache-2.0 라이선스는 [공식 모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)를 확인하십시오.

**학습 루프는 직접 PyTorch로 작성하고 모델·tokenizer 로딩은 Hugging Face Transformers를 사용**합니다. Transformer 아키텍처를 처음부터 구현하는 실습은 아닙니다. `train.py`는 진입점이고 실제 루프는 [torch_text.py](../../finetune_lab/torch_text.py), 토큰 처리 규칙은 [text_encoding.py](../../finetune_lab/text_encoding.py)에 있습니다.

## 1. 무엇을 배우는가

일반 causal language model은 앞의 토큰으로 다음 토큰의 확률을 계산합니다. 예를 들어 `The sky is` 뒤에 `blue`가 나올 확률을 높입니다. Base 모델도 지시를 우연히 따를 수 있지만, 모든 입력을 사용자 요청으로 처리하는 형식이 충분히 학습되어 있다고 가정할 수 없습니다.

Supervised fine-tuning(SFT)은 아래와 같은 정답 예제로 행동을 학습합니다.

```text
### Instruction:
Name two primary colors.

### Response:
Red and blue.<EOS>
```

여기서 teacher forcing은 생성된 이전 답변 대신 **데이터의 정답 토큰**을 다음 토큰 예측의 조건으로 사용하는 방식입니다. 실제 추론에서는 모델이 직접 생성한 이전 토큰을 사용합니다. 목표 loss는 응답 구간의 `-log P(정답 토큰 | 앞선 토큰)` 평균입니다. SFT의 next-token 학습 원리는 [고정 버전 Transformers causal language modeling 안내](https://huggingface.co/docs/transformers/v4.57.6/en/tasks/language_modeling)를 참고하십시오.

Instruction 부분은 조건으로 사용하지만 loss의 정답으로 사용하지 않습니다. 해당 `labels`를 `-100`으로 표시합니다. 응답과 EOS만 정답에 포함합니다. EOS는 답변 종료를 학습합니다. Padding은 배치 길이를 맞추기 위한 자리이므로 loss와 attention에서 제외합니다. PAD와 EOS의 ID가 같더라도 실제 EOS는 학습해야 하므로 **토큰 ID가 아닌 padding 위치**로 masking합니다.

Hugging Face 모델의 `forward(labels=...)`가 causal shift를 수행합니다. 코드에서 정답을 다시 한 칸 이동하면 두 칸 뒤를 예측하게 되므로 이동시키지 않습니다. Full fine-tuning은 원래 가중치를 모두 업데이트합니다. LoRA 옵션은 원래 가중치를 고정하고 작은 저랭크 행렬만 업데이트합니다.

## 2. 실행 환경과 자원

일반 환경은 [빠른 시작](../../docs/00_quickstart.md)의 `.venv-lab`과 `requirements/lab-hf.txt`를 준비합니다. 아래 명령은 프로젝트 루트에서 실행합니다. 작은 텍스트 full·LoRA는 CPU에서도 가능하며, VLM은 모델 크기와 processor 메모리를 확인하고 CUDA를 권장합니다. DGX Spark 사용자는 [전용 환경 안내](../../docs/04_dgx_spark.md)의 NGC overlay를 선택합니다.

```bash
source .venv-lab/bin/activate
```

이후 `python` 명령은 활성화한 실습 환경에서 실행합니다. CUDA 명령에는 `--device cuda`를 명시해 GPU가 준비되지 않았을 때 오류를 확인합니다. CPU 텍스트 실습은 `--device cpu --dtype float32`를 선택합니다.

`--dtype auto`는 지원되는 CUDA에서 **bfloat16 autocast**로 연산하고, 가중치·gradient·Adam 상태는 float32로 유지합니다. `--dtype bfloat16`은 CUDA BF16 지원이 없으면 실패하고, `--dtype float32`는 비교를 위한 FP32 연산입니다. BF16은 FP16보다 넓은 지수 범위를 사용하며 이 루프는 GradScaler를 사용하지 않습니다. Autocast는 연산과 activation에 영향을 주고, optimizer 상태 메모리를 4bit로 줄이는 QLoRA와는 다릅니다.

Autocast의 forward/backward 사용 방식은 [PyTorch AMP 문서](https://docs.pytorch.org/docs/stable/amp.html), BF16 지원 검사는 [is_bf16_supported API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)를 참고하십시오.

가중치·gradient·Adam 두 상태를 각각 4·4·8 bytes/parameter로 근사하면 135M 모델에 약 2.16GB가 필요하고, 여기에 activation·tokenizer·프레임워크 메모리가 추가됩니다. 이것은 예상 계산값이며 실제 필요 메모리는 sequence length와 배치 크기에 따라 더 큽니다. Spark의 통합 메모리 안에서도 모델·데이터·다른 프로세스가 자원을 공유하므로 실제 사용량을 확인하십시오.

## 3. 데이터 준비와 형태

실제 데이터는 [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)를 사용합니다. 영어 instruction/response 약 15,000건이며 `instruction`, `context`, `response`, `category` 열을 제공합니다. 공식 card의 라이선스는 CC-BY-SA-3.0입니다. 원본 `train` split 하나에서 중복 prompt를 제거하고 seed로 섞어 이 실습의 train/validation/test를 만듭니다.

```bash
python -m finetune_lab.prepare_data --task instruction --source hf \
  --train-samples 256 --eval-samples 32
```

다운로드는 이 명령에서만 실행합니다. 학습 스크립트는 준비된 JSONL을 읽고, 없으면 준비 명령을 안내합니다. HF cache는 데이터 전체를 내려받을 수 있으며 `--train-samples`는 **학습용으로 변환할 예제 수**입니다. 256/32/32개가 기본 목표이며 실제 수와 dataset revision/hash는 `data/processed/instruction/manifest.json`을 확인합니다. 같은 위치에 다시 준비하려면 의도를 드러내는 `--overwrite`를 추가하거나 다른 `--output-root`를 사용합니다.

처리된 데이터는 다음 경로에 있습니다.

```text
data/processed/instruction/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── manifest.json
```

JSONL은 한 줄에 한 JSON 객체입니다. `context`가 있다면 `prompt` 안에 `Context:`로 함께 넣습니다.

```json
{"id":"dolly-123","prompt":"Name two primary colors.","completion":"Red and blue."}
```

`--source demo`는 작성된 CC0 예제로 다운로드 없이 구조를 확인합니다. 번호를 그대로 답하는 단순한 예제이며 instruction 성능을 평가하는 데이터는 아닙니다. 기존 실제 데이터와 섞지 않도록 별도 위치에 준비합니다.

```bash
python -m finetune_lab.prepare_data --task instruction --source demo \
  --output-root data/demo
python 01_instruction/pytorch/train.py --dry-run --data-dir data/demo/instruction
```

`--dry-run`은 train/validation/test 세 파일의 schema·예제 수·중복 ID·정규화된 prompt 중복을 검사하며 PyTorch/Transformers를 import하지 않습니다. 따라서 모델 다운로드와 GPU 초기화가 발생하지 않습니다. Test는 누출 검사를 위해서만 읽고 학습 loader에는 넣지 않습니다.

## 4. 먼저 학습 전 기준점 측정하기

```bash
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M \
  --data-dir data/processed/instruction --kind sft --device cuda \
  --output outputs/instruction/baseline.json
```

최종 비교에는 동일한 test set과 prompt format을 사용합니다. 학습 중 출력되는 validation loss로 학습률·step을 선택하고, test set은 최종 판단에 사용하십시오. 응답 loss 감소는 정답 표현의 확률이 높아졌다는 뜻이며, 임의의 지시를 잘 따르게 되었다는 보장은 아닙니다. 자유 응답의 문자열 exact match는 표현 차이에도 실패할 수 있으므로 몇 개 답변을 직접 읽어 의미·지시 준수·반복을 함께 판단합니다.

## 5. 실행 확인과 실제 학습

먼저 아주 작은 random GPT-2로 실제 forward/backward/optimizer/save 과정을 확인합니다. `--smoke-model`은 모델과 tokenizer를 코드에서 생성하므로 인터넷이 필요 없습니다. 이 tokenizer는 작은 어휘를 사용하므로 언어 능력과 실험 성능은 판단할 수 없습니다.

```bash
python 01_instruction/pytorch/train.py --smoke-model --device cuda --dtype auto \
  --data-dir data/demo/instruction --output-dir outputs/instruction/pytorch-smoke \
  --max-steps 2 --max-length 64 --batch-size 2 --gradient-accumulation 4
```

실제 pretrained base 모델은 아래처럼 학습합니다. 첫 실행은 모델과 tokenizer를 Hugging Face에서 다운로드합니다. `--revision`에 모델 commit hash를 지정하면 같은 모델 버전으로 재현할 수 있습니다.

```bash
python 01_instruction/pytorch/train.py --device cuda --dtype auto --method full \
  --max-steps 20 --max-length 256 --batch-size 1 --gradient-accumulation 4 \
  --learning-rate 2e-5 --output-dir outputs/instruction/pytorch-full
```

20 optimizer step은 학습 파이프라인 확인을 위한 시작값입니다. 충분한 instruction 성능을 만드는 고정 레시피가 아닙니다. Validation 결과를 보며 예제 수·step을 늘리십시오. 256개에서 batch=1, accumulation=4라면 한 epoch에 대략 64 update가 발생합니다. `--max-steps`가 더 크면 train loader를 다시 순회합니다.

이미 학습 결과가 있는 output 디렉토리는 덮어쓰지 않습니다. 다음 실험에는 `--output-dir outputs/instruction/pytorch-full-run2`처럼 새 경로를 지정하십시오.

LoRA 비교도 가능합니다. Full 학습보다 높은 learning rate가 필요할 수 있으므로 별도 실험으로 비교합니다. 결과는 adapter이므로 원래 모델과 함께 로드해야 합니다. Random smoke 모델은 adapter의 base 경로가 없으므로 `--smoke-model --method lora` 조합을 거절합니다.

```bash
python 01_instruction/pytorch/train.py --device cuda --dtype auto --method lora --learning-rate 1e-4 \
  --max-steps 20 --output-dir outputs/instruction/pytorch-lora
```

## 6. 학습 루프를 읽는 순서

1. `read_jsonl`과 `validate_splits`가 train/validation/test의 필수 필드·빈 데이터·중복 ID·동일 prompt 누출을 GPU 로딩 전에 검사합니다.
2. `encode_record`가 prompt/응답을 별도로 tokenize하고 `labels`를 만듭니다.
3. 긴 응답은 오른쪽에서 자르고 EOS를 남깁니다. Prompt는 왼쪽에서 잘라 응답 바로 앞의 문맥을 남깁니다. 최소 한 prompt 토큰 + 한 응답 토큰 + EOS를 위해 `max-length >= 3`이 필요합니다. 긴 질문의 중요한 앞부분이 잘릴 수 있으므로 실제 학습 전 길이를 확인하고 length를 늘리거나 예제를 정리하십시오.
4. Collator가 배치 내부 최장 길이에 맞춰 right padding합니다.
5. `validation_loss`가 업데이트 전 기준 loss를 계산합니다.
6. `train_loop`가 forward → backward → clipping → AdamW step을 반복합니다.
7. 마지막 validation과 모델·tokenizer·metadata를 저장합니다.

Gradient accumulation은 여러 작은 microbatch의 gradient를 모은 뒤 한 번 업데이트하는 방식입니다. 각 window의 실제 정답 토큰 수로 loss를 가중합니다. 예제 5개, accumulation=4라면 4개와 1개 두 window가 생기며 마지막 window도 정상적으로 업데이트합니다. 항상 `loss/4`로 처리하는 코드와 달리 마지막 작은 window를 과소평가하지 않습니다. Token 길이가 다른 예제도 실제 target 토큰 수에 비례하여 반영합니다.

`clip_grad_norm_`은 gradient norm을 기본 1.0으로 제한합니다. Training/validation loss나 gradient가 NaN/Inf이면 중단합니다. Forward는 선택한 dtype의 autocast context에서 실행하고 backward는 밖에서 수행합니다. 이 예제에는 scheduler·분산 학습·중간 checkpoint resume가 없으므로 긴 운영 학습에는 Hugging Face 방법과 비교하십시오.

## 7. 저장 모델 재평가와 추론

Full 학습 디렉토리에는 `model.safetensors`, `config.json`, tokenizer 파일, `training_metadata.json`이 생깁니다. LoRA는 `adapter_model.safetensors`와 `adapter_config.json`을 저장합니다. Metadata는 base model/요청·실제 revision/prompt 형식/device/seed/compute·parameter dtype/학습 설정/validation loss/학습 parameter 수를 기록합니다. Adapter 평가 loader가 같은 base model과 revision을 다시 로드할 수 있도록 보존합니다. 모델 가중치는 추론에 재사용할 수 있지만 optimizer state를 저장하지 않으므로 정확한 학습 resume checkpoint는 아닙니다.

```bash
python -m finetune_lab.evaluate --model outputs/instruction/pytorch-full \
  --data-dir data/processed/instruction --kind sft --device cuda \
  --output outputs/instruction/pytorch-full-eval.json
python -m finetune_lab.evaluate --model outputs/instruction/pytorch-lora \
  --data-dir data/processed/instruction --kind sft --device cuda \
  --output outputs/instruction/pytorch-lora-eval.json
```

Full 모델의 단일 질문을 Python에서 직접 생성할 때도 학습과 같은 prompt를 사용합니다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune_lab.text_encoding import format_prompt

path = "outputs/instruction/pytorch-full"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda")
inputs = tokenizer(format_prompt("Name two primary colors."), return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## 8. 실험과 문제 해결

- Full과 LoRA를 같은 데이터·seed·step으로 비교하고 trainable parameter 수와 validation loss를 기록하십시오.
- `max-length`를 128/256/512로 바꿔 긴 질문에 대한 정보 손실과 메모리를 비교하십시오.
- Train loss만 내려가고 validation이 나빠지면 과적합 가능성이 있습니다. Step/learning rate를 줄이고 데이터 품질·중복·분포를 확인하십시오.
- OOM이면 `batch-size`를 먼저 줄이고 `max-length`를 줄입니다. Accumulation을 늘려 update당 예제 수를 유지할 수 있지만 activation memory는 현재 microbatch가 결정합니다.
- 응답이 계속 반복되면 EOS label이 남아 있는지, 정답이 반복되지 않는지 확인하십시오. `tests/test_text_encoding.py`가 EOS/PAD와 truncation 사례를 검사합니다.
- 한국어를 확장할 때는 한국어에 적합한 **base** 모델과 같은 라이선스 검토를 거친 한국어 instruction 데이터를 사용하십시오. 작은 영어 모델에 한국어 데이터 몇 개를 주는 것만으로 충분한 한국어 능력이 생기지는 않습니다.

## 수업 완료 기준

1. 준비: 세 split과 manifest를 확인하고 dry-run의 데이터 수·모델·학습 방법을 설명합니다.
2. 실행: 실제 학습이 유한 loss로 종료되고 예상한 전체 모델 또는 adapter·tokenizer·metadata가 새 출력 경로에 저장됩니다.
3. 재사용: 별도 프로세스에서 저장 결과를 읽어 답변을 생성합니다. Adapter이면 동일 base와 revision을 사용합니다.
4. 해석: 동일 validation의 response loss와 생성 답변을 비교하고 지시 준수·관련성·정확성·종료를 읽어 설명합니다. 설정을 고른 뒤 test를 최종 평가합니다.

실행 성공과 품질 개선은 각각 기록합니다. 수업을 준비했거나 dry-run만 통과한 상태를 학습 완료로 표시하지 않습니다. 다음 단계는 이 문서 첫머리의 수업 경로와 [커리큘럼](../../docs/11_curriculum.md)을 따릅니다.
