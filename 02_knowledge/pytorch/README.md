# PyTorch로 기존 LLM에 추가 지식 학습시키기

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

HF의 C4/S4/K3를 묶은 [Domain 주 수업](../../04_post_training/domain_huggingface/README.md)에서 시작합니다. CPT만/QA만/CPT→QA를 비교한 뒤 [DPO 수업](../../04_post_training/dpo_huggingface/README.md)으로 이동합니다. 다른 도구/기법은 선택 비교입니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, Spark CUDA 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 실습은 이미 instruction 학습을 받은 `HuggingFaceTB/SmolLM2-135M-Instruct`에 과학 문서와 QA를 추가 학습합니다. **문서의 다음 토큰을 학습하는 CPT**와 **질문 → 정답을 학습하는 QA SFT**를 `--stage`로 구분합니다. 학습 loop는 직접 PyTorch, 모델·tokenizer 로딩은 Hugging Face Transformers입니다. 모델은 영어 중심이며 Apache-2.0 라이선스로 제공됩니다. Base와 instruct의 배경은 [공식 모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)를 참고하십시오.

## 1. 지식을 추가한다는 것은 무엇인가

Fine-tuning은 모델 parameter를 바꾸어 새 데이터의 패턴을 더 잘 예측하도록 합니다. 데이터베이스에 한 행을 넣듯 사실을 정확하고 독립적으로 저장하는 과정은 아닙니다. 원래 알고 있던 지식이 간섭을 받고, 질문 표현이 바뀌면 답을 못할 수 있으며, 틀린 답을 자신 있게 만들 수도 있습니다.

| 단계 | 입력 데이터 | Loss의 정답 | 기대하는 변화 |
|---|---|---|---|
| CPT: continued pretraining | 도메인 문서·문단 | 본문의 다음 토큰 전체 | 용어·문체·도메인 분포 적응 |
| QA SFT | 질문과 정답 | 응답 + EOS | 질문에 맞춰 학습 지식을 꺼내 답하는 패턴 |
| RAG: 비교 개념 | 검색 문서가 포함된 질문 | 별도 학습 없이도 가능 | 답변 시점에 최신 근거를 제공 |

CPT는 문장을 익히므로 답을 대화 형식으로 전달하는 능력까지 자동으로 보장하지 않습니다. QA SFT는 답변 형식을 익히지만 다양한 사실을 깊이 이해했는지 따로 평가해야 합니다. 실제 프로젝트에서는 CPT → QA SFT를 연결하거나, QA SFT만 실행하거나, retrieval을 함께 사용합니다.

동일 도메인만 오래 학습하면 원래 지시 준수·일반 지식이 약해지는 **catastrophic forgetting**이 생길 수 있습니다. 학습 전후에 일반 instruction 예제도 평가하고, 필요한 경우 일반 예제 replay를 섞거나 learning rate/step을 줄이십시오. 본 실습은 단계 간 차이를 볼 수 있도록 두 데이터를 별도로 유지합니다.

## 2. 데이터: SciQ의 support 문단과 QA

[allenai/sciq](https://huggingface.co/datasets/allenai/sciq)는 물리·화학·생물 등의 과학 객관식 문제 13,679개입니다. 공식 split은 train 11,679 / validation 1,000 / test 1,000이며 `question`, `correct_answer`, `distractor1/2/3`, `support`를 제공합니다. Support가 없는 문제도 있습니다. 공식 라이선스는 **CC-BY-NC-3.0**이므로 이 데이터 사용 결과를 상업적으로 활용하려면 조건을 확인해야 합니다.

```bash
# DGX Spark 호스트의 저장소 루트에서 컨테이너 시작
bash scripts/spark_container.sh hf
# 이후 컨테이너 셸의 저장소 루트에서 설치와 실습 실행
bash scripts/install_spark.sh hf
source .venv-spark-hf/bin/activate
python -m finetune_lab.prepare_data --task knowledge --source hf \
  --train-samples 256 --eval-samples 32
```

실제 학습 대상은 **DGX Spark의 ARM64 Linux + CUDA**입니다. [DGX Spark 실행 안내](../../docs/04_dgx_spark.md)에서 컨테이너 버전·GPU 확인 절차를 먼저 확인하십시오. NVIDIA 컨테이너의 Spark용 PyTorch/CUDA를 사용하고 다른 플랫폼의 wheel로 교체하지 않습니다. 이후 명령은 모두 열린 컨테이너 셸의 저장소 루트에서 실행합니다. 데이터와 모델은 처음 준비할 때 다운로드하며 HF cache에 저장됩니다. 같은 위치를 덮어쓰려면 `--overwrite`가 필요합니다.

하나의 원본 예제에서 아래 두 표현을 만듭니다.

```text
data/processed/
├── knowledge_cpt/{train,validation,test}.jsonl
└── knowledge_sft/{train,validation,test}.jsonl
```

각 디렉토리에 `manifest.json`도 있으며 실제 예제 수·라이선스·dataset revision·파일 SHA256을 기록합니다. 원본 split을 먼저 선택한 뒤 CPT/QA로 변환합니다. 정확히 같은 문단과 같은 질문의 정규화된 중복을 제거하므로 최종 건수는 요청보다 줄 수 있습니다. Support가 빈 QA는 SFT에는 사용할 수 있지만 CPT에는 들어가지 않습니다. CPT/SFT에서 항상 같은 건수를 가정하지 마십시오.

```json
{"id":"sciq-train-7","text":"Example science supporting paragraph."}
```

```json
{"id":"sciq-train-7","prompt":"Example science question?","completion":"Correct answer"}
```

SFT prompt에는 `support`를 넣지 않습니다. 문서를 보지 않고 답하는 closed-book 과학 QA를 학습합니다. 객관식 오답을 label로 학습하지 않습니다. SFT에서는 아래 명시적인 형식을 사용하고 응답만 loss에 포함합니다.

```text
### Instruction:
Example science question?

### Response:
Correct answer<EOS>
```

모델의 원래 chat template 대신 다른 방법의 텍스트 실습과 같은 이 형식을 의도적으로 사용합니다. 기존 instruct 모델의 원래 대화 분포와 차이가 있으므로 추론에서도 같은 `format_prompt`를 사용하십시오. 실제 서비스용 학습을 확대할 때는 원래 chat template을 유지하는 실험과 일반 instruction 보존 평가를 추가할 수 있습니다.

**SciQ 지식이 pretrained 모델에게 처음 공개되는 정보라고 주장할 수 없습니다.** 이미 사전학습에 존재했을 수 있으므로 이 실습은 도메인 적응과 학습 과정 이해에 적합합니다. 새 사실을 외웠는지 확인하려면 공개되지 않은 가상의 사실과 표현을 바꾼 질문으로 별도 실험하십시오. `--source demo`는 가상의 Nara 우주정거장 코드를 제공합니다. 이 데이터는 파이프라인 확인용이고 train/test에 다른 정거장을 두므로 미학습 정거장의 코드를 맞힐 것으로 기대하지 않습니다.

## 3. 모델을 바꾸기 전 평가

```bash
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-dir data/processed/knowledge_cpt --kind cpt --device cuda \
  --output outputs/knowledge/baseline-cpt.json
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-dir data/processed/knowledge_sft --kind sft --device cuda \
  --output outputs/knowledge/baseline-sft.json
```

CPT의 loss/perplexity는 같은 tokenizer·같은 문단·같은 truncation 기준에서 비교합니다. Perplexity `exp(mean next-token loss)`가 낮아지면 해당 문장을 더 예상하기 쉬워진 것입니다. 사실 QA 정확도까지 높아졌다는 뜻은 아닙니다. SFT loss는 응답 토큰만 포함하므로 CPT loss와 수치 자체를 직접 비교하지 않습니다.

지식 실험은 세 종류를 나누면 해석하기 쉽습니다.

1. 학습한 사실을 같은 질문으로 다시 묻기: 기억 여부 진단, 일반화 성능 수치로 보고하지 않기.
2. 학습한 사실을 새로운 질문 표현으로 묻기: 지식 사용의 표현 일반화 확인.
3. 학습하지 않은 사실을 묻기: 모른다고 답할 수 있는지 또는 기존 일반 지식으로 답하는지 확인.

QA test에 포함된 사실이 train 문서로 노출되지 않았으면, 완전히 새로운 사실을 업데이트에서 얻을 수 없습니다. 반대로 train 사실로 만든 새 질문은 사실 노출이 의도된 **기억 평가**이며 독립적인 지식 일반화 평가와 구별해야 합니다.

## 4. 다운로드 없는 최소 실행

```bash
python -m finetune_lab.prepare_data --task knowledge --source demo --output-root data/demo
python 02_knowledge/pytorch/train.py --stage cpt --dry-run --data-dir data/demo/knowledge_cpt
python 02_knowledge/pytorch/train.py --stage sft --dry-run --data-dir data/demo/knowledge_sft
python 02_knowledge/pytorch/train.py --stage cpt --smoke-model --device cuda --dtype auto \
  --data-dir data/demo/knowledge_cpt --max-steps 2 --max-length 64 \
  --output-dir outputs/knowledge/pytorch-cpt-smoke
```

`--dry-run`은 모델 라이브러리 없이 train/validation/test 세 JSONL의 구조·중복 ID·문단(CPT) 또는 질문(SFT) 누출을 검사합니다. Test는 이 검사용으로만 읽고 학습 loader에는 넣지 않습니다. `--smoke-model`은 tiny random GPT-2/작은 WordLevel tokenizer로 실제 backward와 저장을 확인합니다. 언어·과학·지식 정확도 판단에는 사용할 수 없습니다. `--method full`만 지원하는 smoke 제한은 random base 모델 없이 adapter만 저장하는 문제를 방지합니다.

## 5. CPT와 QA SFT를 독립적으로 비교하기

CPT는 기본 단계입니다. `--stage cpt`를 생략해도 CPT입니다.

```bash
python 02_knowledge/pytorch/train.py --stage cpt --device cuda --dtype auto --method full \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --learning-rate 1e-5 \
  --max-steps 20 --max-length 256 --batch-size 1 --gradient-accumulation 4 \
  --output-dir outputs/knowledge/pytorch-cpt
```

QA SFT 단독 학습도 같은 시작 모델에서 실행하십시오. CPT 모델 위에 학습하는 명령은 다음 절에서 따로 보여 줍니다.

```bash
python 02_knowledge/pytorch/train.py --stage sft --device cuda --dtype auto --method full \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --learning-rate 2e-5 \
  --max-steps 20 --max-length 256 --batch-size 1 --gradient-accumulation 4 \
  --output-dir outputs/knowledge/pytorch-sft
```

20 update는 파이프라인의 시작값입니다. 많은 지식 습득을 보장하는 값이 아닙니다. Optimizer update 수이며 microbatch 수가 아닙니다. 데이터가 끝나면 마지막 작은 accumulation window도 업데이트하고 train loader를 다시 순회합니다. `--device cuda`를 명시해 Spark GPU에서 실행하며, CUDA가 없으면 실패합니다.

`--dtype auto`는 CUDA BF16 지원 시 bfloat16 autocast로 forward를 계산합니다. 학습 가중치·gradient·Adam 상태는 float32로 유지합니다. `--dtype float32`로 FP32 연산과 비교하고, `--dtype bfloat16`으로 BF16 지원을 강제 확인할 수 있습니다. BF16 autocast는 모델을 4bit로 만드는 과정이 아니며 optimizer 상태 메모리는 그대로입니다. 본 루프는 BF16에서 GradScaler를 사용하지 않습니다.

Forward 안에서 autocast를 사용하고 backward는 밖에서 실행하는 방식은 [PyTorch AMP 문서](https://docs.pytorch.org/docs/stable/amp.html)를 참고하십시오.

기존 결과가 있는 output 디렉토리는 덮어쓰지 않습니다. 반복 실험에는 새 `--output-dir`를 지정하십시오. 기본 output도 CPT와 SFT를 서로 다른 디렉토리에 저장합니다.

CPT는 본문을 `max-length - 1`토큰까지 사용하고 마지막에 EOS를 넣습니다. 이 작은 실습은 문서 전체를 overlapping window로 분할하거나 여러 문서를 packing하지 않습니다. 긴 문서의 뒤쪽은 학습하지 않으므로 문서 청킹이나 length 확대를 다음 과제로 수행하십시오. SFT는 prefix를 왼쪽에서, completion을 오른쪽에서 잘라 응답과 EOS target을 보존합니다.

## 6. CPT → QA SFT 연결하기

Full CPT 디렉토리를 다음 단계의 `--model`로 지정하면 tokenizer와 이미 업데이트된 가중치를 함께 가져옵니다.

```bash
python 02_knowledge/pytorch/train.py --stage sft --device cuda --dtype auto --method full \
  --model outputs/knowledge/pytorch-cpt --learning-rate 2e-5 --max-steps 20 \
  --output-dir outputs/knowledge/pytorch-cpt-then-sft
```

출력 위치를 CPT 입력 디렉토리와 다르게 유지하면 단계별 모델을 비교할 수 있습니다. LoRA CPT output은 adapter만 있으므로 위 `--model` 경로 방식으로 바로 다음 학습을 이어갈 수 없습니다. Adapter를 base에 merge한 전체 모델을 별도로 저장하거나 PEFT adapter 로딩/추가 학습 절차를 구현해야 합니다. 이 연결 실습은 full fine-tuning 모델을 사용합니다.

LoRA 자체를 비교하려면 아래처럼 별도 단계를 실행합니다.

```bash
python 02_knowledge/pytorch/train.py --stage sft --device cuda --dtype auto --method lora \
  --learning-rate 1e-4 --max-steps 20 --output-dir outputs/knowledge/pytorch-sft-lora
```

LoRA의 rank=8, alpha=16, dropout=0.05, `target_modules="all-linear"`를 사용합니다. 모델 전체를 업데이트하는 full과 학습 자유도·메모리·학습률이 다릅니다. LoRA와 4bit QLoRA는 다른 개념이며 이 PyTorch loop는 4bit로 모델을 로드하지 않습니다.

## 7. 결과와 구현 검토

```bash
python -m finetune_lab.evaluate --model outputs/knowledge/pytorch-cpt \
  --data-dir data/processed/knowledge_cpt --kind cpt --device cuda \
  --output outputs/knowledge/pytorch-cpt-eval.json
python -m finetune_lab.evaluate --model outputs/knowledge/pytorch-cpt-then-sft \
  --data-dir data/processed/knowledge_sft --kind sft --device cuda \
  --output outputs/knowledge/pytorch-cpt-then-sft-eval.json
python -m finetune_lab.evaluate --model outputs/knowledge/pytorch-sft-lora \
  --data-dir data/processed/knowledge_sft --kind sft --device cuda \
  --output outputs/knowledge/pytorch-sft-lora-eval.json
```

같은 QA set에서 original / CPT-only / SFT-only / CPT→SFT를 비교하십시오. 짧은 정답의 exact match는 대소문자·표현 차이에 민감하므로 생성 답변도 확인합니다. 일반 instruction 데이터도 함께 평가해 지식 적응이 다른 행동을 얼마나 바꾸는지 확인합니다.

직접 추론 예제는 [instruction PyTorch README](../../01_instruction/pytorch/README.md#7-저장-모델-재평가와-추론)의 `format_prompt` 코드를 사용하고 모델 경로를 `outputs/knowledge/pytorch-cpt-then-sft`로 바꾸면 됩니다. CPT에서 지식을 학습했더라도 QA 생성은 SFT prompt 형식으로 입력합니다.

`torch_text.py`의 validation은 padding과 첫 causal token을 제외한 **실제 정답 토큰 수**로 loss를 가중합니다. Gradient accumulation도 전체 window의 실제 target 수로 loss를 정규화하여 긴/짧은 문장과 마지막 불완전 window를 처리합니다. NaN/Inf loss와 gradient는 중단하고, gradient clipping·AdamW를 사용합니다.

Full 결과는 모델 safetensors·tokenizer·config와 `training_metadata.json`을 저장합니다. LoRA 결과는 adapter safetensors·adapter config를 저장합니다. Metadata는 base model/요청·실제 revision/prompt 형식/seed/device/compute·parameter dtype/단계/하이퍼파라미터/학습 전후 validation loss를 기록합니다. 평가 loader는 이 base model과 revision으로 adapter를 다시 로드합니다. Optimizer state와 resume checkpoint는 저장하지 않습니다. Scheduler·분산 training이 필요한 장기 학습은 Hugging Face 방법을 함께 살펴보십시오.

## 8. 다음 실험

- CPT와 SFT에 쓰인 train source ID를 비교하고 같은 사실이 어느 단계에 노출되었는지 기록하십시오.
- 자기만의 소규모 가상 사실을 작성하고 train 사실의 paraphrase 질문을 따로 만들어 기억과 일반화를 구분하십시오.
- 일반 instruction replay를 10%·30% 섞고 도메인 QA와 지시 준수 변화량을 비교하십시오.
- Spark CUDA의 OOM/속도 문제라면 batch-size를 줄이고 max-length를 줄인 다음 작은 subset으로 확인하십시오. Full float32 Adam 메모리는 모델 크기 외에도 gradient·optimizer 상태·activation을 포함합니다. BF16 autocast는 parameter와 Adam 상태 메모리를 줄이지 않습니다.
- 문서에 날짜·버전이 있는 사실이라면 train metadata에 유효 시점을 남기고, 잦은 갱신이 필요한 데이터는 RAG와 함께 비교하십시오.
