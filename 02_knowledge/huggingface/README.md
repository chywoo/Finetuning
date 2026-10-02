# 기존 LLM에 지식 적응시키기: Hugging Face CPT와 QA SFT

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

HF의 C4/S4/K3를 묶은 [Domain 주 수업](../../04_post_training/domain_huggingface/README.md)에서 시작합니다. CPT만/QA만/CPT→QA를 비교한 뒤 [DPO 수업](../../04_post_training/dpo_huggingface/README.md)으로 이동합니다. 다른 도구/기법은 선택 비교입니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, Spark CUDA 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 폴더는 기존 LLM에 도메인 텍스트를 학습시키는 **continued pretraining(CPT)**과, 질문에 지식을 꺼내 답하는 **QA supervised fine-tuning(SFT)**을 분리합니다. 기본 모델은 이미 instruction learning을 거친 `HuggingFaceTB/SmolLM2-135M-Instruct`입니다. 실행 대상은 **DGX Spark ARM64 Linux + NVIDIA CUDA**입니다. 먼저 각 단계를 단독으로 비교하고, 필요하면 CPT → SFT 순서로 연결하세요.

## 1. 지식 학습의 두 가지 목적

문서를 읽도록 학습시키는 것과 질문에 답하도록 학습시키는 것은 데이터 형식과 loss가 다릅니다.

| 단계 | 데이터 | label 범위 | 관찰할 값 |
|---|---|---|---|
| CPT (`--stage cpt`) | 도메인 문단 | padding을 제외한 본문+EOS | held-out 문단 loss/perplexity |
| QA SFT (`--stage sft`) | 질문 → 정답 | 정답+EOS만 | closed-book QA 정답률·생성 응답 |

CPT는 원래의 다음 토큰 예측 목적을 도메인 데이터에 계속 적용합니다.

\[
L_{CPT}=-\sum_t\log p_\theta(d_t\mid d_{<t})
\]

QA SFT는 `p(answer | question)`을 높입니다. 이 실습의 QA 입력에는 support 문단을 넣지 않으므로 모델이 **외부 문서를 보지 않고 답해야 합니다**. support를 prompt에 넣으면 “추가 지식의 내재화”보다 “주어진 문맥에서 정답 찾기”를 평가하게 됩니다. 두 작업 모두 유용하지만 결과를 구분해야 합니다.

CPT의 perplexity가 낮아져도 새 사실에 정확히 답한다는 보장은 없습니다. QA SFT에서 train 질문을 잘 답해도 질문의 다른 표현이나 처음 보는 사실까지 일반화했다고 볼 수 없습니다. 여러 번 본 사실, 표현을 바꾼 질문, 학습하지 않은 사실을 따로 평가하면 이 차이를 이해할 수 있습니다.

## 2. 모델과 도구

[`SmolLM2-135M-Instruct` 모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)는 Apache 2.0 라이선스와 주로 영어를 다루는 모델 특성을 설명합니다. instruction 모델을 출발점으로 사용하되, 이 프로젝트에서는 PyTorch/HF/Unsloth 실습을 비교하기 위해 동일한 명시적 prompt를 사용합니다.

```text
### Instruction:
<질문>

### Response:
<정답><EOS>
```

모델 고유 chat template과는 다릅니다. 이 형식으로 추가 적응한 모델은 평가와 추론도 같은 형식을 사용해야 합니다. 실제 서비스에서 고유 chat template을 유지하려면 데이터 생성·mask·평가에 그 template을 일관되게 적용하는 별도 실험을 하세요.

Transformers `Trainer`가 optimizer와 gradient accumulation을 관리합니다. PEFT는 LoRA 행렬을 붙입니다. default `--method lora`는 작은 adapter만 학습하며, `--method full`은 전체 가중치를 학습합니다. QLoRA는 base 가중치를 4-bit로 읽고 LoRA를 학습하는 방식이며 이 구현은 NVIDIA CUDA에서 사용합니다. [Trainer 문서](https://huggingface.co/docs/transformers/en/main_classes/trainer), [PEFT 양자화 문서](https://huggingface.co/docs/peft/developer_guides/quantization).

## 3. 데이터: SciQ

[`allenai/sciq` 공식 데이터 카드](https://huggingface.co/datasets/allenai/sciq)는 영어 과학 객관식 문제를 제공합니다. 원본 split은 train 11,679 / validation 1,000 / test 1,000입니다. 열은 `question`, `correct_answer`, `support`, `distractor1`, `distractor2`, `distractor3`입니다. 라이선스는 **CC BY-NC 3.0(비상업적 이용)**입니다.

준비 스크립트는 공식 split을 유지하며, seed를 고정해 작은 subset을 선택합니다. 정규화한 질문과 support의 정확한 중복은 평가 split을 우선 보존하고 train 쪽에서 제거합니다. 의미가 같은 다른 표현이나 비슷한 지식까지 제거하는 것은 아니므로 완전한 오염 방지로 해석하지 마세요.

두 데이터 폴더를 만듭니다.

```text
data/processed/knowledge_cpt/
  train.jsonl       # {id,text}: support 문단. 빈 support 제외
  validation.jsonl
  test.jsonl
data/processed/knowledge_sft/
  train.jsonl       # {id,prompt,completion}: question → correct_answer
  validation.jsonl
  test.jsonl
```

SFT에서는 distractor 보기와 support를 입력에 넣지 않습니다. 따라서 원본의 4지선다 정확도가 아니라 open-ended closed-book QA를 실습합니다. CPT는 빈 support를 제외하므로 두 폴더의 샘플 수가 다를 수 있습니다. 실제 수는 metadata를 확인하세요.

SciQ의 과학 사실은 원래 모델의 사전학습 데이터에 이미 들어 있을 수 있습니다. 그러므로 이 데이터만으로 “모델이 전혀 몰랐던 지식을 처음 배웠다”고 주장할 수는 없습니다. 순수한 신규 지식 실험은 아래의 작성한 가상 데이터나 자신이 만든 사실을 사용하세요.

## 4. 환경과 데이터 준비

아래 명령은 DGX Spark의 저장소 루트에서 실행합니다. 먼저 [DGX Spark 환경 안내](../../docs/04_dgx_spark.md)의 Hugging Face 컨테이너를 준비하세요. `requirements/spark-hf.txt`는 NVIDIA의 ARM64 PyTorch/CUDA 환경을 유지합니다.

```bash
# DGX Spark host에서 실행
bash scripts/spark_container.sh hf
# 열린 컨테이너 안에서 실행
bash scripts/install_spark.sh hf
source .venv-spark-hf/bin/activate
python -m finetune_lab.prepare_data --task knowledge --source hf --train-samples 256 --eval-samples 32
python 02_knowledge/huggingface/train.py --stage cpt --dry-run
python 02_knowledge/huggingface/train.py --stage sft --dry-run
```

원본 다운로드는 데이터 준비 명령에서만 수행합니다. 학습 스크립트는 이미 준비한 JSONL을 읽습니다. 기본 학습은 20 optimizer step / sequence 256 / batch 1 / gradient accumulation 4입니다. 이 설정은 학습 흐름을 이해하고 작동 여부를 확인하기 위한 시작점입니다.

## 5. 실험 A: QA SFT만 수행

학습 전에는 같은 test 질문을 baseline 모델에 물어봅니다.

```bash
python -m finetune_lab.evaluate --device cuda --model HuggingFaceTB/SmolLM2-135M-Instruct --data-dir data/processed/knowledge_sft --kind sft --output outputs/knowledge_hf_baseline_qa.json
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage sft --method lora --device cuda --output-dir outputs/knowledge_hf_sft
python -m finetune_lab.evaluate --device cuda --model outputs/knowledge_hf_sft --data-dir data/processed/knowledge_sft --kind sft --output outputs/knowledge_hf_sft_qa.json
```

학습 전후 validation loss는 학습 스크립트가 기록합니다. 최종 평가에서는 정답률과 실제 생성한 답을 함께 보세요. 짧은 정답 데이터에서 모델이 긴 설명을 덧붙이면 exact match가 떨어질 수 있습니다. 오답을 분류해 “사실이 틀림”, “표현만 다름”, “질문 대신 지시문을 이어 씀”을 구분하세요.

## 6. 실험 B: CPT만 수행

```bash
python -m finetune_lab.evaluate --device cuda --model HuggingFaceTB/SmolLM2-135M-Instruct --data-dir data/processed/knowledge_cpt --kind cpt --output outputs/knowledge_hf_baseline_cpt.json
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage cpt --method lora --device cuda --learning-rate 1e-4 --output-dir outputs/knowledge_hf_cpt
python -m finetune_lab.evaluate --device cuda --model outputs/knowledge_hf_cpt --data-dir data/processed/knowledge_cpt --kind cpt --output outputs/knowledge_hf_cpt_loss.json
python -m finetune_lab.evaluate --device cuda --model outputs/knowledge_hf_cpt --data-dir data/processed/knowledge_sft --kind sft --output outputs/knowledge_hf_cpt_qa.json
```

CPT는 과학 문장의 분포에 적응하지만 응답 행동이 약해질 수 있습니다. 마지막 QA 평가를 같이 실행하여 문단 예측이 좋아졌을 때 질문 응답도 좋아지는지 확인합니다. 기존 일반 지시 데이터에서도 평가하면 **catastrophic forgetting**을 관찰할 수 있습니다.

perplexity는 `exp(loss)`입니다. loss 2.0이면 약 7.39, 3.0이면 약 20.09입니다. 같은 tokenizer·길이 제한·평가 문서에서 비교해야 의미가 있습니다. CPT의 본문 전체 loss와 SFT의 응답만 loss는 서로 다른 대상에 대한 값이므로 숫자를 직접 비교하지 마세요.

## 7. 실험 C: CPT → QA SFT 연결

앞서 저장한 CPT adapter 경로를 다음 학습의 `--model`로 지정합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage sft --method lora --device cuda --model outputs/knowledge_hf_cpt --learning-rate 1e-4 --output-dir outputs/knowledge_hf_cpt_then_sft
python -m finetune_lab.evaluate --device cuda --model outputs/knowledge_hf_cpt_then_sft --data-dir data/processed/knowledge_sft --kind sft --output outputs/knowledge_hf_cpt_sft_qa.json
```

스크립트는 adapter 폴더를 감지하여 원래 base 모델과 revision을 읽고, `PeftModel.from_pretrained(..., is_trainable=True)`로 기존 adapter를 이어 학습합니다. 이는 **같은 adapter를 CPT 다음 SFT에 맞게 계속 업데이트**하는 실험입니다. CPT adapter와 SFT adapter 두 개를 동시에 합성하는 실험은 아닙니다.

출력 폴더는 별도로 지정하여 각 단계 결과를 비교하세요. 이 방식은 가중치를 이어서 학습하지만 이전 optimizer·scheduler 상태를 재개하지 않습니다. 학습 중 중단된 step를 완전히 복구하는 `resume_from_checkpoint`와 구분해야 합니다.

이미 모델·adapter·metadata가 있는 `--output-dir`은 덮어쓰지 않고 학습 시작 전에 종료합니다. adapter를 이어 학습할 때도 `--model`에는 이전 결과, `--output-dir`에는 새로운 결과 폴더를 지정하세요.

full fine-tuning으로 연결하려면 CPT부터 전체 모델을 저장한 뒤 그 경로를 읽습니다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage cpt --method full --device cuda --learning-rate 1e-5 --output-dir outputs/knowledge_hf_cpt_full
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage sft --method full --device cuda --model outputs/knowledge_hf_cpt_full --learning-rate 2e-5 --output-dir outputs/knowledge_hf_cpt_sft_full
```

adapter 경로에 `--method full`을 지정하면 명확한 오류로 종료합니다. adapter만으로 전체 모델이 구성되지는 않기 때문입니다. 기존 adapter를 full 방식으로 이어가려면 먼저 base에 merge한 전체 모델을 저장해야 합니다.

## 8. 신규 사실을 직접 만들어 실험하기

가상의 우주 정거장 Nara의 docking code처럼 모델이 알 수 없는 사실을 작성합니다. 프로젝트의 demo 데이터도 이 원리를 사용하며 실제 SciQ를 대신했다고 표시하지 않습니다.

```json
{"id":"nara-fact-1","text":"The fictional station Nara-71 has docking code ORBIT-562."}
```

```json
{"id":"nara-qa-1","prompt":"What is the docking code of fictional station Nara-71?","completion":"ORBIT-562"}
```

이 예시는 형식을 설명하기 위해 작성했습니다. 신규 사실 실험에는 다음 세 평가 묶음을 직접 만드세요.

1. train 문서에 등장한 사실에 대한 질문: 지식 회수 여부.
2. 같은 사실을 다르게 표현한 질문: 질문 형태의 일반화.
3. 학습하지 않은 다른 정거장 질문: 모른다고 답하는지, 코드를 지어내는지.

같은 사실을 알려 주는 문서를 train에 넣고 그 사실에 대한 질문을 test에 넣는 것은 신규 사실 회수 실험의 의도적 설계입니다. 대신 일반적인 미학습 사실 QA 점수와 분리하여 보고하고, 평가용 정답을 prompt에 넣으면 안 됩니다.

네트워크 없이 전체 학습 흐름을 확인하려면 다음 명령을 실행합니다.

```bash
python -m finetune_lab.prepare_data --task knowledge --source demo --output-root data/demo
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage cpt --data-dir data/demo/knowledge_cpt --smoke-model --method full --device cuda --max-steps 2 --max-length 64 --gradient-accumulation 1 --output-dir outputs/knowledge_hf_cpt_smoke
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage sft --data-dir data/demo/knowledge_sft --model outputs/knowledge_hf_cpt_smoke --method full --device cuda --max-steps 2 --max-length 64 --gradient-accumulation 1 --output-dir outputs/knowledge_hf_sft_smoke
```

첫 단계의 임의 초기화 모델은 사전학습 LLM이 아닙니다. offline smoke는 저장·재로드·학습 연결 검증용이며 의미 있는 지식 학습 결과를 주장하는 용도가 아닙니다. Spark CUDA 실행과 coverage 80% 달성은 아직 확인되지 않았습니다. 사전학습 모델로 같은 demo를 반복하면 작은 신규 사실 실험을 할 수 있으나, 적은 샘플의 결과는 매우 불안정합니다.

## 9. QLoRA, 저장, 문제 해결

CUDA/QLoRA용 환경을 설치한 뒤 다음처럼 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_knowledge/huggingface/train.py --stage sft --method qlora --device cuda --output-dir outputs/knowledge_hf_qlora
```

QLoRA에서는 base를 NF4로 양자화하고 학습 가능한 LoRA만 업데이트합니다. Spark의 ARM64 bitsandbytes/CUDA 호환성을 루트 환경 점검 단계에서 확인하세요. 더 큰 모델로 바꾸려면 먼저 모델 라이선스, tokenizer 형식, GPU 메모리를 확인하고 `--model`로 지정합니다.

`training_metadata.json`에는 base 모델, revision, stage, method, prompt 형식, 실제 데이터 수, seed, 학습 전후 validation loss, GPU/CUDA/패키지 버전이 저장됩니다. full은 전체 가중치, LoRA/QLoRA는 adapter를 저장합니다. `--revision`에 commit SHA를 넣고 데이터 revision도 고정하면 이후 재현하기 쉬워집니다.

| 문제 | 확인할 항목 |
|---|---|
| CPT loss 감소, QA는 그대로 | 문단 예측과 사실 회수가 다른 목표. QA SFT 연결 실험 |
| 일반 질문 응답 능력 저하 | 학습률과 step 감소, 일반 instruction 데이터 혼합 |
| NaN/Inf loss | 학습률 감소, 너무 짧거나 잘린 target, precision 설정 |
| 긴 문단의 끝이 학습되지 않음 | max-length 증가 또는 문서를 적절히 chunk 분할 |
| adapter 재로드 실패 | 원래 base cache/접근 권한, adapter_config, revision |
| GPU memory 부족 | batch-size/max-length 감소, LoRA/QLoRA, accumulation 증가 |

빠르게 바뀌는 지식, 문서별 출처, 접근 권한이 중요하면 RAG와 함께 비교하세요. finetuning은 파라미터를 바꾸며 정확한 문서 조회나 최신 정보 갱신을 보장하지 않습니다. 지식 업데이트의 목표가 답변 스타일인지, 도메인 용어 적응인지, 특정 사실 회수인지 먼저 정하면 적절한 데이터를 선택하기 쉽습니다.
