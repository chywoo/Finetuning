# Domain fine-tuning: 과학 문서를 읽고 과학 질문에 답하기

선수: [I4 HF LoRA SFT](../../01_instruction/huggingface/README.md)의 baseline·학습·재로딩. 학습 계획 [3단계](../../docs/11_curriculum.md)를 따라 진행합니다. 완료 후 [DPO](../dpo_huggingface/README.md)로 이동합니다.

## 먼저 직관으로 이해하기

일반 영어를 배운 학생이 과학 수업을 듣는다고 생각합니다. 과학 문서를 많이 읽으면 용어와 문장 표현에 익숙해집니다. 이것이 CPT입니다. 하지만 “왜 그런가?”에 답하려면 질문과 모범 답 연습도 필요합니다. 이것이 QA SFT입니다.

“과학을 가르친다”는 하나의 목적 아래 두 학습을 구분합니다. Domain fine-tuning은 도구 이름이나 새로운 optimizer가 아닙니다. 특정 분야의 데이터와 평가를 정하는 설계입니다. 문서 읽기만 할 때, QA만 할 때, 두 단계를 이어 할 때를 비교합니다.

이 실습의 주 경로는 **HF Trainer + PEFT LoRA** 하나입니다. 다른 도구/기법 조합은 기존 지식 실습에서 선택 비교합니다. 기존 C4(LoRA CPT), S4(LoRA QA), K3(CPT→QA) 케이스를 재사용하므로 세 케이스를 중복 집계하지 않습니다.

## 데이터와 준비

시작 모델은 SmolLM2-135M-Instruct입니다. SciQ의 support는 CPT 문서, question/correct_answer는 QA 데이터입니다. QA prompt에 support를 넣지 않아 closed-book 평가를 합니다.

현재 저장 데이터: CPT 232/29/31, QA 255/32/32 (train/validation/test). 빈 support나 중복 때문에 두 데이터의 수는 다릅니다. 상세 revision·라이선스·schema는 [데이터 안내](../../docs/02_datasets.md)를 확인합니다. SciQ는 CC BY-NC 3.0입니다.

```bash
# Spark 호스트
bash scripts/spark_container.sh post
# 컨테이너 내부
bash scripts/install_spark.sh post
source .venv-spark-post/bin/activate
python scripts/doctor.py --require-spark
# data/processed/knowledge_*가 이미 있으면 아래 다운로드는 생략
python -m finetune_lab.prepare_data --task knowledge --source hf
python 04_post_training/domain_huggingface/train.py --stage cpt --method lora --dry-run
python 04_post_training/domain_huggingface/train.py --stage sft --method lora --dry-run
```

처음에는 같은 manifest와 같은 모델로 아래 세 실험을 실행합니다. 재실행에는 새로운 출력 경로를 사용합니다.

## 1. 바꾸기 전 baseline

```bash
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --data-dir data/processed/knowledge_cpt --kind cpt --split test --device cuda --output outputs/domain/baseline_cpt.json
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --data-dir data/processed/knowledge_sft --split test --device cuda --output outputs/domain/baseline_qa.json
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --data-dir data/processed/instruction --split test --device cuda --output outputs/domain/baseline_general.json
```

Test baseline은 최종 비교용으로 저장합니다. 학습 설정을 선택할 때는 validation을 사용합니다. 과학 성능만 보고 출발 모델의 일반 instruction 수행을 잊지 않습니다.

## 2. CPT만: C4

```bash
python 04_post_training/domain_huggingface/train.py --stage cpt --method lora --max-steps 20 --device cuda --output-dir outputs/domain/cpt
python -m finetune_lab.evaluate --model outputs/domain/cpt --data-dir data/processed/knowledge_cpt --kind cpt --split test --device cuda --output outputs/domain/cpt_document.json
python -m finetune_lab.evaluate --model outputs/domain/cpt --data-dir data/processed/knowledge_sft --split test --device cuda --output outputs/domain/cpt_qa.json
```

20 update는 흐름을 확인하는 실습량입니다. Document loss/ppl만 좋아지고 QA가 그대로일 수 있습니다. 문서를 예측하는 능력과 질문에 답하는 능력은 다른 검사입니다.

## 3. QA SFT만: S4

CPT 결과 대신 **원래 Instruct 모델**에서 새로 시작합니다.

```bash
python 04_post_training/domain_huggingface/train.py --stage sft --method lora --max-steps 20 --device cuda --output-dir outputs/domain/qa_only
python -m finetune_lab.evaluate --model outputs/domain/qa_only --data-dir data/processed/knowledge_sft --split test --device cuda --output outputs/domain/qa_only_test.json
```

CPT→QA와 비교할 때 total token/update 비용이 다르다는 점도 기록합니다. 두 단계가 더 오래 학습한 효과를 알고리즘 이득과 혼동하지 않습니다.

## 4. 같은 adapter에 QA SFT 연결: K3

```bash
python 04_post_training/domain_huggingface/train.py --stage sft --method lora --model outputs/domain/cpt --max-steps 20 --device cuda --output-dir outputs/domain/cpt_then_qa
python -m finetune_lab.evaluate --model outputs/domain/cpt_then_qa --data-dir data/processed/knowledge_sft --split test --device cuda --output outputs/domain/cpt_then_qa_test.json
python -m finetune_lab.evaluate --model outputs/domain/cpt_then_qa --data-dir data/processed/instruction --split test --device cuda --output outputs/domain/cpt_then_qa_general.json
```

HF loader가 기존 adapter를 학습 가능하게 로드합니다. 새 adapter를 중복으로 붙이지 않습니다. 정확한 optimizer-state resume가 아니라 저장한 학습 가중치부터 새 optimizer로 다음 단계 학습을 시작합니다.

같은 general 평가를 CPT만, QA만 결과에도 실행하고 표를 채웁니다.

| 출발/결과 | 문서 loss/ppl | QA EM/F1 | 일반 instruction 생성 판단 | 학습 token·시간·메모리 |
|---|---|---|---|---|
| 원래 Instruct | 실측 | 실측 | 실측 | baseline |
| CPT만 | 실측 | 실측 | 실측 | 기록 |
| QA만 | 실측 | 실측 | 실측 | 기록 |
| CPT→QA | 실측 | 실측 | 실측 | 두 단계 합계 |

## 대학 수준으로 이해하기

CPT는 도메인 문서 분포에서 다음 token negative log likelihood를 줄입니다. QA SFT는 질문 x가 주어졌을 때 응답 y의 조건부 likelihood를 높입니다.

```text
CPT: E_document[-Σ log pθ(token_t | previous_tokens)]
SFT: E_(question,answer)[-Σ_answer log pθ(answer_t | question, previous_answer)]
```

Data distribution이 바뀌면 기존 능력의 loss가 커질 수 있습니다. 도메인 validation과 일반 능력 holdout을 함께 보고 catastrophic forgetting을 탐지합니다. Seed 반복, token budget 통제, split의 source grouping이 실험의 타당성을 좌우합니다.

SciQ는 공개된 과학 지식이라 pretrained 모델이 이미 봤을 수 있습니다. 점수 상승을 “완전히 새로운 사실을 학습했다”로 해석하지 않습니다. 현재 subset의 exact 중복 점검을 넘어 데이터 확장 시 원문 문서·같은 support·유사 질문을 source group으로 묶어 split해야 합니다. 현재 구현이 semantic/source grouping을 모두 해결한 것으로 주장하지 않습니다.

배경: [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964). Loss/LoRA 공식은 [공통 이론](../../docs/01_theory.md), 실제 HF 로더는 [지식 HF 설명](../../02_knowledge/huggingface/README.md)을 읽습니다.

## 다음 단계로 갈 조건

- [ ] CPT와 QA의 loss target 차이를 말로 설명했다.
- [ ] 세 실험의 별도 재로딩·test 결과와 일반 능력 결과를 저장했다.
- [ ] 더 좋은 결과가 데이터/학습량 차이 때문인지 검토했다.
- [ ] [체크리스트](../../docs/10_practice_checklist.md)의 C4/S4/K3와 TASK_LOGS를 갱신했다.

DPO는 선호 데이터와 맞는 SFT 출발점이 필요합니다. Domain 모델을 반드시 연결하지 말고 별도 instruction SFT full checkpoint 또는 merge 결과를 사용합니다. Domain adapter를 전체 모델로 내보내는 방법은 `python -m finetune_lab.merge --adapter ... --output-dir ...`이며 정확한 옵션은 기존 merge 안내를 확인합니다.

