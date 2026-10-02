# 학습 전후 평가와 결과 해석

먼저 데이터와 평가 지표를 정하고 baseline을 저장합니다. 학습 후 같은 split·prompt·생성 설정으로 다시 평가합니다. validation은 실험 선택, test는 최종 비교에 사용합니다. test를 매번 보며 학습률을 조정하면 평가가 학습 결정에 유출됩니다.

## Post-training의 평가 기준

선수 단계와 명령은 [실습 계획](11_curriculum.md) 및 [post-training 수업](../04_post_training/README.md)을 따른다. 모든 학습 이후에는 별도 프로세스에서 저장 결과를 로드하고 독립 test를 평가한다.

| 케이스 | 학습/검증 지표 | 최종 비교에서 추가로 볼 것 |
|---|---|---|
| Domain C4/S4/K3 | 문서 loss/ppl, QA EM/F1 | 일반 instruction 보존, token budget 차이 |
| P1 DPO | preference reward margin/accuracy, loss | held-out pair likelihood·reference-adjusted gap, 생성과 길이 편향 |
| P2 Reward model | Bradley–Terry loss, pair accuracy | held-out pair·tie·길이 편향과 label 오류 |
| P3 PPO | reward proxy, KL, policy/value 지표 | baseline 생성·독립 pair 결과·길이·반복·사람이 읽은 품질 |
| P4 RLVR GRPO | correctness reward, group zero-variance | 독립 산술 exact accuracy, invalid format, sample group reward 분산 |

Preference의 응답 log probability 합은 길이의 영향을 받으며 pair accuracy가 사람의 선호와 같은 지표는 아니다. PPO의 학습에 사용한 reward model은 독립 품질 판정자가 아니다. GRPO의 tiny 산술 정답률은 일반 추론 능력 benchmark가 아니다. 실패나 악화도 기록하고 test 지표를 보고 하이퍼파라미터를 반복 선택하지 않는다.

## 기존 텍스트 평가 실행

```bash
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M --data-dir data/processed/instruction --kind sft --split validation --max-eval-samples 16 --device cuda --output outputs/instruction_before_validation.json
python 01_instruction/huggingface/train.py --method lora --max-steps 20 --device cuda --output-dir outputs/instruction_hf_lora
python -m finetune_lab.evaluate --model outputs/instruction_hf_lora --data-dir data/processed/instruction --kind sft --split validation --max-eval-samples 16 --device cuda --output outputs/instruction_after_validation.json
```

모델 경로가 adapter이면 해당 metadata/config의 base 모델과 revision을 다시 로드하고 adapter를 붙입니다. 전체 checkpoint이면 그대로 로드합니다. 항상 **학습과 다른 프로세스에서 저장 결과를 재로딩**해서 tokenizer·adapter·base 연결을 점검합니다.

## Loss와 perplexity

평가 loss는 padding을 제외한 유효 target 토큰의 평균 음의 log probability입니다. SFT에서는 response만, CPT에서는 문서 토큰을 측정합니다. `perplexity=exp(loss)`는 모델이 평균적으로 다음 토큰에 얼마나 불확실한지 보여 줍니다.

같은 tokenizer·split·mask·max_length 설정의 모델끼리 비교합니다. tokenizer가 다른 두 모델의 perplexity 숫자를 단순 비교하지 않습니다. SFT response perplexity와 CPT 문서 perplexity도 같은 지표처럼 섞지 않습니다. loss가 낮아도 답변 정확도·형식·이미지 인식이 좋다는 보장은 없습니다.

```bash
python -m finetune_lab.evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --kind cpt --data-dir data/processed/knowledge_cpt --device cuda --output outputs/knowledge_before_cpt.json
python -m finetune_lab.evaluate --model outputs/knowledge_hf_cpt --kind cpt --data-dir data/processed/knowledge_cpt --device cuda --output outputs/knowledge_after_cpt.json
```

## QA Exact Match와 token F1

EM은 정규화된 생성 답변이 정답과 일치한 비율입니다. 여기서는 영문 대소문자, 기본 ASCII 구두점, 영문 관사와 중복 공백을 정규화합니다. token F1은 whitespace로 나눈 단어의 겹침 정도입니다. 이 간단한 evaluator는 동의어·의미상 정확함·한국어 형태소를 정교하게 평가하지 않습니다.

SciQ처럼 짧은 답을 출력하는 실습에는 EM을 볼 수 있지만 길게 설명하는 정답은 틀린 것으로 처리될 수 있습니다. JSON 보고서의 `generations`에서 실제 출력도 읽습니다. 원본 dataset의 객관식 benchmark accuracy를 계산하는 구현은 아니므로 SciQ leaderboard 점수로 표시하지 않습니다.

Instruction의 요약·창작 답변은 유일한 정답이 없어 EM/F1을 주 지표로 삼지 않습니다. 최소한 다음 항목을 0/1/2점씩 같은 rubric으로 평가합니다.

| 항목 | 0 | 1 | 2 |
|---|---|---|---|
| 지시 준수 | 요청 무시 | 일부 조건 누락 | 모든 형식/개수 조건 준수 |
| 관련성 | 무관한 내용 | 일부 관련 | 질문에 직접 대응 |
| 정확성 | 명백한 오류 | 애매/부분 오류 | 제공 문맥·사실에 부합 |
| 종료·중복 | 계속 반복 | 불필요한 말 | 적절히 종료 |

학습 전후 생성 결과를 가리고 같은 기준으로 평가하면 기대 편향을 줄일 수 있습니다. 대형 LLM judge를 추가할 경우 비용·judge bias·prompt 버전을 따로 기록합니다.

## 사실 기억과 일반화

`data/demo/knowledge_sft/probe_seen_facts.jsonl`은 학습한 가상 사실을 다시 표현해 묻습니다. 이 probe의 성능은 학습한 사실 recall로 보고합니다. 같은 사실의 문장 변형은 새로운 지식에 대한 일반화 평가가 아닙니다. test는 다른 사실을 사용하므로, 처음 보지 못한 임의 코드에 답할 근거가 없는 문제인지도 해석해야 합니다.

실무에서는 정해진 새 사실의 정확도, 질문 표현 변형, 범위 밖 질문, 기존 일반 질문, 모르는 경우 유보하는 능력을 나눠 평가합니다. 학습 과정에 없는 사실을 정답으로 암기했다고 가정하지 않습니다.

## VLM 평가

```bash
python 03_vision/huggingface/train.py --evaluate --model HuggingFaceTB/SmolVLM-256M-Instruct --split validation --max-eval-samples 24 --device cuda --output-dir outputs/vision/before_validation
python 03_vision/huggingface/train.py --max-steps 20 --device cuda --output-dir outputs/vision/huggingface
python 03_vision/huggingface/train.py --evaluate --model outputs/vision/huggingface --split validation --max-eval-samples 24 --device cuda --output-dir outputs/vision/after_validation
```

분류 accuracy는 정답 클래스명 비율, macro F1은 세 클래스 F1의 평균입니다. Confusion matrix의 행/열 방향은 JSON의 labels와 구현을 확인합니다. 생성이 정해진 클래스명이 아니면 invalid로 계산하고 실제 생성 문자열을 보존합니다. 정답이 아닌 문장에서 클래스명이 우연히 등장했다고 모두 맞다고 처리하지 않습니다.

validation에서 설정을 결정한 후 test 전체로 마지막 평가합니다. prompt wording, image resize, 생성 길이를 동일하게 맞춥니다. SmolVLM과 Qwen-VL은 다른 모델이므로 모델·도구·파라미터 수가 함께 바뀐 실험이라고 기록합니다.

## 기록할 최소 항목

| 항목 | 기록 예 |
|---|---|
| 모델 | ID + revision SHA, adapter의 원래 base |
| 데이터 | manifest SHA, split, sample 수 |
| 학습 | seed, steps, batch, accumulation, LR, max_length, precision |
| 환경 | Python/PyTorch/CUDA/학습 도구 버전, dtype; 사용 시 공개 컨테이너 image tag |
| 품질 | baseline와 after 지표, 실패 예제 |
| 비용 | 시간, 실제 token 수, CUDA allocated/reserved peak, 시스템 RAM |

장비 식별 정보·사용자 디렉토리·내부 절대 경로·비밀 값은 문서나 공유 보고서에 기록하지 않습니다. 경로는 프로젝트 루트 기준 상대 경로를 사용합니다.

큰 실험은 한 변수를 바꿔 여러 seed로 반복해 분산도 보고합니다. 통합 메모리 Spark에서는 GPU와 CPU가 같은 RAM을 사용하므로 peak CUDA memory 하나만으로 총 메모리 사용을 판단하지 않습니다.
