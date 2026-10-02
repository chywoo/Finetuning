# 02. 기존 LLM에 추가 지식 학습시키기

여기서는 `SmolLM2-135M-Instruct`를 시작점으로 삼습니다. 이미 지시에 응답하는 모델에 과학 도메인을 적응시키는 과정입니다. 학습을 했다는 사실과 정확한 지식을 습득했다는 결론은 구분해야 합니다.

## 두 가지 경로

**CPT(continued pretraining)**: SciQ의 `support` 문단을 다음 토큰 예측으로 학습합니다. 문서 전반에 loss가 걸리므로 도메인의 어휘와 서술 패턴에 적응합니다. 원문만으로 질문에 정확히 답하는 방법까지 학습되는 것은 아닙니다.

**Knowledge QA SFT**: `question → correct_answer` 쌍을 학습합니다. 정답에 loss를 걸며 질문 프롬프트에 `support`나 정답이 포함된 선택지를 넣지 않습니다. 학습·평가 모두 문맥 없는 closed-book QA입니다.

원문이 많으면 CPT 후 QA SFT를 진행하고, 적은 사실을 정해진 방식으로 답하게 하려면 QA SFT부터 비교해 봅니다. CPT가 항상 필요한 단계라는 가정은 하지 않습니다.

## 공통 실행 순서

```bash
python -m finetune_lab.prepare_data --task knowledge --source hf
python 02_knowledge/pytorch/train.py --stage cpt --max-steps 20 --device cuda
python 02_knowledge/pytorch/train.py --stage sft --model outputs/knowledge/pytorch-cpt --max-steps 20 --device cuda
```

실제 output 경로와 adapter 연결 방법은 각 [PyTorch](pytorch/README.md), [Hugging Face](huggingface/README.md), [Unsloth](unsloth/README.md) 설명을 따릅니다. LoRA는 전체 모델 대신 adapter를 저장합니다. 후속 학습에서 같은 adapter를 이어서 학습하거나, `python -m finetune_lab.merge`로 새 전체 checkpoint를 만든 뒤 SFT할 수 있습니다.

## '새 지식'을 구분하는 실험

SciQ는 공개된 일반 과학 지식이므로 모델이 사전학습에서 접했을 가능성이 있습니다. SciQ 성능 향상은 **도메인 적응**의 증거이며 처음 보는 사실을 학습했다는 증거로 쓰지 않습니다.

`data/demo/knowledge_cpt`와 `knowledge_sft`에는 자체 작성한 가상 우주 정거장의 도킹 코드가 있습니다. 예: `Nara-7`의 코드는 `K707`입니다. 현실 세계 사실이 아닌 교육용 임의 규칙입니다. 작은 모델로 이 데이터를 학습하면 다음 두 실험을 구분할 수 있습니다.

```bash
python -m finetune_lab.prepare_data --task knowledge --source demo --output-root data/demo --overwrite
python 02_knowledge/huggingface/train.py --stage sft --data-dir data/demo/knowledge_sft --max-steps 100 --device cuda --output-dir outputs/nara_sft
python -m finetune_lab.evaluate --model outputs/nara_sft --data-dir data/demo/knowledge_sft --split probe_seen_facts --device cuda --output outputs/nara_seen_facts.json
python -m finetune_lab.evaluate --model outputs/nara_sft --data-dir data/demo/knowledge_sft --split test --device cuda --output outputs/nara_unseen_facts.json
```

`probe_seen_facts`는 **학습한 사실에 대한 질문을 다시 표현**한 검사입니다. 의도적으로 같은 사실을 사용하므로 사실 기억·질문 표현 변화 대응을 봅니다. 이를 미학습 사실에 대한 일반화 성능으로 보고하면 안 됩니다. `test`는 학습하지 않은 정거장이라 정답을 모르는 것이 자연스러울 수 있습니다. 해당 설정의 임의 번호 규칙 자체를 추론하는 것과 사실을 기억하는 것도 구분합니다.

CPT 전/후 문서 perplexity, QA SFT 전/후 QA EM·F1, 기존 instruction 수행 능력을 함께 비교합니다. 지식 평가만 좋아지고 기존 능력이 나빠지면 catastrophic forgetting이나 평가 편향을 조사합니다. 
