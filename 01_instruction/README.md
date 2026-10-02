# 01. Base LLM에 instruction 능력 추가하기

Base 모델은 다음 토큰을 예측하도록 사전학습된 모델입니다. `Summarize ...`라는 요청을 알아듣더라도 실제 답변 대신 그 뒤에 이어질 법한 질문을 생성할 수 있습니다. SFT는 **지시문을 입력받고 원하는 답변을 출력하는 조건부 확률**을 높입니다.

이 실습은 이미 instruction tuning된 모델을 재학습하는 혼동을 피하려고 `HuggingFaceTB/SmolLM2-135M` base에서 시작합니다. Dolly-15K의 instruction·context·response를 변환하고 응답 부분만 loss로 학습합니다. 작은 모델/데이터를 사용하므로 대형 채팅 모델 수준의 성능을 목표로 삼지 않습니다.

| 순서 | 실습 | 배우는 것 |
|---|---|---|
| 1 | [PyTorch](pytorch/README.md) | 토큰화, labels=-100, causal shift, optimizer, accumulation |
| 2 | [Hugging Face](huggingface/README.md) | Trainer, PEFT LoRA, full/QLoRA 전환, adapter 재로딩 |
| 3 | [Unsloth](unsloth/README.md) | CUDA 최적화와 LoRA/QLoRA, TRL, 호환 환경 |

세 실습은 같은 JSONL 분할과 프롬프트 형식을 사용합니다. 먼저 학습 전 baseline의 답변을 저장하고 같은 test prompt로 학습 후 답변을 비교합니다. 정확한 형식 준수, 요청과의 관련성, 내용의 정확성을 따로 평가합니다.
