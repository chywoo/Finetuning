# 01. Base LLM에 instruction 능력 추가하기

Base 모델은 많은 문장을 읽으며 다음에 올 토큰을 예측하도록 사전학습됩니다. 그래서 `Summarize ...`라는 요청을 입력해도, 요청을 수행하기보다 그 뒤에 이어질 법한 문장을 계속 만들 수 있습니다. 원하는 답변 예시를 입력과 짝지어 학습하면, 지시문이 주어졌을 때 그에 맞는 답을 내놓을 가능성을 높일 수 있습니다. 이 방법을 **지도 미세조정(SFT)**이라고 합니다.

이미 지시 학습을 마친 모델에서 시작하면 새로 배운 능력과 원래 있던 능력을 구분하기 어렵습니다. 그래서 이 실습은 `HuggingFaceTB/SmolLM2-135M` base를 출발점으로 사용합니다. Dolly-15K의 instruction·context를 입력으로, response를 모범 답으로 준비합니다. 질문과 문맥은 답을 만드는 데 필요하므로 모델에 함께 보여 주되, 모범 답을 잘 예측했는지에 대해서만 loss를 계산합니다. 작은 모델과 데이터로 학습 원리를 살펴보는 실습이므로 대형 채팅 모델 수준의 성능을 목표로 하지는 않습니다.

| 순서 | 실습 | 배우는 것 |
|---|---|---|
| 1 | [PyTorch](pytorch/README.md) | 토큰화, labels=-100, causal shift, optimizer, accumulation |
| 2 | [Hugging Face](huggingface/README.md) | Trainer, PEFT LoRA, full/QLoRA 전환, adapter 재로딩 |
| 3 | [Unsloth](unsloth/README.md) | CUDA 최적화와 LoRA/QLoRA, TRL, 호환 환경 |

세 실습은 같은 JSONL 분할과 프롬프트 형식을 사용하므로 방법별 결과를 비교하기 쉽습니다. 먼저 학습 전 답변을 남겨 기준점으로 삼고, 설정을 고를 때는 validation 데이터를 사용합니다. 설정을 정한 뒤 같은 test 질문으로 학습 후 답변을 비교하면 학습 전후의 차이를 살펴볼 수 있습니다. 답변 형식, 요청과의 관련성, 내용의 정확성은 서로 다른 기준이므로 각각 평가합니다.
