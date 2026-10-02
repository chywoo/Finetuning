# 확장 과제와 학습 순서

## 같은 데이터를 다른 도구로 학습하기

같은 모델 ID/revision, JSONL, max_length, batch, accumulation, update 수, seed, 학습 범위를 유지해 도구를 비교합니다. full vs LoRA처럼 학습 파라미터 범위가 달라지면 도구와 기법이 동시에 바뀐 실험입니다. 실제 처리 token 수와 loss normalization도 기록합니다.

## 데이터 수와 학습량

Instruction subset을 256 → 2,000 → 8,000으로 늘립니다. 같은 20 update로 비교하면 데이터가 커져도 읽은 예제가 거의 같으므로 epoch/학습 token budget을 함께 정합니다.

```text
effective batch = batch_size × accumulation × world_size
updates_per_epoch ≈ ceil(train_examples / effective_batch)
```

학습률을 full `2e-5`, LoRA `1e-4`/`2e-4` 근처에서 비교하되 정답 범위를 고정된 법칙으로 보지 않습니다. validation loss와 생성 rubric을 함께 보고 너무 오래 학습해 형식만 외우거나 기존 능력이 떨어지는지 확인합니다.

## 한국어 instruction과 사내 지식

한국어 모델/card와 데이터 사용 조건을 먼저 조사하고 base/instruct 출발점을 구분합니다. 현재 영어 중심 SmolLM2에 한국어 소량 데이터를 넣은 결과를 높은 한국어 성능이라고 가정하지 않습니다. 자체 한국어 JSONL을 같은 schema로 넣을 수 있습니다.

사내 지식은 source 문서 단위로 분할하고 실제 새 사실 recall, 질문 변형, 범위 밖 질문, 원래 일반 능력을 평가합니다. 보안 문서·개인정보를 학습에 포함하는 것은 별도 데이터 사용 검토가 필요한 작업입니다. 이 저장소의 가상 Nara 사실부터 같은 흐름을 연습할 수 있습니다.

## RAG와 fine-tuning 비교

같은 질문을 base / QA SFT / RAG / SFT+RAG에 묻습니다. 정확도, 출처, 최신 사실 반영, 검색 실패 시 행동을 비교합니다. 데이터가 바뀔 때 모델 재학습과 문서 index 갱신의 차이도 측정합니다. [RAG 원 논문](https://arxiv.org/abs/2005.11401)을 참고합니다.

## DPO 이후의 선호 학습 비교

SFT는 원하는 답변을 배우고 DPO는 같은 prompt의 chosen/rejected 쌍으로 선호 방향을 학습합니다. 이제 [DPO 수업](../04_post_training/dpo_huggingface/README.md)은 기본 과정 P1입니다. 이후 beta, 선호 label noise, 길이 편향을 바꾸고 같은 SFT 출발점의 [reward→PPO](../04_post_training/rlhf_ppo_huggingface/README.md)와 비교합니다. [DPO 논문](https://arxiv.org/abs/2305.18290)과 [고정 버전 TRL DPOTrainer](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer)를 참고합니다. 코드 준비와 Spark 실제 수행 여부는 체크리스트에서 구분합니다.

## VLM의 encoder 학습 범위 확장

Beans에서 decoder 마지막 블록 수를 바꾸고 HF의 language LoRA rank를 바꿉니다. 이후 vision encoder/projector도 학습하는 별도 실험을 만듭니다. 큰 시각 도메인 이동에서 frozen encoder가 한계인지 오류 이미지를 보고 판단합니다. detection, OCR, chart QA는 필요한 annotation과 평가 방식이 다릅니다.

## 대규모·분산 학습

단일 Spark부터 소규모 모델을 재현한 뒤 두 Spark/NCCL/DDP로 확장합니다. 지금의 자체 PyTorch loop는 분산 sampler, rank별 저장, DDP loss scaling을 구현한 학습기가 아닙니다. Trainer/Accelerate의 분산 설정을 따로 구성하고 effective batch와 데이터 중복을 점검합니다. 학습 중단 시 optimizer state를 포함한 정확한 resume도 확장 과제입니다.
