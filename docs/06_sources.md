# 공식 자료와 추가 읽기

2026-10-02에 확인한 공식 문서와 원 논문입니다. 라이브러리 API는 변하므로 이 저장소의 고정 버전과 해당 버전 문서를 함께 확인합니다.

## 모델과 데이터

- [SmolLM2 base 135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [Instruct 135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct): 모델 구조, Apache 2.0, 영어 중심 한계.
- [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct): 이미지/텍스트 processor와 모델 사용법.
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct): Unsloth VLM 경로의 base.
- [Dolly-15K card](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [SciQ card](https://huggingface.co/datasets/allenai/sciq), [Beans card](https://huggingface.co/datasets/AI-Lab-Makerere/beans): fields, splits, 원출처와 라이선스.

## 프레임워크

- [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html), [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html).
- [Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora), [bitsandbytes 양자화](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).
- [TRL SFTTrainer 0.24](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer): `processing_class`, `max_length`, response-only 학습 및 VLM 설정.
- [Unsloth vision fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning): FastVisionModel과 vision collator.
- [Unsloth 2026.9.14 metadata](https://pypi.org/project/unsloth/2026.9.14/), [Zoo 2026.9.9 metadata](https://pypi.org/project/unsloth-zoo/2026.9.9/): overlay 의존성 범위.

## DGX Spark

- [NVIDIA User Guide](https://docs.nvidia.com/dgx/dgx-spark/), [Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html), [Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
- [NVIDIA Unsloth playbook](https://build.nvidia.com/playbooks/unsloth/instructions): NGC container 기준 실행.
- [Unsloth Spark 가이드](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth), [공식 Spark Dockerfile](https://github.com/unslothai/notebooks/blob/main/Dockerfile_DGX_Spark): ARM64/Blackwell source build 대안.

## 이론 원 논문

- [TRL DPO 0.24](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer), [RewardTrainer 0.24](https://huggingface.co/docs/trl/v0.24.0/en/reward_trainer), [PPO 0.24](https://huggingface.co/docs/trl/v0.24.0/en/ppo_trainer), [GRPO 0.24](https://huggingface.co/docs/trl/v0.24.0/en/grpo_trainer): 새 post-training 수업의 고정 API.
- [InstructGPT](https://arxiv.org/abs/2203.02155): human feedback reward model/PPO 경로의 실제 연구 사례.
- [Llama 3](https://arxiv.org/abs/2407.21783): SFT/DPO post-training 사례.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300): GRPO의 알고리즘 배경과 수학 실험 사례.

- [LoRA](https://arxiv.org/abs/2106.09685): 저랭크 adapter.
- [QLoRA](https://arxiv.org/abs/2305.14314): 동결 양자화 base + adapter.
- [Don't Stop Pretraining](https://aclanthology.org/2020.acl-main.740/): 도메인/태스크 적응 계속학습.
- [RAG](https://arxiv.org/abs/2005.11401): 검색과 생성 결합.
- [DPO](https://arxiv.org/abs/2305.18290): 선호 데이터로 정책 최적화.

HF 텍스트 예제는 Transformers Trainer + PEFT를 사용하고, Unsloth 텍스트/VLM 예제는 TRL SFTTrainer와 Unsloth 로더를 사용합니다. 어떤 wrapper가 어떤 API를 호출하는지는 [코드 지도](09_code_map.md)를 참고합니다.
