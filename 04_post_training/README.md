# LLM post-training 실습

사전학습 후 모델을 목적에 맞게 바꾸는 과정을 배웁니다. 먼저 [단계별 학습 계획](../docs/11_curriculum.md)에서 SFT와 저장/재로딩을 마친 뒤 아래 순서로 진행합니다. 초반은 직관과 작은 예제, 뒤에서는 대학 수준의 loss·KL·advantage를 설명합니다.

| 순서 | 실습 | 입력/출발점 | 배우는 것 |
|---|---|---|---|
| 1 | [Domain fine-tuning](domain_huggingface/README.md) | SciQ support/QA, Instruct 모델 | CPT만/QA만/CPT→QA 비교와 망각 |
| 2 | [DPO](dpo_huggingface/README.md) | SFT checkpoint, chosen/rejected | 모범 답 따라 쓰기에서 선호 비교로 이동 |
| 3 | [Reward model → PPO RLHF](rlhf_ppo_huggingface/README.md) | 같은 SFT 모델, 학습한 scalar reward model | 채점기 검증 후 rollout/policy update |
| 4 | [RLVR GRPO](rlvr_grpo_huggingface/README.md) | 산술 SFT, 정답 verifier | 검산 가능한 reward와 그룹 내 advantage |

DPO와 PPO는 같은 SFT 출발점의 대안입니다. DPO 모델을 PPO에 반드시 연결하지 않습니다. RLVR는 별도 산술 수업 출발점으로 진행합니다. 자체 작성 선호 데이터는 사람에게 수집한 평가가 아니므로 사람 피드백 RLHF의 알고리즘 경로를 축소 재현하는 예제입니다.

도구는 HF Transformers/PEFT + TRL 한 경로로 제공합니다. 선택 근거와 실제 연구 사례는 [실습 계획](../docs/11_curriculum.md)에 기록했습니다.

```bash
# Spark 호스트
bash scripts/spark_container.sh post
# 컨테이너 내부
bash scripts/install_spark.sh post
source .venv-spark-post/bin/activate
python scripts/doctor.py --require-spark
```

TRL은 0.24.0으로 고정합니다. 각 실습의 dry-run/평가/학습 명령은 해당 README를 따릅니다. 기존 validate_spark.py는 이 post-training 학습 성공을 검증하지 않습니다. CUDA 실제 수행과 품질 향상은 아직 확인하지 않았습니다.
