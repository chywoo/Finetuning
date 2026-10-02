# RLHF의 구조 실습: 선호 데이터 → reward model → PPO

이 실습은 **좋은 답변을 채점하는 모델을 먼저 학습하고, 그 점수를 이용해 답변 모델을 개선**합니다. BF16 지원 Linux CUDA 환경에서 Hugging Face Transformers와 **TRL 0.24.0**을 사용합니다. 주 학습 경로의 출발점은 아래 선수 과정에서 만든 **`outputs/preference_sft_full` SFT checkpoint**입니다. 기존 `HuggingFaceTB/SmolLM2-135M-Instruct`는 reward/PPO 단계만 분리하여 확인하는 선택지입니다.

프로젝트의 기본 선호 데이터는 사람이 실제로 채점한 기록이 아니라 **작성한 synthetic 예제**입니다. 따라서 이것은 RLHF의 reward-model/PPO 계산 과정을 학습하는 실습이며, 실제 human-feedback RLHF 실험을 재현했다는 결과로 해석하면 안 됩니다. 실제 RLHF를 하려면 평가 기준을 정하고 사람에게 동일 prompt의 여러 응답을 비교하게 하여 선호 데이터를 수집해야 합니다.

## 1. 고등학생에게 설명하기: 학생, 채점 선생님, 과격한 변화 방지

SFT는 학생에게 문제와 모범 답안을 보여 주는 수업입니다. 학생이 이미 어느 정도 답할 수 있게 되면 같은 문제의 두 답 중 더 나은 답을 골라 보여 줍니다. **reward model(RM)**은 이런 비교 기록을 보고 답안을 채점하는 선생님 역할을 배웁니다.

**policy**는 답을 작성하는 학생입니다. 학생이 직접 새 답을 만들고, 채점 모델이 점수를 줍니다. PPO는 높은 점수를 받는 답을 더 자주 만들도록 학생을 업데이트합니다. 채점 모델이 불완전하므로 학생이 점수만 높이는 이상한 문장을 배울 수도 있습니다. **reference policy**는 개선을 시작하기 전 학생의 답변 방식이며, KL penalty는 갑자기 너무 멀리 변하지 않도록 합니다.

**value model**은 답을 작성하는 도중 “이 상태에서 어느 정도 점수를 기대할까?”를 예측합니다. PPO는 실제 보상이 이 기대보다 얼마나 좋은지, 즉 advantage를 사용해 업데이트합니다. 채점 모델과 value model은 같은 점수 출력 형태를 가지지만 역할과 학습 대상이 다릅니다.

## 2. 대학생에게 설명하기: 네 모델과 목적함수

| 모델 | 역할 | 이 구현에서 업데이트 |
|---|---|---|
| policy `πθ` | prompt에서 응답 생성 | PPO에서 전체 가중치 학습 |
| reference `πref` | 초기 SFT policy, KL 기준 | 고정 |
| reward `rφ(x,y)` | 완성 응답의 선호 점수 | 선호 학습 단계에서 학습, PPO에서는 고정 |
| value `Vψ(s)` | 생성 중 기대 return | PPO에서 전체 가중치 학습 |

### Reward learning: Bradley–Terry 모델

같은 prompt `x`에서 chosen 응답 `y+`를 rejected `y-`보다 선호한다고 표시한 데이터로 학습합니다.

\[
P(y^+ \succ y^-\mid x)=\sigma(r_\phi(x,y^+)-r_\phi(x,y^-))
\]

\[
L_{RM}=-\mathbb{E}\log\sigma(r_\phi(x,y^+)-r_\phi(x,y^-))
\]

정답은 0~100점 같은 절대 점수가 아닙니다. 두 점수의 차이가 중요합니다. 이 구현은 평균 점수의 불필요한 이동을 줄이는 `center_rewards_coefficient=0.01`을 추가하고, **실제 pretrained backbone + 새 scalar `score` head**를 `RewardTrainer`로 학습합니다. 학습 완료 metadata와 가중치가 있는 RM checkpoint만 PPO에 사용할 수 있습니다. [TRL 0.24 RewardTrainer 문서](https://huggingface.co/docs/trl/v0.24.0/reward_trainer).

### PPO의 보상과 KL

학습 목표를 간단히 쓰면 다음과 같습니다.

\[
J(\theta)=\mathbb{E}_{y\sim\pi_\theta}[r_\phi(x,y)]-\beta D_{KL}(\pi_\theta(\cdot\mid x)\Vert\pi_{ref}(\cdot\mid x))
\]

코드의 `--kl-coef`가 `β`입니다. 구현에서는 생성 토큰별 log probability 차이를 이용해 KL penalty를 계산하고, 완성된 응답에 reward model 점수를 부여합니다. 큰 reward score가 곧 실제 품질이라는 의미는 아닙니다. reward model이 놓친 편법을 policy가 찾아내는 **reward hacking**도 관찰해야 합니다.

### Clipped PPO와 value loss

\[
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
\]

\[
L_{clip}=\mathbb{E}\left[\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right]
\]

여기서 `πold`는 해당 rollout을 생성한 policy이며, **KL 기준인 `πref`와 다른 개념**입니다. PPO는 같은 rollout에 몇 번의 update를 적용하므로 `--ppo-epochs`를 늘리면 기존 샘플을 더 많이 재사용합니다. value loss는 대략 `E[(Vψ(s_t)-R̂_t)²]` 형태이고, TRL은 value clipping과 GAE를 포함합니다. GAE는 `δt = rt + γ V(s_{t+1}) − V(s_t)`를 이용해 advantage를 계산합니다. [PPO 논문](https://arxiv.org/abs/1707.06347), [RLHF PPO 구현 설명](https://huggingface.co/docs/trl/v0.24.0/ppo_trainer).

## 3. 선수 과정: SFT 이후 두 갈래

```text
Base LLM → instruction SFT
                      ├→ preference pairs → DPO
                      └→ preference pairs → reward learning → PPO
```

DPO는 chosen/rejected 쌍과 reference를 사용하여 policy를 직접 업데이트합니다. 이 폴더의 reward→PPO 경로는 별도의 scalar reward model을 학습하고, policy가 학습 중 직접 생성한 응답으로 강화학습을 수행합니다. 서로 다른 과정이므로 데이터·출발 SFT·평가 질문을 맞추어 비교하세요. 학습 순서 전체는 [curriculum 문서](../../docs/11_curriculum.md)를 참고합니다.

처음에는 기존 Instruct 모델로 RLHF 단계만 분리하여 실습할 수 있습니다. 자신의 SFT checkpoint를 쓰려면 instruction 실습에서 `--method full`로 저장한 결과를 지정하세요. adapter 폴더는 이 스크립트가 거부합니다. LoRA SFT를 사용했다면 먼저 base와 merge한 전체 checkpoint가 필요합니다. PPO의 주 실습은 full fine-tuning이며 LoRA/QLoRA 옵션을 제공하지 않습니다.

환경을 준비한 뒤 아래의 full SFT 선수 실습을 진행합니다. [DPO 실습](../dpo_huggingface/README.md)에서 이미 같은 `outputs/preference_sft_full`을 만들었다면 재학습하지 않고 그대로 사용하세요. DPO 문서의 LoRA SFT → merge 경로도 같은 이름의 full checkpoint를 제공합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_instruction/huggingface/train.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --method full --device cuda \
  --data-dir data/demo/post_preferences/sft --max-steps 40 --learning-rate 2e-5 \
  --output-dir outputs/preference_sft_full
python 04_post_training/rlhf_ppo_huggingface/train.py --stage reward --dry-run
```

**동일한 SFT checkpoint에서 DPO 또는 reward→PPO로 갈라집니다.** PPO 전에 DPO를 순서대로 수행해야 하는 경로는 아닙니다. 비교 실험에서는 SFT 출발점·선호 데이터·test 질문을 동일하게 맞추세요.

## 4. 데이터와 입력 형식

공유 데이터 준비 스크립트는 다음 표준 형식을 만듭니다.

```json
{"id":"example-1","prompt":"Return only the sum of 2 and 3.","chosen":"5","rejected":"6"}
```

이 예시는 형식 설명을 위해 작성했습니다. 기본 authored fixture는 산술 질문의 정확한 답과 틀린 답을 비교하는 작은 데이터이며, **실제 사람의 선호 기록이 아닙니다**. 경로는 `data/demo/post_preferences/{train,validation,test}.jsonl`입니다. 기본 데이터만으로 광범위한 유용성이나 안전성을 정렬했다고 주장할 수 없습니다.

기본 split은 train 16 / validation 4 / test 4이며 CC0-1.0으로 작성한 fixture입니다. `sft/`에는 동일 split의 chosen을 모범 응답으로 변환한 데이터도 있습니다. 준비 스크립트는 기존 데이터를 보호하므로 이미 제공된 fixture를 그대로 사용하세요. 재생성 실습은 `python -m finetune_lab.post_data --task preferences --output-root data/reproduced_post`처럼 새 root에 만들고, 이후 모든 명령에 `--data-dir data/reproduced_post/post_preferences`를 지정합니다.

`read_preferences()`는 필수 문자열, chosen/rejected 차이, ID와 정규화한 prompt의 split 중복을 검사합니다. RewardTrainer에는 아래 prefix와 각각의 응답+EOS를 토큰화하여 전달합니다.

```text
### Instruction:
<prompt>

### Response:
<chosen 또는 rejected><EOS>
```

PPO에는 **prompt prefix만** 전달하고 응답은 policy가 새로 생성합니다. chosen/rejected 답은 PPO rollout prompt에 넣지 않습니다. reward model, policy, reference, value는 같은 원래 token ID를 사용해야 합니다. reward 모델의 입력에는 실제 prompt와 생성 응답이 함께 들어갑니다.

PPO의 reward pooling은 padding 위치를 읽기 때문에 **PAD와 EOS가 같으면 안 됩니다**. 코드에서 필요한 경우 별도 PAD를 추가하고 네 모델 모두의 embedding 크기를 맞춥니다. RM tokenizer를 함께 저장하고, PPO와 평가에서 이 tokenizer를 재사용합니다. 너무 긴 pair와 prompt는 조용히 제거하거나 자르지 않고 오류로 안내합니다.

## 5. 실행 환경

먼저 [빠른 시작](../../docs/00_quickstart.md)의 일반 환경과 단일 BF16 CUDA 장치를 준비합니다. Spark를 선택한 경우에만 [전용 안내](../../docs/04_dgx_spark.md)의 `requirements/spark-post.txt` NGC overlay를 사용합니다.

```bash
source .venv-lab/bin/activate
python -m pip install -r requirements/lab-post.txt
python scripts/doctor.py --require-cuda --profile post
# 이미 제공된 data/demo/post_preferences fixture를 사용합니다.
# 다음으로 3절의 full SFT 선수 명령을 실행합니다.
```

실제 학습·평가는 Linux, CUDA 한 장, BF16 지원, `trl==0.24.0`을 확인한 다음 수행합니다. CPU로 자동 전환하지 않습니다. `--dry-run`은 ML 라이브러리를 import하지 않고 schema와 checkpoint 조건만 검사합니다.

**버전의 차이:** TRL 0.24.0의 public API는 `from trl import PPOConfig, PPOTrainer`입니다. 해당 tagged source에는 향후 `trl.experimental`로 이동할 수 있다는 warning이 있지만 0.24.0의 이 코드는 experimental namespace를 import하지 않습니다. 최신 TRL 예제를 섞지 마세요. [고정 버전 PPO source](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/ppo_trainer.py).

## 6. 단계 A: 실제 reward model 학습

```bash
CUDA_VISIBLE_DEVICES=0 python 04_post_training/rlhf_ppo_huggingface/train.py \
  --stage reward --model outputs/preference_sft_full \
  --max-steps 20 --batch-size 2 --gradient-accumulation 4 \
  --learning-rate 1e-5 --output-dir outputs/rlhf_reward_run1
```

다른 SFT 결과를 사용하려면 full checkpoint 경로를 지정합니다. 여기서 classifier의 scalar head가 처음에는 임의 초기화되는 것은 정상입니다. 선호 학습을 완료하고 저장한 이후에만 PPO reward로 사용합니다. 단계를 분리해 보는 선택 실험은 `--model HuggingFaceTB/SmolLM2-135M-Instruct`로 바꾸되, PPO와 baseline 평가에도 동일한 모델을 지정합니다.

학습 전후 validation metric을 기록하고, loss와 로그가 NaN/Inf이면 종료합니다. 출력은 scalar 모델 `config.json`/safetensors, distinct-PAD tokenizer, `training_metadata.json`입니다. metadata에는 실제 완료 step, 재현에 필요한 library/CUDA 버전과 출발 모델 revision을 남기며 내부 장비 식별 정보·비밀 값은 제외합니다.

## 7. 단계 B: held-out reward 평가

```bash
CUDA_VISIBLE_DEVICES=0 python 04_post_training/rlhf_ppo_huggingface/train.py \
  --stage evaluate-reward --reward-model outputs/rlhf_reward_run1 \
  --split test --output-dir outputs/rlhf_reward_test_run1
```

새 프로세스에서 저장한 reward model을 재로드합니다. test에서 `r(chosen)>r(rejected)` 비율, 평균 gap, Bradley–Terry loss와 각 pair의 점수를 저장합니다. 동일 점수는 맞춘 것으로 계산하지 않습니다. train pair 정확도가 높고 test가 낮으면 데이터 암기나 편향을 의심하세요.

## 8. 단계 C: PPO policy 학습

```bash
CUDA_VISIBLE_DEVICES=0 python 04_post_training/rlhf_ppo_huggingface/train.py \
  --stage ppo --model outputs/preference_sft_full \
  --reward-model outputs/rlhf_reward_run1 --total-episodes 32 \
  --batch-size 2 --gradient-accumulation 4 --ppo-epochs 2 \
  --response-length 32 --kl-coef 0.05 --learning-rate 1e-6 \
  --output-dir outputs/rlhf_ppo_run1
```

reward 단계와 동일한 출발 policy/tokenizer를 사용하세요. `--model`이 SFT checkpoint이면 두 단계 모두 같은 경로를 지정합니다. 코드가 원래 vocabulary ID와 backbone 구조를 확인합니다. reference는 PPO 시작 시 policy와 동일한 checkpoint를 별도 모델로 읽어 고정합니다. value는 학습한 reward checkpoint로 초기화하되, PPO에서 따로 업데이트합니다.

이 설정에서 한 rollout batch는 `batch-size × gradient-accumulation = 8`개의 응답입니다. 32 episodes는 4개의 rollout batch이며 각 batch에 PPO epoch 2회를 적용합니다. RewardTrainer의 `--max-steps`와 PPO의 `--total-episodes`는 다른 단위입니다. train prompt가 rollout batch보다 적으면 TRL의 `drop_last` 때문에 학습이 진행되지 않을 수 있어 사전 검사합니다. episode 수는 rollout batch의 배수로 지정합니다.

policy/value는 학습하고 reward/reference는 고정합니다. 길이를 점수로 쓰는 가짜 reward를 사용하지 않습니다. NaN/Inf 로그나 마지막 policy/value 가중치가 발견되면 최종 checkpoint를 저장하지 않습니다. 결과 폴더에는 **policy 전체 모델**, tokenizer, metadata, 별도 `value_model/`이 저장됩니다. optimizer 상태까지 완전히 재개하는 기능은 이 실습에 포함하지 않습니다.

## 9. 단계 D: policy를 다시 읽어 baseline과 비교

```bash
# 학습 전 출발 policy를 같은 test에서 평가
CUDA_VISIBLE_DEVICES=0 python 04_post_training/rlhf_ppo_huggingface/train.py \
  --stage evaluate-policy --model outputs/preference_sft_full \
  --reward-model outputs/rlhf_reward_run1 --split test \
  --output-dir outputs/rlhf_policy_baseline_test
# 저장한 PPO policy를 새 프로세스에서 재로드하여 평가
CUDA_VISIBLE_DEVICES=0 python 04_post_training/rlhf_ppo_huggingface/train.py \
  --stage evaluate-policy --model outputs/rlhf_ppo_run1 \
  --reward-model outputs/rlhf_reward_run1 --split test \
  --output-dir outputs/rlhf_policy_ppo_test
```

각 결과의 `evaluation.json`에는 held-out chosen/rejected의 **응답 전체 log probability 차이**, preference accuracy, 실제 greedy 생성 문장이 저장됩니다. log probability는 길이 정규화를 하지 않으며 응답 길이에 영향을 받습니다. 이 진단 지표가 사람의 평가를 대신하지 않습니다.

정책 평가에서도 전체 preference pair가 길이 제한에 들어오는지 먼저 검사합니다. chosen/rejected를 잘라 버려 다른 응답을 비교하는 일을 막습니다. prompt와 새 응답의 길이 예산이 저장된 모델의 context 길이를 넘으면 오류로 종료합니다.

생성 문장에도 RM score를 기록하지만 **같은 RM은 PPO의 학습 목표이므로 독립적인 품질 척도가 아닙니다**. `training_reward_proxy`라는 이름으로 구분합니다. 생성 내용을 직접 채점하고, 새로운 사람의 비교 기록이나 추가 평가 데이터로 결과를 확인하세요. test chosen/rejected pair를 gradient update에 사용하지 않습니다.

## 10. 메모리와 실험 설계

PPO는 policy·reference·reward·value 네 backbone을 함께 사용합니다. 기본 구현은 가중치를 FP32로 읽고 CUDA BF16 mixed precision으로 학습합니다. 135M 모델의 backbone 네 개는 단순 계산으로 `4 × 135M × 4byte ≈ 2.16GB`입니다. 학습하는 policy/value의 FP32 gradient는 약 1.08GB, Adam의 두 FP32 moment는 약 2.16GB이므로 주요 tensor만 합쳐 약 5.4GB입니다. activation, logits, CUDA allocator와 일시적인 복사본을 더하면 실제 peak는 더 커집니다. 이 숫자는 총 메모리나 실행 측정치가 아닙니다. 1.7B 모델로 바꾸면 backbone 가중치만 약 27.2GB이므로 먼저 135M으로 시작하세요.

Spark 통합 메모리를 모두 모델에 쓸 수 있는 것은 아니며 호스트 프로그램도 공유합니다. 부족하면 response length, prompt length, batch size를 줄이세요. accumulation을 늘리면 PPO의 rollout batch도 커지므로 동시에 늘리지 말고 위 계산식을 확인합니다. QLoRA나 여러 GPU 전략은 이 실습의 PPO 구현 범위에 포함하지 않습니다.

한 번에 한 조건만 바꾸어 비교합니다.

1. reward 학습 step 수를 늘려 train/validation gap 변화 확인. test는 설정 선택 후 최종 비교에 사용.
2. `--kl-coef` 0.02 / 0.05 / 0.1 비교: reward와 KL, 답변 변화.
3. `--ppo-epochs` 1 / 2 비교: 같은 rollout 재사용의 효과.
4. 산술 이외의 설명·정확성·형식 선호 쌍을 직접 작성하거나 사람에게 수집.
5. 동일 SFT와 선호 데이터를 사용한 DPO 결과와 test 생성 비교.

모든 stage는 기존 비어 있지 않은 출력 폴더를 보호합니다. 재실험할 때 새 `--output-dir`을 지정하세요. **이 폴더의 CUDA RewardTrainer/PPO 학습·저장·재로딩은 아직 검증되지 않았습니다.** 정적 검토는 TRL 0.24.0 tagged API를 기준으로 하며 실제 통합 검증은 지원 CUDA 환경에서 별도로 수행합니다.

지원 CUDA 환경에서 schema·출력 보호 테스트와 실제 RM→PPO→재로드 통합 테스트를 실행하려면 다음 명령을 사용합니다. 통합 테스트는 작은 step 수로 파이프라인을 검사하며 응답 품질을 증명하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0 RUN_SPARK_POST_TESTS=1 python -m unittest discover -s tests -p test_post_rlhf.py
# 자신의 full SFT checkpoint로 테스트하려면 위 명령에 환경변수를 함께 지정합니다.
# SPARK_POST_SFT_MODEL=outputs/instruction_hf_full
```
