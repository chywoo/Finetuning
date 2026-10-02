# P4 — RLVR + GRPO: 생성한 답을 검산하며 배우기

[전체 학습 순서](../../docs/11_curriculum.md)의 SFT와 preference/RL 개념을 익힌 뒤 진행합니다. 이 수업은 **쉬운 산술 SFT → full checkpoint → strict verifier → GRPO LoRA**라는 별도 분기입니다. DPO 또는 PPO 모델을 자동으로 이어 붙이지 않습니다. 단일 BF16 지원 CUDA 환경을 요구하며 실제 GPU 학습·재로딩·성능 개선은 아직 검증하지 않았습니다.

## 1. 고등학생 수준: 네 답을 놓고 비교하기

질문 `1 + 2`에 모델이 네 번 답했다고 생각합니다.

| 생성 | 출력 | verifier reward |
|---|---|---|
| 1 | `3` | 1 |
| 2 | `4` | 0 |
| 3 | `3` | 1 |
| 4 | `The answer is 3` | 0 |

이 수업의 지시는 **정수 하나만 출력하기**입니다. 설명이 붙으면 수학 숫자가 포함되어 있어도 형식 조건을 만족하지 않아 0점입니다. 실제 적용에서는 원하는 답변 형식에 맞춰 verifier를 설계해야 합니다. 검산기가 잘못된 조건을 보상하면 모델도 그 조건을 배우게 됩니다.

RLVR(reinforcement learning with verifiable rewards)은 보상이 계산으로 검증된다는 뜻입니다. GRPO(group relative policy optimization)는 같은 질문의 여러 응답을 비교해 policy를 업데이트하는 알고리즘입니다. RLVR와 GRPO는 같은 말이 아닙니다. 여기서는 산술 정답의 정확성으로 RLVR reward를 만들고 GRPO로 최적화합니다.

모든 답이 틀려 `[0,0,0,0]`이면 어떤 답을 더 좋아해야 할지 알 수 없습니다. 모두 맞아 `[1,1,1,1]`여도 그룹 내 차이가 없습니다. 먼저 쉬운 산술 SFT로 적어도 일부 정답을 생성하게 하고, 정답/오답이 함께 나오는 그룹이 있는지 확인합니다. Step 수만 늘리는 것으로 이 문제를 해결한다고 가정하지 않습니다.

## 2. CUDA 환경과 실제 데이터

```bash
source .venv-lab/bin/activate
python -m pip install -r requirements/lab-post.txt
python scripts/doctor.py --require-cuda --profile post
```

[빠른 시작](../../docs/00_quickstart.md)의 환경 준비를 먼저 확인합니다. TRL 0.24.0 / Transformers 4.57.6 / PEFT 0.18.1과 장치에 맞는 PyTorch/CUDA를 사용합니다. 이 실습은 단일 BF16 CUDA GPU를 요구합니다. `use_vllm=False`를 명시하여 모델의 Transformers generation을 사용하므로 vLLM 서버나 별도 설치가 필요하지 않습니다.

동봉 데이터는 작성된 CC0 정수 덧셈 48문제입니다. Download 없이 사용할 수 있습니다.

```text
data/demo/post_math/
├── train.jsonl              # 32개
├── validation.jsonl         # 8개
├── test.jsonl               # 8개
├── manifest.json            # 원본/SFT split 건수와 SHA256
└── sft/{train,validation,test}.jsonl
```

```json
{"id":"math-0","prompt":"Compute 1 + 1. Reply with one integer only; no explanation.","answer":"2","completion":"2"}
```

`answer`는 GRPO verifier가 사용하는 정답이고 `completion`은 선수 SFT 정답입니다. `sft/`는 표준 `{id,prompt,completion}`로 변환한 파일입니다. Split을 먼저 정하고 변환합니다. 학습에는 오른쪽 피연산자 1–4, validation에는 5, test에는 6을 사용하므로 같은 질문을 test에 복사하지 않습니다. 작은 숫자의 새로운 조합을 평가하며 일반적인 수학 추론 benchmark가 아닙니다.

스크립트는 train/validation/test의 ID·질문 중복, 빈 필드, 정수 label, SFT completion=answer를 검사합니다. Test 정답은 학습 설정을 고르거나 rollout reward를 만들 때 쓰지 않습니다. 별도 fixture를 새 경로에 만들려면 다음 명령을 사용합니다.

```bash
python -m finetune_lab.post_data --task math --output-root data/my_post_demo
```

## 3. 선수 단계: 같은 산술 형식으로 SFT하기

완전히 random 모델로 RL을 시작하면 정답을 우연히 생성할 가능성이 너무 낮습니다. 아래 작은 instruct 모델에 산술 SFT를 먼저 합니다. GRPO는 반드시 **그 결과**에서 시작하고 baseline도 같은 결과로 측정합니다.

```bash
python 01_instruction/huggingface/train.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --method full --device cuda \
  --data-dir data/demo/post_math/sft --max-steps 120 --learning-rate 2e-5 \
  --output-dir outputs/math_sft_full
```

135M은 작은 교육용 모델이며 수학 성능을 보장하지 않습니다. 영어 특성과 Apache-2.0 라이선스는 [공식 모델 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)를 참고하십시오. 시작값 120 update를 실행했다고 GRPO에 필요한 reward 다양성이 생겼다고 가정하지 않습니다. 다음 평가와 TRAIN signal probe로 판단합니다.

Full 대신 LoRA로 선수 SFT를 했다면 먼저 merge하십시오. 아래는 대안 경로이며 위 full 명령과 하나를 선택합니다.

```bash
python 01_instruction/huggingface/train.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --method lora --device cuda \
  --data-dir data/demo/post_math/sft --max-steps 120 --learning-rate 2e-4 \
  --output-dir outputs/math_sft_lora
python -m finetune_lab.merge --adapter outputs/math_sft_lora \
  --output-dir outputs/math_sft_full
```

출발 모델과 tokenizer, raw prompt와 `### Instruction/Response` 형식이 SFT/GRPO/평가 모두 같아야 합니다. 기존 LoRA를 그대로 reference로 사용하지 않도록 GRPO train은 로컬 full checkpoint만 받습니다. 새 GRPO LoRA를 끄면 현재 math SFT가 reference가 됩니다.

## 4. Dry-run과 baseline

```bash
python 04_post_training/rlvr_grpo_huggingface/train.py \
  --model outputs/math_sft_full --dry-run
python 04_post_training/rlvr_grpo_huggingface/evaluate.py \
  --model outputs/math_sft_full --split test --num-generations 4 \
  --output outputs/grpo-baseline.json
```

Dry-run은 JSONL과 group/batch/beta 옵션만 확인하며 ML 라이브러리 import와 GPU 초기화가 없습니다. Evaluate는 greedy 답변의 정확도와 형식 오류율을 계산하고, 추가로 질문당 4개 sampling 응답의 reward를 기록합니다. Baseline의 학습 효과를 비교할 때 raw instruct 모델과 GRPO 모델을 바로 비교하면 SFT 효과까지 섞입니다. **같은 math SFT full 모델 전후**를 비교하십시오.

Verifier는 출력 전체에서 하나의 canonical integer만 허용합니다. 바깥 공백은 제거하지만 `03`, `+3`, `3.0`, `3 or 4`, 설명 문장, JSON, 코드 실행 요청은 거절합니다. 최대 19자리로 제한하고 Python `eval`을 사용하지 않습니다. 최종 문자열의 정수와 정답을 비교하여 정확히 맞으면 1, 아니면 0입니다. 숫자를 여러 개 적어 정답 하나가 우연히 포함되는 방식은 점수를 받지 못합니다.

## 5. GRPO 학습과 저장 모델 재평가

```bash
python 04_post_training/rlvr_grpo_huggingface/train.py \
  --model outputs/math_sft_full --num-generations 4 --batch-size 4 \
  --gradient-accumulation 2 --beta 0.04 --temperature 0.9 \
  --max-steps 20 --learning-rate 1e-5 --output-dir outputs/grpo_lora
python 04_post_training/rlvr_grpo_huggingface/evaluate.py \
  --model outputs/grpo_lora --split test --num-generations 4 \
  --output outputs/grpo-after.json
```

학습은 먼저 **TRAIN 질문 8개**에서 sampling group을 생성해 `training_signal_probe.json`을 저장합니다. 모든 그룹이 같은 reward만 가지면 학습을 시작하지 않고 오류로 중단합니다. 이때 파일을 열어 전부 틀리는지, 전부 맞는지, 형식 오류인지 구분합니다. 필요한 조치는 더 적절한 math SFT, 문제 난이도 변경, sampling 설정 변경입니다. Test label을 보고 training 설정을 고르지 않습니다. 실패한 output도 보호하므로 다시 시도할 때 새 경로를 지정합니다.

Default는 4 completions/group, microbatch 4, accumulation 2입니다. 이 단일 GPU 수업은 batch-size가 num-generations 이상이고 나누어 떨어지도록 제한합니다. 한 generation batch는 기본 8개 completion이 되어 완전한 4개 그룹을 유지합니다. `num-generations >= 2`가 필요합니다. OOM이면 먼저 max-completion-length를 줄이거나 두 값을 함께 2로 줄이십시오.

`beta > 0`을 요구하여 reference KL penalty를 유지합니다. 기본 0.04, `scale_rewards="group"`, `loss_type="grpo"`, `num_iterations=1`, fresh rank 8 / alpha 16 LoRA, BF16, vLLM 비활성화입니다. 답을 검증하는 함수에는 dataset의 answer가 전달되지만 정답을 generation prompt에 넣지 않습니다.

저장 결과는 adapter safetensors/config, tokenizer, metadata, metrics/log history, train signal probe, validation report입니다. Test는 별도 evaluate 프로세스에서만 점수를 냅니다. Metadata에는 math SFT full base의 경로와 revision, reward 정의, group/batch/beta/seed, 패키지 버전, 데이터 SHA256을 보존합니다. Base checkpoint를 함께 보관해야 재로딩할 수 있습니다. NaN/Inf metrics 검사는 checkpoint 저장 전에 수행합니다.

## 6. 대학 수준: advantage와 clipped policy objective

같은 질문 $x$에서 $G$개 응답 $y_i$를 생성합니다. 각 응답의 검증 reward가 $r_i$이면 그룹 평균과 표준편차로 상대 점수를 만듭니다.

$$
\bar r=\frac1G\sum_{i=1}^G r_i,\qquad
A_i=\frac{r_i-\bar r}{s_r+10^{-4}}
$$

TRL 0.24.0의 group scaling은 sample standard deviation을 사용합니다. 모든 reward가 같으면 분자는 0이므로 advantage도 0입니다. 작은 epsilon이 0으로 나누는 문제를 막지만 학습 신호를 새로 만들지는 않습니다. 위 `[1,0,1,0]` 예는 평균 0.5이고 정답은 양수, 오답은 음수 advantage를 얻습니다.

이전 rollout policy와 현재 policy의 토큰 확률 비를 $\rho_{i,t}=\pi_\theta(y_{i,t}|x,y_{i,<t})/\pi_{old}(y_{i,t}|x,y_{i,<t})$라고 쓰면, clipped objective의 핵심은 다음과 같습니다.

$$
J(\theta)=\mathbb E\left[
\min\left(\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)
-\beta D_{KL}(\pi_\theta\|\pi_{ref})
\right]
$$

실제 구현은 completion mask, 토큰 평균과 KL 추정량을 포함합니다. 위 수식은 동작을 읽기 위한 요약입니다. Clipping은 한 번의 업데이트에서 확률 비가 과하게 변하는 것을 제한하고, KL penalty는 math SFT reference에서 너무 멀어지는 것을 억제합니다. GRPO는 별도 value/critic 모델 대신 같은 prompt 그룹의 reward를 기준으로 상대 advantage를 만듭니다. 유도는 [DeepSeekMath 원 논문](https://arxiv.org/abs/2402.03300)을 참고하십시오.

이 verifier는 final-answer 정확성만 채점합니다. 추론 과정의 타당성을 채점하지 않으므로 복잡한 reasoning이 학습되었다고 해석할 수 없습니다. 보상이 올라도 잘못된 format 취약점이나 데이터 패턴을 이용했을 수 있으므로 실패 예제를 읽어야 합니다.

## 7. 지표와 실패 해석

- `exact_accuracy`: 독립 split의 greedy 생성이 형식과 정답을 모두 만족한 비율입니다.
- `invalid_format_rate`: 단일 canonical integer가 아닌 출력 비율입니다.
- `sample_groups.mean_reward`: sampling completion 중 verifier를 통과한 비율입니다. Greedy accuracy와 sampling 분포는 다릅니다.
- `frac_reward_zero_std`: 그룹 내 reward가 전부 같은 비율입니다. 1에 가까우면 상대 correctness signal이 부족합니다. 전부 맞아서 1인지 전부 틀려서 1인지 구분하십시오.
- Trainer 로그의 reward/reward_std/KL/생성 길이: update의 변화와 reward hacking을 점검하는 진단입니다. RL loss는 음수가 될 수도 있으므로 SFT loss처럼 숫자가 작아질수록 정확도가 높다고 해석하지 않습니다.

API는 [TRL v0.24.0 GRPOConfig](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py)와 [GRPOTrainer](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py)의 group divisibility, reward 함수 keyword 전달, adapter-disabled reference, advantage 계산을 기준으로 확인했습니다. Source 검토는 실제 GPU 실행 성공을 의미하지 않습니다.

실습 질문: `[0,0,0,0]`과 `[1,1,1,1]`의 accuracy는 다르지만 상대 advantage는 왜 같을까요? 정답을 포함한 긴 문장을 0점으로 만드는 것은 어떤 목표를 반영할까요? Sampling temperature를 높이면 정답률과 그룹 다양성이 어떻게 함께 변할까요?

마지막으로 baseline/after의 동일 test JSON을 비교하고 [전체 학습 순서](../../docs/11_curriculum.md)의 다음 단계인 VLM·확장 실험으로 이동하십시오. 숫자 몇십 문제의 성공을 광범위한 수학·추론 능력 향상으로 보고하지 않습니다.
