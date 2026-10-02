# P1 — DPO: 두 답을 비교해서 선호 순서 배우기

먼저 [학습 순서](../../docs/11_curriculum.md)의 instruction SFT와 모델 저장/재로딩을 완료합니다. 이 실습은 **SFT checkpoint → 전체 모델로 merge → 새로운 DPO LoRA** 순서입니다. 단일 BF16 지원 CUDA 환경을 요구하며 GPU 학습·재로딩·품질 개선은 아직 검증하지 않았습니다.

## 1. 고등학생 수준의 직관

SFT는 한 모범 답을 보여 줍니다. DPO는 같은 질문에 두 답을 놓고 “이 답을 더 좋아한다”는 비교를 보여 줍니다. 질문이 `1 + 2를 정수 하나로 답하라`이면 `3`을 chosen, `4`를 rejected로 지정할 수 있습니다. 모델이 정답을 이미 알고 있어도, 비교 학습은 두 답의 상대 확률을 바꿀 수 있습니다.

이 저장소의 선호는 **사람에게 조사한 평가가 아닌, 산술 정답을 기준으로 만든 합성 선호**입니다. 사람의 취향·안전성·문화적 가치가 학습되었다고 판단하는 자료가 아닙니다. 선택 기준이 산술 정확성이라는 작은 예로 DPO의 구조를 배웁니다. 실제 인간 선호 데이터로 확장할 때는 평가 기준·평가자 일치·정답 충돌·개인정보를 별도로 검토해야 합니다.

정책(policy)은 업데이트할 답변 모델입니다. Reference는 비교 기준으로 고정한 **현재 SFT 모델**입니다. SFT LoRA를 먼저 merge한 전체 모델에서 새 LoRA를 만들면, 새 adapter를 껐을 때 SFT 출발점으로 정확히 돌아갑니다. SFT adapter를 그대로 DPO adapter처럼 취급하면 reference의 의미가 달라질 수 있어 이 실습은 unmerged adapter를 거절합니다.

## 2. CUDA 환경과 데이터

```bash
source .venv-lab/bin/activate
python -m pip install -r requirements/lab-post.txt
python scripts/doctor.py --require-cuda --profile post
```

환경은 TRL 0.24.0, Transformers 4.57.6, PEFT 0.18.1을 대상으로 합니다. 장치에 맞는 PyTorch/CUDA와 [일반 환경 준비](../../docs/00_quickstart.md)를 사용합니다. Spark 선택 환경은 [전용 안내](../../docs/04_dgx_spark.md)를 따릅니다. 이 실습은 한 개의 BF16 지원 CUDA GPU를 요구하며 다른 장치로 자동 전환하지 않습니다.

작은 CC0 fixture를 실제 파일로 동봉했습니다. 다운로드할 필요가 없습니다.

```text
data/demo/post_preferences/
├── train.jsonl              # 16개
├── validation.jsonl         # 4개
├── test.jsonl               # 4개
├── manifest.json            # 원본 및 SFT split 건수·SHA256
└── sft/{train,validation,test}.jsonl
```

```json
{"id":"preference-0","prompt":"Compute 1 + 1. Reply with one integer only; no explanation.","chosen":"2","rejected":"3"}
```

`prompt`에는 원래 질문, `chosen/rejected`에는 답만 넣습니다. 스크립트가 공통 `### Instruction:` / `### Response:` 형식을 적용하고 TRL이 두 응답에 EOS를 붙입니다. 빈 필드·같은 chosen/rejected·중복 ID·split을 넘는 정규화된 질문 중복을 검사합니다. Test는 학습 dataset에 넣지 않습니다. 현재 fixture는 덧셈만 포함하며 광범위한 preference benchmark가 아닙니다.

`sft/`는 chosen을 completion으로 변환한 같은 split입니다. 원본부터 split한 뒤 변환하므로 질문이 train에서 test로 옮겨 들어가지 않습니다. 새 위치에 동일 fixture를 작성하려면 아래 명령을 사용합니다. 이미 동봉된 기본 디렉토리에는 다시 쓰지 않습니다.

```bash
python -m finetune_lab.post_data --task preferences --output-root data/my_post_demo
```

## 3. 선수 실습: 현재 SFT checkpoint 만들기

이미 같은 prompt 형식의 SFT full checkpoint가 있으면 그 경로를 `--model`로 지정해도 됩니다. 아래는 fixture에 맞춘 작은 LoRA SFT를 만든 뒤 전체 모델로 merge하는 완전한 경로입니다.

```bash
python 01_instruction/huggingface/train.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --method lora --device cuda \
  --data-dir data/demo/post_preferences/sft --max-steps 40 --learning-rate 2e-4 \
  --output-dir outputs/preference_sft_lora
python -m finetune_lab.merge --adapter outputs/preference_sft_lora \
  --output-dir outputs/preference_sft_full
```

이 단계는 base→instruction SFT를 다시 재현하려는 것이 아니라, post-training의 출발 policy를 데이터 형식에 맞추는 SFT입니다. 출발 모델의 영어 중심 특성과 Apache-2.0 라이선스는 [공식 SmolLM2 카드](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)를 참고하십시오. SFT가 만든 결과를 보존하고 DPO는 새로운 output에 저장합니다.

## 4. Dry-run과 학습 전 평가

```bash
python 04_post_training/dpo_huggingface/train.py --model outputs/preference_sft_full --dry-run
python 04_post_training/dpo_huggingface/evaluate.py \
  --model outputs/preference_sft_full --split test \
  --output outputs/dpo-baseline.json
```

Dry-run은 세 split의 JSONL만 검사하고 ML 라이브러리를 import하지 않습니다. 실제 train은 `config.json`과 safetensors가 있는 로컬 full checkpoint를 요구합니다. Evaluate는 현재 정책이 chosen을 rejected보다 더 높은 **전체 응답 log probability**로 평가한 쌍의 비율과 실제 생성 답변을 저장합니다. 정책이 비교 문제를 잘 풀어도 greedy 생성은 다른 답을 낼 수 있으므로 두 결과를 함께 봅니다.

## 5. DPO 학습과 저장 모델 재로딩

```bash
python 04_post_training/dpo_huggingface/train.py \
  --model outputs/preference_sft_full --max-steps 20 --beta 0.1 \
  --batch-size 1 --gradient-accumulation 4 --learning-rate 5e-5 \
  --output-dir outputs/dpo_lora
python 04_post_training/dpo_huggingface/evaluate.py \
  --model outputs/dpo_lora --split test --output outputs/dpo-after.json
```

두 번째 명령은 새 프로세스에서 저장 adapter와 원래 SFT full 모델을 다시 로드합니다. 마지막 evaluate까지 성공해야 train→save→reload→evaluate 경로를 확인한 것입니다. 같은 report/output 파일은 덮어쓰지 않으므로 재실험에는 새 이름을 지정합니다.

`DPOTrainer`는 새로운 rank 8 / alpha 16 LoRA를 만들고 base 가중치를 고정합니다. Reference model을 따로 복사하지 않고 adapter를 끈 SFT 가중치를 사용합니다. `reference_free=False`, sigmoid DPO loss, `beta=0.1`, BF16, dropout 비활성화, gradient checkpointing을 명시합니다. Training은 train split, 학습 전후 Trainer 평가에는 validation split만 사용합니다.

길이가 긴 preference를 임의로 잘라 label 의미를 바꾸지 않습니다. Prompt/완전한 응답+EOS가 설정된 길이를 넘으면 학습과 CLI 평가가 모두 실패합니다. `max-prompt-length=128`, `max-completion-length=32`, `max-length=256`이 기본입니다. 형식을 바꾼 데이터라면 실제 tokenizer 길이를 보고 명시적으로 한도를 늘리십시오.

저장되는 파일은 adapter safetensors/config, tokenizer, `training_metadata.json`, `metrics.json`입니다. Metadata에는 full SFT base의 절대 경로, revision, beta, seed, 실제 버전, prompt 형식, 데이터 SHA256과 split 건수를 기록합니다. Base full 디렉토리를 삭제하면 adapter를 재로딩할 수 없습니다. 성공 checkpoint를 저장하기 전에 loss와 로그의 NaN/Inf를 검사합니다.

## 6. 대학 수준: DPO 목적함수

입력 $x$, chosen $y^+$, rejected $y^-$, 학습 policy $\pi_\theta$, 고정 reference $\pi_{ref}$라고 씁니다. 응답 log probability는 응답 각 토큰의 조건부 log probability를 더한 값입니다. Prompt token은 이 합의 정답 구간에 넣지 않습니다.

$$
\Delta_\theta=
\left[\log\pi_\theta(y^+|x)-\log\pi_{ref}(y^+|x)\right]
-\left[\log\pi_\theta(y^-|x)-\log\pi_{ref}(y^-|x)\right]
$$

$$
\mathcal L_{DPO}=-\log\sigma(\beta\Delta_\theta)
$$

$\sigma(z)=1/(1+e^{-z})$입니다. $\Delta$가 커지면 chosen이 reference 대비 더 우세해져 loss가 줄어듭니다. 시작 직후 새 LoRA의 변화가 없다면 policy=reference라서 $\Delta\approx0$, 쌍당 loss는 $\log2\approx0.693$입니다. 모델이 chosen을 원래 더 좋아했어도 reference로부터 **얼마나 바뀌었는지** 계산하면 시작 margin은 0일 수 있습니다.

DPO의 유도와 RLHF 목적 연결은 [원 논문](https://arxiv.org/abs/2305.18290)을 참고하십시오. 별도 학습 reward model을 먼저 만드는 과정 없이 고정 preference 쌍으로 학습합니다. 이 코드에는 online rollout PPO가 없습니다. 알고리즘을 사용한다는 사실과 preference label의 품질은 별개입니다.

`beta`는 reference에서 벗어나는 정도를 제어하는 핵심 값입니다. Learning rate와 데이터를 고정해 beta 0.05/0.1/0.2를 비교하되 한 숫자만 보고 선택하지 않습니다. 작은 합성 데이터의 선호 확률이 높아져도 다른 instruction 능력이 약해질 수 있습니다.

## 7. 지표 읽기와 다음 단계

- `preference_accuracy`: 현재 정책의 raw chosen log probability가 rejected보다 큰 비율입니다.
- `mean_policy_gap`: chosen minus rejected의 평균입니다. 응답 길이가 다른 일반 데이터에서는 길이 편향을 함께 점검합니다.
- `reference_adjusted_accuracy`: 저장 adapter 평가에서 reference 대비 gap 개선이 양수인 비율입니다. Raw accuracy와 의미가 다릅니다. 동률은 성공으로 세지 않습니다.
- Trainer의 `rewards/accuracies`, `rewards/margins`, loss: implicit reward 기준의 학습 진단입니다. 실제 사람이 매긴 보상 점수는 아닙니다.
- JSON `generated`: 질문당 greedy 생성입니다. Pair metric이 좋아져도 숫자 형식과 정확성이 실제 생성에서 좋아졌는지 확인합니다.

실습 질문: chosen/rejected를 의도적으로 뒤집으면 모델이 어떤 행동을 배울까요? 길이만 다른 두 정답을 비교하면 정답성 대신 길이를 학습할 수 있을까요? 현재 SFT full을 reference로 두는 이유를 adapter disable과 연결해 설명하십시오.

API는 최신 문서 대신 이 환경의 [TRL v0.24.0 DPOConfig](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/dpo_config.py)와 [DPOTrainer](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/dpo_trainer.py)의 constructor, tokenizer 처리, reference adapter context를 확인했습니다. 실제 CUDA 성공은 별도 검증 대상입니다.

이후 [P2 reward model → P3 PPO](../rlhf_ppo_huggingface/README.md)와 비교하고, [전체 학습 순서](../../docs/11_curriculum.md)에서 다음 단계의 통과 조건을 확인하십시오. PPO를 DPO 결과에 자동 연결하지 않고 같은 SFT 출발점에서 비교합니다.
