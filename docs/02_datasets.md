# 데이터셋 정보와 전처리

모든 실제 실습 데이터는 Hugging Face 공개 저장소에서 준비합니다. `prepare_data.py`는 dataset ID의 요청 revision을 실제 SHA로 해석하고, 같은 snapshot으로 데이터를 로드합니다. 학습 코드는 다운로드와 전처리를 자동으로 다시 수행하지 않습니다.

## 원본 데이터

| 데이터 | 작성자·출처 | 원본 크기 | 원본 컬럼 | 실습 용도 | 라이선스 |
|---|---|---|---|---|---|
| [Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Databricks 직원 작성 | train 15,011 | instruction, context, response, category | base LLM instruction SFT | CC BY-SA 3.0 |
| [SciQ](https://huggingface.co/datasets/allenai/sciq) | Welbl·Liu·Gardner, AI2 | train 11,679 / validation 1,000 / test 1,000 | question, correct_answer, distractor1/2/3, support | support CPT + closed-book QA SFT | CC BY-NC 3.0 |
| [Beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | Makerere AI Lab | train 1,034 / validation 133 / test 128 | image, labels, image_file_path | 콩잎 3-class VQA | HF card licensing 정보 미기재 |

Dolly 데이터의 출처 표시와 ShareAlike 조건을 확인합니다. SciQ는 비상업 조건이 있어 개인 학습·연구 실습에 맞게 사용하며 상업 데이터로 자동 전용하지 않습니다. Beans의 원출처는 [Makerere ibean 저장소](https://github.com/AI-Lab-Makerere/ibean/)입니다. HF card의 정보가 불완전하므로 명시되지 않은 라이선스를 추측하지 않습니다. 출판·재배포 전에 원작성자의 조건을 확인합니다.

## 준비 명령

Spark 컨테이너의 가상환경을 활성화한 다음 저장소 루트에서 실행합니다.

```bash
python -m finetune_lab.prepare_data --task instruction --source hf --train-samples 256 --eval-samples 32
python -m finetune_lab.prepare_data --task knowledge --source hf --train-samples 256 --eval-samples 32
python -m finetune_lab.prepare_data --task vision --source hf --train-samples 96 --eval-samples 24
```

`--train-samples`는 학습 subset, `--eval-samples`는 validation/test **각각** 요청 수입니다. `--seed 42`, `--revision main`, `--output-root data/processed`를 지정할 수 있습니다. 이미 저장된 데이터는 `--overwrite` 없이 덮어쓰지 않습니다. 큰 실험은 별도 output root를 사용해 작은 분할도 보존합니다.

```bash
python -m finetune_lab.prepare_data --task instruction --source hf --train-samples 8000 --eval-samples 500 --output-root data/experiment_large
```

이 구현은 `load_dataset`으로 원본 snapshot을 다운로드한 뒤 subset을 만듭니다. **subset을 96장으로 지정해도 원본 Beans 약 360 MB 다운로드가 발생할 수 있습니다.** 모델 weights와 HF cache까지 포함한 저장공간을 고려합니다. streaming으로 대규모 원본을 다루는 방법은 확장 과제로 남겼습니다.

## Instruction 변환

```json
{"id":"dolly-예제번호","prompt":"instruction\n\nContext:\ncontext","completion":"response","category":"summarization"}
```

context가 비어 있으면 Context 부분을 넣지 않습니다. 빈 instruction/response를 제외하고, whitespace·대소문자를 정규화한 exact prompt 중복을 제거합니다. seed로 전체 shuffle한 뒤 요청 수에 맞춰 train/validation/test를 나눕니다. Dolly에는 원래 validation/test가 없으므로 이 저장소에서 만든 소규모 holdout이며 공개 표준 benchmark 점수가 아닙니다.

같은 문서의 다른 질문, 표현만 바뀐 지시문은 exact dedup로 모두 발견되지 않습니다. 실제 도메인 데이터에서는 문서·대화·사용자 같은 그룹 단위 분할, 근접 중복 탐지, 수동 검토도 필요합니다.

## Knowledge 변환

SciQ의 공식 split을 유지한 상태에서 shuffle/subset을 선택하고 CPT/QA 형식을 만듭니다.

```json
{"id":"sciq-train-번호","text":"support 문단"}
```

```json
{"id":"sciq-train-번호","prompt":"question","completion":"correct_answer","support":"평가 입력에는 넣지 않는 참고 문단"}
```

SFT 입력은 질문만 사용합니다. distractor와 support를 프롬프트에 넣지 않아 정답 문맥 읽기와 closed-book 지식을 혼동하지 않습니다. CPT는 빈 support를 제외합니다. test → validation → train 순으로 exact content 중복을 제거해 평가 행을 우선 보존합니다. 따라서 최종 요청 수보다 작아질 수 있습니다.

작성 시 snapshot `2c94ad3e1aafab77146f384e23536f97a4849815`에서는 SFT 255/32/32, CPT 232/29/31이 저장되었습니다. split별 CPT text와 QA prompt 중복은 제거하지만 **문서의 의미상 중복·근접 중복까지 보장하는 도구는 아닙니다.** 같은 과학 지식이 여러 문단에 등장할 수 있습니다.

## Vision 변환

```json
{"id":"beans-train-번호","image":"images/train-00001.png","label":"healthy","image_sha256":"..."}
```

label mapping은 `0=angular_leaf_spot`, `1=bean_rust`, `2=healthy`입니다. 원본 split을 유지하고 seed shuffle로 subset을 선택합니다. 이미지를 RGB PNG로 저장하고 디코딩된 pixel bytes의 SHA256으로 split 간 정확한 이미지 복제본을 제외합니다. image 경로는 JSONL이 있는 data-dir의 상대경로입니다. 학습 loader는 경로가 데이터 디렉터리를 벗어나거나 이미지가 없으면 거부합니다.

실제 준비된 클래스 수는 다음과 같습니다.

| split | angular_leaf_spot | bean_rust | healthy |
|---|---:|---:|---:|
| train | 28 | 33 | 35 |
| validation | 10 | 7 | 7 |
| test | 9 | 7 | 8 |

완전한 stratified split은 아니므로 macro F1도 함께 봅니다. 잎 촬영 조건과 병해의 다른 분포에 대한 일반화는 이 작은 데이터만으로 결론 내리지 않습니다.

## Manifest와 snapshot 재현

각 데이터 디렉터리의 `manifest.json`에 source, dataset ID, 실제 revision SHA, seed, 최종 counts, JSONL SHA256, 라이선스와 이미지 클래스 수를 기록합니다. 이번 데이터 snapshot은 다음과 같습니다.

| 데이터 | revision SHA |
|---|---|
| Dolly | bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a |
| SciQ | 2c94ad3e1aafab77146f384e23536f97a4849815 |
| Beans | 27aa014ce09b193e1a6f58112d4a66e0eddb69c5 |

```bash
python -m finetune_lab.prepare_data --task knowledge --source hf --revision 2c94ad3e1aafab77146f384e23536f97a4849815 --output-root data/reproduced
```

HF cache 경로는 Spark launcher가 `.cache/huggingface`에 연결합니다. 데이터·cache·모델 산출물은 source 코드와 분리합니다. data/demo의 자체 작성 text와 합성 이미지는 CC0 교육 fixture로 제공하며, 실제 공개 데이터셋 결과로 표시하지 않습니다.

## Post-training 자체 작성 데이터

새 실습에는 `data/demo/post_preferences/`(train 16 / validation 4 / test 4)와 `data/demo/post_math/`(32 / 8 / 8)을 함께 제공합니다. CC0-1.0 교육 fixture이며 manifest에 출처·counts·SHA256을 기록합니다. 각 `sft/` 하위 폴더에는 동일 source split의 `id/prompt/completion`을 준비해 SFT 출발 모델을 만들 수 있습니다.

Preference schema는 `id/prompt/chosen/rejected`입니다. 작은 덧셈의 올바른 정수 답을 chosen, 틀린 답을 rejected로 작성했습니다. 사람 선호를 실제 수집한 데이터가 아니며, DPO/reward/PPO 경로를 이해하기 위한 synthetic preference입니다. Math schema는 `id/prompt/answer/completion`이며 정답은 하나의 canonical integer입니다. GRPO verifier는 생성 문자열을 코드로 실행하지 않고 단일 정수 형식과 exact match로 검사합니다.

두 데이터는 목적이 다른 별도 분기입니다. 각각 SFT→post-training에서 같은 source split을 유지하고 test를 학습에 사용하지 않습니다. Preference 수업은 pair 지표와 생성 결과를, RLVR는 독립 산술 정답률과 invalid format을 평가합니다. 현재 tiny fixture를 인간 정렬이나 일반 추론 능력 benchmark로 보고하지 않습니다. 공개 preference/math 데이터로 확대하려면 출처·license·label 생성 방식·source grouping을 별도 확인합니다.

준비된 파일은 그대로 사용합니다. 다시 생성하려면 새 출력 root를 정합니다. 기존 디렉토리는 보호됩니다.

```bash
# Spark 컨테이너 내부: 데이터를 새 경로에 생성할 때만
python -m finetune_lab.post_data --task all --output-root data/reproduced_post
```

구체적인 SFT 변환·학습·평가 명령은 [post-training 수업](../04_post_training/README.md)에 있습니다.

작은 fixture를 마친 다음 공개 데이터로 확대하는 계획은 다음과 같습니다. 아래 데이터는 이번 새 실습용으로 다운로드/변환을 완료한 상태가 아닙니다.

| 확장 | 공식 데이터 카드 | 변환 및 평가 계획 |
|---|---|---|
| DPO 선호 | [trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) | chosen/rejected message의 동일 user prompt를 확인하고 pair로 변환; score tie 제외 정책·label 생성 출처를 기록 |
| 사람 feedback RLHF | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | chosen/rejected 대화의 공통 이전 turn과 마지막 assistant 응답을 분리; multi-turn을 단일 turn처럼 자르지 않고 source 대화 단위 split |
| 산술에서 수학 RLVR로 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | question과 answer의 최종 정답을 분리; 공식 train에서 validation을 만들고 test는 보존; reasoning 출력 parser/verifier를 별도 설계 |

원본을 읽을 때 HF dataset revision SHA와 license를 확인하고 manifest에 고정합니다. 본 수업의 문자열 schema와 prompt 형식은 원본 conversational schema와 다를 수 있으므로 그대로 `--data-dir`에 넣지 않습니다. UltraFeedback 계열 feedback을 사람 평가라고 간주하지 않고 label 출처를 확인합니다. 현재 strict-integer GRPO verifier는 전체 reasoning 문자열을 정답으로 받아들이지 않으므로 GSM8K 도입에는 형식·최종 답 parser·오류 사례 tests 보강이 필요합니다.

## 자신의 데이터로 교체하는 절차

동일 JSONL 형식으로 train/validation/test를 준비하고 `--data-dir`로 전달합니다. 개인정보나 비밀 문서는 입력 단계에서 제외합니다. JSONL schema 검사는 내용의 사실성·사용 권한·개인정보 검사를 대신하지 않습니다. 직접 작성한 예제는 정답 오류·언어·길이 분포부터 점검합니다.
