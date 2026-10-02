# Unsloth로 Qwen2.5-VL 이미지 과제 QLoRA 학습하기

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

Post-training의 확률/loss·보상 수업을 마친 뒤 PyTorch V1 → HF V2 순으로 진행합니다. Unsloth V3는 다른 모델을 사용하는 선택 확장입니다. 완료하면 [추가 비교 과제](../../docs/08_next_experiments.md)로 이동합니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, Spark CUDA 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 경로는 `Qwen/Qwen2.5-VL-3B-Instruct`에 콩 잎 분류 과제를 학습한다.
이미지+질문을 입력하면 `angular_leaf_spot`, `bean_rust`, `healthy` 중 하나를
텍스트로 생성한다. Unsloth 공식 vision fine-tuning 문서는 Qwen2.5-VL 계열과
`FastVisionModel`, `UnslothVisionDataCollator`를 다룬다.
[공식 vision 안내](https://unsloth.ai/docs/basics/vision-fine-tuning)

앞의 PyTorch/Hugging Face 실습은 작은 SmolVLM을 사용하고 이 실습은 더 큰
Qwen을 사용한다. 실행 가능 환경과 지원 모델을 고려한 선택이므로 점수·시간의
차이를 Unsloth 효과만으로 해석할 수 없다. Backend 성능을 공정하게 비교하려면
동일 모델·정밀도·학습 모듈·데이터·step 수의 별도 실험이 필요하다.

## 1. 이론: 시각 학습, LoRA, 양자화

VLM에서는 시각 인코더가 픽셀을 특징으로 만들고 connector/projector가 그
특징을 언어 디코더에 연결한다. Decoder는 이미지와 텍스트 조건에서 정답
토큰의 확률을 높이는 supervised fine-tuning을 수행한다.
Qwen2.5-VL은 이미지와 동영상 관련 입력을 처리하는 모델 계열이며 이 예제는
정지 이미지 하나만 사용한다. 모델의 상세 입출력은
[공식 Qwen 모델 카드](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)에서 확인한다.

LoRA는 원래 가중치를 고정한 채 작은 low-rank 추가 행렬을 학습한다.
QLoRA는 여기에 4-bit로 읽은 base model을 함께 사용해 메모리를 더 줄이는
접근이다. 저장하는 adapter는 원본 모델을 대체하는 완전한 checkpoint가 아니다.
양자화된 원본 모델을 직접 보통의 full fine-tuning 방식으로 업데이트하는 실습도 아니다.
메모리에는 base model 외에도 activation, 이미지 입력, gradient, optimizer
상태가 있으므로 4-bit라고 전체 사용량이 정확히 1/4이 되지는 않는다.

이 예제는 language attention의 LoRA를 켜고 vision layers와 MLP adapter를 끈다.
`finetune_vision_layers=False`, `finetune_language_layers=True`,
`finetune_attention_modules=True`, `finetune_mlp_modules=False`이며
rank 8, alpha 16을 사용한다. 고정된 시각 표현을 새 라벨과 연결하도록
언어 쪽이 적응한다. 시각 인코더 자체의 미세한 병변 특징을 바꾸려면 그쪽의
학습 범위와 데이터 양을 별도로 조정한다. 실제 trainable module 목록도 확인한다.

세 문자열에 대한 generative classification이므로 라벨이 여러 토큰으로
나뉠 수 있다. Loss는 assistant 응답만 대상으로 하며 입력 이미지·질문은
정답 예측의 조건이다. 좌표·분할 annotation은 사용하지 않으므로 객체 검출
학습이나 새로운 시각 모달리티 추가와는 학습 목표가 다르다.

## 2. 데이터와 저장 형태

[AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans)는
콩 잎 사진의 3-class 데이터다. 공식 train 1,034, validation 133, test 128 중
기본 실습은 96/24/24장을 준비한다. 라벨 0/1/2는 순서대로
`angular_leaf_spot`, `bean_rust`, `healthy`다. 공식 split 경계를 유지하며
다운로드 revision, 실제 클래스 분포와 사용 수는 `manifest.json`에 기록된다.

```json
{"id":"beans-train-00001","image":"images/train-00001.jpg","label":"healthy"}
```

이미지 경로는 `data/processed/vision`을 기준으로 해석한다.
학습 시 이미지를 RGB PIL 객체로 읽고 user의 image content와 질문, assistant의
문자열 답을 가진 messages로 변환한다. JSONL 파일에 큰 이미지 bytes를 넣지 않는다.
`--source demo`는 병해 의미가 없는 합성 fixture이므로 이 데이터로 얻은 점수는
실제 질병 분류 성능을 뜻하지 않는다. 실제 비교에는 HF 사진을 사용한다.
데이터 카드의 라이선스는 명확하게 지정되어 있지 않으므로 배포·상업 사용 전에
원 출처의 조건을 확인하고 임의로 CC-BY라고 표기하지 않는다.

## 3. NVIDIA CUDA 환경 준비

이 저장소의 실제 실행 장치는 **DGX Spark의 ARM64/GB10 CUDA**다.
먼저 [공통 DGX Spark 환경 문서](../../docs/04_dgx_spark.md)와
[NVIDIA Unsloth playbook](https://build.nvidia.com/spark/unsloth)을 따라
Spark용 Docker 환경을 만든다. 일반 x86 CUDA wheel이나
다른 GPU용 컨테이너가 GB10/ARM64에서도 그대로 동작한다고 가정하지 않는다.
공통 설치 스크립트는 playbook의 NGC 환경을 기준으로 Unsloth dependency를 구성한다.
Unified memory는 CPU/GPU가 공유하므로 전체 메모리 사용량도 함께 확인한다.

PyTorch/HF 실습과 분리한 Unsloth 컨테이너를 사용한다. 아래 명령은 준비한
컨테이너 안에서 저장소 최상위 디렉토리를 기준으로 실행한다. 기존 CUDA
PyTorch를 pip로 덮어쓰지 않고 공식 이미지의 dependency 조합을 유지한다.
호스트에서 `bash scripts/spark_container.sh unsloth`를 실행하고 컨테이너 안에서
`bash scripts/install_spark.sh unsloth`로 설치를 마친다.
설치 후 `source .venv-spark-unsloth/bin/activate`로 환경을 활성화한다.

```bash
python -m finetune_lab.prepare_data --task vision --source hf \
  --train-samples 96 --eval-samples 24
python 03_vision/unsloth/train.py --dry-run
```

실제 모델 로드 시 `unsloth`를 torch/Transformers보다 먼저 import한다.
단순 dry run은 dependency 없이 JSONL과 이미지 경로만 확인하고 GPU 모델은 로드하지 않는다.
준비한 HF 이미지를 기존 환경에서 가져왔다면 데이터 폴더를 재생성할 필요는 없다.

## 4. baseline → 학습 → 재평가

먼저 validation에서 baseline을 기록한다. 설정을 validation에서 정하고
마지막에 같은 test split으로 baseline과 adapter를 비교한다.

```bash
python 03_vision/unsloth/train.py --evaluate --device cuda \
  --split validation --max-eval-samples 24 \
  --output-dir outputs/vision/baseline-unsloth-validation

python 03_vision/unsloth/train.py --device cuda --max-steps 20 \
  --batch-size 1 --gradient-accumulation 4 --learning-rate 2e-4 \
  --output-dir outputs/vision/unsloth

python 03_vision/unsloth/train.py --evaluate --device cuda \
  --model outputs/vision/unsloth --split validation --max-eval-samples 24 \
  --output-dir outputs/vision/unsloth-validation
```

20 step은 빠른 파이프라인 점검이며 수렴을 의미하지 않는다. batch 1,
accumulation 4에서는 optimizer 업데이트마다 약 4개 이미지를 사용한다.
더 긴 학습은 validation macro-F1과 원래 이미지 질의응답 능력을 보며 결정한다.
작은 데이터에 긴 학습을 하면 라벨이나 배경을 외울 수 있다.

메모리가 부족하면 batch를 줄이고 불필요하게 큰 이미지 해상도를 피한다.
이미지 placeholder가 들어간 토큰열을 임의의 max length로 자르면 시각 특징
수와 불일치할 수 있다. 이 구현은 SFTConfig의 `max_length=None`을 사용하고,
native collator가 모델 길이를 상속하는 경우까지 고려해 truncation을 끈다.
긴 이미지 입력에서는 잘라내는 대신 이미지 해상도와 학습 구성을 조정한다.

## 5. 코드 읽기와 response-only loss

`finetune_lab/vision.py`에서 다음을 순서대로 확인한다.
[코드 읽기 안내](../../graft/finetune_lab/vision.md)도 참고한다.

1. `FastVisionModel.from_pretrained(... load_in_4bit=True)`로 base model을 읽는다.
2. `get_peft_model`로 학습할 언어 attention 범위를 지정하고 training 모드로 바꾼다.
3. RGB 이미지와 답을 messages 형태로 변환한다.
4. `UnslothVisionDataCollator`가 이미지 토큰과 pixel 값을 일관되게 처리한다.
5. Qwen chat template의 user/assistant 구분자와 `train_on_responses_only=True`로
   질문을 mask하고 답변에만 손실을 준다. `completion_only_loss=True`도 지정한다.
6. TRL `SFTTrainer`가 accumulation, optimizer, 로그, validation loss를 관리한다.

라벨 문자열이 입력 대화의 assistant 쪽에 존재하는 것은 teacher forcing 학습에
필요한 정상적인 형태다. 평가에서는 user 메시지만 만들고 정답은 넣지 않는다.
`all labels masked` 오류가 나면 해당 모델의 chat template와 response 구분자가
일치하는지 확인한다. 다른 모델로 바꿀 때는 고정된 Qwen 구분자도 함께 바꿔야 한다.

## 6. 저장, 평가, 다음 실습

출력 폴더에는 adapter, processor/tokenizer, `training_metadata.json`을 저장한다.
`training_metrics.json`에는 학습 loss와 validation 로그를 남긴다.
재평가도 Unsloth wrapper를 사용해 같은 4-bit base+adapter로 로드한다.
원본 모델 revision을 고정하려면 `--revision`에 commit SHA를 지정한다.
기존 adapter의 continuation 학습은 이 예제에서 지원하지 않는다. 새 실험은
base model과 새 `--output-dir`을 지정한다. 저장된 모델/adapter 폴더를 출력
경로로 지정하면 덮어쓰기 전에 오류를 낸다.
저장한 adapter를 다른 base revision에 붙이면 동일한 결과를 기대할 수 없다.

`evaluation.json`은 생성 원문과 accuracy, macro-F1, per-class F1, 혼동 행렬을
보관한다. 앞뒤 공백과 대소문자만 정규화한 뒤 세 라벨 중 하나와 정확하게 같아야
유효한 답이다. 설명 문장이나 두 라벨을 생성하면 `invalid` 열에 기록하고 오답으로 센다.
Train loss가 감소해도 이미지 분류가 개선된다는 보장은 없으므로 baseline과
같은 질문·이미지·조건에서 생성 평가를 수행한다.

다음 단계로 vision adapter도 켜서 언어만 학습했을 때와 비교해 볼 수 있다.
그때는 학습 범위, VRAM, 일반 이미지 질문의 성능을 함께 기록한다. 사용자
도메인으로 확장할 때는 정상/이상 클래스, 촬영 기기·조명, 데이터 누수,
annotation 품질을 먼저 점검하고 실제 이미지들을 독립적인 test로 남겨 둔다.
