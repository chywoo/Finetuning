# Hugging Face Trainer와 PEFT로 VLM LoRA 학습하기

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

Post-training의 확률/loss·보상 수업을 마친 뒤 PyTorch V1 → HF V2 순으로 진행합니다. Unsloth V3는 다른 모델을 사용하는 선택 확장입니다. 완료하면 [추가 비교 과제](../../docs/08_next_experiments.md)로 이동합니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, 방법에 맞는 실행 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이미지와 질문을 입력받는 `HuggingFaceTB/SmolVLM-256M-Instruct`를
콩 잎 상태의 문자열을 생성하도록 적응시킨다. `angular_leaf_spot`,
`bean_rust`, `healthy`를 학습하고, 학습 전후 같은 이미지에서 답을 생성해 비교한다.
이미 이미지 입력을 지원하는 모델을 사용하므로 새 모달리티를 처음 만드는 과제와는 다르다.

## 1. VLM과 LoRA의 이론

VLM의 시각 인코더는 RGB 픽셀을 특징으로 변환하고 connector/projector는
그 특징을 언어 디코더의 표현 공간으로 연결한다. 디코더는 이미지 표현과
텍스트 질문을 조건으로 답을 생성한다. 이 실습의 supervision은 클래스
문자열과 assistant 종료 토큰이며 이미지 위치 좌표나 병변 마스크는 없다.
따라서 학습 결과로 객체 검출이나 분할 성능을 주장할 수 없다.

LoRA는 원래 행렬 `W`를 고정하고 작은 행렬 `A`, `B`로 `W' = W + sBA`를
만들며 이 추가 행렬만 학습한다. Rank `r`가 작으면 optimizer 상태와 저장하는
가중치를 줄일 수 있다. 연산과 activation, 원본 모델을 읽는 메모리가 전부
없어지는 것은 아니다. 이 예제는 rank 8, alpha 16, dropout 0.05를 사용한다.
[PEFT 공식 LoRA 문서](https://huggingface.co/docs/peft/developer_guides/lora)

구체적으로 이름이 `text_model.layers.<번호>.self_attn.q_proj` 또는 `v_proj`인
**언어 attention projection**에만 adapter를 붙인다. Vision encoder의
`q_proj`/`v_proj`에도 같은 이름이 있을 수 있어 단순한 suffix 목록 대신 언어
decoder 경로를 포함한 정규식을 사용한다. 모델 구조가 다르면 대상 모듈을
확인해야 하며, 이 예제는 맞는 모듈이 없을 때 오류를 낸다.

Vision encoder와 connector는 고정된다. 언어 쪽이 기존 이미지 특징에서
어떤 상태 라벨을 출력할지 학습한다. 미세한 병변 표현을 인코더 자체에
새로 학습시키려면 시각 쪽 adapter/일부 가중치를 학습하는 별도 설정과
데이터가 필요하다. Adapter만 비교해도 일반 이미지 질문에 대한 답변 능력이
손상될 수 있으므로 과제 평가와 기존 능력 평가를 함께 준비한다.

## 2. 데이터

[AI-Lab-Makerere/beans 공식 카드](https://huggingface.co/datasets/AI-Lab-Makerere/beans)의
콩 잎 사진을 사용한다. 원본 split은 train 1,034, validation 133, test 128이다.
라벨은 `0 → angular_leaf_spot`, `1 → bean_rust`, `2 → healthy`로 해석한다.
HF 준비 명령은 공식 split을 유지한 작은 부분집합을 이미지 파일과 JSONL로 저장한다.
샘플 수와 클래스 분포, 데이터 revision은 `data/processed/vision/manifest.json`에 기록된다.

```json
{"id":"beans-train-00001","image":"images/train-00001.jpg","label":"bean_rust"}
```

`image`는 JSONL 폴더 기준 상대 경로다. JSONL과 `images/`를 함께 보관한다.
데이터 카드의 라이선스는 명확한 허용 조건을 제시하지 않으므로 임의로 CC-BY로
표기하지 않는다. 내려받은 데이터를 재배포하거나 제품에 쓰려면 원 출처의 조건을 확인한다.
`--source demo`로 만드는 그림은 다운로드 없이 실행을 점검하는 합성 fixture이며
콩 병해 사진이 아니다. 학습 점수를 실제 이미지 인식 성능으로 해석하지 않는다.

## 3. 실제 진행

일반 환경은 [빠른 시작](../../docs/00_quickstart.md)의 `.venv-lab`과 `requirements/lab-hf.txt`를 준비합니다. 아래 명령은 프로젝트 루트에서 실행합니다. 작은 텍스트 full·LoRA는 CPU에서도 가능하며, VLM은 모델 크기와 processor 메모리를 확인하고 CUDA를 권장합니다. DGX Spark 사용자는 [전용 환경 안내](../../docs/04_dgx_spark.md)의 NGC overlay를 선택합니다.
작은 batch와 적은 step으로 입출력과 저장·재로드부터 확인한다.

```bash
python -m finetune_lab.prepare_data --task vision --source hf \
  --train-samples 96 --eval-samples 24

python 03_vision/huggingface/train.py --dry-run
python 03_vision/huggingface/train.py --evaluate --device cuda --split validation \
  --max-eval-samples 24 --output-dir outputs/vision/baseline-hf-validation

python 03_vision/huggingface/train.py --device cuda --max-steps 20 \
  --batch-size 1 --gradient-accumulation 4 --learning-rate 2e-4 \
  --output-dir outputs/vision/huggingface

python 03_vision/huggingface/train.py --evaluate --device cuda \
  --model outputs/vision/huggingface --split validation --max-eval-samples 24 \
  --output-dir outputs/vision/hf-validation
```

설정을 validation에서 선택한 뒤 baseline과 adapter를 `--split test`로 평가한다.
비교마다 동일한 질문·이미지·processor 크기를 사용한다. Test를 매번 보고
학습률을 바꾸면 test도 설정 선택에 사용한 데이터가 되어 최종 평가 의미가 줄어든다.

`--max-steps 20`은 optimizer 업데이트 20회다. batch 1, accumulation 4에서는
약 80회 이미지 입력을 처리한다. 짧은 smoke run으로 정확도 개선을 보장하지 않는다.
`--device cuda`를 명시해 CUDA GPU를 사용한다. CUDA에서는 하드웨어에
따라 bf16/fp16 AMP를 사용하며 기본 가중치는 float32로 로드한다.
학습 시작 시 출력되는 trainable parameter 수로 예상한 adapter만 학습하는지 확인한다.

Wrapper를 절대 경로로 실행하면 다른 작업 디렉토리에서도 사용할 수 있다.
기본 data/output 디렉토리는 저장소 기준이고 직접 지정한 상대 경로는 현재
작업 디렉토리 기준이다. 데이터 다운로드와 학습은 별도 명령이므로 준비 후
학습 실행이 데이터를 몰래 다시 내려받지는 않는다.

## 4. Trainer에서 주의할 점

공통 구현 `finetune_lab/vision.py`의 `_run_hf`는 PEFT adapter를 만든 다음
Transformers `Trainer`로 학습한다. Trainer가 optimizer, gradient accumulation,
로그와 마지막 validation loss 계산을 관리한다.
[코드 읽기 안내](../../graft/finetune_lab/vision.md)에서 주요 함수의 관계를 확인한다.
[Trainer 공식 문서](https://huggingface.co/docs/transformers/main_classes/trainer)

- `remove_unused_columns=False`: 원본 `image`, `label` 필드가 collator에 도착해야 한다.
- `VisionCollator`: 이미지와 텍스트를 함께 processor에 넣어 `pixel_values`와
  확장된 `input_ids`를 만든다. SmolVLM longest edge는 512로 제한한다.
- `truncation=False`: 이미지 placeholder 일부가 잘리면 image feature와
  토큰 수가 맞지 않을 수 있어 단순한 텍스트 길이 제한을 사용하지 않는다.
- 손실 mask: 같은 이미지를 사용한 prompt/full 대화의 processor 결과를
  비교해 assistant 정답 이후만 남긴다. Prefix가 다르면 학습을 중단한다.
- Padding: attention mask를 사용해 `-100`으로 제외한다. 실제 EOS는
  pad token ID와 같더라도 정답 토큰으로 남길 수 있다.

이렇게 만든 `labels`에 대해 VLM 자체의 causal language modeling loss를
쓴다. 세 클래스에 대한 softmax head를 별도로 학습하는 접근과 평가·출력
형식이 다르다. 라벨을 포함한 assistant 답은 학습 입력에 존재하지만, 평가
프롬프트에는 답을 넣지 않는다.

## 5. 저장과 재로드

출력 폴더에는 `adapter_config.json`, adapter 가중치, tokenizer/processor와
`training_metadata.json`을 저장한다. 원래 모델의 모든 가중치를 다시 저장하지 않는다.
`training_metrics.json`에는 Trainer의 학습/validation loss와 실행 로그를 기록한다.
따라서 재로드할 때는 동일한 원래 SmolVLM checkpoint와 adapter가 모두 필요하다.
기존 adapter를 입력으로 재학습하는 continuation은 이 예제에서 지원하지 않는다.
새 학습은 base model을 지정하고 새 `--output-dir`을 사용한다. 저장된 모델/adapter
폴더를 출력 경로로 지정하면 덮어쓰기 전에 오류를 낸다.
평가 명령은 adapter 설정과 metadata에서 base model과 revision을 읽고
`PeftModel.from_pretrained`로 결합한다. 정확한 모델 버전은 `--revision`에
Hub commit SHA를 지정한다. 데이터 SHA는 준비 단계 manifest에 저장된다.

`--evaluate`는 최대 16개의 답변 토큰을 greedy 생성해 `evaluation.json`에 저장한다.
앞뒤 공백/대소문자 정규화 후 라벨 문자열과 정확히 같아야 유효하다.
문장으로 답하거나 여러 라벨을 생성하면 `invalid`이며 accuracy와 macro-F1에서
오답으로 계산된다. 혼동 행렬 행은 실제 라벨, 열은 예측 라벨과 invalid다.
Loss 개선과 실제 이미지 분류 개선이 항상 함께 움직이지는 않는다.

## 6. 직접 확인할 실험

- 원본 모델과 adapter의 validation macro-F1, invalid 답변 비율을 비교한다.
- PyTorch의 마지막 decoder block 학습과 LoRA를 같은 데이터에서 비교한다.
- Rank·학습률을 바꿀 때 한 번에 한 변수만 바꾼다. 설정마다 출력 폴더를 바꿔 보관한다.
- 이미지 배경만 보고 예측하는지, 새로운 촬영 조건에서도 맞추는지 별도 사진으로 확인한다.
- 잎 사진 외의 일반 이미지 질문에서도 원래 응답 능력이 유지되는지 기록한다.

## 수업 완료 기준

1. 준비: 세 split과 manifest를 확인하고 dry-run의 데이터 수·모델·학습 방법을 설명합니다.
2. 실행: 실제 학습이 유한 loss로 종료되고 예상한 전체 모델 또는 adapter·tokenizer·metadata가 새 출력 경로에 저장됩니다.
3. 재사용: 별도 프로세스에서 저장 결과를 읽어 답변을 생성합니다. Adapter이면 동일 base와 revision을 사용합니다.
4. 해석: 동일 이미지와 prompt의 accuracy·macro-F1·invalid 출력 및 클래스별 오분류를 비교합니다. 설정을 고른 뒤 test를 최종 평가합니다.

실행 성공과 품질 개선은 각각 기록합니다. 수업을 준비했거나 dry-run만 통과한 상태를 학습 완료로 표시하지 않습니다. 다음 단계는 이 문서 첫머리의 수업 경로와 [커리큘럼](../../docs/11_curriculum.md)을 따릅니다.
