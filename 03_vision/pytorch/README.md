# PyTorch로 VLM에 이미지 분류 과제 학습시키기

## 수업의 위치와 다음 단계

처음이라면 [고등학생 입문 수업](../../docs/12_foundations.md)을 먼저 읽습니다. 이 문서의 심화 수식은 [단계별 계획](../../docs/11_curriculum.md)에 맞춰 대학 수준으로 확장하는 부분입니다.

Post-training의 확률/loss·보상 수업을 마친 뒤 PyTorch V1 → HF V2 순으로 진행합니다. Unsloth V3는 다른 모델을 사용하는 선택 확장입니다. 완료하면 [추가 비교 과제](../../docs/08_next_experiments.md)로 이동합니다.

이 수업의 준비물은 데이터 split/manifest, 출발 모델, 방법에 맞는 실행 환경과 baseline입니다. 아래 데이터·환경·실행 절차를 순서대로 읽고 학습→저장→별도 재로딩→전후 비교를 확인한 후 다음 단계로 이동합니다. [체크리스트](../../docs/10_practice_checklist.md)의 해당 ID와 TASK_LOGS를 갱신합니다.

이 실습은 이미 이미지를 볼 수 있는 `HuggingFaceTB/SmolVLM-256M-Instruct`에
콩 잎의 세 가지 상태를 구분하는 과제를 학습시킨다. 이미지와 질문을 넣으면
`angular_leaf_spot`, `bean_rust`, `healthy` 중 하나를 **문자열로 생성**한다.
이미지 입력이 없는 LLM에 새 시각 모달리티를 만드는 작업, 병변의 위치를 찾는
객체 검출, 의료·농업 현장 진단의 검증까지 포함하는 실습은 아니다.

## 1. 이론: 이미지가 어떻게 다음 토큰 예측에 들어갈까?

대략적인 흐름은 `RGB 이미지 → vision encoder → connector/projector →
이미지 토큰과 질문의 표현 → language decoder → 정답 토큰`이다.
Encoder는 픽셀을 시각 특징으로 바꾸고, connector는 시각 특징을 언어 모델이
처리할 수 있는 표현으로 맞춘다. Decoder는 이미지 표현과 질문을 함께 보면서
정답을 한 토큰씩 생성한다. SmolVLM은 SigLIP 계열 이미지 인코더와 SmolLM2
언어 디코더를 결합하며, 이미지와 텍스트를 입력받아 텍스트를 출력한다.
[공식 모델 카드](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

분류기를 따로 붙이지 않아도 다음과 같이 분류 문제를 시각 질의응답으로 바꿀 수 있다.

```text
user: <이미지> Classify the bean leaf ... exactly one label ...
assistant: bean_rust
```

학습 목적은 이미지 `I`, 질문 `x`가 주어졌을 때 정답 `y`의 확률을 높이는 것이다.
`L = -Σ log p(y_t | I, x, y_<t)`로 생각하면 된다. `bean_rust`도 토크나이저에
따라 여러 토큰이므로 문자열 클래스 하나가 손실 항 하나라는 뜻은 아니다.
질문과 이미지 토큰은 입력 조건이고, 손실은 assistant의 정답과 종료 토큰에만 준다.

이 구현은 **마지막 언어 decoder block 2개와 최종 norm의 일반 가중치**를
업데이트한다. Vision encoder와 connector, 나머지 decoder block은 고정한다.
따라서 이미 계산된 이미지 특징을 새 라벨과 연결하는 언어 쪽 적응을 연습한다.
`--decoder-layers`를 늘리면 학습하는 범위와 optimizer 메모리도 늘어난다.
전체 VLM의 모든 가중치를 갱신하는 full fine-tuning과는 학습 범위가 다르다.

언어 쪽만 학습해도 이미지에 대한 출력은 달라질 수 있다. 다만 인코더가 잘
표현하지 못하는 미세한 병변 정보를 새로 추출하도록 직접 바꾸지는 않는다.
그런 도메인에서는 더 많은 데이터와 함께 encoder/connector까지 학습하는
실험을 별도로 설계해야 한다. 적은 데이터에 큰 학습 범위를 쓰면 기존 능력을
잊거나 배경과 촬영 조건을 외우기 쉽다.

## 2. 데이터와 라벨

실제 데이터는 [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans)다.
콩 잎 사진과 질병/건강 라벨이 있으며 공식 split은 train 1,034장,
validation 133장, test 128장이다. 원본 숫자 라벨 0, 1, 2를 각각
`angular_leaf_spot`(각진 잎반점), `bean_rust`(콩 녹병), `healthy`로 바꾼다.
실습에서는 기본적으로 작은 부분집합을 저장하고 공식 split의 경계를 유지한다.
정확한 사용 샘플 수와 클래스별 수는 `data/processed/vision/manifest.json`에서 확인한다.

```json
{"id":"beans-train-00001","image":"images/train-00001.jpg","label":"healthy"}
```

이미지 경로는 데이터 디렉토리를 기준으로 한 상대 경로다. 파일을 이동할 때는
JSONL과 `images/`를 함께 이동한다. 로더는 파일 존재와 경로 이탈, 라벨을 검사한다.
Hub 카드의 라이선스 항목은 명확한 허용 조건을 제시하지 않는다.
이를 CC-BY 데이터라고 가정하지 말고 배포·상업적 재사용 전 원 출처의 조건을 확인한다.

`--source demo`는 내려받기 없이 파이프라인을 확인하는 합성 그림을 만든다.
이 그림에는 실제 병해 의미가 없고, 점수로 질병 인식 능력을 판단할 수 없다.
실제 성능 비교에는 `--source hf`로 받은 사진을 사용한다.

## 3. 준비와 실행

일반 환경은 [빠른 시작](../../docs/00_quickstart.md)의 `.venv-lab`과 `requirements/lab-hf.txt`를 준비합니다. 아래 명령은 프로젝트 루트에서 실행합니다. 작은 텍스트 full·LoRA는 CPU에서도 가능하며, VLM은 모델 크기와 processor 메모리를 확인하고 CUDA를 권장합니다. DGX Spark 사용자는 [전용 환경 안내](../../docs/04_dgx_spark.md)의 NGC overlay를 선택합니다.
모델과 데이터의 첫 다운로드에는 인터넷이 필요하다. PyTorch 방식도
모델/processor를 읽기 위해 Transformers를 사용하지만 학습 loop와 optimizer는 직접 작성한다.

```bash
python -m finetune_lab.prepare_data --task vision --source hf \
  --train-samples 96 --eval-samples 24
python 03_vision/pytorch/train.py --dry-run
```

다음은 모델을 아직 학습하지 않은 상태의 성능을 저장한다. 일반적으로 먼저
validation에서 실험하고, 설정을 결정한 뒤 test를 최종 비교에 한 번 사용한다.

```bash
python 03_vision/pytorch/train.py --evaluate --device cuda --split validation \
  --max-eval-samples 24 --output-dir outputs/vision/baseline-pytorch-validation

python 03_vision/pytorch/train.py --device cuda --max-steps 20 \
  --batch-size 1 --gradient-accumulation 4 --learning-rate 2e-5 \
  --output-dir outputs/vision/pytorch

python 03_vision/pytorch/train.py --evaluate --device cuda \
  --model outputs/vision/pytorch --split validation --max-eval-samples 24 \
  --output-dir outputs/vision/pytorch-validation
```

`--max-steps`는 optimizer 업데이트 횟수다. 위 설정은 업데이트마다 최대 4개
이미지의 gradient를 누적하므로 20번에 약 80개 이미지 입력을 처리한다.
빠른 실행 점검을 위한 값이며 수렴이나 정확도 개선을 보장하지 않는다.
실제 GPU 실행은 호환되는 CUDA 환경에서 확인한다. `--device cuda`를 명시해
GPU 환경이 준비되지 않았을 때 바로 오류를 확인한다. 이 교육용 직접 loop는
float32를 사용한다. BF16/AMP를 추가하는 최적화는 별도 실험으로 비교할 수 있다.
메모리가 부족하면 batch size와 decoder layer 수를 줄인다.

다른 작업 디렉토리에서 wrapper를 실행할 때는 wrapper 자체를 절대 경로로
지정한다. 기본 data/output 경로는 파일 위치를 기준으로 계산되므로 유지된다.
직접 지정한 상대 `--data-dir`/`--output-dir`은 현재 작업 디렉토리를 기준으로 해석된다.

## 4. 코드와 학습 과정 읽기

공통 구현은 `finetune_lab/vision.py`에 있고 이 폴더의 `train.py`가 PyTorch 방식을 선택한다.
[코드 읽기 안내](../../graft/finetune_lab/vision.md)도 함께 확인한다.

1. `load_rows`가 train/validation JSONL을 검증하고 이미지를 확인한다.
2. `VisionCollator`가 RGB 이미지를 열고 학습 대화를 만든다.
3. Processor가 이미지 크기와 토큰을 함께 처리한다. SmolVLM 실습에서는 longest
   edge를 512로 제한하고 tokenizer truncation을 사용하지 않는다.
4. 이미지가 여러 시각 토큰으로 확장된 **processor 결과**에서 prompt/full prefix를
   비교한다. 질문·이미지·padding은 `-100`으로 손실에서 제외한다.
5. `model(**batch).loss → backward → gradient clipping → AdamW.step`을 실행한다.
   Gradient accumulation에서는 손실을 누적 횟수로 나눈다.
6. 모델 전체 파일과 processor, `training_metadata.json`을 저장한다.

Prompt 길이를 일반 텍스트 토크나이저로만 세면 이미지 토큰 확장 후 길이가
달라져 정답 경계를 잘못 잡을 수 있다. prefix가 일치하지 않으면 이 구현은
계속 학습하지 않고 오류를 낸다. Padding은 attention mask로 제외하므로
pad token과 EOS가 같은 경우에도 실제 정답 종료 토큰은 유지된다.
이런 입력은 텍스트 전용 collator에 맡기기보다 이미지와 텍스트를 함께 처리해야 한다.
[Idefics3 입력/processor 설명](https://huggingface.co/docs/transformers/v4.51.3/model_doc/idefics3)

## 5. 저장 결과와 평가 해석

PyTorch 경로는 adapter가 아니라 **전체 모델 checkpoint**를 저장한다.
`config.json`, 모델 가중치, processor/tokenizer 파일을 함께 보관한다.
`training_metrics.json`은 optimizer step별 학습 loss를 저장한다.
이미 저장된 모델이 있는 output 폴더는 학습으로 덮어쓰지 않으므로 새 실험마다
새 `--output-dir`을 선택한다.
`training_metadata.json`에는 모델 ID, revision, 샘플 수, 설정과 질문을 기록한다.
정확한 재현을 원하면 `--revision`에 모델 commit SHA를 지정한다.

평가는 정답을 프롬프트에 넣지 않고 최대 16개 새 토큰을 greedy 생성한다.
`evaluation.json`에 원문 생성 결과, 라벨, accuracy, 클래스별 F1, macro-F1,
혼동 행렬을 저장한다. 앞뒤 공백과 대소문자를 정규화한 후 세 문자열 중
하나와 정확히 같아야 정답 후보가 된다. `The leaf is healthy`는 `invalid`로
기록하며 오답에 포함한다. 혼동 행렬의 마지막 열은 이 형식 오류를 보여 준다.

Validation loss가 내려가도 test의 이미지 분류 점수가 개선되지 않을 수 있다.
항상 같은 이미지, 질문, processor 크기, 샘플 수로 baseline과 비교한다.
세 클래스의 macro-F1은 클래스별 F1을 같은 가중치로 평균해 다수 클래스만
맞추는 모델을 구분한다. 작은 부분집합에서는 수치 변동이 크므로 충분한
데이터와 여러 seed로 반복하고 일반 이미지 질문의 기존 능력도 확인한다.

## 6. 다음 실험

- `--decoder-layers 1`과 `2`를 비교해 학습 범위의 효과를 확인한다.
- 같은 데이터로 옆 `huggingface/`의 LoRA와 저장 크기·시간·점수를 비교한다.
- 전체 공식 split을 사용하려면 준비 명령의 train/eval 샘플 제한을 늘린다.
  validation/test는 서로 크기가 다르므로 실습 도구가 허용하는 최대치를 확인한다.
- 이미지 분류가 목표라면 작은 전용 ViT/ResNet 분류기와도 비교한다. VLM은
  질문과 이미지 설명으로 확장하기 쉽지만 순수 분류에서 항상 가장 경제적이지는 않다.

## 수업 완료 기준

1. 준비: 세 split과 manifest를 확인하고 dry-run의 데이터 수·모델·학습 방법을 설명합니다.
2. 실행: 실제 학습이 유한 loss로 종료되고 예상한 전체 모델 또는 adapter·tokenizer·metadata가 새 출력 경로에 저장됩니다.
3. 재사용: 별도 프로세스에서 저장 결과를 읽어 답변을 생성합니다. Adapter이면 동일 base와 revision을 사용합니다.
4. 해석: 동일 이미지와 prompt의 accuracy·macro-F1·invalid 출력 및 클래스별 오분류를 비교합니다. 설정을 고른 뒤 test를 최종 평가합니다.

실행 성공과 품질 개선은 각각 기록합니다. 수업을 준비했거나 dry-run만 통과한 상태를 학습 완료로 표시하지 않습니다. 다음 단계는 이 문서 첫머리의 수업 경로와 [커리큘럼](../../docs/11_curriculum.md)을 따릅니다.
