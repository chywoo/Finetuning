# 문제 해결: Spark에서 실패 원인 좁히기

## 설치 / ARM64 / CUDA

`No matching distribution`은 wheel이 ARM64·Python·CUDA 조합을 지원하지 않거나 package version이 없을 때 발생합니다. 호스트에서 generic `pip install torch`로 교체하지 말고 NGC 컨테이너의 runtime과 `requirements/spark-*.txt`를 사용합니다. `scripts/install_spark.sh`가 보호한 runtime constraint와 충돌하는 패키지 이름을 확인합니다.

`no kernel image is available`, `unsupported gpu architecture`, Triton compile 오류는 모델 데이터보다 CUDA/Blackwell build 문제일 수 있습니다. `doctor --require-spark`의 작은 BF16 forward/backward부터 확인하고 [Spark 안내](04_dgx_spark.md)의 공식 source build 대안을 검토합니다. CUDA driver 버전과 컨테이너 CUDA runtime 버전은 다를 수 있습니다.

Unsloth import 이전에 Transformers/TRL을 import한 프로세스는 패치 상태가 달라질 수 있습니다. 새 프로세스에서 `train.py`를 실행합니다. 이 저장소의 Unsloth 경로는 Unsloth를 먼저 import하고 별도 overlay를 사용합니다. notebook에서 실패한 import 뒤에 반복 실행하는 것보다 kernel을 재시작합니다.

## Out of memory / 급격한 속도 저하

1. `--batch-size 1`로 줄입니다.
2. text `--max-length`를 줄입니다. prompt/response가 잘리는 비율을 확인합니다.
3. full 대신 LoRA, 이어 QLoRA를 비교합니다.
4. 모델 크기·학습 범위를 줄입니다. VLM은 이미지 해상도/patch 수를 점검합니다.
5. 메모리 여유가 확인되면 accumulation을 조절합니다. accumulation 증가가 단일 microbatch activation을 줄이는 방법은 아닙니다.

Spark에서는 CPU와 GPU가 RAM을 공유합니다. dataloader와 cache·OS 메모리도 사용량에 포함됩니다. 이미지 token이 잘려서는 안 되므로 VLM text truncation으로 OOM을 우회하지 않습니다. 이미지 processor의 설정을 바꾸면 train/eval 양쪽에 동일하게 적용하고 processor도 저장합니다.

## NaN / Inf / 학습 가능한 파라미터 없음

학습률을 줄이기 전에 labels에 유효 target이 있는지 확인합니다. prompt 전체·padding·모든 EOS를 `-100`으로 만들었다면 loss가 잘못됩니다. text의 causal shift를 모델 내부와 전처리에서 중복 적용하지 않습니다.

LoRA target 이름이 해당 모델 구조에 없는지, vision만 동결하려다 언어까지 동결했는지 trainable parameter 수를 확인합니다. 각 실습은 수정한 파라미터 범위를 출력합니다. 유효 loss 검사 실패를 logging에서 숨기지 않습니다.

## 파일 / schema / revision

데이터가 없으면 `prepare_data`를 별도로 실행합니다. 기존 데이터를 덮어쓰는 경우는 `--overwrite`, 새로운 실험은 별도의 `--output-root`를 사용합니다. 빈 split, 중복 ID/prompt, 잘못된 label, 데이터 디렉터리 밖의 이미지 경로는 수정한 뒤 다시 시도합니다.

Adapter만으로 전체 모델을 로드하면 필요한 base가 빠집니다. `training_metadata.json`, `adapter_config.json`의 base ID·revision을 보존하고 공통 evaluator를 사용합니다. CPT 후 SFT에서 전체 checkpoint와 adapter를 혼동하지 않습니다. merge는 새 output-dir로 저장합니다.

## Loss는 감소하지만 생성 답변은 좋지 않음

학습량이 너무 적거나, 작은 모델의 용량이 제한적이거나, 질문 형식이 학습과 다를 수 있습니다. 먼저 `instruction_response_v1`과 추론 prompt가 일치하는지 확인합니다. 답변 생성 길이와 EOS도 확인합니다.

공개 지식 데이터는 원래 모델의 학습에 이미 포함되었을 수 있습니다. baseline을 함께 보고, 새로운 사실 recall과 미학습 사실 일반화를 나눕니다. 평가 prompt에 정답을 포함한 문맥을 주지 않았는지도 점검합니다. 학습 데이터 train loss만으로 능력 향상을 선언하지 않습니다.
