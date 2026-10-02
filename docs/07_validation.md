# 검증 상태와 Spark 실행 체크

## 현재 상태

새 post-training 범위는 [실습 계획](11_curriculum.md)과 [체크리스트](10_practice_checklist.md)의 P1–P4를 따른다. Domain은 기존 HF CPT/QA 경로를 재사용한다. DPO/Reward/PPO/GRPO는 post overlay에서 각각 학습·저장·별도 재로딩·독립 평가를 실행해야 하며, 아래 기존 HF/Unsloth matrix에 포함된 것으로 표시하지 않는다.

코드·데이터·상세 설명과 Spark 실행 스크립트를 작성했습니다. 실제 공개 데이터 Dolly/SciQ/Beans subset을 다운로드하고 manifest를 저장했습니다. 사용자 지시 이후 **Intel Mac의 학습·환경 검증을 중단했습니다.** 최신 Spark/AMP/의존성 조합에 대한 GPU 실행은 아직 수행하지 않았습니다.

지시 이전의 일부 작은 CPU 파이프라인 실행은 Spark의 ARM64 CUDA·Triton·bitsandbytes 호환성을 증명하지 않습니다. 결과를 Spark 성능 비교로 사용하지 않습니다. 변경 전 unit tests가 통과했더라도 최신 코드의 GPU 성공을 주장하지 않습니다.

코드 리뷰에서 response/EOS mask, causal shift, token-weighted PyTorch accumulation, VLM image-token 위치, adapter/base 재로딩, 경로 검사와 output 덮어쓰기 방지를 점검했습니다. Graft의 파일별 카드로 해당 구현 위치를 찾았습니다.

## Spark에서 실행할 검증

HF profile은 직접 PyTorch SFT/CPT, Trainer full, CPT LoRA → QA SFT adapter 이어 학습, 별도 프로세스 재로딩, merge 후 재로딩을 점검합니다. `--include-vision`을 붙이면 SmolVLM PyTorch/HF image training과 저장·재로딩도 확인합니다.

```bash
# HF overlay가 활성화된 Spark 컨테이너
python scripts/validate_spark.py --profile hf --include-vision
RUN_ML_TESTS=1 python -m pytest --cov=finetune_lab --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

Unsloth profile은 실제 작은 사전학습 모델로 instruction SFT, 지식 CPT/SFT 이어 학습과 저장·재로딩을 확인합니다. vision 옵션은 3B VLM 다운로드를 포함합니다.

```bash
# Unsloth overlay가 활성화된 Spark 컨테이너
python scripts/validate_spark.py --profile unsloth --include-vision
```

검증 script는 Mac/Linux x86에서 GPU 실행을 시작하지 않습니다. 각 subprocess의 명령·종료 코드·로그를 `outputs/spark_validation/<timestamp-profile>/`에 기록하고 실패하면 중단합니다. fixture를 사용하는 짧은 테스트이므로 능력 향상·실제 Beans accuracy를 측정하는 실험이 아닙니다. 그런 비교는 실제 데이터로 별도로 실행합니다.

## Coverage 기준

제공된 AGENTS.md의 목표는 80% 이상입니다. 현재 **전체 coverage 80% 달성은 미확인**이며 gate 명령은 기준 미달 시 실패합니다. CUDA/Unsloth의 mock tests는 제어 흐름 검사에 도움을 주지만 실제 GPU kernel 실행을 대신하지 않습니다. HF와 Unsloth를 한 프로세스에서 섞지 않고 profile별 실행 결과를 기록합니다.

GPU 검증 결과를 받으면 실행 환경 image digest·freeze·logs를 남기고 이 문서를 갱신합니다. 통과하지 않은 항목을 완료로 표시하거나 exception coverage로 감추지 않습니다.
