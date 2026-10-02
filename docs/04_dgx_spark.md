# DGX Spark: ARM64·Blackwell CUDA 환경

이 저장소의 실행 대상은 **DGX Spark 한 대**입니다. ARM64 CPU와 GB10 Blackwell GPU, CPU/GPU가 공유하는 128 GB 통합 메모리를 사용합니다. 이 특성 때문에 x86 GPU 서버의 설치 목록과 고정 VRAM 가정을 그대로 가져오지 않습니다. [NVIDIA 시스템 설명](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html), [포팅 가이드](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)를 참고합니다.

## 1. NGC 컨테이너를 기준으로 삼기

NVIDIA의 [Spark Unsloth playbook](https://build.nvidia.com/playbooks/unsloth/instructions)은 `nvcr.io/nvidia/pytorch:25.11-py3`를 사용합니다. 이 저장소도 해당 tag를 시작점으로 삼고 torch/torchvision/Triton을 NGC의 CUDA build로 보존합니다. 최신 release를 무조건 설치하는 것보다 한 조합에서 작은 학습·저장·재로딩을 먼저 확인합니다.

Spark 호스트에서 Docker와 NVIDIA runtime이 준비되어 있는지 확인합니다. 시스템 설정은 [NVIDIA Container Runtime 안내](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)를 따릅니다.

```bash
nvidia-smi
nvcc --version
docker pull nvcr.io/nvidia/pytorch:25.11-py3
cd ~/finetune
bash scripts/spark_container.sh hf
```

launcher는 폴더를 `~/finetune`에 연결하고 GPU를 노출합니다. datasets/model cache와 outputs도 같은 호스트 폴더에 남습니다. 컨테이너 종료 후에도 데이터와 결과는 보존됩니다. Mac의 `.venv`를 Spark에 복사하지 않습니다.

## 2. PyTorch / HF / VLM profile

컨테이너 내부에서 실행합니다.

```bash
bash scripts/install_spark.sh hf
source .venv-spark-hf/bin/activate
python scripts/doctor.py --require-spark
python scripts/validate_spark.py --profile hf --include-vision
```

설치 스크립트는 `--system-site-packages` 가상환경을 만들고 NGC의 torch/torchvision/Triton 버전을 constraints로 고정합니다. HF 라이브러리는 overlay에서 설치합니다. 충돌 시 기존 CUDA build를 교체하지 않고 설치를 실패시킵니다. `pip check` 오류는 무시하지 말고 어떤 NGC 구성 요소와 충돌했는지 확인합니다.

`doctor --require-spark`는 Linux/aarch64, GB10 capability 12.1, CUDA 가용성과 BF16 matrix forward/backward를 확인합니다. 이름만 CUDA인 wheel이 설치되었다는 사실보다 실제 커널 실행을 점검합니다. CUDA kernel 실패는 작은 모델의 학습 문제와 분리해 해결합니다.

## 3. Unsloth profile

다른 실행 프로세스/overlay에서 준비합니다.

```bash
# Spark 호스트
bash scripts/spark_container.sh unsloth
# 컨테이너 내부
bash scripts/install_spark.sh unsloth
source .venv-spark-unsloth/bin/activate
python scripts/validate_spark.py --profile unsloth --include-vision
```

이 저장소는 다음 호환 범위를 [Unsloth 공식 PyPI metadata](https://pypi.org/project/unsloth/2026.9.14/)와 [Unsloth Zoo metadata](https://pypi.org/project/unsloth-zoo/2026.9.9/)에서 확인하고 고정했습니다.

| 패키지 | 사용 버전 / 출처 |
|---|---|
| torch / torchvision / Triton | NGC 25.11 build를 유지 |
| Transformers | 4.57.6 |
| Datasets | 4.3.0 |
| PEFT | 0.18.1 |
| Unsloth / Zoo | 2026.9.14 / 2026.9.9 |
| TRL | 0.24.0 |
| bitsandbytes | 0.48.2, 공개 aarch64 wheel |

NVIDIA playbook의 예제 설치 목록과 현재 Unsloth metadata가 항상 같은 것은 아닙니다. 지정 Unsloth는 TRL `<=0.24.0`, Datasets `<4.4.0`을 요구하므로 다른 버전을 혼합하지 않습니다. package metadata가 호환된다는 사실만으로 ARM64 커널이 실행된다는 보장은 없으며 **Spark에서 validation 명령을 실제 실행해야 합니다.**

커널 문제가 있으면 [Unsloth의 Spark 안내](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth)와 [공식 Dockerfile_DGX_Spark](https://github.com/unslothai/notebooks/blob/main/Dockerfile_DGX_Spark)를 비교합니다. source build 대안에서는 Triton/xformers와 CUDA architecture `12.1`을 맞춥니다. source build를 이 저장소에서 자동 실행하지 않습니다. 해당 Dockerfile의 구형 TRL 설치 목록을 사용할 때도 이 실습이 사용하는 `max_length`·`processing_class` API와 다시 맞춰야 합니다.

## 4. 작은 모델에서 Spark 실험으로 확장

Post-training 수업은 별도 `post` overlay를 사용합니다. Spark 호스트에서 `bash scripts/spark_container.sh post`, 컨테이너 내부에서 `bash scripts/install_spark.sh post` 후 `source .venv-spark-post/bin/activate`를 실행합니다. `requirements/spark-post.txt`는 HF overlay에 TRL 0.24.0을 추가하며 NGC 런타임 보호 방식은 동일합니다. DPO·Reward/PPO·GRPO의 실제 검증은 [post-training 입구](../04_post_training/README.md)의 각 README를 따릅니다. 기존 `validate_spark.py --profile hf` 결과는 post-training 검증을 포함하지 않습니다.

기본 135M LLM/256M VLM은 원리와 mask/저장을 배우기 위한 선택입니다. 다음으로 같은 계열의 base/instruct를 유지해 모델 크기만 바꿔 비교합니다. [SmolLM2 모델 계열](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)은 1.7B base/instruct도 제공합니다.

```bash
python 01_instruction/huggingface/train.py --model HuggingFaceTB/SmolLM2-1.7B --method lora --device cuda --max-length 1024 --batch-size 1 --gradient-accumulation 8 --max-steps 100 --output-dir outputs/instruction_17b_lora
python 02_knowledge/huggingface/train.py --stage sft --model HuggingFaceTB/SmolLM2-1.7B-Instruct --method lora --device cuda --max-length 1024 --max-steps 100 --output-dir outputs/knowledge_17b_lora
```

그다음 Unsloth의 `--no-load-in-4bit`(LoRA)와 기본 4-bit(QLoRA)를 비교합니다. 기본 3B Qwen VLM도 이어서 실행합니다. 처음부터 70B 이상을 기본값으로 만들지 않습니다. 128 GB 통합 메모리가 있어도 full optimizer state와 activation·OS 메모리까지 고려해야 합니다. 모델 inference 가능 크기와 full fine-tuning 가능 크기는 다릅니다.

## 5. 메모리와 재현성

`nvidia-smi`의 메모리 합계가 `Not Supported`로 보일 수 있습니다. 이는 iGPU/통합 메모리에서 예상되는 표시이며 단독으로 GPU 실패를 뜻하지 않습니다. [NVIDIA known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)를 참고합니다.

```bash
free -h
vmstat 1
```

모델 run의 CUDA allocated/reserved peak와 시스템 available RAM을 함께 기록합니다. swap이 계속 늘면 throughput이 크게 떨어질 수 있으므로 sequence·batch·모델 크기를 줄여 실제 여유를 확인합니다.

NGC tag 외에도 image digest를 저장합니다. overlay freeze는 `outputs/environment/*-freeze.txt`에 기록됩니다.

```bash
# 호스트에서
docker image inspect nvcr.io/nvidia/pytorch:25.11-py3 --format '{{json .RepoDigests}}'
```

훈련 metadata와 data manifest, 실행 명령, 출력 보고서를 함께 보존합니다. 최신 Spark 구성이 실제 GPU에서 검증되었는지는 [검증 상태](07_validation.md)에 명시합니다.
