# 03. 기존 VLM에 이미지 도메인 인식 추가하기

기존 VLM은 이미지 encoder, 이미지 정보를 언어 공간으로 연결하는 connector/projector, 텍스트를 생성하는 decoder를 이미 가지고 있습니다. 이 실습은 콩잎 이미지를 보고 세 클래스 중 하나를 답하게 적응시킵니다.

```text
이미지 + "다음 클래스 중 하나만 답하라"
  → angular_leaf_spot / bean_rust / healthy
```

이는 **기존 이미지 이해 모델의 도메인 적응**입니다. 객체 위치 bounding box를 찾는 detection이나 이미지 입력을 지원하지 않는 LLM에 encoder를 붙이는 구현은 별도의 작업입니다. 이 점을 [이론](../docs/01_theory.md)에서 구분합니다.

| 도구 | 기본 모델 | 업데이트 범위 |
|---|---|---|
| [PyTorch](pytorch/README.md) | SmolVLM-256M-Instruct | 언어 decoder 마지막 블록 일부 |
| [Hugging Face](huggingface/README.md) | SmolVLM-256M-Instruct | 언어 attention LoRA |
| [Unsloth](unsloth/README.md) | Qwen2.5-VL-3B-Instruct | FastVisionModel 언어 LoRA, vision 동결 |

PyTorch/HF는 작은 모델로 구조를 이해하고, Unsloth는 공식 지원 모델로 CUDA 실습을 진행합니다. 모델이 다르므로 시간과 정확도 차이는 모델 크기·학습 범위에도 영향을 받습니다.

```bash
python -m finetune_lab.prepare_data --task vision --source hf --train-samples 96 --eval-samples 24
python 03_vision/huggingface/train.py --dry-run
python 03_vision/huggingface/train.py --evaluate --model HuggingFaceTB/SmolVLM-256M-Instruct --max-eval-samples 24
python 03_vision/huggingface/train.py --max-steps 20
```

각 README에서 baseline/학습 후 평가의 출력 파일을 다르게 지정하는 법을 확인합니다. 정확도뿐 아니라 macro F1·confusion matrix·잘못 생성한 클래스명을 봅니다. 데이터가 세 클래스로 균형을 이루는지도 manifest에서 확인합니다. 같은 이미지의 복제본이 train/test로 나뉘면 실제 인식 능력을 과대평가하므로 원본 이미지 기준 분할을 유지합니다.
