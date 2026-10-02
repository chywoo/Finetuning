# Graft로 코드 연결 관계 읽기

저장소 작업 규칙은 [AGENTS.md](../AGENTS.md)에 있습니다. 최종 보완 후 기본 `graft build`로 post-training·validation 소스를 포함한 51개 카드를 갱신했습니다. Graft는 Git에서 제외되는 로컬 캐시이므로 다른 작업 공간에서는 재생성합니다. 카드가 없을 때는 실제 소스를 읽으며 함수 위치를 추정하지 않습니다.

사용자가 만든 [Graft 저장소 지도](../graft/INDEX.md)와 파일별 markdown 카드를 실제 구현 탐색에 사용했습니다. 카드는 함수·클래스와 생성 시점의 소스 위치를 요약합니다. 이후 소스가 수정되면 line 정보가 오래될 수 있으므로 현재 파일을 확인합니다. 원본 Graft graph/cache는 임의로 수정하지 않습니다.

| 궁금한 내용 | 먼저 볼 카드 | 실제 구현 |
|---|---|---|
| 데이터 schema·중복·split | [data 카드](../graft/finetune_lab/data.md) | [data.py](../finetune_lab/data.py) |
| HF download·JSONL·이미지 저장 | [prepare 카드](../graft/finetune_lab/prepare_data.md) | [prepare_data.py](../finetune_lab/prepare_data.py) |
| 응답만 loss·EOS·padding | [encoding 카드](../graft/finetune_lab/text_encoding.md) | [text_encoding.py](../finetune_lab/text_encoding.py) |
| PyTorch optimizer loop | [torch 카드](../graft/finetune_lab/torch_text.md) | [torch_text.py](../finetune_lab/torch_text.py) |
| HF Trainer·PEFT | [HF 카드](../graft/finetune_lab/hf_text.md) | [hf_text.py](../finetune_lab/hf_text.py) |
| Unsloth FastLanguageModel·TRL | [Unsloth 카드](../graft/finetune_lab/unsloth_text.md) | [unsloth_text.py](../finetune_lab/unsloth_text.py) |
| 이미지 processor·mask·분류 지표 | [vision 카드](../graft/finetune_lab/vision.md) | [vision.py](../finetune_lab/vision.py) |
| adapter 재로딩·텍스트 평가 | [evaluate 카드](../graft/finetune_lab/evaluate.md) | [evaluate.py](../finetune_lab/evaluate.py) |
| manifest·decoded pixel·CPT/QA split 검사 | [validation 카드](../graft/finetune_lab/validation.md) | [validation.py](../finetune_lab/validation.py), [검사 CLI](../scripts/check_data.py) |
| 선호/산술 데이터·DPO·Reward/PPO·GRPO | [post_data 카드](../graft/finetune_lab/post_data.md) | [post_data.py](../finetune_lab/post_data.py), [post_dpo.py](../finetune_lab/post_dpo.py), [post_rlhf.py](../finetune_lab/post_rlhf.py), [post_grpo.py](../finetune_lab/post_grpo.py) |

```text
각 대상/도구의 train.py
  → 해당 공통 engine
  → data.read_jsonl / split 검사
  → text encode_record 또는 VisionCollator
  → 학습 / validation / 저장
  → 별도 evaluate 프로세스에서 재로딩
```

`rg -n 'encode_record|completion_labels|from_pretrained' graft/`로 관련 카드를 먼저 찾을 수 있습니다. Graft CLI가 있는 환경에서는 `graft ask "SFT response masking은 어디서 하나?"` 또는 `graft callers encode_record`로 연결을 조회합니다. 최신 코드를 반영하려면 사용자 환경에서 `graft build`를 실행합니다. 이 작업은 추가 모델 학습/환경 검증과 독립적입니다.
