# Repository agent guide

## 목적과 실행 환경

**현재 작업 모드: 계획·문서 정리만 진행한다. 사용자가 명시적으로 구현 재개를 지시하기 전에는 실행 코드·테스트 구현을 추가하거나 변경하지 않는다.** 중단 전에 작성된 코드는 현황으로 보존하며, 구현 재개 승인으로 간주하지 않는다. 학습·환경 설치도 실행하지 않는다.

이 저장소는 한국어로 설명하는 단계별 LLM/VLM fine-tuning 및 LLM post-training 실습실이다. 작은 영어 모델·공개 데이터로 원리를 배우고 DGX Spark에서 실제 학습한다.

- 실행 대상: **DGX Spark ARM64 Linux / GB10 CUDA**. Intel Mac의 환경 검사, ML library import, pytest, 학습 실행은 사용자 지시로 중단했다. Mac에서는 소스·문서·데이터의 정적 확인만 한다.
- CUDA 실행, 품질 향상, coverage 통과는 실제 근거가 있을 때만 완료로 기록한다.
- 커리큘럼: [docs/11_curriculum.md](docs/11_curriculum.md). 전체 케이스/체크: [docs/10_practice_checklist.md](docs/10_practice_checklist.md). 시간별 기록: [TASK_LOGS.md](TASK_LOGS.md).
- 처음은 고등학생 수준의 직관·작은 계산·용어 설명으로 시작하고, 이후 대학생 수준의 수식·가정·실험 해석으로 확장한다. 실습마다 선수 단계, 시작 파일, 실행 명령, 결과물, 완료 기준, 다음 단계를 제공한다.

## 저장소 구조와 경계

| 경로 | 역할 |
|---|---|
| `01_instruction/{pytorch,huggingface,unsloth}/` | Base LLM instruction SFT |
| `02_knowledge/{pytorch,huggingface,unsloth}/` | 지식 CPT / QA SFT |
| `03_vision/{pytorch,huggingface,unsloth}/` | Beans VLM 이미지 도메인 적응 |
| `04_post_training/` | HF TRL DPO, reward modeling→PPO, RLVR GRPO, HF domain 과정 |
| `finetune_lab/` | 공통 데이터 계약·encoding·학습·평가·저장 엔진 |
| `data/demo/` | 자체 작성 fixture; 사람 선호/실제 품질 benchmark라고 주장하지 않음 |
| `data/processed/` | Dolly/SciQ/Beans 다운로드 subset과 manifest |
| `requirements/`, `scripts/` | Spark profile 설치·런타임 보호·검증 |
| `docs/`, `tests/` | 수업, 출처, 진행 계획, 데이터/계산/재로딩 검사 |
| `graft/` | 사용자가 만든 코드 탐색 카드 |
| `outputs/` | 모델·adapter·명령·평가 보고서; 결과 덮어쓰기 금지 |

Post-training은 도구 3종을 모두 복제하지 않는다. HF Transformers/PEFT + TRL의 한 경로를 가르친다. TRL 0.24.0 API를 기준으로 확인하며 최신 main의 API를 혼합하지 않는다. 업계 점유율 1위라고 근거 없이 표현하지 않는다. Domain fine-tuning은 기존 CPT/SFT 엔진을 재사용한다. DPO와 reward→PPO는 SFT 이후의 대안 경로이며 PPO 앞에 DPO를 필수로 연결하지 않는다.

## 먼저 SKILL을 읽고 적용하기

사용 가능한 SKILL catalog에서 아래 이름을 찾고 **해당 SKILL.md를 읽은 뒤** 필요한 작업에만 적용한다. 설치 경로는 호스트마다 다르므로 개인 Mac 절대 경로를 코드/실습 명령에 고정하지 않는다. `ecc:` prefix는 plugin 제공 skill 이름이다. 파일이 없으면 catalog와 plugin 경로에서 찾아보고, 제공되지 않으면 그 사실을 기록하고 가능한 범위를 진행한다.

| 작업 | 우선 SKILL |
|---|---|
| 데이터 계약·누수·재현성·평가 계획 | `ecc:mle-workflow` |
| PyTorch loss/mask/optimizer/AMP 수정 | `ecc:pytorch-patterns` |
| Python 구조·오류 처리 | `ecc:python-patterns` |
| tests와 coverage gate | `ecc:python-testing` |
| benchmark·지표·회귀 검증 설계 | `ecc:eval-harness` |
| 코드 설명·탐색 안내 | `ecc:codebase-onboarding` 또는 `ecc:code-tour` |

Skill을 처음 적용할 때 사용자에게 어떤 skill을 쓰는지 알린다. Skill은 사용자 지시를 대체하지 않는다. 예를 들어 skill의 실행 검증 권고를 이유로 Intel Mac 검증을 재개하지 않는다. 이 저장소에는 local Spark 실행만 필요하므로 skill에 cloud paid job 기능이 있어도 자동 제출하지 않는다.

## Graft로 탐색하기

1. [graft/INDEX.md](graft/INDEX.md)와 [코드 지도](docs/09_code_map.md)를 먼저 읽는다.
2. `rg -n '검색할 함수|관심 키워드' graft/`로 카드를 찾는다.
3. 카드에 표시된 실제 소스를 읽어 현재 구현을 확인한다. 줄 번호는 생성 후 변경으로 오래될 수 있다.
4. CLI가 있으면 `graft ask "SFT 응답 mask는 어디서 만드는가?"`, `graft callers encode_record` 등을 활용한다. 도구가 없다면 카드와 `rg`로 진행한다.
5. 새 post-training 파일은 아직 카드가 없을 수 있다. 없는 카드를 꾸며내지 않고 소스를 직접 읽는다. 코드 변경이 마무리되면 가능한 환경에서 `graft build`로 갱신한다. 기존 graph/cache를 수작업으로 편집하지 않는다.

## 작업 순서와 협업

- 주요 작업 단위가 끝나면 변경 내용을 검토하고 Conventional Commit으로 local Git commit을 남긴다. 계획/문서만 작업한 단계도 포함한다. 명령·검증 범위와 미확인 항목을 TASK_LOGS에 시각별로 기록한다.
- Commit은 현재 작성한 결과를 기록하는 동작이다. 구현 중단 상태에서 commit 요청을 구현 재개나 push 승인으로 해석하지 않는다. Cache·가상환경·모델 출력·credentials는 commit에 넣지 않는다.

- 복잡한 변경은 먼저 계획하고 파일 소유권을 나눈다. 이 저장소에서는 전문 에이전트의 계획·구현·리뷰와 독립 작업의 병렬 위임을 허용한다.
- Worker에게 소유 파일, 다른 작업자 공존, 타인의 변경을 되돌리지 않을 것을 명시한다.
- 테스트 소스를 먼저 작성한다. Spark에서 의미 있는 unit/integration/학습→저장→재로딩 검증과 **coverage 80% 이상**을 확인한다. 실행 불가 상태는 미확인으로 기록하며 통과했다고 쓰지 않는다.
- 입력 JSONL, split overlap, model/adapter 구분, output 경로와 유한 loss/reward를 검증한다. 학습에 test를 사용하지 않는다.
- 논리·데이터 변환은 가능한 새 객체로 반환하고 불필요한 mutation을 피한다. PyTorch/Trainer의 필요한 optimizer 상태 갱신은 프레임워크 동작이다.
- 함수는 작게 유지하고 파일은 800줄을 넘기지 않는다. 공통 기능은 기존 모듈을 재사용한다.
- 코드 수정 후 correctness/ML 리뷰를 수행한다. Dataset/model card, 논문, 고정 버전 공식 문서/태그 소스를 근거로 API를 확인한다.
- 모델 다운로드/로컬 결과 작성은 요청 범위에 포함된다. 원격 작업 dispatch, 게시, push, 유료 학습, credential 변경은 별도 명시적 승인이 필요하다. Secret을 코드·로그에 넣지 않는다.

## Spark 검사와 기록

설치·실행은 [DGX Spark 안내](docs/04_dgx_spark.md)를 따른다. NGC의 torch/torchvision/Triton CUDA build를 overlay constraints로 보호한다. HF/Unsloth/post profile을 별도 환경과 프로세스로 실행한다.

```bash
# Spark 컨테이너 내부, HF overlay 활성화 후
python scripts/doctor.py --require-spark
python scripts/validate_spark.py --profile hf --include-vision
RUN_ML_TESTS=1 python -m pytest --cov=finetune_lab --cov-report=term-missing --cov-fail-under=80
```

Post-training CLI와 검증 기준은 각 README를 따른다. 기존 `validate_spark.py`의 HF/Unsloth matrix를 post-training 검증으로 간주하지 않는다.

진행이나 범위가 바뀔 때:
- 체크리스트의 해당 단계와 집계를 갱신한다.
- TASK_LOGS에 실제 `America/Toronto` 시각·case ID·변경·근거·남은 일을 **추가**한다. 기존 이벤트를 덮어쓰거나 시간을 소급 추정하지 않는다.
- docs의 준비 상태와 실행 상태를 함께 맞춘다.
