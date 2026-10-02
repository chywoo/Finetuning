# 실습실 구축 완료와 현재 실행 범위

최종 빠른 검증: 2026-10-02 11:35 EDT (America/Toronto). 교육 자료와 실습 구현 보완을 마쳤다. 모델 학습과 오래 걸리는 작업은 사용자 조건에 따라 실행하지 않았다.

## 최초 요청과 제공한 과정

| 학습 목표 | 시작 수업 | 제공 형태 |
|---|---|---|
| Base LLM에 instruction 추가 | [Instruction](../01_instruction/README.md) | PyTorch full/LoRA, HF full/LoRA/QLoRA, Unsloth LoRA/QLoRA |
| 기존 LLM에 지식 추가 | [Knowledge](../02_knowledge/README.md) | CPT·QA SFT·CPT→SFT 연결, 세 도구의 대응 실습 |
| VLM 이미지 도메인 적응 | [Vision](../03_vision/README.md) | Beans, PyTorch decoder 일부/HF·Unsloth 언어 LoRA |
| Domain fine-tuning | [Domain](../04_post_training/domain_huggingface/README.md) | 기존 HF CPT/QA 엔진 재사용과 세 갈래 비교 |
| DPO / RLHF | [DPO](../04_post_training/dpo_huggingface/README.md), [Reward→PPO](../04_post_training/rlhf_ppo_huggingface/README.md) | HF/PEFT + TRL 0.24.0, 동일 SFT에서 대안 경로 비교 |
| RLVR | [GRPO](../04_post_training/rlvr_grpo_huggingface/README.md) | 산술 SFT 선수 과정, strict verifier와 GRPO |

각 방법별 `train.py`와 한국어 README에 이론·데이터·선수 조건·명령·결과물·완료 기준·다음 수업을 제공한다. [입문 수업](12_foundations.md)은 고등학생 수준의 비유·손계산부터 시작하고, 이후 loss·LoRA·KL·PPO·GRPO 수식과 실험 해석으로 확장한다. [커리큘럼](11_curriculum.md)이 주 경로이며 [체크리스트](10_practice_checklist.md)는 선택 비교까지 관리한다. Post-training의 도구 선택 근거는 공식 trainer와 공개 개발 사례이며 도구 점유율 순위를 주장하지 않는다.

## 실제 확인한 근거

| 검사 | 결과와 근거 |
|---|---|
| 전체 빠른 회귀 suite | **172 passed, 2 학습 integration skipped, 37 subtests passed**, 약 5.63초 |
| 전체 branch 포함 coverage | **81.56%, 80% gate 통과**; [coverage JSON](../outputs/readiness/coverage.json), [JUnit](../outputs/readiness/tests.xml) |
| CLI 데이터 진입 | I/C/S/K/V/P 총 **34/34 dry-run 통과**; 학습 선수 checkpoint/kernel 성공을 뜻하지 않음 |
| 복사된 데이터 | manifest 10개·JSONL 36개·이미지 168개; 건수/해시/schema/split/RGB pixel 및 원본→SFT 대응 통과; [보고서](../outputs/readiness/data-audit.json) |
| HF/post 환경 | 고정 조합 import, TRL 실제 생성자 signature, 작은 CUDA 연산과 BF16 지원 확인; [보고서](../outputs/readiness/post-environment.json) |
| 학습 없는 모델 경로 | random tiny GPT-2 CUDA forward, checkpoint 저장 후 별도 subprocess 평가 종료 코드 0/2개 예시; [보고서](../outputs/readiness/offline-reload.json) |
| LoRA 경로 | 학습하지 않은 실제 adapter 저장·재로딩·merge 검사 통과 |
| Trainer/post-training 연결 | mock 경계로 train/validation 분리, reference/reward freeze, value 저장, finite metrics·출력 보호 확인 |
| 정적 검사/탐색 | Python 51개 AST·800줄 제한, Markdown 로컬 링크·shell 문법 정상; 기본 Graft 51개 카드 갱신 |

보고서는 프로젝트 루트의 `outputs/readiness/`에 있으며 Git 제외 대상이다. 다른 작업 공간에서는 재실행하여 생성한다. 기록에는 내부 장비 식별 정보·절대 내부 경로·비밀 값을 넣지 않는다. 프레임워크의 model/adapter 재로딩에 필수인 참조 정보와 사용자용 보고서는 구분한다.

## 설치와 짧은 검증 재현

Python 3.11/3.12와 장치에 맞는 PyTorch를 먼저 설치한다. 일반 고정 실습 조합은 `requirements/lab-hf.txt`, post 과정은 `requirements/lab-post.txt`를 사용한다. 사용자 관리 중인 `pyproject.toml`의 최신 패키지를 기존 실습 API의 검증 조합으로 자동 간주하지 않는다. 기존 환경을 보존하고 별도 `.venv-lab`에서 검사했다.

아래 명령은 설치가 끝난 프로젝트 루트에서 실행하며 실제 LLM 학습을 시작하지 않는다. Unit suite에는 작은 scalar gradient 계산과 mock Trainer가 포함된다.

```bash
source .venv-lab/bin/activate
python scripts/check_data.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/doctor.py --profile post --require-cuda
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_ML_TESTS=0 RUN_POST_TRAINING_TESTS=0 RUN_SPARK_POST_TESTS=0 OMP_NUM_THREADS=1 timeout 90 python -m pytest --cov=finetune_lab --cov-report=term-missing --cov-fail-under=80
```

## 실행하지 않은 항목과 환경 제약

- 실제 공개 모델 학습, baseline/학습 후 품질 비교와 속도·메모리 실험은 실행하지 않았다. 실습 학습 완료는 **0/34**, 추가 과제는 **0/11**이다. 이는 구축 미완성과 다른 상태다.
- Unsloth 고정 버전 2026.9.14는 torch `<2.13`을 요구한다. 기존 torch 2.14.1을 보호한 의존성 dry-run은 실패했다. `requirements/lab-unsloth.txt`를 호환되는 별도 CUDA 환경에 설치해야 한다. Native Unsloth/QLoRA/VLM kernel, Spark 전용 matrix와 실제 post-training rollout은 미확인이다.
- 기존 `nvidia-cusparselt-cu13==0.8.1` 공급 wheel tag 때문에 `uv pip check`/`pip check`가 플랫폼 호환 경고를 냈다. 별도 환경의 같은 버전 재설치로도 해소되지 않았다. 라이브러리 load와 dense CUDA forward는 성공했지만 sparse kernel까지 검증했다는 뜻은 아니다. 태그를 수작업으로 고치거나 검사 결과를 통과로 바꾸지 않았다.
- 기존 PyTorch·Transformers 환경과 사용자 변경은 보존했다. 원격 dispatch·게시·push·유료 작업·credential 변경은 하지 않았다.

학습을 실제로 수행할 때는 [빠른 시작](00_quickstart.md)의 학습 절과 개별 README의 선수 checkpoint·새 출력 경로·baseline 기준을 따른다. Validation으로 설정을 고르고 test는 최종 비교에 사용한다. 모델과 데이터의 공개성·라이선스·synthetic fixture의 한계는 [데이터](02_datasets.md)와 [평가](03_evaluation.md)를 참조한다.
