# DCC 2026 대학부 예선 — eigensinn

데이터+AI 크리에이터 캠프(DCC) 2026 대학부 예선 **「전화대화 음성 및 전사 데이터를 이용한 응급상황 인식」**을 위한 팀 저장소입니다.

이 문서는 저장소의 현재 상태, Mission별 진행 상황, 주요 경로와 작업 시작 방법을 빠르게 파악하기 위한 안내서입니다. 실험 수치의 원본은 [`docs/experiment_log.md`](./docs/experiment_log.md), 모델·방법론의 채택·기각·보류 근거는 [`docs/decisions.md`](./docs/decisions.md)에서 관리합니다.

## Mission 구성

| Mission | 문제 | 입력 | 평가지표 |
|---|---|---|---|
| **M1** | 신고자 성별 분류 | 음성 | Accuracy |
| **M2** | 신고자 / 119 상황실 구분 | 발화 구간 음성 | Accuracy |
| **M3** | 증상 다중 레이블 분류 | 전사 텍스트 | Macro F1 |

M1·M2는 음성 처리 구조를 가능한 범위에서 공유하고, M3는 텍스트 파이프라인으로 진행합니다.

## 현재 진행 상황

아래 내용은 **2026-10-12 기준 `main`에 병합된 결과**입니다. 제출물 동결은 **2026-10-14 09:00 KST**입니다. 수치의 원본은 [`docs/experiment_log.md`](./docs/experiment_log.md)이며, 공식 Validation은 모델 선택이 아니라 채점에만 사용합니다.

| Mission | 제출 모델 | 공식 Validation | 제출 폴더 |
|---|---|---|---|
| **M1** | 52D 음향 특징 + WavLM x-vector(다중 창 평균) 결합 로지스틱 회귀. Training 29,200통화 전체로 재학습(refit) | Accuracy **98.10%** (이전 52D LR 95.11%) | [`submission/mission1/`](./submission/mission1/) |
| **M2** | log-mel 스펙트로그램 + `SpeakerCNNLarge` 3 seed 소프트보팅 앙상블. 사전학습 모델·외부 데이터 없음. 발화 하나의 `startAt`~`endAt` 구간만 사용 | Accuracy **91.49%** | [`submission/mission2/`](./submission/mission2/) |
| **M3** | RoBERTa-large(`klue/roberta-large`) 3 seed + TF-IDF 로지스틱 회귀의 고정 가중치 앙상블(0.55 : 0.45), threshold 0.5 고정·후처리 없음 | Macro F1 **0.6534** (P0 TF-IDF+LR 0.5689) | [`submission/mission3/`](./submission/mission3/) |

- 각 제출 폴더의 `README.md`에 실행 명령, 학습 재현 방법, 제출 전 검증이 있습니다.
- 모델 가중치(M1 `m1_model.pt` 약 386MB, M3 `m3_ensemble_full.pt` 약 4.05GB, M2 앙상블 `.pt`)는 git에 포함하지 않고 패키징 때 폴더에 넣습니다.
- M2 가중치는 팀원 보유본을 전달받아 SHA256을 대조한 뒤 넣습니다.

### Mission별 요약

- **M1**: baseline(52D LR 95.11%) → WavLM x-vector 결합(97.47%) → 다중 창 평균(98.13%) → 전체 학습 재학습(98.10%, 사실상 동등)으로 진행했습니다. ECAPA 점수 결합은 사전 고정 기준에 미달해 채택하지 않았습니다.
- **M2**: L1(LightGBM) 86.44%, L2 CNN 앙상블 91.49%(원본부터 재학습)가 현재 제출 모델입니다. 주최 측 답변(2026-10-05)에 따라 **분류 대상 발화 구간 정보만** 사용하며 같은 통화의 다른 발화 정보는 쓰지 않습니다. 사전학습 SSL 임베딩·미세조정 비교 실험(M2-SSL 계열)은 같은 규정 안에서 진행 중입니다.
- **M3**: 분할·전처리 기준은 JSON 샘플 단위 80:20 multilabel stratified split, seed=42, threshold=0.5입니다. KcBERT(0.6246) → RoBERTa-large(0.6433) → 3 seed + TF-IDF 앙상블(0.6534)로 개선했습니다. threshold 튜닝과 빈 예측 보정 후처리는 규정상 사용하지 않으며, 경계 이동·재정규화·규칙 기반 부정 마킹은 측정상 이득이 거의 없어 중단했습니다.
- **공통**: 데이터 감사, WAV↔JSON 매칭 검증, 통합 EDA(`reports/eda/eda_report.md`), Mission별 실험·결정 기록 체계를 갖추고 있습니다.

### 진행 중 PR

진행 중인 실험과 문서 PR은 병합 전까지 확정 현황에 포함하지 않습니다. 최신 목록은 [GitHub Pull Requests](https://github.com/eigensinn-dcc2026/prelim/pulls)에서 확인합니다.

## 팀 역할

| 팀원 | 담당 |
|---|---|
| **김장수** | Mission 1 |
| **김광희** | Mission 1·2 |
| **지연우** | Mission 2 |
| **이수빈** | Mission 3 + 전체 Validation·오류 분석 취합 |
| **조지은** | Mission 3 |

각 Mission 담당자는 baseline 작성, 실행, 결과 확인, 오류 분석과 다음 실험 결정을 함께 수행합니다.

## 저장소 구조

```text
dcc2026-prelim/
├── .github/               # PR 템플릿, CI 및 병합 판정 설정
├── submission/            # 제출 폴더(mission1·2·3): 추론 코드, 학습 재현 노트북, 폴더별 README
├── notebooks/             # Colab/Jupyter 실험 노트북
├── src/dcc_data/          # 데이터 로딩·검증·Mission별 공통 처리 모듈
├── scripts/               # 데이터 감사, 학습, 평가, 추론 실행 스크립트
├── tests/                 # 공통 모듈과 파이프라인 테스트
├── reports/               # 데이터 감사, EDA, Mission별 결과와 오류 분석
├── ppt/                   # 결과 정리 발표 자료
├── docs/
│   ├── experiment_log.md  # Mission별 실험 결과 인덱스
│   ├── mission2_experiment_log.md  # Mission 2 상세 실험 기록
│   └── decisions.md       # 채택·기각·보류 결정과 근거
├── data/                  # 대회 원본·중간 데이터 (Git 추적 제외)
├── checkpoints/           # 모델 가중치 (Git 추적 제외)
├── requirements.txt       # Python 의존성
├── WORKFLOW.md            # Git/GitHub 협업 규칙
└── README.md
```

- `src/`는 여러 실행 경로에서 재사용하는 코드입니다.
- `scripts/`는 데이터 생성, 학습, 평가와 추론을 직접 실행하는 진입점입니다.
- `reports/`에는 근거 자료를, `docs/`에는 여러 결과를 찾아보기 위한 인덱스와 팀 결정을 기록합니다.
- `data/`, `checkpoints/`, `outputs/`는 `.gitignore` 대상이며 GitHub clone에 포함되지 않습니다.

## 시작하기

### 저장소와 환경 준비

```bash
git clone https://github.com/eigensinn-dcc2026/prelim.git
cd prelim
python -m pip install -r requirements.txt
python -m pytest
```

저장소가 private이면 clone 전에 GitHub 접근 권한과 인증이 필요합니다. 대회 데이터와 모델 체크포인트는 저장소 밖에서 별도로 준비해야 합니다.

현재 의존성에는 `numpy`, `pandas`, `pyarrow`, `librosa`, `scikit-learn`, `lightgbm`, `iterative-stratification`, `tqdm`, `pytest`, `matplotlib`, `torch`, `transformers`, `opensmile`(버전 고정)이 포함됩니다. 전체 목록과 버전은 `requirements.txt`를 기준으로 합니다. 새 외부 패키지를 코드에 도입하면 같은 PR에서 `requirements.txt`도 함께 갱신합니다.

### 데이터와 실행 스크립트 확인

각 스크립트의 최신 인자와 기본 출력 경로는 `--help`로 확인합니다.

```bash
python scripts/build_manifest.py --help
python scripts/build_mission1_dataset.py --help
python scripts/train_mission1_baseline.py --help
python scripts/build_mission2_dataset.py --help
python scripts/train_mission2_baseline.py --help
python scripts/mission2_inference.py --help
python scripts/mission3_baseline.py --help
```

제출용 추론은 각 `submission/missionN/inference.py`가 진입점이며, 실행 명령은 폴더별 README를 따릅니다.

실데이터 파이프라인의 공통 순서는 다음과 같습니다.

```text
원본 데이터 준비
  → WAV·JSON manifest 생성
  → Mission별 dataset/feature 생성
  → 학습 및 평가
  → 구조화된 결과 로그와 보고서 기록
```

실행 전에는 해당 스크립트가 요구하는 데이터 split과 출력 경로를 확인합니다. Validation 데이터는 모델 학습이나 scaler 통계 계산에 사용하지 않습니다.

### 로컬 데이터 경로

공식 Drive에서 받은 데이터는 로컬에 보관해 실행할 수 있습니다. 데이터 출처 제한과 실행 시 저장 위치는 별개입니다. 기본 원본 위치는 저장소의 `data/raw_original`이며 다음 구조를 사용합니다.

```text
data/raw_original/
├── Training/1.원천데이터/TS_서울_구급/       # WAV
├── Training/2.라벨링데이터/TL_서울_구급/     # JSON
├── Validation/1.원천데이터/VS_서울_구급/     # WAV
└── Validation/2.라벨링데이터/VL_서울_구급/   # JSON
```

`eda_dcc_dataset.py`, `analyze_mission1_validation_errors.py`, `verify_timestamps.py`의 원본 경로 우선순위는 `--data-root` → `DCC_DATA_ROOT` → 저장소의 `data/raw_original`입니다. PowerShell에서 환경변수를 설정할 수 있습니다.

```powershell
$env:DCC_DATA_ROOT = 'D:\공식 데이터\raw_original'
```

세 스크립트는 필요한 manifest가 없으면 중단합니다. 원본 폴더를 확인한 뒤 저장소 루트에서 다음 명령으로 manifest를 별도로 준비하세요. 기존 manifest는 자동으로 덮어쓰지 않습니다.

```bash
python scripts/build_manifest.py --data-root data/raw_original --output data/manifests/dcc_manifest_v1.csv
python scripts/verify_timestamps.py --data-root data/raw_original --manifest data/manifests/dcc_manifest_v1.csv --output reports/eda/phase2a_check.json
```

외부 원본 폴더를 사용하는 경우 위 명령의 `--data-root`도 같은 폴더로 지정하세요. EDA와 Mission 1 오류 분석에는 기존 feature·checkpoint·평가 로그 등 각 스크립트의 추가 입력도 필요합니다. 일부 Mission 2 Notebook은 `tmp/mission2_eda`의 manifest와 `data/processed`의 지정된 캐시를 요구하며, 원본 폴더만 준비했다고 실행 준비가 완료되지는 않습니다. 캐시와 provenance 검증 실패를 우회하거나 캐시 metadata를 수정하지 마세요.

`notebooks/mission2_local.ipynb`은 저장소 안에서 실행하며 `DATA_MODE="existing"`이 기본입니다. `DCC_DATA_ROOT`를 지원하고, ZIP 준비는 `DATA_MODE="zip"`과 별도 `ZIP_DIR`·`EXTRACT_DIR`를 지정한 경우에만 수행합니다. ZIP 모드의 기존 Validation 50통화 표본 조건은 유지됩니다. `reports/M3_baseline.ipynb`도 로컬을 기본으로 지원하며 Colab에서는 기존 Drive 경로를 사용합니다. 두 Notebook의 CLI Python은 현재 커널의 `sys.executable`을 사용하고 필요하면 `DCC_PYTHON`에 실행 파일의 전체 경로를 지정합니다. 설정 셀은 추출·학습·추론을 실행하지 않으며 이후 실행 셀은 작업을 수행하므로 필요한 단계만 실행하세요.

`notebooks/mission2_egemaps_colab_resume.ipynb`은 수동 검증 절차를 가진 Colab 전용 재개 도구로 유지합니다. `scripts/audit_dcc_dataset.py`는 원격 Drive와 로컬을 비교하는 도구이므로 원격 접근이 필요합니다.

### Colab 사용

Colab에서도 최신 `main`을 clone한 뒤 의존성을 설치합니다. 특정 과거 작업 브랜치를 checkout할 필요는 없습니다.

```python
!git clone https://github.com/eigensinn-dcc2026/prelim.git /content/prelim
%cd /content/prelim
!python -m pip install -q -r requirements.txt
```

private 저장소 인증과 Google Drive 마운트는 각자 환경에서 별도로 설정합니다. 장시간 생성하는 feature와 checkpoint는 런타임 종료에 대비해 Drive 등 영속 경로에 저장합니다.

## 실험 기록 원칙

- 최종 수치는 JSON 로그, 지표 파일 또는 보고서를 근거로 남깁니다.
- [`docs/experiment_log.md`](./docs/experiment_log.md)는 실험 ID, 데이터 규모, 분할 단위, seed, 결과와 근거 파일을 찾는 인덱스입니다.
- [`docs/decisions.md`](./docs/decisions.md)는 결과를 채택·기각·보류한 이유와 후속 조치를 기록합니다.
- 미병합 PR의 결과나 재현 조건이 확인되지 않은 수치는 최종 성능으로 표현하지 않습니다.

## Git/GitHub 협업

작업은 항상 최신 `main`에서 새 브랜치를 만들어 시작합니다.

```bash
git switch main
git pull --ff-only origin main
git switch -c 이름/작업명
```

브랜치, 커밋, push, PR, 충돌 처리의 전체 규칙은 [`WORKFLOW.md`](./WORKFLOW.md)를 따릅니다. 특히 `main`에 직접 push하지 않고, 같은 `.ipynb` 파일을 여러 명이 동시에 수정하지 않습니다.

`main` 대상 PR에는 자동 병합 판정이 표시됩니다.

| 판정(라벨) | 의미 |
|---|---|
| `병합-가능` | 정책 검사, 자동 테스트, 최신 `main` 반영 조건 충족 |
| `병합-대기` | 자동 검사가 진행 중이거나 최신 `main` 반영을 기다리는 중 |
| `수정-필요` | 충돌, 정책 위반 또는 테스트 실패 해결 필요 |

자동 판정은 merge를 대신하지 않습니다. 팀원 리뷰는 필수가 아니며, `병합-가능`을 확인한 뒤 PR 작성자가 직접 최종 병합합니다. 현재 private 저장소 요금제에서는 판정이 merge 버튼을 강제로 막지 않으므로, `병합-가능`이 아니면 병합하지 않는 것을 팀 규칙으로 적용합니다. 자세한 내용은 [`WORKFLOW.md`](./WORKFLOW.md) 6장을 따릅니다.

## 개발 방향

복잡한 모델부터 시작하기보다 **규칙을 지키는 baseline → Validation 확인 → 오류 분석 → 다음 실험 결정** 순서로 진행합니다.

```text
M1 ─┐
M2 ─┼─ Mission별 검증과 오류 분석 → 실험 기록 → 채택·기각·보류 결정
M3 ─┘
```

README는 온보딩과 현재 상태 요약을 담당합니다. 상세 수치와 의사결정은 `docs/`와 `reports/`의 근거 파일에서 확인합니다.
