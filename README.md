# DCC 2026 대학부 예선 — eigensinn

데이터+AI 크리에이터 캠프(DCC) 2026 대학부 예선 **「전화대화 음성 및 전사 데이터를 이용한 응급상황 인식」**을 위한 팀 저장소입니다.

이 README는 팀원이 저장소에 들어왔을 때 **현재 진행상황, 담당 Mission, 폴더 구조, 작업 시작 방법**을 빠르게 이해하는 것을 목표로 합니다.

## 1. Mission 구성

| Mission | 문제 | 입력 | 평가지표 |
|---|---|---|---|
| **M1** | 신고자 성별 분류 | 음성 | Accuracy |
| **M2** | 신고자 / 119 상황실 구분 | 발화 구간 음성 | Accuracy |
| **M3** | 증상 다중 레이블 분류 | 전사 텍스트 | Macro F1 |

세 Mission은 **병렬로 진행**합니다. M1·M2는 음성 처리 구조를 가능한 범위에서 공유하고, M3는 텍스트 파이프라인으로 독립 진행합니다.

## 2. 현재 진행상황

- ✅ 데이터 전수 감사 및 기본 구조 검증 완료
- ✅ WAV ↔ JSON 매칭 구조 확인 완료
- ✅ Mission 2·3 데이터 처리 기본 모듈 구현
- ✅ 데이터 처리 코드에 대한 기본 테스트 작성
- 🔵 **EDA PR #2 리뷰 중**
  - WAV 32,840개 기본 형식 확인: 모두 mono, 8,000 Hz
  - JSON `startAt`·`endAt`과 실제 WAV 시간정보 정합성 1,050건 확인, 문제 0건
- ➡️ **M1·M2·M3 baseline 병렬 개발 단계**

> 위 EDA 결과와 관련 코드는 아직 `main`에 병합되지 않았으며, [PR #2](https://github.com/eigensinn-dcc2026/prelim/pull/2)에서 리뷰 중입니다.

### 현재 단기 목표

1. M1 baseline + Validation Accuracy 확보
2. M2 baseline + Validation Accuracy 확보
3. M3 baseline + Validation Macro F1 확보
4. Mission별 대표 오류 사례 수집
5. 9/5 1차 멘토링에서 확인할 질문 정리

## 3. 팀 역할

| 팀원 | 담당 |
|---|---|
| **김장수** | Mission 1 |
| **김광희** | Mission 1·2 |
| **지연우** | Mission 2 |
| **이수빈** | Mission 3 + 전체 Validation·오류 분석 취합 |
| **조지은** | Mission 3 |

각 Mission 담당자는 모델 작성과 실험을 완전히 분리하지 않고, **baseline 작성 → 실행 → 결과 확인 → 다음 실험 결정**까지 함께 수행합니다.

## 4. 저장소 구조

```text
dcc2026-prelim/
├── notebooks/              # Colab 실험 노트북 위치
├── src/
│   └── dcc_data/           # 데이터 로딩·검증·공통 처리 모듈
├── scripts/                # 직접 실행하는 데이터 감사·생성 스크립트
├── tests/                  # 공통 코드 테스트
├── reports/                # 데이터 감사·EDA 결과 보고서
├── docs/
│   ├── decisions.md        # 모델·방법론 선택과 보류·기각 이유
│   └── experiment_log.md   # Mission별 실험 결과 요약과 원본 파일 인덱스
├── data/                   # 대회 원본/로컬 데이터 (Git 추적 제외)
├── checkpoints/            # 모델 가중치 (Git 추적 제외)
├── requirements.txt        # 프로젝트에 필요한 Python 외부 패키지 목록
├── WORKFLOW.md             # Git/GitHub 협업 규칙
├── .gitignore
└── README.md
```

`data/`와 `checkpoints/`는 로컬에서는 사용할 수 있지만 `.gitignore`에 포함되어 있어 GitHub에는 올라가지 않습니다.

### 폴더를 쉽게 구분하면

- **`notebooks/`**: 사람이 직접 열어 실험하는 Colab/Jupyter 노트북
- **`src/`**: 여러 코드에서 반복해서 가져다 쓰는 **부품**
- **`scripts/`**: `src/`의 부품 등을 이용해 실제 작업을 수행하는 **실행 프로그램**
- **`tests/`**: 공통 코드가 예상대로 동작하는지 확인하는 테스트
- **`reports/`**: 데이터 감사, EDA, 실험 결과 등을 사람이 읽을 수 있게 정리하는 곳
- **`docs/`**: 실험 결과 요약과 팀의 채택·기각·보류 결정을 추적하는 곳

실험 현황은 **[docs/experiment_log.md](./docs/experiment_log.md)**에서 확인하고, 모델·방법론을
선택하거나 보류한 이유는 **[docs/decisions.md](./docs/decisions.md)**에서 확인합니다.

즉,

```text
src/     = 재사용할 부품
scripts/ = 부품을 사용해 실제 작업을 실행하는 프로그램
```

이라고 생각하면 됩니다.

## 5. 주요 코드 설명

### `src/dcc_data/`

현재 `main`에는 다음 공통 모듈이 있습니다.

| 파일 | 역할 |
|---|---|
| `__init__.py` | `dcc_data`를 Python 패키지로 사용할 수 있게 하는 파일 |
| `audio.py` | WAV 읽기, 시간 구간 처리 등 음성 공통 기능 |
| `guards.py` | 잘못된 데이터 구조나 허용되지 않은 입력 사용을 막기 위한 검증 기능 |
| `manifest.py` | WAV·JSON 파일 목록과 매칭 정보를 관리 |
| `mission2.py` | Mission 2용 데이터 처리 기능 |
| `mission2_features.py` | Mission 2 baseline(L1)용 발화 단위 인덱스·53차원 음향 feature 추출 |
| `mission2_train.py` | Mission 2 baseline(L1) 학습(GroupShuffleSplit)·모델 저장/로드 |
| `mission3.py` | Mission 3용 데이터 처리 기능 |

현재 `mission1.py`는 없습니다. Mission 1은 M2의 음성 처리 구조를 가능한 범위에서 재사용하면서 baseline 단계에서 필요한 전용 코드를 추가합니다.

### `scripts/`

현재 `main`에는 다음 실행 스크립트가 있습니다.

| 파일 | 역할 |
|---|---|
| `audit_dcc_dataset.py` | 전체 데이터 구조·파일·JSON/WAV 무결성 감사 |
| `build_manifest.py` | WAV와 JSON을 연결한 manifest 생성 |
| `build_mission2_dataset.py` | Mission 2 baseline(L1): Seoul/Training 통화 샘플링 + 발화 인덱스 + feature 캐시 생성 |
| `train_mission2_baseline.py` | Mission 2 baseline(L1) 학습 및 모델·실험 로그 저장 |
| `mission2_inference.py` | Mission 2 추론 (대회 CLI 규격: `--audio_dir --label_dir --ckpt_path --output`) |

EDA 관련 추가 스크립트는 현재 PR #2에서 리뷰 중입니다.

### `tests/`

현재 데이터 처리와 Mission 2·3 기본 기능을 검사하는 테스트가 있습니다.

```text
tests/
├── conftest.py
├── test_manifest_and_guards.py
├── test_mission2.py
└── test_mission3.py
```

공통 모듈을 수정했다면 기존 기능을 깨뜨리지 않았는지 테스트를 확인해야 합니다.

### `reports/`

`main`에는 현재 데이터 감사 결과가 `reports/data_audit/` 아래에 정리되어 있습니다.

EDA 보고 파일은 PR #2 병합 후 `main`에 반영될 예정입니다.

### `notebooks/`

Mission별 Colab 실험 노트북을 둘 위치입니다. 현재 `main`에는 아직 실제 baseline 노트북이 들어 있지 않습니다.

새 Mission 폴더나 노트북 구조가 필요할 경우, 팀에서 구조를 정한 뒤 추가합니다.

## 6. 데이터와 Git 추적 정책

대회 원본 데이터와 모델 가중치는 GitHub에 올리지 않습니다.

`.gitignore`에서 다음 항목을 제외하고 있습니다.

```text
data/
checkpoints/
```

따라서 GitHub를 새로 clone해도 데이터와 모델 가중치는 자동으로 내려오지 않습니다.

- **대회 데이터:** 공식 배포본을 각자 지정된 위치에 준비
- **모델 체크포인트:** 로컬 또는 별도 허용 저장소에서 관리
- **코드·설정·작은 보고서:** GitHub에서 공유

## 7. 작업 시작 방법

### 처음 저장소를 받는 경우

```bash
git clone https://github.com/eigensinn-dcc2026/prelim.git
cd prelim
```

> 저장소가 private이므로 GitHub 접근 권한과 인증이 필요합니다.

### 작업을 시작하기 전

먼저 최신 `main`을 받습니다.

```bash
git switch main
git pull --ff-only origin main
```

그다음 자기 작업 브랜치를 만듭니다.

```bash
git switch -c 이름/작업명
```

예:

```bash
git switch -c jangsoo/m1-baseline
```

세부적인 브랜치, commit, push, PR 규칙은 **[WORKFLOW.md](./WORKFLOW.md)**를 기준으로 합니다.

## 8. Colab에서 사용할 때

Colab에서는 저장소를 가져온 뒤 저장소 폴더로 이동해 작업합니다.

```bash
!git clone https://github.com/eigensinn-dcc2026/prelim.git
%cd prelim
```

private 저장소이므로 실제 clone 과정에서는 GitHub 인증이 추가로 필요할 수 있습니다.

`data/`는 GitHub에 포함되지 않기 때문에 별도로 준비해야 합니다.

Google Drive를 사용하는 경우 예시는 다음과 같습니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 9. `requirements.txt` 관리

`requirements.txt`는 **프로젝트 코드를 실행하는 데 필요한 외부 Python 패키지 목록**입니다.

현재 파일은 존재하지만 아직 비어 있습니다. baseline 개발을 시작하면서 실제 사용 패키지를 누적합니다.

관리 원칙은 다음과 같습니다.

- 각 Mission 담당자는 자기 코드에서 새 외부 패키지를 사용하면 기록합니다.
- 예: `numpy`, `pandas`, `scikit-learn`, `librosa`, `torch`
- `os`, `json`, `pathlib` 같은 Python 기본 라이브러리는 적지 않습니다.
- baseline 초기에는 필요한 패키지명을 중심으로 관리합니다.
- 코드가 안정된 뒤 필요한 패키지 버전을 고정합니다.
- 최종 통합·정리는 김장수가 담당합니다.

최종적으로 새 Colab 환경에서 다음 명령만으로 필요한 패키지를 설치하고, M1·M2·M3 코드가 정상 실행되는지 확인하는 것이 목표입니다.

```bash
pip install -r requirements.txt
```

## 10. Git 협업 규칙

README에는 Git 사용법을 전부 중복해서 적지 않습니다.

브랜치 생성, commit, push, Pull Request, 충돌 방지 규칙은 **[WORKFLOW.md](./WORKFLOW.md)**를 확인합니다.

특히 `.ipynb` 파일은 충돌 해결이 어렵기 때문에 **같은 노트북을 여러 사람이 동시에 수정하지 않는 것**을 원칙으로 합니다.

## 11. 현재 개발 방향

현재는 복잡한 모델보다 **규칙을 지키는 빠른 baseline → Validation 결과 확인 → 오류 분석 → 점진적 개선** 순서로 진행합니다.

```text
M1 baseline ─┐
M2 baseline ─┼─ 병렬 진행 → Validation / 오류 분석 → 다음 실험
M3 baseline ─┘
```

README는 저장소 구조와 현재 개발 상태를 설명하는 문서입니다. 세부 일정, 멘토링 준비, 팀 운영 현황은 팀 Notion에서 관리합니다.

## 12. Mission 2 baseline (L1) 실행 방법

L1 = 음향 통계 feature(53차원: duration + MFCC + RMS + ZCR + spectral 계열) + GBM.
`startAt` 절대값·발화 순번 등 순서를 복원할 수 있는 값은 feature로 사용하지 않는다 (대회 규정).
Train은 Seoul/Training 데이터만 사용하며, Validation 폴더는 이 파이프라인에 들어오지 않는다
(`dcc_data.guards.ensure_training_rows`로 강제).

Colab 셀 순서 (실제 검증된 버전 -- 전체 실행 가능한 정리본은 저장소 밖
`M2_Colab_실행스크립트.py` 참고):

```python
# 1) 저장소 clone (private repo이므로 GitHub PAT 필요:
#    https://github.com/settings/tokens/new -> scope "repo")
from getpass import getpass
token = getpass("GitHub PAT: ")
!git clone https://{token}@github.com/eigensinn-dcc2026/prelim.git /content/prelim

# 2) 이 baseline 브랜치로 전환 + 패키지 설치
%cd /content/prelim
!git checkout gwanghee/mission2-baseline
!pip install -q -r requirements.txt

# 3) Google Drive 마운트 + 데이터/결과 경로 지정
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["DATA_ROOT"] = "/content/drive/MyDrive/데이터 크리에이터 캠프/2026 DCC 대학부 데이터/대학부 데이터"
os.environ["M2_CACHE_DIR"] = "/content/drive/MyDrive/데이터 크리에이터 캠프/m2_cache"
os.makedirs(os.environ["M2_CACHE_DIR"], exist_ok=True)
```

```bash
# 4) wav<->json manifest 생성 (한 번만 하면 재사용 가능 -- 이미 있으면
#    FileExistsError가 뜨는데 정상이니 다음 단계로 넘어가면 된다)
%cd /content/prelim
!python scripts/build_manifest.py \
    --data-root "$DATA_ROOT" \
    --output data/manifests/dcc_manifest_v1.csv

# 5) Mission 2 발화 인덱스 + 53차원 feature 캐시 생성 (통화 3,000건 샘플,
#    seed=42). 가장 오래 걸리는 단계(1.5~2시간대) -- 결과를 처음부터
#    $M2_CACHE_DIR(Drive)에 저장해야 런타임이 끊겨도 안전하다.
%cd /content/prelim
!python scripts/build_mission2_dataset.py \
    --manifest data/manifests/dcc_manifest_v1.csv \
    --data-root "$DATA_ROOT" \
    --n-calls 3000 --seed 42 \
    --index-output "$M2_CACHE_DIR/m2_utterances_train.parquet" \
    --features-output "$M2_CACHE_DIR/m2_features_train.parquet" \
    --failures-log "$M2_CACHE_DIR/m2_build_failures.jsonl"

# 6) baseline 학습 (call_id GroupShuffleSplit 8:2 + 비교용 random split)
%cd /content/prelim
!python scripts/train_mission2_baseline.py \
    --features "$M2_CACHE_DIR/m2_features_train.parquet" \
    --model-output "$M2_CACHE_DIR/m2_baseline_lgbm.pkl" \
    --log-output "$M2_CACHE_DIR/m2_exp_log.json"

# 7) 추론 (대회가 실제로 실행할 명령과 동일한 형태)
%cd /content/prelim
!python scripts/mission2_inference.py \
    --audio_dir {wav folder} --label_dir {json folder} \
    --ckpt_path "$M2_CACHE_DIR/m2_baseline_lgbm.pkl" \
    --output "$M2_CACHE_DIR/mission2.csv"
```

Colab에서 겪었던 함정들:

- Google Drive에 저장하지 않고 `/content/` 로컬 경로에만 결과를 쓰면, 런타임이
  끊기는 순간 몇 시간짜리 5단계 결과가 통째로 사라진다. 항상 `$M2_CACHE_DIR`
  (Drive)로 출력 경로를 지정할 것.
- `!cmd "{python_변수}"` 형태의 IPython 문자열 치환은 이 환경에서 간헐적으로
  실패했다(치환 안 되고 글자 그대로 들어감). `$ENV_VAR`(bash 환경변수, 위처럼
  `os.environ[...]`으로 미리 설정) 방식은 매번 정상 동작했으니 그쪽을 쓸 것.
- 런타임이 재시작되면 clone·마운트·환경변수가 전부 초기화된다. 스크립트 실행
  셀 맨 앞에 항상 `%cd /content/prelim`을 넣어 cwd를 강제로 맞출 것.
- 한글 폴더명(`TS_서울_구급` 등)이 zip 압축을 거치며 유니코드 NFD로 인코딩된
  적이 있었다 (Seoul+Training 필터가 0건으로 잡히는 문제) --
  `path_region()`에서 NFC로 정규화해서 수정됨 (커밋 `416e843`).

산출물:

- `data/m2_utterances_train.parquet` — 발화 단위 인덱스 (Git 추적 제외)
- `data/m2_features_train.parquet` — 53차원 feature 캐시 (Git 추적 제외)
- `checkpoints/mission2/m2_baseline_lgbm.pkl` — 학습된 모델 (Git 추적 제외)
- `reports/mission2/m2_exp_log.json` — GroupSplit/random split accuracy, feature importance top 20 등 실험 기록

`reports/mission2/m2_exp_log.json`의 `verdict` 필드는 실제 데이터로 학습한 뒤 직접 채워 넣는다
(`측정된 accuracy가 채택 기준 0.70 이상인지, 기각 기준 0.60 미만인지`).

로컬 smoke test (실데이터 없이 합성 fixture로 파이프라인 전체를 검증):

```bash
pytest tests/test_mission2_baseline.py -v
```
