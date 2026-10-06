# 팀원 Colab 실행 준비 (경로 수정 없이 실행)

노트북·스크립트는 `src/dcc_data/colab_env.py`가 경로를 자동으로 찾습니다. 아래 두 가지만 하면 됩니다.

1. 내 드라이브(My Drive) 최상단에 **바로가기 추가** (우클릭 → 정리 → 바로가기 추가, 이름은 그대로)
   - `대학부 데이터` (공식 배포 폴더, `Training/`·`Validation/`가 바로 보이는 폴더)
   - `dcc_runs` (편집자로 공유받은 실행 폴더: 체크포인트·로그 저장)
2. 노트북 첫 설정 셀 실행 → Drive 마운트 후 자동 탐색

## 탐색 순서

| 대상 | 우선순위 |
|---|---|
| 데이터 | 인자 → `DCC_DATA_ROOT` → `data/raw_original` → `MyDrive/대학부 데이터` → `.shortcut-targets-by-id/*/대학부 데이터` → 공유 드라이브 |
| 출력 | 인자 → `DCC_RUN_ROOT` → `MyDrive/dcc_runs` (Colab) → `outputs/` (로컬) |

- `DCC_DATA_ROOT`를 지정했는데 경로가 틀리면 다른 곳으로 넘어가지 않고 오류를 냅니다.
- 못 찾으면 시도한 경로를 모두 출력합니다.
- 로컬은 `DCC_DATA_ROOT`만 지정하면 됩니다. 기존 `M1_DATA_ROOT`도 계속 인식합니다.
