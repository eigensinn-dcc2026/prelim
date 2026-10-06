# Mission 3 P0 로컬 재현 로그 (TF-IDF + LR)

## 목적
- M3 보험 제출본(P0: TF-IDF + OvR Logistic Regression)의 로컬 재현 기록.
- 팀 규칙에 따라 노트북 셀 출력을 보존한 실행본(`*_executed.ipynb`)을 그대로 커밋함.
- 코드 변경 없음, 기록만. 체크포인트(`.joblib`)와 `validation_predictions.csv`는 커밋하지 않음.

## 결과: PASS
| 항목 | 값 |
|---|---|
| Macro F1 | **0.567552** (ref 0.5676, diff -0.00005, tol ±0.003) → **PASS** |
| Micro F1 | 0.5779 |
| N | 29,200 (train 23,408 / val 5,792) |
| 로드 실패 | 0 |
| 실행 시간 | 15.9분 (스크립트 학습 구간 933.4s) |

- 증상별 CSV(`per_label_metrics.csv`)와 val 예측(`validation_predictions.csv`)은 이전 실행과 **바이트 단위로 동일**.
- 체크포인트 sha256: `d9b9df04f90784b499b5b2cc2eb55d630d899d44bbe35e72fdeaceaa0c30beb9`
  - joblib은 커밋하지 않음. 로컬 `outputs/m3_p0_rerun/`에 보관.
  - 이전 실행의 joblib 해시와는 다르나, 예측이 동일하므로 직렬화 차이로 판단.

## 재현 정보
- 실행 커밋: `a1ad073`
- 스크립트: `scripts/mission3_baseline 최종 수정본.py`
  - sha256 `fee2bef3d52568fd8ecfb31773f0b8987a3dc4a3517508d858fdac9fc7209103`
- 인자: `--test_size 0.2 --seed 42 --threshold 0.5`
- 환경: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, joblib 1.5.3, iterative-stratification 0.1.9 (전체 목록: `pip_freeze.txt`)

## min-1 증상 후처리 (규칙: 0.5 이상 라벨이 없으면 argmax 라벨을 1로)
- 정답 라벨 0개인 콜: 0건
- 영향받은 val 행: 293 / 5,792
- Macro F1 0.56755 → 0.56798 (+0.0004)
- 오심/두통 F1 사실상 변화 없음
- 고정 규칙이며 임계값 튜닝 없음, 공식 Val 미사용.
- 참고: min1 노트북에는 `repo_commit` 35c6075가 기록됨. 세션 도중 공유 작업 트리의 HEAD가 바뀌었기 때문이며, M3 스크립트 sha256은 동일함(위 값과 일치).

## 파일
| 파일 | 내용 |
|---|---|
| `mission3_p0_rerun_executed.ipynb` | 재현 실행 노트북 (셀 출력 포함) |
| `metrics.json` | 스크립트 출력 지표 |
| `per_label_metrics.csv` | 증상별 P/R/F1 |
| `verdict.json` | ref 대비 판정, 해시, 커밋 |
| `run_log.txt` | 스크립트 stdout 로그 |
| `pip_freeze.txt` | 실행 환경 패키지 목록 |
| `mission3_p0_min1_postproc_executed.ipynb` | min-1 후처리 평가 노트북 (셀 출력 포함) |
| `min1_metrics.json` | 후처리 전/후 지표 |
| `min1_per_label.csv` | 후처리 전/후 증상별 지표 |
