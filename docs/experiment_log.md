# 실험 로그 (Experiment Log) — 사람이 읽는 요약판

> 이 파일은 실험을 한눈에 찾기 위한 인덱스입니다. 최종 수치의 source of truth는 각 행의
> `관련 파일`에 연결된 JSON 로그, 지표 파일 또는 보고서입니다. 구조화된 로그가 없는 실험은
> 보고서를 임시 근거로 사용하되, 후속 실행부터 기계 판독 가능한 로그를 남깁니다.
> 채택·기각 이유와 방향 변경은 `docs/decisions.md`에 기록합니다.
>
> 기록 필수 항목: 실험 ID, `YYYY-MM-DD` 날짜, 데이터 규모, 분할 단위, seed, 주요 결과, 상태, 관련 파일.
>
> 실험 ID 규칙: `M{미션번호}-{종류}-{일련번호}` (예: `M2-SCALE-002`). 실제 JSON에 기록된 ID는
> `M1-EXP-001`, `M1-EXP-001-val`, `M1-VAL-ERR-001`, `M2-EXP-001`입니다. `M2-EXP-002`는
> 오류 분석 보고서에서만 사용됐고 별도 JSON은 없습니다. 그 밖의 ID는 이 문서를 만들며 소급해
> 부여했습니다. 새 실험은 표에서 다음 번호를 확인해 할당하고 실행 로그에도 같은 ID를 남기세요.

## Mission 1 — 신고자 성별 분류 (Accuracy)

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| M1-EXP-001 (internal) | 2026-09-10 | 52차원 acoustic + GBM/LR, Training 내부 stratified 8:2 sanity split, seed=42 | GBM 94.78% / LR 95.02% | 참고용(내부) | [`m1_exp_log.json`](../reports/mission1/m1_exp_log.json) |
| M1-EXP-001-val | 2026-09-10 | 공식 Validation 평가, 3,640건(F 1,959/M 1,681) | GBM 94.92% / LR 95.11% | LR 1차 채택 후보 | [`m1_val_eval.json`](../reports/mission1/m1_val_eval.json) |
| M1-EXP-001-val 파생 | 2026-09-10 | Bootstrap 95% CI, n_boot=10,000, seed=42 | GBM [94.20, 95.60] / LR [94.40, 95.82] | 불확실성 확인용 | [`m1_val_bootstrap_ci.json`](../reports/mission1/m1_val_bootstrap_ci.json) |
| M1-VAL-ERR-001 | 2026-09-10 | 오류 분석(duration/overlap/feature 분포) | 상세 결과는 보고서 참조 | 잠정 종료 | [`m1_val_error_analysis.json`](../reports/mission1/m1_val_error_analysis.json), [`M1_error_analysis_report.md`](../reports/mission1/M1_error_analysis_report.md) |

## Mission 2 — 신고자/119대원 분류 (Accuracy)

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| M2-EXP-001 | 2026-09-03 | Baseline: 53차원 acoustic + LightGBM 기본값, 3,000통화, `call_id` GroupShuffleSplit 8:2, seed=42 | 83.78% (random split 84.42%, gap +0.65%p) | baseline 채택 | [`m2_exp_log.json`](../reports/mission2/m2_exp_log.json) |
| M2-SCALE-001 | 2026-09-15 | 같은 파이프라인, 9,000통화 | 84.90% (3,000건 대비 +1.12%p) | 성능 기준 채택 후보 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 0·3절 |
| M2-SCALE-002 | 2026-09-15 | 같은 파이프라인, 29,200통화 전체 | 84.59% (9,000건 대비 -0.31%p, gap -0.05%p) | 오류 분석 기준 실행·성능 선택 보류 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 0·3절 |
| M2-EXP-002 | 2026-09-15 | Threshold 보정(0.5→다른 값) | 최적값 0.5, 개선 없음 | 기각 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 1-4·3절 |
| M2-EXP-003 | 2026-09-15 | `class_weight="balanced"` 재학습 | +0.02%p | 기각(잡음 수준) | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 3절 |
| M2-EXP-004 | 2026-09-15 | 겹침 학습 데이터 제외 | -0.25%p | 기각 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 3절 |
| M2-EXP-005 | 2026-09-15 | 겹침 학습 데이터 가중치 0.3x | +0.14%p | 기각(잡음 수준) | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 3절 |
| M2-VAL-001 | — | 공식 Validation 세트 평가 | 아직 수행 안 됨 | 미실행 | [`eval_mission1_validation.py`](../scripts/eval_mission1_validation.py)를 참고해 작성 필요 |

> M2-SCALE-001 이후 실험은 현재 별도 JSON 로그가 없어 오류 분석 보고서를 근거로 사용합니다.
> 재실행 시 실험 ID, 설정과 결과를 JSON으로 함께 저장해야 합니다.

## Mission 3 — 환자 증상 인식 (F1 Macro, 9개 증상)

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| M3-EXP-001 | 2026-09-11 | TF-IDF + One-vs-Rest LR, `recordId` 단위 다중레이블 층화 분할 | PR 문서에 `[실행 후 입력]` placeholder가 남아 있어 결과 미기재 | 미확정 | [`mission3_baseline.py`](../scripts/mission3_baseline.py), [`M3_PR.docx`](../scripts/M3_PR.docx) |
| M3-EXP-002 | 2026-09-19 | TF-IDF + LR, JSON 행 단위 MultilabelStratifiedShuffleSplit, seed=42 | 보고값: Macro F1 0.3955→0.4530, Precision 0.5639 / Recall 0.4110 | 보류 — 분할 감사 전 비교 불가 | [`mission3_baseline 최종 수정본.py`](<../scripts/mission3_baseline 최종 수정본.py>), [`F1 성능 검증 과정 및 결과.docx`](<../scripts/F1 성능 검증 과정 및 결과.docx>), [`decisions.md`](./decisions.md) |

> M3는 현재 구조화된 실험 로그가 없습니다. 다음 실행에서는 전체 행 수, 고유 `recordId` 수,
> 중복 그룹 수, train/validation의 `recordId` overlap, 분할 단위와 지표를 JSON으로 저장해야 합니다.

---

새 실험은 해당 Mission 표의 맨 아래에 추가합니다.
