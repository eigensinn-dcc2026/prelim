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
| M3-KCBERT-001 | 2026-09-01 | KcBERT 다중 라벨 baseline, 앞 256토큰, Training 90%/내부 검증 10%, seed=42, 비가중 BCE, 3 epochs | 내부 검증 Macro F1 최고 0.5861(epoch 2) | baseline 완료 | [`m3_kcbert_baseline_train.log`](../reports/mission3/m3_kcbert_baseline_train.log) |
| M3-EXP-001 | 2026-09-11 | TF-IDF + One-vs-Rest LR, `recordId` 단위 다중레이블 층화 분할 | PR 문서에 `[실행 후 입력]` placeholder가 남아 있어 결과 미기재 | 미확정 | [`mission3_baseline.py`](../scripts/mission3_baseline.py), [`M3_PR.docx`](../scripts/M3_PR.docx) |
| M3-CHUNK-001 | 2026-09-16 | 전체 토큰 청크, Training 20%, 256토큰·overlap 64·max pooling, 공식 Validation 3,640건 평가 | Macro F1 0.5496 / Micro F1 0.6054 / subset accuracy 0.4253 | 완료 | [`m3_chunk_validation_metrics.json`](../reports/mission3/m3_chunk_validation_metrics.json) |
| M3-TURN-001 | 2026-09-18 | 질문과 답변의 경계를 보존하는 발화 턴 청크, 공식 Validation 3,640건 평가 | Macro F1 0.5365 / Micro F1 0.5932 / subset accuracy 0.4080 | 실행 완료·학습 조건 추가 확인 필요 | [`m3_turn_chunk_validation_metrics.json`](../reports/mission3/m3_turn_chunk_validation_metrics.json) |
| M3-EXP-002 | 2026-09-19 | TF-IDF + LR, JSON 행 단위 MultilabelStratifiedShuffleSplit, seed=42 | 보고값: Macro F1 0.3955→0.4530, Precision 0.5639 / Recall 0.4110 | 과거 실행값 — 최신 80:20 기준 재실행 전 직접 비교하지 않음 | [`mission3_baseline 최종 수정본.py`](<../scripts/mission3_baseline 최종 수정본.py>), [`F1 성능 검증 과정 및 결과.docx`](<../scripts/F1 성능 검증 과정 및 결과.docx>), [`decisions.md`](./decisions.md) |
| M3-WEIGHT-001 | 2026-09-19 | 턴 청크 모델에 증상별 양성 손실 가중치 적용, Training 50% | 결과 원본 파일 추가 확인 필요 | 실행 완료·결과 확인 필요 | [`m3_weighted_diagnosis_sources.json`](../reports/mission3/m3_weighted_diagnosis_sources.json) |
| M3-THRESH-001 | 2026-09-19 | weighted 모델의 내부 검증 결과를 이용해 증상별 판정 임계값 조정 | 선택된 임계값 배열이 남아 있지 않음 | 실행 완료·재현 정보 부족 | [`m3_threshold_comparison.json`](../reports/mission3/m3_threshold_comparison.json) |
| M3-THRESH-002 | 2026-09-19 | 증상별 임계값 조정을 중단하고 모든 증상을 0.5로 고정해 재평가 | 별도 평가 결과 파일 확인 필요 | 평가 방식 변경·결과 확인 필요 | [`MISSION3_Colab_가중치모델_임계값점검.ipynb`](../notebooks/mission3/MISSION3_Colab_가중치모델_임계값점검.ipynb) |
| M3-WEIGHT-002 | 2026-09-20 | 완화 weighted BCE, Training 40%, scale 0.25, max 3.0, effective batch 16, 3 epochs | 체크포인트 생성, 평가 결과 확인 필요 | 학습 완료·평가 필요 | [`train_weighted_40pct_vscode.py`](../scripts/mission3/train_weighted_40pct_vscode.py) |

> 최초 KcBERT의 0.5861은 Training **90:10 내부 검증** 결과이고, 청크 모델의 0.5496/0.5365는
> **공식 Validation** 결과이므로 점수만으로 직접 우열을 판단하지 않습니다.
> 2026-09-30부터 Mission 3의 현재 비교 기준은 **JSON 샘플 단위 80:20 multilabel stratified split,
> seed=42, threshold=0.5**입니다. 이전 분할에서 나온 점수는 과거 기록으로 보존하고 최신 기준 재실행 전 직접 비교하지 않습니다.
> 임계값·가중치 실험은 누락된 결과와 설정을 확인한 뒤 수치를 보완해야 합니다.

---

새 실험은 해당 Mission 표의 맨 아래에 추가합니다.
