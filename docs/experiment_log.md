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
| M1-INF-001 | 2026-10-05 | 제출용 CLI 추론(`scripts/mission1_inference.py`): WAV+JSON → speaker=1 구간 52D → LR → CSV(`audio file name`, `gender`). 새 학습 없음 | 구현 완료. 스모크(헤더·행 수·UTF-8)는 미확인 | 제출 점검 진행 중 | [`m1_submission_checklist.md`](../reports/mission1/m1_submission_checklist.md) |

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
| M2-VAL-001 | 2026-10-05 | 공식 Validation 평가 — 아래 M2-MFCC/FEAT/MODEL 행은 공식 Val(111,947발화)에서 평가됨. **L1-LGBM 9k 후보는 ckpt가 없어 미실행**(스펙만 존재) | MLP 트랙 85.29% / L1-LGBM 9k 미평가 | 부분 완료 | [`m2_official_val_spec.md`](../reports/mission2/m2_official_val_spec.md) |
| M2-MFCC-001 | 소급(≈2026-09, 노트북) | MFCC40 + StandardScaler + LR, 공식 Val 평가. majority baseline 51.90% | **80.91%** (speaker0 recall 약 0.765 / speaker1 약 0.850) | 이후 feature·모델 실험의 고정 비교점(E00) | [`mission2_yeonwoo_baseline.ipynb`](../notebooks/mission2_yeonwoo_baseline.ipynb) |
| M2-FEAT-E01~E04 | 소급(≈2026-09-14~27) | MFCC40에 feature 1개씩 추가: E01 duration_ms, E02 spectral_rolloff, E03 zcr, E04 spectral_centroid. Training 889,358발화(internal 711,467 / 177,891, call_id GroupSplit, seed=42), 공식 Val | E01 80.91%(+0.00%p) / E02 81.15%(+0.24) / E03 81.01%(+0.10) / E04 81.05%(+0.14) | E02 최대, E01 효과 없음. E03·E04는 Training 순위로 사전 확정 | [`mission2_feature_experiments.ipynb`](../notebooks/mission2_feature_experiments.ipynb) |
| M2-FEAT-E05 | 소급(≈2026-09-27) | MFCC40 + rolloff + zcr (42D). Training-only 중복성·오류 보완성 진단 후 사전 확정 | 공식 Val **81.31%**(+0.40%p, +449건). speaker0 recall +0.94%p, speaker1 -0.10%p, 고확신 오답 2,086→2,158 | 42D 고정 후 모델 비교로 이행. **v1 cache 기준 결과이며 v2 코드로 재평가되지 않음.** 단일 feature 간 유의성 검정 없음 | [`mission2_feature_experiments.ipynb`](../notebooks/mission2_feature_experiments.ipynb) |
| M2-MODEL-M00~M04 | 소급(≈2026-09-21) | 42D 고정, 모델만 변경. 동일 Training 내부 split | 내부: M00 LR 81.73 / M01 HGB 84.19 / M02 ExtraTrees 82.00 / M03 QDA 79.73 / M04 MLP(64, 30 epoch) 85.80. 공식 Val: M01 83.73, **M04 85.29(+1.57%p vs M01)** | M04 MLP를 42D 기준 best 후보로 선정(최종 아님). M04만 공식 Val 승격 | [`mission2_model_experiments.ipynb`](../notebooks/mission2_model_experiments.ipynb) |
| M2-MODEL-O | 소급(≈2026-09-21) | M04 max_iter만 변경(내부 split) | 30→60: 85.80→85.88(+0.08%p), 30→100: 85.95(+0.16%p). 모두 ConvergenceWarning 지속 | 수렴 효과 작음. 이후 feature 스크리닝 기준을 M04-100으로 사용 | [`mission2_model_experiments.ipynb`](../notebooks/mission2_model_experiments.ipynb) |
| M2-AF-R1 | 소급(≈2026-09-21) | 47D 기준(E05 42D + 후보 block 1개씩): AF01 energy / AF02 pitch·voicing / AF03 spectral flux. M04-100, 내부 split | 기준 85.95%(재현값 0.859532). **AF02만 86.63%(+0.67%p)**, AF01·AF03은 하락 | AF02 채택 → 47D. AF01·AF03 기각 | [`mission2_acoustic_feature_expansion.ipynb`](../notebooks/mission2_acoustic_feature_expansion.ipynb) |
| M2-AF-R2 | 소급(≈2026-09-21) | 47D에 AF04 CPP·AF05 delta-MFCC를 각각 추가(Round 2) → 고정한 49D 기준에 AF07/08/09를 하나씩 추가 | 49D 기준 86.86%. AF07 +0.05%p(speaker0 recall 하락, HOLD), AF08 -0.02%p(DROP 유력), AF09 +0.03%p | 모두 잡음 수준 | 같은 노트북 |
| M2-AF-R3 | 소급(≈2026-09-21) | 52D에 AF10 speech activity 4D + AF11 spectral entropy 3D → 59D. 내부 split | 86.95%→**87.46%**(+0.50%p, +894건), speaker0/1 recall 모두 상승, 고확신 오답 2,590→2,377 | **59D를 handcrafted reference로 채택** | 같은 노트북 |
| M2-AF-R4 | 소급(≈2026-09-21) | 59D + AF05(delta-MFCC 2D) = 61D | 87.46%→87.47%(+0.018%p), speaker0 recall +1.47 / speaker1 -1.32%p | 보류. handcrafted 탐색 종료 | 같은 노트북 |
| M2-REP-001 | 2026-10-04 | 표준 표현 대조: clean 59D vs eGeMAPSv02 88D. 동일 M04-100·internal split(177,891발화), 0.5 threshold | 59D **87.38%** vs 88D 86.40%(-0.98%p). speaker0 recall 84.67→82.36%, 고확신 오답 2,378→2,733 | 88D 열세. 0.5–1s 구간 82.5→81.8%. 59D 유지 | [`mission2_standard_representation_comparison.ipynb`](../notebooks/mission2_standard_representation_comparison.ipynb) |

> M2-SCALE-001 이후 L1 실험은 별도 JSON 로그가 없어 오류 분석 보고서를 근거로 사용합니다.
> **M2-MFCC/FEAT/MODEL/AF/REP 행은 2026-10-05에 노트북 출력과 사용자 해석 셀에서 소급 기록했습니다.**
> 수치의 출처는 노트북이며 별도 JSON 로그가 없으므로, 제출 근거로 쓰기 전에 로그 JSON 확보가 필요합니다.
> **비교 주의:** (1) 3,000~29,200통화 L1+LGBM 계열(83~85%)은 call_id GroupSplit **내부** 점수입니다.
> (2) MFCC/FEAT/MODEL 계열은 **공식 Val 111,947발화**, AF·REP 계열은 Training **내부 177,891발화** 점수입니다.
> 두 계열을 직접 비교하지 않습니다. (3) FEAT 계열은 v1 cache 결과이며, 87.38%(clean 59D)는 이전에 기록된 87.46%와 재현 차이가 있습니다.

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
| M3-REEVAL-P0 | 2026-10-05 | 재평가 P0: TF-IDF+OvR-LR, 80:20 multilabel strat, seed=42, th=0.5. 파이프라인·메트릭 sanity | 미실행(`m3_reeval_tfidf_lr/` 비어 있음) | 승인 대기 | [`m3_reeval_spec.md`](../reports/mission3/m3_reeval_spec.md) |
| M3-REEVAL-P1~P3 | 2026-10-05 | P1 KcBERT 앞 256 / P2 token vs turn 청크 동일조건 대조 / P3 완화 가중 BCE | 후보 정의만 있음, 미실행 | 후보(P0 이후) | [`m3_next_experiment_candidates.md`](../reports/mission3/m3_next_experiment_candidates.md) |

> 최초 KcBERT의 0.5861은 Training **90:10 내부 검증** 결과이고, 청크 모델의 0.5496/0.5365는
> **공식 Validation** 결과이므로 점수만으로 직접 우열을 판단하지 않습니다.
> 2026-09-30부터 Mission 3의 현재 비교 기준은 **JSON 샘플 단위 80:20 multilabel stratified split,
> seed=42, threshold=0.5**입니다. 이전 분할에서 나온 점수는 과거 기록으로 보존하고 최신 기준 재실행 전 직접 비교하지 않습니다.
> 임계값·가중치 실험은 누락된 결과와 설정을 확인한 뒤 수치를 보완해야 합니다.


## 공통 — EDA

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| COM-EDA-001 | 2026-09-21~10-05 | 전수(32,840통화) EDA와 심화 분석: Train/Val 분포 이동(KS, Wasserstein), 겹침 시간 비율, 성별·화자 효과크기(Cliff's δ), 증상 lift. 집계 통계만 저장 | 요약은 보고서 7장. 예: M2 spectral centroid 화자 δ≈-0.72, M3는 목표 증상 0개 통화가 Training·Val 모두 0건 | 완료 | [`eda_report.md`](../reports/eda/eda_report.md), [`eda_deep_summary.json`](../reports/eda/eda_deep_summary.json) |

---

새 실험은 해당 Mission 표의 맨 아래에 추가합니다.
