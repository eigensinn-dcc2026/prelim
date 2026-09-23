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
| M2-EXP-006 | 2026-09-10 | pitch/energy 단독 피처 추가(f0_mean, voiced_ratio 등) | 효과 거의 없음(~0.15%p) | 기각 — 기존 spectral_centroid/MFCC0과 중복 | [`m2_progress_narrative.md`](../reports/mission2/m2_progress_narrative.md) §"막다른 길" |
| M2-EXP-007 | 2026-09-10 | harmonicity(overlap-detection) 피처, 자기상관 피크 기반 | 겹침 구간 오류율 8.38%p→8.66%p 악화 | 기각 — 합성 테스트만 통과, 실데이터 역효과 | [`m2_progress_narrative.md`](../reports/mission2/m2_progress_narrative.md) §"막다른 길" |
| M2-EXP-008 | 실행 2026-09-10, PR#11 머지 09-15, 되돌림 09-18 | call-relative(자기 통화 내 z-score) 피처 — f0_mean/rms_p90/duration_sec을 통화 내 평균·표준편차로 정규화 | 내부 +0.1~0.3%p였으나 **대회 FAQ(09-15)로 규정 위반 확정** | **되돌림(무효)** — `decisions.md` 참조 | [`m2_progress_narrative.md`](../reports/mission2/m2_progress_narrative.md) §"시도했다가 철회한 것" |
| M2-TUNE-001 | 날짜 미상(09-10~09-15 사이로 추정) | LightGBM 하이퍼파라미터 재튜닝(`num_leaves=255, lr=0.02, n_estimators≈1150`), 단 이 시점 64차원(규정 위반 피처 포함) | 84.59%→86.9%(+2.3%p) | 오염된 피처 기준이라 공식 수치로 미인용 — M2-COMPARE-001에서 53차원 재검증 완료 | 별도 committed 리포트 없음(TODO) |
| M2-LABEL-001 | 2026-09-10 | 라벨 스왑 최초 감사 — 극단적 저성능 통화의 전사문(text)을 사람이 1회성으로 읽어 확인, 모델 feature로는 미사용 | 확정 전체 뒤집기 4건, 자동 스캔(19,969/29,200통화) 17건 후보 중 3건 추가 확정·2건 오탐·10건 보류 | 라벨 4건 수정 반영 | [`m2_label_swap_audit.md`](../reports/mission2/m2_label_swap_audit.md) |
| M2-LABEL-002 | 2026-09-14 | 라벨 불연속 자동 탐지기 — "전체 뒤집기"로 이미 고쳤다고 본 통화가 실은 통화 중간부터만 뒤바뀐 부분 뒤집기였음을 모델 예측 패턴으로 발견 | 2건 추가 확정(28.57%→100%, 27.78%→100%) | 라벨 2건 추가 수정, held-out val만 스캔 가능(전체의 20%) | [`m2_label_swap_audit.md`](../reports/mission2/m2_label_swap_audit.md) |
| M2-COMPARE-001 | 2026-09-18 | L1 모델 계열 공정 비교(GaussianNB/로지스틱/RandomForest/XGBoost/LightGBM, 53차원 순정, 동일 GroupSplit) + LightGBM 최종 튜닝 | **LightGBM 튜닝 86.77% 최고**(84.59% 기준선 대비 +2.18%p) | L1 최종 후보 확정 | [`m2_l1_model_comparison_results.md`](../reports/mission2/m2_l1_model_comparison_results.md) |
| M2-EXP-009 | 2026-09-18 | 대원 recall 밀어보기 — threshold(0.5→0.75)/class_weight(1.0→3.0) 스윕 | 모든 조정이 accuracy 손해(threshold=0.55: −0.20%p) | 기각(적용 안 함) — 채점이 accuracy 단일 지표 | [`m2_l1_daewon_recall_push_results.md`](../reports/mission2/m2_l1_daewon_recall_push_results.md) |
| M2-VAL-001 | 2026-09-18 | 공식 Validation 세트(3,640통화/111,947발화) 평가, LightGBM 튜닝 체크포인트, 재학습 없음, 1회성 | **86.40%**(내부 split 86.77% 대비 −0.37%p) | **확정 — L1 공식 후보** | [`m2_l1_official_validation_results.md`](../reports/mission2/m2_l1_official_validation_results.md) |
| M2-CNN-001 | 2026-09-10 | L2(Mel-spectrogram+CNN) 파이프라인 구축 + 3,000통화 파일럿(SpeakerCNN 소형) | 86.56%(수렴), 같은 표본 L1 최고 상회하나 격차 4.95%p→9.20%p 확대 | 참고용(파일럿) | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-002 | 2026-09-14 | L2 CNN 전체 데이터(29,200통화) SpeakerCNN 소형 | 89.71%(수렴), 격차도 27.57%p→7.66%p로 축소 | 참고용 | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-003 | 2026-09-14 | L2 CNN SpeakerCNNLarge(4 conv 블록, 421,954 파라미터) | **91.22%**(최종 수렴), 격차 7.27%p | 내부 검증 최고 기록, 공식 Validation 미검증·`main` 미병합 | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-004 | 2026-09-16 | L2 화자 격차 겨냥 A: SpeakerCNNLargeWithDuration(duration_sec 명시 입력) | accuracy +0.30%p이나 격차 8.45%p→9.17~9.35%p 확대 | 기각 — 가설(격차 원인) 미검증 | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-005 | 2026-09-16 | L2 화자 격차 겨냥 B: hop-length 절반 축소(고해상도 스펙트로그램) | 인프라만 구축(프레임 수 16→31, 76→151 확인) | 미평가 — L1 우선순위로 보류 | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-AUDIT-001 | 2026-09-20 | PR #4/#9/#11/#14 사후 코드 리뷰 — 셀프 머지 여부 및 코드 정합성 확인 | PR #9/#11/#14 셀프 머지 확인, `build_mission2_dataset.py --resume`이 학습 통화 29,200개 중 2,170개(7.4%)를 조용히 유실시키는 버그 발견 | 버그 수정 PR #19 팀원 리뷰 후 머지 완료(2026-09-21) — 누락 통화 재처리·재학습 대기 | `scripts/build_mission2_dataset.py`, PR #19 |
| M2-AUDIT-002 | 2026-09-20 | 브랜치/PR 간 상호작용 감사(L2 CNN 트랙 중심) | `mission2-l2-cnn-clean`(PR #17, 미머지)에 라벨 스왑 수정이 통째로 빠져있음을 발견 — cherry-pick 시 파일은 챙겼지만 그 파일이 나르던 수정 내용은 놓친 사례 | L2 재개 전 라벨 수정 rebase 필요 — 09-22 완료(아래 M2-VAL-002 참조) | 세션 기록만(별도 리포트 없음) |
| M2-LABEL-003 | 2026-09-22 | 라벨 스왑 방향 재검토 — `651e4c754428de718c773b56`을 전사문 21개 발화 전체 재확인, 부분 뒤집기 경계(`best_flip_point`)가 이미 한 번 뒤집힌 라벨 기준으로 계산돼 반대 방향(`>=` 대신 `<`)으로 적용됐어야 했음을 발견 | 21개 발화 전부와 100% 일치하는 방향으로 수정, `PARTIAL_LABEL_SWAP_FLIP_BEFORE_UTT_IDX` 규칙 추가 | 라벨 로직 수정, 회귀 테스트 추가(`test_mission2_label_swap_fix.py`) | `src/dcc_data/mission2_features.py`, `tests/test_mission2_label_swap_fix.py` |
| M2-VAL-002 | 2026-09-21~22 | M2-AUDIT-001 후속 — 누락 통화 재처리(889,358행, 자세한 수는 `m2_exp_log_postfix2.json` 참조) + M2-LABEL-003 라벨 수정 반영해 재학습, 공식 Validation 재평가(1회성, 재튜닝 없음) | 내부 GroupSplit **86.93%**(86.77% 대비 +0.16%p), 공식 Validation **86.44%**(86.40%→+0.04%p, 사실상 무변화 — 이 통화가 Training split에 속해 있었음을 시사) | **확정 — M2-VAL-001을 대체하는 L1 최종 후보**(`m2_baseline_lgbm_tuned_postfix2.pkl`) | [`m2_exp_log_postfix2.json`](../reports/mission2/m2_exp_log_postfix2.json), [`m2_l1_official_validation_result_postfix2.json`](../reports/mission2/m2_l1_official_validation_result_postfix2.json) |
| M2-VAL-003 | 2026-09-22~23 | L2(SpeakerCNNLarge, M2-CNN-003 체크포인트에 라벨 수정 반영해 재학습) 공식 Validation 최초 평가(1회성, 재튜닝 없음) — 로컬 CPU와 Colab GPU 양쪽에서 같은 체크포인트(epoch 21)로 재확인 | **90.88~90.89%**(CPU 90.89%/GPU 90.88%, 오분류 약 250건이 CPU·GPU 부동소수점 차이로 이동 — 재학습·재튜닝 없이 같은 체크포인트를 두 하드웨어에서 잰 것이므로 "1회성 확인" 원칙 위반 아님). L1(M2-VAL-002) 대비 **+4.44~4.45%p** | **확정 — 현재 L1·L2 통틀어 최고 기록, 제출 후보 1순위**. 타 팀 조사에서 확인된 최고 기록(eunkyo00, 90.25%)도 상회 | [`m2_l2_official_validation_result_local_cpu.json`](../reports/mission2/m2_l2_official_validation_result_local_cpu.json), [`m2_l2_official_validation_result_colab_gpu.json`](../reports/mission2/m2_l2_official_validation_result_colab_gpu.json) |
| M2-COMPARE-002 | 2026-09-23 | M2-VAL-002(L1)·M2-VAL-003(L2) per-utterance 비교(공식 Validation 111,947발화) — 겹침 여부/발화 길이/정답 클래스별 accuracy, 동의·불일치 구조 | 전체 L1 86.44% / L2 90.88%(+4.4%p). 안 겹침(n=63,084) L1 90.01%/L2 92.44%(+2.43%p) vs 겹침(n=48,863) L1 81.83%/L2 88.88%(+7.05%p) — CNN 우위가 겹침 구간에 집중. 길이별도 전 구간 L2 우세, 0.5~2초 짧은 발화에서 격차 최대(0.5-1s 81.9%→86.9%, 1-2s 84.2%→89.0%). 동의 구조: 둘 다 맞음 94,545 / 둘 다 틀림 7,980 / L1만 틀림(L2가 고침) 7,198 / L2만 틀림(L1이 고침) 2,224 — L2가 L1 오류를 고치는 경우가 반대보다 약 3.2배 많음 | 참고용 — L2 우위가 겹침·단발화 구간에서 비롯됨을 정량 확인, L2 채택 결정(2026-09-23 decisions.md) 근거 보강. 앙상블 등 추가 조치는 미결 | [`m2_l1_l2_official_validation_comparison.json`](../reports/mission2/m2_l1_l2_official_validation_comparison.json), [`m2_l1_l2_official_validation_comparison.csv`](../reports/mission2/m2_l1_l2_official_validation_comparison.csv) |

> M2-SCALE-001 이후 실험 중 별도 JSON 로그가 없는 항목은 보고서를 근거로 사용합니다(위 표의
> "별도 committed 리포트 없음"/"세션 기록만" 표시 참조). 재실행 시 실험 ID, 설정과 결과를
> JSON으로 함께 저장해야 합니다.

## Mission 3 — 환자 증상 인식 (F1 Macro, 9개 증상)

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| M3-EXP-001 | 2026-09-11 | TF-IDF + One-vs-Rest LR, `recordId` 단위 다중레이블 층화 분할 | PR 문서에 `[실행 후 입력]` placeholder가 남아 있어 결과 미기재 | 미확정 | [`mission3_baseline.py`](../scripts/mission3_baseline.py), [`M3_PR.docx`](../scripts/M3_PR.docx) |
| M3-EXP-002 | 2026-09-19 | TF-IDF + LR, JSON 행 단위 MultilabelStratifiedShuffleSplit, seed=42 | 보고값: Macro F1 0.3955→0.4530, Precision 0.5639 / Recall 0.4110 | 보류 — 분할 감사 전 비교 불가 | [`mission3_baseline 최종 수정본.py`](<../scripts/mission3_baseline 최종 수정본.py>), [`F1 성능 검증 과정 및 결과.docx`](<../scripts/F1 성능 검증 과정 및 결과.docx>), [`decisions.md`](./decisions.md) |

> M3는 현재 구조화된 실험 로그가 없습니다. 다음 실행에서는 전체 행 수, 고유 `recordId` 수,
> 중복 그룹 수, train/validation의 `recordId` overlap, 분할 단위와 지표를 JSON으로 저장해야 합니다.

---

새 실험은 해당 Mission 표의 맨 아래에 추가합니다.
