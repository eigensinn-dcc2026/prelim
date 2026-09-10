# M1 (신고자 성별 분류) — Baseline · 공식 Validation · 오류 분석 보고서

- 실험 ID: M1-EXP-001 / M1-VAL-ERR-001
- 작성일: 2026-09-10
- 작성자: 김장수 (Claude 보조 정리)
- 데이터: Seoul Training 29,200건 (baseline 학습), 공식 Validation 3,640건 (모델 선택·평가)
- 상태: **잠정 종료 (Tentatively Closed)** — 아래 8절 참조

> 이 문서는 관찰된 사실과, 그로부터 나온 해석/추정을 엄격히 구분해서 씁니다.
> - **[확정된 관찰]**: 로그 파일에 그대로 찍힌 숫자, 재현 가능한 사실.
> - **[탐색적 해석]**: 관찰로부터 사람이 내린 해석. 반증 가능하고, 추가 검증 전까지는 잠정적.
> - **[후속 가설]**: 다음 실험에서 확인해볼 아이디어. 아직 증거가 아님.

---

## 목차

1. M1 baseline 요약
2. 공식 Validation 결과
3. 오류 분석 방법
4. duration / utterance count / overlap 결과
5. 52차원 acoustic feature 탐색 결과
6. 해석상의 한계
7. 후속 실험 후보 (F0 등)
8. 현재 M1 상태 — 잠정 종료

---

## 1. M1 baseline 요약

**[확정된 관찰]**

- 가설: 통화 단위 caller waveform의 52차원 음향 통계(acoustic_feature_vector)만으로 신고자 성별(F/M)을 분류할 수 있다.
- 고정 조건: Seoul Training 29,200건, seed=42, 52-d feature, duration_sec 제외, 비교 모델은 LightGBM(GBM)과 StandardScaler+LogisticRegression(LR).
- Majority baseline (Training, F 15,542/29,200) = **53.23%**. 이 값이 "아무것도 안 해도 나오는 바닥값"이며, 이후 모든 정확도는 이 값 대비로 읽어야 함.
- 내부 목표(internal_target)는 70%였고, 이는 팀 내부 1차 기준일 뿐 공식 채점 기준은 아님.
- 내부 sanity holdout (Training 내부 8:2 stratified split, n_holdout=5,840): GBM 94.78%, LR 95.02%.

**[탐색적 해석]**

- 내부 holdout과 아래 2절의 공식 Validation 결과가 서로 근접(0.1~0.3%p 차이)하다는 점은, 현재 평가 파이프라인에서 뚜렷한 leakage 징후는 관찰되지 않았다는 정도로만 말할 수 있음. 이 관찰이 leakage가 없음을 증명하는 것은 아니며, 코드 상 Training-only fit을 강제하는 안전장치(SplitContaminationError)가 실질적인 방어 수단임.

---

## 2. 공식 Validation 결과

**[확정된 관찰]** (source: `scripts/eval_mission1_validation.py`, `scripts/bootstrap_ci_mission1.py`, n_validation=3,640, F 1,959 / M 1,681)

| 모델 | Accuracy | 95% CI (bootstrap, n_boot=10,000, seed=42) | Recall F | Recall M | 오분류 건수 |
|---|---|---|---|---|---|
| Majority (F) | 53.82% | — | — | — | — |
| GBM | 94.92% | [94.20%, 95.60%] | 95.61% | 94.11% | 185건 (F→M 86, M→F 99) |
| LR | 95.11% | [94.40%, 95.82%] | 95.87% | 94.23% | **178건 (F→M 81, M→F 97)** |

- 두 모델의 정확도 차이는 0.19%p (오분류 건수 기준 7건 차이)로 작음.
- CI는 confusion matrix 기반 nonparametric bootstrap과 이항분포 정규근사(Wald) CI를 교차검증하여 소수점 셋째 자리까지 일치함을 확인함.
- **채택 후보: LR.** 근거는 (a) 정확도 차이가 작다는 점, (b) LR이 단순하고 해석이 쉽다는 점. GBM은 향후 feature 확장 실험용 비교 baseline으로 유지.
- recall_F/recall_M이 두 모델 모두 93~96% 범위로 균형적 → 다수 클래스(F)만 반복적으로 예측하는 단순 majority-class 편향은 관찰되지 않았다.

**[탐색적 해석 — 반드시 지켜야 할 표현 규칙]**

- 두 모델의 95% CI가 겹친다는 사실만으로 "두 모델이 통계적으로 동급"이라고 단정하지 않음. GBM과 LR은 **같은 3,640건에 대한 paired 예측**이므로, 엄밀히 비교하려면 McNemar test나 paired bootstrap처럼 두 모델의 판정을 짝지어 검정해야 함.
- 현재 평가 로그에는 행 단위 paired prediction이 저장되어 있지 않아 이 검정은 **아직 수행하지 않았음**. `m1_features_validation.parquet`과 두 checkpoint가 남아 있으므로 필요 시 재현해서 검정 가능.
- 이후 본 문서(및 발표 자료)에서 이 결과를 인용할 때 "CI가 겹치므로 두 모델은 동일하다"는 표현은 사용하지 않는다.

이하 3~7절의 오류 분석은 **LR 모델의 Validation 예측 178건 오분류**를 대상으로 함 (`m1_val_error_rows.parquet`, `m1_val_error_analysis.json`).

---

## 3. 오류 분석 방법

**[확정된 관찰]**

- 대상: 공식 Validation 3,640건 중 LR이 틀린 178건 (F→M 81건, M→F 97건).
- 검증 정책: "공식 Validation은 소수의 의미 있는 실험 비교에만 쓰고, fit·scaler 통계·역전파에는 절대 쓰지 않는다. 이 스크립트는 고정된(frozen) LR로 예측만 한다." — 이 정책이 그대로 준수됨.
- 세 갈래로 나눠 확인함:
  1. 관찰량(발화 길이 duration, 발화 조각 수 n_caller_utterances)이 부족하면 오류가 느는가.
  2. 신고자·119대원 발화의 시간적 겹침(overlap)이 크면 오류가 느는가.
  3. 52차원 acoustic feature 중 오분류 샘플이 정답 샘플과 다르게 분포하는 축이 있는가.
- 세 갈래 모두 quintile(5분위) 구간별 오류율과, 정답/오답 그룹 간 기술통계(mean/median/p10/p90 또는 Cohen's d)로 비교함.
- 원본 분석 스크립트 자체가 "가설 간 우선순위를 자동으로 매기지 않으며, 사람이 각 블록을 보고 다음 실험을 판단한다"고 명시하고 있음. 본 문서도 이 원칙을 따름.

---

## 4. duration / utterance count / overlap 결과

**[확정된 관찰]**

| 축 | Q1 오류율 | Q5 오류율 | Q1−Q5 차이 |
|---|---|---|---|
| duration (caller union, sec) | 6.04% (Q1, 최단) | 4.40% (Q5, 최장) | 1.65%p |
| n_caller_utterances | 4.12% (Q1, 최소) | 5.88% (Q5, 최대) | −1.77%p (방향 반대) |
| overlap ratio | 5.08% (Q1, 최소) | 5.49% (Q5, 최대) | 0.41%p |

- "짧고 발화 수도 적은(short_and_few)" 조합(n=1,506)의 오류율은 4.38%로, 전체 평균 오류율(4.89%)보다 오히려 낮음.
- overlap=0 구간(n=137)의 오류율은 2.92%로 낮게 나타나지만, 원본 로그에 "Training 감사에서 대부분 overlap>0으로 나와 이 구간 비교는 참고용이며 가설 순위 판단에 쓰지 않는다"고 명시되어 있음.

**[탐색적 해석]**

- duration, n_caller_utterances, overlap 세 축 모두 quintile 간 오류율 차이가 최대 1.8%p 수준으로 작고, duration과 n_caller_utterances는 방향성마저 서로 반대로 나타남. 즉 **단조적인(monotonic) 관계가 뚜렷하지 않음.**
- 이 결과는 "발화가 짧거나 적거나 겹침이 크면 오분류가 늘어난다"는 가설을 강하게 뒷받침하지 않는다. **"세 축이 주된 오류 원인이라는 근거는 약하다"** 정도로 정리하는 것이 데이터에 맞는 표현이며, 세 가설을 완전히 기각한다고 단정하지도 않는다 (표본 수, quintile 구간 설정 방식에 따라 결과가 달라질 여지는 있음).

---

## 5. 52차원 acoustic feature 탐색 결과

**[확정된 관찰]**

- LR 계수 절댓값 상위 15개 컬럼: `mfcc_std_19`, `mfcc_mean_16`, `mfcc_mean_15`, `mfcc_mean_6`, `mfcc_mean_17`, `mfcc_mean_18`, `mfcc_std_3`, `mfcc_std_11`, `mfcc_mean_8`, `spectral_centroid_std`, `mfcc_mean_19`, `mfcc_std_7`, `mfcc_std_16`, `mfcc_std_9`, `spectral_rolloff_std`. — 이 중 `zcr_mean`, `spectral_centroid_mean`, `mfcc_mean_0~3`처럼 피치·성별 음향 특성과 연관될 가능성이 있는 간접 spectral 지표로 분류된 컬럼은 포함되지 않음.
- 오분류 vs 정답 샘플의 Cohen's d(동일 실제 성별 내부 비교)를 계산한 결과, 다음과 같이 |d| > 1인 축이 다수 관찰됨.
  - F 오류(n=81): `mfcc_std_19` d=−1.24, `mfcc_std_17` d=−1.15, `mfcc_std_16` d=−1.10
  - M 오류(n=97): `mfcc_std_17` d=+1.83, `mfcc_std_16` d=+1.69, `mfcc_std_19` d=+1.65, `mfcc_std_18` d=+1.37, `mfcc_std_13` d=+1.34
- "간접 spectral 지표 6개(zcr_mean, spectral_centroid_mean, mfcc_mean_0~3)" 기준으로, 오분류 샘플의 평균이 반대 성별 평균에 더 가까운 지표 개수는 F 오류에서 6개 중 2개, M 오류에서 6개 중 3개.

**[탐색적 해석 — 확정적 표현 금지]**

- 이 Cohen's d 상위 목록은 **52개 feature를 사후적으로(post-hoc) 전부 탐색한 뒤 값이 큰 축만 골라 보여준 것**이다. 따라서 다중비교(multiple comparisons) 문제와 post-hoc selection bias의 영향을 받을 수 있으며, 이 목록만으로 어떤 축이 "진짜" 오류 원인인지 판단할 수 없음.
- 오분류 표본 수 자체가 F→M 81건, M→F 97건으로 크지 않아, Cohen's d 추정치에도 상당한 불확실성(표준오차)이 있음.
- 따라서 **"고차 MFCC가 오류의 원인이다"** 또는 **"feature confusion이 주원인이다"**처럼 확정적으로 서술하지 않는다. 이 절의 결과는 **"일부 MFCC 축에서 탐색적 패턴이 관찰되었다"** 수준으로만 기술한다.
- 피치·성별 음향 특성과 연관될 가능성이 있는 간접 spectral 지표가 반대 성별에 더 가깝다는 결과도 F 6개 중 2개, M 6개 중 3개로 과반에 미치지 못하거나 절반 수준이라, "오분류된 목소리는 실제로 피치가 애매하다"는 가설을 확정적으로 지지하지 않음. (이 지표들은 실제 F0의 proxy로 측정된 것이 아니며, 그렇게 해석하지 않음.)
- **이 절의 결과는 다음 실험의 가설을 만드는 용도로만 사용하며, 별도 검증(예: pre-registered 재확인, 독립 데이터에서의 재현) 없이 확정적 결론이나 발표 핵심 주장으로 사용하지 않는다.**

---

## 6. 해석상의 한계

**[확정된 관찰 / 방법론적 한계 — 사실 그대로 기술]**

1. **다중비교·post-hoc selection bias**: 52개 feature 중 사후적으로 큰 값만 뽑아 Cohen's d를 제시했으므로, 우연히 크게 나온 축이 섞여 있을 수 있음.
2. **표본 크기**: 오분류 표본이 F→M 81건, M→F 97건으로 작아, 관찰된 효과크기(Cohen's d)의 불확실성이 큼.
3. **Paired 비교 미실시**: GBM과 LR을 같은 3,640건에 대해 짝지어 검정(McNemar 등)하지 않았으므로, 두 모델 성능 차이의 통계적 유의성은 아직 확인되지 않음.
4. **quintile 분석의 한계**: duration/utterance/overlap 분석은 5분위 구간 오류율 비교로, 비선형 관계나 특정 구간 내 국소적 패턴은 놓칠 수 있음.
5. **leakage 부재는 "증명"되지 않음**: 내부 holdout과 공식 Validation 결과가 근접하다는 것은 leakage 미관찰 정황이지, leakage가 없다는 증명이 아님. Training-only fit 강제 장치(SplitContaminationError)가 실질적 방어선임.

---

## 7. 후속 실험 후보 (F0 등)

**[후속 가설 — 아직 증거 아님]**

- 4절 결과로 duration/발화 수/overlap이 주된 오류 원인이라는 근거가 약해진 상태이고, 5절 결과는 방법론적 한계 때문에 확정적 결론으로 쓸 수 없음. 이 두 가지를 종합하면, 다음으로 시도할 후보 중 하나로 **"기존 52-d와는 다른 종류의 음성 정보를 추가하는 방향"**이 고려 대상에 오르며, 그 후보 중 F0(기본 주파수, pitch)의 상대적 우선순위가 올라간 상태라고 말할 수 있다.
- 다만 이는 **"F0가 답이다" 혹은 "F0를 추가해야 한다"는 결론이 아니다.** 이번 분석은 F0의 필요성을 증명한 것이 아니라, 다른 후보들(duration 계열)의 근거가 약해지면서 상대적으로 F0의 순위가 올라간 것일 뿐이다.
- 그 외 후속 가설 후보 (아직 우선순위 미확정):
  - 52개 feature에 대한 사전 등록(pre-registered) 방식의 재검증 — 단, 이번 분석에 사용된 Validation을 다시 보거나 같은 데이터를 재추출하는 것은 독립 검증으로 간주하지 않음. 이번 분석에 사용되지 않은 독립 평가 표본이 확보될 경우, 5절 Cohen's d 상위 축이 그 표본에서도 재현되는지 확인한다.
  - GBM vs LR paired 비교(McNemar test / paired bootstrap) — 두 모델 차이의 유의성 확정.
  - F→M, M→F 오분류 각각을 개별 화자 단위로 다시 들여다보는 정성 분석(소수의 이례적 화자가 오류를 몰아서 만드는지 확인).

---

## 8. 현재 M1 상태 — 잠정 종료

- **현재까지 완료**: baseline 수립(1절), 공식 Validation 평가 및 모델 선정(2절), 오류 분석(3~5절), 방법론적 한계 정리(6절), 후속 가설 도출(7절).
- **EXP-002는 아직 착수하지 않음.** 본 문서 작성 시점까지는 분석·기록만 진행하고, 새 실험(F0 추가 등)은 실행하지 않았다.
- 기존 M1 baseline 코드(`scripts/eval_mission1_validation.py`, `scripts/bootstrap_ci_mission1.py` 등)와 실험 결과 파일(`m1_exp_log.json`, `m1_val_eval.json`, `m1_val_bootstrap_ci.json`, `m1_val_error_analysis.json`, `m1_val_error_rows.parquet`)은 이번 문서화 과정에서 **수정하지 않았음.**
- 따라서 M1은 "LR 채택 후보 + Validation 95.11% + 오류 분석 완료 + 후속 가설(F0 등) 도출" 상태로 **잠정 종료**하며, 위 후속 가설 중 하나를 EXP-002로 착수할지는 팀 논의 후 별도로 결정한다.
