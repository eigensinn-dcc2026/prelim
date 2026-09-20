# EXPERIMENT_LOG.md — 실험 기록

**돌린 실험은 전부 여기에 기록한다. 기록 없는 실험 결과는 성과로 인정하지 않는다.**
이 로그가 있어야 (1) 재현이 가능하고, (2) 발표에서 성능 향상 과정을 보여줄 수 있다.

> **범위**: 이 문서는 **미션 2(119 신고 통화 발화 단위 화자 분류: 대원=0 / 신고자=1)** 실험만
> 다룬다. 미션 1/3은 별도 로그가 필요하면 각자 만든다.
>
> **현재 상태(2026-09-20 갱신, 원본 근거: `reports/mission2/m2_progress_narrative.md`
> 2026-09-19 요약 + git 커밋 로그 대조)**:
> - **L1(LightGBM, 53차원 통계 피처) 공식 후보 확정**: 공식 Validation(3,640통화, 111,947발화)
>   Accuracy **86.40%** (M2-EXP-017). 내부 split 기준 튜닝 결과는 86.77%(M2-EXP-015).
>   규정 위반 피처(call-relative) 제거 완료, 라벨 스왑 6건 반영 완료.
> - **L2(Mel-spectrogram + CNN, SpeakerCNNLarge) 내부 검증 91.22%로 L1보다 우수하지만
>   2026-09-18부로 잠정 보류** (DECISIONS [004]). 공식 Validation 미검증, `main` 미병합.
> - 채점 지표는 **accuracy 단일 지표**로 확정(2026-09-18) — 대원/신고자 recall 격차(8~9%p대)는
>   참고 지표일 뿐 채점에 반영되지 않는다. 격차를 줄이려는 threshold/class_weight 조정은
>   전부 기각됨(M2-EXP-016, DECISIONS [006]).
> - 다음 결정 포인트: L1을 어디까지 다듬고 나서 L2를 재개/공식 검증/병합할지 — **아직 미정**.

작성 방법: 새 실험은 고유 번호(`M2-EXP-XXX`)로 추가한다. **번호는 절대 재사용하지 않는다**
(과거 `m2_exp_log.json`과 `m2_exp_log_tuned.json`이 둘 다 `M2-EXP-001`을 썼던 것이 실수 사례 —
아래 M2-EXP-001 항목의 "주의" 참조). 실행일과 검증·채택일이 다르면 둘 다 기록한다.
날짜를 특정할 수 없으면 "미상"이라고 정직하게 쓰고 추정 범위만 남긴다 — 지어내지 않는다.

---

## 평가 지표

| 지표 | 의미 | 비고 |
|---|---|---|
| `Accuracy` | 발화 단위 대원/신고자 이진 분류 정확도 | **대회 공식 채점 지표(단일)** — DECISIONS [006] |
| `대원(0) recall` | 실제 대원 발화 중 대원으로 맞춘 비율 | 항상 신고자보다 낮음(구조적, §해석 참조) |
| `신고자(1) recall` | 실제 신고자 발화 중 신고자로 맞춘 비율 | 항상 대원보다 높음 |
| `격차` | 신고자 recall − 대원 recall | **채점에 미반영** — 참고/오류분석 전용 지표. 이 지표를 올리려고 accuracy를 깎는 시도는 하지 않는다(DECISIONS [006]) |
| `Leakage gap` | random split accuracy − GroupSplit(call_id) accuracy | 크면 화자 암기(같은 통화 발화가 train/val에 동시에 있어 생기는 부풀림) 의심 |

> **입력 제약(대회 규정, 위반 시 실격)** — DECISIONS [001]: 분류 입력은 해당 발화 구간
> (`startAt`~`endAt`) 하나뿐이다. 같은 통화의 다른 발화에서 나온 정보는 순서·형태 무관 전부 금지
> (평균·표준편차 등 집계값 포함). 발화 자신의 `duration_sec`만 예외적으로 허용된다.
> 새 피처를 추가할 때마다 이 규정 위반 여부를 **머지 전에** 확인한다(M2-EXP-007 참조).

## 템플릿 (복사해서 사용)

```
### M2-EXP-XXX (YYYY-MM-DD) — 한 줄 제목

- 유형/상태: (`DATA`/`AUDIT`/`EVAL`/`TRAIN`/`SMOKE`; `planned`/`completed`/`invalidated`/`superseded`)
- 목적/가설: (이 실험으로 무엇을 확인하려 하는가)
- 데이터 버전: (통화 수, 발화 수, 서울/Training or Validation 등)
- split 방식: (call_id 기준 GroupShuffleSplit, seed=42, 8:2 등)
- 모델: (LightGBM, SpeakerCNN 등)
- 코드/환경: (Git commit, 브랜치, Colab/로컬, CPU/GPU)
- 하이퍼파라미터: (num_leaves, learning_rate, epochs 등)
- 모델 선택 기준: (validation 지표·체크포인트 선택 규칙)
- 채점 지표 사용: (내부 split / 공식 Validation 사용 여부, 공식 Validation은 회차 기록 필수)
- 평가 지표 결과:
    - Accuracy:
    - 대원(0) recall:
    - 신고자(1) recall:
    - 격차:
    - Leakage gap:
- 결과 요약: (숫자로 한두 줄)
- 해석: (왜 이런 결과가 나왔는가, 무엇을 알게 됐는가)
- 다음 실험: (이 결과를 바탕으로 다음에 무엇을 할 것인가)
- 산출물 경로: (리포트/코드/체크포인트 위치)
- 보존 상태: (Git 추적 여부, 브랜치, 병합 여부)
- 한계: (표본, 커버리지, 미검증 조건 등)
```

---

## 실험 기록 (최신순)

### M2-EXP-017 (2026-09-18) — L1 공식 Validation 최초 검증

- 유형/상태: `EVAL` / `completed` (재학습 없음)
- 목적/가설: 그동안 모든 수치가 내부 split(Training 20%)이었다. 대회가 실제로 채점에 쓰는
  공식 Validation에서도 성능이 유지되는지, 재학습 없이 저장된 체크포인트로 **1회만** 확인한다.
- 데이터 버전: Seoul Validation 전체 3,640통화, 111,947발화 (추출 실패 0건).
  데이터 위치 주의: OneDrive `대학부 데이터/` 폴더는 Validation의 12%(424/3,640)만 있는
  불완전 사본 — 전체는 `C:\dcc_data_local_full\Validation`에 있었다.
- split 방식: 해당 없음 (공식 Validation 전체 평가, split 아님)
- 모델: LightGBM 튜닝 체크포인트(`checkpoints/mission2/m2_baseline_lgbm_tuned.pkl`, M2-EXP-015와 동일 가중치)
- 코드/환경: `dcc_data/mission2_features.py`에 `select_seoul_validation_rows`/
  `build_seoul_validation_dataset` 추가(기존 Training용 함수와 동일 구조, VALIDATION 스플릿으로 게이트)
- 모델 선택 기준: 해당 없음 (기존 체크포인트 평가만)
- 채점 지표 사용: **공식 Validation 1회 사용**(DECISIONS [007] 원칙의 첫 적용 사례)
- 평가 지표 결과 (n=111,947):
    - Accuracy: **86.40%**
    - 대원(0) recall: 81.68%
    - 신고자(1) recall: 90.77%
    - 격차: 9.09%p
    - 참고: 내부 split(n=164,983) 대비 Accuracy −0.37%p, 격차는 9.12%p→9.09%p로 거의 동일
- 결과 요약: 내부 split과 공식 Validation 차이가 −0.37%p뿐 — 과적합으로 부풀려진 수치가 아니라
  실제 일반화되는 성능. 화자 비대칭 패턴도 그대로 재현됨.
- 해석: 대원 recall < 신고자 recall 격차가 데이터를 바꿔도 사라지지 않는 구조적 특성임이
  재확인됨. yeonwoo의 MFCC40+로지스틱회귀(공식 Validation 81.31%)보다 +5.09%p 높음.
- 다음 실험: 없음(추가 튜닝 시 반드시 Training 내부 split으로 먼저 하고, 최종 후보가 바뀔 때만
  공식 Validation을 다시 봄 — DECISIONS [007]).
- 산출물 경로: `reports/mission2/m2_l1_official_validation_results.md`,
  `m2_l1_official_validation_result.json`
- 보존 상태: `gwanghee/mission2-l1-model-comparison` 브랜치, `main` 미병합
- 한계: 예비로 424통화만 돌렸던 결과(86.24%)와 0.16%p 차이 — 작은 표본도 이미 안정적이었음.

### M2-EXP-016 (2026-09-18) — 대원 recall 밀어보기 (threshold / class_weight 스윕)

- 유형/상태: `EVAL` / `completed`, **결론: 기각(적용 안 함)**
- 목적/가설: M2-EXP-015 최종 후보(격차 9.12%p)의 판정 기준(threshold)이나 클래스 가중치를
  조정하면 대원 recall을 올릴 수 있는지 확인한다.
- 데이터 버전/split: M2-EXP-015와 동일(내부 split, n=164,983)
- 모델: LightGBM 튜닝 (재학습 없이 threshold만 이동 / class_weight는 재학습)
- 코드/환경: `scripts/mission2_l1_push_daewon_recall.py`
- 주의(실수 기록): 1차 실행에서 threshold를 0.5→0.25로 **잘못된 방향**으로 낮춰서 의도와
  반대로 신고자 recall만 더 오르는 결과가 나왔다. 방향을 0.5→0.75로 바로잡아 재실행함.
- 채점 지표 사용: 내부 split만, 공식 Validation 미사용
- 평가 지표 결과 (threshold 스윕, 기본 0.5 대비):
    - threshold 0.55: Accuracy 86.57%(−0.20%p), 대원 recall 84.97%(+2.93%p), 신고자 recall 88.06%(−3.10%p), 격차 +3.09%p
    - threshold 0.60: Accuracy 85.78%, 대원 recall 90.21%, 신고자 recall 81.67%, 격차 **역전(−8.54%p)**
    - class_weight=1.2: Accuracy 86.69%(−0.08%p), 대원 recall 84.45%(+2.41%p), 격차 +4.32%p
    - class_weight=1.3: Accuracy 86.34%, 격차 **역전(−2.55%p)**
- 결과 요약: 두 방법 다 대원 recall을 올리는 만큼 신고자 recall이 거의 대칭적으로(제로섬에 가깝게)
  깎이고, 전체 Accuracy는 항상 손해를 본다.
- 해석: 대회 채점이 accuracy 단일 지표로 확정됨(2026-09-18, DECISIONS [006]) — recall 격차는
  채점과 무관하므로 이 트레이드오프를 감수할 이유가 없다. `m2_error_analysis.md`에서 확인된
  구조적 원인(발화 겹침, 0.5~1초 구간, 대원 발화가 통계적으로 더 짧음)은 그대로 남아 있다 —
  "모델이 똑똑해진 것"이 아니라 "어느 쪽 오답을 봐줄지 옮긴 것"뿐.
- 다음 실험: 없음. 기본 threshold=0.5, class_weight 없음(86.77%)을 그대로 최종 L1 후보로 유지.
- 산출물 경로: `reports/mission2/m2_l1_daewon_recall_push_results.md`,
  `m2_l1_daewon_recall_push.json`
- 보존 상태: `gwanghee/mission2-l1-model-comparison` 브랜치, `main` 미병합
- 한계: 없음(결론이 명확한 실험).

### M2-EXP-015 (2026-09-18) — L1 모델 계열 공정 비교 + LightGBM 최종 튜닝 (53차원 clean)

- 유형/상태: `EVAL`+`TRAIN` / `completed`
- 목적/가설: 규정 위반 피처(call-relative, M2-EXP-007)를 제거한 순정 53차원 피처에서
  GaussianNB/로지스틱회귀/RandomForest/XGBoost/LightGBM 중 무엇이 최선인지, 그리고 이전에
  64차원 기준으로 검증됐던 튜닝(M2-EXP-008, 84.59%→86.9%)이 순정 피처에서도 유효한지 확인한다.
- 데이터 버전: `data/local_cache/m2_features_train_local_full.parquet`, 826,333발화(53차원)
- split 방식: `make_group_split`(call_id 기준 GroupShuffleSplit, seed=42, test_size=0.2) —
  기준선(LightGBM)과 완전히 동일한 split. train 661,350 / val 164,983
  (`m2_error_analysis.md`의 val 건수와 정확히 일치 — split 재현 확인됨)
- 모델: LightGBM(기준선·n=300·튜닝), GaussianNB, 로지스틱회귀, RandomForest(n=300), XGBoost(n=300)
- 코드/환경: `scripts/compare_mission2_l1_models.py`, 브랜치 `gwanghee/mission2-l1-model-comparison`
- 하이퍼파라미터(최종 채택): LightGBM `num_leaves=255, learning_rate=0.02, n_estimators=1150`, early stopping(`best_iteration=1150` — 끝까지 다 씀, 더 큰 n_estimators면 소폭 더 오를 여지 있으나 우선순위 낮음)
- 모델 선택 기준: 동일 split Accuracy 최댓값. 1차 실행에서 XGBoost/RandomForest만 `n_estimators=300`,
  LightGBM은 기본값(100그루)으로 돌려 불공정했던 것을 발견 → 트리 개수를 300으로 통일해 재비교.
- 채점 지표 사용: 내부 split만, 공식 Validation 미사용(선택 단계이므로 DECISIONS [007] 원칙상 미사용)
- 평가 지표 결과 (n=164,983):

  | 모델 | Accuracy | 대원 recall | 신고자 recall | 격차 |
  |---|---|---|---|---|
  | LightGBM 기준선(n=100 기본값) | 84.59% | 79.75% | 89.08% | 9.33%p |
  | GaussianNB | 69.70% | 80.96% | 59.26% | −21.70%p(역전) |
  | 로지스틱회귀 | 83.24% | 79.53% | 86.68% | 7.14%p |
  | RandomForest(n=300) | 84.05% | 81.17% | 86.72% | 5.55%p |
  | LightGBM(n=300) | 86.00% | 81.36% | 90.30% | 8.94%p |
  | XGBoost(n=300) | 86.47% | 82.17% | 90.44% | 8.27%p |
  | **LightGBM 튜닝(최종)** | **86.77%** | 82.04% | **91.16%** | 9.12%p |

- 결과 요약: "XGBoost가 기준선보다 +1.88%p 낫다"의 75%는 모델 차이가 아니라 트리 개수 차이였다
  (LightGBM만 n=300으로 올려도 +1.41%p). 트리 개수를 통일하면 XGBoost 우위는 +0.47%p로 줄고,
  LightGBM을 제대로 튜닝하면 그 XGBoost를 다시 앞선다(+0.30%p).
- 해석: 부스팅 계열(LightGBM/XGBoost)이 이 53차원 피처의 실질적 상한이고, 그 안에서는
  "모델 선택"보다 "용량(트리 개수/깊이)"과 "튜닝 여부"가 더 크게 작용한다. RandomForest는
  트리 수를 맞춰도 부스팅에 못 미치며 학습 시간이 가장 길다(14분, LightGBM의 약 80배).
  M2-EXP-008에서 64차원(규정위반 피처 포함) 기준으로 검증됐던 튜닝 조합이 순정 53차원에서도
  거의 동일한 개선폭(84.59%→86.77%, 그때는 →86.9%)으로 재현됨 — 튜닝 효과가 피처 오염과
  무관하게 진짜였음을 확인.
- 다음 실험: 이 설정(LightGBM 튜닝)을 `dcc_data/mission2_train.py`의 `_build_model()`에 반영해
  L1 기준선 자체를 갱신 → M2-EXP-016(대원 recall 밀어보기), M2-EXP-017(공식 Validation)로 이어짐.
- 산출물 경로: `reports/mission2/m2_l1_model_comparison_results.md`,
  `m2_l1_model_comparison.json`, `m2_exp_log_tuned.json`
- 보존 상태: `gwanghee/mission2-l1-model-comparison` 브랜치, `main` 미병합
- 한계: `m2_exp_log_tuned.json`은 27,030통화(train 21,624/val 5,406) 기준 수치로, 이 표의
  826,333발화(29,200통화) 전체 기준과 통화 수 집계가 다르다 — Accuracy 값(86.77%)은 동일하나
  두 파일의 모집단 표기가 정확히 일치하지 않는 점을 인지하고 재확인이 필요하다(TODO).

### M2-EXP-014 (2026-09-16) — L2 화자 격차 겨냥 실험 B: hop-length 절반 축소 (미평가)

- 유형/상태: `DATA`+`SMOKE` / **미완료 — 재학습·평가 안 함**
- 목적/가설: `HOP_LENGTH=160` 기준 짧은 발화가 스펙트로그램에서 1~2 타임스텝까지만 살아남는
  문제를, hop_length를 절반으로 줄여 프레임 수를 약 2배로 늘려서 완화할 수 있는지 확인한다.
- 데이터 버전: 스모크 테스트로 프레임 수 16→31, 76→151 증가만 확인, 별도 `--output-dir`에
  고해상도 스펙트로그램 인프라만 구축(기존 91%+ 체크포인트가 쓰는 스펙트로그램은 건드리지 않음)
- 모델: 해당 없음(재학습 안 함)
- 코드/환경: `scripts/build_mission2_spectrograms.py --hop-length` 옵션 추가,
  브랜치 `gwanghee/mission2-l2-cnn` / `mission2-l2-cnn-clean`
- 채점 지표 사용: 없음
- 평가 지표 결과: *(미실행 — 수치 없음)*
- 결과 요약: 인프라만 만들어두고 2026-09-19 기준 재학습·평가 전 상태로 L2 전체가 보류에 들어감.
- 해석: 아직 판단할 근거 없음.
- 다음 실험: L2 재개 시 가장 먼저 이 설정으로 재학습/평가부터 확인할 것.
- 산출물 경로: `reports/mission2/m2_l2_cnn_results.md` §"보류 직전 마지막 시도"
- 보존 상태: `gwanghee/mission2-l2-cnn`, `mission2-l2-cnn-clean` 브랜치, `main` 미병합, PR #17은 병합 없이 닫힘
- 한계: 평가 자체가 없으므로 효과 유무를 전혀 알 수 없음.

### M2-EXP-013 (2026-09-16) — L2 화자 격차 겨냥 실험 A: SpeakerCNNLargeWithDuration

- 유형/상태: `TRAIN`+`EVAL` / `completed`, **결론: 기각**
- 목적/가설: 3번의 MaxPool(stride 2) 이후 짧은 발화가 남기는 프레임 수가 거의 없어지는 문제를,
  발화 자신의 `duration_sec`(log1p 압축)을 classifier head에 명시적으로 추가해 보완할 수 있는지
  확인한다. (발화 자신의 속성이므로 규정 위반 아님 — DECISIONS [001] 예외 조항)
- 데이터 버전/split: M2-EXP-012와 동일(전체 29,200통화, GroupShuffleSplit seed=42)
- 모델: `SpeakerCNNLargeWithDuration`(SpeakerCNNLarge + duration 입력)
- 코드/환경: Colab GPU(T4), 브랜치 `gwanghee/mission2-l2-cnn` / `mission2-l2-cnn-clean`
- 채점 지표 사용: 내부 split만
- 평가 지표 결과: Accuracy **+0.30%p** (M2-EXP-012 91.22% 대비), 격차 8.45%p → **9.17~9.35%p로 확대**
- 결과 요약: 전체 정확도는 소폭 개선됐지만 격차는 오히려 벌어졌다.
- 해석: duration 정보 추가가 짧은 발화의 대원 편향을 줄이지 못했다 — 격차의 원인이 duration
  정보 부재가 아니라 다른 곳에 있음을 시사.
- 다음 실험: 이 방향은 폐기. 기존 91.22% 체크포인트/학습 경로는 그대로 두고 별도 클래스로만
  코드에 남김(참고용).
- 산출물 경로: `reports/mission2/m2_l2_cnn_results.md` §"보류 직전 마지막 시도"
- 보존 상태: `gwanghee/mission2-l2-cnn`, `mission2-l2-cnn-clean` 브랜치, `main` 미병합
- 한계: Colab 1회 실행 기준, 반복 seed 검증 없음.

### M2-EXP-012 (2026-09-14) — L2 CNN SpeakerCNNLarge (4 conv 블록, 421,954 파라미터)

- 유형/상태: `TRAIN` / `completed` (내부 검증), **공식 Validation 미검증**
- 목적/가설: SpeakerCNN 소형(M2-EXP-010)이 89.71%에서 수렴한 뒤, "모델 용량이 부족했다"는
  가설을 검증하기 위해 더 큰 모델로 처음부터 재학습한다.
- 데이터 버전: 전체 29,200통화(라벨 수정 2건 추가 반영, M2-EXP-011)
- split 방식: call_id 기준 GroupShuffleSplit(seed=42, test_size=0.2) — L1과 동일한 split
- 모델: `SpeakerCNNLarge` (conv 블록 4개, 421,954 파라미터, 소형 대비 16배)
- 코드/환경: Colab GPU(T4), `scripts/train_mission2_cnn.py --model large --resume`
- 하이퍼파라미터: 12에포크 시점 LR=2.5e-4(아직 감쇠 전, 학습 계속 중이었음)
- 모델 선택 기준: val accuracy 최댓값(수렴 확인 전까지 계속 학습)
- 채점 지표 사용: 내부 split만, 공식 Validation 미사용
- 평가 지표 결과 (12에포크 시점, 아직 수렴 전):
    - Accuracy: **91.02%** (목표 90% 최초 돌파), 15에포크 90.83%로 피크 근처 유지
    - 대원 recall: 87.24% / 신고자 recall: 94.51% / 격차: **7.27%p**(소형 모델 7.66%p보다 개선)
    - 이후 추가 학습으로 최종 **91.22%**까지 수렴(정확한 최종 에포크 수는 로컬 리포트
      스냅샷에 기록되지 않음 — 팀 progress narrative 요약치, TODO: 정확한 최종 로그 확보)
- 결과 요약: 목표였던 90%를 처음 돌파. 모델을 키우자 정확도와 격차가 함께 개선됨 —
  소형 모델과 반대로, "더 학습"과 "더 큰 모델" 둘 다 격차를 줄이는 방향으로 작동.
- 해석: 요약통계(L1)가 버리는 정보(발화 내부 시간-주파수 구조)가 화자 비대칭의 원인 중
  하나였을 가능성을 뒷받침. 겹침 구간 정확도도 L1(79.90%) 대비 L2(87.04%)로 크게 개선.
- 다음 실험: SpeakerCNNLarge 수렴 확인 후 오류 분석 재실행 → M2-EXP-013, M2-EXP-014.
- 산출물 경로: `reports/mission2/m2_l2_cnn_results.md`
- 보존 상태: `gwanghee/mission2-l2-cnn`, `mission2-l2-cnn-clean` 브랜치, `main` 미병합(PR #17 닫힘)
- 한계: 공식 Validation 미검증. 최종 91.22% 수렴 시점의 상세 로그(에포크·LR)가 committed 리포트에
  없어 재현성이 M2-EXP-017(L1 공식 검증)만큼 확보되지 않음 — L2 재개 시 최우선 보완 필요.

### M2-EXP-011 (2026-09-14) — 라벨 불연속 자동 탐지기 + 부분 뒤집기 2건 확정

- 유형/상태: `AUDIT`+`DATA` / `completed`
- 목적/가설: M2-EXP-004에서 발견한 "통화 중간부터 라벨이 뒤바뀌는" 패턴(부분 뒤집기)을
  수동 확인이 아니라 모델 예측 패턴으로 체계적으로 찾아내는 도구를 만든다.
- 데이터 버전: held-out val 통화 5,840건 전체 스캔 (train은 모델이 그 라벨로 학습하므로 이 방법이
  안 통함 — 한계로 명시)
- 모델: 해당 없음(기존 L1/L2 예측 결과 재분석)
- 코드/환경: `dcc_data.mission2_analysis.best_flip_point()` /
  `find_label_discontinuity_candidates()`, `scripts/find_mission2_label_discontinuities.py`
- 채점 지표 사용: 없음(데이터 감사)
- 평가 지표 결과:
    - `651e4c754428de718c773b56`: 원래 28.57%(전체 뒤집기 적용 후에도) → utt_idx 6부터 뒤집으면 **100.00%**
    - `651e4e712e98ee712096903c`: 원래 27.78% → utt_idx 5부터 뒤집으면 **100.00%** (신규 발견)
- 결과 요약: 첫 번째 건은 원래 "전체 뒤집기" 목록에 있었으나 실제로는 통화 중간(utt_idx 6)부터만
  뒤바뀐 것으로 재확인 — 전체 뒤집기 수정이 오히려 앞 6개 발화를 망가뜨리고 있었음.
- 해석: 모델은 두 목소리를 거의 완벽히 구분하고 있었는데(0.87~1.0 확신도), 원본 라벨 자체가
  중간에 끊겨 있었다 — 데이터 문제였지 모델/feature 한계가 아니었음.
  `651e537c201879a4721c618c`(30~40번 구간만 오답)처럼 "구간 중간만" 틀린 케이스는 이 알고리즘이
  못 잡음(어느 지점부터 끝까지 뒤집기만 지원) — 실제 청음 필요, 보류.
- 다음 실험: 전체 데이터 루트에서 재실행해 나머지 ~9,200통화(zip 기반 스캔의 사각지대) 커버,
  cross-validation으로 train 쪽까지 확장하는 방안 검토(M2-EXP-004 한계와 동일).
- 산출물 경로: `reports/mission2/m2_label_swap_audit.md` §"부분 뒤집기" 이하
- 보존 상태: `gwanghee/mission2-l2-cnn` 관련 브랜치 커밋으로 반영, `main` 미병합
- 한계: val만 스캔 가능(전체 29,200건 중 5,840건, 20%). train 쪽 라벨 스왑은 여전히 미확인.

### M2-EXP-010 (2026-09-14) — L2 CNN 전체 데이터(29,200통화) SpeakerCNN 소형

- 유형/상태: `TRAIN` / `completed`
- 목적/가설: M2-EXP-009(3,000통화 파일럿)에서 "더 학습해도 격차가 안 줄었던" 결과가 데이터
  규모 문제였는지, 전체 데이터로 확인한다.
- 데이터 버전: 전체 29,200통화 (라벨 수정 2건 반영, M2-EXP-004)
- split 방식: call_id 기준 GroupShuffleSplit(seed=42, test_size=0.2) — L1과 동일
- 모델: `SpeakerCNN`(소형, conv 블록 3개, 25.7k 파라미터)
- 코드/환경: Colab GPU(T4), `scripts/build_mission2_spectrograms.py` + `scripts/train_mission2_cnn.py`
- 채점 지표 사용: 내부 split만
- 평가 지표 결과:

  | 누적 에포크 | Accuracy | 대원 recall | 신고자 recall | 격차 |
  |---|---|---|---|---|
  | 5 | 84.13% | 69.79% | 97.36% | 27.57%p |
  | 10 | 87.90% | 81.34% | 93.95% | 12.61%p |
  | 20 | 89.27% | 85.00% | 93.21% | 8.21%p |
  | 35 | 89.60% | 85.20% | 93.66% | 8.46%p |
  | **45(수렴, LR→6.25e-05)** | **89.71%** | 85.72% | 93.38% | **7.66%p** |

- 결과 요약: 데이터를 3,000→29,200(약 9.7배)으로 늘리자 정확도뿐 아니라 격차도 계속 줄었다 —
  3,000통화 버전(M2-EXP-009)과 정반대 방향.
- 해석: 3,000통화 규모에서는 "더 학습"이 격차를 못 줄였지만, 데이터 규모 자체가 핵심
  레버였음이 확인됨. 36~45에포크 구간 val_acc가 89.4~89.7% 좁은 밴드에서 수렴 확인.
- 다음 실험: 모델을 키운 SpeakerCNNLarge로 재학습 → M2-EXP-012.
- 산출물 경로: `reports/mission2/m2_l2_cnn_results.md`
- 보존 상태: `gwanghee/mission2-l2-cnn` 브랜치, `main` 미병합
- 한계: 공식 Validation 미검증.

### M2-EXP-009 (2026-09-10) — L2 CNN 파이프라인 구축 + 3,000통화 파일럿

- 유형/상태: `TRAIN` / `completed` (파일럿, 최종 결론 아님)
- 목적/가설: L1(음향 통계+GBM)이 84.59%에서 정체된 후, mel-spectrogram을 CNN에 직접 학습시켜
  L1이 놓치는 시간-주파수 구조를 잡을 수 있는지 파일럿으로 확인한다.
- 데이터 버전: 3,000통화(L1 M2-EXP-001과 동일 표본, train 73,374 / val 18,134)
- split 방식: call_id 기준 GroupShuffleSplit(seed=42) — L1과 동일
- 모델: `SpeakerCNN`(소형, conv 블록 3개)
- 코드/환경: Colab GPU(T4), `scripts/build_mission2_spectrograms.py`, `scripts/train_mission2_cnn.py`
- 채점 지표 사용: 내부 split만
- 평가 지표 결과:

  | 누적 에포크 | Accuracy | 대원 recall | 신고자 recall | 격차 |
  |---|---|---|---|---|
  | 20 | 86.11% | 83.56% | 88.51% | 4.95%p |
  | 40(수렴, LR→6.25e-06) | 86.56% | 81.80% | 91.00% | 9.20%p |

- 결과 요약: 같은 3,000통화 기준 L1 최고 기록(85.6~85.9%)을 이미 넘었지만, 더 학습해도 격차는
  줄지 않고 오히려 커졌다(4.95%p→9.20%p).
- 해석: 이 표본 크기에서는 "더 학습"이 격차를 줄이지 못한다 — 데이터 규모 자체가 핵심 변수일
  가능성 제기 → M2-EXP-010에서 전체 데이터로 검증.
- 다음 실험: 전체 29,200통화로 확장.
- 산출물 경로: `reports/mission2/m2_l2_cnn_results.md`
- 보존 상태: **프로세스 사고 발생** — 이 CNN 작업 브랜치(`mission2-l2-cnn`)가 `main`이 아니라
  아직 실험 중이던 GBM 피처 브랜치(`mission2-feature-fixes`) 위에서 이어붙여져, CNN 작업과
  피처 작업 커밋이 뒤섞였다. `main` 자체는 오염되지 않았음을 확인했으나(PR 베이스가 `main`이
  아니었음), 재발 방지를 위해 DECISIONS [003](항상 main에서 분기)을 수립하고 이후
  `mission2-l2-cnn-clean`/`mission2-feature-fixes-clean`으로 재구성함.
- 한계: 파일럿 표본(3,000통화)이라 최종 결론 아님.

### M2-EXP-008 (날짜 미상 — 2026-09-10~09-15 사이로 추정, 별도 committed 리포트 없음) — LightGBM 하이퍼파라미터 재튜닝 (64차원)

- 유형/상태: `TRAIN` / `completed`, **이후 M2-EXP-015에서 53차원으로 재검증**
- 목적/가설: 기본값(`num_leaves=31, n_estimators=100`)이 826k행 규모에서 과소적합 상태로
  의심됨 — 데이터 규모에 맞게 재튜닝하면 개선되는지 확인한다.
- 데이터 버전: 전체 29,200통화, **64차원 피처**(53차원 순정 + call-relative 등 규정 위반 11개
  포함 시점 — M2-EXP-007 참조). 이 시점 이후 규정 위반이 확인되어 피처가 되돌려졌으므로
  **이 결과 자체는 오염된 피처 기준**이라는 점에 유의.
- split 방식: call_id 기준 GroupShuffleSplit(seed=42, test_size=0.2)
- 모델: LightGBM
- 코드/환경: 미상(별도 committed 스크립트/리포트 없이 진행됨 — TODO: 실행 코드/커밋 확인 필요)
- 하이퍼파라미터: `num_leaves=255, learning_rate=0.02, n_estimators≈1150`, early stopping
- 채점 지표 사용: 내부 split만
- 평가 지표 결과: Accuracy **84.59% → 86.9%** (+2.3%p, 이 시점까지 가장 큰 단일 개선)
- 결과 요약: 하이퍼파라미터 재튜닝이 지금까지 시도 중 가장 큰 개선폭을 보임.
- 해석: 기본 파라미터가 데이터 규모(89만 발화)에 비해 과소적합이었다는 가설이 맞았음.
- 다음 실험: 규정 위반 피처(call-relative) 제거 후 순정 53차원으로 이 튜닝이 여전히 유효한지
  재검증 필요 → M2-EXP-015 (84.59%→86.77%로 거의 동일한 개선폭 재확인됨).
- 산출물 경로: `[[project-mission2-confirmed-wins]]`(팀 메모)에만 기록, 이 저장소에는 별도
  committed 리포트 없음 — **TODO: 원 실행 스크립트/커밋/날짜를 찾아 이 항목을 보강할 것.**
- 보존 상태: 불명확 (TODO)
- 한계: **이 실험 자체는 규정 위반 피처가 섞인 채로 실행됨** — 86.9%라는 숫자를 피처 정합성
  검증 없이 공식 수치로 인용하지 않는다. 공식 수치는 M2-EXP-015의 86.77%(순정 53차원).

### M2-EXP-007 (실행 2026-09-10, PR #11 머지 2026-09-15, 되돌림 2026-09-18) — call-relative(자기 통화 내 z-score) 피처

- 유형/상태: `TRAIN` / **invalidated (규정 위반으로 되돌림)**
- 목적/가설: `f0_mean`, `rms_p90`, `duration_sec`을 **같은 통화 내** 평균/표준편차로
  z-scoring하면 화자 판별에 도움이 되는지 확인한다.
- 데이터 버전: 전체 29,200통화, 53차원 + call-relative 3종(총 56~64차원, harmonicity/pitch 포함 시 64차원)
- 모델: LightGBM
- 코드/환경: 브랜치 `gwanghee/mission2-feature-fixes`, PR #11
- 채점 지표 사용: 내부 split만
- 평가 지표 결과: 내부 테스트에서 **+0.1~0.3%p** 꾸준한 개선(작지만 일관적)
- 결과 요약: 처음엔 개선으로 보여 PR #11로 `main` 계열에 병합됨.
- 해석(사후): **2026-09-15 대회 측 FAQ로 "같은 통화의 다른 발화 정보는 어떤 형태로든(집계값
  포함) 사용 금지"가 확정**되면서, 같은 통화 내 평균/표준편차를 쓰는 이 피처가 **규정 위반**으로
  판명됨(DECISIONS [001]). 정확도 숫자만 보고 규정 확인을 나중에 한 것이 원인.
- 다음 실험: 2026-09-18, `gwanghee/mission2-feature-fixes`/`-clean` 두 브랜치 모두에서 revert
  커밋으로 11개 피처(call-relative 3 + pitch/energy 5 + harmonicity 3) 전부 제거, 53차원
  순정으로 복귀 → M2-EXP-015에서 재검증.
- 산출물 경로: `reports/mission2/m2_progress_narrative.md` §"시도했다가 철회한 것"
- 보존 상태: `main`은 오염되지 않음(PR들의 base가 `main`이 아니라 feature 브랜치였음, git으로
  확인). revert 커밋: "Mission2: revert PR#11's 11 added features, restore 53-dim baseline"
  (2026-09-18, 양쪽 브랜치)
- 한계: **머지 전 규정 확인을 안 한 프로세스 실수** — 이후 원칙은 DECISIONS [001]/[005] 참조.

### M2-EXP-006 (2026-09-10) — harmonicity(overlap-detection) 피처

- 유형/상태: `TRAIN` / `completed`, **결론: 기각**
- 목적/가설: 자기상관 피크 기반으로 발화 겹침을 탐지하는 피처를 추가하면, 겹침으로 인한
  오류(전체 오류의 57.3%, M2-EXP-002)를 줄일 수 있는지 확인한다.
- 데이터 버전: 전체 29,200통화, 53차원 + harmonicity 3종
- 모델: LightGBM
- 코드/환경: 브랜치 `gwanghee/mission2-feature-fixes`
- 채점 지표 사용: 내부 split만
- 평가 지표 결과: 겹침 구간 오류율 **8.38%p → 8.66%p로 오히려 악화**
- 결과 요약: 합성 톤(synthetic tone) 테스트에서는 깨끗하게 작동했지만 실제 데이터에서는 역효과.
- 해석: 합성 테스트가 "동일 음량 겹침"을 가정했는데, 실제로 겹치는 목소리는 음량 차이가 훨씬
  불균등해서 그 가정이 깨진 것으로 추정.
- 다음 실험: 이 방향 폐기(재시도 안 함). M2-EXP-007과 함께 2026-09-18 revert로 제거.
- 산출물 경로: `reports/mission2/m2_progress_narrative.md` §"막다른 길"
- 보존 상태: revert 커밋으로 제거됨(2026-09-18)
- 한계: 합성 테스트만으로 실데이터 효과를 예측할 수 없었던 사례 — 앞으로 합성 검증은
  실데이터 검증 전 단계로만 취급.

### M2-EXP-005 (2026-09-10) — pitch/energy 단독 피처 추가

- 유형/상태: `TRAIN` / `completed`, **결론: 기각**
- 목적/가설: `f0_mean`, `voiced_ratio` 등 pitch/energy 피처를 단독 추가하면 화자 판별에
  도움이 되는지 확인한다.
- 데이터 버전: 전체 29,200통화, 53차원 + pitch/energy 5종
- 모델: LightGBM
- 코드/환경: 브랜치 `gwanghee/mission2-feature-fixes`
- 채점 지표 사용: 내부 split만
- 평가 지표 결과: 효과 거의 없음(~0.15%p)
- 결과 요약: 개선폭이 잡음 수준.
- 해석: 이미 있는 `spectral_centroid`/`mfcc_0`과 정보가 중복돼 새 신호를 못 줌.
- 다음 실험: 폐기. M2-EXP-007과 함께 2026-09-18 revert로 제거.
- 산출물 경로: `reports/mission2/m2_progress_narrative.md` §"막다른 길"
- 보존 상태: revert 커밋으로 제거됨(2026-09-18)
- 한계: 없음(결론이 명확한 실험).

### M2-EXP-004 (2026-09-14) — 라벨 스왑 2차 확인 (부분 뒤집기 포함)

- 유형/상태: `AUDIT`+`DATA` / `completed`
- 목적/가설: M2-EXP-003에서 발견한 라벨 스왑 의심 통화 외에, L2 CNN 오류 분석 과정에서
  추가로 발견된 이상 패턴을 확인·수정한다.
- 데이터 버전: L2 오류 분석 대상 val 통화
- 코드/환경: `dcc_data.mission2_analysis.corrected_speaker()`, index-patching 스크립트
- 평가 지표 결과: `651e4c754428de718c773b56` — L2 CNN 오류 분석에서 발견. 원래 "전체 뒤집기"
  목록에 있었는데(텍스트 근거: speaker=1이 "네, 119입니다"), 전체 뒤집기로 수정해도 정확도가
  28.57%에 그침 → utt_idx 0~5는 원래 라벨이 맞고(확신도 0.87~0.9996), 6번부터 뒤바뀐
  **부분 뒤집기**였음이 발화별 예측 분석으로 드러남(→ 상세 재확인은 M2-EXP-011).
- 결과 요약: "전체를 뒤집는" 단순 수정이 오히려 앞부분(원래 멀쩡했던 0~5번)을 망가뜨리고
  있었다는 것을 발견 — 라벨 스왑은 "전체"뿐 아니라 "통화 중간부터"도 존재함을 확인.
- 해석: M2-EXP-003의 "전체 뒤집기만 있다"는 전제가 불완전했음 — 자동 탐지 도구
  (`find_label_discontinuity_candidates`, M2-EXP-011)를 만드는 계기가 됨.
- 다음 실험: 전체 데이터셋 체계적 스캔 → M2-EXP-011.
- 산출물 경로: `reports/mission2/m2_label_swap_audit.md` §"부분 뒤집기"
- 보존 상태: 커밋 "Mission2: fix 2 confirmed label discontinuities", "Mission2: extract
  corrected_speaker(), add index-patching script" (2026-09-14)
- 한계: 텍스트 기반 확인(아래 M2-EXP-003의 방법론 참조)이라 재현율 한계 동일.

### M2-EXP-003 (2026-09-10) — 라벨 스왑 최초 감사 (전체 뒤집기)

- 유형/상태: `AUDIT`+`DATA` / `completed`
- 목적/가설: M2-EXP-002 오류 분석에서 정확도가 극단적으로 낮은(5~41%) 통화 상위 4건 중 2건이
  "예측이 정답의 정반대"인 패턴을 보여, 원본 라벨(대원/신고자)이 통화 단위로 뒤바뀌었을
  가능성이 제기됨 — 실제로 그런지 확인한다.
- 확인 방법: 원래 계획은 청음이었으나, **텍스트만 읽어도 화자 역할이 거의 항상 명확히
  드러남**(119 상황실은 "네, 119입니다"로 시작하는 고정 인사말, 신고자는 사고/증상 설명, 대원은
  위치·의식 확인 후 "출동할게요"로 마무리)을 이용해 텍스트로 대체 확인함.
  **`text`는 대회 규정상 모델 feature로 사용 금지**이고 실제 파이프라인도 `text`를 읽지 않는다
  — 여기서는 **사람이 라벨 품질을 검증하는 1회성 감사 목적으로만** 사용했다(DECISIONS [002]).
- 데이터 버전: 정확도 극단적으로 낮은 상위 통화 + zip 기반 부분 스캔(19,969/29,200통화 커버)
- 코드/환경: `dcc_data.mission2_features.CONFIRMED_LABEL_SWAPPED_CALL_IDS`,
  `dcc_data.mission2_analysis.find_label_swap_candidates()`,
  `scripts/scan_mission2_label_swaps.py`
- 평가 지표 결과:
    - **확정 전체 뒤집기 4건**: `651e5550bfd63782eb3866a5`(5.0%), `651e4b1829d82455557a23b2`(14.3%),
      `651e50c72f06ed4a6e31e3d4`, `651e4c754428de718c773dab`(이 중 마지막 1건은 이후 M2-EXP-004에서
      부분 뒤집기로 재정정됨)
    - 자동 규칙(인사말 매칭) 스캔: 19,969통화 커버, **17건 후보** → 수동 검증 결과 확정 3건,
      오탐 2건(119-to-119 지역 이송 통화 등), 보류 10건(겹침 발화로 텍스트만으론 불확실)
- 결과 요약: 4건은 텍스트 근거가 명확해 즉시 수정. 자동 스캔 도구로 추가 후보를 찾았으나
  대부분 청음 확인이 필요한 애매한 케이스로 남음.
- 해석: 라벨 품질 문제는 "몇 건의 예외"가 아니라 체계적으로 찾아야 하는 문제 — 이후
  M2-EXP-011의 자동 탐지기로 이어짐.
- 다음 실험: 보류 10건 실제 청음 확인, 전체 데이터 루트에서 재스캔(나머지 ~9,200통화),
  train 쪽까지 cross-validation으로 확장.
- 산출물 경로: `reports/mission2/m2_label_swap_audit.md`
- 보존 상태: 커밋 "Mission2: pitch/energy + call-relative features, confirm & fix 5
  label-swapped calls" (2026-09-10, 이 커밋이 피처 실험과 라벨 수정을 한 커밋에 같이
  담고 있음 — 서로 무관한 작업을 한 커밋에 섞은 사례로, WORKFLOW.md "한 브랜치=한 작업"
  원칙 위반 소지 있음)
- 한계: **재현율이 낮다** — 인사말이 "119입니다" 계열이 아니면 놓친다(실제로 확정 스왑 중
  1건은 인사말이 "안녕하세요"여서 자동 규칙이 못 찾고 수동 확인으로만 잡음). zip 기반 스캔은
  부분 커버리지(19,969/29,200). train 쪽 라벨 스왑은 이 방법으로 확인 불가.

### M2-EXP-002 (2026-09-09) — L1 전체 데이터 오류 분석

- 유형/상태: `AUDIT` / `completed`
- 목적/가설: M2-EXP-001 baseline(전체 29,200통화, 84.59%)의 오류가 어떤 패턴을 갖는지
  분석해 다음 개선 방향을 찾는다.
- 데이터 버전: 서울/Training 전체 29,200통화(889,358발화, feature 추출 성공 826,333발화)
- split 방식: call_id 기준 GroupShuffleSplit(seed=42, test_size=0.2), val 164,983발화
- 모델: `scripts/train_mission2_baseline.py` (음향 통계 53차원 + LightGBM 기본 파라미터)
- 코드/환경: `scripts/analyze_mission2_errors.py`
- 채점 지표 사용: 내부 split만
- 평가 지표 결과 (val, n=164,983, 오류 25,419건/15.41%):
    - 전체 Accuracy: **84.59%** (leakage gap −0.05%p — random split과 거의 동일, 진짜 일반화 확인)
    - 데이터 규모별: 3,000통화 83.78%(leakage +0.64%p) / 9,000통화 84.90%(−0.28%p) /
      29,200통화(전체) 84.59%(−0.05%p) — 9,000→전체(3.2배)에서 오히려 소폭 하락(표준오차
      ±0.09%p 수준이라 우연이라기엔 큰 차이, 기본 파라미터를 규모에 안 맞춘 게 원인 추정)
    - 발화 겹침: 안 겹침 88.27% vs **겹침 79.90%** — 겹침이 전체의 43.9%인데 전체 오류의 **57.3%** 차지
    - 발화 길이: 0.5~1초 구간이 가장 취약(80.00%, 전체 val의 24%) — <0.5s(84.03%)보다도 낮음
    - 화자 비대칭: 대원(0) 79.75% vs 신고자(1) 89.08% — **9.33%p 격차**, 데이터를 10배 늘려도
      줄지 않음(구조적 특성)
    - 확신도: 정답 시 평균 confidence 85.43%, 오답 시 67.06% — 오답도 확신에 차 있음(순수
      추측이 아님, threshold 보정으로 못 고치는 이유)
- 결과 요약: 오류는 (1) 발화 겹침(두 클래스에 공평하게 손해), (2) 0.5~1초 구간(대원에 불리),
  (3) 화자 비대칭(구조적, 데이터량 무관) 세 축으로 설명됨. 부록으로 정확도가 극단적으로
  낮은 통화 상위 10개를 조사하다 라벨 스왑 의심 사례를 발견(→ M2-EXP-003).
- 해석: 발화 길이가 짧을수록(0.5~1초) 대원-신고자 격차가 최대(14.70%p)이고 길수록 줄어든다
  (2-4s +6.48%p, ≥4s +7.68%p) — 대원 발화가 통계적으로 더 짧고(맞장구·질문 위주), 짧은 구간에서
  모델이 "신고자"로 편향되게 찍는 경향과 결합된 결과로 해석.
- 다음 실험: (1) 라벨 스왑 청음 확인(→M2-EXP-003), (2) 하이퍼파라미터 재튜닝(→M2-EXP-008),
  (3) duration/pitch 겨냥 피처 추가(→M2-EXP-005/006/007), (4) 겹침은 학습 데이터가 아니라
  오디오/feature 처리 쪽에서 다뤄야 함, (5) L2는 후순위로 제안했으나 실제로는 M2-EXP-009부터
  병행 착수됨.
- 산출물 경로: `reports/mission2/m2_error_analysis.md`,
  `m2_error_summary_full.json` (상세 CSV는 14MB라 gitignore, 재생성 명령은 리포트 부록 참고)
- 보존 상태: 커밋 "docs: 전체 29,200통화 기준 Mission 2 오류 분석 리포트 작성" (2026-09-09)
- 한계: threshold 보정(개선 없음), `class_weight='balanced'`(+0.02%p 잡음), 겹침 데이터 제외
  (−0.25%p 손해), 겹침 가중치 낮춤(+0.14%p 잡음) 등은 이미 시도·기각됨(리포트 3절 표) —
  같은 방향 재시도 불필요.

### M2-EXP-001 (실행 2026-09-03, 파이프라인 구축 2026-09-02) — L1 baseline 최초 수립 (3,000통화)

- 유형/상태: `TRAIN`+`EVAL` / `completed`
- 목적/가설: 발화 조각의 음향 통계 feature와 duration만으로 speaker(대원/신고자)를 유의미하게
  분류할 수 있는가?
- 데이터 버전: Seoul Training 3,000통화 샘플(발화 91,508개: train 2,400통화/73,374발화,
  val 600통화/18,134발화)
- split 방식: call_id 기준 GroupShuffleSplit(seed=42, test_size=0.2)
- 모델: LightGBM(미탑재 시 HistGradientBoostingClassifier로 대체), 기본 파라미터
- 코드/환경: Colab, `scripts/train_mission2_baseline.py`
- 모델 선택 기준: 채택 기준 GroupSplit accuracy ≥ 0.70 / 기각 기준 < 0.60(→L2 조기 전환)
- 채점 지표 사용: 내부 split만
- 평가 지표 결과 (n=18,134):
    - Accuracy: **83.78%** (GroupSplit) vs 84.42%(random split) — leakage gap **0.65%p로 매우 작음**
    - 다수결 baseline(this sample): 52.05%
    - feature importance: 특정 변수 하나가 지배하지 않고 MFCC/RMS/spectral centroid에 고르게 분산
- 결과 요약: 채택 기준(0.70)을 크게 상회. leakage gap이 작아 화자 암기가 아니라 실제 일반화된
  음향 신호 기반 분류로 판단됨.
- 해석: 첫 baseline부터 채택 기준을 크게 넘겨, L2(딥러닝) 조기 전환 없이 L1 경로로 계속
  진행하기로 함. 3,000→9,000→29,200통화로 규모를 넓히는 다음 실험으로 이어짐(→M2-EXP-002).
- 다음 실험: 데이터 규모 확장(9,000 → 29,200통화 전체), 그 다음 오류 분석.
- 산출물 경로: `reports/mission2/m2_exp_log.json`
- 보존 상태: 커밋 "docs: M2-EXP-001 실데이터 결과 기록 (GroupSplit accuracy 0.838, 채택)" (2026-09-03)
- 한계: 3,000통화 파일럿 표본. 대회 규모 전체(29,200통화)로는 아직 미확인이었음(→M2-EXP-002에서 해소).
- **주의(번호 재사용 사고 기록)**: `m2_exp_log_tuned.json` 파일이 이 실험과 **같은 ID
  "M2-EXP-001"**을 재사용해 27,030통화·86.77%라는 **전혀 다른(사실상 M2-EXP-015와 동일한)**
  결과를 담고 있었다. 이 로그에서는 원 실행일(2026-09-03)이 명확한 이 3,000통화 결과만
  M2-EXP-001로 유지하고, 튜닝된 86.77% 결과는 M2-EXP-015로 별도 번호를 부여했다.
  **앞으로 실험 ID는 절대 재사용하지 않는다.**

---

*관련 문서: `DECISIONS.md`(의사결정), `WORKFLOW.md`(Git 협업 규칙),
`reports/mission2/m2_progress_narrative.md`(팀용 시간순 서술, 이 로그의 1차 원천).*
