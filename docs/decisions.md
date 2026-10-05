# 팀 결정 기록 (Decisions Log)

> 미션마다 흩어진 `verdict`만으로는 "왜 이렇게 하기로 했는지"가 한눈에 안 보입니다.
> 이 파일은 팀이 내린 판단과 그 이유만 모아둡니다. 수치와 실험 결과는
> `docs/experiment_log.md`에서 찾고, 최종 근거는 각 행에 연결된 로그·보고서를 확인하세요.
>
> 형식: 날짜 역순(최신이 위). 항목당 결정, 이유, 상태와 후속 조치를 간결하게 기록.
> 갱신 시점: 실험 결과를 채택·기각·보류할 때, 또는 방법론과 방향을 바꿀 때마다 한 항목 추가.
> 날짜는 `YYYY-MM-DD` 형식으로 쓰며, 근사 날짜라면 본문에 표시합니다.

---

## 2026-10-05 — [M2] 제출 후보 충돌 정리 필요 (미결, 사령탑 판단 요청)

- **현황**: 후보가 두 갈래다. (a) 스펙상 L1 53D + LightGBM 9k는 내부 84.90%이고 ckpt가 없어 공식 Val 미평가.
  (b) MLP 트랙(feature·모델 실험)은 공식 Val 85.29%(42D), 내부 87.38%(59D, M04-100)까지 기록됐으나 제출용 ckpt·inference가 정리되지 않음.
- **비교 불가 사유**: (a)는 call_id GroupSplit 내부 점수, (b)는 공식 Val 또는 다른 내부 split이므로 점수를 직접 비교하지 않음.
- **결정**: 확정 안 됨. 같은 공식 Val 기준으로 (a)를 채점하거나 (b)를 ckpt로 고정해 비교한 뒤 정함.
- **관련**: [`m2_official_val_spec.md`](../reports/mission2/m2_official_val_spec.md), [`experiment_log.md`](./experiment_log.md) M2-MODEL·M2-AF.

## 2026-10-05 — [M1] 1차 제출 후보를 LR로 고정, 새 학습 없이 제출 패키징

- **결정**: 공식 Val 95.11%의 LR(`m1_baseline_lr.pkl`)을 제출 후보로 고정하고 새 학습·MFCC/F0 개선은 보류함.
  제출 CLI(`mission1_inference.py`)와 CSV 열(`audio file name`, `gender`)은 공식 PDF 기준을 따름. Zoom 안내의 startAt/endAt 포함설은 채택하지 않음.
- **이유**: 남은 기간 대비 안전한 제출본 확보가 우선. ckpt는 Git에 올리지 않고 SHA256만 기록.
- **한계**: 규정상 가중치 확장자(`.pt/.pth/.ckpt`) 대응과 스모크 테스트가 남아 있음.
- **관련**: [`m1_submission_checklist.md`](../reports/mission1/m1_submission_checklist.md).

## 2026-10-05 — [M3] 80:20 기준 재평가 순서와 보류 항목 확정

- **결정**: 재평가는 P0 TF-IDF → P1 KcBERT 앞 256 → P2 토큰 청크 vs 턴 청크(동일 조건) → P3 완화 가중 BCE 순.
  증상별 threshold 재튜닝과 강한 가중 BCE는 보류함. 전사 텍스트만 입력으로 사용함.
- **이유**: 과거 점수는 분할·Train 비율·임계값이 달라 직접 비교가 안 됨(스펙 표 참조).
- **상태**: 모두 승인 대기, 미실행.
- **관련**: [`m3_reeval_spec.md`](../reports/mission3/m3_reeval_spec.md), [`m3_next_experiment_candidates.md`](../reports/mission3/m3_next_experiment_candidates.md).

## 2026-10-05 — [M2] 같은 통화의 다른 발화 정보 사용 금지 (주최 측 공식 답변)

- **결정**: M2는 분류 대상 발화 조각(startAt~endAt)의 음성 정보만 사용함. 같은 통화의 다른 발화로 계산한
  길이·평균 길이·상대 길이 등 모든 값과 통화 단위 정규화, 통화 내 클러스터링·화자 분리를 사용하지 않음.
- **근거**: 멘토링 게시판 질의에 대한 주최 측 답변: "분류 대상이 되는 해당 발화 조각(startAt~endAt)의 정보만 사용",
  "발화의 순서를 직접 사용하지 않더라도, 다른 발화의 정보를 현재 발화의 입력에 활용하는 방식에 해당",
  "각 발화는 독립적으로 입력된다고 가정하고 모델을 구성".
- **현황 점검**: 현 L1 파이프라인은 통화 WAV를 한 번 읽은 뒤 발화 구간만 잘라 특징을 계산하므로 이 기준에 맞음.
  사전학습 음성 임베딩 등 이후 실험도 발화별 독립 추론과 Training 고정 통계만 사용함.
- **관련**: [`mission2_features.py`](../src/dcc_data/mission2_features.py) (`read_call_waveform` → `slice_and_pad`).

## 2026-10-04 — [M2] 표준 표현(eGeMAPS 88D) 기각, clean 59D 유지

- **결정**: eGeMAPSv02 88D는 채택하지 않고 handcrafted 59D를 유지함.
- **이유**: 같은 M04-100·split에서 87.38% → 86.40%(-0.98%p), speaker0 recall 하락, 고확신 오답 증가.
  매우 짧은 발화에서는 eGeMAPS가 NaN/inf를 반환함.
- **한계**: 내부 split 결과이며 공식 Val 재평가는 하지 않음. 최종 모델 확정은 아님.
- **관련**: [`mission2_standard_representation_comparison.ipynb`](../notebooks/mission2_standard_representation_comparison.ipynb), M2-REP-001.

## 2026-09-30 — [M3] 현재 baseline 분할·전처리 기준 확정

- **결정**: `recordId` 기반 grouping은 사용하지 않고, JSON 샘플 단위로 **80:20 multilabel stratified split**을 사용함.
  seed는 42, 모든 증상의 예측 임계값은 0.5로 고정함.
- **전처리**: `utterances`에 들어 있는 기존 순서를 그대로 유지하고 `text`만 이어 붙여 사용함.
  `startAt`, `id`를 이용한 재정렬은 하지 않으며, 제공된 `symptom` 라벨은 현재 단계에서 그대로 사용함.
- **검증 원칙**: 성능은 Macro F1만 보지 않고 증상별 precision/recall/F1과 오류 사례를 함께 확인함.
  한 실험에서는 핵심 변수 하나만 변경해 baseline 대비 개선 여부를 판단함.
- **과거 결과 해석**: 이전 `recordId` grouping 또는 90:10 random split에서 나온 점수는 과거 실행 기록으로 보존하되,
  현재 80:20 multilabel stratified 기준의 성능과 직접 비교하지 않음. 최신 기준으로 재실행한 결과를 이후 비교 기준으로 사용함.
- **관련**: [`mission3_baseline 최종 수정본.py`](<../scripts/mission3_baseline 최종 수정본.py>),
  [`mission3_jieun_baseline.ipynb`](../notebooks/mission3_jieun_baseline.ipynb).

## 2026-09-27 — [M2] feature 단계 종료: rolloff+zcr 조합(E05)을 42D로 고정

- **결정**: MFCC40에 rolloff·zcr을 더한 42D를 고정하고, 모델 구조만 바꾸는 단계로 이행함. 추가 feature 조합을 공식 Val에서 반복 탐색하지 않음.
- **이유**: 단일 feature는 rolloff +0.24%p가 최대였고 조합 E05가 +0.40%p. 조합은 Training-only 중복성·오류 보완성 진단 후 Val을 보기 전에 사전 확정함. duration은 효과 없음.
- **한계**: v1 cache 결과이며 v2 코드로 재평가하지 않음. E05와 E02 차이의 유의성은 미검정.
- **관련**: M2-FEAT-E01~E05.

## 2026-09-21 — [M2] handcrafted feature 탐색 종료, 59D를 reference로 채택

- **결정**: AF10(speech activity)+AF11(spectral entropy)을 더한 59D를 reference로 채택하고, AF05(delta-MFCC) 추가(61D)는 보류함.
  AF01·AF03·AF08은 기각, AF07·AF09는 효과가 잡음 수준이라 보류함. 이후 handcrafted 추가 탐색은 중단하고 표준 표현과 비교.
- **이유**: 52D→59D는 +0.50%p이고 두 speaker recall이 함께 올랐다. 61D는 +0.018%p에 speaker1 recall -1.32%p로 trade-off만 있었음.
  AF02(pitch/voicing)는 47D 단계에서 +0.67%p로 채택됨.
- **관련**: M2-AF-R1~R4, [`mission2_acoustic_feature_expansion.ipynb`](../notebooks/mission2_acoustic_feature_expansion.ipynb).

## 2026-09-21 — [M2] 모델을 HGB에서 MLP로 승격

- **결정**: 42D에서 MLP(64 hidden, M04)를 best 후보로 선정함. ExtraTrees·QDA는 기각, LR은 비교점으로 유지.
- **이유**: 내부 85.80%(HGB 84.19% 대비 +1.61%p)와 공식 Val 85.29%(+1.57%p)의 개선폭이 거의 같아 내부 우연으로 보기 어렵고, speaker 0·1 recall이 모두 상승함.
- **한계**: 30 epoch에서 미수렴. max_iter 60·100 확장은 +0.08~0.16%p로 효과 작음. 별도 유의성 검정 없음.
- **관련**: M2-MODEL-M00~M04, M2-MODEL-O.

## 2026-09-21 — [공통] PR 자동 병합 게이트와 기록 운영 원칙

- **결정**: PR마다 CI와 verdict 기록을 거치는 병합 게이트를 도입함(9/21~22). 체크포인트·데이터는 Git에 올리지 않고 SHA256만 기록하며,
  데이터 루트는 `DCC_DATA_ROOT`(`data/raw_merged`)로 고정함(10/2).
- **이유**: 여러 담당자가 병렬로 작업하므로 재현성과 규정 준수를 자동 점검하기 위함.
- **관련**: [`pr-ci.yml`](../.github/workflows/pr-ci.yml), [`WORKFLOW.md`](../WORKFLOW.md).

## 2026-09-20 — [M3] 양성 손실 가중치를 완화해 후속 실험

- **결정**: 희소 증상의 낮은 recall을 보완하기 위해 weighted BCE를 사용하되,
  과다 양성 예측을 줄이도록 `pos_weight_scale=0.25`, 상한 3.0으로 완화함.
- **이유**: 초기 강한 가중치 모델은 recall은 높아졌지만 precision과 subset accuracy가 낮아져
  오탐 증가 가능성이 확인됨.
- **상태/후속**: Training 40%, effective batch size 16, 3 epochs 학습 체크포인트는 생성됨.
  평가 결과를 확인한 뒤 비가중 모델과 동일 조건으로 비교해야 함.
- **관련**: [`m3_weighted_diagnosis_sources.json`](../reports/mission3/m3_weighted_diagnosis_sources.json),
  [`train_weighted_40pct_vscode.py`](../scripts/mission3/train_weighted_40pct_vscode.py).
## 2026-09-19 — [M3] 증상별 임계값 조정을 중단하고 전 증상 임계값을 0.5로 고정

- **결정**: 증상별 임계값 조정을 중단하고 모든 증상의 판정 임계값을 0.5로 고정함.
- **이유**: 임계값 조정 후 오탐이 증가했고, 당시 선택된 증상별 임계값 배열이 남아 있지 않아
  결과를 재현할 수 없음. 공식 Validation에 맞춘 사후 조정 위험도 있음.
- **상태/후속**: 증상별 임계값 실험은 참고용으로만 유지하고, 이후 모델은 임계값 0.5에서 비교함.
- **관련**: [`m3_threshold_comparison.json`](../reports/mission3/m3_threshold_comparison.json),
  [`MISSION3_Colab_가중치모델_임계값점검.ipynb`](../notebooks/mission3/MISSION3_Colab_가중치모델_임계값점검.ipynb).

## 2026-09-18 — [M3] 단순 토큰 청크를 발화 턴 기반 청크로 변경

- **결정**: 같은 화자의 연속 발화를 하나의 턴으로 묶고, 질문과 다음 답변이 가능한 한
  같은 청크에 포함되도록 `dialogue_turn_overlap_v1`을 사용함.
- **이유**: 토큰 수만 기준으로 자르면 119 상담의 질문과 증상 답변이 서로 다른 청크로 분리될 수 있음.
- **상태/후속**: 공식 Validation Macro F1 0.5365를 기록함. 토큰 청크와 학습 조건을 동일하게 맞춘
  재실험 전까지 두 방식의 우열은 확정하지 않음.
- **관련**: [`m3_turn_chunk_validation_metrics.json`](../reports/mission3/m3_turn_chunk_validation_metrics.json).

## 2026-09-16 — [M3] 긴 대화의 전체 내용을 사용하기 위해 토큰 청크 도입

- **결정**: 전체 대화를 256토큰·overlap 64의 여러 청크로 나누고,
  청크별 logit을 max pooling해 통화 단위 예측을 생성함.
- **이유**: 기존 baseline에서는 256토큰 이후의 증상 표현이 모델 입력에서 제외됨.
- **상태/후속**: 공식 Validation Macro F1 0.5496을 기록함. 질문과 답변이 청크 경계에서
  분리되는 문제를 보완하기 위해 이후 발화 턴 청크를 실험함.
- **관련**: [`m3_chunk_validation_metrics.json`](../reports/mission3/m3_chunk_validation_metrics.json).
## 2026-09-15 — [M2] 라벨이 뒤바뀐 것으로 의심되는 통화 청음 확인 (미결)

- **현황**: 전체 규모 오류 분석에서, 정확도가 극단적으로 낮은 통화 일부가 "어려워서 틀린" 것이 아니라
  **정답의 정반대를 예측하는** 패턴을 보임. 상위 2건은 `651e5550bfd63782eb3866a5`(정확도 5.0%,
  발화 20개 — 라벨을 뒤집으면 95%), `651e4b1829d82455557a23b2`(14.3%, 21개 — 뒤집으면 85.7%).
- **가설**: 모델·feature의 한계가 아니라 원본 라벨의 speaker 값이 통화 단위로 뒤바뀌어 있을 가능성.
- **결정**: 아직 확정 안 됨. 해당 통화의 원본 오디오를 청음해 라벨이 실제 화자와 맞는지 확인하기 전까지 보류.
  라벨이 맞으면 "매우 어려운 통화"로 분류하고 이 가설은 기각, 틀리면 전체 29,200건에서 같은 패턴을
  체계적으로 탐색(학습·평가 양쪽에 잡음이 섞여 있는 것이므로 교정 시 정확도가 오를 여지가 있음).
- **담당/완료 조건**: 김광희 확인 요청. 청음 결과와, 스왑이 확인될 경우 탐색 범위·방법을 기록하면 종결.
- **관련**: [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 2절.

## 2026-09-15 — [M2] 9,000건을 성능 후보로 채택, 29,200건은 오류 분석 기준으로 보류

- **결정**: 3,000건 최초 baseline은 유효한 출발점으로 유지하고, 9,000건 결과(84.90%)를 현재
  성능 기준 채택 후보로 선정. 29,200건 전체 결과(84.59%)는 오류 분석 기준 실행으로 활용하되
  성능 기준 최종 선택은 보류.
- **이유**: 3,000→9,000건은 +1.12%p 개선됐지만 9,000→29,200건은 -0.31%p 하락함.
  전체 데이터 하락 원인은 기본 LightGBM 하이퍼파라미터가 데이터 규모에 맞지 않기 때문으로 추정하나 미검증.
- **후속 실험**: 전체 데이터에서 하이퍼파라미터를 재튜닝한 뒤 9,000건 결과와 같은 분할 기준으로 비교.
- **관련**: [`m2_exp_log.json`](../reports/mission2/m2_exp_log.json),
  [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md).

## 2026-09-15 — [M2] class_weight·threshold·겹침 데이터 처리 실험 기각

- **결정**: threshold 보정, `class_weight="balanced"`, 겹침 학습 데이터 제외와 가중치 조정을 기각.
- **이유**: threshold는 최적값이 0.5로 유지됐고, class weight와 가중치 조정은 잡음 수준
  (각각 +0.02%p, +0.14%p)이었으며, 겹침 데이터 제외는 -0.25%p로 성능이 하락함.
- **관련**: [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 3절.

## 2026-09-10 — [M1] GBM vs LR, LR을 1차 채택 후보로 결정

- **결정**: 공식 Validation에서 LR(95.11%)을 GBM(94.92%)보다 1차 채택 후보로 선정. GBM은 향후
  feature 확장 실험용 비교 baseline으로 유지.
- **이유**: 정확도 차이 0.19%p(오분류 7건 차이)로 작고, LR이 더 단순하며 해석하기 쉬움.
- **관련**: [`m1_exp_log.json`](../reports/mission1/m1_exp_log.json),
  [`m1_val_eval.json`](../reports/mission1/m1_val_eval.json),
  [`m1_val_bootstrap_ci.json`](../reports/mission1/m1_val_bootstrap_ci.json).

## 2026-09-10 — [M1] baseline 단계에서 k-fold 교차검증 생략

- **결정**: baseline 단계에서는 별도 k-fold 교차검증을 수행하지 않음.
- **이유**: 대규모 고정 Validation 세트와 내부 holdout 결과가 근접했고, baseline 단계의 시간·계산 비용을
  고려해 우선 생략함. Bootstrap CI와 Wald CI의 일치는 불확실성 계산의 교차 확인일 뿐,
  학습 데이터 분할에 따른 모델 안정성을 검증한 근거로 사용하지 않음.
- **한계/후속**: 현재 결과는 학습 분할 민감도를 검증하지 않았음. 최종 모델 확정 전에 필요하면
  반복 holdout, k-fold 또는 paired 비교를 수행.

## 2026-09-10 — [M1] "통계적으로 동급" 표현 수정 (외부 리뷰 반영)

- **결정**: PR 설명과 verdict의 "두 모델 신뢰구간이 겹쳐 통계적으로 동급" 표현을
  "차이가 작음(0.19%p)"으로, "과적합·리키지 징후 없음"을 "뚜렷한 징후가 관찰되지 않음"으로 수정.
- **이유**: GBM과 LR은 같은 3,640개 샘플에 대한 paired 비교이므로 marginal CI 겹침만으로는
  유의성을 판단할 수 없음. 엄밀한 비교에는 McNemar test나 paired bootstrap이 필요함.

## 2026-09-01 — [M3] KcBERT 다중 라벨 모델을 최초 신경망 baseline으로 사용

- **결정**: 앞 256토큰을 사용하는 KcBERT 9종 다중 라벨 모델을 최초 신경망 baseline으로 사용함.
- **이유**: 한 통화에 여러 증상이 존재해 다중 라벨 분류가 필요하며,
  Training 내부 검증 Macro F1 최고 0.5861을 기록함.
- **상태/후속**: baseline으로 유지함. 약 29.88%의 대화가 256토큰을 초과해
  후반 문맥이 잘리는 문제는 청크 방식으로 보완함.
- **관련**: [`m3_kcbert_baseline_train.log`](../reports/mission3/m3_kcbert_baseline_train.log).
---

새 결정은 첫 번째 결정 항목 위에 추가합니다.
