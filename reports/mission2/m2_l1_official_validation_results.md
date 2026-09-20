# Mission 2 L1: 공식 Validation 셋 첫 검증

지금까지 L1(LightGBM 튜닝, 53차원)은 Training 29,200통화 안에서 우리가 직접 뗀
GroupShuffleSplit(8:2, seed=42)로만 평가해왔다 — 대회가 실제로 제공하는 공식
Validation 폴더는 한 번도 안 썼다(EDA 단계 결정: "Validation 폴더는 분포를 보지
않는다", `DCC2026_EDA_요약.md`). 재학습 없이, 이미 저장된 체크포인트
(`checkpoints/mission2/m2_baseline_lgbm_tuned.pkl`)를 공식 Validation에 **딱 한 번**
평가해 봤다.

## 데이터 위치

처음엔 OneDrive `대학부 데이터/` 폴더에서 찾았는데 그 안엔 서울 Validation의 12%(424/
3,640통화)만 있었다(wav/json도 서로 안 맞았음). **실제 전체 데이터는
`C:\dcc_data_local_full\Validation`에 있었다** — Seoul Validation wav 3,640개/json
3,640개, 완벽하게 페어링됨. `prelim/data/manifests/dcc_manifest_local_full.csv`(2026-09-09
빌드, Training 29,200 + Validation 3,640)와 정확히 일치하는 그 원본 루트였다.

새로 만든 파이프라인(`dcc_data/mission2_features.py`에
`select_seoul_validation_rows`/`build_seoul_validation_dataset` 추가, 기존
`select_seoul_training_rows`/`build_seoul_training_dataset`과 동일 구조지만
`VALIDATION` 스플릿으로 게이트)으로 전체 3,640통화의 53차원 feature를 뽑았다:
**111,947발화, 추출 실패 0건** — yeonwoo 실험 코드의 `EXPECTED_UTTERANCE_COUNTS
["Validation"]`과 정확히 일치, 전체 공식 Validation임이 확인됨.

## 결과 (전체 3,640통화, 111,947발화)

| | 내부 split(Training 20%) | 공식 Validation(전체, 이번이 처음) |
|---|---|---|
| 발화 수 | 164,983 | 111,947 |
| Accuracy | 86.77% | **86.40%** |
| 대원(0) recall | 82.04% | 81.68% |
| 신고자(1) recall | 91.16% | 90.77% |
| 격차 | 9.12%p | 9.09%p |

(참고: 데이터 위치를 못 찾아 424통화만으로 먼저 돌렸던 예비 결과는 86.24%로, 전체
결과와 0.16%p밖에 안 차이났다 — 작은 표본이었는데도 이미 안정적인 추정이었다.)

## 해석

**내부 split과 공식 Validation의 차이는 −0.37%p뿐이다.** 대회가 실제로 검증에 쓰는
것과 동일한 데이터, 전체 3,640통화로 확인한 결과이므로 신뢰도가 높다 — 내부 split에
과적합돼서 숫자가 부풀려진 게 아니라 실제로 일반화되는 성능이다. 화자 비대칭 패턴
(대원 recall < 신고자 recall, 9%p대 격차)도 정확히 재현됐다 — 데이터가 달라져도
사라지지 않는 구조적 특성임이 재확인됨.

**yeonwoo의 MFCC40+로지스틱회귀 최종 결과(공식 Validation 전체 3,640통화 기준
81.31%)보다 +5.09%p 높다** — 이제 둘 다 같은 공식 Validation 전체 데이터 기준이므로
완전히 공정한 비교다.

## 다음 단계

이 체크는 **한 번만** 하는 게 원칙 — 여기서 나온 숫자를 보고 다시 하이퍼파라미터를
조정하면 yeonwoo처럼 Training-only screening 없이 Validation을 반복 조회하는 셈이
되어 이 검증의 의미가 없어진다. 추가 튜닝이 필요하면 반드시 Training 내부 split으로
먼저 하고, 최종 후보가 바뀔 때만 공식 Validation을 다시 본다. **현재 결론: L1 최종
후보(LightGBM 튜닝, threshold=0.5, class_weight 없음)는 공식 Validation 86.40%로
확정.**
