# M2 최종 후보 비교 — 2026-10-10

기존 결과를 감사해 비교했다. 이번 점검에서 새 M2 학습이나 공식 평가를 수행하지 않았다.

| 후보 | 평가 분할 | Accuracy | 판단 |
|---|---|---|---|
| 연우 MFCC40 + LR | 공식 Validation | 80.91% | 초기 baseline |
| 연우 42D + MLP | 공식 Validation | 85.29% | 연우 계열 중 공식 평가를 확인한 최고 결과 |
| 연우 59D + MLP | Training 내부 holdout | 87.4558% | 공식 점수 미확인 |
| 연우 clean 59D + MLP (최신 PR #35) | Training 내부 holdout | 87.3760% | 공식 점수·가중치 미확인 |
| 제공 라벨 53D LightGBM | 공식 Validation | 86.44% | CNN보다 낮음 |
| 기존 CNN 3-seed 앙상블 | 공식 Validation | 91.55% | 기존 최고, 102,485 / 111,947 |
| 원본부터 재학습한 CNN 3-seed 앙상블 | 공식 Validation | 91.49% | **우선 제출 후보**, 102,419 / 111,947 |

## 결정

**M2-RERUN-001을 우선 제출 후보로 선택한다.** 원본 데이터부터 재생성·학습한 로그와 제출 경로의 공식 평가가 기록되어 있고, 이전 앙상블과의 차이는 66발화(약 −0.06%p)이다. 연우 계열 중 확인된 최고 공식 점수 85.29%보다 6.20%p 높다. 다만 **최신 59D MLP와 공식 성능 우열을 확인했다는 뜻은 아니다.** 내부 87.38%를 공식 91.49%와 직접 비교하지 않는다.

최신 공개 결과는 [PR #35](https://github.com/eigensinn-dcc2026/prelim/pull/35), 이전 59D 결과는 [PR #25](https://github.com/eigensinn-dcc2026/prelim/pull/25)이다. 작성자 `wldusdn03`의 전체 PR·이슈, yeonwoo 브랜치, 로컬 대회 폴더, 로컬 Drive와 Google Drive 연결 도구의 공유 파일까지 검색했으나 최신 59D 가중치·공식 점수를 찾지 못했다. 해당 모델은 최종 제출 모델로 확정된 것이 아니라 representation 비교용 내부 reference라고 PR에 명시돼 있다.

## 후보 선택과 실행 반영의 구분

재학습본의 보관 경로는 기존 로그상 `Drive m2_cache/m2_rerun_from_raw/rerun_2026-10-06b/`, 앙상블 SHA256은 `19ea482a…3246`이다. 현재 계정의 접근 가능한 Drive에는 그 경로가 없었다. **이번 작업에서 가중치 실물·전체 해시를 검증했다고 표시하지 않는다.** 파일 확보와 전체 SHA256 대조, 제출 폴더에서 오프라인 실행 확인 전에는 기존 M2 기본 경로를 바꾸지 않는다.

상세 비교와 검색 범위는 `m2_final_candidate_comparison.json`에 기록했다.

## 사용자 공유 폴더 확인 — 2026-10-10

[공유 comparison 폴더](https://drive.google.com/drive/folders/17D_6n2E36GTlRE8Yj3HcaNTYeSr8-ZXx)에는 `summary.json`, `comparison.csv`, `paired_slices.csv`, `validation_predictions.parquet` 네 파일이 있다. 가중치 파일은 없다.

보고서의 `official_validation_used=false`, 분할의 Training 내부 80:20(총 889,358발화 중 177,891발화, 5,840통화), 예측 `record_key`의 모든 접두사가 `Training`임을 확인했다. 공식 Validation의 3,640통화·111,947발화와 다르다. Parquet 전체 SHA256이 보고서와 일치하고, 예측을 독립 재채점한 결과도 정확히 일치했다.

- clean 59D MLP: **87.375977%**, 155,434 / 177,891.
- eGeMAPS 88D MLP: **86.398975%**, 153,696 / 177,891.

따라서 최신 59D의 내부 점수와 예측은 실물로 확인됐지만, **가중치와 공식 점수는 이 폴더에서 확인되지 않았다.** CNN 공식 91.49%와 같은 분할에서 우열을 확인했다는 뜻이 아니며 기존 우선 후보 결정을 유지한다. 상세 검증은 `yeonwoo_shared_folder_audit.json`에 기록했다.
