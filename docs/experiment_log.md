# 실험 로그 (Experiment Log) — 사람이 읽는 요약판

> 이 파일은 우리가 해본 실험들을 한눈에 찾아보는 목록입니다. 진짜 정확한 숫자는 표 안이 아니라
> 각 줄의 `관련 파일`에 연결된 파일(로그, 결과 파일, 보고서)에 있습니다 — 표는 어디까지나 요약입니다.
> "왜 이걸 채택했는지·기각했는지", "왜 방향을 바꿨는지"는 `docs/decisions.md`에 따로 적어둡니다.
>
> 한 줄을 쓸 때 꼭 넣을 것: 실험 ID, 날짜(`YYYY-MM-DD`), 데이터가 몇 개였는지, 어떻게 나눠서
> 검증했는지, seed, 결과, 지금 상태, 관련 파일.
>
> **자주 나오는 말**
> - **seed**: 실행할 때마다 결과가 똑같이 나오게 고정해두는 난수 값. `seed=42`면 데이터를
>   섞거나 나누는 방식이 항상 같아서, 나중에 다시 실행해도 같은 결과가 나옵니다.
> - **분할(split) / GroupShuffleSplit**: 데이터를 학습용과 검증용으로 나누는 것. "Group"이
>   붙으면 같은 통화의 발화들이 학습 쪽과 검증 쪽에 나뉘어 들어가지 않도록 통화 단위로 나눴다는
>   뜻입니다. `8:2`는 80%는 학습용, 20%는 검증용으로 나눴다는 뜻입니다.
> - **val / Validation**: 학습에는 쓰지 않고 "얼마나 잘 맞히는지" 확인용으로만 쓰는 데이터.
>   공식 Validation은 대회 주최측이 따로 준 검증 데이터를 말합니다.
> - **%p (퍼센트 포인트)**: 두 정확도 값의 차이. 86%에서 87%가 되면 "+1%p"입니다.
> - **gap(격차)**: 대원(0)과 신고자(1) 두 그룹의 정답률(recall) 차이. 대회 채점에는 안 들어가고
>   오류 분석용 참고 지표입니다.
> - **수렴(converge)**: 더 학습시켜도 정확도가 더 이상 안 오르는 상태.
> - **채택 / 기각**: 이 방법을 실제로 쓰기로 했는지(채택), 안 쓰기로 했는지(기각).
>
> 실험 ID 이름 짓는 법: `M{미션 번호}-{종류}-{번호}` (예: `M2-SCALE-002`). 실제 JSON 파일에
> 남아있는 ID는 `M1-EXP-001`, `M1-EXP-001-val`, `M1-VAL-ERR-001`, `M2-EXP-001` 이렇게 4개뿐이고,
> `M2-EXP-002`는 오류 분석 보고서에만 쓰였을 뿐 따로 JSON은 없습니다. 그 밖의 ID는 이 문서를
> 만들면서 나중에 붙인 것입니다. 새 실험을 추가할 땐 표에서 다음 번호를 확인해서 붙이고, 실행
> 로그에도 같은 ID를 남겨주세요.

## Mission 1 — 신고자 성별 분류 (Accuracy)

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| M1-EXP-001 (internal) | 2026-09-10 | 소리에서 뽑은 특징 52개로 GBM과 로지스틱회귀(LR) 학습. 학습 데이터를 클래스 비율 맞춰 8:2로 나눠 내부 확인, seed=42 | GBM 94.78% / LR 95.02% | 참고용 (아직 내부 확인 단계) | [`m1_exp_log.json`](../reports/mission1/m1_exp_log.json) |
| M1-EXP-001-val | 2026-09-10 | 주최측 공식 Validation으로 평가, 통화 3,640건(여성 1,959 / 남성 1,681) | GBM 94.92% / LR 95.11% | LR을 1차 채택 후보로 정함 | [`m1_val_eval.json`](../reports/mission1/m1_val_eval.json) |
| M1-EXP-001-val 파생 | 2026-09-10 | 정확도가 얼마나 믿을만한지 확인하는 통계 방법(부트스트랩)으로 95% 신뢰구간 계산, 10,000번 반복, seed=42 | GBM [94.20, 95.60] / LR [94.40, 95.82] | 정확도 수치의 신뢰도 확인용 | [`m1_val_bootstrap_ci.json`](../reports/mission1/m1_val_bootstrap_ci.json) |
| M1-VAL-ERR-001 | 2026-09-10 | 어떤 경우에 틀리는지 분석(발화 길이 / 목소리 겹침 / 피처 분포별) | 상세 결과는 보고서 참조 | 일단 마무리 | [`m1_val_error_analysis.json`](../reports/mission1/m1_val_error_analysis.json), [`M1_error_analysis_report.md`](../reports/mission1/M1_error_analysis_report.md) |

## Mission 2 — 신고자/119대원 분류 (Accuracy)

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| M2-EXP-001 | 2026-09-03 | 첫 baseline: 소리 특징 53개 + LightGBM 기본 설정, 통화 3,000건. 같은 통화가 학습/검증에 같이 안 들어가게 통화 단위로 8:2 분할, seed=42 | 정확도 83.78% (통화 단위 안 지키고 무작위로 나누면 84.42%, 차이 +0.65%p) | baseline으로 채택 | [`m2_exp_log.json`](../reports/mission2/m2_exp_log.json) |
| M2-SCALE-001 | 2026-09-15 | 같은 방식, 통화 수만 9,000건으로 늘림 | 84.90% (3,000건 대비 +1.12%p) | 지금까지 중 제일 좋아서 채택 후보로 정함 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 0·3절 |
| M2-SCALE-002 | 2026-09-15 | 같은 방식, 가진 통화 전체(29,200건) 사용 | 84.59% (9,000건보다 -0.31%p 낮음, 격차도 -0.05%p) | 오류 분석용으로만 쓰고, 최종 성능 후보 선택은 보류 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 0·3절 |
| M2-EXP-002 | 2026-09-15 | 정답 기준선(threshold)을 0.5에서 다른 값으로 바꿔봄 | 최적값은 그대로 0.5, 개선 없음 | 기각 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 1-4·3절 |
| M2-EXP-003 | 2026-09-15 | 두 클래스 비율 차이를 보정하는 옵션(`class_weight="balanced"`) 켜서 재학습 | +0.02%p | 기각 (차이가 너무 작아 의미 없는 수준) | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 3절 |
| M2-EXP-004 | 2026-09-15 | 목소리가 겹치는 학습 데이터를 아예 빼고 학습 | -0.25%p | 기각 | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 3절 |
| M2-EXP-005 | 2026-09-15 | 목소리가 겹치는 학습 데이터의 학습 비중을 0.3배로 낮춤 | +0.14%p | 기각 (잡음 수준) | [`m2_error_analysis.md`](../reports/mission2/m2_error_analysis.md) 3절 |
| M2-EXP-006 | 2026-09-10 | 목소리 높낮이·에너지 관련 피처 추가(f0_mean, voiced_ratio 등) | 효과 거의 없음(~0.15%p) | 기각 — 이미 쓰던 피처(spectral_centroid, MFCC0)와 겹치는 정보라 소용없었음 | [`m2_progress_narrative.md`](../reports/mission2/m2_progress_narrative.md) §"막다른 길" |
| M2-EXP-007 | 2026-09-10 | 목소리 겹침을 잡아내려고 만든 피처(harmonicity — 소리 파형의 반복 패턴 기반) | 겹침 구간 오류율 8.38%p→8.66%p로 악화 | 기각 — 인공적으로 만든 테스트에서는 잘 됐지만 실제 데이터에서는 오히려 나빠짐 | [`m2_progress_narrative.md`](../reports/mission2/m2_progress_narrative.md) §"막다른 길" |
| M2-EXP-008 | 실행 2026-09-10, PR#11 머지 09-15, 되돌림 09-18 | 같은 통화 안 다른 발화들과 비교해 표준화한 피처 — f0_mean/rms_p90/duration_sec을 그 통화의 평균·표준편차 기준으로 변환 | 내부적으로는 +0.1~0.3%p 좋아졌지만, **대회 FAQ(9/15)로 이 방식이 규정 위반이라는 게 확정됨**(같은 통화의 다른 발화 정보를 쓰면 안 된다는 규칙) | **되돌림(무효)** — `decisions.md` 참조 | [`m2_progress_narrative.md`](../reports/mission2/m2_progress_narrative.md) §"시도했다가 철회한 것" |
| M2-TUNE-001 | 날짜 미상(09-10~09-15 사이로 추정) | LightGBM 설정값 재조정(`num_leaves=255, 학습률=0.02, 나무 개수≈1150`) — 단, 이때는 규정 위반 피처가 섞인 64차원 기준 | 84.59%→86.9%(+2.3%p) | 규정 위반 피처가 섞여 있어 공식 수치로는 안 씀 — 이후 M2-COMPARE-001에서 문제없는 53차원 기준으로 다시 확인 완료 | 별도 committed 리포트 없음(TODO) |
| M2-LABEL-001 | 2026-09-10 | 라벨(정답)이 뒤바뀐 통화가 있는지 첫 점검 — 정확도가 유난히 낮은 통화의 대화 내용을 사람이 딱 한 번 읽어서 확인(모델 학습에는 이 텍스트를 안 씀) | 라벨이 통째로 뒤집힌 통화 4건 확정. 자동으로 훑어본 19,969/29,200건 중 의심 후보 17건 중 3건 추가 확정·2건 오탐·10건 보류 | 라벨 4건 수정 반영 | [`m2_label_swap_audit.md`](../reports/mission2/m2_label_swap_audit.md) |
| M2-LABEL-002 | 2026-09-14 | 라벨이 통화 중간에서만 바뀐 경우를 찾는 자동 탐지기 — "통째로 고쳤다"고 봤던 통화 중 일부가 사실은 중간부터만 뒤집혀 있었다는 걸 모델이 틀리는 패턴으로 발견 | 2건 추가 확정(28.57%→100%, 27.78%→100%) | 라벨 2건 추가 수정. 검증용으로 떼어둔 20%에서만 이 방법을 쓸 수 있음(나머지 80%는 모델이 이미 그 라벨로 학습해서 같은 방법이 안 통함) | [`m2_label_swap_audit.md`](../reports/mission2/m2_label_swap_audit.md) |
| M2-COMPARE-001 | 2026-09-18 | L1(피처 기반) 모델 5종을 공정 비교(GaussianNB/로지스틱회귀/RandomForest/XGBoost/LightGBM, 규정 위반 없는 순정 53차원 피처, 같은 분할 기준) + LightGBM 최종 튜닝 | **LightGBM 튜닝 86.77%로 최고**(84.59% 기준 대비 +2.18%p) | L1 최종 후보 확정 | [`m2_l1_model_comparison_results.md`](../reports/mission2/m2_l1_model_comparison_results.md) |
| M2-EXP-009 | 2026-09-18 | 대원 쪽 정답률(recall)을 올려보려는 시도 — 정답 기준선(threshold 0.5→0.75)과 class_weight(1.0→3.0)를 여러 값으로 바꿔봄 | 모든 조정이 전체 정확도 손해(threshold=0.55: −0.20%p) | 기각(적용 안 함) — 대회 채점이 정확도 하나만 보기 때문에 오히려 손해 | [`m2_l1_daewon_recall_push_results.md`](../reports/mission2/m2_l1_daewon_recall_push_results.md) |
| M2-VAL-001 | 2026-09-18 | 주최측 공식 Validation(통화 3,640건/발화 111,947개)으로 평가, LightGBM 튜닝 모델을 다시 학습하지 않고 그대로 1회만 확인 | **86.40%**(내부 확인 86.77% 대비 −0.37%p) | **확정 — L1 공식 후보** | [`m2_l1_official_validation_results.md`](../reports/mission2/m2_l1_official_validation_results.md) |
| M2-CNN-001 | 2026-09-10 | L2(소리를 이미지처럼 바꾼 mel-spectrogram + CNN) 파이프라인을 처음 만들고, 통화 3,000건으로 소형 CNN(SpeakerCNN) 시험 실행 | 86.56%(수렴). 같은 통화 수 기준 L1 최고 기록보다 높지만, 대원/신고자 정답률 격차는 4.95%p→9.20%p로 더 벌어짐 | 참고용 (시험 실행 단계) | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-002 | 2026-09-14 | L2 CNN을 전체 통화(29,200건)로 확대, 여전히 소형 모델(SpeakerCNN) | 89.71%(수렴), 격차도 27.57%p→7.66%p로 축소 | 참고용 | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-003 | 2026-09-14 | L2 CNN을 더 큰 모델(SpeakerCNNLarge, 층 4개, 파라미터 42만 개)로 교체 | **91.22%**(수렴), 격차 7.27%p | 우리끼리 나눈 검증에서는 최고 기록. 공식 Validation은 아직 안 해봤고 팀 공용 main에도 안 합쳐짐 | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-004 | 2026-09-16 | 대원/신고자 격차를 줄여보려는 시도 A: 발화 길이(duration_sec)를 모델에 직접 알려주는 SpeakerCNNLargeWithDuration | 정확도 +0.30%p이나 격차는 8.45%p→9.17~9.35%p로 확대 | 기각 — "격차의 진짜 원인"이라는 가정을 제대로 검증 못 함 (**2026-09-27: 이 기각 자체가 재검증에서 뒤집힘 → M2-CNN-006 참조**) | [`m2_l2_cnn_results.md`](../reports/mission2/m2_l2_cnn_results.md) |
| M2-CNN-005 | 2026-09-16 인프라, pilot 평가 09-23 | 격차를 줄여보려는 시도 B: 소리를 이미지로 바꿀 때 시간 간격(hop-length)을 절반으로 줄여 더 촘촘하게 만듦 — 통화 3,000건으로 기존 방식(hop160)과 새 방식(hop80)을 비교. SpeakerCNNLarge를 20번 학습시킨 뒤 15번 더 학습시켜(총 35번) 다시 비교 | 1차(20번 학습, 아직 다 안 배운 상태): 목표로 삼은 짧은 발화 구간에서 <0.5초 +1.15%p, 1~2초 +0.70%p로 새 방식이 더 나음. 2차(35번 학습, 더 배워도 그대로라 수렴 확인): 같은 두 구간이 **-0.19%p / -0.32%p로 정반대로 뒤집힘**, 전체 정확도 우위도 뒤집힘(+0.27%p→-0.14%p) — 원인은 실행마다 모델 초기값과 데이터 섞는 순서를 고정하지 않아서, 짧은 발화가 매번 다른 발화들과 묶여 학습되는 바람에 진짜 효과인지 우연인지 구분이 안 된 것 | **기각 — 다시 해봐도 똑같이 나온다는 보장이 없어서, 전체 데이터(3시간+ 예상)까지 갈 필요 없다고 판단.** "이 아이디어 자체가 틀렸다"가 아니라 "이번 확인 방법으로는 맞는지 틀린지 알 수 없다"는 뜻 | `M2_Colab_L2_hop80_실험_스크립트.py`(DCC 루트, git 미포함) |
| M2-AUDIT-001 | 2026-09-20 | PR #4/#9/#11/#14를 나중에 다시 살펴봄 — 본인이 자기 PR을 직접 승인해서 합친 건 아닌지, 코드에 문제는 없는지 확인 | PR #9/#11/#14가 실제로 셀프 머지였음을 확인. 그 과정에서 `build_mission2_dataset.py --resume`이 통화 29,200개 중 2,170개(7.4%)를 아무 경고 없이 빠뜨리는 버그를 발견 | 버그를 고친 PR #19는 팀원 검토 후 머지(9/21 완료) — 빠졌던 통화는 재처리·재학습 대기 | `scripts/build_mission2_dataset.py`, PR #19 |
| M2-AUDIT-002 | 2026-09-20 | L2 CNN 관련 브랜치/PR들이 서로 잘 맞물려 있는지 점검 | `mission2-l2-cnn-clean`(PR #17, 미머지)에 라벨 수정 내용이 통째로 빠져 있는 걸 발견 — 파일만 그대로 가져오고(cherry-pick) 그 파일에 있던 수정 내용은 놓친 것 | L2 작업 재개 전 라벨 수정을 반영해야 함 — 9/22 완료(아래 M2-VAL-002 참조) | 세션 기록만(별도 리포트 없음) |
| M2-LABEL-003 | 2026-09-22 | 라벨을 어느 방향으로 뒤집어야 하는지 재확인 — 통화 `651e4c754428de718c773b56`의 대화 21개를 전부 다시 읽음. 일부 구간만 뒤집는 경계값을 계산할 때, 이미 한 번 뒤집은 라벨을 기준으로 계산해버려서 방향이 반대(`<` 써야 할 걸 `>=`로 씀)로 적용돼 있었던 걸 발견 | 21개 발화 전부와 100% 일치하는 방향으로 수정, `PARTIAL_LABEL_SWAP_FLIP_BEFORE_UTT_IDX` 규칙 추가 | 라벨 처리 코드 수정, 같은 실수 재발 방지용 테스트 추가(`test_mission2_label_swap_fix.py`) | `src/dcc_data/mission2_features.py`, `tests/test_mission2_label_swap_fix.py` |
| M2-VAL-002 | 2026-09-21~22 | M2-AUDIT-001 후속 — 빠졌던 통화 재처리(발화 889,358개, 자세한 숫자는 `m2_exp_log_postfix2.json` 참고) + M2-LABEL-003 라벨 수정 반영해 재학습, 공식 Validation 재평가(1회만, 추가 튜닝 없이) | 내부 확인 **86.93%**(전보다 +0.16%p), 공식 Validation **86.44%**(전보다 +0.04%p, 거의 그대로 — 빠졌던 통화가 원래 학습용 쪽에 있었다는 뜻) | **확정 — M2-VAL-001을 대체하는 L1 최종 후보**(`m2_baseline_lgbm_tuned_postfix2.pkl`) | [`m2_exp_log_postfix2.json`](../reports/mission2/m2_exp_log_postfix2.json), [`m2_l1_official_validation_result_postfix2.json`](../reports/mission2/m2_l1_official_validation_result_postfix2.json) |
| M2-VAL-003 | 2026-09-22~23 | L2(SpeakerCNNLarge, M2-CNN-003 모델에 라벨 수정 반영해 재학습)를 처음으로 공식 Validation에 평가(1회만, 추가 튜닝 없이) — 내 컴퓨터(CPU)와 Colab(GPU) 양쪽에서 같은 저장 모델(21번째 학습)로 재확인 | **90.88~90.89%**(CPU 90.89%/GPU 90.88% — 컴퓨터 종류에 따른 계산 오차로 틀린 답 약 250건이 서로 달랐을 뿐, 같은 모델을 두 컴퓨터에서 잰 것이라 "1회만 확인" 원칙 위반 아님). L1(M2-VAL-002)보다 **+4.44~4.45%p 높음** | **확정 — 지금까지 우리 기록 중 최고, 제출 후보 1순위**. 다른 팀들 기록 중 제일 높았던 것(eunkyo00, 90.25%)보다도 높음 | [`m2_l2_official_validation_result_local_cpu.json`](../reports/mission2/m2_l2_official_validation_result_local_cpu.json), [`m2_l2_official_validation_result_colab_gpu.json`](../reports/mission2/m2_l2_official_validation_result_colab_gpu.json) |
| M2-COMPARE-002 | 2026-09-23 | L1(M2-VAL-002)과 L2(M2-VAL-003)를 발화 하나하나 단위로 비교(공식 Validation 111,947개) — 목소리 겹침 여부·발화 길이·정답 클래스별 정확도, 두 모델이 같이 맞았는지·틀렸는지도 확인 | 전체 L1 86.44% / L2 90.88%(+4.4%p). 안 겹침(63,084개) L1 90.01%/L2 92.44%(+2.43%p) vs 겹침(48,863개) L1 81.83%/L2 88.88%(+7.05%p) — L2가 잘하는 부분이 겹치는 목소리 구간에 몰려 있음. 발화 길이로도 전 구간 L2 우세, 특히 0.5~2초 짧은 발화에서 차이 최대(0.5-1초 81.9%→86.9%, 1-2초 84.2%→89.0%). 둘 다 맞음 94,545개 / 둘 다 틀림 7,980개 / L1만 틀림(L2가 고침) 7,198개 / L2만 틀림(L1이 고침) 2,224개 — L2가 L1의 오답을 고치는 경우가 반대보다 약 3.2배 많음 | 참고용 — L2가 왜 더 나은지(겹침·짧은 발화 처리) 숫자로 확인, L2를 쓰기로 한 결정(2026-09-23 decisions.md)을 뒷받침. 두 모델을 합치는(앙상블) 방법은 아직 안 정함 | [`m2_l1_l2_official_validation_comparison.json`](../reports/mission2/m2_l1_l2_official_validation_comparison.json), [`m2_l1_l2_official_validation_comparison.csv`](../reports/mission2/m2_l1_l2_official_validation_comparison.csv) |
| M2-AUDIT-003 | 2026-09-23 | hop80 실험 준비하다가 제출용 코드도 점검 — 대회가 정해준 실행 방식인 `scripts/mission2_inference.py`가 L2(CNN)도 돌릴 수 있는지 확인 | **L1 전용으로 확인됨** — L1(GBM) 예측 코드만 있고 L2(CNN) 코드는 전혀 없음. 지금까지 만든 모든 브랜치를 찾아봐도 L2용 실행 코드가 없음 | **아직 안 풀림 — 제출 전에 반드시 해결해야 함.** 9/23에 "L2를 제출하기로" 정했었지만(decisions.md), 지금 이대로는 L2를 제출할 방법이 없음 — L2를 제출하려면 `mission2_inference.py`를 새로 만들거나 고쳐서 CNN 모델을 읽고 소리를 처리해서 예측하는 코드를 넣어야 함 | `scripts/mission2_inference.py` |
| M2-CNN-007 | 2026-09-27 | 주파수 해상도(n_mels) 확대 — 소리를 이미지로 바꿀 때 주파수를 40단계 대신 80단계로 나눔(`n_mels` 40→80, 시간 간격은 그대로). `--torch-seed` 1/2/3으로 3번씩 학습(통화 3,000건, 각 35번 학습, 데이터 나누는 방식은 고정) | 전체 정확도 차이(n_mels=80−40): -0.58%p/-0.63%p/-0.82%p로 **3번 다 나쁨**(평균 -0.68%p, 같은 조건 반복 실행 편차 0.14~0.15%p의 4~5배 커서 우연이 아님). <0.5초 구간이 제일 크게 나빠짐(-2.29%p), 0.5~2초도 일관되게 나쁨. 2초 이상은 거의 차이 없음(부호도 섞임) — 나쁜 영향이 정확히 짧은 발화에 집중됨 | **기각 — 재현이 아니라 명확하게 나쁨.** 발화 길이 알려주기(M2-CNN-004/006), 시간 해상도 확대(M2-CNN-005)에 이어 "입력 해상도를 올려서 짧은 발화를 돕는다"는 방향 세 번째도 실패 — 이 방향 자체를 접기로 함 | [`m2_cnn_nmels_seed_recheck.md`](../reports/mission2/m2_cnn_nmels_seed_recheck.md) |
| M2-CNN-006 | 2026-09-26~27 | M2-CNN-004를 다시 확인 — `train_mission2_cnn.py`에 `--torch-seed`(모델 초기값 + 데이터 섞는 순서를 고정하는 값, 데이터 나누는 seed와는 별개)를 추가하고, `large`와 `large-duration` 두 모델을 이 값 1/2/3으로 바꿔가며 3번씩 학습(통화 3,000건, 각 35번 학습, 데이터 나누는 방식은 고정) | large-duration이 large보다 나은지 3번 비교: -0.21%p/-0.22%p/-0.01%p로 **3번 다 오히려 더 나쁨**(평균 -0.15%p). 원래 목표였던 0.5~1초 구간도 -0.77%p/-0.95%p/-0.39%p로 **3번 다 더 나쁨**(평균 -0.70%p, 실행 편차로 보기엔 너무 큰 차이). <0.5초 구간은 좋아졌다 나빠졌다 왔다갔다 해서 판단 불가 | **기각 — M2-CNN-004에서 봤던 +0.30%p 개선은 다시 해보니 안 나옴, 오히려 나빠지는 쪽.** M2-CNN-005 때와 똑같이 "그때는 우연이 신호처럼 보였다"는 결론 | [`m2_cnn_duration_seed_recheck.md`](../reports/mission2/m2_cnn_duration_seed_recheck.md) |

> M2-SCALE-001 이후 실험 중 따로 저장된 JSON 로그가 없는 건 보고서를 근거 자료로 씁니다(위
> 표에서 "별도 committed 리포트 없음"/"세션 기록만"이라고 적힌 것들). 나중에 다시 실행할 땐
> 실험 ID·설정·결과를 JSON 파일로도 같이 남겨야 합니다.

## Mission 3 — 환자 증상 인식 (F1 Macro, 9개 증상)

| 실험 ID | 날짜 | 실험 | 결과 | 상태 | 관련 파일 |
|---|---|---|---|---|---|
| M3-EXP-001 | 2026-09-11 | 단어 빈도 기반(TF-IDF) + 클래스별 로지스틱회귀(One-vs-Rest LR), 통화(`recordId`) 단위로 여러 라벨 비율 맞춰 나눔 | PR 문서에 결과를 나중에 채운다는 placeholder만 있고 실제 숫자는 안 적혀 있음 | 미확정 | [`mission3_baseline.py`](../scripts/mission3_baseline.py), [`M3_PR.docx`](../scripts/M3_PR.docx) |
| M3-EXP-002 | 2026-09-19 | 단어 빈도 기반(TF-IDF) + 로지스틱회귀, JSON 한 줄(발화) 단위로 여러 라벨 비율 맞춰 나눔, seed=42 | 보고값: Macro F1 0.3955→0.4530, Precision 0.5639 / Recall 0.4110 | 보류 — 데이터를 나눈 방식이 맞는지 확인 전이라 비교할 수 없음 | [`mission3_baseline 최종 수정본.py`](<../scripts/mission3_baseline 최종 수정본.py>), [`F1 성능 검증 과정 및 결과.docx`](<../scripts/F1 성능 검증 과정 및 결과.docx>), [`decisions.md`](./decisions.md) |

> M3는 아직 정리된 실험 로그가 없습니다. 다음에 실행할 때는 전체 행 수, 고유한 `recordId` 개수,
> 중복된 그룹 수, 학습/검증에 같은 `recordId`가 겹치는지, 어떻게 나눴는지와 지표를 JSON으로
> 저장해야 합니다.

---

새 실험은 해당 Mission 표의 맨 아래에 추가합니다.
