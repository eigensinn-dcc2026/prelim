# M2-CNN-006 — duration-conditioning(SpeakerCNNLargeWithDuration) seed 통제 재검증

## 배경

M2-CNN-004(2026-09-16)에서 `SpeakerCNNLargeWithDuration`(발화 자신의 `duration_sec`을
classifier head에 명시 입력)이 전체 정확도 +0.30%p, 대원/신고자 recall 격차 확대(8.45%p→
9.17~9.35%p)로 측정돼 "격차를 못 줄인다"는 이유로 기각됐다. 그런데 이 측정은 `--seed`가
train/val 분할에만 쓰이고 모델 초기화·`DataLoader` shuffle 순서는 통제되지 않은 1회 실행
결과였다. `collate_batch`는 배치마다 그 배치의 최장 발화 길이로 padding하므로, shuffle
순서가 다르면 짧은 발화가 어떤 발화들과 한 배치에 묶이는지도 달라진다 — M2-CNN-005(hop80
pilot)가 20에포크→35에포크 사이에 부호가 통째로 뒤집힌 것과 같은 종류의 confound다.

이번 재검증은 두 가지를 분리했다:
- `scripts/train_mission2_cnn.py`에 `--torch-seed`를 추가(모델 초기화 + train shuffle 순서).
  `--seed`(=42, train/val 분할)는 모든 실행에서 고정해 같은 held-out set을 유지.
- `--torch-seed`를 1/2/3으로 바꿔가며 `large`/`large-duration`을 각각 학습, 같은 torch-seed의
  두 모델은 shuffle 순서가 동일(같은 generator seed)하므로 배치 구성 confound가 쌍 안에서는
  제거된다.

판정 기준은 실행 전에 고정: 전체 accuracy 기준으로 3개 seed 전부 양수이고 평균 +0.5%p 이상이면
가설 지지(Phase B 진행), 부호가 갈리거나 평균 |차이| < 0.5%p면 잡음으로 종료.

## 설정

- 표본: 3,000통화 pilot(M2-CNN-005와 동일 표본, split seed=42).
- 모델: `SpeakerCNNLarge` vs `SpeakerCNNLargeWithDuration`, 각 35에포크(20 + `--resume` 15).
- torch-seed: 1, 2, 3 (모델 초기화 + train shuffle 순서만 변경, held-out set은 불변).
- 평가: 같은 held-out set(18,134발화)에서 duration bucket별 정확도.

## 결과

held-out val: 18,134개 발화 (`<0.5s` 1,570 / `0.5-1s` 4,400 / `1-2s` 5,869 / `2-4s` 4,240 /
`>=4s` 2,055)

| torch_seed | model | acc | best_epoch/total | <0.5s | 0.5-1s | 1-2s | 2-4s | >=4s |
|---|---|---|---|---|---|---|---|---|
| 1 | large | 0.8904 | 29/35 | 0.8745 | 0.8486 | 0.8731 | 0.9300 | 0.9601 |
| 1 | large-duration | 0.8883 | 22/35 | 0.8796 | 0.8409 | 0.8702 | 0.9318 | 0.9586 |
| 2 | large | 0.8876 | 34/35 | 0.8732 | 0.8486 | 0.8669 | 0.9288 | 0.9562 |
| 2 | large-duration | 0.8854 | 30/35 | 0.8682 | 0.8391 | 0.8680 | 0.9262 | 0.9635 |
| 3 | large | 0.8874 | 29/35 | 0.8720 | 0.8450 | 0.8669 | 0.9295 | 0.9616 |
| 3 | large-duration | 0.8873 | 32/35 | 0.8803 | 0.8411 | 0.8680 | 0.9283 | 0.9620 |

seed별 차이 (large-duration − large, %p):

| torch_seed | acc | <0.5s | 0.5-1s | 1-2s | 2-4s | >=4s |
|---|---|---|---|---|---|---|
| 1 | -0.21 | +0.51 | -0.77 | -0.29 | +0.19 | -0.15 |
| 2 | -0.22 | -0.51 | -0.95 | +0.10 | -0.26 | +0.73 |
| 3 | -0.01 | +0.83 | -0.39 | +0.10 | -0.12 | +0.05 |

요약:

| | mean_diff | std_diff | n_positive/3 |
|---|---|---|---|
| acc | **-0.15%p** | 0.12%p | 0 |
| <0.5s | +0.28%p | 0.70%p | 2 |
| 0.5-1s | **-0.70%p** | 0.29%p | 0 |
| 1-2s | -0.03%p | 0.23%p | 2 |
| 2-4s | -0.06%p | 0.23%p | 1 |
| >=4s | +0.21%p | 0.46%p | 2 |

같은 모델의 seed 간 accuracy std(순수 실행 노이즈 기준선): `large` 0.17%p, `large-duration`
0.15%p.

## 해석

- **전체 accuracy: 3개 seed 전부 음수(0/3), 평균 -0.15%p.** 판정 기준(전부 양수 & 평균
  ≥+0.5%p)에 명백히 못 미침 — 오히려 반대 방향. M2-CNN-004의 +0.30%p는 재현되지 않는다.
- **가설이 정확히 겨냥했던 0.5-1s 버킷도 3개 seed 전부 음수(0/3), 평균 -0.70%p** — 이 폭은
  같은 모델의 seed 간 noise 기준선(~0.15~0.17%p)보다 뚜렷이 크고 방향도 일관돼서, 우연한
  산포보다는 실제 효과(duration 입력이 이 버킷에 오히려 해가 됨)일 가능성이 있다. 다만 n=3라
  통계적으로 확정하기는 이르다.
- **<0.5s 버킷은 부호가 갈림(+0.51, -0.51, +0.83)** — 신호 없음, 판단 보류.
- M2-CNN-005(hop80)와 같은 결론: 두 실험 모두 "메커니즘이 틀렸다"가 아니라 "seed/배치 순서
  미통제 1회 실행이 노이즈를 신호로 오인하게 만들었다"는 것이 재현됨.

## 결론 및 후속

**기각.** `SpeakerCNNLargeWithDuration`을 채택하지 않는다. M2-CNN-004의 측정치는 무효로
처리한다. 짧은 발화(0.5~1초 backchannel) 병목에는 다른 메커니즘(예: `n_mels` 확대, duration
bucket 가중 샘플링)을 다음 후보로 검토한다. `--torch-seed`는 앞으로 이런 A/B 비교의 기본
절차로 유지한다.
