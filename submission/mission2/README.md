# Mission 2 — 화자 분류 (대원 0 / 신고자 1)

발화 하나의 소리만 보고 대원(0)인지 신고자(1)인지 고른다. 성격이 다른 모델 두 종류를 섞는다.

| 구성 | 내용 | 공식 Validation (단독) |
|---|---|---|
| CNN 앙상블 | log-mel 스펙트로그램(40 mel, 8 kHz, FFT 512, hop 160) + `SpeakerCNNLarge` 3개(torch seed 1/2/3), 처음부터 학습 | 91.49% |
| WavLM-large 1 | 공개 사전학습 모델 `microsoft/wavlm-large`를 미세조정 (seed 1) | 92.40% |
| WavLM-large 2 | 같은 방법, seed 2 | 92.71% |

**최종 판정**: `p = 0.4 × P_CNN(신고자) + 0.3 × P_WavLM1(신고자) + 0.3 × P_WavLM2(신고자)`, `p >= 0.5`이면 1, 아니면 0.
비율(0.4 : 0.3 : 0.3)은 Training 안에서 떼어 둔 발화 20,000개로 골랐다(공식 Validation으로 고르지 않음). 판정 기준은 0.5 고정.

공식 Validation 정확도(3,640통화, 111,947발화): **92.90%** (103,994 / 111,947 = 92.8957%).
이 폴더를 압축한 파일 그대로 Colab A100에서 채점 명령으로 실행해 잰 값이다(아래 "제출 전 검증"). 모델별 확률을 따로 저장했다가 섞은
연구용 계산(92.894%)과는 발화 2개가 다르다.

## 추론 (채점 명령)

```
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} \
    --ckpt_path checkpoints/m2_final_ensemble.pt --output ./outputs/mission2.csv
```

- **실행 환경**: Python 3.12 또는 3.13 + `requirements.txt`(고정 버전). 공식 Validation 전체 검증은 Colab의 Python 3.13.15에서,
  PC 대조 시험은 Python 3.12.13에서 했다. GPU가 있으면 자동으로 GPU를 쓴다(WavLM은 float16 autocast, 정확도를
  잰 설정과 같음). GPU가 없으면 CPU로 돌지만 매우 느리다(아래 계산 효율성 참고).
- **인터넷 불필요**: 체크포인트 하나(`checkpoints/m2_final_ensemble.pt`)에 CNN 3개, WavLM 2개의 가중치와 모델 구조 설정, 섞는 비율이
  모두 들어 있다. 실행 중 아무것도 내려받지 않는다(`HF_HUB_OFFLINE=1`로 확인).
- 발화 하나하나를 그 발화의 `startAt`/`endAt` 구간 소리만 잘라 추론한다. 라벨 JSON에서는 `startAt`/`endAt`만 읽는다
  (`text`, 발화 순서, 같은 통화의 다른 발화는 쓰지 않음).
  - CNN: 그 구간의 스펙트로그램, 짧으면 무음(고정된 바닥값)으로 300프레임까지 채움.
  - WavLM: 그 구간을 16 kHz로 바꾼 뒤 가운데 3초(짧으면 그 발화를 반복해 3초), 그 3초 안에서만 평균 0·분산 1로 정규화.
  - 속도를 위해 한 통화의 발화들을 WavLM에 한 묶음으로 넣지만, 발화끼리 섞이는 계산이 없어 결과는 하나씩 넣을 때와 같다
    (BatchNorm은 학습 때 고정된 값만 쓰고, WavLM은 발화별 layer normalization. 18발화로 확인: 확률 최대 차이 9e-9).
- **출력 CSV**: 첫 줄은 출제문제 11쪽 컬럼 이름 그대로 `audio file name,startAt,endAt,speaker`, `speaker`는 정수 0/1,
  `startAt`/`endAt`은 라벨 JSON 값 그대로. 행 순서는 파일 이름 정렬, 같은 파일 안에서는 라벨 JSON의 `utterances` 순서
  (순서는 출력에만 쓰고 예측에는 쓰지 않음). WAV와 JSON의 짝이 안 맞거나 출력 행 수가 라벨 발화 수와 다르면 CSV를 쓰지 않고 멈춘다.

## 대회 규정 확인

| 규정 | 이 제출물 |
|---|---|
| 공개 사전학습 모델 사용 가능, 채점 때 다운로드 금지 | `microsoft/wavlm-large`(Hugging Face 공개 모델)를 미세조정. 가중치는 체크포인트에 포함 |
| 상용 AI API 금지 | 사용하지 않음 |
| 제공 데이터만 사용 | 대회 제공 Training(서울)만 학습에 사용 |
| Validation으로 학습(역전파) 금지 | 공식 Validation은 평가에만 사용. 섞는 비율도 Training 안의 떼어 둔 발화로 고름 |
| 발화 하나의 정보만 사용 | 위 "추론" 참고. 학습도 발화 하나씩(같은 통화 정보 없음) |
| `text`로 라벨 보정 금지 | 제공된 `speaker` 라벨을 그대로 학습 (CNN 체크포인트에 `label_source: provided_json` 기록, 실행 시 검사) |
| 판정 기준 0.5 고정 | 섞은 확률 `>= 0.5`. 클래스별 기준이나 기준 이동 없음 |

참고로 공식 Validation은 모델들의 성능을 비교하고 최종 구성을 정할 때 여러 번 평가에 썼다(규정이 허용하는 모델 선택 용도).
학습에는 한 번도 쓰지 않았다.

## 학습 (재현)

공통: Training 서울 29,200통화, 라벨은 제공된 JSON의 `speaker` 그대로. 통화 단위로 20%(5,840통화)를 떼어 두고(`--split-seed 42`,
CNN과 WavLM이 같은 통화를 떼어 둠) 나머지 80%로만 학습했다. 떼어 둔 통화는 최고 시점 선택과 섞는 비율 선택에만 썼다.

**CNN 앙상블** — `train_mission2_l2_from_raw.ipynb` 하나가 원본 zip → 음성 추출 → 스펙트로그램 → 학습(시드 3개) → 앙상블 →
공식 Validation 평가까지 실행한다. Adam 1e-3, 배치 64, ReduceLROnPlateau(0.5배, patience 2), 최대 30에폭, 5에폭 연속 개선이 없으면
조기 종료. Colab T4에서 데이터 준비 약 1시간 + 학습 약 8시간. 런타임이 끊긴 구간은 노트북 로그에 "세션 재시작" 줄로 남아 있다.
노트북의 마지막 단계(CNN 앙상블만 공식 Validation 평가)는 당시의 `inference.py`로 실행했는데, 지금 `inference.py`는 최종 묶음
체크포인트용이다. CNN만 다시 평가하려면 같은 코드인 `python scripts/mission2_inference.py --ckpt_path <CNN 앙상블 .pt> ...`를 쓴다.

**WavLM-large** — `scripts/train_mission2_wavlm.py` (실행 기록: `train_mission2_wavlm_seed1.ipynb`, `train_mission2_wavlm_seed2.ipynb`.
각 노트북 맨 위의 "제출용 안내" 셀에 실행 이력을 적었다)

- 모델: WavLM-large(24층) 25개 층 출력의 학습되는 가중합 → 시간축 평균·표준편차 → MLP(256) → 2개 클래스. 파형 CNN 부분만 고정,
  나머지 전부 미세조정.
- 입력: 위 "추론"과 같은 3초 창. 학습 때만 3초를 발화 안의 무작위 위치에서 자른다(두 클래스에 똑같이 적용).
- AdamW(본체 2e-5, 분류 head 3e-4, weight decay 0.01), 500 step 웜업 뒤 코사인으로 0까지 감소, 처음 300 step은 head만 학습,
  배치 32, float16 혼합 정밀도, 기울기 클리핑 5.0, 42,500 step. Colab A100(40GB), step당 약 0.18초.
- 1,000 step마다 떼어 둔 발화 12,000개로 정확도를 재서 최고 시점을 `best.pt`로 저장. 학습이 끝난 뒤 `best.pt`와 마지막 step 모델을
  **새로 뽑은 떼어 둔 발화 20,000개**로 다시 비교해 더 좋은 쪽을 골랐다(두 시드 모두 `best.pt`).
  - seed 1: step 15,000 모델 (발화 20,000개 정확도 92.73%, 마지막 step 92.60%)
  - seed 2: step 20,000 모델 (발화 20,000개 정확도 92.99%, 마지막 step 92.57%)
- **seed 1 실행 기록 안내**: 첫 Colab 계정의 컴퓨팅 단위가 바닥나 학습이 중간에 끊겼고, Drive에는 같은 설정으로 학습한 step 1,000
  체크포인트만 남았다. 새 계정에서 그 step 1,000 가중치로부터 이어서(옵티마이저는 새로 시작, 200 step 동안 학습률을 다시 올림)
  42,500 step까지 학습했다. 제출한 seed 1 모델은 이 두 번째 실행의 결과다.

**섞는 비율 선택** — `scripts/predict_mission2_wavlm.py`, `scripts/predict_mission2_cnn_probs.py`로 세 모델의 P(신고자)를 떼어 둔
발화 20,000개에 대해 구한 뒤, `scripts/choose_mission2_blend_weights.py`로 0.1 간격의 모든 비율을 비교했다.
최고는 CNN 0.4 / WavLM 0.3 / 0.3(93.21%). 최고에서 0.05%p 이내인 비율이 여러 개라 결과가 비율에 민감하지 않다.
같은 비율을 공식 Validation에 적용한 결과가 92.89%(연구용 계산 92.894%, 제출 코드로는 92.8957%)다.
마지막으로 `scripts/pack_mission2_final_ensemble.py`가 세 체크포인트와 비율을 `checkpoints/m2_final_ensemble.pt` 하나로 묶는다.

참고로 다른 사전학습 모델(`facebook/hubert-large-ll60k`)도 같은 방법으로 미세조정해 보았다(공식 Validation 단독 91.88%).
하지만 떼어 둔 발화 20,000개에서 섞어도 정확도가 오르지 않아(93.210% → 93.215%) 최종 구성에서 뺐다.

## 제출 폴더 정리 (2026-10-13)

제출 직전에 최종 모델에 쓰이지 않는 코드를 지우고, 모든 설명문과 주석을 한글로 바꿨다(실행 코드는 아래 6곳 외에 그대로).

**지운 코드** (모두 최종 모델이 쓰지 않는 초기 실험용, 약 750줄)
- 초기 L1 모델(LightGBM): 53차원 음향 통계 특징과 그 데이터셋 만들기(`mission2_features.py`), 학습·저장·불러오기(`mission2_train.py`),
  추론 분기(`scripts/mission2_inference.py`). 그래서 `requirements.txt`에서 `lightgbm`도 뺐다.
- 쓰지 않은 CNN 실험 모델: 작은 `SpeakerCNN`, 발화 길이를 추가 입력으로 넣은 `SpeakerCNNLargeWithDuration`과 그 데이터셋·묶음 함수,
  `predict_with_confidence`(`mission2_cnn.py`), 학습 스크립트의 `--model small` / `--model large-duration` 선택지.
- 초기 데이터셋 클래스 `Mission2Dataset`(`mission2.py`), 쓰이지 않는 검사 함수 `ensure_training_batch`(`guards.py`).

**실행 코드가 바뀐 6곳** (그 밖의 코드는 주석과 설명문만 바뀜. 파일마다 "주석을 뺀 코드"를 정리 전과 비교해 확인)
| 위치 | 바뀐 내용 | 결과에 영향이 없는 이유 |
|---|---|---|
| `scripts/mission2_inference.py` `load_predictor` | `.pt`가 아닌 파일을 L1으로 넘기던 것을, 오류를 내고 멈추게 바꿈 | CNN 체크포인트(.pt) 경로는 그대로 |
| `scripts/train_mission2_cnn.py` 모델 선택지 표 3개 + `parse_args` | 선택지를 `large` 하나로, 기본값을 `small`에서 `large`로 | 학습 노트북은 항상 `--model large`를 명시함. 학습 흐름(`main`)은 그대로 |
| `src/dcc_data/mission2_wavlm.py` `configure_trainable` | HuBERT용으로 넣었던 분기를 빼고 WavLM을 학습한 당시의 코드로 되돌림 | 학습 당시 코드와 정확히 같음. 추론에는 쓰이지 않음 |

**정리 후 시험** (정리한 코드로 실행)
- 학습/검증 분할 재현: 학습 711,467발화/23,360통화, 떼어 둔 177,891발화/5,840통화, 분할 지문 `8d59ba35c2f327e0` — WavLM 학습 로그와 같음.
  공식 Validation 111,947발화/3,640통화.
- CNN 단독 추론(`scripts/mission2_inference.py`): 공식 Validation 3통화 90발화에서 검증된 CNN 판정과 다른 발화 0개.
- CNN 학습 전체 과정: Training 12통화로 스펙트로그램 만들기 → 묶기 → 학습(seed 1, 2) → `--resume` → 앙상블 묶기 → 추론까지 오류 없이 실행.
- 최종 추론(`inference.py`): 공식 Validation 3통화 90발화에서 정리 전 코드의 결과와 발화별 확률(소수점 8자리)·판정·출력 CSV가 완전히 같음.

## 노트북에 남은 실패 기록

세 학습 노트북은 실제 실행 기록을 그대로 담고 있어, 중간에 실패했다가 고쳐서 다시 실행한 기록도 남아 있다. 지우지 않고 그대로 두었고,
각 노트북 맨 위 "제출용 안내" 셀에 설명을 덧붙였다. 모두 원인을 고친 뒤 다시 실행해 성공했으며 모델과 결과에는 영향이 없다.

| 노트북 | 남은 기록 | 원인과 이후 |
|---|---|---|
| `train_mission2_wavlm_seed1.ipynb` | 마지막 셀 로그의 `FAILED (exit 2)` 9줄 (10/12 08:41~09:07) | 학습 시작 전 속도 측정 명령에서 띄어쓰기가 있는 Drive 경로를 따옴표로 감싸지 않아 명령이 거부됨. 고친 뒤 09:08부터 학습 정상 시작 |
| `train_mission2_wavlm_seed2.ipynb` | 7-2 셀의 `AttributeError` | 점수를 읽는 정규식의 `\d`가 노트북 생성기 실수로 `d`가 되어 있었음. 정규식만 고쳐 다시 실행해 모델 선택 완료(같은 셀 출력과 마지막 셀 로그에 선택 결과) |
| `train_mission2_l2_from_raw.ipynb` | 마지막 셀 로그의 `ValueError (0 WAV stems ...)` 2건 | 공식 Validation 폴더 경로를 잘못 줌(실제로는 `대학부 데이터/Validation`에 풀림). 추론 코드가 짝이 안 맞으면 CSV를 쓰지 않고 멈춰 잘못된 결과는 없음. 올바른 경로로 다시 실행해 111,947행 출력 |

## 결과와 계산 효율성

| 항목 | 값 |
|---|---|
| 공식 Validation 정확도 | **92.90%** (103,994 / 111,947), 대원 recall 88.36% / 신고자 recall 97.10% |
| (1) 파라미터 수 | CNN 421,954 × 3 = 1,265,862 + WavLM 315,977,179 × 2 = 631,954,358 → **Total 633,220,220 / Active 633,220,220** (발화마다 다섯 모델을 모두 실행) |
| (2) 컴퓨팅 환경 | 학습: CNN은 Colab Tesla T4, WavLM은 Colab A100-SXM4-40GB. 추론: Colab NVIDIA A100-SXM4-40GB 1개 (WavLM float16 autocast, CNN float32) |
| (3) Validation 추론 batch size | CNN 1 (발화 하나씩), WavLM 한 통화의 발화들(최대 64) |
| (4) Validation 추론 시간 | 전체 **20.2분**, 발화당 평균 **10.8ms** (음성 읽기, 리샘플링, 스펙트로그램 계산 포함, 위 GPU 기준, 최종 코드로 잰 값). 모델 로딩 27초 별도. 참고: GPU 없이 CPU(노트북 PC, Intel Core i5-1240P)만 쓰면 발화당 약 1.3~6.6초(메모리 상황에 따라) -- 11만 발화면 며칠이 걸려 GPU가 필요하다 |

## 제출 전 검증 (2026-10-13)

- **이 폴더 그대로 공식 Validation 전체 실행** (코드 정리 전과 후 두 번): 폴더를 압축한 파일을 Colab(A100)에서 그대로 풀고(체크포인트 sha256 일치 확인),
  `requirements.txt`만 설치한 뒤(Python 3.13.15, torch 2.11.0+cu130, transformers 5.19.0), Hugging Face 접속을 막은 상태(`HF_HUB_OFFLINE=1`)로 위 채점 명령을 실행했다.
  3,640통화 111,947행을 출력했고(대체 예측 0건), 라벨 JSON의 `speaker`와 비교한 정확도는 92.8957%, 행 순서와 `startAt`/`endAt`은 라벨과 같았다.
  코드 정리 후(최종 코드)에 같은 방식으로 다시 실행해 정확도 92.8957%, 정리 전 결과와 111,947발화 모두 판정이 같음을 확인했다
  (추론 20.2분, 발화당 10.8ms). 실행 기록: `verify_mission2_final_on_gpu.ipynb` (최종 코드의 재검증, 출력 포함).
- **연구용 계산과의 대조 (PC, CPU)**: 처음 3통화 90발화를 이 폴더의 `inference.py`로 돌려, 비율 선택에 쓴 Colab 확률 파일과 발화별로 비교했다.
  최종 판정이 다른 발화 0개, CNN 확률 차이 최대 1.5e-7, WavLM 확률 차이 최대 2.9e-3(CPU float32와 GPU float16의 차이).
- **코드 변경 확인**: WavLM 모델 코드를 일반화하기 전과 후로 같은 체크포인트를 불러 실제 발화를 넣었을 때 확률 차이 0, 발화 구간만 읽은 소리와
  통화 전체를 읽고 자른 소리의 차이 0, 18발화를 한 묶음으로 넣을 때와 하나씩 넣을 때의 확률 차이 최대 9e-9.

## 파일

| 파일 | 역할 |
|---|---|
| `inference.py` | 채점용 진입점 (구현은 `scripts/mission2_final_inference.py`) |
| `checkpoints/m2_final_ensemble.pt` | 최종 가중치 하나: CNN 3개 + WavLM 2개 + 비율 (sha256 `35e5e586d33a8f7ab5d0bf174ecbba33cf8de152395b9b97ea31a99bb73cac94`) |
| `scripts/mission2_final_inference.py` | 최종 추론 (CNN과 WavLM 확률을 섞어 0.5로 판정) |
| `scripts/mission2_inference.py` | CNN 단독 제출본의 추론 코드. 라벨 짝짓기·출력 검사 함수를 최종 추론이 그대로 가져다 씀 |
| `scripts/train_mission2_cnn.py`, `scripts/pack_mission2_l2_ensemble.py` | CNN 학습, CNN 3개를 하나로 묶기 |
| `scripts/train_mission2_wavlm.py` | WavLM 미세조정 |
| `scripts/predict_mission2_wavlm.py`, `scripts/predict_mission2_cnn_probs.py` | 떼어 둔 발화·공식 Validation의 발화별 확률 계산 (비율 선택용) |
| `scripts/choose_mission2_blend_weights.py` | 섞는 비율 비교 |
| `scripts/pack_mission2_final_ensemble.py` | 최종 체크포인트 만들기 |
| `scripts/extract_dcc_sample_from_zips.py`, `scripts/build_manifest.py` | 원본 zip 풀기, wav-json 짝 목록 |
| `scripts/build_mission2_spectrograms.py`, `scripts/pack_mission2_spectrograms.py` | CNN용 스펙트로그램 만들기 |
| `src/dcc_data/mission2_cnn.py` | CNN 모델 코드 (`SpeakerCNNLarge`) |
| `src/dcc_data/mission2_wavlm.py` | WavLM 모델 코드 (입력 창 만들기, 층 가중합 + 분류 head) |
| `src/dcc_data/` (나머지) | 음성 읽기, 데이터 분할, 체크포인트 검사, 규정 검사(Training 전용, 제공 라벨) |
| `train_mission2_l2_from_raw.ipynb` | CNN 학습 노트북 (에폭별 로그와 결과, 맨 위에 제출용 안내 셀) |
| `train_mission2_wavlm_seed1.ipynb`, `train_mission2_wavlm_seed2.ipynb` | WavLM 학습 노트북 (step별 로그, 모델 선택, 공식 Validation 평가) |
| `verify_mission2_final_on_gpu.ipynb` | 이 폴더 그대로 공식 Validation 전체를 돌린 검증 기록 (위 "제출 전 검증") |
| `SOURCE_COMMIT.txt` | 이 코드를 가져온 팀 저장소 커밋 |

## 사용한 공개 모델

- `microsoft/wavlm-large` (Hugging Face, Microsoft 공개). Hugging Face 모델 카드에는 라이선스 칸이 비어 있고, 원래 코드 저장소
  (microsoft/unilm)는 MIT 라이선스다. 가중치를 내려받아 미세조정한 뒤 우리 체크포인트 안에 저장했다.
