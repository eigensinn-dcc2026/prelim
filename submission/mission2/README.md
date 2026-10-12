# Mission 2 — 화자 분류 (대원 0 / 신고자 1)

log-mel 스펙트로그램(40 mel, 8 kHz, FFT 512, hop 160) + CNN(`SpeakerCNNLarge`, 421,954 파라미터) 3개를
torch seed 1/2/3으로 학습해 소프트보팅(평균 확률 >= 0.5 이면 1)한다. 사전학습 모델, 외부 데이터, 외부 API는 쓰지 않는다.

## 추론 (채점 명령)

```
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} \
    --ckpt_path checkpoints/m2_cnn_large_rerun_ensemble.pt --output ./outputs/mission2.csv
```

- 발화 하나하나를 그 발화의 `startAt`/`endAt` 구간만 잘라 따로 추론한다. 라벨 JSON에서는 `startAt`/`endAt`만 읽는다
  (`text`, 발화 순서, 같은 통화의 다른 발화는 쓰지 않음).
- 정규화와 길이 맞춤(짧은 발화를 무음으로 300프레임까지 채움)도 그 발화 자신만 사용한다.
- 출력 CSV: 첫 줄은 출제문제 11쪽 컬럼 이름 그대로 `audio file name,startAt,endAt,speaker`, `speaker`는 정수 0/1,
  `startAt`/`endAt`은 라벨 JSON 값 그대로. 행 순서는 파일 이름 정렬, 같은 파일 안에서는 라벨 JSON의 `utterances` 순서
  (순서는 출력에만 쓰고 예측에는 쓰지 않음). 라벨 JSON의 발화 수와 출력 행 수가 다르면 CSV를 쓰지 않고 멈춘다.
- 앙상블 체크포인트 하나에 모델 3개가 들어 있다. CPU에서 동작하고, 인터넷 연결이나 외부 파일이 필요 없다.
- 실행 환경: **Python 3.12** + `requirements.txt` (고정 버전). 새로 만든 Python 3.12 가상환경에 `requirements.txt`만 설치해서
  위 명령으로 공식 Validation 전체를 실행해 확인했다 (아래 "제출 전 검증").

## 학습 (재현)

`train_mission2_l2_from_raw.ipynb` 하나가 대회 제공 원본 zip → 음성 추출 → 스펙트로그램 생성 → 학습(시드 3개)
→ 앙상블 → 공식 Validation 평가까지 이 폴더의 코드만으로 실행한다. 에폭별 학습 로그와 최종 결과가 노트북 출력에 남아 있다.

- 학습 데이터: Training 서울 29,200통화. 라벨은 제공된 JSON의 `speaker` 그대로. Validation은 학습에 쓰지 않는다.
- 최고 에폭 선택: Training 안에서 통화 단위로 떼어 둔 20%(`--seed 42`).
- Adam 1e-3, 배치 64, ReduceLROnPlateau(0.5배, patience 2), 최대 30에폭, 5에폭 연속 개선이 없으면 조기 종료.
- 다시 실행하려면: Colab(GPU)에서 노트북 1번 셀의 `DRIVE_ROOT`/`CODE_DIR`/`ZIP_DIR`를 이 폴더와 원본 zip이 있는 Drive
  경로로 바꾸고 "모두 실행". 예선에서는 T4 GPU 1개로 데이터 준비 약 1시간 + 학습 약 8시간 걸렸다.
- **실행 기록 안내**: Colab 런타임이 끊기면 Drive에 저장된 마지막 에폭부터 이어서 학습하도록 했다(난수 상태까지 이어받음).
  ts2의 7번째, ts3의 9번째 에폭은 학습이 끝난 직후 Drive 저장 전에 런타임이 끊겨 직전 저장 지점부터 다시 학습했으므로
  로그에 두 번 나타난다. 최종 모델에는 다시 학습한 쪽이 반영됐고, 빠진 에폭은 없다. 끊긴 시점은 로그의
  "세션 재시작 -- 같은 실행을 이어감" 줄로 남아 있다.

## 결과와 계산 효율성 (학습 2026-10-05~06, 노트북 로그 기준)

| 항목 | 값 |
|---|---|
| 공식 Validation 정확도 | **91.49%** (102,419 / 111,947 발화), 대원 recall 87.15% / 신고자 recall 95.51% |
| (1) 파라미터 수 | 모델 1개 421,954 × 3개 = **Total 1,265,862 / Active 1,265,862** (발화마다 3개를 모두 실행) |
| (2) 컴퓨팅 환경 | 학습: Colab NVIDIA Tesla T4 1개(혼합 정밀도), 에폭당 약 7분, 총 66에폭(시드별 22/24/20, 조기 종료). 추론: Colab CPU Intel Xeon 2.00GHz 2코어, GPU 미사용 |
| (3) Validation 추론 batch size | **1** (발화 하나씩 따로 추론) |
| (4) Validation 추론 시간 | 전체 **87.1분**, 발화당 평균 **46.7ms** (음성 읽기와 스펙트로그램 계산 포함, 위 CPU 기준, 노트북 마지막 실행 2026-10-12). 같은 환경의 첫 측정(2026-10-06)은 85.3분 / 45.7ms |

## 제출 전 검증 (2026-10-12)

이 폴더 그대로(코드 + `checkpoints/`), 새로 만든 Python 3.12 가상환경에 `requirements.txt`만 설치하고 위 채점 명령으로
공식 Validation 전체(3,640통화, 111,947발화)를 실행했다.

- 결과 CSV의 예측값과 행 순서가 학습 노트북(Colab)의 예측과 **바이트 단위로 동일** (111,947행, 정확도 91.49%, 대체 예측 0건). 첫 줄만 출제문제 11쪽의 컬럼 이름(`audio file name`)으로 쓴다.
- 소요 시간: 133.6분과 154.4분 (두 번 실행, Windows 노트북, Intel Core i5-1240P, CPU만 사용, batch 1). 같은 PC에서 다른 작업이 함께 돌면 느려진다.

## 파일

| 파일 | 역할 |
|---|---|
| `inference.py` | 채점용 진입점 (구현은 `scripts/mission2_inference.py`) |
| `scripts/extract_dcc_sample_from_zips.py`, `scripts/build_manifest.py` | 원본 zip 풀기, wav-json 짝 목록 |
| `scripts/build_mission2_spectrograms.py`, `scripts/pack_mission2_spectrograms.py` | 발화별 스펙트로그램 생성, 학습용 파일 하나로 묶기 |
| `scripts/train_mission2_cnn.py` | 학습 |
| `scripts/pack_mission2_l2_ensemble.py` | 모델 3개를 체크포인트 하나로 묶기 |
| `src/dcc_data/mission2_cnn.py` | **모델 코드** (`SpeakerCNNLarge`), 학습 루프 |
| `src/dcc_data/` (나머지) | 스펙트로그램 생성, 데이터 분할, 체크포인트 검사, 규정 검사(Training 전용, 제공 라벨) 코드 |
| `SOURCE_COMMIT.txt` | 이 코드를 가져온 팀 저장소 커밋 |
| `checkpoints/m2_cnn_large_rerun_ensemble.pt` | 학습된 앙상블 가중치 (sha256 `19ea482a80a2ceb2ea5b87981be0a381a11e01fd576b3ba6860ee5d7144e3246`) |
| `train_mission2_l2_from_raw.ipynb` | 학습 노트북 (에폭별 로그와 결과 포함) |
