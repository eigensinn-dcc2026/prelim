# Mission 1 제출 폴더 (팀 eigensinn)

통화 녹음에서 신고자(speaker=1) 음성만 잘라 성별(F/M)을 분류합니다. 라벨 JSON에서는 `startAt`, `endAt`, `speaker`만 읽습니다.

## 구성

| 파일 | 설명 |
|---|---|
| `inference.py` | 공식 실행 명령 진입점 |
| `mission1_model.py` | 모델 코드(WavLM x-vector, 특징 결합, 예측) |
| `audio_features.py` | 신고자 구간 자르기, 52D 음향 특징, WAV·JSON 짝짓기. 팀 데이터 유틸에서 추론에 쓰는 18개 정의만 옮겼다(주석은 한글, 코드는 원본 그대로이며 40통화에서 파형과 52D 벡터가 원본과 비트 단위로 같음) |
| `requirements.txt` | 추론용 패키지, 버전 고정 |
| `m1_model.pt` | 가중치. 모델 전체를 `.pt` 한 파일에 담았다(형식 `dcc-m1-v1`): 표준화·로지스틱 회귀 헤드(564 입력, 텐서)와 고정 x-vector 인코더 `microsoft/wavlm-base-plus-sv`(revision `1bfd64eca136543feb28c5ffaf05381c6af33121`)의 가중치와 설정 텍스트. 약 386MB라 git에 없고 패키징 때 넣는다. SHA256 `b81a36d56fd55dd9d98fea08e8775082ee35d0c056b3952c6c53573c40de1905`. 인터넷 다운로드 없음 |
| `mission1_train.ipynb` | 최종 학습 노트북(실행 로그 포함). Training 29,200통화 특징 캐시로 80:20 holdout(0.9774)을 재현하고, 전체 데이터로 재학습한 모델이 제출 체크포인트(`m1_model.pt` 안의 헤드)와 파라미터까지 완전히 같음을 확인한다. 제출 체크포인트와 제출 코드로 Training 50통화를 다시 추출해 캐시와 일치(52D 오차 6e-5, x-vector 오차 1e-6)함도 보인다. WavLM 특징 추출(GPU 약 5시간)은 별도 스크립트로 실행해 만든 캐시를 읽는다. 실행에는 Drive의 특징 캐시와 Training 데이터가 필요하다 |
| `train/build_model_pt.py` | 학습한 헤드(sklearn 파이프라인)와 WavLM 스냅샷을 `m1_model.pt` 하나로 묶는 스크립트 |
| `train/requirements-train.txt` | 학습 노트북과 위 스크립트용 패키지(추론용 `requirements.txt`와 별개) |
| `train/extract_features_colab.py` | 위 노트북이 읽는 Training 특징 캐시(52D + 창별 x-vector)를 만든 추출 스크립트. 실제 실행에 쓴 파일이며(커밋 `bb9921b`, 주석만 한글로 옮기고 코드는 같다), 팀 저장소의 `dcc_data`와 Colab 러너 환경(`LAB_OUT`, `DCC_RUN_ROOT`)을 전제로 하므로 이 폴더만으로는 다시 실행되지 않는다. 같은 추출을 제출 코드(`mission1_model.extract_features`)가 하며, 노트북 5단계가 둘의 일치를 확인한다 |

## 실행

```bash
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path m1_model.pt --output ./outputs/mission1.csv
```

출력 컬럼은 `audio file name,gender`입니다. 특징 추출에 실패한 통화는 Training 다수 클래스 `F`로 채우고 stderr에 기록합니다(행 수 유지).

## 상태 (2026-10-12)

- 코드 경로 확인: 공식 Validation 20건(스모크)에서 실패 0건, 정답 20/20, 기존 `src/dcc_inference/mission1.py` 출력과 동일. 점수 근거로는 쓰지 않는다.
- **공식 Validation 3,640건 정확도 98.10%** (3,571건, 여성 recall 98.26%, 남성 recall 97.92%). 이 체크포인트(Training 29,200통화 전체로 재학습)를, M1-XVEC-002-OFFVAL이 Drive에 남긴 Validation 특징 캐시(52D + 다중 창 평균 x-vector)에 적용해 계산했다. 재학습·튜닝 없음(공식 Val 3회째 사용). 학습 80% 구간 모델(98.13%, 3,572건)과 예측 99.48%가 같고, 갈린 19건은 10 대 9로 차이가 없다(McNemar 정확검정 p=1).
- 전체 경로 확인: 공식 Validation 203건(3,640건 중 18건마다 1건)을 이 폴더의 `inference.py`와 `m1_model.pt` 하나로(인터넷 차단 환경, 외부 모델 폴더 없음) 처음부터 끝까지 실행한 결과가 캐시 기반 예측과 203/203 일치, 실패 0건. 이 203건의 정확도는 98.52%이며 점수 근거로는 쓰지 않는다. 전체 3,640건을 `inference.py`로 끝까지 돌린 것은 아니다(CPU로 약 3.6시간).
- 가중치를 `.pt` 한 파일로 묶은 뒤 검증: 복원한 WavLM의 모든 가중치가 원본과 같고 x-vector 출력 차이가 0이며, `.pt`의 numpy 헤드가 sklearn 파이프라인과 캐시된 Validation 3,640건·Training 29,200건에서 모두 같은 예측을 낸다.

## 계산 효율 정보

| 항목 | 값 |
|---|---|
| 파라미터 수 | Total 101,112,718개 (Dense 모델이라 Active도 동일). WavLM-base-plus-sv 101,112,153 + 표준화기·로지스틱 회귀 헤드 565(+표준화 통계 1,128) |
| 컴퓨팅 환경 | 학습: 특징 추출은 Colab Tesla T4, LR 학습은 CPU(수 초). 추론 측정: Intel Core i5-13420H(12스레드), GPU 없음 |
| Validation 추론 batch size | 1 (통화 하나씩, 통화당 20초 창 최대 4개를 하나씩 처리) |
| Validation 추론 시간 | **203건 표본** 731초(모델 로딩 포함, `m1_model.pt` 기준), 통화당 약 3.6초 (CPU). 전체 3,640건으로 환산하면 약 3.6시간. 전체를 직접 측정한 값은 아님 |

참고: T4 GPU에서는 같은 모델의 Validation 3,640건 추론이 33.2분(M1-XVEC-002-OFFVAL)이었다.
