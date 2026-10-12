# Mission 3 제출 폴더 (팀 eigensinn)

통화 전사문에서 9개 증상(고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란)을 다중 라벨로 예측합니다.

## 모델

최종 모델은 **RoBERTa-large 3 seed와 TF-IDF의 고정 가중치 앙상블**입니다(M3-FINAL-002).

- `klue/roberta-large`(revision `28d9112`)를 Training 29,200통화 전체로 seed 42·43·44 세 번 학습했습니다(앞 256토큰, BCE + soft-F1, 3 epoch).
- TF-IDF(5만 단어) + 로지스틱 회귀 하나를 Training 전체로 학습했습니다.
- 확률은 `p = 0.55 × (RoBERTa 3 seed 평균) + 0.45 × TF-IDF`로 합치고, 모든 증상에 `p ≥ 0.5 → 1`만 적용합니다. 다른 후처리는 없습니다.
- 가중치 0.55 : 0.45는 Training 내부 80:20 홀드아웃에서 미리 정했고, 공식 Validation은 선택에 쓰지 않고 채점에만 썼습니다.

## 구성

| 파일 | 설명 |
|---|---|
| `inference.py` | 공식 실행 명령 진입점 |
| `mission3_model.py` | 모델 코드(분류기, 앙상블 체크포인트 로딩, 확률 계산) |
| `requirements.txt` | 추론용 패키지, 버전 고정 |
| `m3_ensemble_full.pt` | 가중치. 앙상블 전체(RoBERTa-large 3개 + TF-IDF)를 한 파일에 담았다. 약 4.05GB라 git에 없고 패키징 때 넣는다. SHA256 `73c10afcee23f2cf1aba3ece5f758056f9ec0099fc3022aac469b5d5f10818d8` |
| `m3_final_train.executed.ipynb` | 학습 로그가 포함된 실행된 노트북(Colab A100, M3-FINAL-002). 학습 29,200통화, seed별·앙상블 공식 Val 점수 출력이 들어 있다 |

학습 노트북의 원본과 실행 방법은 `scripts/m3_final/`(최종 학습 키트)에 있습니다.

가중치에는 tokenizer·config 텍스트가 함께 들어 있어 **인터넷 없이** 실행됩니다. `--ckpt_path`는 앙상블 `.pt`(`dcc-m3-ens-v1`)와 단일 모델 `.pt`(`dcc-m3-bert-v1`)를 모두 받습니다.

## 실행

```bash
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path m3_ensemble_full.pt --output ./outputs/mission3.csv
```

- 입력은 라벨 JSON의 `utterances[].text`뿐입니다(`--audio_dir`은 받기만 하고 읽지 않음).
- 출력 컬럼은 `label file name,symptom`, 셀은 `['두통', '복통']` 형식(없으면 `[]`)입니다.
- 앙상블 멤버는 한 번에 하나씩 메모리에 올려 계산합니다. 다만 `.pt` 파일 전체(약 4GB)를 먼저 읽으므로 여유 메모리가 필요합니다.

## 결과 (공식 Validation 3,640건, 채점 전용)

| 모델 | Macro F1 | Micro F1 | 빈 예측 |
|---|---|---|---|
| RoBERTa seed 42 | 0.6479 | 0.6637 | 111 |
| RoBERTa seed 43 | 0.6441 | 0.6594 | 90 |
| RoBERTa seed 44 | 0.6445 | 0.6607 | 120 |
| **앙상블 (제출)** | **0.6534** | **0.6690** | 149 |

값은 `m3_final_train.executed.ipynb`의 실행 출력입니다. Training 100%로 학습해 holdout 점수를 낼 수 없으므로 공식 Validation은 채점에만 썼습니다.

코드 확인: 80% 학습 모델로 만든 앙상블 `.pt`(같은 형식)에서 앙상블 확률이 `0.55 × 개별 RoBERTa 평균 + 0.45 × TF-IDF`와 최대 오차 5e-8로 일치하고 판정이 같음을 Validation 12건으로 확인했습니다.

## 계산 효율 정보

| 항목 | 값 |
|---|---|
| 파라미터 수 | Total 1,010,496,836개 (Dense 모델이라 Active도 동일, 세 모델과 TF-IDF가 모두 실행됨). RoBERTa-large 336,665,609개 × 3 + TF-IDF 500,009개 |
| 컴퓨팅 환경 | 학습: Colab NVIDIA A100-SXM4-40GB 1개(노트북 출력 기준 seed 42·43·44가 각각 48.5·29.6·29.6분). 추론 측정: GPU는 같은 A100, CPU는 Intel Core i5-13420H(12스레드), RAM 15.6GB, GPU 없음 |
| Validation 추론 batch size | 32 |
| Validation 추론 시간 | GPU(A100): 3,640건 전체 109초(`m3_final_train.executed.ipynb`). CPU: 공식 Validation 203건(3,640건 중 18건마다 1건)을 이 폴더의 `inference.py`로 실행해 **462초**(모델 로딩 포함), 건당 약 2.3초. 3,640건 전체로 환산하면 약 2.3시간이다(로딩 시간까지 건수에 비례한다고 본 상한이며, 전체를 직접 돌린 값은 아님) |
| 최대 메모리 (CPU 추론) | 6.51GB (위 203건 실행, 앙상블 `.pt` 약 4GB를 먼저 읽음) |

이전 후보였던 KcBERT 단일 모델(파라미터 108.9M, CPU 샘플당 약 0.17초)은 최종 제출 모델이 아니므로 위 표에 쓰지 않습니다. 최종 앙상블은 RoBERTa-large 3개를 모두 실행해 파라미터가 약 9.3배이고, CPU 추론은 건당 약 13배 느립니다(0.17초 → 2.3초).
