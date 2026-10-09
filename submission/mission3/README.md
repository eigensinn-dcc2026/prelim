# Mission 3 제출 폴더 (팀 eigensinn)

통화 전사문에서 9개 증상(고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란)을 다중 라벨로 예측합니다.

## 구성

| 파일 | 설명 |
|---|---|
| `inference.py` | 공식 실행 명령 진입점 |
| `mission3_model.py` | 모델 코드(KcBERT 분류기, 체크포인트 로딩, 확률 계산) |
| `requirements.txt` | 추론용 패키지, 버전 고정 |
| `m3_kcbert005_seed42.pt` | 가중치. 용량(436MB) 때문에 git에 없고 패키징 때 넣는다 (SHA256 `77519f12…0f3b`, 전체값은 `checkpoints/mission3/*.pt.json`) |
| 학습 ipynb | 최종 학습 로그. 최종 설정 확정 후 추가 예정 |

가중치에는 tokenizer·config 텍스트가 함께 들어 있어 **인터넷 없이** 실행됩니다.

## 실행

```bash
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path m3_kcbert005_seed42.pt --output ./outputs/mission3.csv
```

- 입력은 라벨 JSON의 `utterances[].text`뿐입니다(`--audio_dir`은 받기만 하고 읽지 않음).
- 판정은 모든 증상에 `확률 ≥ 0.5 → 1`만 적용하고, 다른 후처리는 없습니다.
- 출력 컬럼은 `label file name,symptom`, 셀은 `['두통', '복통']` 형식(없으면 `[]`)입니다.

## 계산 효율 정보

| 항목 | 값 |
|---|---|
| 파라미터 수 | Total 108,925,449개 (Dense 모델이라 Active도 동일). KcBERT-base 108.9M + 분류층 6.9K |
| 컴퓨팅 환경 | 학습: Colab Tesla T4. 추론 측정: Intel Core i5-13420H(12스레드), RAM 15.6GB, GPU 없음 |
| Validation 추론 batch size | 32 |
| Validation 추론 시간 | 3,640건 전체 613초, 샘플당 평균 약 0.17초 (CPU) |

이 폴더의 `inference.py`로 공식 Validation 3,640건을 다시 돌린 결과(2026-10-10, 같은 PC, CPU):
Macro F1 0.6246(기존 기록과 일치), 모델 로딩 포함 883초(샘플당 약 0.24초). 위 613초와 차이가 나는 것은 측정 시점의 PC 부하 차이로 보이며, PPT에는 두 값을 함께 적는다.
