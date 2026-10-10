# Mission 3 제출 폴더 (팀 eigensinn)

통화 전사문에서 9개 증상(고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란)을 다중 라벨로 예측합니다.

## 구성

| 파일 | 설명 |
|---|---|
| `inference.py` | 공식 실행 명령 진입점 |
| `mission3_model.py` | 모델 코드(RoBERTa 분류기, 체크포인트 로딩, 확률 계산) |
| `requirements.txt` | 추론용 패키지, 버전 고정 |
| `m3_roberta003_seed42.pt` | **최종 후보 가중치**, 1,347,790,829 bytes. Git 제외. SHA256 `bbc0bd7a5f320df78bc20e59fc98827c6d9bb9367d23ac45367d7c2ff9a56e7e` |
| 학습 ipynb | 최종 학습 로그. 최종 설정 확정 후 추가 예정 |

가중치에는 tokenizer·config 텍스트가 함께 들어 있어 **인터넷 없이** 실행됩니다.

## 실행

```bash
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path m3_roberta003_seed42.pt --output ./outputs/mission3.csv
```

- 입력은 라벨 JSON의 `utterances[].text`뿐입니다(`--audio_dir`은 받기만 하고 읽지 않음).
- 판정은 모든 증상에 `확률 ≥ 0.5 → 1`만 적용하고, 다른 후처리는 없습니다.
- 출력 컬럼은 `label file name,symptom`, 셀은 `['두통', '복통']` 형식(없으면 `[]`)입니다.

## 모델 선택 — 2026-10-10

`M3-ROBERTA-003`(klue/roberta-large, 강화 가중 BCE + soft macro-F1, 앞 256토큰, seed 42, epoch 3)을 선택했다. 공식 Validation Macro F1 **0.6433**으로 KCBERT-005(0.6246) 대비 **+0.0187**, 채택 기준 +0.01을 넘는다. 3-seed 내부 평균 0.6427도 ROBERTA-002(0.6241)보다 +0.0186 높다. seed 42는 사전에 정한 대표 seed다.

3모델 앙상블(공식 0.6510)은 이 단일 모델 대비 +0.0077로 추가 모델 채택 기준 +0.01에 못 미쳐 사용하지 않는다. threshold 0.5와 빈 예측 `[]` 처리도 유지한다. 이번 선택은 검증된 학습 체크포인트의 교체이며, 전체 Training 최종 재학습을 수행했다는 뜻은 아니다. 학습 ipynb와 최종 제출 전 새 환경 재현은 별도 준비 항목이다.

근거: `reports/mission3/official_val/README.md`, `reports/mission3/ensemble/results/ensemble_multi_result.json`.

## 계산 효율 정보

| 항목 | 값 |
|---|---|
| 파라미터 수 | **336,665,609개**, RoBERTa-large + 9개 출력층(Dense 모델) |
| 컴퓨팅 환경 | 학습: Colab Tesla T4. 추론 측정: Intel Core i5-13420H(12스레드), RAM 15.6GB, GPU 없음. 이번 재검증은 PyTorch 4스레드, M1 작업과 동시 실행 |
| Validation 추론 batch size | 32 |
| Validation 추론 시간 | 이번 제출 폴더 재검증: 3,640건 전체 5,477.6초, 샘플당 약 1.505초(CPU, 다른 작업 동시 실행). 기존 단독 공식 평가 기록: 2,098초, 약 0.58초/건 |

이전 KcBERT의 613초/883초 수치는 현재 RoBERTa의 효율 수치로 사용하지 않는다. 이번 제출 폴더 실행으로 Macro F1 0.6433196, Micro F1 0.6580992, 빈 예측 123건을 재현했다. 실행 결과와 해시 검증은 `reports/mission3/official_val/roberta003_submission_check.json`에 기록한다.
