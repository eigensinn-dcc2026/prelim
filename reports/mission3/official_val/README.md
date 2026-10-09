# Mission 3 공식 Validation 채점 (2026-10-10)

공식 Validation 3,640건(`Validation/2.라벨링데이터/VL_서울_구급`)을 **평가에만** 썼습니다. 학습·선택·조정에는 쓰지 않았습니다.
실행은 제출과 같은 경로입니다.

```bash
python inference.py --audio_dir <Validation/1.원천데이터> --label_dir <Validation/2.라벨링데이터> --ckpt_path <ckpt> --output outputs/mission3.csv
```

채점은 `scripts/mission3_score_csv.py`(9개 증상 Macro F1)로 했습니다. threshold는 `>= 0.5` 고정이고 후처리는 없습니다. 라벨 0개면 `[]`입니다.

| 모델 | 체크포인트 | 공식 Val Macro F1 | Micro F1 | 빈 예측 |
|---|---|---|---|---|
| P0 TF-IDF + OvR LR (현 제출본) | `mission3_tfidf_logreg.joblib` (SHA256 `6dc08608…5ff7b8`) | 0.5689 | 0.5802 | 183 |
| **KCBERT-005 seed 42** | `m3_kcbert005_seed42.pt` (SHA256 `77519f12…6c0d3b`) | **0.6246** | **0.6417** | 299 |

- 차이는 **+0.0557**로 채택 기준(+0.01)을 충족합니다.
- KCBERT-005 seed 42는 lab 스크립트가 미리 정한 대표 seed입니다. 홀드아웃 점수로 고르지 않았습니다.
- 증상별 F1(KCBERT-005): 열상 0.864, 복통 0.802, 고열 0.671, 호흡곤란 0.651, 어지러움 0.640, 구토 0.563, 전신쇠약 0.534, 두통 0.518, 오심 0.378.
  - 오류 분석(`../error_analysis/README.md`)에서 예상한 대로 오심·두통·전신쇠약이 가장 낮습니다.

## 추론 경로 일치 확인

내보낸 `.pt`로 lab과 같은 80:20 홀드아웃 5,792건을 `inference.py`로 다시 채점했습니다(`kcbert005_seed42_holdout_rescore.json`).
Macro F1 0.6195615502539225로 lab 기록(`dcc-lab results/M3-KCBERT-005`, seed 42)과 소수점 16자리까지 같고, 증상별 F1도 모두 같습니다.
오프라인 환경(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)에서 실행했습니다.

## 실행 환경과 시간

- CPU만 사용했습니다(torch 2.12.1+cpu, transformers 4.57.6).
- 공식 Val 3,640건은 613초, 홀드아웃 5,792건은 911초 걸렸습니다.
- 체크포인트는 436,017,645 bytes(fp32)입니다. 원본 lab ckpt의 SHA256은 `m3_kcbert005_seed42.pt.json`에 있습니다.

## 파일

- `kcbert005_seed42_val_score.json`, `p0_tfidf_val_score.json`: 공식 Val 점수
- `kcbert005_seed42_holdout_rescore.json`: 홀드아웃 재채점
- `m3_kcbert005_seed42.pt.json`: 내보내기 기록(원본, SHA256, 크기)
