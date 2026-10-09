# Mission 3 앙상블: KcBERT + TF-IDF 확률 평균 (2026-10-10)

`p = w · p_KcBERT + (1 − w) · p_TF-IDF`, 판정 `p >= 0.5`, 후처리 없음.

- KcBERT: KCBERT-005 seed 42 (`m3_kcbert005_seed42.pt`)
- TF-IDF: P0 (`reports/mission3/m3_reeval_tfidf_lr/mission3_tfidf_logreg.joblib`, SHA256 `6dc08608…5ff7b8`)
- 두 모델 모두 같은 80:20 분할의 Training 23,408건으로 학습했습니다. 홀드아웃 5,792건의 파일 목록이 완전히 같음을 확인했습니다.

## 가중치 선택 규칙
- 가중치 `w`는 **내부 홀드아웃에서만** 0~1 범위를 0.05 간격으로 비교해 골랐습니다. 동점이면 KcBERT 비중이 큰 쪽을 택합니다.
- 공식 Validation은 고른 `w` 하나로 **한 번만** 채점했습니다.
- 가중치는 전체에 하나입니다. 증상별 가중치는 증상별 기준선 조정과 비슷해지므로 쓰지 않았습니다.
- 규정상 앙상블 가중치를 Validation으로 조정하는 것은 허용됩니다(문제명세 Validation 활용 기준).

## 결과

| | 홀드아웃 | 공식 Val |
|---|---|---|
| TF-IDF | 0.5676 | 0.5689 |
| KcBERT | 0.6196 | 0.6246 |
| **앙상블 (w = 0.55)** | **0.6279** | **0.6348** |
| KcBERT 대비 차이 (paired bootstrap 1,000회, 95% CI) | +0.0082 [+0.0037, +0.0126] | +0.0102 [+0.0052, +0.0158] |

- 홀드아웃 곡선은 w = 0.45~0.70에서 0.625~0.628로 평평합니다. 고른 값이 특정 점에 민감하지 않다는 뜻입니다.
- 공식 Val에서 9개 증상 F1이 모두 올랐습니다(+0.003~+0.015).
- 빈 예측 행은 KcBERT 299, 앙상블 306입니다.
- 앙상블이 효과가 있는 이유는 `../error_analysis/README.md` 4.2절에 있습니다.

## 다음 단계
제출 추론 경로에 넣으려면 여러 모델과 가중치를 담는 체크포인트 형식이 필요합니다(아직 구현 전). 최종 학습 때 TF-IDF도 Training 전체로 다시 학습하고, `w = 0.55`는 그대로 씁니다.

## 파일
- `m3_dump_probs.py`: `dcc_inference.mission3`와 같은 경로로 확률을 뽑아 `probs/*.npz`(names, probs, gold)를 만듭니다.
- `m3_ensemble.py`: 가중치 곡선, 선택, 부트스트랩을 계산해 `ensemble_result.json`을 씁니다.
- `probs/`: 홀드아웃과 공식 Val의 확률입니다. 파일명, 확률, 정답만 들어 있고 전사 원문은 없습니다.

```bash
python reports/mission3/ensemble/m3_dump_probs.py <repo>/src <repo> <label_dir> <ckpt> reports/mission3/ensemble/probs/<name>.npz
python reports/mission3/ensemble/m3_ensemble.py reports/mission3/ensemble/probs
```
