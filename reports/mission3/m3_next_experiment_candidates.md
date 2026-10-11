# Mission 3 — 다음 실험 후보 초안 (게이트 대기용)

P0(TF-IDF) 실행·채택 전 **후보만** 기록. 장시간 학습·리서치 요청 없음.
공통 고정: 80:20 multilabel strat · seed=42 · threshold=0.5 · `DCC_DATA_ROOT` 동일 · 전사만.

| ID | 가설 | 스펙 한 줄 | 예상 이득 | 위험 |
|---|---|---|---|---|
| **P1** KcBERT 앞256 | 문맥 encoder가 TF-IDF보다 증상 표현을 잘 잡음 | `beomi/kcbert-base` · max_len=256 · 비가중 BCE · T4/GPU | Macro F1 상승(특히 희소 증상 recall) | 앞 256 truncate로 후반 증상 누락 · T4 시간 |
| **P2a/b** token vs turn chunk | 질문–답 경계를 지키면 오심/구토 공동오류↓ | 256·overlap64·max-pool · **학습조건 완전 동일** 1회 대조 | 긴 통화 Macro F1·공출현 슬라이스 개선 | 조건 불일치 시 우열 무효 · 연산 2배 |
| **P3** 완화 가중 BCE | scale0.25/max3.0이 오심 등 희소 recall↑ | P1/P2 **승자**와 동일 분할·청크만 변경 | 오심 F1 회복 | precision·subset Acc 하락(과거 강한 가중 교훈) |

**보류(이번 후보 아님)**
- 증상별 threshold 재튜닝 (재현 배열 없음·Val 사후조정 위험)
- WEIGHT-001 강한 가중 / 새 아키텍처·리서치

**실행 순서**: P0 결과 검토 → P1 → P2 동일조건 대조 → (이득 시) P3. 각각 @DCC 사령탑 게이트 후 @DCC 실행.
