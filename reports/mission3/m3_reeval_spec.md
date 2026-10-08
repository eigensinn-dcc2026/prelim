# Mission 3 재평가 스펙 (80:20 · threshold 0.5)

- **상태**: 게이트용 초안 (승인 전 새 학습·Colab 큐 투입 금지)
- **기준일**: 2026-10-05
- **비교 기준** (`docs/decisions.md` 2026-09-30): JSON 샘플 단위 **80:20 multilabel stratified split**, **seed=42**, 전 증상 **threshold=0.5**
- **데이터 루트**: `DCC_DATA_ROOT` → `data/raw_merged` (개인 PC 절대경로·`raw_original` 하드코딩 금지)
- **입력 제약**: 전사 텍스트만. 그 외 annotation 사용 금지
- **지표**: Macro F1 + 증상별 P/R/F1. 슬라이스: 구토·오심 공동오류, 2개 이상 라벨 통화
- **Colab**: `daulkim2001@gmail.com` Pro, **이번 주 T4 가정**. ckpt는 Drive/로컬만, Git 커밋 금지
- **제출 정렬**: 최종 추론은 `python inference.py --audio_dir {wav} --label_dir {json} --ckpt_path {ckpt} --output ./outputs/mission3.csv` · CSV 열 `[label file name]`, `[symptom]`

## 왜 재평가가 필요한가

과거 점수는 분할·Train 비율·임계값이 섞여 있어 **직접 우열 비교 불가**.

| 실험 ID | 과거 점수 | 왜 미확인인가 |
|---|---|---|
| M3-KCBERT-001 | 내부 Val Macro F1 0.5861 | Training 90:10 내부검증 |
| M3-CHUNK-001 | 공식 Val 0.5496 | Train 20% 등, 최신 80:20과 불일치 |
| M3-TURN-001 | 공식 Val 0.5365 | CHUNK와 학습 조건 동일성 미확인 |
| M3-EXP-001/002 | TF-IDF+LR | recordId 또는 구 분할 · 결과 미기재/과거값 |
| M3-WEIGHT-001/002 | 화면 전사·ckpt만 | 원본 메트릭/설정 누락 |
| M3-THRESH-001/002 | th 배열 없음 | 재현 불가 · th=0.5 고정만 유효 |

## 재실행 순서 (승인 후 @DCC 실행(colab))

### P0 — TF-IDF + OvR-LR (빠른 sanity, GPU 불필요)

- **스크립트**: `scripts/mission3_baseline 최종 수정본.py` (또는 경로 인자화한 동일 CLI)
- **설정**: `--test_size 0.2 --seed 42 --threshold 0.5`
- **데이터**: Training JSON만 (`DCC_DATA_ROOT` 하위 서울 Training 라벨)
- **산출**: `reports/mission3/m3_reeval_tfidf_lr.json`
- **목적**: 파이프라인·메트릭·분할이 맞는지 확인. 최종 후보 아님

### P1 — KcBERT 앞 256토큰 baseline (T4)

- **모델**: `beomi/kcbert-base`, 9 sigmoid, 비가중 BCE
- **분할**: 80:20 multilabel strat · seed=42 (과거 90:10 점수 폐기 비교)
- **학습**: epochs·batch는 T4 메모리에 맞게 스펙에 기록 (실효 batch·lr·max_length=256 고정)
- **평가**: threshold=0.5 · Macro F1 + 증상별 표
- **산출**: Drive ckpt + SHA-256 · `reports/mission3/m3_reeval_kcbert_front256.json`
- **과거 0.5861과 숫자 비교 금지** (분할이 다름)

### P2 — 토큰 청크 vs 발화 턴 청크 (동일 조건 대조, T4)

양쪽을 **완전 동일**하게 맞춤 후 1회씩만 비교.

| 항목 | 고정값 |
|---|---|
| split | 80:20 multilabel strat · seed=42 |
| max_length | 256 |
| chunk_overlap | 64 |
| pooling | chunk logit **max pooling** → 통화 단위 |
| threshold | 0.5 |
| Train 비율·epoch·batch·lr | 양쪽 동일 (실행 시 확정값을 JSON에 기록) |

- **A** `token_chunk`: 기존 전체 토큰 청크
- **B** `dialogue_turn_overlap_v1`: 발화 턴 청크
- **산출**: `m3_reeval_chunk_token.json` / `m3_reeval_chunk_turn.json`
- **판정**: Macro F1 + 구토·오심 슬라이스. 우열은 이 대조 후에만 확정

### P3 — 완화 가중 BCE (P1/P2 승자 확정 후)

- **대상**: `M3-WEIGHT-002` 설정만 (`pos_weight_scale=0.25`, `max=3.0`)
- **조건**: P1 또는 P2 승자와 **동일 분할·동일 청크 전략**
- **제외**: WEIGHT-001(강한 가중)·THRESH 증상별 배열 — 재현 정보 부족, 참고용만
- **산출**: `m3_reeval_weight025.json` + Drive ckpt SHA

## 경로·재현 규칙 (하드코딩 제거)

1. 코드/노트북의 `C:\Users\…`, `조지은\…` 등 절대경로 삭제
2. 필수 인자/환경변수만 사용: `DCC_DATA_ROOT`, `--data_root`, `--label_dir`, `--ckpt_path`, `--output`
3. `checkpoints/`, `data/` 는 Git에 넣지 않음. Drive 경로 + SHA-256을 리포트 JSON에 기록
4. 단일 추론 명령 형식을 최종 후보에 맞출 것 (제출 게이트 전)

## Colab 입력 체크리스트 (@DCC 실행)

- [ ] `DCC_DATA_ROOT` → `data/raw_merged` 마운트 확인
- [ ] deps: `requirements.txt` + `iterative-stratification` + `transformers` / `beomi/kcbert-base` 로컬 캐시 (상용 API 호출 금지)
- [ ] T4: batch/accumulation을 OOM 없이 기록
- [ ] P0 → P1 → P2 → (선택) P3 순서. **한 번에 한 실험**
- [ ] 산출 JSON을 `reports/mission3/`에 두고 스펙 md와 함께 브랜치 커밋 대상은 **리포트만** (ckpt 제외)
- [ ] Val backprop 금지. Val은 평가·EarlyStopping·모델 선택만

## 이번 스펙에서 하지 않는 것

- 새 모델 아키텍처 탐색 / 리서치 요청
- 증상별 threshold 재튜닝
- `git add .` · `data/` · `checkpoints/` 커밋
- 사람 팀 수동 Colab 조작에 의존 (봇·실행 큐로 밀어감)

## 승인 후 다음 액션

1. 이 파일 push + PR (합의 후)
2. @DCC 실행(colab)이 P0부터 T4 큐 투입
3. 각 단계 종료 시 Macro F1·증상별 표·ckpt SHA를 방/사령탑에 보고
4. P2 우열 확정 전 P3 착수 금지
