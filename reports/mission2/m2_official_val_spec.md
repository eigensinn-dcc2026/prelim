# Mission 2 공식 Val 스펙 (`M2-L1-9k`)

상태: **사령탑 승인됨** (2026-10-05) · 스펙 md 커밋 후 합의→PR · **새 학습은 9k ckpt 부재 시 별도 승인 후에만**

기준 문서: `reports/mission2/m2_error_analysis.md`, `reports/eda/eda_report.md` (§7–8), `competition_rule/`

---

## 1. 후보 정의

| 항목 | 값 |
|---|---|
| ID | `M2-L1-9k` |
| 모델 | 음향 통계 **53차원** + LightGBM 기본 파라미터 |
| 학습 스크립트 | `scripts/train_mission2_baseline.py` |
| Train 범위 | 서울 Training **9,000통화** 샘플 · `seed=42` |
| 내부 참고 점수 | GroupShuffleSplit(call_id, test_size=0.2) **84.90%** — **공식 Val 아님** |
| 제외 | 전체 29,200통화 내부 84.59%(9k 대비 열위) · Spectral centroid ablation(재학습 필요 → Val 이후 게이트) |

입력 제약(대회): 한 발화 조각 음성 + JSON의 `startAt`/`endAt`만. text·기타 annotation 금지.  
지표: **Accuracy**.  
제출 CSV 열: `audio_file_name`, `startAt`, `endAt`, `speaker` (`speaker`: 대원=0, 신고자=1).

데이터 루트(팀 고정): `DCC_DATA_ROOT` → `data/raw_merged` (기본 `raw_original` 아님).

---

## 2. 체크포인트 현황 (2026-10-05, @DCC 실행 확인)

- `checkpoints/mission2/` **없음**
- 문서상 `data/local_cache/m2_baseline_lgbm_local_full.pkl` · feature parquet **없음**
- 리포 `.pkl`은 Mission1만 → **현재 공식 Val 추론 불가**

→ **9k 동결 재학습 1회**는 사령탑 **별도 승인** 후에만 실행. 이 스펙만으로는 학습을 시작하지 않는다.

---

## 3. 공식 Val 절차 (승인 후 · Val backprop 금지)

### 3.1 (조건부) 9k 동결 재학습 — 별도 승인 필요

목표: 내부 실험과 동일 후보를 재현 가능한 ckpt로 고정.

1. `DCC_DATA_ROOT`를 `data/raw_merged`로 고정
2. 서울 Training에서 **9,000통화** 샘플(seed=42) feature 추출  
   - `scripts/build_mission2_dataset.py` (인자·캐시 경로는 실행 로그에 기록)
3. 학습(Train만, Val로 가중치 갱신 금지):
   ```bash
   python scripts/train_mission2_baseline.py \
     --features <9k_features.parquet> \
     --seed 42 \
     --test-size 0.2 \
     --model-output checkpoints/mission2/m2_baseline_lgbm_9k.pkl \
     --log-output reports/mission2/m2_exp_log_9k_rebuild.json
   ```
4. ckpt는 Drive 등 외부 저장 + SHA256을 스펙/로그에만 기록 (**Git 커밋 금지**)

내부 GroupSplit 점수는 참고용. **채택 판정은 아래 공식 Val Accuracy만.**

### 3.2 공식 Validation 추론·채점

```bash
python scripts/mission2_inference.py \
  --audio_dir <Val_wav_dir> \
  --label_dir <Val_json_dir> \
  --ckpt_path checkpoints/mission2/m2_baseline_lgbm_9k.pkl \
  --output outputs/mission2_val.csv
```

- Val JSON의 `speaker`와 CSV를 대조해 **공식 Val Accuracy** 산출
- Val 구간으로 학습·EarlyStopping 외 **backprop 금지**(대회 규정)
- 산출 메타: `reports/mission2/m2_official_val_summary.json`  
  (Accuracy, n_utt, fallback 수, ckpt SHA, data-root, 시각)

### 3.3 오류 슬라이스 (채점 후 · 학습 없음)

EDA·오류분석 함의 고정 슬라이스:

1. **통화당 겹침시간 비율** 상위 (p90 ≥ ~10%; 겹침 *유무*는 통화 ~98%라 슬라이스 불가)
2. **짧은 발화**: `<0.5s` / **`0.5–1s`(취약 구간)** / `<1s`
3. (참고) 대원 vs 신고자 recall 격차

슬라이스 표는 `reports/mission2/m2_official_val_slices.json` (+ 필요 시 짧은 md).

범위 밖(다음 게이트): Spectral centroid ablation, HP 튜닝, pitch 등 feature 확장.

---

## 4. Colab / @DCC 실행 입력 목록

- env: `DCC_DATA_ROOT=…/data/raw_merged`
- deps: `requirements.txt` (기존 M2 LightGBM·librosa 스택)
- 스크립트: `build_mission2_dataset.py` → (승인 시) `train_mission2_baseline.py` → `mission2_inference.py`
- 산출: 위 summary/slices JSON · `outputs/mission2_val.csv`  
- **금지**: `data/`·`checkpoints/`·대용량 parquet/pkl Git 커밋 · `git add .` · `main` 직접 push

단일 제출 CLI 형식(마감 게이트):  
`python inference.py --audio_dir {wav} --label_dir {json} --ckpt_path {ckpt} --output ./outputs/mission2.csv`  
(현재 개발 엔트리는 `scripts/mission2_inference.py` — 패키징은 마감 직전)

---

## 5. 게이트 / 다음 액션

| 단계 | 담당 | 상태 |
|---|---|---|
| 스펙 md | @DCC M2 | 본 문서 |
| 9k ckpt 확인 | @DCC 실행(colab) | **없음** 확인됨 |
| 9k 동결 재학습 1회 | @DCC 실행(colab) | **별도 승인 대기** |
| 공식 Val 추론·슬라이스 | @DCC 실행 → @DCC M2 요약 | 재학습 승인·ckpt 확보 후 |
| Spectral centroid ablation 등 | — | Val 이후 게이트 |

승인 요청(실행): 사령탑이 **9k 동결 재학습 1회**를 승인하면 @DCC 실행(colab)이 위 3.1→3.2→3.3 순으로 진행.