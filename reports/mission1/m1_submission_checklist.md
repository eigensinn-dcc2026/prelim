# M1 제출 패키징 체크리스트 (초안)

- 작성: 2026-10-05 · DCC M1
- 후보: **LR** (`M1-EXP-001`) · 공식 Val Accuracy **95.11%** · 새 학습 금지 · MFCC/F0 개선 보류
- 기준: `competition_rule/` 공식 PDF · 단일 명령 `python inference.py --audio_dir … --label_dir … --ckpt_path … --output …`

---

## 1. 로컬 ckpt 확인

| 항목 | 상태 | 경로 / 메모 |
|---|---|---|
| 채택 ckpt (LR) | 있음 | `checkpoints/mission1/m1_baseline_lr.pkl` (3,702 bytes, 2026-09-04) |
| LR SHA256 | 기록 | `924B5620E1DE723D6B7821EB4849B4EA4B981C4BFD6621B469C5D0753F7934F8` |
| 비교용 GBM | 있음(비채택) | `checkpoints/mission1/m1_baseline_lgbm.pkl` |
| Git 커밋 | 금지 | `checkpoints/` · `data/` 대용량 제외 · Drive/외부만 제출 패키지에 포함 |

---

## 2. `inference.py` CLI · CSV 점검

| 점검 항목 | 규정 | 현재 상태 | 갭 |
|---|---|---|---|
| 엔트리포인트 | 제출용 `inference.py` 한 방 | `scripts/mission1_inference.py` 추가 (제출 시 루트 `inference.py`로 복사) | 패키징 시 이름만 맞추면 됨 |
| 인자 | `--audio_dir` `--label_dir` `--ckpt_path` `--output` 필수 | `scripts/mission1_inference.py` 구현 | 완료 |
| CSV 열 | `[audio file name]`, `[gender]` (공식 PDF; Zoom의 startAt/endAt 포함설은 비채택) | `scripts/mission1_inference.py` writer | 완료 |
| gender 값 | 라벨링 `gender` (학습·평가 로그상 `F`/`M`) | 추론은 WAV+JSON 구간→52-d→LR | 완료(스모크는 별도) |
| 허용 annotation (inference) | `startAt`,`endAt`,`Speaker`만 | 기존 Val eval은 feature cache 사용 | 제출 추론은 WAV+JSON 구간으로 52-d 재추출 필요 |
| Val backprop | 금지 | 기존 eval은 frozen ckpt만 | 유지 |

참고 재현(개발용, 제출 형식 아님):

```bash
python scripts/eval_mission1_validation.py \
  --features data/m1_features_validation.parquet \
  --lr-ckpt checkpoints/mission1/m1_baseline_lr.pkl \
  --log-output reports/mission1/m1_val_eval.json
```

제출 목표 명령:

```bash
python scripts/mission1_inference.py \\
  --audio_dir {wav} \
  --label_dir {json} \
  --ckpt_path checkpoints/mission1/m1_baseline_lr.pkl \
  --output ./outputs/mission1.csv
```

---

## 3. 제출 패키지 항목 (마감 게이트용)

- [x] `scripts/mission1_inference.py` (제출 시 `inference.py`로 복사) — WAV+JSON → 52-d → LR → CSV
- [ ] 학습 `.ipynb` + 로그 (기존 baseline 기록 연결)
- [ ] 가중치: `m1_baseline_lr.pkl` (+ scaler가 파이프라인에 포함됨)
- [ ] 모델 `.py` (`src/dcc_data/mission1_*` 등)
- [ ] `requirements.txt`
- [ ] 결과 PPT (사령탑 취합)
- [ ] CSV 스모크: 헤더 철자·공백·UTF-8 · 행 수=입력 수

---

## 4. 다음 액션 (승인 후)

1. **합의 후** 본 체크리스트만 브랜치 커밋 → PR (`git add .`·ckpt 금지)
2. 스모크: Val 소수 통화로 CSV 헤더·행 수 확인 (새 학습 없음)
3. MFCC/F0 개선은 계속 보류

@DCC 사령탑 합의 요청: 이 초안을 `reports/mission1/m1_submission_checklist.md`로 커밋해도 될까요?

