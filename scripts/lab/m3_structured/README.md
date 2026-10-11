# DCC2026 M3 — 구조화 출력층 · LLM 교사 증류 골격

```
losses.py             공용 손실(pos_weight, soft macro-F1[beta/EMA], 가중 BCE)·지표
structured_head.py    제안 2: 라벨 집합 softmax 헤드 (B1: 공집합 포함 / B2: --exclude_empty) + 손실
train_structured.py   RoBERTa 학습 (holdout / 5-fold OOF / 증류)     <- 로컬 4060·A100
train_llm_teacher.py  제안 1: 8B LoRA 교사, 5-fold OOF, 게이트       <- A100 전용
measure_allzero.py    B2 우선순위 판정 진단 (5분)
make_split.py         홀드아웃 인덱스 npz 생성
test_head.py          헤드 수식 검증 (CPU)
data/                 train.csv(text + 9개 증상 0/1 열), holdout42.npz
runs/                 출력
```

## 준비
```
python -m pip install -r requirements.txt   # bitsandbytes 는 선택(4060 --adam8bit / --load_in_4bit)
python test_head.py                     # "all tests passed" 확인
python make_split.py --rows_json <ROBERTA-003 런>/rows.json --split_json <ROBERTA-003 런>/split.json
```
`data/train.csv`(rows.json 순서) 열: `call_id`, `text`, 고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란 (0/1).
split.json 을 그대로 쓰므로 홀드아웃은 기존 5,792건과 같습니다.

## 실행 순서
```bash
# 0) 진단
python measure_allzero.py --probs <holdout_probs.npy> --labels <holdout_labels.npy>

# 1) 대조군 -> B1. 대조군과 B1 은 W(pairwise) 하나만 다름. seed 42/43/44 를 모두 돌려 평균으로 비교
#    batch 8 x accum 2 = ROBERTA-003 과 같은 soft-F1 마이크로배치(8). warmup 0.1, 마지막 epoch(--select last)가 기본값
#    B2(--exclude_empty)는 진단 결과(재정규화 Δ ≤ 0)로 보류
python train_structured.py --data_csv data/train.csv --mode holdout --split_npz data/holdout42.npz ^
  --fp16 --grad_ckpt --batch 8 --accum 2 --seed 42 ^
  --no_pairwise --lambda_set 0 --lambda_bce 1 --head_lr 1e-5 --out_dir runs/control_s42
python train_structured.py --data_csv data/train.csv --mode holdout --split_npz data/holdout42.npz ^
  --fp16 --grad_ckpt --batch 8 --accum 2 --seed 42 ^
  --lambda_set 0 --lambda_bce 1 --head_lr 1e-5 --w_lr 1e-2 --out_dir runs/B1_s42

# 2) A100 세션 3개 — 교사 (out_dir 은 Drive)
python train_llm_teacher.py --data_csv data/train.csv --folds 0   --gate 0.635 --out_dir <drive>/teacher
python train_llm_teacher.py --data_csv data/train.csv --folds 1,2 --gate 0     --out_dir <drive>/teacher
python train_llm_teacher.py --data_csv data/train.csv --folds 3,4 --gate 0     --out_dir <drive>/teacher
python train_llm_teacher.py --data_csv data/train.csv --merge --out_dir <drive>/teacher

# 3) A100 — RoBERTa 5-fold OOF (현 레시피)
python train_structured.py --data_csv data/train.csv --mode kfold --select last --bf16 --batch 8 --accum 2 \
  --no_pairwise --lambda_set 0 --lambda_bce 1 --head_lr 1e-5 --out_dir runs/rob_5fold

# 4) 학생 증류
python train_structured.py --data_csv data/train.csv --mode kfold --select last --bf16 --batch 8 --accum 2 \
  --soft_target_npy <drive>/teacher/oof_probs.npy --soft_mix 0.5 --out_dir runs/student
```
(Windows cmd 는 `^`, Colab/bash 는 `\` 줄바꿈)

## 주의
- 4060 메모리 부족 시 `--adam8bit` 추가(대조군·B1 에 똑같이). `--batch` 를 8 아래로 내리면 soft-F1 마이크로배치가 ROBERTA-003 과 달라지므로 대조군부터 다시 잽니다.
- 대조군·B1 모두 3 seed 평균으로만 판정합니다. seed 간 편차(약 ±0.004)가 기대 이득보다 큽니다.
- 교사 게이트: fold0 1 epoch(약 1.2~1.5h) 후 F1 < 0.635면 자동 중단(중단 기준일 뿐). 진행 기준은 같은 fold 의 RoBERTa(`runs/rob_5fold/fold0_log.json`) 대비 +0.01 이상.
- 교사 풀링은 BOS 토큰을 평균에서 뺍니다(어텐션에는 포함).
- 학생 OOF 는 교사 OOF 를 통해 검증 fold 정보가 새므로 낙관적입니다. 학생 판정은 공식 Val 로.
- 교사 OOF는 Training만. `--predict_csv`(공식 Val) 확률을 학생 타깃에 섞으면 규정 위반.
- `measure_allzero.py`의 재정규화는 진단 전용. inference.py에 넣으면 후처리(금지).
- B2(`--exclude_empty`)·OOF F1 유도 pos_weight는 멘토 확인 후 적용.

## dcc-lab 큐 실행 (M3-ROB5F-001, M3-TEACHER-001)
- 러너가 재시도하면 끝난 fold(`fold{k}_probs.npy`, `oof_fold{k}.npy`)는 건너뜁니다.
- 두 스크립트 모두 `$LAB_OUT/metrics.json`을 남깁니다. 교사 게이트 탈락도 정상 종료로 기록(`gate_fail: true`).
- `--revision`으로 HF 커밋을 고정합니다 (klue/roberta-large `28d9112`, kanana-1.5-8b-base `5a5aad5`).
