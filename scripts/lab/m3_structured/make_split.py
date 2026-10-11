"""기존 80:20 multilabel stratified 홀드아웃(seed 42)을 train.csv / holdout42.npz 로 재현.
권장: 기존 런(ROBERTA-003 등)의 rows.json·split.json 을 그대로 받아 쓴다. 그래야 행 순서와 분할이 0.6427 과 같다.
  python make_split.py --rows_json <run>/rows.json --split_json <run>/split.json
    -> data/train.csv (rows.json 순서, text + 9개 증상) + data/holdout42.npz
  python make_split.py --data_csv data/train.csv   (split.json 이 없을 때만: 같은 방식으로 다시 나눔)
"""
import argparse, json, numpy as np, pandas as pd
from losses import SYMPTOMS

ap = argparse.ArgumentParser()
ap.add_argument("--rows_json", default=None, help="기존 런의 rows.json -> train.csv 생성")
ap.add_argument("--split_json", default=None, help="기존 런의 split.json (train/holdout 인덱스)")
ap.add_argument("--data_csv", default="data/train.csv"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="data/holdout42.npz")
a = ap.parse_args()

if a.rows_json:
    rows = json.load(open(a.rows_json, encoding="utf-8"))
    df = pd.DataFrame({"call_id": [r["call_id"] for r in rows], "text": [r["text"] for r in rows]})
    df[SYMPTOMS] = np.array([r["labels"] for r in rows], dtype=int)
    df.to_csv(a.data_csv, index=False, encoding="utf-8")
    print(f"saved {a.data_csv}: {len(df)} rows (rows.json 순서)")
Y = pd.read_csv(a.data_csv)[SYMPTOMS].to_numpy()

if a.split_json:
    sp = json.load(open(a.split_json, encoding="utf-8"))
    tr, va = np.array(sp["train"]), np.array(sp["holdout"])
    assert sorted(tr.tolist() + va.tolist()) == list(range(len(Y))), "split.json 과 train.csv 행 수/순서 불일치"
else:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    # ROBERTA-003 과 같은 호출. 단, train.csv 행 순서가 rows.json(정렬된 파일 경로) 순서와 같아야 같은 분할이 나옴
    tr, va = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed).split(np.arange(len(Y)), Y))
np.savez(a.out, train_idx=tr, val_idx=va)
print(f"saved {a.out}: train {len(tr)} / val {len(va)}  (기존 기준: 23,408 / 5,792)")
