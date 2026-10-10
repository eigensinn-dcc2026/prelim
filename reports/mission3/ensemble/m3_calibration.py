"""Probability-quality metrics for Mission 3 models (analysis only).

python m3_calibration.py <probs_dir> <tag> [<tag> ...]
Each tag needs <tag>_holdout.npz and/or <tag>_val.npz (names, probs, gold).

Per symptom, then averaged over the 9 symptoms:
- ECE: 15 equal-width bins, |mean prob - positive rate| weighted by bin size
- Brier: mean (p - y)^2
- bias: mean predicted probability - actual positive rate (> 0 = probabilities pushed up)
- predicted positive rate at p >= 0.5 vs actual positive rate
Macro F1 at p >= 0.5 is printed next to them so the trade-off is visible.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
BINS = 15


def ece(p, y):
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, BINS - 1)
    total = 0.0
    for b in range(BINS):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


def metrics(p, y):
    per = {}
    for j, s in enumerate(LABELS):
        per[s] = dict(ece=round(ece(p[:, j], y[:, j]), 4), brier=round(float(((p[:, j] - y[:, j]) ** 2).mean()), 4),
                      bias=round(float(p[:, j].mean() - y[:, j].mean()), 4),
                      pred_pos_rate=round(float((p[:, j] >= .5).mean()), 4), true_pos_rate=round(float(y[:, j].mean()), 4))
    avg = {k: round(float(np.mean([v[k] for v in per.values()])), 4) for k in ("ece", "brier", "bias", "pred_pos_rate", "true_pos_rate")}
    avg["macro_f1"] = round(float(f1_score(y, (p >= .5).astype(int), average="macro", zero_division=0)), 4)
    return avg, per


def main():
    d, tags = sys.argv[1], sys.argv[2:]
    out = {}
    for t in tags:
        for split in ("holdout", "val"):
            f = Path(d) / f"{t}_{split}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            avg, per = metrics(z["probs"].astype(float), z["gold"].astype(int))
            out[f"{t}/{split}"] = dict(avg=avg, per_symptom=per)
            print(f"{t:6s} {split:7s} F1 {avg['macro_f1']:.4f}  ECE {avg['ece']:.4f}  Brier {avg['brier']:.4f}  "
                  f"bias {avg['bias']:+.4f}  pred+ {avg['pred_pos_rate']:.3f} vs true {avg['true_pos_rate']:.3f}")
    Path(d, "calibration_" + "+".join(tags) + ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
