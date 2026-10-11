"""Two-stage stacking for Mission 3: a linear meta model over base-model logits (analysis only).

python m3_stacking.py <probs_dir> <tag> [<tag> ...]

Features: logit(p) of every base model for all 9 symptoms (9 x M). Meta model: one linear layer
-> 9 logits, trained with the same loss as the base models (BCE with Training-split pos_weight
scale 0.5 / cap 6.0, + 1.0 * (1 - soft macro-F1)), full batch, fixed hyperparameters.
Two variants are compared by 5-fold CV on the internal holdout only:
  diag  - symptom j sees only symptom j of each model (a learned per-model weight + bias per symptom)
  full  - symptom j sees all 9 x M features (can learn co-occurrence: 구토/어지러움 -> 오심)
The chosen variant is refit on the whole holdout and scored once on the official Validation.
Decision p >= 0.5, no post-processing. The holdout is Training data; the official Val is never fitted.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

EPOCHS, LR, WD, SEED = 400, 0.05, 1e-3, 42


def load(d, tags, split):
    probs, ref = [], None
    for t in tags:
        z = np.load(Path(d) / f"{t}_{split}.npz")
        if ref is None:
            ref = (z["names"], z["gold"].astype(int))
        assert (z["names"] == ref[0]).all()
        probs.append(z["probs"].astype(np.float64))
    p = np.stack(probs, 1)  # n x M x 9
    x = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    return x, ref[1], p


def pos_weight(y):
    pos = y.sum(0)
    return np.minimum(6.0, np.maximum(1.0, np.sqrt((len(y) - pos) / np.maximum(pos, 1)) * 0.5))


def fit(x, y, variant):
    torch.manual_seed(SEED)
    n, m, k = x.shape
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    if variant == "diag":
        w = torch.nn.Parameter(torch.full((m, k), 1.0 / m))
        b = torch.nn.Parameter(torch.zeros(k))
        fwd = lambda a: (a * w).sum(1) + b
    else:
        flat = xt.reshape(n, m * k)
        lin = torch.nn.Linear(m * k, k)
        with torch.no_grad():  # start from the plain average of the same symptom
            lin.weight.zero_()
            for j in range(k):
                for i in range(m):
                    lin.weight[j, i * k + j] = 1.0 / m
            lin.bias.zero_()
        w, b = lin.weight, lin.bias
        fwd = lambda a: lin(a.reshape(a.shape[0], m * k))
    opt = torch.optim.Adam([w, b], lr=LR, weight_decay=WD)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight(y), dtype=torch.float32))
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = fwd(xt)
        p = logits.sigmoid()
        soft_f1 = (2 * (p * yt).sum(0) / (p.sum(0) + yt.sum(0) + 1e-8)).mean()
        loss = bce(logits, yt) + 1.0 * (1 - soft_f1)
        loss.backward()
        opt.step()
    return lambda a: fwd(torch.tensor(a, dtype=torch.float32)).sigmoid().detach().numpy()


def macro(p, g):
    return float(f1_score(g, (p >= 0.5).astype(int), average="macro", zero_division=0))


def per(p, g):
    labels = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
    return dict(zip(labels, np.round(f1_score(g, (p >= .5).astype(int), average=None, zero_division=0), 4).tolist()))


def main():
    d, tags = sys.argv[1], sys.argv[2:]
    xh, gh, ph = load(d, tags, "holdout")
    xv, gv, pv = load(d, tags, "val")
    cv = {}
    for variant in ("diag", "full"):
        oof = np.zeros(gh.shape)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(xh):
            oof[te] = fit(xh[tr], gh[tr], variant)(xh[te])
        cv[variant] = round(macro(oof, gh), 4)
    mean_avg_h = round(macro(ph.mean(1), gh), 4)
    chosen = max(cv, key=lambda v: (cv[v], v == "diag"))  # tie -> simpler
    model = fit(xh, gh, chosen)
    val = model(xv)
    result = dict(models=tags, holdout_cv=cv, holdout_plain_mean=mean_avg_h, chosen=chosen,
                  official_val=dict(stacking=round(macro(val, gv), 4), plain_mean=round(macro(pv.mean(1), gv), 4),
                                    empty=int(((val >= .5).sum(1) == 0).sum()), per_symptom=per(val, gv),
                                    per_symptom_plain_mean=per(pv.mean(1), gv)))
    Path(d, f"stacking_{'+'.join(tags)}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "official_val"}, ensure_ascii=False))
    print("official Val:", {k: v for k, v in result["official_val"].items() if not k.startswith("per_")})


if __name__ == "__main__":
    main()
