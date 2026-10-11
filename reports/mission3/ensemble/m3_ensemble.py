"""KcBERT + TF-IDF probability-average ensemble (analysis only).

One global weight w: p = w * p_bert + (1 - w) * p_tfidf, decision p >= 0.5, no post-processing.
w is chosen on the internal 80:20 holdout only; the official Validation is scored once at the chosen w.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

D = Path(sys.argv[1])
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


def load(name):
    z = np.load(D / name)
    return z["names"], z["probs"].astype(np.float64), z["gold"].astype(int)


def aligned(a, b):
    na, pa, ga = load(a)
    nb, pb, gb = load(b)
    assert (na == nb).all() and (ga == gb).all()
    return pa, pb, ga


def macro(p, g):
    return f1_score(g, (p >= 0.5).astype(int), average="macro", zero_division=0)


def per(p, g):
    return dict(zip(LABELS, np.round(f1_score(g, (p >= 0.5).astype(int), average=None, zero_division=0), 4).tolist()))


def boot_diff(pa, pb, g, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ya, yb = (pa >= 0.5).astype(int), (pb >= 0.5).astype(int)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(g), len(g))
        diffs.append(f1_score(g[i], ya[i], average="macro", zero_division=0)
                     - f1_score(g[i], yb[i], average="macro", zero_division=0))
    return np.percentile(diffs, [2.5, 50, 97.5]).round(4).tolist(), float(np.mean(np.array(diffs) > 0))


kb, tb, g = aligned("k5_holdout.npz", "p0_holdout.npz")
print("holdout sanity: KcBERT", round(macro(kb, g), 6), "TF-IDF", round(macro(tb, g), 6))
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
curve = {float(w): round(macro(w * kb + (1 - w) * tb, g), 4) for w in grid}
print("holdout curve:", curve)
w_star = max(curve, key=lambda w: (curve[w], w))  # ties -> more KcBERT weight
ens_h = w_star * kb + (1 - w_star) * tb
ci, p_pos = boot_diff(ens_h, kb, g)
print(f"chosen w={w_star}: holdout ens {macro(ens_h, g):.4f} vs KcBERT {macro(kb, g):.4f}; "
      f"paired bootstrap diff 95% CI {ci}, P(diff>0)={p_pos:.3f}")

kv, tv, gv = aligned("k5_val.npz", "p0_val.npz")
ens_v = w_star * kv + (1 - w_star) * tv
result = dict(
    w=w_star, holdout_curve=curve,
    holdout=dict(kcbert=macro(kb, g), tfidf=macro(tb, g), ensemble=macro(ens_h, g),
                 diff_ci95=ci, p_diff_pos=p_pos, per_kcbert=per(kb, g), per_ensemble=per(ens_h, g),
                 empty_kcbert=int(((kb >= .5).sum(1) == 0).sum()), empty_ensemble=int(((ens_h >= .5).sum(1) == 0).sum())),
    official_val=dict(kcbert=macro(kv, gv), tfidf=macro(tv, gv), ensemble=macro(ens_v, gv),
                      per_kcbert=per(kv, gv), per_ensemble=per(ens_v, gv),
                      empty_kcbert=int(((kv >= .5).sum(1) == 0).sum()), empty_ensemble=int(((ens_v >= .5).sum(1) == 0).sum())),
)
cv, pv = boot_diff(ens_v, kv, gv)
result["official_val"].update(diff_ci95=cv, p_diff_pos=pv)
print("official Val:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in result["official_val"].items() if not k.startswith("per_")}, ensure_ascii=False))
(D / "ensemble_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
