"""Probability-average ensembles over several Mission 3 models (analysis only).

python m3_ensemble_multi.py <probs_dir> <tag> [<tag> ...]
Each tag needs <tag>_holdout.npz and <tag>_val.npz (names, probs, gold) for the same rows.

For every non-empty subset of the tags, global weights on a 0.05 simplex grid are compared on the
internal 80:20 holdout only; ties go to the simpler combination (fewer models, then the earlier tag).
The official Validation is scored once per subset at its chosen weights, plus each single model.
Decision p >= 0.5, no post-processing, one weight per model (never per symptom).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

STEP = 20  # 1 / 0.05


def load(d, tag, split):
    z = np.load(Path(d) / f"{tag}_{split}.npz")
    return z["names"], z["probs"].astype(np.float64), z["gold"].astype(int)


def macro(p, g):
    return float(f1_score(g, (p >= 0.5).astype(int), average="macro", zero_division=0))


def simplex(k):
    for c in itertools.product(range(STEP + 1), repeat=k):
        if sum(c) == STEP and all(c):
            yield np.array(c) / STEP


def boot_diff(pa, pb, g, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ya, yb = (pa >= .5).astype(int), (pb >= .5).astype(int)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(g), len(g))
        d.append(f1_score(g[i], ya[i], average="macro", zero_division=0) - f1_score(g[i], yb[i], average="macro", zero_division=0))
    return np.percentile(d, [2.5, 97.5]).round(4).tolist()


def main():
    d, tags = sys.argv[1], sys.argv[2:]
    data = {}
    for split in ("holdout", "val"):
        ref = None
        for t in tags:
            n, p, g = load(d, t, split)
            if ref is None:
                ref = (n, g)
            assert (n == ref[0]).all() and (g == ref[1]).all(), (t, split)
            data[(t, split)] = p
        data[("gold", split)] = ref[1]
    gh, gv = data[("gold", "holdout")], data[("gold", "val")]
    rows = []
    for k in range(1, len(tags) + 1):
        for subset in itertools.combinations(tags, k):
            best = None
            for w in simplex(k):
                ph = sum(wi * data[(t, "holdout")] for wi, t in zip(w, subset))
                s = macro(ph, gh)
                if best is None or s > best[0] + 1e-12:
                    best = (s, w)
            s, w = best
            pv = sum(wi * data[(t, "val")] for wi, t in zip(w, subset))
            rows.append(dict(models=list(subset), weights=[round(float(x), 2) for x in w], holdout=round(s, 4),
                             val=round(macro(pv, gv), 4), val_empty=int(((pv >= .5).sum(1) == 0).sum()), _pv=pv))
    # Selection is by holdout only; the official Val column is a single confirmation per row.
    rows.sort(key=lambda r: (-r["holdout"], len(r["models"])))
    best_single = max((r for r in rows if len(r["models"]) == 1), key=lambda r: r["holdout"])
    for r in rows:
        r["val_diff_vs_best_single_ci95"] = boot_diff(r["_pv"], best_single["_pv"], gv) if r is not best_single else None
    out = [{k: v for k, v in r.items() if k != "_pv"} for r in rows]
    Path(d, "ensemble_multi_result.json").write_text(json.dumps(dict(best_single=best_single["models"], rows=out),
                                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"best single on holdout: {best_single['models']}")
    for r in out:
        print(f"{'+'.join(r['models']):22s} w={r['weights']}  holdout {r['holdout']:.4f}  val {r['val']:.4f}  "
              f"empty {r['val_empty']}  diff_ci {r['val_diff_vs_best_single_ci95']}")


if __name__ == "__main__":
    main()
