"""rob(OOF) + tfidf_lr + e_lgbm 교차 적합 평가. 가중치는 4 fold에서 고르고 남은 fold에서 채점. threshold 0.5."""
import sys, json, itertools, numpy as np
rob_dir, e_dir = sys.argv[1], sys.argv[2]
rob = np.load(f"{rob_dir}/oof_probs.npy"); lr = np.load(f"{e_dir}/tfidf_lr_oof.npy"); gb = np.load(f"{e_dir}/lgbm_oof.npy")
idx = [np.load(f"{e_dir}/fold{k}_idx.npy") for k in range(5)]
for k in range(5):
    r = f"{rob_dir}/fold{k}_idx.npy"
    try: assert (np.load(r) == idx[k]).all(), f"fold{k} 분할 불일치"
    except FileNotFoundError: print("fold idx 대조 불가", r)
Y = np.load(f"{e_dir}/labels.npy")
def macro(P, Y):
    p = P >= .5; y = Y > .5; tp = (p & y).sum(0); d = p.sum(0) + y.sum(0)
    return float(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0).mean())
g = np.round(np.arange(0, 1.001, .05), 2)
W2 = [(a, round(1 - a, 2), 0.) for a in g]; W3 = [w for w in itertools.product(g, g, g) if abs(sum(w) - 1) < 1e-9]
def run(W):
    out = np.zeros_like(Y); picks = []
    for j in range(5):
        fit = np.concatenate([idx[i] for i in range(5) if i != j])
        w = max(W, key=lambda w: macro(w[0]*rob[fit] + w[1]*lr[fit] + w[2]*gb[fit], Y[fit]))
        out[idx[j]] = w[0]*rob[idx[j]] + w[1]*lr[idx[j]] + w[2]*gb[idx[j]]; picks.append(w)
    return out, picks
a, pa = run(W2); b, pb = run(W3)
print("rob alone", macro(rob, Y)); print("rob+lr", macro(a, Y), pa); print("rob+lr+lgbm", macro(b, Y), pb)
rng = np.random.default_rng(0); d = []
for _ in range(1000):
    s = rng.integers(0, len(Y), len(Y)); d.append(macro(b[s], Y[s]) - macro(a[s], Y[s]))
print("gain", macro(b, Y) - macro(a, Y), "CI95", np.percentile(d, [2.5, 97.5]).tolist())
