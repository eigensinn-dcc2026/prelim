"""제안 3 (TF-IDF 공간 kNN) + TF-IDF·LR 5-fold OOF, CPU 전용.
  fold 분할 = train_structured.py --mode kfold --seed 42 (M3-ROB5F-001) 와 같음 -> fold{k}_idx.npy 로 검증
  TF-IDF·LR 설정 = P0 (word 1-2gram, min_df 2, max_features 50000, sublinear_tf, OvR LR balanced liblinear)
  kNN: p_i(x) = Σ_n softmax(cos(x,n)/τ) y_{n,i}, 이웃은 해당 fold 의 학습 통화만 (자기 자신·검증 통화 제외)
  평가: 멤버 가중치·(k, τ)를 4개 fold 에서 고르고 남은 fold 에서 채점(교차 적합) -> 선택 편향 없는 이득 추정
  threshold 0.5 고정, 후처리 없음, 공식 Val 미사용.
python knn_tfidf_oof.py --rows_json <M3-ROBERTA-003>/rows.json --rob_dir <M3-ROB5F-001> --out_dir <out>
"""
import argparse, itertools, json, os, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

KS, TAUS = (20, 50, 100), (0.02, 0.05, 0.1)


def macro(P, Y):
    """sklearn f1_score(average="macro", zero_division=0) 와 같은 값, 가중치 탐색용으로 빠르게"""
    pred = P >= 0.5; y = Y > 0.5
    tp = (pred & y).sum(0); denom = pred.sum(0) + y.sum(0)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def knn_probs(Xq, Xr, Yr, chunk=1024):
    """모든 (k, τ) 조합의 kNN 확률. 반환 {(k, τ): [n_q, 9]}"""
    out = {c: np.zeros((Xq.shape[0], Yr.shape[1]), np.float32) for c in itertools.product(KS, TAUS)}
    kmax = min(max(KS), Xr.shape[0])
    for s in range(0, Xq.shape[0], chunk):
        S = (Xq[s:s + chunk] @ Xr.T).toarray()                 # cosine (tfidf 는 L2 정규화됨)
        top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
        ts = np.take_along_axis(S, top, 1)
        order = np.argsort(-ts, 1); top, ts = np.take_along_axis(top, order, 1), np.take_along_axis(ts, order, 1)
        for k, tau in out:
            w = np.exp((ts[:, :k] - ts[:, :1]) / tau); w /= w.sum(1, keepdims=True)
            out[(k, tau)][s:s + chunk] = np.einsum("qk,qkl->ql", w, Yr[top[:, :k]])
    return out


def simplex(n, step=0.05):
    g = np.round(np.arange(0, 1 + 1e-9, step), 2)
    return [w for w in itertools.product(g, repeat=n) if abs(sum(w) - 1) < 1e-9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_json", required=True); ap.add_argument("--rob_dir", required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    rows = json.load(open(a.rows_json, encoding="utf-8"))
    texts = [r["text"] for r in rows]; Y = np.array([r["labels"] for r in rows], np.float32)
    splits = list(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed).split(np.zeros((len(Y), 1)), Y))
    for k, (_, va) in enumerate(splits):
        assert (np.load(os.path.join(a.rob_dir, f"fold{k}_idx.npy")) == va).all(), f"fold{k} 분할이 ROB5F 와 다름"
    rob = np.load(os.path.join(a.rob_dir, "oof_probs.npy"))
    lr = np.zeros_like(Y); knn = {c: np.zeros_like(Y) for c in itertools.product(KS, TAUS)}
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr]).astype(np.float32); Xva = vec.transform([texts[i] for i in va]).astype(np.float32)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=a.seed), n_jobs=1)
        clf.fit(Xtr, Y[tr]); lr[va] = clf.predict_proba(Xva)
        for c, p in knn_probs(Xva, Xtr, Y[tr]).items():
            knn[c][va] = p
        print(f"fold{k}: LR {macro(lr[va], Y[va]):.4f} | kNN best {max(macro(knn[c][va], Y[va]) for c in knn):.4f} ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(a.out_dir, "tfidf_lr_oof.npy"), lr)
    for (kk, tau), p in knn.items():
        np.save(os.path.join(a.out_dir, f"knn_k{kk}_t{tau}_oof.npy"), p)

    res = dict(single=dict(rob=macro(rob, Y), tfidf_lr=macro(lr, Y), **{f"knn_k{c[0]}_t{c[1]}": macro(p, Y) for c, p in knn.items()}))
    # 교차 적합: fold j 를 뺀 4개 fold 에서 (구성, 가중치) 선택 -> fold j 채점. 두 조합(rob+lr / rob+lr+knn) 비교
    W2, W3 = simplex(2), simplex(3)
    cross = {"rob+lr": np.zeros_like(Y), "rob+lr+knn": np.zeros_like(Y)}; picks = []
    for j, (_, va) in enumerate(splits):
        fit = np.concatenate([splits[i][1] for i in range(5) if i != j])
        w2 = max(W2, key=lambda w: macro(w[0] * rob[fit] + w[1] * lr[fit], Y[fit]))
        cross["rob+lr"][va] = w2[0] * rob[va] + w2[1] * lr[va]
        c3, w3 = max(((c, w) for c in knn for w in W3),
                     key=lambda cw: macro(cw[1][0] * rob[fit] + cw[1][1] * lr[fit] + cw[1][2] * knn[cw[0]][fit], Y[fit]))
        cross["rob+lr+knn"][va] = w3[0] * rob[va] + w3[1] * lr[va] + w3[2] * knn[c3][va]
        picks.append(dict(fold=j, rob_lr=w2, rob_lr_knn=w3, knn=c3))
        print(f"fold{j}: rob+lr {w2} -> {macro(cross['rob+lr'][va], Y[va]):.4f} | rob+lr+knn {w3} k,τ={c3} -> {macro(cross['rob+lr+knn'][va], Y[va]):.4f}", flush=True)
    res["cross_fitted"] = {n: macro(p, Y) for n, p in cross.items()}
    res["cross_fitted"]["knn_gain"] = res["cross_fitted"]["rob+lr+knn"] - res["cross_fitted"]["rob+lr"]
    # 부트스트랩 95% CI (통화 단위)
    rng = np.random.default_rng(0); d = []
    for _ in range(1000):
        b = rng.integers(0, len(Y), len(Y))
        d.append(macro(cross["rob+lr+knn"][b], Y[b]) - macro(cross["rob+lr"][b], Y[b]))
    res["cross_fitted"]["knn_gain_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    res["picks"] = [dict(p, knn=list(p["knn"])) for p in picks]
    json.dump(res, open(os.path.join(a.out_dir, "knn_tfidf_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res["single"], indent=1, default=float)); print(json.dumps(res["cross_fitted"], indent=1, default=float))


if __name__ == "__main__":
    main()
