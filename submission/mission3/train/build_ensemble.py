"""Pack several Mission 3 BERT checkpoints and one TF-IDF model into one ensemble .pt (dcc-m3-ens-v1).

    python train/build_ensemble.py --bert seed42.pt seed43.pt seed44.pt --bert_weight 0.55 \
        --tfidf tfidf.joblib --tfidf_weight 0.45 --out m3_ensemble.pt

The BERT weight is split equally over the given checkpoints (their probabilities are averaged), so
p = 0.55 * mean(seeds) + 0.45 * tfidf. Weights must be fixed in advance (chosen on the internal
Training holdout, never on the official Validation). The TF-IDF joblib is the P0 checkpoint dict
({vectorizer, classifier, target_symptoms}); it is stored as plain arrays so the result loads with
torch.load(weights_only=True) and inference does not unpickle sklearn objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
TFIDF_PARAMS = ("analyzer", "ngram_range", "lowercase", "token_pattern", "sublinear_tf", "norm",
                "use_idf", "smooth_idf", "strip_accents", "binary")


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def tfidf_member(path, weight):
    import joblib
    import numpy as np
    import torch

    ck = joblib.load(path)
    vec, clf = ck["vectorizer"], ck["classifier"]
    if [str(x) for x in ck["target_symptoms"]] != LABELS:
        raise ValueError("TF-IDF target_symptoms differ from the 9 labels")
    params = vec.get_params()
    if params["norm"] != "l2" or not params["use_idf"] or params["binary"] or params["analyzer"] != "word":
        raise ValueError(f"unsupported TF-IDF parameters for the array form: {params}")
    if not getattr(clf, "multilabel_", False) or len(clf.estimators_) != len(LABELS):
        raise ValueError("expected a multilabel OneVsRestClassifier over the 9 labels")
    vocab = [None] * len(vec.vocabulary_)
    for term, i in vec.vocabulary_.items():
        vocab[i] = term
    if any(not hasattr(e, "coef_") for e in clf.estimators_):
        raise ValueError("a symptom has only one class in the TF-IDF training data (constant predictor)")
    coef = np.vstack([e.coef_.ravel() for e in clf.estimators_])
    intercept = np.array([float(e.intercept_.ravel()[0]) for e in clf.estimators_])
    return dict(kind="tfidf", weight=float(weight), source=dict(file=Path(path).name, sha256=sha256(path)),
                params={k: (list(params[k]) if k == "ngram_range" else params[k]) for k in TFIDF_PARAMS},
                vocabulary=vocab, idf=torch.tensor(vec.idf_, dtype=torch.float64),
                coef=torch.tensor(coef, dtype=torch.float64), intercept=torch.tensor(intercept, dtype=torch.float64))


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bert", type=Path, nargs="+", required=True)
    ap.add_argument("--bert_weight", type=float, required=True)
    ap.add_argument("--tfidf", type=Path)
    ap.add_argument("--tfidf_weight", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if abs(args.bert_weight + (args.tfidf_weight if args.tfidf else 0.0) - 1.0) > 1e-9:
        parser_error = "weights must sum to 1"
        raise SystemExit(parser_error)
    members = []
    for p in args.bert:
        ck = torch.load(p, map_location="cpu", weights_only=True)
        if ck.get("format") != "dcc-m3-bert-v1" or [str(x) for x in ck["labels"]] != LABELS:
            raise ValueError(f"{p} is not a 9-label dcc-m3-bert-v1 checkpoint")
        members.append(dict(kind="bert", weight=args.bert_weight / len(args.bert),
                            source=dict(file=p.name, sha256=sha256(p)), checkpoint=ck))
    if args.tfidf:
        members.append(tfidf_member(args.tfidf, args.tfidf_weight))
    payload = dict(format="dcc-m3-ens-v1", labels=LABELS, threshold=0.5,
                   rule="p = sum(weight * member probability); label if p >= 0.5; no post-processing",
                   members=members)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    torch.load(args.out, map_location="cpu", weights_only=True)  # must load in the safe mode used at inference
    record = dict(output=args.out.name, output_bytes=args.out.stat().st_size, output_sha256=sha256(args.out),
                  members=[dict(kind=m["kind"], weight=m["weight"], **m["source"]) for m in members])
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
