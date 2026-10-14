"""Stage 0: headroom of rule-based negation marking, on the internal 80:20 holdout only (no official Val).

- rule precision: of (call, symptom) cells marked neg_only, how many are gold-negative
- per model: macro F1 if neg_only cells are forced to 0 (hard rule, diagnostic only — not a submission
  post-process) and if only the FPs among them are flipped (oracle upper bound)
- truncation: share of neg_only cells whose negated mentions all lie beyond RoBERTa's 256-token window
Only aggregates are written.

python stage0_probe.py <probs_dir> <training_label_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from negation_rules import SYM, flags, mentions, transcript  # noqa: E402

ENSEMBLE = {"r3": .45, "k12": .15, "p0": .40}  # holdout-selected weights (../ensemble/README.md, 2nd round)
MAX_CONTENT_TOKENS = 254  # max_length 256 minus <s> and </s>


def macro(gold, pred):
    return float(f1_score(gold, pred, average="macro", zero_division=0))


def main():
    probs_dir, label_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    z = {t: np.load(probs_dir / f"{t}_holdout.npz", allow_pickle=True) for t in ENSEMBLE}
    names = list(z["r3"]["names"])
    gold = z["r3"]["gold"].astype(int)
    for t in ENSEMBLE:
        assert list(z[t]["names"]) == names and (z[t]["gold"] == gold).all(), t
    probs = {t: z[t]["probs"].astype(float) for t in ENSEMBLE}
    probs["ens3"] = sum(w * probs[t] for t, w in ENSEMBLE.items())

    paths = {p.name: p for p in label_dir.rglob("*.json")}
    docs = [json.loads(paths[n].read_text(encoding="utf-8-sig")) for n in names]
    F = [flags(d) for d in docs]
    neg_only = np.array([[f[s]["neg_only"] for s in SYM] for f in F])
    mentioned = np.array([[f[s]["mentioned"] for s in SYM] for f in F])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("klue/roberta-large", local_files_only=True)
    beyond = np.zeros_like(neg_only)
    long_calls = 0
    for i, d in enumerate(docs):
        enc = tok(transcript(d), add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if len(offs) > MAX_CONTENT_TOKENS:
            long_calls += 1
        limit_char = offs[MAX_CONTENT_TOKENS][0] if len(offs) > MAX_CONTENT_TOKENS else float("inf")
        neg_pos = {}
        for s, status, pos in mentions(d):
            if status == "neg":
                neg_pos.setdefault(s, []).append(pos)
        for j, s in enumerate(SYM):
            if neg_only[i, j]:
                beyond[i, j] = min(neg_pos[s]) >= limit_char

    res = dict(rows=len(names), split="internal 80:20 holdout (Training only)",
               long_calls_over_256_tokens=round(long_calls / len(names), 3), by_symptom={}, models={})
    for j, s in enumerate(SYM):
        n, m = neg_only[:, j], mentioned[:, j]
        res["by_symptom"][s] = dict(
            neg_only=int(n.sum()),
            neg_only_gold_negative=round(float((gold[n, j] == 0).mean()), 3) if n.any() else None,
            mentioned_gold_negative=round(float((gold[m, j] == 0).mean()), 3) if m.any() else None,
            all_gold_negative=round(float((gold[:, j] == 0).mean()), 3),
            neg_only_beyond_256=int(beyond[:, j].sum()),
        )
    for t, p in probs.items():
        pred = (p >= .5).astype(int)
        fp = (pred == 1) & (gold == 0)
        tp = (pred == 1) & (gold == 1)
        hard = np.where(neg_only, 0, pred)
        oracle = np.where(neg_only & fp, 0, pred)
        oracle_seen = np.where(neg_only & fp & ~beyond, 0, pred)
        base = macro(gold, pred)
        res["models"][t] = dict(
            macro_f1=round(base, 4),
            fp=int(fp.sum()), fp_neg_only=int((fp & neg_only).sum()), tp_neg_only=int((tp & neg_only).sum()),
            fp_neg_only_beyond_256=int((fp & neg_only & beyond).sum()),
            delta_hard_rule=round(macro(gold, hard) - base, 4),
            delta_oracle=round(macro(gold, oracle) - base, 4),
            delta_oracle_within_256=round(macro(gold, oracle_seen) - base, 4),
            per_symptom_fp_neg_only={s: int((fp & neg_only)[:, j].sum()) for j, s in enumerate(SYM)},
        )
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
