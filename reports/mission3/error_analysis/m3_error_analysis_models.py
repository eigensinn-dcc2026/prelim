"""M3 error analysis by model: TF-IDF (P0), KcBERT (KCBERT-005 seed 42), and their ensemble.

Inputs are probability dumps (npz: names, probs, gold) for the same rows, plus the label JSON folder
for the transcript text. Evidence = the symptom's keyword regex (KW in m3_error_analysis.py) matches
the transcript. Decisions are p >= 0.5. Only aggregates are written; no transcript text.

python m3_error_analysis_models.py <probs_dir> <holdout_label_dir> <val_label_dir> <out.json>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
SYM = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
KW = {
    "오심": r"메스껍|메슥|울렁|구역|토할\s*것|토할\s*거|속이\s*(안\s*좋|이상|불편|뒤집)|오심",
    "구토": r"토했|토해|토하|토를|구토|게워|토가|오바이트",
    "두통": r"머리.{0,8}(아프|아파|아퍼|깨질|지끈|띵|쑤)|두통|편두통",
    "어지러움": r"어지러|어지럽|어질|빙빙|핑\s*돌|현기증",
    "전신쇠약": r"기운이?\s*없|힘이?\s*없|쳐지|처지|축\s*늘어|기력|무기력|못\s*일어|쇠약",
    "고열": r"열이|고열|열나|열 나|\d{2}\.\d\s*도|체온",
    "복통": r"배.{0,6}(아프|아파|아퍼)|복통|배가|명치|아랫배|윗배",
    "열상": r"찢어|베였|베이|피가|피 나|출혈|꿰매|상처",
    "호흡곤란": r"숨.{0,6}(못|안|차|막|쉬기|가빠|헐떡)|호흡|숨이",
}
# A negation right after the keyword ("열은 없어요", "숨은 안 차요").
NEG_AFTER = r".{0,12}?(없|아니|않|안\s*(나|해|하|아프|차))"
W_ENSEMBLE = 0.55  # chosen on the internal holdout (see official_val README); never on the official Val


def transcript(doc):
    texts = []
    for u in doc.get("utterances", []) if isinstance(doc.get("utterances"), list) else []:
        if isinstance(u, dict) and isinstance(u.get("text"), str):
            t = re.sub(r"\s+", " ", u["text"]).strip()
            if t:
                texts.append(t)
    return " ".join(texts)


def load(probs_dir, split):
    out = {}
    for tag, name in (("tfidf", "p0"), ("kcbert", "k5")):
        z = np.load(Path(probs_dir) / f"{name}_{split}.npz")
        out[tag] = z["probs"].astype(float)
        names, gold = z["names"], z["gold"].astype(int)
    out["ensemble"] = W_ENSEMBLE * out["kcbert"] + (1 - W_ENSEMBLE) * out["tfidf"]
    return list(names), gold, {k: (v >= 0.5).astype(int) for k, v in out.items()}


def analyse(names, gold, preds, label_dir):
    paths = {p.name: p for p in Path(label_dir).rglob("*.json")}
    texts = [transcript(json.loads(paths[n].read_text(encoding="utf-8-sig"))) for n in names]
    ev = np.array([[bool(re.search(KW[s], t)) for s in SYM] for t in texts])
    neg = np.array([[bool(re.search("(" + KW[s] + ")" + NEG_AFTER, t)) for s in SYM] for t in texts])
    res = {"rows": len(names), "by_symptom": {}, "complementarity": {}, "summary": {}}
    for j, s in enumerate(SYM):
        pos, e = gold[:, j] == 1, ev[:, j]
        row = {"support": int(pos.sum()), "evidence_rate_in_pos": round(float(e[pos].mean()), 3)}
        for m, p in preds.items():
            y = p[:, j] == 1
            tp, fp = int((y & pos).sum()), int((y & ~pos).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, int(pos.sum()))
            row[m] = dict(
                f1=round(2 * prec * rec / max(1e-9, prec + rec), 3), precision=round(prec, 3), recall=round(rec, 3),
                recall_with_evidence=round(float(y[pos & e].mean()), 3) if (pos & e).any() else None,
                recall_without_evidence=round(float(y[pos & ~e].mean()), 3) if (pos & ~e).any() else None,
                fp_with_keyword=int((y & ~pos & e).sum()),
                fp_with_negated_keyword=int((y & ~pos & neg[:, j]).sum()),
            )
        res["by_symptom"][s] = row
    # Complementarity on positives: who catches what the other misses.
    t, k = preds["tfidf"] == 1, preds["kcbert"] == 1
    P = gold == 1
    res["complementarity"] = dict(
        positives=int(P.sum()),
        both_hit=int((t & k & P).sum()), only_tfidf_hit=int((t & ~k & P).sum()),
        only_kcbert_hit=int((~t & k & P).sum()), both_miss=int((~t & ~k & P).sum()),
        both_miss_without_evidence=int((~t & ~k & P & ~ev).sum()),
        false_pos_only_tfidf=int((t & ~k & ~P).sum()), false_pos_only_kcbert=int((~t & k & ~P).sum()),
    )
    E = P & ev
    res["summary"] = {m: dict(
        recall_with_evidence=round(float((p == 1)[E].mean()), 3),
        recall_without_evidence=round(float((p == 1)[P & ~ev].mean()), 3),
        empty_rows=int((p.sum(1) == 0).sum()),
        positive_labels_with_evidence=int(E.sum()), positive_labels_without_evidence=int((P & ~ev).sum()),
    ) for m, p in preds.items()}
    return res


def main():
    probs_dir, hold_dir, val_dir, out = sys.argv[1:5]
    result = {"ensemble_weight_kcbert": W_ENSEMBLE}
    for split, label_dir in (("holdout", hold_dir), ("val", val_dir)):
        names, gold, preds = load(probs_dir, split)
        result[split] = analyse(names, gold, preds, label_dir)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for split in ("holdout", "val"):
        r = result[split]
        print(f"== {split} rows={r['rows']}")
        for m, v in r["summary"].items():
            print(f"  {m:9s} recall_w_ev={v['recall_with_evidence']} recall_wo_ev={v['recall_without_evidence']} empty={v['empty_rows']}")
        print("  complementarity:", r["complementarity"])
        for s in SYM:
            row = r["by_symptom"][s]
            print(f"  {s:5s} ev={row['evidence_rate_in_pos']:.2f} " + " | ".join(
                f"{m[:2]} F1 {row[m]['f1']:.3f} R+ {row[m]['recall_with_evidence']} R- {row[m]['recall_without_evidence']} "
                f"FPneg {row[m]['fp_with_negated_keyword']}" for m in ("tfidf", "kcbert", "ensemble")))


if __name__ == "__main__":
    main()
