"""Holdout residual probe for the char n-gram gate.

Proxy baseline: 0.55 * ROBERTA-003 seed 42 + 0.45 * P0 word TF-IDF.
Official Validation probabilities are not read. speaker and other metadata are not read.
Aggregates only are written.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

LABELS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
WEAK = ("오심", "두통", "전신쇠약")
STRONG = ("열상", "복통")
RELATED = {
    "오심": ("어지러움", "구토", "복통"),
    "두통": ("어지러움",),
    "전신쇠약": ("어지러움", "고열", "호흡곤란"),
}
STRICT = {
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
# Same stems after whitespace removal. A hit here and a miss on STRICT is a spacing-gap mention.
LOOSE = {
    "오심": r"메스껍|메슥|울렁|구역|토할것|토할거|속이안좋|속이이상|속이불편|속뒤집|오심",
    "구토": r"토했|토해|토하|토를|구토|게워|토가|오바이트",
    "두통": r"머리.{0,8}(아프|아파|아퍼|깨질|지끈|띵|쑤)|두통|편두통",
    "어지러움": r"어지러|어지럽|어질|빙빙|핑돌|현기증",
    "전신쇠약": r"기운이?없|힘이?없|쳐지|처지|축늘어|기력|무기력|못일어|쇠약",
    "고열": r"열이|고열|열나|\d{2}\.\d도|체온",
    "복통": r"배.{0,6}(아프|아파|아퍼)|복통|배가|명치|아랫배|윗배",
    "열상": r"찢어|베였|베이|피가|피나|출혈|꿰매|상처",
    "호흡곤란": r"숨.{0,6}(못|안|차|막|쉬기|가빠|헐떡)|호흡|숨이",
}
STRICT_RE = {k: re.compile(v) for k, v in STRICT.items()}
LOOSE_RE = {k: re.compile(v) for k, v in LOOSE.items()}
MACRO_GATE = 0.004


def utterances(doc: dict) -> list[str]:
    out = []
    for utterance in doc.get("utterances", []):
        if not isinstance(utterance, dict):
            continue
        text = utterance.get("text", "")
        if not isinstance(text, str):
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return out


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def macro_from_counts(counts: list[tuple[int, int, int]]) -> float:
    return float(np.mean([f1_from_counts(*c) for c in counts]))


def load_holdout(probs_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r3 = np.load(probs_dir / "r3_holdout.npz", allow_pickle=False)
    p0 = np.load(probs_dir / "p0_holdout.npz", allow_pickle=False)
    if r3["names"].shape != p0["names"].shape or not np.array_equal(r3["names"], p0["names"]):
        raise SystemExit("r3 and p0 holdout names differ")
    if not np.array_equal(r3["gold"], p0["gold"]):
        raise SystemExit("r3 and p0 holdout gold differ")
    blend = 0.55 * r3["probs"].astype(np.float64) + 0.45 * p0["probs"].astype(np.float64)
    pred = (blend >= 0.5).astype(np.int8)
    return r3["names"], r3["gold"].astype(np.int8), pred, blend


def index_labels(label_dir: Path, wanted: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in label_dir.rglob("*.json"):
        if path.name in wanted:
            found[path.name] = path
    missing = wanted - found.keys()
    if missing:
        raise SystemExit(f"missing {len(missing)} holdout json files")
    return found


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    probs_dir = repo / "reports" / "mission3" / "ensemble" / "probs"
    label_dir = repo / "data" / "raw_merged" / "Training" / "2.라벨링데이터"
    if len(sys.argv) == 3:
        probs_dir = Path(sys.argv[1])
        label_dir = Path(sys.argv[2])
    names, gold, pred, _blend = load_holdout(probs_dir)
    paths = index_labels(label_dir, set(names.tolist()))
    n, k = gold.shape
    if k != len(LABELS):
        raise SystemExit(f"expected {len(LABELS)} labels, got {k}")

    counts = []
    for j in range(k):
        tp = int(((gold[:, j] == 1) & (pred[:, j] == 1)).sum())
        fp = int(((gold[:, j] == 0) & (pred[:, j] == 1)).sum())
        fn = int(((gold[:, j] == 1) & (pred[:, j] == 0)).sum())
        counts.append((tp, fp, fn))
    base_macro = macro_from_counts(counts)

    per_label = {
        label: {
            "support": 0, "fn": 0, "fp": 0, "tp": 0,
            "fn_strict": 0, "fn_no_mention": 0, "fn_spacing_gap": 0,
            "fp_related": 0, "fp_question_only": 0,
        }
        for label in WEAK
    }
    gap_fn = np.zeros(k, dtype=np.int32)
    question_fp = np.zeros(k, dtype=np.int32)
    for i, name in enumerate(names.tolist()):
        doc = json.loads(paths[name].read_text(encoding="utf-8-sig"))
        utts = utterances(doc)
        text = " ".join(utts)
        flat = re.sub(r"\s+", "", text)
        symptom = doc.get("symptom", [])
        if isinstance(symptom, str):
            symptom = [symptom]
        gold_names = {
            unicodedata.normalize("NFC", x.strip())
            for x in symptom
            if isinstance(x, str)
        }
        for j, label in enumerate(LABELS):
            strict_hit = bool(STRICT_RE[label].search(text))
            loose_hit = bool(LOOSE_RE[label].search(flat))
            is_fn = gold[i, j] == 1 and pred[i, j] == 0
            is_fp = gold[i, j] == 0 and pred[i, j] == 1
            if is_fn and loose_hit and not strict_hit:
                gap_fn[j] += 1
            if is_fp:
                hit_utts = [u for u in utts if STRICT_RE[label].search(u) or LOOSE_RE[label].search(re.sub(r"\s+", "", u))]
                if hit_utts and all(("?" in u or "？" in u) for u in hit_utts):
                    question_fp[j] += 1
        for j, label in enumerate(LABELS):
            if label not in WEAK:
                continue
            strict_hit = bool(STRICT_RE[label].search(text))
            loose_hit = bool(LOOSE_RE[label].search(flat))
            slot = per_label[label]
            if gold[i, j] == 1:
                slot["support"] += 1
            if gold[i, j] == 1 and pred[i, j] == 1:
                slot["tp"] += 1
            if gold[i, j] == 1 and pred[i, j] == 0:
                slot["fn"] += 1
                if strict_hit:
                    slot["fn_strict"] += 1
                elif loose_hit:
                    slot["fn_spacing_gap"] += 1
                else:
                    slot["fn_no_mention"] += 1
            if gold[i, j] == 0 and pred[i, j] == 1:
                slot["fp"] += 1
                if any(rel in gold_names for rel in RELATED[label]):
                    slot["fp_related"] += 1
                hit_utts = [u for u in utts if STRICT_RE[label].search(u) or LOOSE_RE[label].search(re.sub(r"\s+", "", u))]
                if hit_utts and all(("?" in u or "？" in u) for u in hit_utts):
                    slot["fp_question_only"] += 1

    gap_counts = [(tp + int(gap_fn[j]), fp, fn - int(gap_fn[j])) for j, (tp, fp, fn) in enumerate(counts)]
    q_counts = [(tp, fp - int(question_fp[j]), fn) for j, (tp, fp, fn) in enumerate(counts)]
    weak_q_counts = []
    for j, (tp, fp, fn) in enumerate(counts):
        drop = int(question_fp[j]) if LABELS[j] in WEAK else 0
        weak_q_counts.append((tp, fp - drop, fn))
    gap_macro = macro_from_counts(gap_counts)
    question_macro = macro_from_counts(q_counts)
    weak_question_macro = macro_from_counts(weak_q_counts)
    weak_question_share = {
        label: round(per_label[label]["fp_question_only"] / per_label[label]["fp"], 4) if per_label[label]["fp"] else 0.0
        for label in WEAK
    }
    # A turn-separator probe is justified only when question-only hits are a large share of weak-label FPs.
    substantial_question = sum(share >= 0.25 for share in weak_question_share.values()) >= 2
    summary = {
        "baseline": "0.55 * r3_holdout + 0.45 * p0_holdout, threshold 0.5",
        "n": int(n),
        "macro_f1": round(base_macro, 4),
        "per_label_f1": {
            LABELS[j]: round(f1_from_counts(*counts[j]), 4) for j in range(k)
        },
        "weak": per_label,
        "spacing_gap_fn": {LABELS[j]: int(gap_fn[j]) for j in range(k)},
        "spacing_gap_fn_total": int(gap_fn.sum()),
        "spacing_gap_oracle_macro": round(gap_macro, 4),
        "spacing_gap_oracle_delta": round(gap_macro - base_macro, 4),
        "question_only_fp": {LABELS[j]: int(question_fp[j]) for j in range(k)},
        "question_only_fp_total": int(question_fp.sum()),
        "question_oracle_macro": round(question_macro, 4),
        "question_oracle_delta": round(question_macro - base_macro, 4),
        "weak_question_fp_share": weak_question_share,
        "weak_question_oracle_delta": round(weak_question_macro - base_macro, 4),
        "run_char_model": bool(gap_macro - base_macro >= MACRO_GATE),
        "run_turn_probe": bool(substantial_question and weak_question_macro - base_macro >= MACRO_GATE),
        "gate": MACRO_GATE,
        "official_val_used": False,
    }
    out = Path(__file__).resolve().parent / "residual_probe.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "n", "macro_f1", "spacing_gap_fn_total", "spacing_gap_oracle_delta",
        "question_only_fp_total", "question_oracle_delta", "weak_question_fp_share",
        "weak_question_oracle_delta", "run_char_model", "run_turn_probe",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
