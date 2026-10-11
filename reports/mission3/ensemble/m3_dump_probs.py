"""Dump per-symptom probabilities of one Mission 3 checkpoint on a label folder (analysis only).

python m3_dump_probs.py <main_src_dir> <repo_root> <label_dir> <ckpt> <out.npz>
Uses the same code paths as dcc_inference.mission3.predict (text rule per checkpoint kind).
"""
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

src, repo, label_dir, ckpt, out = map(Path, sys.argv[1:6])
sys.path.insert(0, str(src))
from dcc_inference import mission3, mission3_bert  # noqa: E402

paths = sorted(label_dir.rglob("*.json"))
names = [p.name for p in paths]
gold = []
for p in paths:
    doc = json.loads(p.read_text(encoding="utf-8-sig"))
    s = doc.get("symptom", [])
    s = {unicodedata.normalize("NFC", x.strip()) for x in ([s] if isinstance(s, str) else s) if isinstance(x, str)}
    gold.append([int(lab in s) for lab in mission3_bert.LABELS])

t0 = time.time()
if mission3_bert.is_bert_checkpoint(ckpt):
    texts = [mission3_bert.read_text(p) for p in paths]
    model, tok, labels, ml = mission3_bert.load_checkpoint(ckpt)
    probs = mission3_bert.probabilities(model, tok, texts, ml)
else:
    script = mission3._script()
    texts = [str(script.read_one_json(p)["text"]) for p in paths]
    vec, clf, labels = mission3.load_checkpoint(ckpt)
    probs = mission3._probabilities(vec, clf, texts, len(labels))
assert list(labels) == list(mission3_bert.LABELS), labels
np.savez(out, names=np.array(names), probs=np.asarray(probs, dtype=np.float32), gold=np.array(gold, dtype=np.int8))
print(f"{out.name}: rows={len(names)} sec={time.time() - t0:.0f}")
