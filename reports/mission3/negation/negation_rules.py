"""M3 rule-based negation marking. Frozen before any measurement (stage 0, 2026-10-12).

Rules come only from the Training-holdout error analysis (../error_analysis/README.md, sections 1-3).
Input is the transcript only; the same function runs at train and inference time.

For every keyword mention of a symptom (KW, same as the error analysis):
- question utterance (contains "?" after the mention): look at the next utterance by another speaker.
  It starts with a no-word -> negated; with a yes-word -> affirmed; otherwise unknown (ignored).
- otherwise: a negation within 10 chars after the mention and inside the same clause, "안" right before it,
  or "안 차/아파" inside the match itself -> negated; else affirmed.
Per call and symptom: mentioned / negated_any / affirmed_any, and neg_only = negated_any and not affirmed_any.
"""
import re

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
# Topic-marked forms the evidence regex misses ("열은 없어요", "토는 안 했어요", "피는 안 나요").
KW_EXTRA = {"고열": r"열은|열도|열\s?없", "구토": r"토는|토도", "열상": r"피는|피도"}
NEG_AFTER = re.compile(r"없|아니|않|안\s*(나|해|하|했|아프|아파|차)|괜찮")
NEG_INSIDE = re.compile(r"안\s*(차|아프|아파|아퍼)")  # "숨은 안 차", "머리는 안 아파" match the keyword itself
NEG_BEFORE = re.compile(r"안\s*$")  # "안 어지러워요"
CLAUSE_END = re.compile(r"[.?!,]|고\s|는데|지만|서\s")
WINDOW = 10
NO_ANSWER = re.compile(r"^\s*(아니|아뇨|아녜|아냐|없|안\s)")
YES_ANSWER = re.compile(r"^\s*(네|예|응|어|맞|좀|있)")


def clean(text):
    return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""


def utterances(doc):
    out = []
    for u in doc.get("utterances", []) if isinstance(doc.get("utterances"), list) else []:
        if isinstance(u, dict):
            t = clean(u.get("text", ""))
            if t:
                out.append((u.get("speaker"), t))
    return out


def mentions(doc):
    """Yield (symptom, status, char_start_in_joined_transcript); status is 'neg', 'aff' or 'unk'."""
    utts = utterances(doc)
    offset = 0
    for i, (spk, text) in enumerate(utts):
        for s in SYM:
            pattern = KW[s] + ("|" + KW_EXTRA[s] if s in KW_EXTRA else "")
            for m in re.finditer(pattern, text):
                if "?" in text[m.end():]:
                    nxt = next((t for sp, t in utts[i + 1:] if sp != spk), None)
                    status = "unk" if nxt is None else "neg" if NO_ANSWER.match(nxt) else "aff" if YES_ANSWER.match(nxt) else "unk"
                else:
                    after = text[m.end():m.end() + WINDOW]
                    cut = CLAUSE_END.search(after)
                    after = after[:cut.start()] if cut else after
                    neg = (NEG_AFTER.search(after) or NEG_INSIDE.search(m.group())
                           or NEG_BEFORE.search(text[max(0, m.start() - 3):m.start()]))
                    status = "neg" if neg else "aff"
                yield s, status, offset + m.start()
        offset += len(text) + 1  # joined with a single space, like the training transcript()


def transcript(doc):
    return " ".join(t for _, t in utterances(doc))


def flags(doc):
    """Per symptom: dict(mentioned, negated_any, affirmed_any, neg_only)."""
    f = {s: dict(mentioned=False, negated_any=False, affirmed_any=False) for s in SYM}
    for s, status, _ in mentions(doc):
        f[s]["mentioned"] = True
        if status == "neg":
            f[s]["negated_any"] = True
        elif status == "aff":
            f[s]["affirmed_any"] = True
    for v in f.values():
        v["neg_only"] = v["negated_any"] and not v["affirmed_any"]
    return f
