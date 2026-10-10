"""Text-only evidence windows and conservative Training weak supervision.

Weak states describe a local window, never the absence of a call-level symptom.
They do not use call labels, speaker IDs, or any patient metadata.
"""
from __future__ import annotations
import re
try:
    from dcc_inference.mission3_bert import LABELS, transcript
except ImportError:
    from mission3_bert import LABELS, transcript

IGNORE = -100
POSITIVE, NEGATIVE, UNKNOWN = 0, 1, 2
QUESTION = re.compile(r"\?|(?<!주)세요|나요|습니까|어때|어떻|있으신|있는지")
YES = re.compile(r"^(네|예|응|맞아|맞습니다|있어요|있습니다)[\s,.!~]*$")
NO = re.compile(r"^(아니요|아니오|아뇨|아니|없어요|없습니다)[\s,.!~]*$")
# Auxiliary states use narrower cues than the legacy diagnostic regex.
# "호흡", "배가", and "피가" alone do not assert a symptom.
STATE_KEYWORDS = {
    "고열":r"고열|열이?\s*(?:많|높|나|있)|체온.{0,5}(?:38|39|40|41)",
    "구토":r"토했|토해|토하|토를|구토|게워|오바이트",
    "두통":r"머리.{0,8}(?:아프|아파|아퍼|깨질|지끈|띵|쑤)|두통|편두통",
    "복통":r"배.{0,6}(?:아프|아파|아퍼)|복통|명치.{0,5}(?:아프|아파)",
    "어지러움":r"어지러|어지럽|어질|빙빙|핑\s*돌|현기증",
    "열상":r"찢어|베였|베이|피가?\s*(?:나|흐르)|출혈|꿰매|상처",
    "오심":r"메스껍|메슥|울렁|구역|토할\s*(?:것|거)|속이\s*(?:안\s*좋|이상|불편|뒤집)|오심",
    "전신쇠약":r"기운이?\s*없|힘이?\s*없|기력이?\s*없|쳐지|처지|축\s*늘어|무기력|못\s*일어|쇠약",
    "호흡곤란":r"호흡곤란|숨.{0,6}(?:못|안\s*쉬|차|막|가쁘|가빠|헐떡|힘들)|호흡.{0,5}(?:어렵|곤란|힘들)"}


def utterance_spans(doc):
    parts = []
    utterances = doc.get("utterances", []) if isinstance(doc, dict) else []
    for item in utterances if isinstance(utterances,list) else []:
        text = item.get("text", "") if isinstance(item, dict) else ""
        text = re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""
        if text:
            parts.append(text)
    spans, cursor = [], 0
    for text in parts:
        spans.append((cursor, cursor + len(text)))
        cursor += len(text) + 1
    return parts, spans


def weak_states(parts, keywords=None):
    """Conservative weak rules; conflicting evidence is ignored."""
    keywords=keywords or STATE_KEYWORDS
    states = []
    for label in LABELS:
        cue = re.compile(keywords[label])
        found = set()
        for i, text in enumerate(parts):
            matches = list(cue.finditer(text))
            if not matches:
                continue
            if QUESTION.search(text):
                answers = []
                for following in parts[i+1:i+3]:
                    if QUESTION.search(following):break
                    answers.append(following)
                yes = any(YES.fullmatch(answer) for answer in answers)
                no = any(NO.fullmatch(answer) for answer in answers)
                if yes and not no:
                    found.add(POSITIVE)
                elif no and not yes:
                    found.add(NEGATIVE)
                elif not yes and not no:
                    found.add(UNKNOWN)
                else:
                    found.update((POSITIVE, NEGATIVE))
            else:
                for match in matches:
                    # Absence inside "기력이 없" or "속이 안 좋" asserts a symptom.
                    # Ignore unrelated preceding "없고" and discourse "아니".
                    matched=match.group()
                    tail=text[match.end():match.end()+14]
                    embedded=re.search(r"안\s*(?:아프|아파|울렁|메스)|지\s*않",matched)
                    following=re.match(r"\s*(?:(?:은|는|이|가|도)\s*)?(?:없|아니|지\s*않|하지\s*않)",tail)
                    found.add(NEGATIVE if embedded or following else POSITIVE)
        decisive = found - {UNKNOWN}
        states.append(next(iter(decisive)) if len(decisive) == 1 else
                      UNKNOWN if found == {UNKNOWN} else IGNORE)
    return states


def tokenize_views(tokenizer, doc, mode="multi", max_length=256, max_windows=6):
    """Return token windows and original character ranges; first view stays exact."""
    if mode not in ("front", "head_tail", "multi", "states", "states_removed"):
        raise ValueError(f"Unknown input mode: {mode}")
    text = transcript(doc)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 4 or max_windows < 2:
        raise ValueError("Window capacity >=4 and max_windows >=2 required")
    def view(start, end):
        char_range = (offsets[start][0], offsets[end-1][1]) if end > start else (0, 0)
        return dict(ids=tokenizer.build_inputs_with_special_tokens(ids[start:end]),
                    token_range=(start, end), char_range=char_range)
    front = view(0, min(capacity, len(ids)))
    if mode == "front" or len(ids) <= capacity:
        return [front]
    if mode == "head_tail":
        if tokenizer.sep_token_id is None:
            raise ValueError("head_tail requires an explicit separator")
        left = (capacity-1+1)//2
        right = capacity-1-left
        return [dict(ids=tokenizer.build_inputs_with_special_tokens(
                         ids[:left] + [tokenizer.sep_token_id] + ids[-right:]),
                     token_ranges=((0, left), (len(ids)-right, len(ids))),
                     char_range=None)]
    _, spans = utterance_spans(doc)
    ends = {end for _, end in spans}
    starts = {start for start, _ in spans}
    # Offset boundaries remain those of the exact normalized full transcript.
    boundaries = [i+1 for i, (_, end) in enumerate(offsets) if end in ends]
    turn_starts = [i for i, (start, _) in enumerate(offsets) if start in starts]
    chunks, start = [front], 0
    while start < len(ids):
        limit = min(start+capacity, len(ids))
        candidates = [end for end in boundaries if start < end <= limit]
        end = max(candidates) if candidates and limit < len(ids) else limit
        chunks.append(view(start, end))
        if end == len(ids):
            break
        preceding = [i for i in turn_starts if start < i < end]
        overlap = preceding[-2] if len(preceding) >= 2 else end
        start = overlap if overlap > start else end
    unique, seen = [], set()
    for chunk in chunks:
        if chunk["token_range"] not in seen:
            unique.append(chunk); seen.add(chunk["token_range"])
    if len(unique) > max_windows:
        # Bounded uniform selection retains front and final evidence; report coverage.
        positions = sorted({round(i*(len(unique)-1)/(max_windows-1)) for i in range(max_windows)})
        unique = [unique[i] for i in positions]
    return unique


def window_states(doc, views, keywords):
    parts, spans = utterance_spans(doc)
    result = []
    for view in views:
        region = view.get("char_range")
        if region is None:
            result.append([IGNORE]*len(LABELS)); continue
        a, b = region
        # Supervision only from fully visible utterances; truncated answers cannot label a window.
        visible = [part for part, (start, end) in zip(parts, spans) if a <= start and end <= b]
        result.append(weak_states(visible, keywords))
    return result
