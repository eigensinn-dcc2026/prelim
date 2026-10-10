"""M3 error analysis on the P0 80:20 holdout predictions (TF-IDF). Text-only, Training data only."""
import ast, json, os, re, sys, collections
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ["DCC_DATA_ROOT"])
PRED = Path(sys.argv[1])
OUT = Path(sys.argv[2])
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
NEG = r"(안\s*(나|하|해|했|아프)|없|아니|않)"

idx = {p.name: p for p in (ROOT / "Training").rglob("*.json")}
df = pd.read_csv(PRED, encoding="utf-8-sig")
rows = []
for r in df.itertuples(index=False):
    d = json.loads(idx[r[0]].read_text(encoding="utf-8"))
    utt = d["utterances"]
    rows.append(dict(
        name=r[0], true=set(ast.literal_eval(r[1])), pred=set(ast.literal_eval(r[2])),
        all_labels=d.get("symptom", []),
        caller=" ".join(u["text"] for u in utt if u.get("speaker") == 1),
        oper=" ".join(u["text"] for u in utt if u.get("speaker") == 0),
        text=" ".join(u["text"] for u in utt),
    ))
print("n =", len(rows))

rep = {}
for s in SYM:
    kw = re.compile(KW[s])
    pos = [x for x in rows if s in x["true"]]
    neg = [x for x in rows if s not in x["true"]]
    tp = [x for x in pos if s in x["pred"]]; fn = [x for x in pos if s not in x["pred"]]
    fp = [x for x in neg if s in x["pred"]]
    P = len(tp) / max(1, len(tp) + len(fp)); R = len(tp) / max(1, len(pos))
    kw_pos = sum(bool(kw.search(x["text"])) for x in pos) / len(pos)
    kw_neg = sum(bool(kw.search(x["text"])) for x in neg) / len(neg)
    kw_pos_caller = sum(bool(kw.search(x["caller"])) for x in pos) / len(pos)
    # 키워드가 있는데 라벨이 없는 샘플 수 / 키워드 있는 샘플 중 비율
    kw_any = [x for x in rows if kw.search(x["text"])]
    kw_unlabeled = [x for x in kw_any if s not in x["true"]]
    # FP 때 실제 라벨로 무엇이 붙어 있었나, FN 때 대신 무엇을 예측했나
    fp_true = collections.Counter(t for x in fp for t in x["true"])
    fn_pred = collections.Counter(t for x in fn for t in (x["pred"] - x["true"]))
    fn_nokw = sum(not kw.search(x["text"]) for x in fn) / max(1, len(fn))
    rep[s] = dict(
        support=len(pos), prevalence=round(len(pos) / len(rows), 3),
        precision=round(P, 3), recall=round(R, 3), f1=round(2 * P * R / max(1e-9, P + R), 3),
        kw_rate_in_pos=round(kw_pos, 3), kw_rate_in_pos_caller=round(kw_pos_caller, 3),
        kw_rate_in_neg=round(kw_neg, 3),
        label_given_kw=round(1 - len(kw_unlabeled) / max(1, len(kw_any)), 3),
        fn_without_kw=round(fn_nokw, 3),
        fp_top_true=fp_true.most_common(4), fn_top_pred_instead=fn_pred.most_common(4),
    )

# 오심/구토 공동 라벨 일관성
both = lambda a, b: sum(a in x["true"] and b in x["true"] for x in rows)
pair = {
    "오심만": sum("오심" in x["true"] and "구토" not in x["true"] for x in rows),
    "구토만": sum("구토" in x["true"] and "오심" not in x["true"] for x in rows),
    "둘다": both("오심", "구토"),
}
# 구토 표현만 있고(메스꺼움 표현 없음) 오심 라벨이 붙은 경우 등
kv, kn = re.compile(KW["구토"]), re.compile(KW["오심"])
pair["오심라벨_구토표현만"] = sum(1 for x in rows if "오심" in x["true"] and kv.search(x["text"]) and not kn.search(x["text"]))
pair["오심라벨_표현둘다없음"] = sum(1 for x in rows if "오심" in x["true"] and not kv.search(x["text"]) and not kn.search(x["text"]))
pair["메스꺼움표현있음_오심라벨없음"] = sum(1 for x in rows if kn.search(x["text"]) and "오심" not in x["true"])
pair["메스꺼움표현있음_전체"] = sum(bool(kn.search(x["text"])) for x in rows)

# 두통: 상담원 질문만 있고 신고자 부정 답변 가능성
kh = re.compile(KW["두통"])
pair["두통라벨_신고자발화에만표현없음(상담원만)"] = sum(1 for x in rows if "두통" in x["true"] and kh.search(x["oper"]) and not kh.search(x["caller"]))
pair["두통라벨_표현없음"] = sum(1 for x in rows if "두통" in x["true"] and not kh.search(x["text"]))
nonmet = collections.Counter(l for x in rows for l in x["all_labels"] if l not in SYM)

# 사례 추출 (오심 FP/FN, 두통 FN 각 6건, 앞 300자)
def ex(sel, k=6):
    return [dict(name=x["name"], true=sorted(x["true"]), pred=sorted(x["pred"]), caller=x["caller"][:300]) for x in sel[:k]]
cases = {
    "오심_FN": ex([x for x in rows if "오심" in x["true"] and "오심" not in x["pred"]]),
    "오심_FP": ex([x for x in rows if "오심" not in x["true"] and "오심" in x["pred"]]),
    "두통_FN": ex([x for x in rows if "두통" in x["true"] and "두통" not in x["pred"]]),
}
OUT.write_text(json.dumps(dict(per_symptom=rep, pair=pair, non_target_labels=nonmet.most_common(10), cases=cases),
                          ensure_ascii=False, indent=1), encoding="utf-8")
for s, v in rep.items():
    print(s, {k: v[k] for k in ("f1", "precision", "recall", "kw_rate_in_pos", "kw_rate_in_neg", "label_given_kw", "fn_without_kw")})
    print("   FP true:", v["fp_top_true"], "| FN pred instead:", v["fn_top_pred_instead"])
print(pair)
print("non-target:", nonmet.most_common(10))

# 텍스트 근거 유무별 recall (근거 = 해당 증상 표현이 통화에 있음)
print("\n증상 | 근거있는 양성 recall (n) | 근거없는 양성 recall (n)")
ev = {}
for s in SYM:
    kw = re.compile(KW[s]); pos = [x for x in rows if s in x["true"]]
    a = [x for x in pos if kw.search(x["text"])]; b = [x for x in pos if not kw.search(x["text"])]
    ra = sum(s in x["pred"] for x in a) / max(1, len(a)); rb = sum(s in x["pred"] for x in b) / max(1, len(b))
    ev[s] = dict(recall_with_evidence=round(ra, 3), n_with=len(a), recall_without=round(rb, 3), n_without=len(b))
    print(s, f"{ra:.3f} ({len(a)})", f"{rb:.3f} ({len(b)})")
j = json.loads(OUT.read_text(encoding="utf-8")); j["recall_by_evidence"] = ev
OUT.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
