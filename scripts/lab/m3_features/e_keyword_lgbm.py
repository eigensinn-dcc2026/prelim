"""M3-E: 증상 키워드·부정 구조 특징 + LightGBM 5-fold OOF (CPU 전용).

  입력 = 전사 텍스트만(발화 text 이어붙임, speaker 등 메타데이터 미사용).
  분할 = MultilabelStratifiedKFold(5, shuffle, seed 42), 행 순서 = 파일명 정렬(M3-ROB5F-001과 동일 규약).
  특징 = 증상별 키워드 (전체/비부정/부정 횟수, 첫 등장 위치, 앞 30% 안 등장) + 문맥 단어 횟수 + 길이.
  판정 threshold=0.5 고정, 후처리 없음, 공식 Val 미사용. 확률 calibration 은 하지 않음.

python e_keyword_lgbm.py --data_dir <.../TL_서울_구급> --out_dir <out>
"""
import argparse, json, os, re, time, unicodedata
import numpy as np
import lightgbm as lgb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
KW = {
    "고열": r"(?<![가-힣])열(이|은|도|나|을|가|\s)|고열|발열|체온|\d{2}(\.\d)?\s?도|펄펄|끓",
    "구토": r"구토|토했|토하|토해|토를|토가|토도|토할|토사|게워|토 했|토 해",
    "두통": r"두통|머리(가|를|는|도)?\s*(아|지끈|깨질|띵|무거)|머리 통증|뒷골",
    "복통": r"복통|배(가|를|는|도)?\s*(아|쑤|땡|당기|꼬이)|아랫배|윗배|명치|복부",
    "어지러움": r"어지|어질|현기|핑\s?(돌|도)|빙글|휘청|비틀",
    "열상": r"열상|찢어|찢긴|베였|베이|베인|피가|피를|출혈|피나|피 나|상처|다쳤|절단|찔렸|긁|찰과",
    "오심": r"메스꺼|울렁|속이 안|속 안|속이 좋지|구역|니글|토할 것|오심|속이 불편|속이 미식|속이 이상",
    "전신쇠약": r"기운(이)?\s*(없|빠)|힘(이|도)?\s*(없|빠|들)|무기력|쇠약|축 처|늘어|탈진|쓰러|못 일어|기력|맥이",
    "호흡곤란": r"숨(이|을|도)?\s*(차|쉬|못|가쁘|막|안)|호흡|가쁘|헐떡|숨쉬|질식|컥",
}
CTX = ["흉통|가슴", "넘어|떨어|부딪|낙상", "사고|교통", "설사", "소변|대변|변이", "의식|반응|깨우", "경련|발작|떨",
       "당뇨|혈당", "혈압", "술|음주", "약|복용", "백신|주사", "임신", "알레르기|가려|두드러기", "붓|부어", "마비|저리|감각",
       "입술|청색|창백", "식은땀|땀", "기침|가래", "감기|코로나", "열감", "목|인후", "허리|등", "다리|발", "팔|손", "눈|시야", "귀", "코피",
       "아프|아파", "괜찮|없어요|없고", "못 |못해", "계속", "갑자기", "어제|며칠|오래"]
NEG = re.compile(r"없|않|아니|안 |괜찮|못 ")
W = 10  # 부정 판단 창(문자 수)


def load_rows(data_dir):
    rows = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".json"):
            continue
        doc = json.load(open(os.path.join(data_dir, name), encoding="utf-8-sig"))
        text = " ".join(re.sub(r"\s+", " ", u["text"]).strip() for u in doc.get("utterances", [])
                        if isinstance(u, dict) and isinstance(u.get("text"), str) and u["text"].strip())
        sym = doc.get("symptom", [])
        sym = [sym] if isinstance(sym, str) else sym
        sym = {unicodedata.normalize("NFC", s.strip()) for s in sym if isinstance(s, str)}
        rows.append((unicodedata.normalize("NFC", name), text, [int(s in sym) for s in LABELS]))
    return rows


def feats(text, kw_re, ctx_re):
    n = max(len(text), 1)
    f = [len(text), text.count(" ")]
    for p in kw_re:
        pos = neg = 0; first = first_pos = -1.0
        for m in p.finditer(text):
            around = text[max(0, m.start() - W): m.end() + W]
            if NEG.search(text[m.end(): m.end() + W]):
                neg += 1
            else:
                pos += 1
                if first_pos < 0:
                    first_pos = m.start() / n
            if first < 0:
                first = m.start() / n
        f += [pos + neg, pos, neg, first, first_pos, float(0 <= first_pos < 0.3)]
    f += [len(p.findall(text)) for p in ctx_re]
    return f


def macro(P, Y, thr=0.5):
    pred = P >= thr; y = Y > 0.5
    tp = (pred & y).sum(0); d = pred.sum(0) + y.sum(0)
    per = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
    return float(per.mean()), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    rows = load_rows(a.data_dir)
    texts = [r[1] for r in rows]; Y = np.array([r[2] for r in rows], np.float32)
    print(f"rows {len(rows)} pos/label {Y.sum(0).astype(int).tolist()}", flush=True)
    kw_re = [re.compile(KW[s]) for s in LABELS]; ctx_re = [re.compile(c) for c in CTX]
    t0 = time.time(); X = np.array([feats(t, kw_re, ctx_re) for t in texts], np.float32)
    print(f"features {X.shape} {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(a.out_dir, "features.npy"), X)

    splits = list(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed).split(np.zeros((len(Y), 1)), Y))
    for k, (_, va) in enumerate(splits):
        np.save(os.path.join(a.out_dir, f"fold{k}_idx.npy"), va)
    gbm = np.zeros_like(Y); lr = np.zeros_like(Y)
    for k, (tr, va) in enumerate(splits):
        t0 = time.time()
        for j in range(len(LABELS)):
            pos = Y[tr, j].sum(); spw = float(np.clip(0.5 * np.sqrt((len(tr) - pos) / max(pos, 1)), 1, 6))
            m = lgb.LGBMClassifier(n_estimators=a.rounds, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                   scale_pos_weight=spw, random_state=a.seed, verbose=-1)
            m.fit(X[tr], Y[tr, j]); gbm[va, j] = m.predict_proba(X[va])[:, 1]
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr]).astype(np.float32); Xva = vec.transform([texts[i] for i in va]).astype(np.float32)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=a.seed))
        clf.fit(Xtr, Y[tr]); lr[va] = clf.predict_proba(Xva)
        print(f"fold{k}: lgbm {macro(gbm[va], Y[va])[0]:.4f} | tfidf-lr {macro(lr[va], Y[va])[0]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(a.out_dir, "lgbm_oof.npy"), gbm); np.save(os.path.join(a.out_dir, "tfidf_lr_oof.npy"), lr)

    res = dict(lgbm=macro(gbm, Y)[0], tfidf_lr=macro(lr, Y)[0],
               lgbm_per=dict(zip(LABELS, macro(gbm, Y)[1].tolist())), lr_per=dict(zip(LABELS, macro(lr, Y)[1].tolist())))
    res["blend"] = {f"w_lgbm={w:.1f}": macro(w * gbm + (1 - w) * lr, Y)[0] for w in np.arange(0, 1.01, 0.1)}
    res["corr_err"] = float(np.corrcoef(((gbm >= .5) != (Y > .5)).ravel(), ((lr >= .5) != (Y > .5)).ravel())[0, 1])
    json.dump(res, open(os.path.join(a.out_dir, "e_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
