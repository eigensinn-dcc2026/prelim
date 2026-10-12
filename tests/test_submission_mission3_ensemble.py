"""submission/mission3 ensemble checkpoint (dcc-m3-ens-v1): build, safe load, exact weighted average, TF-IDF parity."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
joblib = pytest.importorskip("joblib")

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "submission" / "mission3", ROOT / "scripts" / "m3_final"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import build_ensemble  # noqa: E402
import mission3_model  # noqa: E402
from dcc_inference import mission3_bert  # noqa: E402

LABELS = list(mission3_bert.LABELS)
TEXTS = ["열이 나요 머리 아파요", "숨이 차요 가슴 답답", "배가 아파요 토했어요", "다리 찢어져서 피 나요",
         "어지러워요 쓰러질 것 같아요", "기운이 없어요 열이 나요", "토할 것 같아요 속이 울렁", "머리 아파요 어지러워요"]


def _bert_pt(tmp_path: Path, name: str, seed: int) -> Path:
    folder = tmp_path / "tiny-bert"
    if not folder.exists():
        folder.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set("".join(TEXTS).replace(" ", "")))
        (folder / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        (folder / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": False, "model_max_length": 300}), encoding="utf-8")
        transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=32, max_position_embeddings=300).save_pretrained(folder)
    config = transformers.AutoConfig.from_pretrained(folder, local_files_only=True)
    torch.manual_seed(seed)
    model = mission3_bert._classifier(config, len(LABELS))
    lab = tmp_path / f"{name}.ckpt"
    torch.save(dict(config=dict(labels=LABELS, max_length=256, dropout=0.1, exp_id="T"), seed=seed, epoch=3,
                    model=model.state_dict()), lab)
    out = tmp_path / f"{name}.pt"
    mission3_bert.export_checkpoint(lab, folder, out)
    return out


def _tfidf_joblib(tmp_path: Path) -> Path:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    y = np.array([[(i + j) % 2 for j in range(len(LABELS))] for i in range(len(TEXTS))], dtype=int)
    assert (y.min(0) == 0).all() and (y.max(0) == 1).all()  # every label has both classes
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42))
    clf.fit(vec.fit_transform(TEXTS), y)
    path = tmp_path / "tfidf.joblib"
    joblib.dump(dict(vectorizer=vec, classifier=clf, target_symptoms=LABELS), path)
    return path


def test_tfidf_array_form_matches_sklearn(tmp_path: Path):
    path = _tfidf_joblib(tmp_path)
    ck = joblib.load(path)
    member = build_ensemble.tfidf_member(path, 1.0)
    unseen = TEXTS + ["처음 보는 단어만 있어요", ""]
    np.testing.assert_allclose(mission3_model.tfidf_probabilities(member, unseen),
                               ck["classifier"].predict_proba(ck["vectorizer"].transform(unseen)), atol=1e-6)


def test_ensemble_is_the_fixed_weighted_average_and_loads_safely(tmp_path: Path):
    bert = [_bert_pt(tmp_path, f"s{s}", s) for s in (42, 43)]
    tfidf = _tfidf_joblib(tmp_path)
    ens = tmp_path / "ens.pt"
    build_ensemble.main(["--bert", *map(str, bert), "--bert_weight", "0.55", "--tfidf", str(tfidf),
                         "--tfidf_weight", "0.45", "--out", str(ens)])
    assert torch.load(ens, map_location="cpu", weights_only=True)["format"] == "dcc-m3-ens-v1"
    got = mission3_model.predict_probabilities(ens, TEXTS, 4, device="cpu")
    singles = [mission3_model.predict_probabilities(p, TEXTS, 4, device="cpu") for p in bert]
    ck = joblib.load(tfidf)
    expected = 0.55 * (singles[0] + singles[1]) / 2 + 0.45 * ck["classifier"].predict_proba(ck["vectorizer"].transform(TEXTS))
    np.testing.assert_allclose(got, expected, atol=1e-5)
    record = json.loads((tmp_path / "ens.pt.json").read_text(encoding="utf-8"))
    assert [m["weight"] for m in record["members"]] == [0.275, 0.275, 0.45]


def test_weights_must_sum_to_one(tmp_path: Path):
    bert = [_bert_pt(tmp_path, "s42", 42)]
    with pytest.raises(SystemExit):
        build_ensemble.main(["--bert", str(bert[0]), "--bert_weight", "0.5", "--out", str(tmp_path / "x.pt")])


def test_submission_inference_runs_on_the_ensemble(tmp_path: Path):
    bert = [_bert_pt(tmp_path, "s42", 42)]
    ens = tmp_path / "ens.pt"
    build_ensemble.main(["--bert", str(bert[0]), "--bert_weight", "0.55", "--tfidf", str(_tfidf_joblib(tmp_path)),
                         "--tfidf_weight", "0.45", "--out", str(ens)])
    labels = tmp_path / "labels"
    labels.mkdir()
    for i, t in enumerate(TEXTS[:3]):
        (labels / f"c{i}.json").write_text(json.dumps({"utterances": [{"text": t}]}, ensure_ascii=False), encoding="utf-8")
    csv = tmp_path / "out" / "mission3.csv"
    run = subprocess.run([sys.executable, str(ROOT / "submission" / "mission3" / "inference.py"), "--audio_dir", str(tmp_path),
                          "--label_dir", str(labels), "--ckpt_path", str(ens), "--output", str(csv)],
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    frame = pd.read_csv(csv, encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(frame.columns) == ["label file name", "symptom"] and len(frame) == 3
    for cell in frame["symptom"]:
        assert set(ast.literal_eval(cell)) <= set(LABELS)
