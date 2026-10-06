"""dcc_inference.mission3.predict: official M3 columns, one row per label JSON, P0 text,
threshold >= 0.5, min-1. Synthetic joblib only — no real checkpoint or data."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcc_inference import mission3, registry  # noqa: E402

SYMPTOMS = [
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
]


def test_apply_min1_cases():
    probabilities = np.array(
        [
            [0.1, 0.4, 0.2],  # none >= 0.5 -> argmax (index 1)
            [0.2, 0.2, 0.1],  # tie -> first index
            [0.3, 0.3, 0.3],  # all-way tie -> first index
            [0.9, 0.1, 0.7],  # already positive; argmax must not clear the others
            [0.1, 0.95, 0.6],  # already positive; keep both, including the lower one
            [0.5, 0.49, 0.0],  # exact 0.5 counts as positive
        ],
        dtype=float,
    )
    original = probabilities.copy()
    binary = mission3.apply_min1(probabilities, threshold=0.5)
    assert binary.tolist() == [
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
    ]
    assert np.array_equal(probabilities, original)


def test_format_symptom_cell_matches_p0_list_repr():
    """Cell text is P0 ``_labels_to_string``. UNVERIFIED vs official PDF p.11."""

    script = mission3._script()
    labels = ["두통", "복통"]
    assert mission3.format_symptom_cell(labels) == "['두통', '복통']"
    assert mission3.format_symptom_cell(labels) == mission3.SYMPTOM_CELL_EXAMPLE
    assert mission3.format_symptom_cell(labels) == script._labels_to_string(labels)
    assert mission3.format_symptom_cell(["고열"]) == "['고열']"
    assert mission3.format_symptom_cell([]) == "[]"
    # The formatter does not reorder; predict() passes TARGET_SYMPTOMS order.
    assert mission3.format_symptom_cell(["복통", "두통"]) == "['복통', '두통']"
    assert list(script.TARGET_SYMPTOMS) == SYMPTOMS
    assert mission3.positive_labels([0, 0, 1, 1, 0, 0, 0, 0, 0], SYMPTOMS) == ["두통", "복통"]


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _write_checkpoint(path: Path) -> None:
    """P0's joblib dict, from a tiny fitted TF-IDF + OvR logistic pipeline.

    Coefficients are then set so each fixture transcript has known probabilities.
    The stored ``threshold`` is 0.9 on purpose: predict() must still use 0.5.
    """

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    docs = ["zz", "half", "head belly", "tie"]
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=200))),
        ]
    )
    vectorizer = pipe.named_steps["tfidf"]
    classifier = pipe.named_steps["clf"]
    features = vectorizer.fit_transform(docs)
    both = np.vstack([np.ones((len(docs), len(SYMPTOMS))), np.zeros((len(docs), len(SYMPTOMS)))])
    classifier.fit(np.vstack([features.toarray(), features.toarray()]), both)

    vocab = vectorizer.vocabulary_
    for estimator in classifier.estimators_:
        estimator.coef_[:] = 0
        estimator.intercept_[:] = _logit(0.1)
    # "zz": every label 0.1 except 두통 at 0.4 -> min-1 turns 두통 on.
    classifier.estimators_[2].intercept_[:] = _logit(0.4)
    # "half": 고열 exactly 0.5 (counts) and 두통 at 0.8 (already positive).
    fever = classifier.estimators_[0]
    fever.coef_[0, vocab["half"]] = -float(fever.intercept_[0])
    headache = classifier.estimators_[2]
    headache.coef_[0, vocab["half"]] = _logit(0.8) - float(headache.intercept_[0])
    # "head belly": 두통 and 복통 both above 0.5. JSON gold below is a different label.
    headache.coef_[0, vocab["head"]] = 5.0
    classifier.estimators_[3].coef_[0, vocab["belly"]] = 5.0
    # "tie": every label 0.2 -> min-1 keeps the first symptom, 고열.
    for estimator in classifier.estimators_:
        estimator.coef_[0, vocab["tie"]] = _logit(0.2) - float(estimator.intercept_[0])

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "target_symptoms": list(SYMPTOMS),
            "threshold": 0.9,
            "model_name": "synthetic-tfidf-ovr-logreg",
        },
        path,
    )


def _write_labels(label_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "nested").mkdir()
    documents = {
        # BOM + P0 cleaning: newline inside text becomes a space, blanks are dropped.
        "nested/zz.json": {
            "symptom": ["호흡곤란"],
            "utterances": [{"text": "zz"}, {"text": "   "}, "nope", {"text": 3}],
        },
        "half.json": {
            "symptom": ["구토"],
            "utterances": [{"speaker": 0, "text": "half", "startAt": 10}],
        },
        "pair.json": {
            "symptom": ["호흡곤란", "기타"],
            "utterances": [{"text": "head\nbelly"}, {"text": "   "}],
        },
        "tie.json": {
            "utterances": [{"text": "tie"}],
        },
    }
    for name, document in documents.items():
        path = label_dir / name
        raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
        if name.endswith("zz.json"):
            raw = b"\xef\xbb\xbf" + raw
        path.write_bytes(raw)


def test_predict_contract_and_symptom_cells(tmp_path: Path):
    ckpt = tmp_path / "mission3_tfidf_logreg.joblib"
    _write_checkpoint(ckpt)
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "labels"
    _write_labels(label_dir)

    frame = mission3.predict(audio_dir, label_dir, ckpt)

    registry.validate_frame(3, frame)
    assert tuple(frame.columns) == registry.MISSION_COLUMNS[3] == mission3.COLUMNS
    assert list(frame.columns) == ["label file name", "symptom"]
    expected_names = [path.name for path in sorted(label_dir.rglob("*.json"))]
    # Path order, not basename order: nested/zz.json sorts between half.json and pair.json.
    assert expected_names == ["half.json", "zz.json", "pair.json", "tie.json"]
    assert list(frame["label file name"]) == expected_names
    assert len(frame) == 4

    by_name = dict(zip(frame["label file name"], frame["symptom"]))
    # exact 0.5 is positive, and the higher 두통 stays on too (not collapsed to argmax)
    assert by_name["half.json"] == "['고열', '두통']"
    # TARGET_SYMPTOMS order, comma+space, single quotes — not the JSON's 호흡곤란/기타
    assert by_name["pair.json"] == mission3.SYMPTOM_CELL_EXAMPLE == "['두통', '복통']"
    # all below 0.5 and tied -> first label
    assert by_name["tie.json"] == "['고열']"
    # all below 0.5 -> the single argmax label; utf-8-sig still read
    assert by_name["zz.json"] == "['두통']"
    assert set(frame["symptom"]) == {
        mission3.format_symptom_cell(labels)
        for labels in (["고열", "두통"], ["두통", "복통"], ["고열"], ["두통"])
    }
    assert not (tmp_path / "outputs").exists()


def test_default_ckpt_is_used_when_none_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ckpt = tmp_path / "mission3_tfidf_logreg.joblib"
    _write_checkpoint(ckpt)
    label_dir = tmp_path / "labels"
    _write_labels(label_dir)
    assert mission3.DEFAULT_CKPT.parent == ROOT / "checkpoints" / "mission3"
    assert mission3.DEFAULT_CKPT.name == "mission3_tfidf_logreg.joblib"
    monkeypatch.setattr(mission3, "DEFAULT_CKPT", ckpt)
    frame = mission3.predict(tmp_path / "audio", label_dir)
    assert len(frame) == 4
    monkeypatch.setattr(mission3, "DEFAULT_CKPT", tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError):
        mission3.predict(tmp_path / "audio", label_dir)


def test_root_entry_writes_official_mission3_csv(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("dcc_root_inference_m3", ROOT / "inference.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    ckpt = tmp_path / "m3.joblib"
    _write_checkpoint(ckpt)
    label_dir = tmp_path / "labels"
    _write_labels(label_dir)
    out = tmp_path / "outputs" / "mission3.csv"

    assert entry.main(
        [
            "--audio_dir",
            str(tmp_path / "audio"),
            "--label_dir",
            str(label_dir),
            "--ckpt_path",
            str(ckpt),
            "--output",
            str(out),
        ]
    ) == 0

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    with out.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["label file name", "symptom"]
    assert rows[1:] == [
        ["half.json", "['고열', '두통']"],
        ["zz.json", "['두통']"],
        ["pair.json", "['두통', '복통']"],
        ["tie.json", "['고열']"],
    ]
    # The comma inside the cell is CSV-quoted; the cell text itself keeps the quotes and space.
    assert "pair.json,\"['두통', '복통']\"" in raw.decode("utf-8")
