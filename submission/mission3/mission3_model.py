"""Mission 3 모델 코드: 오프라인 체크포인트 하나에서 BERT 계열 다중 라벨 분류기를 읽어 쓴다.

체크포인트 형식은 두 가지이고, 둘 다 ``torch.load(weights_only=True)``로 읽는 ``.pt`` 하나이다.

- ``dcc-m3-bert-v1``: 인코더 하나(KcBERT / RoBERTa)와 선형 헤드. config.json / 토크나이저 파일의
  내용이 함께 들어 있어 추론할 때 내려받는 것이 없다.
- ``dcc-m3-ens-v1``: 위와 같은 인코더 여러 개와 TF-IDF + 로지스틱 회귀 모델 하나의 확률을 고정 가중치로
  평균한 것(어휘, idf, 계수는 일반 배열로 저장). ``p = sum_k w_k * p_k``이고 가중치는 만들 때 정해 둔다
  (Training 내부 홀드아웃에서 정했고, 공식 Validation으로는 정하지 않았다). 판정은 똑같이 ``p >= 0.5``이다.

입력은 통화 전사문(``utterances[].text``)뿐이다.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FORMAT = "dcc-m3-bert-v1"
ENSEMBLE_FORMAT = "dcc-m3-ens-v1"
# 평가 대상 증상 9개. 출력 셀에는 이 순서대로 적는다.
LABELS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
MAX_LENGTH = 256
THRESHOLD = 0.5  # 주최 측이 정한 값이다. 증상별 조정이나 후처리는 하지 않는다.


def transcript(doc: Any) -> str:
    """``utterances[].text``를 공백 한 칸으로 이어 붙인다(연속 공백은 하나로 줄인다)."""

    if not isinstance(doc, dict) or not isinstance(doc.get("utterances"), list):
        return ""
    texts = []
    for utterance in doc["utterances"]:
        text = utterance.get("text", "") if isinstance(utterance, dict) else ""
        text = re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""
        if text:
            texts.append(text)
    return " ".join(texts)


def read_text(json_path: str | Path) -> str:
    with Path(json_path).open("r", encoding="utf-8-sig") as handle:
        return transcript(json.load(handle))


def _build(files: dict[str, str], dropout: float, n_labels: int):
    """BERT 인코더 + 드롭아웃 + 첫 토큰 위의 선형 헤드."""

    import torch
    from transformers import AutoConfig, AutoModel

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "config.json").write_text(files["config.json"], encoding="utf-8")
        config = AutoConfig.from_pretrained(folder, local_files_only=True)

    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bert = AutoModel.from_config(config)
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(config.hidden_size, n_labels)

        def forward(self, ids, mask):
            hidden = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            return self.classifier(self.dropout(hidden))

    return Classifier()


def load_checkpoint(path: str | Path, device: str | None = None, payload: dict | None = None):
    """``(모델, 토크나이저, 라벨, max_length)``를 완전히 오프라인으로 만든다."""

    import torch
    from transformers import AutoTokenizer

    if payload is None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    labels = tuple(str(label) for label in payload["labels"])
    if labels != LABELS:
        raise ValueError(f"checkpoint labels {labels} != {LABELS}")
    files = payload["files"]
    with tempfile.TemporaryDirectory() as folder:
        for name, text in files.items():
            (Path(folder) / Path(name).name).write_text(text, encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = _build(files, float(payload.get("dropout", 0.1)), len(labels))
    model.load_state_dict(payload["model"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, list(labels), int(payload["max_length"])


def probabilities(model, tokenizer, texts: Sequence[str], max_length: int = MAX_LENGTH,
                  batch_size: int = 32) -> np.ndarray:
    """입력 순서대로 시그모이드 확률 ``(len(texts), 9)``를 돌려준다."""

    import torch

    device = next(model.parameters()).device
    # 길이순으로 정렬해 패딩된 배치를 짧게 유지하고, 결과는 입력 순서로 되돌려 놓는다.
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            encoded = tokenizer([texts[index] for index in chunk], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            out[chunk] = logits.float().sigmoid().cpu().numpy()
    return out


def symptom_cell(binary_row: Sequence[int]) -> str:
    """``str(list)`` of the positive labels in LABELS order, e.g. ``['두통', '복통']``."""

    return str([label for label, flag in zip(LABELS, binary_row) if int(flag) == 1])


def tfidf_probabilities(tfidf: dict, texts: Sequence[str]) -> np.ndarray:
    """TF-IDF(sublinear tf, smooth idf, l2) + 증상별 로지스틱 회귀를 일반 배열만으로 계산한다.

        저장된 파라미터와 어휘로 sklearn ``TfidfVectorizer``와 같은 특징을 만들고, 확률은
        ``OneVsRestClassifier(LogisticRegression).predict_proba``(다중 라벨)와 같다.
    """

    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    params = dict(tfidf["params"])
    params["ngram_range"] = tuple(params["ngram_range"])
    analyzer = TfidfVectorizer(**params).build_analyzer()  # 토큰화만 한다. 학습(fit)은 하지 않는다.
    index = {term: i for i, term in enumerate(tfidf["vocabulary"])}
    rows, cols, vals = [], [], []
    for r, text in enumerate(texts):
        counts: dict[int, int] = {}
        for token in analyzer(text):
            j = index.get(token)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            rows.append(r); cols.append(j); vals.append(c)
    x = csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(len(texts), len(index)))
    if params.get("sublinear_tf"):
        x.data = np.log(x.data) + 1.0
    x = x.multiply(np.asarray(tfidf["idf"], dtype=np.float64)).tocsr()
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    x = csr_matrix(x.multiply(1.0 / norms[:, None]))
    logits = x @ np.asarray(tfidf["coef"], dtype=np.float64).T + np.asarray(tfidf["intercept"], dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def predict_probabilities(path: str | Path, texts: Sequence[str], batch_size: int = 32,
                          device: str | None = None) -> np.ndarray:
    """두 체크포인트 형식 모두에 대해 ``(len(texts), 9)`` 확률을 돌려준다(멤버는 한 번에 하나씩 실행)."""

    import gc
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    fmt = payload.get("format") if isinstance(payload, dict) else None
    if fmt == FORMAT:
        model, tokenizer, _, max_length = load_checkpoint(path, device, payload)
        return probabilities(model, tokenizer, texts, max_length, batch_size)
    if fmt != ENSEMBLE_FORMAT:
        raise ValueError(f"{path} is neither {FORMAT} nor {ENSEMBLE_FORMAT}")
    if tuple(str(label) for label in payload["labels"]) != LABELS:
        raise ValueError("ensemble labels differ from the 9 Mission 3 labels")
    total = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
    weight_sum = 0.0
    for member in payload["members"]:
        w = float(member["weight"])
        if member["kind"] == "bert":
            model, tokenizer, _, max_length = load_checkpoint(path, device, dict(member["checkpoint"]))
            total += w * probabilities(model, tokenizer, texts, max_length, batch_size)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif member["kind"] == "tfidf":
            total += w * tfidf_probabilities(member, texts)
        else:
            raise ValueError(f"unknown ensemble member kind {member['kind']!r}")
        weight_sum += w
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"ensemble weights sum to {weight_sum}, not 1")
    return total.astype(np.float32)
