"""Mission 3 BERT 계열 분류기: 자급자족 체크포인트, 내보내기, 확률 계산.

lab 실험(``scripts/lab/m3-kcbert-005.py`` 등)은 ``best.ckpt`` =
``{"config", "seed", "epoch", "model": state_dict, ...}``를 저장하고, 이 state dict는 아래
``Classifier``(``bert`` 인코더 + 드롭아웃 + 첫 토큰 위의 Linear)의 것이다. 그 파일들은 옆에
Hugging Face 모델 폴더가 있어야 한다. :func:`export_checkpoint`는 그 폴더의 설정·토크나이저 파일 내용을
``.pt`` 하나에 합쳐, 그 파일 하나로 오프라인 추론이 되게 한다. :func:`load_checkpoint`와
:func:`probabilities`가 이를 사용한다.

입력 텍스트는 :func:`transcript`이며, lab 스크립트가 학습할 때 쓴 규칙과 같다:
``utterances[].text``만 쓰고, 연속 공백은 하나로 줄이고, 공백 한 칸으로 이어 붙인다.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FORMAT = "dcc-m3-bert-v1"
LABELS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
MAX_LENGTH = 256
# 가중치는 state dict에 들어 있으므로, 이 파일들은 내보내기에 복사하지 않는다.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack")
_SKIP_FILES = {"dcc_revision.json"}


def transcript(doc: Any) -> str:
    """lab 스크립트의 학습용 텍스트 규칙. JSON의 다른 필드는 절대 읽지 않는다."""

    if not isinstance(doc, dict):
        return ""
    utterances = doc.get("utterances", [])
    if not isinstance(utterances, list):
        return ""
    texts = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        text = utterance.get("text", "")
        text = re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""
        if text:
            texts.append(text)
    return " ".join(texts)


def read_text(json_path: str | Path) -> str:
    with Path(json_path).open("r", encoding="utf-8-sig") as handle:
        return transcript(json.load(handle))


def _classifier(config: Any, n_labels: int, dropout: float = 0.1):
    import torch
    from transformers import AutoModel

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


def _text_files(model_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(model_dir.iterdir()):
        if not path.is_file() or path.name in _SKIP_FILES or path.suffix in _WEIGHT_SUFFIXES:
            continue
        try:
            files[path.name] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"tokenizer/config file is not text, cannot embed: {path}") from error
    if "config.json" not in files:
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    return files


def export_checkpoint(lab_ckpt: str | Path, model_dir: str | Path, output: str | Path) -> dict:
    """자급자족 추론 체크포인트를 쓰고 그 메타데이터(텐서 제외)를 돌려준다."""

    import torch

    lab_ckpt, model_dir, output = Path(lab_ckpt), Path(model_dir), Path(output)
    source = torch.load(lab_ckpt, map_location="cpu", weights_only=False)
    config = source.get("config", {})
    labels = tuple(config.get("labels", ()))
    if labels != LABELS:
        raise ValueError(f"checkpoint labels {labels} != Mission 3 labels {LABELS}")
    if int(config.get("max_length", -1)) != MAX_LENGTH:
        raise ValueError(f"checkpoint max_length {config.get('max_length')} != {MAX_LENGTH}")
    files = _text_files(model_dir)
    digest = hashlib.sha256()
    with lab_ckpt.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    meta = dict(
        format=FORMAT,
        labels=list(LABELS),
        max_length=MAX_LENGTH,
        dropout=float(config.get("dropout", 0.1)),
        source=dict(exp_id=config.get("exp_id"), model=config.get("model"),
                    revision=config.get("revision"), seed=source.get("seed"),
                    epoch=source.get("epoch"), lab_ckpt_sha256=digest.hexdigest()),
    )
    state = {key: value.contiguous() for key, value in source["model"].items()}
    # state dict가 분류기와 맞지 않으면 제출 시점이 아니라 지금 바로 실패시킨다.
    _build(files, meta).load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta, files=files, model=state), output)
    return meta


def _build(files: dict[str, str], meta: dict):
    from transformers import AutoConfig

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "config.json").write_text(files["config.json"], encoding="utf-8")
        config = AutoConfig.from_pretrained(folder, local_files_only=True)
    return _classifier(config, len(meta["labels"]), meta.get("dropout", 0.1))


def is_bert_checkpoint(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"}


def load_checkpoint(path: str | Path, device: str | None = None):
    """내보낸 체크포인트에서 오프라인으로 ``(모델, 토크나이저, 라벨, max_length)``를 만든다."""

    import torch
    from transformers import AutoTokenizer

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a {FORMAT} checkpoint; export the lab best.ckpt with "
            "scripts/mission3_export_bert_ckpt.py"
        )
    labels = [str(label) for label in payload["labels"]]
    if tuple(labels) != LABELS:
        raise ValueError(f"checkpoint labels {labels} != Mission 3 labels {list(LABELS)}")
    files = payload["files"]
    with tempfile.TemporaryDirectory() as folder:
        for name, text in files.items():
            (Path(folder) / Path(name).name).write_text(text, encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = _build(files, payload)
    model.load_state_dict(payload["model"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, labels, int(payload["max_length"])


def probabilities(model, tokenizer, texts: Sequence[str], max_length: int = MAX_LENGTH,
                  batch_size: int = 32) -> np.ndarray:
    """입력 순서대로 시그모이드 확률 ``(len(texts), n_labels)``(fp32)를 돌려준다."""

    import torch

    device = next(model.parameters()).device
    # 길이순으로 정렬해 패딩된 배치마다 길이를 짧게 하고, 결과는 입력 순서로 되돌린다.
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    out = np.zeros((len(texts), model.classifier.out_features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            encoded = tokenizer([texts[index] for index in chunk], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            out[chunk] = logits.float().sigmoid().cpu().numpy()
    return out
