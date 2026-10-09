"""Mission 3 BERT path: lab best.ckpt -> export -> offline predict(). Tiny random BERT only."""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcc_inference import mission3, mission3_bert, registry  # noqa: E402

LABELS = list(mission3_bert.LABELS)
CHARS = list("가나다라마바사아자차카타파하열배머리숨 0123456789.,?")


def _model_dir(root: Path) -> Path:
    folder = root / "tiny-bert"
    folder.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + CHARS
    (folder / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": False, "model_max_length": 300}), encoding="utf-8")
    transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=300,
    ).save_pretrained(folder)
    (folder / "model.safetensors").write_bytes(b"not embedded")  # weights are skipped
    (folder / "dcc_revision.json").write_text("{}", encoding="utf-8")
    return folder


def _lab_ckpt(root: Path, model_dir: Path, bias: list[float] | None = None) -> tuple[Path, object]:
    config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
    torch.manual_seed(0)
    model = mission3_bert._classifier(config, len(LABELS))
    if bias is not None:
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor(bias))
    path = root / "best.ckpt"
    lab_config = dict(exp_id="M3-TEST", model="tiny", revision="0", max_length=256, dropout=0.1,
                      labels=LABELS, threshold=0.5)
    torch.save(dict(config=lab_config, seed=42, epoch=3, model=model.state_dict(),
                    model_local_path=str(model_dir), pos_weight=[1.0] * 9), path)
    return path, model.eval()


def _labels(root: Path) -> Path:
    folder = root / "labels"
    (folder / "nested").mkdir(parents=True)
    docs = {
        "b.json": {"symptom": ["고열"], "utterances": [{"text": "열이  나요\n머리"}, {"text": 3}, "x"]},
        "nested/a.json": {"utterances": [{"text": "숨 차요"}], "address": "not read"},
        "c.json": {"utterances": []},
    }
    for name, doc in docs.items():
        raw = json.dumps(doc, ensure_ascii=False).encode("utf-8")
        (folder / name).write_bytes(b"\xef\xbb\xbf" + raw if name == "b.json" else raw)
    return folder


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access during Mission 3 inference")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_transcript_matches_the_lab_training_rule():
    spec = importlib.util.spec_from_file_location("m3_kcbert_005", ROOT / "scripts" / "lab" / "m3-kcbert-005.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    docs = [
        {"utterances": [{"text": " 열이\t\n나요 "}, {"text": ""}, {"text": None}, 5, {"speaker": 1}]},
        {"utterances": [{"text": "a  b"}, {"text": "c"}]},
        {"utterances": []},
        {},
    ]
    for doc in docs:
        assert mission3_bert.transcript(doc) == lab.transcript(doc)
    assert mission3_bert.transcript({"utterances": "bad"}) == ""
    assert mission3_bert.transcript([]) == ""


def test_export_embeds_text_files_and_matches_the_lab_model(tmp_path: Path, no_network):
    model_dir = _model_dir(tmp_path)
    lab_ckpt, lab_model = _lab_ckpt(tmp_path, model_dir)
    out = tmp_path / "m3.pt"
    meta = mission3_bert.export_checkpoint(lab_ckpt, model_dir, out)
    assert meta["source"]["seed"] == 42 and len(meta["source"]["lab_ckpt_sha256"]) == 64

    payload = torch.load(out, map_location="cpu", weights_only=True)
    assert payload["format"] == mission3_bert.FORMAT
    assert set(payload["files"]) == {"config.json", "tokenizer_config.json", "vocab.txt"}

    model, tokenizer, labels, max_length = mission3_bert.load_checkpoint(out, device="cpu")
    assert labels == LABELS and max_length == 256
    texts = ["열이 나요", "숨 차요 " * 200, ""]
    got = mission3_bert.probabilities(model, tokenizer, texts, max_length, batch_size=2)
    reference_tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # The lab tokenizes with padding="max_length"; batch padding must give the same scores.
    encoded = reference_tokenizer(texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        expected = lab_model(encoded["input_ids"], encoded["attention_mask"]).sigmoid().numpy()
    assert got.shape == (3, 9)
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_export_rejects_mismatched_labels_and_lengths(tmp_path: Path):
    model_dir = _model_dir(tmp_path)
    lab_ckpt, _ = _lab_ckpt(tmp_path, model_dir)
    state = torch.load(lab_ckpt, weights_only=False)
    for key, value in (("labels", LABELS[::-1]), ("max_length", 512)):
        broken = dict(state, config=dict(state["config"], **{key: value}))
        torch.save(broken, tmp_path / "broken.ckpt")
        with pytest.raises(ValueError):
            mission3_bert.export_checkpoint(tmp_path / "broken.ckpt", model_dir, tmp_path / "x.pt")


def test_load_rejects_a_raw_lab_checkpoint(tmp_path: Path):
    model_dir = _model_dir(tmp_path)
    lab_ckpt, _ = _lab_ckpt(tmp_path, model_dir)
    with pytest.raises(Exception):
        mission3_bert.load_checkpoint(lab_ckpt)


def test_predict_with_bert_checkpoint_uses_ge_half_and_allows_empty(tmp_path: Path, no_network):
    model_dir = _model_dir(tmp_path)
    # 고열 logit 0 -> probability exactly 0.5 (positive); 두통 +3; the rest -3.
    bias = [-3.0] * 9
    bias[LABELS.index("고열")] = 0.0
    bias[LABELS.index("두통")] = 3.0
    lab_ckpt, _ = _lab_ckpt(tmp_path, model_dir, bias=bias)
    out = tmp_path / "m3.pt"
    mission3_bert.export_checkpoint(lab_ckpt, model_dir, out)
    labels = _labels(tmp_path)

    frame = mission3.predict(tmp_path / "audio", labels, out)
    registry.validate_frame(3, frame)
    assert list(frame["label file name"]) == ["b.json", "c.json", "a.json"]  # sorted path order
    assert set(frame["symptom"]) == {"['고열', '두통']"}

    lab_ckpt, _ = _lab_ckpt(tmp_path, model_dir, bias=[-3.0] * 9)
    mission3_bert.export_checkpoint(lab_ckpt, model_dir, out)
    frame = mission3.predict(tmp_path / "audio", labels, out)
    assert list(frame["symptom"]) == ["[]", "[]", "[]"]  # no min-1 fill (organizer ruling 2026-10-09)


def test_export_cli_and_score_cli(tmp_path: Path):
    model_dir = _model_dir(tmp_path)
    bias = [-3.0] * 9
    bias[LABELS.index("고열")] = 3.0
    lab_ckpt, _ = _lab_ckpt(tmp_path, model_dir, bias=bias)
    out = tmp_path / "ckpt" / "m3.pt"
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "mission3_export_bert_ckpt.py"),
                          "--lab_ckpt", str(lab_ckpt), "--model_dir", str(model_dir), "--output", str(out)],
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    record = json.loads((tmp_path / "ckpt" / "m3.pt.json").read_text(encoding="utf-8"))
    assert record["output_bytes"] == out.stat().st_size and len(record["output_sha256"]) == 64

    labels = _labels(tmp_path)
    csv_path = tmp_path / "outputs" / "mission3.csv"
    run = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--audio_dir", str(tmp_path / "audio"),
                          "--label_dir", str(labels), "--ckpt_path", str(out), "--output", str(csv_path)],
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    read = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
    assert [ast.literal_eval(cell) for cell in read["symptom"]] == [["고열"]] * 3

    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "mission3_score_csv.py"),
                          "--csv", str(csv_path), "--label_dir", str(labels)],
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    score = json.loads(run.stdout)
    # Only b.json has gold 고열 -> 고열 F1 = 2*1/(3+1) = 0.5; others have no gold and no prediction.
    assert score["rows"] == 3 and score["empty_predictions"] == 0
    assert score["per_symptom_f1"]["고열"] == pytest.approx(0.5)
    assert score["macro_f1"] == pytest.approx(0.5 / 9)
