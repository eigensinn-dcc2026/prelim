"""dcc_inference.mission2.predict: official M2 columns, one row per utterance, same speakers as the
frozen scripts/mission2_inference.py (random-weight / tiny checkpoints, no real ckpt or data needed)."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _root in (ROOT / "src", ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from dcc_data.mission2_features import FEATURE_COLUMNS  # noqa: E402
from dcc_data.mission2_train import save_model  # noqa: E402
from dcc_inference import mission2, registry  # noqa: E402

SAMPLE_RATE = 8000
OFFICIAL_HEADER = "audio file name,startAt,endAt,speaker"
CALLS = {
    "call1": [(0, 90), (200, 1800), (1900, 3400), (3300, 5200)],
    "call2": [(0, 500), (650, 2650)],
}


def _write_wav(path: Path, duration_s: float, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.random.default_rng(seed).standard_normal(int(duration_s * SAMPLE_RATE)) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def _write_calls(tmp_path: Path, calls=CALLS, *, reverse: bool = False) -> tuple[Path, Path]:
    audio_dir, label_dir = tmp_path / "audio", tmp_path / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for seed, (stem, utterances) in enumerate(calls.items()):
        _write_wav(audio_dir / f"{stem}.wav", max(e for _, e in utterances) / 1000.0 + 0.5, seed=seed)
        ordered = list(reversed(utterances)) if reverse else list(utterances)
        document = {
            "_id": stem,
            # text/speaker present on purpose: the module must ignore both
            "utterances": [
                {"startAt": s, "endAt": e, "text": f"t{i}", "speaker": i % 2} for i, (s, e) in enumerate(ordered)
            ],
        }
        (label_dir / f"{stem}.json").write_text(json.dumps(document), encoding="utf-8")
    return audio_dir, label_dir


def _l2_metadata(**overrides) -> dict:
    from dcc_data.mission2_checkpoint import build_checkpoint_metadata
    from dcc_data.mission2_features import HOP_LENGTH, N_FFT
    from dcc_data.mission2_spectrogram import LABEL_SOURCE_PROVIDED, N_MELS

    metadata = build_checkpoint_metadata(
        model_name="large",
        model_class="SpeakerCNNLarge",
        index_provenance={
            "split": "Training", "n_mels": N_MELS, "hop_length": HOP_LENGTH, "n_fft": N_FFT,
            "sample_rate": SAMPLE_RATE, "label_source": LABEL_SOURCE_PROVIDED,
        },
        split_seed=42,
        test_size=0.2,
        torch_seed=42,
    )
    metadata.update(overrides)
    return metadata


@pytest.fixture()
def l2_ckpt(tmp_path: Path) -> Path:
    import torch

    from dcc_data.mission2_cnn import SpeakerCNNLarge

    torch.manual_seed(0)
    path = tmp_path / "l2_random.pt"
    torch.save({"best_model_state_dict": SpeakerCNNLarge().state_dict(), "metadata": _l2_metadata()}, path)
    return path


@pytest.fixture()
def l2_ensemble_ckpt(tmp_path: Path) -> Path:
    import torch

    from dcc_data.mission2_checkpoint import ENSEMBLE_STATE_DICTS_KEY, build_ensemble_metadata
    from dcc_data.mission2_cnn import SpeakerCNNLarge

    torch.manual_seed(1)
    path = tmp_path / "l2_ensemble.ckpt"  # .ckpt on purpose: torch suffix, not an L1 wrapper
    torch.save(
        {
            ENSEMBLE_STATE_DICTS_KEY: [SpeakerCNNLarge().state_dict() for _ in range(3)],
            "metadata": build_ensemble_metadata([_l2_metadata(torch_seed=s) for s in (1, 2, 3)]),
        },
        path,
    )
    return path


@pytest.fixture()
def l1_pkl(tmp_path: Path) -> Path:
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, len(FEATURE_COLUMNS)))
    y = (x[:, 0] > 0).astype(int)
    path = tmp_path / "l1.pkl"
    save_model(path, HistGradientBoostingClassifier(random_state=0).fit(x, y),
               {"feature_columns": FEATURE_COLUMNS, "majority_class": 1})
    return path


def _script_frame(audio_dir: Path, label_dir: Path, ckpt: Path, out: Path) -> pd.DataFrame:
    """Run the frozen script's main() and read its CSV back, renamed to the official header."""

    script = mission2._script()
    old_argv = sys.argv
    sys.argv = ["mission2_inference.py", "--audio_dir", str(audio_dir), "--label_dir", str(label_dir),
                "--ckpt_path", str(ckpt), "--output", str(out)]
    try:
        assert script.main() == 0
    finally:
        sys.argv = old_argv
    with out.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["audio_file_name", "startAt", "endAt", "speaker"]
    return rows[1:]


def _frame_rows(frame: pd.DataFrame, tmp_path: Path) -> list[list[str]]:
    out = tmp_path / "module.csv"
    frame.to_csv(out, index=False, encoding=registry.CSV_ENCODING)
    with out.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert ",".join(rows[0]) == OFFICIAL_HEADER
    return rows[1:]


def _check_contract(frame: pd.DataFrame, calls=CALLS) -> None:
    registry.validate_frame(2, frame)
    assert tuple(frame.columns) == registry.MISSION_COLUMNS[2] == mission2.COLUMNS
    assert len(frame) == sum(len(u) for u in calls.values())
    assert set(frame["speaker"].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(frame["speaker"])  # speaker=0/1 as int (docs/submission_format.md)
    expected = [(f"{stem}.wav", s, e) for stem in sorted(calls) for s, e in calls[stem]]
    assert list(zip(frame["audio file name"], frame["startAt"], frame["endAt"])) == expected
    # startAt/endAt are the label JSON values unchanged: same Python type too (no int -> float upcast)
    assert [(type(s), type(e)) for s, e in zip(frame["startAt"], frame["endAt"])] == [
        (type(s), type(e)) for stem in sorted(calls) for s, e in calls[stem]
    ]


@pytest.mark.parametrize("ckpt_fixture", ["l2_ckpt", "l2_ensemble_ckpt", "l1_pkl"])
def test_predict_matches_frozen_script_with_official_columns(tmp_path, request, ckpt_fixture):
    ckpt = request.getfixturevalue(ckpt_fixture)
    audio_dir, label_dir = _write_calls(tmp_path)

    frame = mission2.predict(audio_dir, label_dir, ckpt)

    _check_contract(frame)
    assert _frame_rows(frame, tmp_path) == _script_frame(audio_dir, label_dir, ckpt, tmp_path / "script.csv")


def test_l2_sets_torch_threads_to_cpu_count(tmp_path, l2_ckpt):
    import torch

    torch.set_num_threads(1)
    audio_dir, label_dir = _write_calls(tmp_path)
    mission2.predict(audio_dir, label_dir, l2_ckpt)
    assert torch.get_num_threads() == (os.cpu_count() or 1)


def test_prediction_ignores_utterance_order_text_and_labels(tmp_path, l2_ensemble_ckpt):
    forward = mission2.predict(*_write_calls(tmp_path / "fwd"), l2_ensemble_ckpt)
    backward = mission2.predict(*_write_calls(tmp_path / "bwd", reverse=True), l2_ensemble_ckpt)

    key = ["audio file name", "startAt", "endAt"]
    merged = forward.merge(backward, on=key, suffixes=("_fwd", "_bwd"))
    assert len(merged) == len(forward) == len(backward)
    assert (merged["speaker_fwd"] == merged["speaker_bwd"]).all()


def test_l1_pkl_to_ckpt_converter_gives_identical_predictions(tmp_path, l1_pkl):
    spec = importlib.util.spec_from_file_location("convert_mission2_l1_pkl", ROOT / "scripts" / "convert_mission2_l1_pkl.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    audio_dir, label_dir = _write_calls(tmp_path)
    ckpt = tmp_path / "l1.ckpt"

    assert converter.main(["--pkl", str(l1_pkl), "--output", str(ckpt), "--verify",
                           "--audio_dir", str(audio_dir), "--label_dir", str(label_dir)]) == 0

    _, _, kind = mission2.load_predictor(ckpt)
    assert kind == "l1-ckpt"
    frame_ckpt = mission2.predict(audio_dir, label_dir, ckpt)
    _check_contract(frame_ckpt)
    assert frame_ckpt.equals(mission2.predict(audio_dir, label_dir, l1_pkl))


def test_tampered_l1_ckpt_is_refused(tmp_path, l1_pkl):
    import torch

    from dcc_data.mission2_checkpoint import CheckpointConfigError

    ckpt = tmp_path / "l1.ckpt"
    mission2.save_l1_torch_ckpt(l1_pkl, ckpt)
    checkpoint = torch.load(ckpt, weights_only=True)
    checkpoint["l1_pickle_sha256"] = "0" * 64
    torch.save(checkpoint, ckpt)
    with pytest.raises(CheckpointConfigError):
        mission2.load_predictor(ckpt)


def test_default_ckpt_is_used_when_none_given(tmp_path, monkeypatch, l1_pkl):
    audio_dir, label_dir = _write_calls(tmp_path)
    assert mission2.DEFAULT_CKPT.parent == ROOT / "checkpoints" / "mission2"
    monkeypatch.setattr(mission2, "DEFAULT_CKPT", l1_pkl)
    _check_contract(mission2.predict(audio_dir, label_dir))
    monkeypatch.setattr(mission2, "DEFAULT_CKPT", tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        mission2.predict(audio_dir, label_dir)


def test_rows_sorted_by_file_name_then_label_json_order(tmp_path, l1_pkl):
    """docs/submission_format.md (M2): rows sorted by file name, then the label JSON's utterances order
    within a file; startAt/endAt written exactly as the label JSON has them."""
    # dict (write) order not sorted, utterances not chronological, fractional values: all on purpose
    calls = {"b_call": [(1900, 3400), (0, 500.5)], "a_call": [(650.25, 2650), (0, 90)]}
    audio_dir, label_dir = _write_calls(tmp_path, calls)
    frame = mission2.predict(audio_dir, label_dir, l1_pkl)
    _check_contract(frame, calls)
    expected = [[f"{stem}.wav", json.dumps(s), json.dumps(e)] for stem in sorted(calls) for s, e in calls[stem]]
    assert [row[:3] for row in _frame_rows(frame, tmp_path)] == expected


def test_unreadable_wav_keeps_rows_with_fallback(tmp_path, l1_pkl):
    audio_dir, label_dir = _write_calls(tmp_path)
    (audio_dir / "call2.wav").write_bytes(b"not a wav")
    frame = mission2.predict(audio_dir, label_dir, l1_pkl)
    _check_contract(frame)
    assert (frame.loc[frame["audio file name"] == "call2.wav", "speaker"] == 1).all()


def test_missing_label_refuses(tmp_path, l1_pkl):
    audio_dir, label_dir = _write_calls(tmp_path)
    (label_dir / "call2.json").unlink()
    with pytest.raises(ValueError, match="call2"):
        mission2.predict(audio_dir, label_dir, l1_pkl)


def test_root_entry_writes_official_mission2_csv(tmp_path, l2_ckpt):
    spec = importlib.util.spec_from_file_location("dcc_root_inference_m2", ROOT / "inference.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    audio_dir, label_dir = _write_calls(tmp_path)
    out = tmp_path / "outputs" / "mission2.csv"

    assert entry.main(["--audio_dir", str(audio_dir), "--label_dir", str(label_dir),
                       "--ckpt_path", str(l2_ckpt), "--output", str(out)]) == 0

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8").splitlines()
    assert lines[0] == OFFICIAL_HEADER
    assert len(lines) == 1 + sum(len(u) for u in CALLS.values())
    assert all(line.rsplit(",", 1)[1] in {"0", "1"} for line in lines[1:])
    # file name order, then label JSON utterances order; startAt/endAt text unchanged
    expected = [f"{stem}.wav,{s},{e}" for stem in sorted(CALLS) for s, e in CALLS[stem]]
    assert [line.rsplit(",", 1)[0] for line in lines[1:]] == expected
