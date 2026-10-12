"""Smoke tests for WavLM windowing and the CNN+WavLM packed checkpoint.

These use a synthetic WAV and a tiny randomly initialized WavLM so the pack/load
round trip does not download weights or read competition audio.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

mission2_inference = importlib.import_module("mission2_inference")
pack_mission2_ensemble = importlib.import_module("pack_mission2_ensemble")

from dcc_data.mission2_checkpoint import ENSEMBLE_STATE_DICTS_KEY, build_ensemble_metadata  # noqa: E402
from dcc_data.mission2_cnn import SpeakerCNNLarge  # noqa: E402
from dcc_data.mission2_wavlm import (  # noqa: E402
    SR,
    WIN,
    fit_length,
    read_utterance_signal,
    window_starts,
)

SAMPLE_RATE = 8000


def _l2_metadata(torch_seed: int) -> dict:
    from dcc_data.mission2_checkpoint import build_checkpoint_metadata
    from dcc_data.mission2_features import HOP_LENGTH, N_FFT
    from dcc_data.mission2_spectrogram import LABEL_SOURCE_PROVIDED, N_MELS

    return build_checkpoint_metadata(
        model_name="large",
        model_class="SpeakerCNNLarge",
        index_provenance={
            "split": "Training",
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "sample_rate": SAMPLE_RATE,
            "label_source": LABEL_SOURCE_PROVIDED,
        },
        split_seed=42,
        test_size=0.2,
        torch_seed=torch_seed,
    )


def _write_wav(path: Path, duration_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(duration_s * SAMPLE_RATE))
    samples = np.full(n, 1600, dtype="<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def _tiny_wavlm_config() -> dict:
    from transformers import WavLMConfig

    config = WavLMConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        conv_dim=[8, 8, 8],
        conv_stride=[5, 2, 2],
        conv_kernel=[10, 3, 3],
        num_feat_extract_layers=3,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        layerdrop=0.1,
    )
    return config.to_dict()


def test_window_starts_and_fit_length_on_a_synthetic_wav(tmp_path: Path) -> None:
    assert window_starts(1) == [0]
    assert window_starts(WIN) == [0]
    assert window_starts(WIN + 1) == [0, 1]
    long_length = SR * 100
    starts = window_starts(long_length)
    assert len(starts) == 8
    assert starts[0] == 0
    assert starts[-1] == long_length - WIN

    wav_path = tmp_path / "call.wav"
    _write_wav(wav_path, 0.25)  # 2000 samples at 8 kHz -> 4000 samples at 16 kHz
    signal = read_utterance_signal(wav_path, 0, 250)
    assert signal.shape == (4000,)
    assert torch.isfinite(signal).all()

    window = fit_length(signal, start=0)
    assert window.shape == (WIN,)
    assert torch.isfinite(window).all()
    assert abs(float(window.mean())) < 1e-4
    # A short utterance is repeated, not zero-padded, before that window normalization.
    repeated = signal.repeat(int(np.ceil(WIN / signal.numel())))[:WIN]
    assert torch.allclose(window, (repeated - repeated.mean()) / repeated.std(unbiased=False))

    exact = fit_length(torch.arange(WIN, dtype=torch.float32))
    assert exact.shape == (WIN,)
    cropped = fit_length(torch.arange(WIN + 10, dtype=torch.float32), start=4)
    assert cropped.shape == (WIN,)


def test_pack_roundtrip_keeps_weights_metadata_and_runs(tmp_path: Path) -> None:
    from dcc_data.mission2_wavlm import build_wavlm_classifier

    cnn_path = tmp_path / "cnn_ensemble.pt"
    member_metadata = [_l2_metadata(seed) for seed in (1, 2)]
    cnn_states = [SpeakerCNNLarge().state_dict() for _ in range(2)]
    torch.save(
        {
            ENSEMBLE_STATE_DICTS_KEY: cnn_states,
            "metadata": build_ensemble_metadata(member_metadata),
        },
        cnn_path,
    )

    config = _tiny_wavlm_config()
    model = build_wavlm_classifier(config)
    wavlm_path = tmp_path / "best.ckpt"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": {key: value.half() if value.is_floating_point() else value for key, value in model.state_dict().items()},
            "label_source": "provided_json",
            "architecture": "wavlm-base-weighted-mean-std",
            "revision": "abc",
            "layers": 2,
            "hidden_size": 32,
            "sr": 16000,
            "win_sec": 3.0,
            "inference": {"stride_sec": 1.5, "max_windows": 8, "aggregation": "mean_logit", "threshold": 0.5},
        },
        wavlm_path,
    )

    output = tmp_path / "packed.pt"
    packed = pack_mission2_ensemble.build_prob_ensemble(
        cnn_path,
        [wavlm_path],
        [config_path],
        weights={"cnn": 1.0, "wavlm-base-weighted-mean-std": 3.0},
        combine="weighted_prob_mean",
    )
    torch.save(packed, output)
    loaded = torch.load(output, map_location="cpu")

    assert loaded["format"] == "mission2_prob_ensemble_v1"
    assert loaded["metadata"]["combine"] == "weighted_prob_mean"
    assert loaded["metadata"]["threshold"] == 0.5
    assert loaded["metadata"]["label_source"] == "provided_json"
    assert [item["name"] for item in loaded["metadata"]["members"]] == ["cnn", "wavlm-base-weighted-mean-std"]
    assert loaded["members"][1]["weight"] == 3.0
    for key, value in cnn_states[0].items():
        assert torch.equal(loaded["members"][0]["state_dicts"][0][key], value)
    reloaded = build_wavlm_classifier(loaded["members"][1]["config"])
    reloaded.load_state_dict(
        {
            key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in loaded["members"][1]["state_dict"].items()
        }
    )
    for key, value in model.state_dict().items():
        got = reloaded.state_dict()[key]
        if value.is_floating_point():
            assert torch.allclose(got, value.half().float(), atol=1e-3, rtol=1e-3)
        else:
            assert torch.equal(got, value)

    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "label"
    _write_wav(audio_dir / "synthetic.wav", 1.0)
    label_dir.mkdir()
    (label_dir / "synthetic.json").write_text(
        json.dumps({"utterances": [{"startAt": 0, "endAt": 400}, {"startAt": 400, "endAt": 900}]}),
        encoding="utf-8",
    )
    csv_path = tmp_path / "mission2.csv"
    probs_path = tmp_path / "probs.csv"
    old_argv = sys.argv
    sys.argv = [
        "mission2_inference.py",
        "--audio_dir",
        str(audio_dir),
        "--label_dir",
        str(label_dir),
        "--ckpt_path",
        str(output),
        "--output",
        str(csv_path),
        "--save_probs",
        str(probs_path),
    ]
    try:
        assert mission2_inference.main() == 0
    finally:
        sys.argv = old_argv

    with csv_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["speaker"] for row in rows] == ["0", "1"] or all(row["speaker"] in {"0", "1"} for row in rows)
    assert len(rows) == 2
    assert rows[0]["startAt"] == "0" and rows[1]["endAt"] == "900"
    with probs_path.open(encoding="utf-8") as stream:
        prob_rows = list(csv.DictReader(stream))
    assert list(prob_rows[0]) == [
        "audio_file_name",
        "startAt",
        "endAt",
        "cnn",
        "wavlm-base-weighted-mean-std",
        "p_mean",
    ]
    cnn_p = float(prob_rows[0]["cnn"])
    wavlm_p = float(prob_rows[0]["wavlm-base-weighted-mean-std"])
    mean_p = float(prob_rows[0]["p_mean"])
    assert 0.0 <= cnn_p <= 1.0 and 0.0 <= wavlm_p <= 1.0
    assert abs(mean_p - (cnn_p + 3 * wavlm_p) / 4) < 1e-6
    assert rows[0]["speaker"] == str(int(mean_p >= 0.5))
